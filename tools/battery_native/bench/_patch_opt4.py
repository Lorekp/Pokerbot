# OPT-4: precomputed exact-mixture range sampler for deal_opponent_holes.
import io

p = 'src/subgame.rs'
t = io.open(p, encoding='utf-8').read()
R = []

# ---- new RangeSampler machinery, inserted right after the RangeTable impl ----
R.append(("""/// Build the per-seat continuation-weight table for every non-hero seat over every unseen combo.""",
"""/// OPT-4 (2026-07-10) — PRECOMPUTED EXACT-MIXTURE RANGE SAMPLER. The legacy `sample_range_combo`
/// rebuilt (3 Vec allocs + ~3 passes over ~1000 combos) PER SEAT PER ITERATION (~15us/iter, 20-70%
/// of solve_total on fast classes). This sampler precomputes, ONCE per solve: the combo list over
/// the root-unseen cards, per-seat prefix-sum CDFs of the range weights, per-seat totals, and
/// per-card weight sums. A seat draw is then the EXACT same mixture the legacy code sampled —
/// `(1-floor) * (range restricted+renormalized over available combos) + floor * uniform(available)`
/// — via component split (u < floor => uniform branch) + prefix binary search with rejection on the
/// few (<= 8) cards dealt to earlier seats (mass corrected by inclusion-exclusion; bounded retries
/// fall back to an exact linear scan). DISTRIBUTION-identical, but the rng DRAW PATTERN differs
/// (documented statistical equivalence; under the deployed `decouple_sampler=true` the hole deal has
/// its own rng stream, so the traversal rng and every in-tree draw are UNAFFECTED). Degenerate
/// (~zero available range mass) mirrors legacy: uniform when floor>0, the last available combo when
/// floor<=0.
pub(crate) struct RangeSampler {
    /// All c0<c1 pairs over the root-unseen cards, in the legacy scan order (i<j over ascending
    /// unseen cards) — the same order the legacy CDF walked.
    combos: Vec<(i32, i32)>,
    /// Per-seat prefix sums of the range weight over `combos` (seat-major, len = N*combos.len()).
    prefix: Vec<f64>,
    /// Per-seat total range mass over ALL root-unseen combos.
    total: Vec<f64>,
    /// Per-seat per-card sum of range weights over combos containing the card (seat-major, N*52).
    card_sum: Vec<f64>,
    /// Ascending root-unseen card list (indexable for uniform draws).
    unseen: Vec<i32>,
}

impl RangeSampler {
    pub(crate) fn build<const N: usize>(
        rtab: &RangeTable,
        root: &EnvStateN<N>,
        hero: usize,
    ) -> RangeSampler {
        let known = hero_board_known(root, hero);
        let unseen: Vec<i32> = (0..52).filter(|&c| !known[c as usize]).collect();
        let m = unseen.len();
        let nc = if m >= 2 { m * (m - 1) / 2 } else { 0 };
        let mut combos: Vec<(i32, i32)> = Vec::with_capacity(nc);
        for i in 0..m {
            for j in (i + 1)..m {
                combos.push((unseen[i], unseen[j]));
            }
        }
        let mut prefix = vec![0.0f64; N * nc];
        let mut total = vec![0.0f64; N];
        let mut card_sum = vec![0.0f64; N * 52];
        for seat in 0..N {
            if seat == hero {
                continue;
            }
            let mut acc = 0.0f64;
            for (k, &(c0, c1)) in combos.iter().enumerate() {
                let r = rtab.get(seat, c0, c1);
                acc += r;
                prefix[seat * nc + k] = acc;
                card_sum[seat * 52 + c0 as usize] += r;
                card_sum[seat * 52 + c1 as usize] += r;
            }
            total[seat] = acc;
        }
        RangeSampler {
            combos,
            prefix,
            total,
            card_sum,
            unseen,
        }
    }
}

/// One seat draw from the precomputed sampler. `used` is the running removal bitmap (root-known +
/// cards dealt to earlier seats this iteration); `extra` lists ONLY the dealt-beyond-known cards
/// (for the inclusion-exclusion mass correction). Returns None when < 2 cards remain available.
fn sample_range_combo_fast(
    rtab: &RangeTable,
    rs: &RangeSampler,
    seat: usize,
    used: &[bool; 52],
    extra: &[i32],
    floor: f64,
    rng: &mut Rng,
) -> Option<(i32, i32)> {
    let m_root = rs.unseen.len();
    let m_avail = m_root - extra.len();
    if m_avail < 2 {
        return None;
    }
    let nc = rs.combos.len();
    if nc == 0 {
        return None;
    }
    // Available range mass by inclusion-exclusion over the (<= 8) extra-used cards.
    let mut mass = rs.total[seat];
    for &c in extra {
        mass -= rs.card_sum[seat * 52 + c as usize];
    }
    for i in 0..extra.len() {
        for j in (i + 1)..extra.len() {
            mass += rtab.get(seat, extra[i], extra[j]);
        }
    }
    let degenerate = !(mass > 1e-12);
    if degenerate && floor <= 0.0 {
        // Legacy mirror: total weight ~0 => the CDF scan returned the LAST available combo, which in
        // the legacy avail-ordered combo list is the pair of the two HIGHEST available cards.
        let mut hi = [-1i32; 2];
        for &c in rs.unseen.iter().rev() {
            if !used[c as usize] {
                if hi[1] < 0 {
                    hi[1] = c;
                } else {
                    hi[0] = c;
                    break;
                }
            }
        }
        return Some((hi[0], hi[1]));
    }
    let use_uniform = degenerate || (floor > 0.0 && rng.next_f64() < floor.min(1.0));
    if use_uniform {
        // Uniform over available combos == two distinct uniform available cards (unordered).
        // Rejection over the static root-unseen list; acceptance >= (m_root-8)/m_root.
        loop {
            let c0 = rs.unseen[rng.next_below(m_root as u64) as usize];
            if used[c0 as usize] {
                continue;
            }
            let c1 = rs.unseen[rng.next_below(m_root as u64) as usize];
            if c1 == c0 || used[c1 as usize] {
                continue;
            }
            return Some(if c0 < c1 { (c0, c1) } else { (c1, c0) });
        }
    }
    // Range component: sample proportional to r over ALL root-unseen combos via prefix binary
    // search, reject combos touching a used card => exact r|available. Bounded retries, then an
    // exact linear fallback (same conditional distribution).
    let pfx = &rs.prefix[seat * nc..(seat + 1) * nc];
    let tot = rs.total[seat];
    if tot > 1e-12 {
        for _ in 0..64 {
            let v = rng.next_f64() * tot;
            let k = pfx.partition_point(|&p| p <= v).min(nc - 1);
            let (c0, c1) = rs.combos[k];
            if !used[c0 as usize] && !used[c1 as usize] {
                return Some((c0, c1));
            }
        }
    }
    // Exact fallback: inverse-CDF over the available combos only (mass computed above).
    let v = rng.next_f64() * mass.max(1e-12);
    let mut acc = 0.0f64;
    let mut last: Option<(i32, i32)> = None;
    for (k, &(c0, c1)) in rs.combos.iter().enumerate() {
        if used[c0 as usize] || used[c1 as usize] {
            continue;
        }
        let r = if k == 0 { pfx[0] } else { pfx[k] - pfx[k - 1] };
        acc += r;
        last = Some((c0, c1));
        if v < acc {
            return Some((c0, c1));
        }
    }
    last
}

/// Build the per-seat continuation-weight table for every non-hero seat over every unseen combo."""))

# ---- deal_opponent_holes: add sampler param + fast path ----
R.append(("""fn deal_opponent_holes<const N: usize>(
    env: &mut EnvStateN<N>,
    hero: usize,
    rng: &mut Rng,
    scfg: &SubgameConfig,
    rtab: Option<&RangeTable>,
) {""",
"""fn deal_opponent_holes<const N: usize>(
    env: &mut EnvStateN<N>,
    hero: usize,
    rng: &mut Rng,
    scfg: &SubgameConfig,
    rtab: Option<&RangeTable>,
    rs: Option<&RangeSampler>,
) {"""))

R.append(("""    // RANGE-AWARE: per seat, weight every currently-unseen distinct combo by (1-floor)*range + floor*unif,
    // sample one combo, mark its two cards used, continue to the next seat. `used` is the running removal
    // set (starts at the hero+board known set). Matches the deal shape of the uniform path (ascending seat
    // order, folded seats included) so the world distribution differs ONLY by the range tilt.
    let floor = scfg.range_uniform_floor.clamp(0.0, 1.0);
    let mut used = known;
    for seat in 0..N {
        if seat == hero {
            continue;
        }
        if let Some((c0, c1)) = sample_range_combo(rtab, seat, &used, floor, rng) {
            env.holes[seat][0] = c0;
            env.holes[seat][1] = c1;
            used[c0 as usize] = true;
            used[c1 as usize] = true;
        }
    }
}""",
"""    // RANGE-AWARE: per seat, weight every currently-unseen distinct combo by (1-floor)*range + floor*unif,
    // sample one combo, mark its two cards used, continue to the next seat. `used` is the running removal
    // set (starts at the hero+board known set). Matches the deal shape of the uniform path (ascending seat
    // order, folded seats included) so the world distribution differs ONLY by the range tilt.
    let floor = scfg.range_uniform_floor.clamp(0.0, 1.0);
    let mut used = known;
    // OPT-4 fast path: precomputed exact-mixture sampler (distribution-identical; see RangeSampler).
    if let Some(rs) = rs {
        let mut extra: [i32; 16] = [0; 16];
        let mut ne = 0usize;
        for seat in 0..N {
            if seat == hero {
                continue;
            }
            if let Some((c0, c1)) =
                sample_range_combo_fast(rtab, rs, seat, &used, &extra[..ne], floor, rng)
            {
                env.holes[seat][0] = c0;
                env.holes[seat][1] = c1;
                used[c0 as usize] = true;
                used[c1 as usize] = true;
                if ne + 2 <= extra.len() {
                    extra[ne] = c0;
                    extra[ne + 1] = c1;
                    ne += 2;
                }
            }
        }
        return;
    }
    for seat in 0..N {
        if seat == hero {
            continue;
        }
        if let Some((c0, c1)) = sample_range_combo(rtab, seat, &used, floor, rng) {
            env.holes[seat][0] = c0;
            env.holes[seat][1] = c1;
            used[c0 as usize] = true;
            used[c1 as usize] = true;
        }
    }
}"""))

# ---- solve_stream: pass sampler through ----
R.append(("""    root_nd: &NodeDig,
    root_key: u128,
    traversers: &[usize],
    rtab: Option<&RangeTable>,
    stream_seed: u64,
) -> ([f64; NUM_ACTIONS], [f64; NUM_ACTIONS], bool) {""",
"""    root_nd: &NodeDig,
    root_key: u128,
    traversers: &[usize],
    rtab: Option<&RangeTable>,
    rs: Option<&RangeSampler>,
    stream_seed: u64,
) -> ([f64; NUM_ACTIONS], [f64; NUM_ACTIONS], bool) {"""))

R.append(("""        let mut world = *root;
        {
            let hrng: &mut Rng = hole_rng.as_mut().unwrap_or(&mut rng);
            deal_opponent_holes(&mut world, hero, hrng, scfg, rtab);
        }
        if scfg.repair_deck {
            prof_scope!(crate::prof::P_REPAIR);""",
"""        let mut world = *root;
        {
            let hrng: &mut Rng = hole_rng.as_mut().unwrap_or(&mut rng);
            deal_opponent_holes(&mut world, hero, hrng, scfg, rtab, rs);
        }
        if scfg.repair_deck {
            prof_scope!(crate::prof::P_REPAIR);"""))

# ---- solve_rows: build sampler + thread through ----
R.append(("""    let rtab = build_rtab(root, hero, bp, pcfg, scfg);

    // Sum the per-stream ROOT strategy_sum""",
"""    let rtab = build_rtab(root, hero, bp, pcfg, scfg);
    // OPT-4: build the exact-mixture sampler once per solve (pure precompute over rtab).
    let rs = rtab.as_ref().map(|rt| RangeSampler::build(rt, root, hero));

    // Sum the per-stream ROOT strategy_sum"""))

R.append(("""        let (ss, rr, has) = solve_stream(
            root, hero, &ctx, &root_nd, root_key, &traversers, rtab.as_ref(), public_seed,
        );
        (ss, rr, has)""",
"""        let (ss, rr, has) = solve_stream(
            root, hero, &ctx, &root_nd, root_key, &traversers, rtab.as_ref(), rs.as_ref(),
            public_seed,
        );
        (ss, rr, has)"""))

R.append(("""                let seed = public_seed ^ (idx as u64).wrapping_mul(ENSEMBLE_SEED_ODD);
                solve_stream(root, hero, &ctx, &root_nd, root_key, &traversers, rtab.as_ref(), seed)""",
"""                let seed = public_seed ^ (idx as u64).wrapping_mul(ENSEMBLE_SEED_ODD);
                solve_stream(
                    root, hero, &ctx, &root_nd, root_key, &traversers, rtab.as_ref(),
                    rs.as_ref(), seed,
                )"""))

# ---- solve_exploit: build sampler + pass ----
R.append(("""    let rtab = build_rtab(root, hero, bp, pcfg, scfg);
    let iters = scfg.max_iterations.max(0);
    for t in 0..iters {
        let mut world = *root;
        {
            let hrng: &mut Rng = hole_rng.as_mut().unwrap_or(&mut rng);
            deal_opponent_holes(&mut world, hero, hrng, scfg, rtab.as_ref());
        }""",
"""    let rtab = build_rtab(root, hero, bp, pcfg, scfg);
    // OPT-4: the exploit path shares the WORLD SAMPLER speedup (distribution-identical; the
    // frozen-read exploit math is untouched — same firewall rationale as range_aware itself).
    let rs = rtab.as_ref().map(|rt| RangeSampler::build(rt, root, hero));
    let iters = scfg.max_iterations.max(0);
    for t in 0..iters {
        let mut world = *root;
        {
            let hrng: &mut Rng = hole_rng.as_mut().unwrap_or(&mut rng);
            deal_opponent_holes(&mut world, hero, hrng, scfg, rtab.as_ref(), rs.as_ref());
        }"""))

for old, new in R:
    n = t.count(old)
    assert n == 1, (n, old[:70])
    t = t.replace(old, new)
io.open(p, 'w', encoding='utf-8').write(t)
print('OPT-4 edits applied:', len(R))
