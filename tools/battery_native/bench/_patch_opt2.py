# OPT-2 patch: digest keys + FxHashMap in subgame.rs (GTO path only). One-shot; idempotence guarded
# by the count==1 asserts.
import io, sys

p = 'src/subgame.rs'
t = io.open(p, encoding='utf-8').read()
R = []

# ---- biased_profile_value ----
R.append(("""fn biased_profile_value<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    choices: &[i32; N],
    ctx: &Ctx,
    tab: &mut Tables,
) -> f64 {
    let mut key = leaf_key(env);
    if ctx.two_sided {
        key.push(hero as i64);
    }
    key.push(CONT_PROFILE_MARKER);
    for s in 0..N {
        key.push(choices[s] as i64);
    }
    let m = ctx.leaf_worlds;
    let (sum0, n0) = match tab.leaf.get(&key) {
        Some(&(s, n)) => {
            if n >= m {
                return s / n as f64;
            }
            (s, n)
        }
        None => (0.0, 0),
    };
    let base = leaf_seed(&key, ctx.public_seed);""",
"""fn biased_profile_value<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    choices: &[i32; N],
    ctx: &Ctx,
    tab: &mut Tables,
    nd: &NodeDig,
) -> f64 {
    // OPT-2: the cache is keyed by the profile-extended leaf DIGEST; the legacy Vec key is rebuilt
    // ONLY on an eval (cache-miss path) to derive the exact legacy `leaf_seed` (bit-identical rng).
    let trav = if ctx.two_sided { Some(hero) } else { None };
    let dkey = prof_dig(leaf_dig(nd, env, trav), choices);
    let m = ctx.leaf_worlds;
    let (sum0, n0) = match tab.leaf.get(&dkey) {
        Some(&(s, n)) => {
            if n >= m {
                return s / n as f64;
            }
            (s, n)
        }
        None => (0.0, 0),
    };
    let mut key = leaf_key(env);
    if ctx.two_sided {
        key.push(hero as i64);
    }
    key.push(CONT_PROFILE_MARKER);
    for s in 0..N {
        key.push(choices[s] as i64);
    }
    let base = leaf_seed(&key, ctx.public_seed);"""))

R.append(("""    let sum = if n0 == 0 { v } else { sum0 + v };
    let n = n0 + 1;
    tab.leaf.insert(key, (sum, n));
    sum / n as f64
}

/// R2 — the biased-continuation LEAF NODE""",
"""    let sum = if n0 == 0 { v } else { sum0 + v };
    let n = n0 + 1;
    tab.leaf.insert(dkey, (sum, n));
    sum / n as f64
}

/// R2 — the biased-continuation LEAF NODE"""))

# ---- leaf_value_biased ----
R.append(("""fn leaf_value_biased<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    ctx: &Ctx,
    tab: &mut Tables,
    t: i32,
) -> f64 {
    let mut choices: [i32; N] = [-1; N];
    for s in 0..N {
        if s == hero || env.seat_status[s] != SEAT_LIVE {
            continue;
        }
        let ck = cont_choice_key(env, s);
        let regs = *tab.regrets.get(&ck).unwrap_or(&ZEROS);
        choices[s] = sample_cont_choice(&regs, &ck, t, ctx.public_seed) as i32;
    }
    if env.seat_status[hero] != SEAT_LIVE {
        return biased_profile_value(env, hero, &choices, ctx, tab);
    }""",
"""fn leaf_value_biased<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    ctx: &Ctx,
    tab: &mut Tables,
    t: i32,
    nd: &NodeDig,
) -> f64 {
    // OPT-2: choice-infoset REGRET/STRAT rows are keyed by the domain-separated infoset digest; the
    // legacy Vec `cont_choice_key` is still built for the pure-function `cont_choice_u` draw, so the
    // sampled continuations (and thus all values) are bit-identical to the legacy binary.
    let mut choices: [i32; N] = [-1; N];
    for s in 0..N {
        if s == hero || env.seat_status[s] != SEAT_LIVE {
            continue;
        }
        let ckd = cont_dig(iset_dig(nd, env, s));
        let ck = cont_choice_key(env, s);
        let regs = *tab.regrets.get(&ckd).unwrap_or(&ZEROS);
        choices[s] = sample_cont_choice(&regs, &ck, t, ctx.public_seed) as i32;
    }
    if env.seat_status[hero] != SEAT_LIVE {
        return biased_profile_value(env, hero, &choices, ctx, tab, nd);
    }"""))

R.append(("""    let ck = cont_choice_key(env, hero);
    let regs = *tab.regrets.get(&ck).unwrap_or(&ZEROS);
    let strat = regret_match(&regs, &CONT_LEGAL);
    let mut cv = [0.0f64; NUM_ACTIONS];
    for c in 0..playout::NUM_CONTINUATIONS {
        choices[hero] = c as i32;
        cv[c] = biased_profile_value(env, hero, &choices, ctx, tab);
    }""",
"""    let ckd = cont_dig(iset_dig(nd, env, hero));
    let regs = *tab.regrets.get(&ckd).unwrap_or(&ZEROS);
    let strat = regret_match(&regs, &CONT_LEGAL);
    let mut cv = [0.0f64; NUM_ACTIONS];
    for c in 0..playout::NUM_CONTINUATIONS {
        choices[hero] = c as i32;
        cv[c] = biased_profile_value(env, hero, &choices, ctx, tab, nd);
    }"""))

R.append(("""    tab.regrets.insert(ck.clone(), newr);
    // Strategy-sum accumulation: identical gating/weighting to the action nodes (delayed averaging +
    // CFR+ linear weight), so the choice node is a first-class citizen of the served average.
    if t >= ctx.avg_skip {
        let w = if ctx.scfg.linear_averaging {
            (t + 1) as f64
        } else {
            1.0
        };
        let ss = tab.strat_sum.entry(ck).or_insert([0.0; NUM_ACTIONS]);""",
"""    tab.regrets.insert(ckd, newr);
    // Strategy-sum accumulation: identical gating/weighting to the action nodes (delayed averaging +
    // CFR+ linear weight), so the choice node is a first-class citizen of the served average.
    if t >= ctx.avg_skip {
        let w = if ctx.scfg.linear_averaging {
            (t + 1) as f64
        } else {
            1.0
        };
        let ss = tab.strat_sum.entry(ckd).or_insert([0.0; NUM_ACTIONS]);"""))

# ---- leaf_value (uniform) ----
R.append(("""fn leaf_value<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    ctx: &Ctx,
    tab: &mut Tables,
    t: i32,
) -> f64 {
    if env.done {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    // R2 — Pluribus biased continuation leaves (gated, default OFF => the legacy uniform leaf below
    // runs byte-identically; `t` is only read on the biased path).
    if ctx.leaf_biased {
        return leaf_value_biased(env, hero, ctx, tab, t);
    }""",
"""fn leaf_value<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    ctx: &Ctx,
    tab: &mut Tables,
    t: i32,
    nd: &NodeDig,
) -> f64 {
    if env.done {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    // R2 — Pluribus biased continuation leaves (gated, default OFF => the legacy uniform leaf below
    // runs byte-identically; `t` is only read on the biased path).
    if ctx.leaf_biased {
        return leaf_value_biased(env, hero, ctx, tab, t, nd);
    }"""))

R.append(("""    let mut key = leaf_key(env);
    if ctx.two_sided {
        key.push(hero as i64);
    }
    prof_count!(crate::prof::C_LEAF_HIT); // counts every non-terminal uniform-leaf visit (hits = this - miss)""",
"""    // OPT-2: keyed by the leaf DIGEST; the legacy Vec key is rebuilt only on the eval (miss) path to
    // derive the exact legacy `leaf_seed` (bit-identical rng stream).
    let trav = if ctx.two_sided { Some(hero) } else { None };
    let dkey = leaf_dig(nd, env, trav);
    prof_count!(crate::prof::C_LEAF_HIT); // counts every non-terminal uniform-leaf visit (hits = this - miss)"""))

R.append(("""    let m = ctx.leaf_worlds;
    let (sum0, n0) = match tab.leaf.get(&key) {
        Some(&(s, n)) => {
            if n >= m {
                return s / n as f64;
            }
            (s, n)
        }
        None => (0.0, 0),
    };
    prof_count!(crate::prof::C_LEAF_MISS);
    prof_scope!(crate::prof::P_LEAF_PLAYOUT);
    let base = leaf_seed(&key, ctx.public_seed);""",
"""    let m = ctx.leaf_worlds;
    let (sum0, n0) = match tab.leaf.get(&dkey) {
        Some(&(s, n)) => {
            if n >= m {
                return s / n as f64;
            }
            (s, n)
        }
        None => (0.0, 0),
    };
    prof_count!(crate::prof::C_LEAF_MISS);
    prof_scope!(crate::prof::P_LEAF_PLAYOUT);
    let mut key = leaf_key(env);
    if ctx.two_sided {
        key.push(hero as i64);
    }
    let base = leaf_seed(&key, ctx.public_seed);"""))

R.append(("""    let sum = if n0 == 0 { v } else { sum0 + v };
    let n = n0 + 1;
    tab.leaf.insert(key, (sum, n));
    sum / n as f64
}

fn cfv<const N: usize>(""",
"""    let sum = if n0 == 0 { v } else { sum0 + v };
    let n = n0 + 1;
    tab.leaf.insert(dkey, (sum, n));
    sum / n as f64
}

#[allow(clippy::too_many_arguments)]
fn cfv<const N: usize>("""))

# ---- cfv ----
R.append(("""fn cfv<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    depth: i32,
    ctx: &Ctx,
    tab: &mut Tables,
    rng: &mut Rng,
    t: i32,
) -> f64 {
    if env.done {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    if core::is_chance_node(env) {
        return leaf_value(env, hero, ctx, tab, t);
    }
    if depth >= ctx.scfg.max_depth {
        return leaf_value(env, hero, ctx, tab, t);
    }
    let actor_i = env.action_seat;
    if actor_i < 0 {
        return leaf_value(env, hero, ctx, tab, t);
    }""",
"""fn cfv<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    depth: i32,
    ctx: &Ctx,
    tab: &mut Tables,
    rng: &mut Rng,
    t: i32,
    nd: &NodeDig,
) -> f64 {
    if env.done {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    if core::is_chance_node(env) {
        return leaf_value(env, hero, ctx, tab, t, nd);
    }
    if depth >= ctx.scfg.max_depth {
        return leaf_value(env, hero, ctx, tab, t, nd);
    }
    let actor_i = env.action_seat;
    if actor_i < 0 {
        return leaf_value(env, hero, ctx, tab, t, nd);
    }"""))

R.append(("""    if !any_legal(&legal) {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    let key = infoset_key(env, actor);

    if actor != hero {
        prof_count!(crate::prof::C_NODE_OPP);""",
"""    if !any_legal(&legal) {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    let key = iset_dig(nd, env, actor);

    if actor != hero {
        prof_count!(crate::prof::C_NODE_OPP);"""))

R.append(("""        let u = rng.next_f64();
        let a = sample_action(&strat, &legal, u);
        let child = core::apply_action(env, a);
        return cfv(&child, hero, depth + 1, ctx, tab, rng, t);
    }

    // hero node: branch all (abstracted-)legal actions, CFR+ regret update.
    prof_count!(crate::prof::C_NODE_HERO);
    let regrets = *tab.regrets.get(&key).unwrap_or(&ZEROS);
    let strat = regret_match(&regrets, &legal);
    let mut av = [0.0f64; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            let child = core::apply_action(env, a);
            av[a] = cfv(&child, hero, depth + 1, ctx, tab, rng, t);
        }
    }""",
"""        let u = rng.next_f64();
        let a = sample_action(&strat, &legal, u);
        let child = core::apply_action(env, a);
        let cd = child_dig(nd, env.hist_len, actor, a);
        return cfv(&child, hero, depth + 1, ctx, tab, rng, t, &cd);
    }

    // hero node: branch all (abstracted-)legal actions, CFR+ regret update.
    prof_count!(crate::prof::C_NODE_HERO);
    let regrets = *tab.regrets.get(&key).unwrap_or(&ZEROS);
    let strat = regret_match(&regrets, &legal);
    let mut av = [0.0f64; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            let child = core::apply_action(env, a);
            let cd = child_dig(nd, env.hist_len, actor, a);
            av[a] = cfv(&child, hero, depth + 1, ctx, tab, rng, t, &cd);
        }
    }"""))

R.append(("""    tab.regrets.insert(key.clone(), newr);
    // BUG-5 (partial) — DELAYED AVERAGING: the SERVED average excludes iterations t < avg_skip
    // (floor(avg_skip_frac * K), 0 = legacy). Regret updates above are NEVER skipped.
    if t >= ctx.avg_skip {""",
"""    tab.regrets.insert(key, newr);
    // BUG-5 (partial) — DELAYED AVERAGING: the SERVED average excludes iterations t < avg_skip
    // (floor(avg_skip_frac * K), 0 = legacy). Regret updates above are NEVER skipped.
    if t >= ctx.avg_skip {"""))

# ---- solve_stream ----
R.append(("""fn solve_stream<const N: usize>(
    root: &EnvStateN<N>,
    hero: usize,
    ctx: &Ctx,
    root_key: &[i64],
    traversers: &[usize],
    rtab: Option<&RangeTable>,
    stream_seed: u64,
) -> ([f64; NUM_ACTIONS], [f64; NUM_ACTIONS], bool) {
    let scfg = ctx.scfg;
    let mut tab = Tables {
        regrets: HashMap::new(),
        strat_sum: HashMap::new(),
        leaf: HashMap::new(),
    };""",
"""fn solve_stream<const N: usize>(
    root: &EnvStateN<N>,
    hero: usize,
    ctx: &Ctx,
    root_nd: &NodeDig,
    root_key: u128,
    traversers: &[usize],
    rtab: Option<&RangeTable>,
    stream_seed: u64,
) -> ([f64; NUM_ACTIONS], [f64; NUM_ACTIONS], bool) {
    let scfg = ctx.scfg;
    let mut tab = Tables {
        regrets: FxHashMap::default(),
        strat_sum: FxHashMap::default(),
        leaf: FxHashMap::default(),
    };"""))

R.append(("""        let traverser = traversers[(t as usize) % traversers.len()];
        cfv(&world, traverser, 0, ctx, &mut tab, &mut rng, t);
    }
    // Return the ROOT rows raw (unnormalized) so the ensemble can SUM strat_sum across streams.
    let ss = tab.strat_sum.get(root_key).copied().unwrap_or(ZEROS);
    let has_avg = ss.iter().sum::<f64>() > 1e-12;
    let rr = tab.regrets.get(root_key).copied().unwrap_or(ZEROS);
    (ss, rr, has_avg)
}""",
"""        let traverser = traversers[(t as usize) % traversers.len()];
        cfv(&world, traverser, 0, ctx, &mut tab, &mut rng, t, root_nd);
    }
    // Return the ROOT rows raw (unnormalized) so the ensemble can SUM strat_sum across streams.
    let ss = tab.strat_sum.get(&root_key).copied().unwrap_or(ZEROS);
    let has_avg = ss.iter().sum::<f64>() > 1e-12;
    let rr = tab.regrets.get(&root_key).copied().unwrap_or(ZEROS);
    (ss, rr, has_avg)
}"""))

# ---- solve_rows ----
R.append(("""    let root_key = infoset_key(root, hero);
    // TWO-SIDED traverser list: cycle over the LIVE seats (SEAT_LIVE) at the solve root; hero is always""",
"""    // OPT-2: the root digest lanes are a pure function of the root's public board/street/history —
    // invariant across worlds AND ensemble streams — so they are built once here. The extraction key
    // is the root infoset digest (the digest twin of the legacy `infoset_key(root, hero)`; the root
    // state's spr/hist_len are world-invariant, so this is valid for every world).
    let root_nd = root_dig(root);
    let root_key = iset_dig(&root_nd, root, hero);
    // TWO-SIDED traverser list: cycle over the LIVE seats (SEAT_LIVE) at the solve root; hero is always"""))

R.append(("""    let (sum_ss, sum_rr, any_avg): ([f64; NUM_ACTIONS], [f64; NUM_ACTIONS], bool) = if n_streams <= 1 {
        let (ss, rr, has) = solve_stream(
            root, hero, &ctx, &root_key, &traversers, rtab.as_ref(), public_seed,
        );
        (ss, rr, has)
    } else {""",
"""    let (sum_ss, sum_rr, any_avg): ([f64; NUM_ACTIONS], [f64; NUM_ACTIONS], bool) = if n_streams <= 1 {
        let (ss, rr, has) = solve_stream(
            root, hero, &ctx, &root_nd, root_key, &traversers, rtab.as_ref(), public_seed,
        );
        (ss, rr, has)
    } else {"""))

R.append(("""                let seed = public_seed ^ (idx as u64).wrapping_mul(ENSEMBLE_SEED_ODD);
                solve_stream(root, hero, &ctx, &root_key, &traversers, rtab.as_ref(), seed)""",
"""                let seed = public_seed ^ (idx as u64).wrapping_mul(ENSEMBLE_SEED_ODD);
                solve_stream(root, hero, &ctx, &root_nd, root_key, &traversers, rtab.as_ref(), seed)"""))

for old, new in R:
    n = t.count(old)
    assert n == 1, (n, old[:70])
    t = t.replace(old, new)
io.open(p, 'w', encoding='utf-8').write(t)
print('OPT-2 edits applied:', len(R))
