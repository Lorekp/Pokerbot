//! Native depth-limited CFR subgame solve — the GTO-base (opponent_model=None) path of
//! `pokerbot.search.pluribus_search.PluribusReBeLSolver.solve` / `_cfv`, run ENTIRELY in Rust.
//!
//! WHY: the Python `_cfv` recursion is the eval's true bottleneck. It marshals a full EnvState across
//! the FFI boundary PER NODE (`_to_flat`/`_from_flat`, profiled ~12.5s / 20% of a run) and runs the
//! CFR+ orchestration in interpreted Python. Porting only the leaf (see `playout.rs`) won 1.33x because
//! the leaf was 27%; the marshaling + orchestration is the rest. This module does one native call per
//! hero decision: opponent-hole sampling, the CFR+ recursion, regret-matching, the blueprint-mixed
//! opponent prior, and the continuation-gadget leaf (reusing `playout::uniform_leaf_value`) — zero
//! per-node Python round-trips.
//!
//! FIDELITY: this mirrors the model=None branch of `_cfv` exactly:
//!   * abstracted legal mask  = full legal AND (allowed_pre if street==0 else allowed_post)
//!   * opponent node          = (1-mix)*regret_match + mix*blueprint, then sample ONE action
//!   * hero node              = branch all legal, CFR+ regret update, accumulate strategy_sum
//!   * leaf (chance / depth-limit / no-actor) = uniform continuation-gadget value, cached by PUBLIC
//!     leaf key (board+street+action-seq+pot) with first-touch holes — same estimator shape as
//!     Python's `_leaf_value` cache (minus suit-iso canonicalization, a statistical-equivalence
//!     preserving difference: suit-isomorphic leaves get independent — equal-expectation — samples).
//!   * root strategy          = strategy_sum[root]*legal normalized, else regret_match fallback.
//!
//! It is STATISTICALLY EQUIVALENT, not bit-identical, to the Python solve: a fast native xoshiro RNG
//! replaces numpy's PCG64 for hole/action sampling and board runouts, and the leaf cache is keyed
//! without suit canonicalization. The exploit tail (opponent_model != None) is NEVER taken here — the
//! caller falls back to Python whenever a model is set, so the firewalled exploitation math is
//! untouched. The opponent-hole deal assumes UNIFORM beliefs (the eval base arm), matching
//! `PBSTracker.sample_opponent_holes` under the default uniform belief.

use std::collections::HashMap;

use crate::core::{self, EnvStateN, NUM_ACTIONS, SEAT_FOLDED};
use crate::fingerprint::{fingerprint_fast, CHIP_UNIT};
use crate::playout::{self, Blueprint, PlayoutConfig, Rng};

const ZEROS: [f64; NUM_ACTIONS] = [0.0; NUM_ACTIONS];

/// CFR-loop knobs (distinct from the leaf playout's `PlayoutConfig`, whose `max_depth` bounds the
/// per-leaf rollout — `max_depth` here bounds the betting-round CFR recursion).
pub struct SubgameConfig {
    pub max_iterations: i32,
    pub max_depth: i32,
    pub blueprint_mix: f64,
    pub rollout_samples: i32,
    pub resample_board: bool,
    /// CFR+ recency-weighted (linear) strategy averaging: weight iteration `t`'s strategy by (t+1) in
    /// the strategy-sum accumulation. Mirrors PluribusReBeLConfig.linear_averaging on the Python side;
    /// MUST match it or the python/native crosscheck diverges.
    pub linear_averaging: bool,
}

/// Immutable per-solve context handed down the recursion.
struct Ctx<'a> {
    bp: &'a Blueprint,
    pcfg: &'a PlayoutConfig,
    scfg: &'a SubgameConfig,
    public_seed: u64,
}

/// Mutable per-solve tables (regrets + strategy sums keyed by infoset, leaf-value cache keyed by
/// public leaf state).
struct Tables {
    regrets: HashMap<Vec<i64>, [f64; NUM_ACTIONS]>,
    strat_sum: HashMap<Vec<i64>, [f64; NUM_ACTIONS]>,
    leaf: HashMap<Vec<i64>, f64>,
}

// --------------------------------------------------------------------------------------------------
// masks / strategy
// --------------------------------------------------------------------------------------------------

#[inline]
fn abstracted_legal<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    pcfg: &PlayoutConfig,
) -> [bool; NUM_ACTIONS] {
    let full = core::legal_action_mask(s, seat as i32);
    let allowed = if s.street == 0 {
        &pcfg.allowed_pre
    } else {
        &pcfg.allowed_post
    };
    let mut m = [false; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        m[a] = full[a] && allowed[a];
    }
    m
}

#[inline]
fn any_legal(m: &[bool; NUM_ACTIONS]) -> bool {
    m.iter().any(|&x| x)
}

/// `_regret_match`: positive regrets over legal, renormalized; uniform-over-legal when no positive mass.
#[inline]
fn regret_match(regrets: &[f64; NUM_ACTIONS], legal: &[bool; NUM_ACTIONS]) -> [f64; NUM_ACTIONS] {
    let mut pos = [0.0f64; NUM_ACTIONS];
    let mut s = 0.0;
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            let v = regrets[a].max(0.0);
            pos[a] = v;
            s += v;
        }
    }
    if s > 0.0 {
        for a in 0..NUM_ACTIONS {
            pos[a] /= s;
        }
        return pos;
    }
    let nl = legal.iter().filter(|&&x| x).count().max(1) as f64;
    let mut out = [0.0f64; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        out[a] = if legal[a] { 1.0 / nl } else { 0.0 };
    }
    out
}

/// Blueprint policy at (state, seat): fingerprint lookup -> masked by the abstracted legal set ->
/// renormalized; uniform over the abstracted legal set on a miss. Mirrors `lbr.make_blueprint_policy`
/// (== `playout::blueprint_policy`, re-implemented here to keep `playout.rs` byte-unchanged).
fn blueprint_policy<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    abst: &[bool; NUM_ACTIONS],
    bp: &Blueprint,
    num_buckets: i32,
) -> [f64; NUM_ACTIONS] {
    let key = fingerprint_fast(s, seat, num_buckets);
    let mut p = [0.0f64; NUM_ACTIONS];
    if let Some(row) = bp.get(&key) {
        let mut tot = 0.0;
        for a in 0..NUM_ACTIONS {
            let v = if abst[a] { row[a] } else { 0.0 };
            p[a] = v;
            tot += v;
        }
        if tot > 1e-12 {
            for a in 0..NUM_ACTIONS {
                p[a] /= tot;
            }
            return p;
        }
    }
    let nl = abst.iter().filter(|&&x| x).count().max(1) as f64;
    for a in 0..NUM_ACTIONS {
        p[a] = if abst[a] { 1.0 / nl } else { 0.0 };
    }
    p
}

/// `_opponent_strategy` (model=None): blueprint-mixed regret-match over the abstracted legal set.
fn opponent_strategy<const N: usize>(
    s: &EnvStateN<N>,
    actor: usize,
    legal: &[bool; NUM_ACTIONS],
    regrets: &[f64; NUM_ACTIONS],
    ctx: &Ctx,
) -> [f64; NUM_ACTIONS] {
    let rm = regret_match(regrets, legal);
    let bp = blueprint_policy(s, actor, legal, ctx.bp, ctx.pcfg.num_buckets);
    let s_bp: f64 = bp.iter().sum();
    if s_bp <= 1e-12 {
        return rm;
    }
    let mix = ctx.scfg.blueprint_mix;
    let mut strat = [0.0f64; NUM_ACTIONS];
    let mut ss = 0.0;
    for a in 0..NUM_ACTIONS {
        strat[a] = (1.0 - mix) * rm[a] + mix * (bp[a] / s_bp);
        ss += strat[a];
    }
    if ss > 1e-12 {
        for a in 0..NUM_ACTIONS {
            strat[a] /= ss;
        }
        strat
    } else {
        let nl = legal.iter().filter(|&&x| x).count().max(1) as f64;
        let mut out = [0.0f64; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            out[a] = if legal[a] { 1.0 / nl } else { 0.0 };
        }
        out
    }
}

/// Categorical inverse-CDF sample of an action over the legal set given `strat` and u in [0,1).
fn sample_action(strat: &[f64; NUM_ACTIONS], legal: &[bool; NUM_ACTIONS], u: f64) -> usize {
    let mut idx: [usize; NUM_ACTIONS] = [0; NUM_ACTIONS];
    let mut n = 0usize;
    let mut s = 0.0;
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            idx[n] = a;
            n += 1;
            s += strat[a];
        }
    }
    if n == 0 {
        return 0;
    }
    if s <= 1e-12 {
        return idx[0];
    }
    let mut cum = 0.0;
    for i in 0..n {
        cum += strat[idx[i]] / s;
        if u < cum {
            return idx[i];
        }
    }
    idx[n - 1]
}

// --------------------------------------------------------------------------------------------------
// keys
// --------------------------------------------------------------------------------------------------

/// `_infoset_key(env, seat)` as a collision-free i64 vector (length-prefixed so field boundaries are
/// unambiguous): [seat, street, n_board, board.., hist_len, hist_seat/action interleaved.., spr].
fn infoset_key<const N: usize>(s: &EnvStateN<N>, seat: usize) -> Vec<i64> {
    let hist_len = s.hist_len.max(0) as usize;
    let mut k: Vec<i64> = Vec::with_capacity(8 + hist_len * 2);
    k.push(seat as i64);
    k.push(s.street as i64);
    // board (only revealed cards, >=0), length-prefixed
    let mut nb = 0i64;
    for &c in s.board.iter() {
        if c >= 0 {
            nb += 1;
        }
    }
    k.push(nb);
    for &c in s.board.iter() {
        if c >= 0 {
            k.push(c as i64);
        }
    }
    // history (seats then actions, interleaved is fine as long as consistent), length-prefixed
    k.push(hist_len as i64);
    for i in 0..hist_len {
        k.push(s.hist_seat[i] as i64);
        k.push(s.hist_action[i] as i64);
    }
    k.push(spr_bucket(s));
    k
}

/// `spr_bucket = int(pot / max(active_mean, 1.0) * 4)` with pot = pot_dead + sum(committed_street),
/// active_mean = mean stack over non-folded seats (else mean over all). `as i64` truncates toward
/// zero, matching Python `int()` on the non-negative ratio.
fn spr_bucket<const N: usize>(s: &EnvStateN<N>) -> i64 {
    let mut pot = s.pot_dead as f64;
    for i in 0..N {
        pot += s.committed_street[i] as f64;
    }
    let mut sm = 0.0;
    let mut cnt = 0i32;
    for i in 0..N {
        if s.seat_status[i] != SEAT_FOLDED {
            sm += s.stacks[i] as f64;
            cnt += 1;
        }
    }
    let active_mean = if cnt > 0 {
        sm / cnt as f64
    } else {
        let mut t = 0.0;
        for i in 0..N {
            t += s.stacks[i] as f64;
        }
        t / N as f64
    };
    (pot / active_mean.max(1.0) * 4.0) as i64
}

/// `_leaf_key(env)` (pre-canonicalization): [street, n_board, board.., hist_len, action-seq.., pot*2].
/// Uses the action sequence only (not seats) and pot*2, mirroring the Python leaf cache key inputs.
fn leaf_key<const N: usize>(s: &EnvStateN<N>) -> Vec<i64> {
    let hist_len = s.hist_len.max(0) as usize;
    let mut k: Vec<i64> = Vec::with_capacity(6 + hist_len);
    k.push(s.street as i64);
    let mut nb = 0i64;
    for &c in s.board.iter() {
        if c >= 0 {
            nb += 1;
        }
    }
    k.push(nb);
    for &c in s.board.iter() {
        if c >= 0 {
            k.push(c as i64);
        }
    }
    k.push(hist_len as i64);
    for i in 0..hist_len {
        k.push(s.hist_action[i] as i64);
    }
    let mut pot = s.pot_dead as f64;
    for i in 0..N {
        pot += s.committed_street[i] as f64;
    }
    k.push((pot * 2.0) as i64);
    k
}

/// FNV-1a over the i64 key, mixed with the public seed: a deterministic per-leaf base seed so a given
/// public leaf integrates a fixed board-runout stream within the solve (the cache makes that moot, but
/// it also keeps distinct leaves independent).
fn leaf_seed(key: &[i64], public_seed: u64) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in key {
        let xb = x as u64;
        for b in xb.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h ^ public_seed.rotate_left(32) ^ public_seed
}

// --------------------------------------------------------------------------------------------------
// opponent-hole sampling (uniform belief)
// --------------------------------------------------------------------------------------------------

/// Deal 2 distinct unseen cards to every non-hero seat (ascending seat order, folded seats included —
/// matching `PBSTracker._opponent_seats` + uniform-belief `sample_opponent_holes`: a uniform random
/// distinct allocation from the cards not in the hero's hand or on the board).
fn deal_opponent_holes<const N: usize>(env: &mut EnvStateN<N>, hero: usize, rng: &mut Rng) {
    let mut known = [false; 52];
    for k in 0..2 {
        let c = env.holes[hero][k];
        if c >= 0 {
            known[c as usize] = true;
        }
    }
    for &c in env.board.iter() {
        if c >= 0 {
            known[c as usize] = true;
        }
    }
    let mut unseen: Vec<i32> = (0..52).filter(|&c| !known[c as usize]).collect();
    // Fisher-Yates over the whole unseen pool, then deal sequentially.
    let n = unseen.len();
    if n > 1 {
        for i in (1..n).rev() {
            let j = rng.next_below((i + 1) as u64) as usize;
            unseen.swap(i, j);
        }
    }
    let mut p = 0usize;
    for seat in 0..N {
        if seat == hero {
            continue;
        }
        if p + 1 < unseen.len() {
            env.holes[seat][0] = unseen[p];
            env.holes[seat][1] = unseen[p + 1];
            p += 2;
        }
    }
}

// --------------------------------------------------------------------------------------------------
// recursion
// --------------------------------------------------------------------------------------------------

fn leaf_value<const N: usize>(env: &EnvStateN<N>, hero: usize, ctx: &Ctx, tab: &mut Tables) -> f64 {
    if env.done {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    let key = leaf_key(env);
    if let Some(&v) = tab.leaf.get(&key) {
        return v;
    }
    let seed = leaf_seed(&key, ctx.public_seed);
    let v = playout::uniform_leaf_value(
        env,
        hero,
        ctx.bp,
        ctx.pcfg,
        seed,
        ctx.scfg.rollout_samples,
        ctx.scfg.resample_board,
    );
    tab.leaf.insert(key, v);
    v
}

fn cfv<const N: usize>(
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
        return leaf_value(env, hero, ctx, tab);
    }
    if depth >= ctx.scfg.max_depth {
        return leaf_value(env, hero, ctx, tab);
    }
    let actor_i = env.action_seat;
    if actor_i < 0 {
        return leaf_value(env, hero, ctx, tab);
    }
    let actor = actor_i as usize;
    let legal = abstracted_legal(env, actor, ctx.pcfg);
    if !any_legal(&legal) {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    let key = infoset_key(env, actor);

    if actor != hero {
        let regrets = *tab.regrets.get(&key).unwrap_or(&ZEROS);
        let strat = opponent_strategy(env, actor, &legal, &regrets, ctx);
        let u = rng.next_f64();
        let a = sample_action(&strat, &legal, u);
        let child = core::apply_action(env, a);
        return cfv(&child, hero, depth + 1, ctx, tab, rng, t);
    }

    // hero node: branch all (abstracted-)legal actions, CFR+ regret update.
    let regrets = *tab.regrets.get(&key).unwrap_or(&ZEROS);
    let strat = regret_match(&regrets, &legal);
    let mut av = [0.0f64; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            let child = core::apply_action(env, a);
            av[a] = cfv(&child, hero, depth + 1, ctx, tab, rng, t);
        }
    }
    let mut ev = 0.0;
    for a in 0..NUM_ACTIONS {
        ev += strat[a] * av[a];
    }
    let mut newr = regrets;
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            newr[a] = (newr[a] + av[a] - ev).max(0.0);
        }
    }
    tab.regrets.insert(key.clone(), newr);
    // CFR+ linear averaging: weight iteration t's strategy by (t+1) (matches the Python solver).
    let w = if ctx.scfg.linear_averaging {
        (t + 1) as f64
    } else {
        1.0
    };
    let ss = tab.strat_sum.entry(key).or_insert([0.0; NUM_ACTIONS]);
    for a in 0..NUM_ACTIONS {
        ss[a] += w * strat[a];
    }
    ev
}

/// Run the depth-limited CFR subgame solve and return the root action probabilities (length 11).
pub fn solve<const N: usize>(
    root: &EnvStateN<N>,
    hero: usize,
    bp: &Blueprint,
    pcfg: &PlayoutConfig,
    scfg: &SubgameConfig,
    public_seed: u64,
) -> [f64; NUM_ACTIONS] {
    let legal_root = abstracted_legal(root, hero, pcfg);
    let mut probs = [0.0f64; NUM_ACTIONS];
    if !any_legal(&legal_root) {
        return probs;
    }
    let ctx = Ctx {
        bp,
        pcfg,
        scfg,
        public_seed,
    };
    let mut tab = Tables {
        regrets: HashMap::new(),
        strat_sum: HashMap::new(),
        leaf: HashMap::new(),
    };
    let root_key = infoset_key(root, hero);
    let mut rng = Rng::seed_from_u64(public_seed);
    let iters = scfg.max_iterations.max(0);
    for t in 0..iters {
        let mut world = *root;
        deal_opponent_holes(&mut world, hero, &mut rng);
        cfv(&world, hero, 0, &ctx, &mut tab, &mut rng, t);
    }

    // Root strategy = strategy_sum (masked, normalized) when it has mass, else regret-match fallback.
    let mut used_avg = false;
    if let Some(ss) = tab.strat_sum.get(&root_key) {
        let tot: f64 = ss.iter().sum();
        if tot > 1e-12 {
            for a in 0..NUM_ACTIONS {
                probs[a] = if legal_root[a] { ss[a] } else { 0.0 };
            }
            used_avg = true;
        }
    }
    if !used_avg {
        let regrets = *tab.regrets.get(&root_key).unwrap_or(&ZEROS);
        probs = regret_match(&regrets, &legal_root);
    }
    // Final mask + renormalize (mirrors `probs = probs*legal; probs/=max(sum,1e-12)`).
    let mut s = 0.0;
    for a in 0..NUM_ACTIONS {
        if !legal_root[a] {
            probs[a] = 0.0;
        }
        s += probs[a];
    }
    let denom = s.max(1e-12);
    for a in 0..NUM_ACTIONS {
        probs[a] /= denom;
    }
    probs
}
