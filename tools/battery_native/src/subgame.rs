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

use crate::core::{self, EnvStateN, ALL_IN, BET_BUCKET_START, NUM_ACTIONS, SEAT_FOLDED};
use crate::fingerprint::{fingerprint_fast, CHIP_UNIT};
use crate::playout::{self, Blueprint, PlayoutConfig, Rng};

const ZEROS: [f64; NUM_ACTIONS] = [0.0; NUM_ACTIONS];

/// Precomputed PURE frozen-deviation table (built in Python by `exploit_table.build_frozen_table`):
/// `(actor:u8, ctx_id:u16, legal_bits:u16) -> frozen[f64;11]`. Native NEVER computes the exploit math
/// (context_read / posterior / cap_eff) — it only LOOKS UP this output (data-not-code firewall) and
/// applies the documented overwrite / blend at opponent nodes. A MISS (no read at that context, or the
/// mask not tabled) falls through to the GTO base, identical to Python's `read is None` path.
pub type FrozenTable = HashMap<(u8, u16, u16), [f64; NUM_ACTIONS]>;

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
// context-key id (firewall-safe KEY recompute; a PORT of `_context_key_from_env`, NOT exploit math)
// --------------------------------------------------------------------------------------------------

/// Board-revealed-card count per street (mirrors turbo_rebel_search._CTX_BOARD_LEN = (0,3,4,5)).
const CTX_BOARD_LEN: [usize; 4] = [0, 3, 4, 5];

/// Board texture id (mirrors context.board_texture_rs): 0=dry, 1=wet, 2=paired. `ranks` are 2..14,
/// `suits` are 0..3. Returns None for <3 cards (preflop has no texture in the ctx-id map).
fn board_texture_id(ranks: &[i32], suits: &[i32]) -> Option<u8> {
    let n = ranks.len();
    if n < 3 {
        return None;
    }
    // paired: any duplicate rank (len(set(ranks)) < n).
    let mut seen = [false; 32];
    let mut distinct = 0usize;
    for &r in ranks {
        let idx = r as usize;
        if idx < 32 && !seen[idx] {
            seen[idx] = true;
            distinct += 1;
        }
    }
    if distinct < n {
        return Some(2); // paired
    }
    // wet (flush-heavy): max suit count >= 3.
    let mut suit_cnt = [0i32; 4];
    for &s in suits {
        if (0..4).contains(&s) {
            suit_cnt[s as usize] += 1;
        }
    }
    if suit_cnt.iter().cloned().max().unwrap_or(0) >= 3 {
        return Some(1); // wet
    }
    // straight-heavy: tight cluster (<=4 span) or >=2 close connections (gaps<=2) over sorted-unique.
    let mut rs: Vec<i32> = ranks.to_vec();
    rs.sort_unstable();
    rs.dedup();
    let span = rs[rs.len() - 1] - rs[0];
    let mut close = 0usize;
    for i in 0..rs.len() - 1 {
        if rs[i + 1] - rs[i] <= 2 {
            close += 1;
        }
    }
    if span <= 4 || close >= 2 {
        return Some(1); // wet
    }
    Some(0) // dry
}

/// Is an engine action index aggressive (bet/raise/all-in) — counts toward preflop facing depth.
/// Mirrors context.is_aggressive_action (action_bin in {AGG_SMALL, AGG_LARGE}); equivalently a>=3.
#[inline]
fn is_aggressive_action_rs(a: i32) -> bool {
    let a = a as usize;
    (BET_BUCKET_START..=ALL_IN).contains(&a)
}

/// Raw per-street aggression count for the facing bucket (mirrors `_facing_count_from_env`):
/// preflop = #aggressive entries over the whole flat history; postflop = binary to_call signal.
fn facing_count<const N: usize>(s: &EnvStateN<N>, street: i32, actor: usize) -> i32 {
    if street == 0 {
        let hist_len = s.hist_len.max(0) as usize;
        let mut n = 0i32;
        for i in 0..hist_len {
            if is_aggressive_action_rs(s.hist_action[i]) {
                n += 1;
            }
        }
        return n;
    }
    let to_call = s.current_bet - s.committed_street[actor];
    if to_call <= 0 {
        0
    } else {
        1
    }
}

/// Port of `pluribus_search._context_key_from_env` -> the `all_contexts()` index (0..21). This is the
/// SHARED firewall contract: native recomputes the KEY, Python computed the VALUE. The (street,
/// texture, facing) -> ctx_id map MIRRORS list(all_contexts()) order EXACTLY (asserted equal to the
/// Python list in a test). Returns None for an out-of-range street (the caller treats it as a MISS).
///
/// Layout (= list(all_contexts()) order):
///   preflop (street 0): ctx_id = facing_idx, facing in {unopened:0, facing_bet:1, facing_raise:2,
///                       facing_3bet:3} by raw aggression count capped at 3 (full preflop depth).
///   postflop (street 1..3): ctx_id = 4 + (street-1)*6 + texture_idx*2 + facing_bin
///                       texture in {dry:0, wet:1, paired:2}; facing_bin = min(facing,1) (binary).
fn ctx_key_id<const N: usize>(s: &EnvStateN<N>, actor: usize) -> Option<u16> {
    let street = s.street;
    if !(0..4).contains(&street) {
        return None;
    }
    let n_board = CTX_BOARD_LEN[street as usize];
    let mut ranks: Vec<i32> = Vec::with_capacity(n_board);
    let mut suits: Vec<i32> = Vec::with_capacity(n_board);
    for i in 0..n_board {
        let c = s.board[i];
        if c < 0 {
            continue;
        }
        ranks.push(c / 4 + 2);
        suits.push(c % 4);
    }
    let n_aggr = facing_count(s, street, actor);
    if street == 0 {
        // preflop: full 4-level facing depth (cap at 3).
        let f = n_aggr.clamp(0, 3) as u16;
        return Some(f);
    }
    // postflop: texture (None when <3 revealed, but street>=1 always has >=3 on a settled tick).
    let tex = board_texture_id(&ranks, &suits)?;
    let facing_bin = if n_aggr > 0 { 1u16 } else { 0u16 };
    let base = 4u16 + (street as u16 - 1) * 6 + (tex as u16) * 2 + facing_bin;
    Some(base)
}

/// Public concrete-N wrapper of `ctx_key_id` for the Python cross-check (`subgame_ctx_key_id`).
pub fn ctx_key_id_pub<const N: usize>(s: &EnvStateN<N>, actor: usize) -> Option<u16> {
    ctx_key_id(s, actor)
}

/// 11-bit legal mask packed into a u16 (bit `a` set iff action `a` is in the abstracted legal set).
#[inline]
fn legal_bits(legal: &[bool; NUM_ACTIONS]) -> u16 {
    let mut b = 0u16;
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            b |= 1 << a;
        }
    }
    b
}

/// Exploit per-solve context: the GTO `Ctx` plus the precomputed frozen table + rnr scalars.
struct CtxExploit<'a> {
    base: Ctx<'a>,
    frozen: &'a FrozenTable,
    rnr: &'a [f64; 6],
    // Carried for parity with the Python config; the overwrite-vs-blend decision is driven by the NaN
    // sentinel in `rnr` (Python sets every entry NaN when rnr_restrict is off), so the flag itself is
    // not read in the hot path. Kept so the FFI signature mirrors pluribus_search.
    #[allow(dead_code)]
    rnr_restrict: bool,
    frozen_read_response: bool,
}

/// `_opponent_strategy` with the precomputed pure-frozen exploit deviation applied at opponent nodes.
///
/// Computes the SAME native base (blueprint-mixed regret-match) as `opponent_strategy`, then — iff the
/// frozen table is engaged and `frozen_read_response` — recomputes the context KEY + legal_bits, looks
/// up the frozen vector, and applies (mirroring pluribus_search.py:548-557):
///   * rnr[actor] is NaN (rnr is None)  -> RETURN frozen                    (R1 overwrite)
///   * else                              -> normalize((1-rnr)*base + rnr*frozen) over legal (R2 blend)
/// A MISS (no read at ctx, or this mask not tabled) RETURNS the GTO base — identical to Python's
/// `read is None` path and SAFER than the rare `fs<=1e-12 -> apply_context_bias` fallback we keep in
/// Python (native treats it as a table miss => less deviation).
fn opponent_strategy_exploit<const N: usize>(
    s: &EnvStateN<N>,
    actor: usize,
    legal: &[bool; NUM_ACTIONS],
    regrets: &[f64; NUM_ACTIONS],
    cx: &CtxExploit,
) -> [f64; NUM_ACTIONS] {
    let base = opponent_strategy(s, actor, legal, regrets, &cx.base);
    if !cx.frozen_read_response || actor >= 6 {
        return base;
    }
    let ctx_id = match ctx_key_id(s, actor) {
        Some(c) => c,
        None => return base,
    };
    let bits = legal_bits(legal);
    let frozen = match cx.frozen.get(&(actor as u8, ctx_id, bits)) {
        Some(f) => f,
        None => return base, // MISS -> GTO base (the firewall-safe fallback)
    };
    let r = cx.rnr[actor];
    if r.is_nan() {
        // R1: rnr is None -> overwrite the base with the frozen posterior (already normalized over
        // legal in the Python precompute).
        return *frozen;
    }
    // R2: normalize((1-r)*base + r*frozen) over the legal set (== pluribus_search.py:555-557).
    let mut mixed = [0.0f64; NUM_ACTIONS];
    let mut ms = 0.0;
    for a in 0..NUM_ACTIONS {
        let v = (1.0 - r) * base[a] + r * frozen[a];
        mixed[a] = v;
        ms += v;
    }
    if ms > 1e-12 {
        for a in 0..NUM_ACTIONS {
            mixed[a] /= ms;
        }
        mixed
    } else {
        *frozen
    }
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

// --------------------------------------------------------------------------------------------------
// EXPLOIT solve: same CFR loop, opponent nodes apply the precomputed pure-frozen deviation, and the
// root per-action CFVs are accumulated (CFR+-weighted) for the AIVAT decision baseline.
// --------------------------------------------------------------------------------------------------

/// Tables for the exploit solve: the GTO tables plus the AIVAT root-CFV accumulators (mirrors
/// pluribus_search.py:692-697 — `_root_av_sum` / `_root_av_wsum`, CFR+-weighted like strat_sum).
struct TablesExploit {
    regrets: HashMap<Vec<i64>, [f64; NUM_ACTIONS]>,
    strat_sum: HashMap<Vec<i64>, [f64; NUM_ACTIONS]>,
    leaf: HashMap<Vec<i64>, f64>,
    root_av_sum: [f64; NUM_ACTIONS],
    root_av_wsum: f64,
}

fn leaf_value_x<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    cx: &CtxExploit,
    tab: &mut TablesExploit,
) -> f64 {
    if env.done {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    let key = leaf_key(env);
    if let Some(&v) = tab.leaf.get(&key) {
        return v;
    }
    let seed = leaf_seed(&key, cx.base.public_seed);
    let v = playout::uniform_leaf_value(
        env,
        hero,
        cx.base.bp,
        cx.base.pcfg,
        seed,
        cx.base.scfg.rollout_samples,
        cx.base.scfg.resample_board,
    );
    tab.leaf.insert(key, v);
    v
}

#[allow(clippy::too_many_arguments)]
fn cfv_exploit<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    depth: i32,
    cx: &CtxExploit,
    tab: &mut TablesExploit,
    rng: &mut Rng,
    t: i32,
    want_root_q: bool,
) -> f64 {
    if env.done {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    if core::is_chance_node(env) {
        return leaf_value_x(env, hero, cx, tab);
    }
    if depth >= cx.base.scfg.max_depth {
        return leaf_value_x(env, hero, cx, tab);
    }
    let actor_i = env.action_seat;
    if actor_i < 0 {
        return leaf_value_x(env, hero, cx, tab);
    }
    let actor = actor_i as usize;
    let legal = abstracted_legal(env, actor, cx.base.pcfg);
    if !any_legal(&legal) {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    let key = infoset_key(env, actor);

    if actor != hero {
        let regrets = *tab.regrets.get(&key).unwrap_or(&ZEROS);
        let strat = opponent_strategy_exploit(env, actor, &legal, &regrets, cx);
        let u = rng.next_f64();
        let a = sample_action(&strat, &legal, u);
        let child = core::apply_action(env, a);
        return cfv_exploit(&child, hero, depth + 1, cx, tab, rng, t, want_root_q);
    }

    // hero node: branch all (abstracted-)legal actions, CFR+ regret update.
    let regrets = *tab.regrets.get(&key).unwrap_or(&ZEROS);
    let strat = regret_match(&regrets, &legal);
    let mut av = [0.0f64; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            let child = core::apply_action(env, a);
            av[a] = cfv_exploit(&child, hero, depth + 1, cx, tab, rng, t, want_root_q);
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
    let w = if cx.base.scfg.linear_averaging {
        (t + 1) as f64
    } else {
        1.0
    };
    let ss = tab.strat_sum.entry(key).or_insert([0.0; NUM_ACTIONS]);
    for a in 0..NUM_ACTIONS {
        ss[a] += w * strat[a];
    }
    // AIVAT decision-baseline capture (mirrors pluribus_search.py:692-697): at the ROOT decision node
    // (depth==0) accumulate the per-action CFVs CFR+-weighted exactly like the served strategy, so the
    // AIVAT baseline is CONSISTENT with the served average policy.
    if want_root_q && depth == 0 {
        for a in 0..NUM_ACTIONS {
            tab.root_av_sum[a] += w * av[a];
        }
        tab.root_av_wsum += w;
    }
    ev
}

/// Run the exploit subgame solve. Returns `(root_probs[11], root_q[11])` where `root_q[a]` is the
/// CFR+-weighted average per-action CFV at the root (the AIVAT decision baseline) — all zeros when
/// `want_root_q` is false.
#[allow(clippy::too_many_arguments)]
pub fn solve_exploit<const N: usize>(
    root: &EnvStateN<N>,
    hero: usize,
    bp: &Blueprint,
    pcfg: &PlayoutConfig,
    scfg: &SubgameConfig,
    public_seed: u64,
    frozen: &FrozenTable,
    rnr: &[f64; 6],
    rnr_restrict: bool,
    frozen_read_response: bool,
    want_root_q: bool,
) -> ([f64; NUM_ACTIONS], [f64; NUM_ACTIONS]) {
    let legal_root = abstracted_legal(root, hero, pcfg);
    let mut probs = [0.0f64; NUM_ACTIONS];
    let root_q = [0.0f64; NUM_ACTIONS];
    if !any_legal(&legal_root) {
        return (probs, root_q);
    }
    let cx = CtxExploit {
        base: Ctx {
            bp,
            pcfg,
            scfg,
            public_seed,
        },
        frozen,
        rnr,
        rnr_restrict,
        frozen_read_response,
    };
    let mut tab = TablesExploit {
        regrets: HashMap::new(),
        strat_sum: HashMap::new(),
        leaf: HashMap::new(),
        root_av_sum: [0.0; NUM_ACTIONS],
        root_av_wsum: 0.0,
    };
    let root_key = infoset_key(root, hero);
    let mut rng = Rng::seed_from_u64(public_seed);
    let iters = scfg.max_iterations.max(0);
    for t in 0..iters {
        let mut world = *root;
        deal_opponent_holes(&mut world, hero, &mut rng);
        cfv_exploit(&world, hero, 0, &cx, &mut tab, &mut rng, t, want_root_q);
    }

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

    // Root-CFV average (the AIVAT baseline): root_av_sum / root_av_wsum (mirrors _leaf_q extraction).
    let mut rq = [0.0f64; NUM_ACTIONS];
    if want_root_q && tab.root_av_wsum > 1e-12 {
        for a in 0..NUM_ACTIONS {
            rq[a] = tab.root_av_sum[a] / tab.root_av_wsum;
        }
    }
    (probs, rq)
}
