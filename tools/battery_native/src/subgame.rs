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

use crate::core::{self, EnvStateN, ALL_IN, BET_BUCKET_START, FOLD, NUM_ACTIONS, SEAT_FOLDED, SEAT_LIVE};
use crate::fingerprint::CHIP_UNIT;
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
    /// STALE-DECK REPAIR (mirrors `config.repair_deck` / `_set_world` + `_reshuffle_undealt`): after the
    /// per-world opponent-hole deal, rebuild the UNDEALT deck tail from the unseen complement so every
    /// sampled world integrates over a RANDOM board runout instead of replaying the ONE frozen base-state
    /// runout on all-in lines. Default OFF => deck untouched => byte/stat-identical to today. Applies to
    /// BOTH the GTO and exploit paths (Python `_set_world` reshuffles for every solve).
    pub repair_deck: bool,
    /// TWO-SIDED SUBGAME CFR (mirrors `config.two_sided`, GTO/base path ONLY): cycle the traverser across
    /// the live seats so every player learns its OWN regret-matched strategy (converges to a subgame
    /// equilibrium instead of a one-sided best-response to a fixed opponent). Default OFF => traverser is
    /// always the hero => byte/stat-identical. The EXPLOIT path hard-codes this false (one-sided BR to the
    /// modeled opponent is the intended firewalled behaviour).
    pub two_sided: bool,
    /// Q2 — DECOUPLE THE OPPONENT-HOLE SAMPLER's RNG (mirrors `config.decouple_sampler`). When ON, the
    /// per-world opponent-hole deal (and the range weighting) draws from its OWN deterministic sub-stream
    /// (seed = public_seed ^ 0x9E3779B97F4A7C15) so its per-world draw count can't desync the in-tree
    /// action/rollout `rng`. Native's `deal_opponent_holes` is a fixed-count Fisher-Yates (already low-
    /// variance in the UNIFORM case), so this is chiefly for config-parity + correctness; it matters most
    /// once range-aware sampling makes the per-world draw count DATA-dependent. Default OFF => the deal uses
    /// the shared `rng` exactly as before => byte/stat-identical.
    pub decouple_sampler: bool,
    /// Q3 — RANGE-AWARE OPPONENT-HOLE SAMPLING (mirrors the intended `config.range_aware`). When ON, each
    /// opponent's hole is drawn from a blueprint-RANGE-weighted distribution over unseen combos instead of
    /// uniform: the hero best-responds to a REALISTIC (tighter) opponent that would not have folded the
    /// action so far. The range weight per candidate hole = the blueprint CONTINUATION probability at the
    /// solve root for that seat holding that hole (1 - P_blueprint(FOLD); 1.0 when FOLD is illegal). Default
    /// OFF => uniform deal => byte/stat-identical.
    pub range_aware: bool,
    /// Q3 — UNIFORM ROBUSTNESS FLOOR for range-aware sampling (mirrors the RNR restricted-response cap): the
    /// per-combo sampling probability is `(1-floor)*range + floor*uniform`. A PRO who bluffs off-range is
    /// still sampled with weight >= floor/|unseen combos| per seat, so the hero keeps defending off-range /
    /// bluff hands. floor=1.0 => pure uniform (== range_aware OFF); floor=0.0 => the honest-range extreme
    /// (exploitable by an off-range bluffer). Only read when `range_aware`.
    pub range_uniform_floor: f64,
    /// ENSEMBLE PARALLELISM (2026-07-01). Run this many INDEPENDENT CFR streams in parallel (rayon), each
    /// with its OWN regret/strategy tables + its OWN RNG (seed = public_seed ^ (stream_idx*ODD_CONST)), then
    /// MERGE by summing the per-stream root strategy_sum (and regrets, for the fallback) before extracting
    /// the served root policy. No shared mutable state across streams => embarrassingly parallel => ~linear
    /// core scaling, AND averaging N independent CFR averages is variance reduction (~N x effective samples
    /// in ~1x wall-clock — what the under-converged live-budget search needs). The subgame tables are small
    /// (a few thousand infosets) so N copies is cheap. `<= 1` => the single-stream loop, byte/stat-identical
    /// to the pre-ensemble binary (stream 0's seed == the legacy `public_seed`). Deterministic per-stream
    /// seeds keep it CRN-stable across processes/arms. Applies to the GTO `solve` path; the exploit path is
    /// left single-stream for now (see solve_exploit — firewalled, follow-up).
    pub ensemble_streams: i32,
    /// BUG-1 FIX (2026-07-02) — LEAF CACHE HOLE-WORLD INTEGRATION. The leaf cache is keyed by PUBLIC
    /// state only, but the env reaching a leaf carries THAT ITERATION's sampled opponent holes, so the
    /// cached value is a single-hole-world draw reused for all remaining ~K iterations (an arbitrary
    /// per-leaf bias the CFR best-responds to — the frozen-villain-hands sibling of the stale-deck bug).
    /// When > 1, the cached leaf becomes a RUNNING MEAN over the first `leaf_cache_worlds` VISITING
    /// worlds: each hit while n < M re-evaluates under the CURRENT iteration's sampled holes (with a
    /// deterministic per-(leaf, n) seed, preserving the CRN/pure-function contract) and returns the
    /// incremental mean; after M samples the mean freezes. Default 1 => first-touch freeze, byte-identical
    /// to today (n=0 uses EXACTLY the legacy leaf seed). GTO `solve` path only; `solve_exploit` forces
    /// the no-op value (its leaf stays first-touch — firewall: the certified exploit path is unchanged).
    pub leaf_cache_worlds: i32,
    /// BUG-5 (partial, 2026-07-02) — DELAYED AVERAGING (DeepStack/Supremus practice: exclude the noisy
    /// early iterates from the SERVED average). strat_sum accumulation begins only at iteration
    /// t >= floor(avg_skip_frac * max_iterations); regret updates are NEVER skipped (the CFR+ clamp and
    /// traversal are untouched — that is a separate experiment). Default 0.0 => accumulate from t=0,
    /// byte-identical to today. GTO `solve` path only; `solve_exploit` forces 0.0 (firewall: the exploit
    /// serve + its AIVAT root-CFV baseline stay byte-identical regardless of this knob).
    pub avg_skip_frac: f64,
    /// R2 (2026-07-02) — PLURIBUS BIASED CONTINUATION LEAVES. When ON, every depth-limit leaf becomes a
    /// 4-way CONTINUATION-CHOICE node for each player still able to act (SEAT_LIVE): (0) unaltered
    /// blueprint, (1) fold-biased x`leaf_bias_mult`, (2) call-biased, (3) raise-biased — the Pluribus
    /// depth-limited-solving construction (Science 2019 supp; Modicum's ablation: k=1 pure-blueprint
    /// leaves are WORSE than no search at all). The choice is made by the SAME CFR that solves the
    /// subgame: the traverser BRANCHES its 4 continuations at the leaf (regret update + strategy-sum,
    /// exactly like an action node — the choice node belongs to the player whose continuation it is);
    /// every other live seat SAMPLES its choice from its own learned 4-way strategy (learned on the
    /// iterations where IT is the traverser under two_sided; uniform-forever under one-sided, where
    /// non-traversers never update — matching the one-sided mode's existing semantics). The choice
    /// infoset is keyed like every other subgame infoset (`infoset_key(env, seat)` + a marker), so it
    /// is IDENTICAL across all leaf nodes in the same infoset, and the non-traverser draw is a pure
    /// function of (choice infoset, iteration t, public_seed) — consistent within an iteration by
    /// construction. Leaf value = playout under the chosen JOINT continuation profile (playout_fixed:
    /// per-seat bias applied to the blueprint probs before sampling; hero's continuation biases hero's
    /// own playout actions too — BOTH hero and opponents choose, per Pluribus; opponent-only choice is
    /// documented to make the searcher defensive/lower-EV). Default OFF => the uniform-draw leaf,
    /// byte-identical to today. GTO `solve` path only; `solve_exploit` forces OFF (firewall) and its
    /// FFI does not even expose the knob (defense in depth).
    pub leaf_biased_continuations: bool,
    /// R2 — bias multiplier for the biased continuations (fold/call/raise mass x this, renormalized).
    /// Pluribus used 5, Modicum 10. Only read when `leaf_biased_continuations` is ON (the legacy
    /// uniform leaf keeps using the adapter-level PlayoutConfig.bias_mult, untouched).
    pub leaf_bias_mult: f64,
}

/// Immutable per-solve context handed down the recursion.
struct Ctx<'a> {
    bp: &'a Blueprint,
    pcfg: &'a PlayoutConfig,
    scfg: &'a SubgameConfig,
    public_seed: u64,
    /// Cached `scfg.two_sided` for the hot cfv branch (GTO path only; the exploit CtxExploit wraps a
    /// base Ctx built with this forced false).
    two_sided: bool,
    /// Precomputed floor(scfg.avg_skip_frac * max_iterations): strat_sum accumulation begins at
    /// iteration t >= avg_skip (0 = legacy accumulate-from-t=0). The exploit CtxExploit wraps a base
    /// Ctx built with this forced 0 (firewall: solve_exploit is byte-identical regardless of the knob).
    avg_skip: i32,
    /// Cached scfg.leaf_cache_worlds.max(1) for the leaf cache (1 = legacy first-touch freeze). The
    /// exploit path never reads it (leaf_value_x keeps its own single-world cache untouched).
    leaf_worlds: i32,
    /// R2 — cached scfg.leaf_biased_continuations (GTO path only; the exploit CtxExploit wraps a base
    /// Ctx built with this forced false, so the certified exploit leaf can never take the biased path).
    leaf_biased: bool,
    /// R2 — cached scfg.leaf_bias_mult (only read when leaf_biased; forced to the 5.0 default on the
    /// exploit Ctx, where it is never read anyway).
    leaf_bias_mult: f64,
}

/// Mutable per-solve tables (regrets + strategy sums keyed by infoset, leaf-value cache keyed by
/// public leaf state).
struct Tables {
    regrets: HashMap<Vec<i64>, [f64; NUM_ACTIONS]>,
    strat_sum: HashMap<Vec<i64>, [f64; NUM_ACTIONS]>,
    /// Leaf value cache: `(sum, n)` running mean over the first `leaf_cache_worlds` VISITING hole-worlds
    /// (n frozen at that cap). With leaf_cache_worlds=1 (default) this is exactly the legacy first-touch
    /// single-value cache: `(v, 1)` inserted on the miss, `sum / 1` (== v bit-for-bit) on every hit.
    leaf: HashMap<Vec<i64>, (f64, i32)>,
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
    pcfg: &PlayoutConfig,
) -> [f64; NUM_ACTIONS] {
    // Key abstraction (made-hand vs EMD frozen-lookup) is decided by `pcfg.fp_key` — the SAME
    // dispatch the leaf playout uses, so subgame CFR + leaf + exploit all key the blueprint
    // identically (the exploit path's `opponent_strategy_exploit` reaches here through
    // `opponent_strategy`; ONLY the key changes there, its frozen-table firewall knobs do not).
    let key = pcfg.fp_key(s, seat);
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
    let bp = blueprint_policy(s, actor, legal, ctx.bp, ctx.pcfg);
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

/// The hero+board removal set at the solve root (a card here is never dealt to an opponent). INVARIANT
/// across worlds, so the Q3 range table is built against it once.
fn hero_board_known<const N: usize>(root: &EnvStateN<N>, hero: usize) -> [bool; 52] {
    let mut known = [false; 52];
    for k in 0..2 {
        let c = root.holes[hero][k];
        if c >= 0 {
            known[c as usize] = true;
        }
    }
    for &c in root.board.iter() {
        if c >= 0 {
            known[c as usize] = true;
        }
    }
    known
}

/// Q3 precomputed per-seat blueprint CONTINUATION-weight table (built ONCE per solve, reused across all K
/// worlds). `w[seat][c0*52 + c1]` (c0 < c1) is that seat's blueprint continuation weight (1 - P(FOLD) at the
/// solve root holding `(c0,c1)`; 1.0 when FOLD is illegal). The fingerprint+blueprint lookup depends only on
/// `(root, seat, hole)` — INVARIANT across worlds — so hoisting it out of the per-world deal turns range-
/// aware sampling from O(K * combos) blueprint lookups into O(combos) (K× cheaper). `None` when range-aware
/// is off (the uniform path never touches it).
struct RangeTable {
    // Flattened [seat][52*52]; only the c0<c1 upper triangle is populated (others 0, never read).
    w: Vec<f64>,
    stride: usize,
}

impl RangeTable {
    #[inline]
    fn get(&self, seat: usize, c0: i32, c1: i32) -> f64 {
        let (a, b) = if c0 < c1 { (c0, c1) } else { (c1, c0) };
        self.w[seat * self.stride + (a as usize) * 52 + (b as usize)]
    }
}

/// Build the per-seat continuation-weight table for every non-hero seat over every unseen combo. `known`
/// is the hero+board removal set (a card in it is never dealt, so its weights are left 0). One blueprint
/// lookup per (seat, unseen-combo); the fingerprint substitutes only `holes[seat]` so the lookup is well
/// defined. Firewall-safe (a pure blueprint read; no re-simulation, no exploit math).
fn build_range_table<const N: usize>(
    root: &EnvStateN<N>,
    hero: usize,
    known: &[bool; 52],
    ctx: &Ctx,
) -> RangeTable {
    let stride = 52 * 52;
    let mut w = vec![0.0f64; N * stride];
    let unseen: Vec<i32> = (0..52).filter(|&c| !known[c as usize]).collect();
    for seat in 0..N {
        if seat == hero {
            continue;
        }
        let base = seat * stride;
        for i in 0..unseen.len() {
            for j in (i + 1)..unseen.len() {
                let c0 = unseen[i];
                let c1 = unseen[j];
                let cw = continue_weight(root, seat, c0, c1, ctx);
                w[base + (c0 as usize) * 52 + (c1 as usize)] = cw;
            }
        }
    }
    RangeTable { w, stride }
}

/// Deal 2 distinct unseen cards to every non-hero seat (ascending seat order, folded seats included —
/// matching `PBSTracker._opponent_seats` + uniform-belief `sample_opponent_holes`: a uniform random
/// distinct allocation from the cards not in the hero's hand or on the board).
///
/// Q3 RANGE-AWARE (`rtab` is `Some`): instead of a uniform allocation, each opponent's 2-card combo is drawn
/// from `(1-floor)*blueprint_range + floor*uniform` over the currently-unseen combos, where the blueprint
/// range weight per combo is the precomputed `rtab` continuation weight (1 - P(FOLD) at the solve root). Card
/// removal is maintained sequentially across seats (a card dealt to an earlier seat is unavailable to a later
/// one), mirroring Python's `sample_opponent_holes`. Degenerate (all-zero range) seats fall back to uniform.
///
/// `root` is unused in the uniform path; in range-aware it is only needed for `rtab` (already built). `env`
/// is the per-world copy whose opponent holes we fill.
fn deal_opponent_holes<const N: usize>(
    env: &mut EnvStateN<N>,
    hero: usize,
    rng: &mut Rng,
    scfg: &SubgameConfig,
    rtab: Option<&RangeTable>,
) {
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

    let rtab = match (scfg.range_aware, rtab) {
        (true, Some(rt)) => rt,
        _ => {
            // UNIFORM (byte/stat-identical to the pre-Q3 deal): Fisher-Yates over the whole unseen pool.
            let mut unseen: Vec<i32> = (0..52).filter(|&c| !known[c as usize]).collect();
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
            return;
        }
    };

    // RANGE-AWARE: per seat, weight every currently-unseen distinct combo by (1-floor)*range + floor*unif,
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
}

/// Blueprint CONTINUATION probability for `seat` holding `(c0,c1)` at the solve root: 1 - P_blueprint(FOLD)
/// over the ABSTRACTED legal set. Returns 1.0 when FOLD is not in the abstracted legal set (the seat cannot
/// fold, so every hole "continues" — no range information). A pure lookup on the native blueprint (the same
/// fingerprint + legal-renorm used everywhere), so it is firewall-safe and needs no re-simulation.
fn continue_weight<const N: usize>(
    root: &EnvStateN<N>,
    seat: usize,
    c0: i32,
    c1: i32,
    ctx: &Ctx,
) -> f64 {
    // Substitute the candidate hole into a scratch copy (only holes[seat] is read by the fingerprint).
    let mut s = *root;
    s.holes[seat][0] = c0;
    s.holes[seat][1] = c1;
    let legal = abstracted_legal(&s, seat, ctx.pcfg);
    if !legal[FOLD] {
        return 1.0; // cannot fold => no range narrowing from a continuation signal.
    }
    let pol = blueprint_policy(&s, seat, &legal, ctx.bp, ctx.pcfg);
    (1.0 - pol[FOLD]).clamp(0.0, 1.0)
}

/// Sample one distinct 2-card combo for `seat` from the range-tilted distribution over combos of unseen
/// cards. Weight(combo) = (1-floor)*range + floor*uniform, normalized over all candidate combos, where
/// `range` is the seat's PRECOMPUTED blueprint continuation weight (`rtab`) and `uniform` is 1/#combos.
/// Returns None when there are <2 unseen cards. Falls back to the uniform tilt for this seat if the range
/// mass is degenerate (all zero). Cheap per world: only the removal-filter + inverse-CDF, no lookups.
fn sample_range_combo(
    rtab: &RangeTable,
    seat: usize,
    used: &[bool; 52],
    floor: f64,
    rng: &mut Rng,
) -> Option<(i32, i32)> {
    let avail: Vec<i32> = (0..52).filter(|&c| !used[c as usize]).collect();
    let m = avail.len();
    if m < 2 {
        return None;
    }
    let n_combos = m * (m - 1) / 2;
    let unif = 1.0 / n_combos as f64;
    let mut combos: Vec<(i32, i32)> = Vec::with_capacity(n_combos);
    let mut range: Vec<f64> = Vec::with_capacity(n_combos);
    let mut range_mass = 0.0f64;
    for i in 0..m {
        for j in (i + 1)..m {
            let c0 = avail[i];
            let c1 = avail[j];
            let r = rtab.get(seat, c0, c1);
            range_mass += r;
            combos.push((c0, c1));
            range.push(r);
        }
    }
    // Mix normalized blueprint-range with uniform; if the range mass is degenerate collapse to uniform.
    let range_norm = if range_mass > 1e-12 { 1.0 / range_mass } else { 0.0 };
    let mut total = 0.0f64;
    let mut weights: Vec<f64> = Vec::with_capacity(n_combos);
    for &r in range.iter() {
        let mixed = (1.0 - floor) * (r * range_norm) + floor * unif;
        weights.push(mixed);
        total += mixed;
    }
    let u = rng.next_f64() * total.max(1e-12);
    let mut cum = 0.0f64;
    for (k, &wt) in weights.iter().enumerate() {
        cum += wt;
        if u < cum {
            return Some(combos[k]);
        }
    }
    Some(*combos.last().unwrap())
}

/// Deterministic per-world reshuffle seed = fold over the SAMPLED opponent seats (ascending seat order,
/// == Python `sorted(sampled_holes)`) of `seed*131 + seat*2704 + c0*52 + c1`, masked to 32 bits. A pure
/// function of the sampled holes, so a given decision samples the SAME worlds across processes / CRN arms
/// (mirrors `_set_world`, pluribus_search.py:774-778). Skips the hero and any un-dealt (holes<0) seat,
/// exactly as Python only folds seats present in `sampled_holes`.
#[inline]
fn repair_deck_seed<const N: usize>(env: &EnvStateN<N>, hero: usize) -> u64 {
    let mut seed: u64 = 0;
    for seat in 0..N {
        if seat == hero {
            continue;
        }
        let c0 = env.holes[seat][0];
        let c1 = env.holes[seat][1];
        if c0 < 0 || c1 < 0 {
            continue;
        }
        seed = (seed
            .wrapping_mul(131)
            .wrapping_add((seat as u64) * 2704)
            .wrapping_add((c0 as u64) * 52)
            .wrapping_add(c1 as u64))
            & 0xffff_ffff;
    }
    seed
}

// --------------------------------------------------------------------------------------------------
// recursion
// --------------------------------------------------------------------------------------------------

/// Odd mixing constant for the per-(leaf, n) re-evaluation seeds (BUG-1 fix): world n's leaf eval uses
/// `leaf_seed ^ (n * ODD)`. n=0 XORs 0 => EXACTLY the legacy seed (byte-identity at leaf_cache_worlds=1).
/// Distinct from ENSEMBLE_SEED_ODD / the decouple-sampler constant so the seed domains never alias.
const LEAF_WORLD_SEED_ODD: u64 = 0xA24BAED4963EE407;

// --------------------------------------------------------------------------------------------------
// R2 — Pluribus biased continuation leaves (gated on scfg.leaf_biased_continuations, default OFF)
// --------------------------------------------------------------------------------------------------

/// NEGATIVE sentinel appended to a continuation-CHOICE infoset key. Every element of a regular action
/// `infoset_key` (seat, street, board cards, lengths, history, spr_bucket) is NON-NEGATIVE, so a key
/// ending in this marker can never collide with any action infoset — the 4-way choice regrets/strategy
/// live safely in the SAME Tables maps as the action nodes (per the R2 spec).
pub(crate) const CONT_CHOICE_MARKER: i64 = -0xC047_C401;

/// NEGATIVE sentinel separating the public leaf key from the appended JOINT continuation profile in
/// the leaf-value cache key. Every element of `leaf_key` (street, board, lengths, actions, pot*2) and
/// the optional two-sided traverser id is non-negative, so profile-extended keys are disjoint from the
/// legacy (feature-OFF) leaf keys by construction.
const CONT_PROFILE_MARKER: i64 = -0xC047_9203;

/// Odd mixing constant for the per-(choice-infoset, iteration) sampling seeds: the non-traverser
/// continuation draw is a PURE FUNCTION of (choice key, t, public_seed) — never the shared traversal
/// rng — so the same choice infoset gets the SAME continuation everywhere within an iteration (the
/// Pluribus per-infoset-consistency requirement) and the solve stays CRN-stable across processes/arms.
const CONT_ITER_SEED_ODD: u64 = 0x9E6C_63D0_54F3_AA6B;

/// Legal mask for a 4-way continuation-choice node: the first NUM_CONTINUATIONS "actions" are the
/// continuations (0=blueprint, 1=fold-biased, 2=call-biased, 3=raise-biased); the rest are dead. Lets
/// the choice node reuse `regret_match` / `sample_action` / the `[f64; NUM_ACTIONS]` table rows as-is.
const CONT_LEGAL: [bool; NUM_ACTIONS] = {
    let mut m = [false; NUM_ACTIONS];
    let mut i = 0;
    while i < playout::NUM_CONTINUATIONS {
        m[i] = true;
        i += 1;
    }
    m
};

/// The continuation-choice infoset key for `seat` at leaf `env`: keyed EXACTLY like every other
/// subgame infoset (`infoset_key(env, seat)` — public state + betting history + spr; seat-scoped)
/// plus the negative marker. NOTE (honest scope): like all subgame infoset keys in this engine, it
/// carries NO private-hand component (the documented BUG-2/R3 limitation) — hero's hand is fixed per
/// solve so hero's choice is exactly Pluribus-keyed; a two-sided opponent's choice is range-average
/// across its resampled worlds, the same approximation its ACTION infosets already make.
pub(crate) fn cont_choice_key<const N: usize>(env: &EnvStateN<N>, seat: usize) -> Vec<i64> {
    let mut k = infoset_key(env, seat);
    k.push(CONT_CHOICE_MARKER);
    k
}

/// Deterministic u in [0,1) for a non-traverser's continuation draw: pure function of
/// (choice infoset key, iteration t, public_seed). Same infoset + same iteration => same u => the
/// SAME sampled continuation at every leaf in that infoset within the iteration (given the strategy
/// cannot change mid-iteration: a non-traverser's choice regrets only update on ITS traverser turns).
pub(crate) fn cont_choice_u(key: &[i64], t: i32, public_seed: u64) -> f64 {
    let h = leaf_seed(key, public_seed);
    let mut r = Rng::seed_from_u64(h ^ ((t as u64).wrapping_add(1)).wrapping_mul(CONT_ITER_SEED_ODD));
    r.next_f64()
}

/// Sample a NON-traverser seat's continuation: regret-matched 4-way strategy over its choice-infoset
/// regrets (uniform until that seat has traversed), drawn with the pure-function u above. Exposed
/// pub(crate) so the consistency gate can assert "same infoset + same iteration => same continuation"
/// directly (the Pluribus per-infoset consistency requirement).
pub(crate) fn sample_cont_choice(
    regrets: &[f64; NUM_ACTIONS],
    key: &[i64],
    t: i32,
    public_seed: u64,
) -> usize {
    let strat = regret_match(regrets, &CONT_LEGAL);
    let u = cont_choice_u(key, t, public_seed);
    sample_action(&strat, &CONT_LEGAL, u)
}

/// Leaf value under the JOINT continuation profile `choices` (one entry per seat; -1 = seat has no
/// choice node — folded/all-in — and never acts in the playout), computed from `hero`'s (the current
/// traverser's) perspective, cached + M-world-integrated exactly like the legacy leaf.
///
/// LEAF-CACHE KEY DESIGN (the R2 spec's "think this through" item): the cache key is
///   `leaf_key(env)  [+ traverser when two_sided]  + CONT_PROFILE_MARKER + choices[0..N]`
/// i.e. a cached value is valid ONLY for the exact continuation profile it was computed under, and
/// the `leaf_cache_worlds` running mean applies PER (leaf, profile) entry. Worst-case profile space
/// is 4^n_live per public leaf, but entries are created LAZILY on visits only: per leaf visit the
/// traverser branches its 4 choices while every other seat's choice is SAMPLED (one profile per
/// visit) — this sampling IS Pluribus's compression insight (one action per abstract infoset per
/// continuation; revisits negligible), so the map grows by at most 4 entries per leaf visit
/// (<= 4 * leaf visits per solve, a few MB at K=2000) and the 4^n blowup never materializes.
fn biased_profile_value<const N: usize>(
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
    let base = leaf_seed(&key, ctx.public_seed);
    let seed = base ^ (n0 as u64).wrapping_mul(LEAF_WORLD_SEED_ODD);
    let v = playout::biased_leaf_value(
        env,
        hero,
        ctx.bp,
        ctx.pcfg,
        seed,
        ctx.scfg.rollout_samples,
        ctx.scfg.resample_board,
        choices,
        ctx.leaf_bias_mult,
    );
    let sum = if n0 == 0 { v } else { sum0 + v };
    let n = n0 + 1;
    tab.leaf.insert(key, (sum, n));
    sum / n as f64
}

/// R2 — the biased-continuation LEAF NODE: a 4-way continuation-choice decision for every seat that
/// can still act (SEAT_LIVE), solved by the SAME CFR as the rest of the subgame.
///
///   * every live NON-traverser seat SAMPLES its continuation from its own learned 4-way strategy
///     (regret-matched over its choice-infoset regrets; uniform until it has traversed) — the draw is
///     a pure function of (choice infoset, t), so it is consistent across all leaves in the infoset
///     within the iteration;
///   * the TRAVERSER (if live) BRANCHES all 4 continuations, evaluates each joint profile via
///     `biased_profile_value`, and does the standard CFR+ regret update + (avg-gated, weighted)
///     strategy-sum accumulation on its choice infoset — "essentially just another action" (Pluribus);
///   * a traverser that cannot act beyond the leaf (folded / all-in) has no choice: the value is the
///     single playout under the others' sampled profile.
///
/// Seats that are folded or all-in get NO choice node (choices[s] = -1): a folded seat has no stake
/// and an all-in seat has no remaining decisions, so a continuation strategy cannot affect either.
fn leaf_value_biased<const N: usize>(
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
    }
    // Traverser's choice node: branch all 4 continuations (hero's own continuation biases hero's own
    // playout actions too — both hero and opponents choose, per Pluribus).
    let ck = cont_choice_key(env, hero);
    let regs = *tab.regrets.get(&ck).unwrap_or(&ZEROS);
    let strat = regret_match(&regs, &CONT_LEGAL);
    let mut cv = [0.0f64; NUM_ACTIONS];
    for c in 0..playout::NUM_CONTINUATIONS {
        choices[hero] = c as i32;
        cv[c] = biased_profile_value(env, hero, &choices, ctx, tab);
    }
    let mut ev = 0.0;
    for c in 0..playout::NUM_CONTINUATIONS {
        ev += strat[c] * cv[c];
    }
    let mut newr = regs;
    for c in 0..playout::NUM_CONTINUATIONS {
        newr[c] = (newr[c] + cv[c] - ev).max(0.0);
    }
    tab.regrets.insert(ck.clone(), newr);
    // Strategy-sum accumulation: identical gating/weighting to the action nodes (delayed averaging +
    // CFR+ linear weight), so the choice node is a first-class citizen of the served average.
    if t >= ctx.avg_skip {
        let w = if ctx.scfg.linear_averaging {
            (t + 1) as f64
        } else {
            1.0
        };
        let ss = tab.strat_sum.entry(ck).or_insert([0.0; NUM_ACTIONS]);
        for c in 0..playout::NUM_CONTINUATIONS {
            ss[c] += w * strat[c];
        }
    }
    ev
}

fn leaf_value<const N: usize>(
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
    }
    // TWO-SIDED: the leaf continuation value is computed FROM the traverser's perspective (`hero` here is
    // the current traverser), so the cache MUST be keyed per-traverser or one traverser's value poisons
    // another's (mirrors pluribus_search.py:619-620 `key = (key, int(hero))`). One-sided => seat-agnostic.
    let mut key = leaf_key(env);
    if ctx.two_sided {
        key.push(hero as i64);
    }
    // BUG-1 FIX: the cache entry is a RUNNING MEAN (sum, n) over the first `leaf_worlds` VISITING
    // hole-worlds. The env at each visit carries THAT iteration's sampled opponent holes, so while
    // n < M every hit re-evaluates under the CURRENT world (deterministic per-(leaf, n) seed => the
    // CRN/pure-function contract holds) and the mean integrates M independent hole-worlds instead of
    // freezing the first-touch draw. n >= M => frozen mean (M=1 default == legacy first-touch freeze).
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
    let base = leaf_seed(&key, ctx.public_seed);
    let seed = base ^ (n0 as u64).wrapping_mul(LEAF_WORLD_SEED_ODD); // n0=0 => base exactly (legacy)
    let v = playout::uniform_leaf_value(
        env,
        hero,
        ctx.bp,
        ctx.pcfg,
        seed,
        ctx.scfg.rollout_samples,
        ctx.scfg.resample_board,
    );
    // n0=0: sum IS v (not 0.0+v, which would flip a -0.0 sign) and v/1.0 == v bit-for-bit, so the
    // M=1 default path returns/caches EXACTLY the legacy first-touch value.
    let sum = if n0 == 0 { v } else { sum0 + v };
    let n = n0 + 1;
    tab.leaf.insert(key, (sum, n));
    sum / n as f64
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
        return leaf_value(env, hero, ctx, tab, t);
    }
    if depth >= ctx.scfg.max_depth {
        return leaf_value(env, hero, ctx, tab, t);
    }
    let actor_i = env.action_seat;
    if actor_i < 0 {
        return leaf_value(env, hero, ctx, tab, t);
    }
    let actor = actor_i as usize;
    let legal = abstracted_legal(env, actor, ctx.pcfg);
    if !any_legal(&legal) {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    let key = infoset_key(env, actor);

    if actor != hero {
        let regrets = *tab.regrets.get(&key).unwrap_or(&ZEROS);
        // TWO-SIDED (GTO): the non-traverser samples from ITS OWN learned regret-match (uniform on first
        // touch), and its regrets update on the iterations where IT is the traverser -> a subgame
        // equilibrium (mirrors pluribus_search.py:722-723). ONE-SIDED: fixed blueprint-mixed prior.
        let strat = if ctx.two_sided {
            regret_match(&regrets, &legal)
        } else {
            opponent_strategy(env, actor, &legal, &regrets, ctx)
        };
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
    // BUG-5 (partial) — DELAYED AVERAGING: the SERVED average excludes iterations t < avg_skip
    // (floor(avg_skip_frac * K), 0 = legacy). Regret updates above are NEVER skipped.
    if t >= ctx.avg_skip {
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
    }
    ev
}

/// Odd multiplier that de-correlates the per-stream RNG seeds in the ensemble (a large odd constant so
/// `public_seed ^ (idx*ODD)` gives well-separated xoshiro states; stream 0 keeps `public_seed` exactly so
/// the single-stream path is byte/stat-identical to the pre-ensemble binary).
const ENSEMBLE_SEED_ODD: u64 = 0x2545F4914F6CDD1D;

/// One INDEPENDENT CFR stream: builds its OWN `Tables` + `Rng` (seeded from `stream_seed`), runs the
/// full iteration loop (identical to the single-stream body), and returns the ROOT infoset's raw
/// `strategy_sum` row (with a `bool` = "had usable avg mass") and `regrets` row. The ensemble merges
/// these across streams; extraction (mask+normalize / regret-match fallback) is done once by the caller.
/// `root_key` and the range table are passed in so they are built once (they are stream-invariant).
#[allow(clippy::too_many_arguments)]
fn solve_stream<const N: usize>(
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
    };
    let mut rng = Rng::seed_from_u64(stream_seed);
    // Q2 DECOUPLE: the opponent-hole deal draws from its OWN deterministic sub-stream (derived from this
    // stream's seed) so it can't desync the in-tree action/rollout `rng`. OFF => the deal uses `rng`.
    let mut hole_rng = if scfg.decouple_sampler {
        Some(Rng::seed_from_u64(stream_seed ^ 0x9E3779B97F4A7C15))
    } else {
        None
    };
    let iters = scfg.max_iterations.max(0);
    for t in 0..iters {
        let mut world = *root;
        {
            let hrng: &mut Rng = hole_rng.as_mut().unwrap_or(&mut rng);
            deal_opponent_holes(&mut world, hero, hrng, scfg, rtab);
        }
        if scfg.repair_deck {
            // Rebuild the undealt deck tail AFTER the deal (seed + known set both depend on the sampled
            // holes). A SEPARATE Rng stream (deterministic per-world seed) so the solve's iteration `rng`
            // is unperturbed and worlds stay CRN-stable across processes/arms.
            let sd = repair_deck_seed(&world, hero);
            let mut drng = Rng::seed_from_u64(sd);
            playout::reshuffle_undealt(&mut world, &mut drng);
        }
        let traverser = traversers[(t as usize) % traversers.len()];
        cfv(&world, traverser, 0, ctx, &mut tab, &mut rng, t);
    }
    // Return the ROOT rows raw (unnormalized) so the ensemble can SUM strat_sum across streams.
    let ss = tab.strat_sum.get(root_key).copied().unwrap_or(ZEROS);
    let has_avg = ss.iter().sum::<f64>() > 1e-12;
    let rr = tab.regrets.get(root_key).copied().unwrap_or(ZEROS);
    (ss, rr, has_avg)
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
        two_sided: scfg.two_sided,
        // Delayed averaging: skip the first floor(avg_skip_frac * K) iterations of strat_sum
        // accumulation (0.0 default => 0 => accumulate from t=0, byte-identical to today).
        avg_skip: (scfg.avg_skip_frac.clamp(0.0, 1.0) * scfg.max_iterations.max(0) as f64).floor()
            as i32,
        leaf_worlds: scfg.leaf_cache_worlds.max(1),
        // R2 — Pluribus biased continuation leaves (default OFF => the legacy uniform leaf).
        leaf_biased: scfg.leaf_biased_continuations,
        leaf_bias_mult: scfg.leaf_bias_mult,
    };
    let root_key = infoset_key(root, hero);
    // TWO-SIDED traverser list: cycle over the LIVE seats (SEAT_LIVE) at the solve root; hero is always
    // included so its served average still builds (mirrors pluribus_search.py:858-865). OFF => [hero], so
    // the traverser is always hero and the loop is byte-identical to today.
    let traversers: Vec<usize> = if scfg.two_sided {
        let mut v: Vec<usize> = (0..N).filter(|&s| root.seat_status[s] == SEAT_LIVE).collect();
        if !v.contains(&hero) {
            v = vec![hero];
        }
        v
    } else {
        vec![hero]
    };
    // Q3 RANGE TABLE: build the per-seat blueprint continuation-weight table ONCE (invariant across worlds
    // AND across ensemble streams -- it is a pure blueprint read of the fixed root).
    let rtab = if scfg.range_aware {
        let known = hero_board_known(root, hero);
        Some(build_range_table(root, hero, &known, &ctx))
    } else {
        None
    };

    // Sum the per-stream ROOT strategy_sum (mass-weighted average of N independent CFR averages == variance
    // reduction) and root regrets (for the no-avg fallback). Single stream => stream 0 with seed exactly
    // `public_seed` => byte/stat-identical to the pre-ensemble binary.
    let n_streams = scfg.ensemble_streams.max(1) as usize;
    let (sum_ss, sum_rr, any_avg): ([f64; NUM_ACTIONS], [f64; NUM_ACTIONS], bool) = if n_streams <= 1 {
        let (ss, rr, has) = solve_stream(
            root, hero, &ctx, &root_key, &traversers, rtab.as_ref(), public_seed,
        );
        (ss, rr, has)
    } else {
        use rayon::prelude::*;
        // Each stream is fully independent (own Tables + Rng); the shared refs (bp/pcfg/scfg/root/root_key/
        // traversers/rtab) are read-only => Sync => no locks, no false sharing on the hot tables. Distinct
        // per-stream seed = public_seed ^ (idx*ODD). Reduce = elementwise sum of the ROOT rows only.
        (0..n_streams)
            .into_par_iter()
            .map(|idx| {
                let seed = public_seed ^ (idx as u64).wrapping_mul(ENSEMBLE_SEED_ODD);
                solve_stream(root, hero, &ctx, &root_key, &traversers, rtab.as_ref(), seed)
            })
            .reduce(
                || (ZEROS, ZEROS, false),
                |(mut ass, mut arr, aa), (bss, brr, ba)| {
                    for a in 0..NUM_ACTIONS {
                        ass[a] += bss[a];
                        arr[a] += brr[a];
                    }
                    (ass, arr, aa || ba)
                },
            )
    };

    // Root strategy = SUMMED strategy_sum (masked, normalized) when any stream had mass, else regret-match
    // fallback on the SUMMED regrets. Identical extraction shape to the single-stream path.
    let mut used_avg = false;
    if any_avg {
        let tot: f64 = sum_ss.iter().sum();
        if tot > 1e-12 {
            for a in 0..NUM_ACTIONS {
                probs[a] = if legal_root[a] { sum_ss[a] } else { 0.0 };
            }
            used_avg = true;
        }
    }
    if !used_avg {
        probs = regret_match(&sum_rr, &legal_root);
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
            // FIREWALL: the exploit path is one-sided by design (best-respond to the modeled opponent).
            // Force two_sided false regardless of scfg so even a caller mistake can't cycle traversers or
            // per-traverser-key the leaf here.
            two_sided: false,
            // FIREWALL: leaf_cache_worlds / avg_skip_frac are GTO-path knobs; force their no-op values
            // here so solve_exploit stays byte-identical regardless of scfg (leaf_value_x keeps its own
            // single-world cache and cfv_exploit accumulates strat_sum + AIVAT root CFVs from t=0).
            avg_skip: 0,
            leaf_worlds: 1,
            // FIREWALL: R2 biased continuation leaves are a GTO-path knob; force OFF here regardless of
            // scfg (defense in depth — the exploit FFI does not expose the knob either), so the
            // certified exploit leaf (leaf_value_x) is byte-identical no matter what a caller sets.
            leaf_biased: false,
            leaf_bias_mult: 5.0,
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
    // Q2 DECOUPLE on the exploit path too (world sampling only; the exploit MATH is untouched — firewall).
    let mut hole_rng = if scfg.decouple_sampler {
        Some(Rng::seed_from_u64(public_seed ^ 0x9E3779B97F4A7C15))
    } else {
        None
    };
    // Q3 RANGE TABLE (built against the exploit path's base Ctx — a pure blueprint read, firewall-safe).
    let rtab = if scfg.range_aware {
        let known = hero_board_known(root, hero);
        Some(build_range_table(root, hero, &known, &cx.base))
    } else {
        None
    };
    let iters = scfg.max_iterations.max(0);
    for t in 0..iters {
        let mut world = *root;
        {
            let hrng: &mut Rng = hole_rng.as_mut().unwrap_or(&mut rng);
            deal_opponent_holes(&mut world, hero, hrng, scfg, rtab.as_ref());
        }
        // repair_deck applies to the exploit path too (Python `_set_world` reshuffles for every solve).
        // two_sided does NOT — the exploit traverser is always `hero` (one-sided BR).
        if cx.base.scfg.repair_deck {
            let sd = repair_deck_seed(&world, hero);
            let mut drng = Rng::seed_from_u64(sd);
            playout::reshuffle_undealt(&mut world, &mut drng);
        }
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
