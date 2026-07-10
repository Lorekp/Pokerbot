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

use rustc_hash::FxHashMap;

use crate::core::{
    self, EnvStateN, ALL_IN, BB_CHIPS, BET_BUCKET_START, FOLD, NUM_ACTIONS, SB_CHIPS, SEAT_ALLIN,
    SEAT_FOLDED, SEAT_LIVE,
};
use crate::fingerprint::CHIP_UNIT;
use crate::playout::{self, Blueprint, PlayoutConfig, Rng};

const ZEROS: [f64; NUM_ACTIONS] = [0.0; NUM_ACTIONS];

// PERF-BENCH (feature `prof`, default OFF => macros expand to nothing; zero shipped-binary impact).
macro_rules! prof_scope {
    ($i:expr) => {
        #[cfg(feature = "prof")]
        let _prof_scope = crate::prof::Scope::new($i);
    };
}
macro_rules! prof_count {
    ($i:expr) => {
        #[cfg(feature = "prof")]
        crate::prof::count($i);
    };
}

// --------------------------------------------------------------------------------------------------
// OPT-2 (2026-07-10) — INCREMENTAL 128-BIT INFOSET/LEAF DIGEST KEYS (GTO solve path only).
//
// The legacy `infoset_key`/`leaf_key` build a length-prefixed Vec<i64> PER NODE (heap alloc +
// O(hist_len) writes) and hash it 2-3x through SipHash per map op — the dominant traversal cost
// (~55-75% of solve_total in the baseline profile). Within one solve, every interior node shares the
// root's board/street (chance nodes are leaves) and differs only by (appended subgame actions, actor,
// spr_bucket, pot). So the CFR recursion carries a small running digest (`NodeDig`, two independent
// 64-bit lanes over the (seat, action) history — plus two action-only lanes for the leaf key) updated
// in O(1) per applied action, and the tables are keyed by the finalized 128-bit digest (FxHashMap).
//
// EQUIVALENCE, not byte-layout identity: a digest key IDENTIFIES the same infoset partition as the
// legacy Vec key up to 128-bit hash collisions (~n^2/2^128, vanishing at n~1e5 infosets/solve). The
// solve's VALUES are bit-identical: traversal order, rng draw sequence, regret arithmetic and the
// leaf-eval SEEDS are unchanged (the legacy Vec leaf key is still built on cache-miss evals — a cold
// path — so `leaf_seed` keeps its exact FNV stream). History-append digests mirror `apply_action`'s
// MAX_HISTORY drop (appends beyond 64 entries do not change the key), preserving the legacy
// saturation aliasing exactly. The EXPLOIT path (cfv_exploit/TablesExploit) keeps the legacy Vec
// keys untouched (firewall: zero implementation drift on the certified exploit solve).
// --------------------------------------------------------------------------------------------------

/// OPT-3 (2026-07-10) — route the GTO subgame's in-tree `apply_action` through the perfect-hash
/// settle (`trainer::compute_returns_phe`, feature phe_showdown+phe_site1, default ON). The settle
/// is CHIP-IDENTICAL to `core::compute_returns` (existing gate:
/// `apply_action_with_phe_matches_scalar_byte_for_byte`), so outputs are byte-unchanged — this is
/// the same certified per-terminal speed lever the trainer ships with, now applied to the search's
/// all-in run-out terminals (the profile showed apply_action = 35-70% of solve on settle-heavy
/// states). Feature OFF => exactly `core::apply_action` (the scalar A/B baseline). The exploit
/// path's cfv_exploit keeps `core::apply_action` untouched (firewall).
#[cfg(all(feature = "phe_showdown", feature = "phe_site1"))]
#[inline(always)]
pub(crate) fn subgame_apply<const N: usize>(env: &EnvStateN<N>, action: usize) -> EnvStateN<N> {
    core::apply_action_with(env, action, settle_checked::<N>)
}

/// The PHE settle with a DEGENERATE-WORLD guard. Under `repair_deck=false` (legacy/test configs)
/// the solver's per-world envs can carry a STALE deck: opponent holes are re-dealt independently of
/// the deck tail, so an in-tree all-in run-out can put a card on the board that duplicates a hole
/// card (the documented stale-deck bug repair_deck exists to fix). The scalar `compute_returns`
/// tolerates such garbage inputs (defined-but-meaningless scores — the certified legacy behavior);
/// the perfect-hash evaluator PANICS on them (5-of-a-rank quinary). This wrapper routes showdown
/// settles with a full, duplicate-free, in-range (board + eligible holes) card set through the
/// certified chip-identical `compute_returns_phe`, and EVERYTHING else through the scalar
/// `core::compute_returns` — bit-identical to the legacy settle in all cases (valid worlds:
/// PHE == scalar by the chip-identity gate; degenerate worlds: exactly the scalar path).
#[cfg(all(feature = "phe_showdown", feature = "phe_site1"))]
fn settle_checked<const N: usize>(
    holes: &[[i32; 2]; N],
    board: &[i32; core::MAX_BOARD],
    seat_status: &[i8; N],
    committed_total: &[i32; N],
    button: i32,
    n_showdown: i32,
) -> [i32; N] {
    if n_showdown > 1 {
        let mut mask: u64 = 0;
        let mut ok = true;
        'validate: {
            for i in 0..core::MAX_BOARD {
                let c = board[i];
                if !(0..52).contains(&c) {
                    ok = false;
                    break 'validate;
                }
                let b = 1u64 << c;
                if mask & b != 0 {
                    ok = false;
                    break 'validate;
                }
                mask |= b;
            }
            let elig = core::eligible_to_win(seat_status);
            for s in 0..N {
                if !elig[s] {
                    continue;
                }
                for k in 0..2 {
                    let c = holes[s][k];
                    if !(0..52).contains(&c) {
                        ok = false;
                        break 'validate;
                    }
                    let b = 1u64 << c;
                    if mask & b != 0 {
                        ok = false;
                        break 'validate;
                    }
                    mask |= b;
                }
            }
        }
        if ok {
            return crate::trainer::compute_returns_phe(
                holes,
                board,
                seat_status,
                committed_total,
                button,
                n_showdown,
            );
        }
    }
    // Fold-win (n_showdown <= 1, no eval) or a degenerate world: the scalar legacy settle.
    core::compute_returns(holes, board, seat_status, committed_total, button, n_showdown)
}
#[cfg(not(all(feature = "phe_showdown", feature = "phe_site1")))]
#[inline(always)]
pub(crate) fn subgame_apply<const N: usize>(env: &EnvStateN<N>, action: usize) -> EnvStateN<N> {
    core::apply_action(env, action)
}

/// splitmix64 finalizer — full-avalanche 64-bit mix.
#[inline(always)]
fn fmix64(mut z: u64) -> u64 {
    z ^= z >> 30;
    z = z.wrapping_mul(0xBF58476D1CE4E5B9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[inline(always)]
fn fold1(h: u64, x: u64) -> u64 {
    fmix64(h ^ x.wrapping_mul(0x9E3779B97F4A7C15))
}

#[inline(always)]
fn fold2(h: u64, x: u64) -> u64 {
    fmix64(h ^ x.wrapping_mul(0xC2B2AE3D27D4EB4F))
}

/// Running per-node digest lanes: `f*` fold the (seat, action) history (infoset keys), `a*` fold the
/// action-only history (leaf keys). Board/street are absorbed at the root (fixed within a solve —
/// interior chance nodes are leaves by construction).
#[derive(Clone, Copy)]
pub(crate) struct NodeDig {
    f1: u64,
    f2: u64,
    a1: u64,
    a2: u64,
}

#[inline(always)]
fn pair_code(seat: i32, action: i32) -> u64 {
    ((seat as u64) << 8) | ((action as u64) & 0xff) | 0x1_0000
}

/// Digest lanes for a solve-root state: absorb street + revealed board, then the recorded history.
pub(crate) fn root_dig<const N: usize>(s: &EnvStateN<N>) -> NodeDig {
    let mut f1 = 0x243F6A8885A308D3u64;
    let mut f2 = 0x452821E638D01377u64;
    f1 = fold1(f1, s.street as u64 + 1);
    f2 = fold2(f2, s.street as u64 + 1);
    for &c in s.board.iter() {
        if c >= 0 {
            f1 = fold1(f1, (c + 1) as u64);
            f2 = fold2(f2, (c + 1) as u64);
        }
    }
    let mut a1 = f1;
    let mut a2 = f2;
    let hl = s.hist_len.clamp(0, core::MAX_HISTORY as i32) as usize;
    for i in 0..hl {
        let pr = pair_code(s.hist_seat[i], s.hist_action[i]);
        f1 = fold1(f1, pr);
        f2 = fold2(f2, pr);
        let acx = (s.hist_action[i] + 1) as u64;
        a1 = fold1(a1, acx);
        a2 = fold2(a2, acx);
    }
    NodeDig { f1, f2, a1, a2 }
}

/// O(1) child digest for `apply_action(env, action)` by `seat`. Mirrors `apply_action`'s history
/// drop: an action applied at hist_len >= MAX_HISTORY is NOT recorded, so it must not change the key
/// (preserving the legacy saturation aliasing byte-for-byte).
#[inline(always)]
fn child_dig(d: &NodeDig, parent_hist_len: i32, seat: usize, action: usize) -> NodeDig {
    if parent_hist_len as usize >= core::MAX_HISTORY {
        return *d;
    }
    let pr = pair_code(seat as i32, action as i32);
    let acx = (action + 1) as u64;
    NodeDig {
        f1: fold1(d.f1, pr),
        f2: fold2(d.f2, pr),
        a1: fold1(d.a1, acx),
        a2: fold2(d.a2, acx),
    }
}

/// Finalized 128-bit INFOSET key for (node digest, actor, spr_bucket, hist_len) — the digest twin of
/// `infoset_key(env, seat)` (street/board live in the lanes; hist entries in the fold; seat/spr/len
/// mixed here).
#[inline(always)]
pub(crate) fn iset_dig<const N: usize>(d: &NodeDig, env: &EnvStateN<N>, seat: usize) -> u128 {
    let spr = spr_bucket(env) as u64;
    let hl = env.hist_len.max(0) as u64;
    let d1 = fmix64(
        d.f1 ^ (seat as u64).wrapping_mul(0xD6E8FEB86659FD93)
            ^ spr.wrapping_mul(0xA5A3564FD3C4B4F9)
            ^ hl.wrapping_mul(0x9E3779B97F4A7C15),
    );
    let d2 = fmix64(
        d.f2 ^ (seat as u64).wrapping_mul(0x8CB92BA72F3D8DD7)
            ^ spr.wrapping_mul(0x94D049BB133111EB)
            ^ hl.wrapping_mul(0xBF58476D1CE4E5B9),
    );
    ((d1 as u128) << 64) | d2 as u128
}

/// Finalized 128-bit LEAF key — the digest twin of `leaf_key(env)` (+ traverser under two_sided):
/// action-only lanes + pot*2 + hist_len (+ traverser+1, 0 = one-sided).
#[inline(always)]
fn leaf_dig<const N: usize>(d: &NodeDig, env: &EnvStateN<N>, trav: Option<usize>) -> u128 {
    let mut pot = env.pot_dead as f64;
    for i in 0..N {
        pot += env.committed_street[i] as f64;
    }
    let p2 = (pot * 2.0) as i64 as u64;
    let hl = env.hist_len.max(0) as u64;
    let tv = trav.map(|h| h as u64 + 1).unwrap_or(0);
    let d1 = fmix64(
        d.a1 ^ p2.wrapping_mul(0xD6E8FEB86659FD93)
            ^ hl.wrapping_mul(0x9E3779B97F4A7C15)
            ^ tv.wrapping_mul(0xA5A3564FD3C4B4F9),
    );
    let d2 = fmix64(
        d.a2 ^ p2.wrapping_mul(0x8CB92BA72F3D8DD7)
            ^ hl.wrapping_mul(0xBF58476D1CE4E5B9)
            ^ tv.wrapping_mul(0x94D049BB133111EB),
    );
    ((d1 as u128) << 64) | d2 as u128
}

/// Domain-separating xor for the R2 continuation-CHOICE keys (the digest twin of appending
/// CONT_CHOICE_MARKER to an infoset key): a fixed-constant xor is a bijection, so choice keys are
/// exactly as collision-free as the underlying infoset digests.
#[inline(always)]
fn cont_dig(iset: u128) -> u128 {
    iset ^ 0xC047_C401_5AFE_D00D_9E37_79B9_7F4A_7C15u128
}

/// Digest twin of the R2 profile-extended leaf cache key (leaf key + CONT_PROFILE_MARKER + choices).
#[inline(always)]
fn prof_dig<const N: usize>(leaf: u128, choices: &[i32; N]) -> u128 {
    let mut d1 = (leaf >> 64) as u64 ^ 0xC047_9203_5AFE_D00Du64;
    let mut d2 = leaf as u64 ^ 0x9203_C047_D00D_5AFEu64;
    for s in 0..N {
        let x = (choices[s] + 2) as u64;
        d1 = fold1(d1, x);
        d2 = fold2(d2, x);
    }
    ((d1 as u128) << 64) | d2 as u128
}

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
    /// (exploitable by an off-range bluffer). Only read when `range_aware` (or `range_reach_full`).
    pub range_uniform_floor: f64,
    /// R1 (2026-07-03) — REACH-CONSISTENT OPPONENT ROOT RANGES (the honest fix for the BUG-2 "opponents
    /// may deviate above the node" half; replaces — does not stack on — the Q3 fold-now proxy). When ON,
    /// each opponent seat's range weight per candidate hole combo is the BLUEPRINT REACH PRODUCT along the
    /// hand's OBSERVED public history: `w_s[combo] = Π_{observed decisions t of seat s} P_bp(a_t | combo,
    /// state_t)` — "how likely is this hand GIVEN every action the seat actually took", the Pluribus/
    /// Libratus root-belief construction (Bayes under the blueprint, never re-sampled) instead of the
    /// one-node "would this hand fold NOW" proxy. The per-decision states are reconstructed by REPLAYING
    /// the root's recorded history (`hist_seat`/`hist_action`) forward from a rebuilt hand-start state
    /// through the SAME `core::apply_action`/`sample_chance` engine, and the replay is VERIFIED per step
    /// (recorded amounts) and at the end (public-field equality with the root) — on ANY mismatch the
    /// table falls back to the legacy behavior (`range_aware` fold-now proxy if set, else uniform), so a
    /// mis-keyed reach weight is structurally impossible (the miskey lesson: worse than uniform).
    /// Blueprint lookups use the SAME `pcfg.fp_key` EMD-correct serve keying as every other lookup
    /// (preflop lossless keys for preflop actions). HERO's decisions are SKIPPED (hero's hand is a known
    /// point mass; his actions carry no information about opponent holes). Blueprint MISSES factor to the
    /// decision's mean hit probability (no per-combo information, scale-neutral); a decision whose
    /// observed action is outside the abstracted legal set, or with no blueprint hits at all, is skipped
    /// entirely. The existing `range_uniform_floor` blend applies unchanged at sample time, so support is
    /// NEVER zero (an off-blueprint line is still sampled at >= floor*uniform — RNR-style robustness).
    /// Weights are computed ONCE per solve at the root (pure function of the public history). Applies to
    /// BOTH the GTO and exploit paths (like `range_aware`: GTO-consistent DATA about the public history
    /// entering only the world sampler — the frozen-read exploit math is untouched). Default OFF =>
    /// byte/stat-identical to today.
    pub range_reach_full: bool,
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
    /// R5a (2026-07-02) — SERVE THE FINAL ITERATE (Pluribus, Science 2019 supp: the bot "plays the
    /// final iterate, not the average" — the weighted average retains mass on poor actions that were
    /// not completely eliminated; belief updates still use the average). When ON, the served ROOT
    /// policy is the REGRET-MATCHING distribution over the root infoset's CUMULATIVE (CFR+-clamped)
    /// regrets AFTER the last iteration — i.e. the current (final-iterate) strategy the CFR would
    /// play on iteration K, exactly the expression the existing no-average fallback already computes
    /// — instead of the (linear-)averaged `strategy_sum`. NOTHING else changes: the traversal,
    /// regret updates, and strategy_sum accumulation are untouched (the average is still built; it
    /// is just not served). Under `ensemble_streams > 1` the final iterate is the regret-match of
    /// the SUMMED per-stream root regrets (the same merge the existing fallback uses; deployed
    /// streams=1). Default OFF => the averaged extraction, byte-identical to today. GTO `solve`
    /// path only; `solve_exploit` NEVER reads this field (its extraction is hard-coded to the
    /// certified averaged serve) and its FFI does not expose the knob (firewall, defense in depth).
    pub serve_final_iterate: bool,
    /// STAGE-1 RIVER OVERBET / OFF-TREE ROOT UN-MASK (2026-07-03, artifacts/offtree_resolve_design.md
    /// §3/§8): bitmask of action indices to OR into the ABSTRACTED legal mask at the ROOT node ONLY
    /// (`depth == 0` in `cfv` — the root state's actor is always the hero seat, so depth==0 is a
    /// complete and precise trigger; it mirrors the Python `_root_extra_key` root-only mechanism).
    /// Every un-masked bit is still ANDed with FULL engine legality (`core::legal_action_mask`), so
    /// an engine-illegal action (e.g. 2x pot > stack) can never enter the tree. Interior nodes keep
    /// the static keep-list untouched. The same bits widen `solve`'s root extraction mask so the
    /// served child is not zeroed at extraction. 0 = no widening, byte-identical to today. GTO
    /// `solve` path only; `solve_exploit` NEVER reads it (its Ctx hard-codes 0 and its FFI does not
    /// expose the param — firewall, defense in depth, mirroring leaf_biased/serve_final_iterate).
    pub root_unmask: u32,
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
    /// STAGE-1 — cached scfg.root_unmask (root-only OR-mask, full-legal-ANDed, applied at depth==0
    /// in `cfv`; 0 = byte-identical). The exploit CtxExploit wraps a base Ctx built with this
    /// FORCED 0 (firewall: solve_exploit's tree can never be widened).
    root_unmask: u32,
}

/// One infoset's mutable state: cumulative CFR+ regrets + the (linear-)averaged strategy sum.
/// OPT-5: merged into ONE row so a hero-node update is a single map probe (entry) instead of the
/// legacy regrets.insert + strat_sum.entry pair, and an opponent read touches one map. Arithmetic
/// and update order are unchanged (bit-identical outputs).
#[derive(Clone, Copy)]
struct IsetRow {
    reg: [f64; NUM_ACTIONS],
    ss: [f64; NUM_ACTIONS],
}

const ROW_ZERO: IsetRow = IsetRow {
    reg: [0.0; NUM_ACTIONS],
    ss: [0.0; NUM_ACTIONS],
};

/// Mutable per-solve tables (regrets + strategy sums keyed by infoset, leaf-value cache keyed by
/// public leaf state). OPT-2: keyed by the 128-bit incremental digests (see `NodeDig`) in FxHashMaps
/// — no per-node key allocation, O(1) hashing. The EXPLOIT path keeps its own legacy-keyed
/// `TablesExploit` (firewall).
struct Tables {
    rows: FxHashMap<u128, IsetRow>,
    /// Leaf value cache: `(sum, n)` running mean over the first `leaf_cache_worlds` VISITING hole-worlds
    /// (n frozen at that cap). With leaf_cache_worlds=1 (default) this is exactly the legacy first-touch
    /// single-value cache: `(v, 1)` inserted on the miss, `sum / 1` (== v bit-for-bit) on every hit.
    leaf: FxHashMap<u128, (f64, i32)>,
}

// --------------------------------------------------------------------------------------------------
// masks / strategy
// --------------------------------------------------------------------------------------------------

#[inline]
pub(crate) fn abstracted_legal<const N: usize>(
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
pub(crate) fn regret_match(regrets: &[f64; NUM_ACTIONS], legal: &[bool; NUM_ACTIONS]) -> [f64; NUM_ACTIONS] {
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
pub(crate) struct RangeTable {
    // Flattened [seat][52*52]; only the c0<c1 upper triangle is populated (others 0, never read).
    w: Vec<f64>,
    stride: usize,
}

impl RangeTable {
    #[inline]
    pub(crate) fn get(&self, seat: usize, c0: i32, c1: i32) -> f64 {
        let (a, b) = if c0 < c1 { (c0, c1) } else { (c1, c0) };
        self.w[seat * self.stride + (a as usize) * 52 + (b as usize)]
    }

    /// Flattened seat-major copy of the weight table (52*52 per seat, c0<c1 upper triangle) — the
    /// probe/debug FFI surface (`SubgameSolver.range_weights`), used by the offline sanity/log-loss
    /// validation of the reach-range plumbing. Never on the solve hot path.
    pub(crate) fn to_flat(&self) -> Vec<f64> {
        self.w.clone()
    }
}

/// OPT-4 (2026-07-10) — PRECOMPUTED EXACT-MIXTURE RANGE SAMPLER. The legacy `sample_range_combo`
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

/// Build the per-seat continuation-weight table for every non-hero seat over every unseen combo. `known`
/// is the hero+board removal set (a card in it is never dealt, so its weights are left 0). One blueprint
/// lookup per (seat, unseen-combo); the fingerprint substitutes only `holes[seat]` so the lookup is well
/// defined. Firewall-safe (a pure blueprint read; no re-simulation, no exploit math).
fn build_range_table<const N: usize>(
    root: &EnvStateN<N>,
    hero: usize,
    known: &[bool; 52],
    bp: &Blueprint,
    pcfg: &PlayoutConfig,
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
                let cw = continue_weight(root, seat, c0, c1, bp, pcfg);
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
    rs: Option<&RangeSampler>,
) {
    prof_scope!(crate::prof::P_DEAL);
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

    let rtab = match (scfg.range_aware || scfg.range_reach_full, rtab) {
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
    bp: &Blueprint,
    pcfg: &PlayoutConfig,
) -> f64 {
    // Substitute the candidate hole into a scratch copy (only holes[seat] is read by the fingerprint).
    let mut s = *root;
    s.holes[seat][0] = c0;
    s.holes[seat][1] = c1;
    let legal = abstracted_legal(&s, seat, pcfg);
    if !legal[FOLD] {
        return 1.0; // cannot fold => no range narrowing from a continuation signal.
    }
    let pol = blueprint_policy(&s, seat, &legal, bp, pcfg);
    (1.0 - pol[FOLD]).clamp(0.0, 1.0)
}

// --------------------------------------------------------------------------------------------------
// R1 — reach-consistent opponent root ranges (range_reach_full; gated, default OFF)
// --------------------------------------------------------------------------------------------------

/// `core::first_live_after` twin (core's is private and core.rs is frozen — reimplemented verbatim):
/// the first SEAT_LIVE seat clockwise after `from_seat`, or -1 when none.
fn first_live_after_rs<const N: usize>(seat_status: &[i8; N], from_seat: i32) -> i32 {
    for off in 1..=(N as i32) {
        let seat = ((from_seat + off).rem_euclid(N as i32)) as usize;
        if seat_status[seat] == SEAT_LIVE {
            return seat as i32;
        }
    }
    -1
}

/// `core::nth_live_after` twin: the n-th live seat clockwise after `from_seat` (n >= 1).
fn nth_live_after_rs<const N: usize>(seat_status: &[i8; N], from_seat: i32, n: i32) -> i32 {
    let mut seat = from_seat;
    for _ in 0..n {
        seat = first_live_after_rs(seat_status, seat);
        if seat < 0 {
            return -1;
        }
    }
    seat
}

/// One observed decision from the replayed public history: `(seat, action, state)` where `state` is
/// the PUBLIC state at which the decision was taken. Holes are NOT reconstructed (they are unknowable
/// and unused): the reach builder substitutes each candidate combo into `state.holes[seat]` before the
/// fingerprint lookup — the only reader of hole data on this path.
pub(crate) type ReplayStep<const N: usize> = (usize, usize, EnvStateN<N>);

/// PUBLIC-field equality between the replayed end state and the solve root — the replay's acceptance
/// gate. Deliberately excludes private/derived fields the replay cannot or need not reproduce: holes
/// (unknown), deck/deck_ptr/rng_key (the replay deck is a synthetic stuffing of the observed board),
/// rewards (root is non-terminal) and the hist arrays (verified per step during the replay).
fn public_eq<const N: usize>(a: &EnvStateN<N>, b: &EnvStateN<N>) -> bool {
    a.street == b.street
        && a.board == b.board
        && a.pot_dead == b.pot_dead
        && a.current_bet == b.current_bet
        && a.min_raise == b.min_raise
        && a.button == b.button
        && a.action_seat == b.action_seat
        && a.last_aggressor == b.last_aggressor
        && a.last_raise_full == b.last_raise_full
        && a.done == b.done
        && a.pending_deal == b.pending_deal
        && a.hist_len == b.hist_len
        && a.seat_status == b.seat_status
        && a.stacks == b.stacks
        && a.committed_street == b.committed_street
        && a.committed_total == b.committed_total
        && a.acted == b.acted
}

/// Replay the root's recorded history forward from a rebuilt hand-start state (given per-seat antes),
/// snapshotting the PUBLIC state before every decision. Returns None on ANY inconsistency: seat/action
/// out of range, wrong actor, illegal action, a recorded amount that does not reproduce, or a final
/// state that is not publicly identical to the root. The start state mirrors `core::init_hand_n` /
/// `post_blinds_n` exactly (blinds from SB_CHIPS/BB_CHIPS clamped to stacks, antes to pot_dead +
/// committed_total, all-in marking, UTG/HU first actor); the deck is stuffed with the observed board
/// so `sample_chance` reveals exactly the real runout at each street close.
pub(crate) fn replay_with_ante<const N: usize>(
    root: &EnvStateN<N>,
    ante_each: &[i32; N],
) -> Option<Vec<ReplayStep<N>>> {
    let hist_len = root.hist_len;
    if root.done || !(0..=core::MAX_HISTORY as i32).contains(&hist_len) {
        return None;
    }
    let hist_len = hist_len as usize;
    let mut s = rebuild_hand_start(root, ante_each)?;

    // ---- forward replay with per-step verification ----
    let mut steps: Vec<ReplayStep<N>> = Vec::with_capacity(hist_len);
    for t in 0..hist_len {
        let mut guard = 0;
        while core::is_chance_node(&s) {
            s = core::sample_chance(&s);
            guard += 1;
            if guard > 4 {
                return None;
            }
        }
        if s.done {
            return None;
        }
        let seat = root.hist_seat[t];
        let action = root.hist_action[t];
        if seat < 0 || seat >= N as i32 || !(0..NUM_ACTIONS as i32).contains(&action) {
            return None;
        }
        if s.action_seat != seat {
            return None;
        }
        let full = core::legal_action_mask(&s, seat);
        if !full[action as usize] {
            return None;
        }
        steps.push((seat as usize, action as usize, s));
        s = core::apply_action(&s, action as usize);
        // The replayed engine re-records the action it just applied; the recorded (seat, action,
        // amount) triple must reproduce the root's history entry exactly (amount mismatches catch a
        // wrong start-state reconstruction immediately, e.g. mis-derived antes or blinds).
        if s.hist_len as usize != t + 1
            || s.hist_seat[t] != root.hist_seat[t]
            || s.hist_action[t] != root.hist_action[t]
            || s.hist_amount[t] != root.hist_amount[t]
        {
            return None;
        }
    }
    let mut guard = 0;
    while core::is_chance_node(&s) {
        s = core::sample_chance(&s);
        guard += 1;
        if guard > 4 {
            return None;
        }
    }
    if !public_eq(&s, root) {
        return None;
    }
    Some(steps)
}

/// Hand-start (pre-preflop-action) state reconstruction shared by every replay attempt: blinds from
/// SB_CHIPS/BB_CHIPS clamped to stacks, antes to pot_dead + committed_total, all-in marking, UTG/HU
/// first actor, deck stuffed with the observed board. Extracted VERBATIM from `replay_with_ante`
/// (Stage 2a refactor) so the amount-faithful replay reuses the identical, already-certified
/// construction — behavior of the legacy attempts is unchanged.
fn rebuild_hand_start<const N: usize>(
    root: &EnvStateN<N>,
    ante_each: &[i32; N],
) -> Option<EnvStateN<N>> {
    // ---- hand-start reconstruction ----
    let mut status0 = [core::SEAT_EMPTY; N];
    for i in 0..N {
        if root.seat_status[i] != core::SEAT_EMPTY {
            status0[i] = SEAT_LIVE;
        }
    }
    let mut stacks0 = [0i32; N];
    for i in 0..N {
        stacks0[i] = root.stacks[i] + root.committed_total[i];
    }
    let button = root.button;
    let sb = nth_live_after_rs(&status0, button, 1);
    let bb = nth_live_after_rs(&status0, button, 2);
    if sb < 0 || bb < 0 || sb == bb {
        return None;
    }
    let (sb, bb) = (sb as usize, bb as usize);
    let mut committed = [0i32; N];
    // post_blinds_n order: blinds first (clamped to the pre-ante stack), then antes (clamped to the
    // post-blind stack). sb/bb are SEAT_LIVE by construction of nth_live_after_rs.
    let sb_paid = SB_CHIPS.min(stacks0[sb].max(0));
    let bb_paid = BB_CHIPS.min(stacks0[bb].max(0));
    stacks0[sb] -= sb_paid;
    stacks0[bb] -= bb_paid;
    committed[sb] = sb_paid;
    committed[bb] = bb_paid;
    let mut committed_total0 = [0i32; N];
    let mut pot_dead0 = 0i32;
    for i in 0..N {
        let a = ante_each[i];
        if status0[i] == core::SEAT_EMPTY {
            if a != 0 {
                return None;
            }
            continue;
        }
        if a < 0 || a > stacks0[i] {
            return None;
        }
        stacks0[i] -= a;
        pot_dead0 += a;
        committed_total0[i] = committed[i] + a;
    }
    for i in 0..N {
        if status0[i] != core::SEAT_EMPTY && stacks0[i] == 0 {
            status0[i] = core::SEAT_ALLIN;
        }
    }
    let mut current_bet0 = 0i32;
    for i in 0..N {
        if committed[i] > current_bet0 {
            current_bet0 = committed[i];
        }
    }
    let mut n_live = 0i32;
    for i in 0..N {
        if status0[i] == SEAT_LIVE {
            n_live += 1;
        }
    }
    let btn_first = nth_live_after_rs(&status0, button, 1);
    let utg = nth_live_after_rs(&status0, button, 3);
    let action_seat0 = if n_live <= 2 { btn_first } else { utg };
    if action_seat0 < 0 {
        return None;
    }

    // Deck stuffed with the observed board at the front so street closes reveal the real runout;
    // the tail (never dealt during a valid replay) is filled with the remaining distinct cards.
    let mut deck = [0i32; 52];
    let mut used = [false; 52];
    let mut p = 0usize;
    for &c in root.board.iter() {
        if c >= 0 {
            if !(0..52).contains(&c) || used[c as usize] {
                return None;
            }
            deck[p] = c;
            used[c as usize] = true;
            p += 1;
        }
    }
    for c in 0..52i32 {
        if !used[c as usize] {
            deck[p] = c;
            p += 1;
        }
    }

    Some(EnvStateN::<N> {
        holes: [[-1i32; 2]; N],
        board: [-1i32; core::MAX_BOARD],
        seat_status: status0,
        stacks: stacks0,
        committed_street: committed,
        committed_total: committed_total0,
        street: core::STREET_PREFLOP,
        pot_dead: pot_dead0,
        current_bet: current_bet0,
        min_raise: BB_CHIPS,
        button,
        action_seat: action_seat0,
        last_aggressor: -1,
        acted: [false; N],
        last_raise_full: true,
        deck,
        deck_ptr: 0,
        rng_key: root.rng_key,
        hist_seat: [-1i32; core::MAX_HISTORY],
        hist_action: [-1i32; core::MAX_HISTORY],
        hist_amount: [0i32; core::MAX_HISTORY],
        hist_len: 0,
        done: false,
        rewards: [0i32; N],
        pending_deal: false,
    })
}

// --------------------------------------------------------------------------------------------------
// Stage 2a — AMOUNT-FAITHFUL history replay (offtree_resolve_design.md §2.3a, live-critical).
//
// WHY: live histories record the client's TRUE amounts — bet/raise entries carry the bet-TO street
// TOTAL (normalized_to_state.py Action.amount_bb -> state_bridge.py `_bb_to_chips`), and calls carry
// whatever the client sent (a total, or 0) — while `replay_with_ante` requires every replayed step's
// ENGINE-recomputed amount (`to_call + max(cur_bet, f32_trunc(pot*frac)) - committed`, core.rs
// action_amount) to equal the recorded one. These never agree on real client sizes, so the certified
// R1 reach conditioning (`range_reach_full`) silently fell back to the uniform deal on 100% of live
// states (Stage-0 telemetry, artifacts/live_reach_replay_telemetry.md). The functions below replay
// COMMITTING THE RECORDED AMOUNT for bet-family actions (fold/check/call/all-in amounts are
// engine-FORCED — computed from the state, the recorded value is ignored) and re-verify with:
//   * per-step action-CLASS legality (actor match, live seat, reopen rule, actually-aggressive,
//     full-raise-or-all-in),
//   * per-seat chip conservation (commit clamps to stack; negative/zero increments refused),
//   * the SAME final `public_eq` acceptance gate against the root's true public fields (pot,
//     committed, stacks, street, ...). NO epsilon: ANY inconsistency returns None and the caller
//     falls through to the legacy attempts and then the legacy range fallback (uniform deal) — the
//     miskey lesson: a wrong reach weight is worse than uniform, every path stays fail-closed.
// core.rs is FROZEN: the divergent-amount apply below locally mirrors `core::apply_action`
// (precedent: `first_live_after_rs`); whenever the recorded amount EQUALS the engine-recomputed one
// the step is delegated to `core::apply_action` verbatim (engine-history passthrough), so
// engine-generated histories reproduce byte-identical steps — and therefore byte-identical reach
// tables — through this new first attempt.
// --------------------------------------------------------------------------------------------------

/// `core::live_count` twin (private in frozen core.rs): counts CONTESTING seats — SEAT_LIVE
/// **or** SEAT_ALLIN (an all-in seat still contests the pot; only folded/empty seats drop out).
fn live_count_rs<const N: usize>(seat_status: &[i8; N]) -> i32 {
    let mut n = 0;
    for i in 0..N {
        if seat_status[i] == SEAT_LIVE || seat_status[i] == SEAT_ALLIN {
            n += 1;
        }
    }
    n
}

/// `core::pot_for_sizing` twin (private in frozen core.rs).
fn pot_for_sizing_rs<const N: usize>(s: &EnvStateN<N>) -> i32 {
    let mut p = s.pot_dead;
    for i in 0..N {
        p += s.committed_street[i];
    }
    p
}

/// `core::bucket_prod` twin: f32-truncate bet target `np.int32(np.float32(pot) * frac)`.
#[inline]
fn bucket_prod_rs(pot: i32, frac: f32) -> i32 {
    ((pot as f32) * frac) as i32
}

/// `core::action_amount` twin (private in frozen core.rs): the ENGINE-computed commit increment for
/// `action` by `seat` at `s` — the amount `core::apply_action` would commit and record.
fn engine_action_amount_rs<const N: usize>(s: &EnvStateN<N>, seat: usize, action: usize) -> i32 {
    let committed = s.committed_street[seat];
    let to_call = (s.current_bet - committed).max(0);
    let stack = s.stacks[seat];
    let pot = pot_for_sizing_rs(s);
    if action == core::CALL {
        return to_call;
    }
    if action == ALL_IN {
        return stack;
    }
    if (BET_BUCKET_START..ALL_IN).contains(&action) {
        let frac = core::BET_BUCKETS_F32[action - BET_BUCKET_START];
        let prod = bucket_prod_rs(pot, frac);
        let target = s.current_bet.max(prod);
        let bucket_val = (target - committed).max(0);
        return to_call + bucket_val;
    }
    0
}

/// `core::street_complete` twin (private in frozen core.rs).
fn street_complete_rs<const N: usize>(s: &EnvStateN<N>) -> bool {
    let cur = s.current_bet;
    let mut n_can_act = 0;
    let mut matched = true;
    let mut all_acted = true;
    for i in 0..N {
        if s.seat_status[i] == SEAT_LIVE {
            n_can_act += 1;
            if s.committed_street[i] < cur {
                matched = false;
            }
            if !s.acted[i] {
                all_acted = false;
            }
        }
    }
    let back_to_aggressor = s.action_seat == s.last_aggressor;
    let no_aggressor = s.last_aggressor < 0;
    (n_can_act == 0) || (matched && all_acted && (back_to_aggressor || no_aggressor))
}

/// `core::run_out` twin (private in frozen core.rs): deal the remaining board from the deck.
fn run_out_rs<const N: usize>(s: &mut EnvStateN<N>) {
    let mut ptr = s.deck_ptr as usize;
    for _ in 0..core::MAX_BOARD {
        let nb = s.board.iter().filter(|&&c| c >= 0).count();
        if nb < core::MAX_BOARD {
            s.board[nb] = s.deck[ptr];
            ptr += 1;
        }
    }
    s.deck_ptr = ptr as i32;
    s.street = core::STREET_TERMINAL;
    s.pending_deal = false;
}

/// `core::close_street` twin (private in frozen core.rs).
fn close_street_rs<const N: usize>(s: &mut EnvStateN<N>) {
    let mut pot_dead = s.pot_dead;
    for i in 0..N {
        pot_dead += s.committed_street[i];
    }
    s.pot_dead = pot_dead;
    s.committed_street = [0; N];
    s.current_bet = 0;
    s.pending_deal = true;
    s.action_seat = -1;
    let actors = s.seat_status.iter().any(|&st| st == SEAT_LIVE);
    if !actors && s.street < core::STREET_RIVER {
        run_out_rs(s);
    }
}

/// Apply one recorded history step, committing the RECORDED amount for bet-family actions.
///
///   * fold / check / call / all-in: the amount is engine-FORCED (no free parameter) — computed
///     from the state via `core::apply_action`; the recorded value is deliberately IGNORED (live
///     records call amounts as bet-TO totals; the engine stores the increment — telemetry finding 2).
///   * bet-family (buckets 3..=9) whose recorded amount EQUALS the engine-recomputed one: delegated
///     to `core::apply_action` verbatim (engine-history passthrough — bit-identical semantics).
///   * bet-family with a DIVERGENT recorded amount: the commit increment is interpreted per
///     `total_convention` (true: recorded is the bet-TO street total, the live convention =>
///     inc = recorded - committed_street; false: recorded IS the increment), clamped to stack, and
///     applied with min-raise/reopen state recomputed from the TRUE committed totals — a local
///     mirror of the frozen `core::apply_action` body (chip flow, acted/full-raise reset,
///     current_bet/min_raise/last_aggressor/last_raise_full, street close, run-out, next actor).
///
/// Returns None on ANY class-legality or chip-conservation violation: dead/folded actor, reopen
/// rule violated, non-positive increment (this includes a total below the seat's already-committed
/// chips), a "bet" that does not exceed the standing bet (call-for-less corruption), or a short
/// raise that is not an all-in. Terminal steps mark `done` and zero rewards (the replay REJECTS a
/// history that terminates before the root, so settle amounts are never consumed).
pub(crate) fn apply_action_recorded<const N: usize>(
    s_in: &EnvStateN<N>,
    action: usize,
    recorded: i32,
    total_convention: bool,
) -> Option<EnvStateN<N>> {
    if s_in.action_seat < 0 || s_in.action_seat >= N as i32 || action >= NUM_ACTIONS {
        return None;
    }
    let seat = s_in.action_seat as usize;
    let full = core::legal_action_mask(s_in, seat as i32);
    let engine_amt = engine_action_amount_rs(s_in, seat, action);

    // Engine-forced classes + exact-amount passthrough: the frozen engine IS the semantics.
    if action < BET_BUCKET_START || action == ALL_IN || recorded == engine_amt {
        if !full[action] {
            return None; // exact class legality per the engine mask.
        }
        return Some(core::apply_action(s_in, action));
    }

    // ---- bet-family with a divergent recorded amount ----
    let mut s = *s_in;
    if s.seat_status[seat] != SEAT_LIVE {
        return None;
    }
    let committed = s.committed_street[seat];
    let to_call = (s.current_bet - committed).max(0);
    let reopen_ok = !s.acted[seat] || s.last_raise_full || to_call == 0;
    if !reopen_ok {
        return None; // facing a non-full raise the seat already acted on: cannot re-raise.
    }
    if recorded < 0 {
        return None;
    }
    let inc = if total_convention {
        recorded - committed // live bet-TO total => chips added this action.
    } else {
        recorded
    };
    if inc <= 0 {
        return None; // chip conservation: a bet must add chips (total below committed refused).
    }
    let amount = inc.min(s.stacks[seat]); // clamp to stack (mirrors core::commit).
    if amount <= 0 {
        return None;
    }
    // commit (core::commit twin: stack/committed flow + all-in marking).
    s.stacks[seat] -= amount;
    s.committed_street[seat] += amount;
    s.committed_total[seat] += amount;
    if s.stacks[seat] <= 0 {
        s.seat_status[seat] = SEAT_ALLIN;
    }

    let prev_bet = s_in.current_bet;
    let new_street_bet = s.committed_street[seat];
    if new_street_bet <= prev_bet {
        return None; // recorded "bet/raise" does not exceed the standing bet: corrupt data.
    }
    let full_raise = new_street_bet >= prev_bet + s.min_raise;
    if !full_raise && s.stacks[seat] > 0 {
        return None; // a short raise is only legal as an all-in.
    }
    s.acted[seat] = true;
    if full_raise {
        s.acted = [false; N];
        s.acted[seat] = true;
    }
    // raised == true here by the new_street_bet > prev_bet check above.
    let raise_size = new_street_bet - prev_bet;
    s.current_bet = new_street_bet;
    s.min_raise = BB_CHIPS.max(raise_size);
    s.last_aggressor = seat as i32;
    s.last_raise_full = full_raise;

    let idx = s.hist_len as usize;
    if idx < core::MAX_HISTORY {
        s.hist_seat[idx] = seat as i32;
        s.hist_action[idx] = action as i32;
        s.hist_amount[idx] = amount;
    }
    s.hist_len = ((idx + 1).min(core::MAX_HISTORY)) as i32;

    let n_live = live_count_rs(&s.seat_status);
    let mut terminal = n_live <= 1;
    if terminal {
        s.street = core::STREET_TERMINAL;
        s.pending_deal = false;
    } else if street_complete_rs(&s) {
        close_street_rs(&mut s);
    }
    terminal = terminal || s.street >= core::STREET_TERMINAL || n_live <= 1;
    s.action_seat = if terminal || s.pending_deal {
        -1
    } else {
        first_live_after_rs(&s.seat_status, s.action_seat)
    };
    s.done = terminal;
    s.rewards = [0; N]; // never consumed: a mid-replay terminal is rejected by the caller.
    Some(s)
}

/// The engine's street-close rule (`core::street_complete`) checks `action_seat ==
/// last_aggressor` BEFORE the actor advances, so after a bet-call(-call) sequence the action
/// returns to the AGGRESSOR, who must CHECK to close the street — a synthetic "aggressor option"
/// node that engine-generated histories contain but REAL live histories never record (real poker
/// closes the street on the last call). True exactly when the replay stands at that node: the
/// actor is the last aggressor, faces no bet, and every live seat has matched and acted — i.e.
/// the actor's CHECK closes the street and commits nothing.
fn phantom_check_applicable<const N: usize>(s: &EnvStateN<N>) -> bool {
    if s.done || core::is_chance_node(s) {
        return false;
    }
    let actor = s.action_seat;
    if actor < 0 || actor >= N as i32 || actor != s.last_aggressor {
        return false;
    }
    let a = actor as usize;
    if s.seat_status[a] != SEAT_LIVE {
        return false;
    }
    if (s.current_bet - s.committed_street[a]).max(0) != 0 {
        return false;
    }
    for i in 0..N {
        if s.seat_status[i] == SEAT_LIVE && (s.committed_street[i] < s.current_bet || !s.acted[i]) {
            return false;
        }
    }
    true
}

/// Apply the phantom aggressor-option CHECK via the frozen engine, then ROLL BACK its history
/// entry (restore the slot, decrement hist_len): the inserted action is synthetic — the live
/// root's hist_len counts only RECORDED actions, and keeping the counts aligned preserves the
/// full final `public_eq` (including hist_len) with zero relaxation. The check itself commits no
/// chips; its only effects are the street close + deal bookkeeping, which is exactly the bridge
/// a live history needs. The blueprint keying never reads the hist arrays (playout::fp_key ->
/// fingerprint state scalars), so the rollback has no keying side effects.
fn apply_phantom_check<const N: usize>(s: &EnvStateN<N>) -> Option<EnvStateN<N>> {
    let full = core::legal_action_mask(s, s.action_seat);
    if !full[core::CHECK] {
        return None;
    }
    let before_len = s.hist_len;
    let idx = before_len as usize;
    if idx >= core::MAX_HISTORY {
        return None; // cannot keep recorded counts aligned past MAX_HISTORY — refuse.
    }
    let mut out = core::apply_action(s, core::CHECK);
    if out.hist_len as usize != idx + 1 {
        return None;
    }
    out.hist_seat[idx] = -1;
    out.hist_action[idx] = -1;
    out.hist_amount[idx] = 0;
    out.hist_len = before_len;
    Some(out)
}

/// Amount-faithful forward replay (Stage 2a): identical skeleton to `replay_with_ante` — same
/// hand-start reconstruction, same actor/bounds checks, same chance handling, same FINAL
/// `public_eq` acceptance — but each step is applied via `apply_action_recorded` under the given
/// amount convention, and the per-step "recorded amount must equal the engine-recomputed amount"
/// verification is REPLACED by the class-legality + chip-conservation checks inside it.
/// `insert_phantom_checks`: when true, the replay inserts the engine's synthetic
/// aggressor-option CHECK (see `phantom_check_applicable`) wherever the engine requires it —
/// before a recorded step whose street the engine has not closed yet, and after the last
/// recorded step while the replay is still on an earlier street than the root. Attempted only
/// AFTER the plain pass fails, so engine-generated histories (which record the closing action)
/// never take it.
fn replay_recorded<const N: usize>(
    root: &EnvStateN<N>,
    ante_each: &[i32; N],
    total_convention: bool,
    insert_phantom_checks: bool,
) -> Option<Vec<ReplayStep<N>>> {
    let hist_len = root.hist_len;
    if root.done || !(0..=core::MAX_HISTORY as i32).contains(&hist_len) {
        return None;
    }
    let hist_len = hist_len as usize;
    let mut s = rebuild_hand_start(root, ante_each)?;

    let mut steps: Vec<ReplayStep<N>> = Vec::with_capacity(hist_len);
    for t in 0..hist_len {
        let mut guard = 0;
        loop {
            while core::is_chance_node(&s) {
                s = core::sample_chance(&s);
                guard += 1;
                if guard > 12 {
                    return None;
                }
            }
            if !(insert_phantom_checks && phantom_check_applicable(&s)) {
                break;
            }
            s = apply_phantom_check(&s)?;
            guard += 1;
            if guard > 12 {
                return None;
            }
        }
        if s.done {
            return None;
        }
        let seat = root.hist_seat[t];
        let action = root.hist_action[t];
        if seat < 0 || seat >= N as i32 || !(0..NUM_ACTIONS as i32).contains(&action) {
            return None;
        }
        if s.action_seat != seat {
            return None;
        }
        steps.push((seat as usize, action as usize, s));
        s = apply_action_recorded(&s, action as usize, root.hist_amount[t], total_convention)?;
        if s.hist_len as usize != t + 1 {
            return None;
        }
    }
    let mut guard = 0;
    loop {
        while core::is_chance_node(&s) {
            s = core::sample_chance(&s);
            guard += 1;
            if guard > 12 {
                return None;
            }
        }
        if !(insert_phantom_checks && s.street < root.street && phantom_check_applicable(&s)) {
            break;
        }
        s = apply_phantom_check(&s)?;
        guard += 1;
        if guard > 12 {
            return None;
        }
    }
    if !public_eq(&s, root) {
        return None;
    }
    Some(steps)
}

/// Stage 2a entry: amount-faithful replay attempts, tried BEFORE the legacy engine-amount attempts
/// (design §2.3a / §6 Stage 2). Convention order is total-then-increment (live records bet-TO
/// street totals; engine-shaped increments are the fallback reading); each convention tries the
/// derived-ante identity first (exact for engine histories — where every step takes the
/// `core::apply_action` passthrough and reproduces the legacy steps bit-for-bit), then zero antes
/// (exact for the ante-free live cash regimes; the totals convention breaks the increment-sum
/// identity, so its derived antes are usually refused and zero antes carry the live path). Every
/// attempt is fully verified per-step + final `public_eq`; a wrong convention or ante guess can
/// only produce None, never a mis-keyed table.
pub(crate) fn replay_recorded_amounts<const N: usize>(root: &EnvStateN<N>) -> Option<Vec<ReplayStep<N>>> {
    let hist_len = root.hist_len;
    if root.done || !(0..=core::MAX_HISTORY as i32).contains(&hist_len) {
        return None;
    }
    let hl = hist_len as usize;
    // Derived-ante identity (mirrors the legacy derivation in `replay_public_states`): valid only
    // under the increment reading of the recorded amounts, but exact for engine histories.
    let mut status0 = [core::SEAT_EMPTY; N];
    for i in 0..N {
        if root.seat_status[i] != core::SEAT_EMPTY {
            status0[i] = SEAT_LIVE;
        }
    }
    let mut stacks0 = [0i64; N];
    for i in 0..N {
        stacks0[i] = root.stacks[i] as i64 + root.committed_total[i] as i64;
    }
    let sb = nth_live_after_rs(&status0, root.button, 1);
    let bb = nth_live_after_rs(&status0, root.button, 2);
    if sb < 0 || bb < 0 || sb == bb {
        return None;
    }
    let mut blind = [0i64; N];
    blind[sb as usize] = (SB_CHIPS as i64).min(stacks0[sb as usize].max(0));
    blind[bb as usize] = (BB_CHIPS as i64).min(stacks0[bb as usize].max(0));
    let mut hist_commit = [0i64; N];
    for t in 0..hl {
        let seat = root.hist_seat[t];
        if seat < 0 || seat >= N as i32 {
            return None;
        }
        hist_commit[seat as usize] += root.hist_amount[t].max(0) as i64;
    }
    let mut ante = [0i32; N];
    let mut derived_ok = true;
    let mut any_nonzero = false;
    for i in 0..N {
        let a = root.committed_total[i] as i64 - blind[i] - hist_commit[i];
        if a < 0 {
            derived_ok = false;
            break;
        }
        if a > 0 {
            any_nonzero = true;
        }
        ante[i] = a as i32;
    }
    for &total_convention in &[true, false] {
        for &phantom in &[false, true] {
            if derived_ok {
                if let Some(v) = replay_recorded(root, &ante, total_convention, phantom) {
                    return Some(v);
                }
            }
            if !derived_ok || any_nonzero {
                if let Some(v) = replay_recorded(root, &[0i32; N], total_convention, phantom) {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// Replay the root's recorded public history. Stage 2a: the AMOUNT-FAITHFUL attempts run FIRST
/// (they replay the client's true recorded amounts — the live path; engine histories take the
/// per-step engine passthrough inside them and reproduce the legacy steps bit-for-bit). The legacy
/// engine-amount attempts are KEPT as the fallback chain, deriving per-seat antes from the chip
/// identity `ante[i] = committed_total[i] - blind[i] - Σ recorded amounts of seat i` (exact for
/// every normal hand; a clamped call-for-less overstates the recorded amount, driving the derived
/// ante negative — in that case retry with zero antes, which is exact for the ante-free cash
/// regimes this serves). Every attempt is fully verified (per-step + final public equality), so a
/// wrong guess can never produce states — only a None (=> the caller falls back to the legacy
/// range behavior).
pub(crate) fn replay_public_states<const N: usize>(root: &EnvStateN<N>) -> Option<Vec<ReplayStep<N>>> {
    // Stage 2a — amount-faithful attempts FIRST (fully verified; None falls through to legacy).
    if let Some(v) = replay_recorded_amounts(root) {
        return Some(v);
    }
    let hist_len = root.hist_len;
    if root.done || !(0..=core::MAX_HISTORY as i32).contains(&hist_len) {
        return None;
    }
    let hl = hist_len as usize;
    // Blind seats/amounts from the reconstructed start occupancy (mirrors replay_with_ante).
    let mut status0 = [core::SEAT_EMPTY; N];
    for i in 0..N {
        if root.seat_status[i] != core::SEAT_EMPTY {
            status0[i] = SEAT_LIVE;
        }
    }
    let mut stacks0 = [0i64; N];
    for i in 0..N {
        stacks0[i] = root.stacks[i] as i64 + root.committed_total[i] as i64;
    }
    let sb = nth_live_after_rs(&status0, root.button, 1);
    let bb = nth_live_after_rs(&status0, root.button, 2);
    if sb < 0 || bb < 0 || sb == bb {
        return None;
    }
    let mut blind = [0i64; N];
    blind[sb as usize] = (SB_CHIPS as i64).min(stacks0[sb as usize].max(0));
    blind[bb as usize] = (BB_CHIPS as i64).min(stacks0[bb as usize].max(0));
    let mut hist_commit = [0i64; N];
    for t in 0..hl {
        let seat = root.hist_seat[t];
        if seat < 0 || seat >= N as i32 {
            return None;
        }
        hist_commit[seat as usize] += root.hist_amount[t].max(0) as i64;
    }
    let mut ante = [0i32; N];
    let mut derived_ok = true;
    let mut any_nonzero = false;
    for i in 0..N {
        let a = root.committed_total[i] as i64 - blind[i] - hist_commit[i];
        if a < 0 {
            derived_ok = false;
            break;
        }
        if a > 0 {
            any_nonzero = true;
        }
        ante[i] = a as i32;
    }
    if derived_ok {
        if let Some(v) = replay_with_ante(root, &ante) {
            return Some(v);
        }
        if !any_nonzero {
            return None; // zero-ante attempt already failed; a retry would be identical.
        }
    }
    replay_with_ante(root, &[0i32; N])
}

/// Blueprint probability of `action` for `seat` at state `s` (holes already substituted), masked +
/// renormalized over the abstracted legal set — the SAME `pcfg.fp_key` serve keying as every other
/// blueprint lookup. Returns None on a blueprint MISS or zero masked mass: the row carries no
/// per-combo information (the caller substitutes the decision's mean hit probability, scale-neutral).
fn blueprint_action_prob<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    abst: &[bool; NUM_ACTIONS],
    action: usize,
    bp: &Blueprint,
    pcfg: &PlayoutConfig,
) -> Option<f64> {
    let key = pcfg.fp_key(s, seat);
    let row = bp.get(&key)?;
    let mut tot = 0.0;
    for a in 0..NUM_ACTIONS {
        if abst[a] {
            tot += row[a].max(0.0);
        }
    }
    if tot <= 1e-12 {
        return None;
    }
    Some(row[action].max(0.0) / tot)
}

/// R1 — build the REACH-CONSISTENT range table: `w_s[combo] = Π_{observed decisions t of seat s}
/// P_bp(a_t | combo, state_t)` over the replayed public history. Hero's decisions are skipped (his
/// hand is a known point mass). Per decision: the abstracted legal mask is hole-independent (computed
/// once); a decision whose observed action falls outside the abstraction, or with no blueprint hits
/// (or all-zero hit mass — e.g. the action is never taken by ANY holding under the blueprint), is
/// skipped as carrying no per-combo information; blueprint-MISS combos factor the decision's mean hit
/// probability (neutral — no information about THAT combo, without distorting the hit combos' scale).
/// Returns None when the history cannot be replayed + verified (caller falls back to legacy behavior).
pub(crate) fn build_range_table_reach<const N: usize>(
    root: &EnvStateN<N>,
    hero: usize,
    known: &[bool; 52],
    bp: &Blueprint,
    pcfg: &PlayoutConfig,
) -> Option<RangeTable> {
    let steps = replay_public_states(root)?;
    let stride = 52 * 52;
    let mut w = vec![0.0f64; N * stride];
    let unseen: Vec<i32> = (0..52).filter(|&c| !known[c as usize]).collect();
    let m = unseen.len();
    if m < 2 {
        return None;
    }
    let mut combos: Vec<(i32, i32)> = Vec::with_capacity(m * (m - 1) / 2);
    for i in 0..m {
        for j in (i + 1)..m {
            combos.push((unseen[i], unseen[j]));
        }
    }
    // Prior: weight 1.0 (no information) for every candidate combo of every non-hero seat — a seat
    // with no observed decisions (or only uninformative ones) stays uniform, exactly like today.
    for seat in 0..N {
        if seat == hero {
            continue;
        }
        let base = seat * stride;
        for &(c0, c1) in &combos {
            w[base + (c0 as usize) * 52 + c1 as usize] = 1.0;
        }
    }
    let mut p = vec![0.0f64; combos.len()];
    let mut hit = vec![false; combos.len()];
    for (seat, action, st) in steps.iter() {
        let (seat, action) = (*seat, *action);
        if seat == hero {
            continue;
        }
        let legal = abstracted_legal(st, seat, pcfg);
        if !legal[action] {
            continue; // observed action outside the search abstraction: no info under this keying.
        }
        let mut scratch = *st;
        let mut sum_hit = 0.0f64;
        let mut n_hit = 0usize;
        let mut max_p = 0.0f64;
        for (k, &(c0, c1)) in combos.iter().enumerate() {
            scratch.holes[seat][0] = c0;
            scratch.holes[seat][1] = c1;
            match blueprint_action_prob(&scratch, seat, &legal, action, bp, pcfg) {
                Some(prob) => {
                    p[k] = prob;
                    hit[k] = true;
                    sum_hit += prob;
                    n_hit += 1;
                    if prob > max_p {
                        max_p = prob;
                    }
                }
                None => hit[k] = false,
            }
        }
        if n_hit == 0 || max_p <= 1e-12 {
            continue; // no per-combo signal at this decision (all-miss, or action prob 0 everywhere).
        }
        let mean_hit = sum_hit / n_hit as f64;
        let base = seat * stride;
        for (k, &(c0, c1)) in combos.iter().enumerate() {
            let f = if hit[k] { p[k] } else { mean_hit };
            w[base + (c0 as usize) * 52 + c1 as usize] *= f;
        }
    }
    Some(RangeTable { w, stride })
}

/// The range-table dispatcher shared by `solve_rows` and `solve_exploit` (and the probe FFI):
///   * `range_reach_full` => the R1 reach table; on replay failure fall back to the legacy behavior
///     implied by `range_aware` alone (fold-now proxy if set, else None => uniform deal);
///   * `range_aware` only => the Q3 fold-now proxy (unchanged);
///   * neither => None (the uniform deal, byte-identical to today).
pub(crate) fn build_rtab<const N: usize>(
    root: &EnvStateN<N>,
    hero: usize,
    bp: &Blueprint,
    pcfg: &PlayoutConfig,
    scfg: &SubgameConfig,
) -> Option<RangeTable> {
    if !(scfg.range_aware || scfg.range_reach_full) {
        return None;
    }
    prof_scope!(crate::prof::P_RTAB);
    let known = hero_board_known(root, hero);
    if scfg.range_reach_full {
        if let Some(rt) = build_range_table_reach(root, hero, &known, bp, pcfg) {
            return Some(rt);
        }
        if !scfg.range_aware {
            return None;
        }
    }
    Some(build_range_table(root, hero, &known, bp, pcfg))
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
    tab.leaf.insert(dkey, (sum, n));
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
        let regs = tab.rows.get(&ckd).map(|r| r.reg).unwrap_or(ZEROS);
        choices[s] = sample_cont_choice(&regs, &ck, t, ctx.public_seed) as i32;
    }
    if env.seat_status[hero] != SEAT_LIVE {
        return biased_profile_value(env, hero, &choices, ctx, tab, nd);
    }
    // Traverser's choice node: branch all 4 continuations (hero's own continuation biases hero's own
    // playout actions too — both hero and opponents choose, per Pluribus).
    let ckd = cont_dig(iset_dig(nd, env, hero));
    let regs = tab.rows.get(&ckd).map(|r| r.reg).unwrap_or(ZEROS);
    let strat = regret_match(&regs, &CONT_LEGAL);
    let mut cv = [0.0f64; NUM_ACTIONS];
    for c in 0..playout::NUM_CONTINUATIONS {
        choices[hero] = c as i32;
        cv[c] = biased_profile_value(env, hero, &choices, ctx, tab, nd);
    }
    let mut ev = 0.0;
    for c in 0..playout::NUM_CONTINUATIONS {
        ev += strat[c] * cv[c];
    }
    let mut newr = regs;
    for c in 0..playout::NUM_CONTINUATIONS {
        newr[c] = (newr[c] + cv[c] - ev).max(0.0);
    }
    let row = tab.rows.entry(ckd).or_insert(ROW_ZERO);
    row.reg = newr;
    // Strategy-sum accumulation: identical gating/weighting to the action nodes (delayed averaging +
    // CFR+ linear weight), so the choice node is a first-class citizen of the served average.
    if t >= ctx.avg_skip {
        let w = if ctx.scfg.linear_averaging {
            (t + 1) as f64
        } else {
            1.0
        };
        for c in 0..playout::NUM_CONTINUATIONS {
            row.ss[c] += w * strat[c];
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
    nd: &NodeDig,
) -> f64 {
    if env.done {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    // R2 — Pluribus biased continuation leaves (gated, default OFF => the legacy uniform leaf below
    // runs byte-identically; `t` is only read on the biased path).
    if ctx.leaf_biased {
        return leaf_value_biased(env, hero, ctx, tab, t, nd);
    }
    // PERF (2026-07-10, VALUE-IDENTICAL): a RIVER chance-node leaf (river betting closed,
    // pending_deal set) is a DETERMINISTIC playout — `core::sample_chance` advances the street past
    // RIVER with n_new=0 and actor=-1, the playout loop breaks immediately, and returns the
    // non-terminal state's rewards[hero], which `apply_action` zeroed. No action-rng is consumed and
    // no board card can be dealt (the resample-board reshuffle fills deck slots that are never
    // read), so every rollout of every visiting world returns EXACTLY 0.0 — the engines' certified
    // shared "unsettled showdown" semantics (see optlog.md; correctness audit spawned separately).
    // Return it directly instead of paying rollout_samples x (2 state copies + reshuffle + 2 rng
    // seedings) per visit plus a leaf-cache entry per leaf. The cache skip is unobservable (entries
    // are solve-internal) and the shared traversal `rng` was never touched by leaf evals (they use
    // derived seeds), so the solve output is bit-identical. GTO path only — solve_exploit's
    // leaf_value_x is deliberately left untouched (firewall).
    if core::is_chance_node(env) && env.street == core::STREET_RIVER {
        return 0.0;
    }
    // TWO-SIDED: the leaf continuation value is computed FROM the traverser's perspective (`hero` here is
    // the current traverser), so the cache MUST be keyed per-traverser or one traverser's value poisons
    // another's (mirrors pluribus_search.py:619-620 `key = (key, int(hero))`). One-sided => seat-agnostic.
    // OPT-2: keyed by the leaf DIGEST; the legacy Vec key is rebuilt only on the eval (miss) path to
    // derive the exact legacy `leaf_seed` (bit-identical rng stream).
    let trav = if ctx.two_sided { Some(hero) } else { None };
    let dkey = leaf_dig(nd, env, trav);
    prof_count!(crate::prof::C_LEAF_HIT); // counts every non-terminal uniform-leaf visit (hits = this - miss)
    // BUG-1 FIX: the cache entry is a RUNNING MEAN (sum, n) over the first `leaf_worlds` VISITING
    // hole-worlds. The env at each visit carries THAT iteration's sampled opponent holes, so while
    // n < M every hit re-evaluates under the CURRENT world (deterministic per-(leaf, n) seed => the
    // CRN/pure-function contract holds) and the mean integrates M independent hole-worlds instead of
    // freezing the first-touch draw. n >= M => frozen mean (M=1 default == legacy first-touch freeze).
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
    prof_count!(crate::prof::C_LEAF_MISS);
    prof_scope!(crate::prof::P_LEAF_PLAYOUT);
    let mut key = leaf_key(env);
    if ctx.two_sided {
        key.push(hero as i64);
    }
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
    tab.leaf.insert(dkey, (sum, n));
    sum / n as f64
}

/// OPT-7 (2026-07-10, BIT-IDENTICAL) — inline fold-terminal value. When the acting seat folds with
/// exactly TWO contesting seats (SEAT_LIVE/SEAT_ALLIN), `apply_action` would mark the folder,
/// find n_live==1 => terminal, and settle via the `n_showdown <= 1` fold-win branch of
/// `compute_returns`: every seat -committed_total, the single remaining eligible seat +total. The
/// traverser's component of that settle is computed here directly — skipping the ~1.3KB child state
/// copy + betting bookkeeping for the single scalar the recursion consumes. Returns None unless the
/// fold is immediately terminal (multiway folds continue the hand — and under the engine's
/// unsettled-showdown semantics a folded hero's DOWNSTREAM value is path-dependent, so no shortcut
/// is valid there). Mirrors compute_returns' fold_win branch exactly (winner = first eligible seat
/// after the fold), so the value is bit-identical to the applied child's rewards[hero] / CHIP_UNIT.
#[inline]
fn fold_terminal_value<const N: usize>(env: &EnvStateN<N>, hero: usize) -> Option<f64> {
    if live_count_rs(&env.seat_status) != 2 {
        return None;
    }
    let folder = env.action_seat;
    debug_assert!(folder >= 0);
    // Winner = first eligible (non-folded, non-empty) seat once the folder is excluded — exactly
    // core::eligible_to_win on the post-fold status (the folder becomes SEAT_FOLDED).
    let mut status = env.seat_status;
    status[folder as usize] = SEAT_FOLDED;
    let elig = core::eligible_to_win(&status);
    let mut winner = 0usize;
    for i in 0..N {
        if elig[i] {
            winner = i;
            break;
        }
    }
    let total: i32 = env.committed_total.iter().sum();
    let mut r = -env.committed_total[hero];
    if winner == hero {
        r += total;
    }
    Some(r as f64 / CHIP_UNIT)
}

#[allow(clippy::too_many_arguments)]
fn cfv<const N: usize>(
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
    }
    let actor = actor_i as usize;
    #[cfg(feature = "prof")]
    let _t_legal = crate::prof::tsc();
    let mut legal = abstracted_legal(env, actor, ctx.pcfg);
    #[cfg(feature = "prof")]
    crate::prof::TSC[crate::prof::P_LEGAL].fetch_add(
        crate::prof::tsc().wrapping_sub(_t_legal),
        std::sync::atomic::Ordering::Relaxed,
    );
    // STAGE-1 ROOT UN-MASK (root_unmask, artifacts/offtree_resolve_design.md §3/§8): at the ROOT
    // node ONLY (depth == 0 — the root actor is always the hero seat), OR the un-mask bits into the
    // abstracted keep-list, ANDed with FULL engine legality so an engine-illegal action can never
    // enter. Applies in BOTH the traverser and non-traverser branches below by position (in
    // two-sided mode the root actor samples from its own regret-match over the same widened mask).
    // 0 (default) => byte-identical.
    if depth == 0 && ctx.root_unmask != 0 {
        let full = core::legal_action_mask(env, actor as i32);
        for a in 0..NUM_ACTIONS {
            if ctx.root_unmask >> a & 1 == 1 && full[a] {
                legal[a] = true;
            }
        }
    }
    if !any_legal(&legal) {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    let key = iset_dig(nd, env, actor);

    if actor != hero {
        prof_count!(crate::prof::C_NODE_OPP);
        let regrets = tab.rows.get(&key).map(|r| r.reg).unwrap_or(ZEROS);
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
        // OPT-7: immediately-terminal fold => the settle's hero component, no child state build.
        if a == FOLD {
            if let Some(v) = fold_terminal_value(env, hero) {
                return v;
            }
        }
        #[cfg(feature = "prof")]
        let _t_ap = crate::prof::tsc();
        let child = subgame_apply(env, a);
        #[cfg(feature = "prof")]
        crate::prof::TSC[crate::prof::P_APPLY].fetch_add(
            crate::prof::tsc().wrapping_sub(_t_ap),
            std::sync::atomic::Ordering::Relaxed,
        );
        let cd = child_dig(nd, env.hist_len, actor, a);
        return cfv(&child, hero, depth + 1, ctx, tab, rng, t, &cd);
    }

    // hero node: branch all (abstracted-)legal actions, CFR+ regret update.
    prof_count!(crate::prof::C_NODE_HERO);
    let regrets = tab.rows.get(&key).map(|r| r.reg).unwrap_or(ZEROS);
    let strat = regret_match(&regrets, &legal);
    let mut av = [0.0f64; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            // OPT-7: immediately-terminal fold => the settle's hero component, no child state build.
            if a == FOLD {
                if let Some(v) = fold_terminal_value(env, hero) {
                    av[a] = v;
                    continue;
                }
            }
            #[cfg(feature = "prof")]
            let _t_ap = crate::prof::tsc();
            let child = subgame_apply(env, a);
            #[cfg(feature = "prof")]
            crate::prof::TSC[crate::prof::P_APPLY].fetch_add(
                crate::prof::tsc().wrapping_sub(_t_ap),
                std::sync::atomic::Ordering::Relaxed,
            );
            let cd = child_dig(nd, env.hist_len, actor, a);
            av[a] = cfv(&child, hero, depth + 1, ctx, tab, rng, t, &cd);
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
    let row = tab.rows.entry(key).or_insert(ROW_ZERO);
    row.reg = newr;
    // BUG-5 (partial) — DELAYED AVERAGING: the SERVED average excludes iterations t < avg_skip
    // (floor(avg_skip_frac * K), 0 = legacy). Regret updates above are NEVER skipped.
    if t >= ctx.avg_skip {
        // CFR+ linear averaging: weight iteration t's strategy by (t+1) (matches the Python solver).
        let w = if ctx.scfg.linear_averaging {
            (t + 1) as f64
        } else {
            1.0
        };
        for a in 0..NUM_ACTIONS {
            row.ss[a] += w * strat[a];
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
    root_nd: &NodeDig,
    root_key: u128,
    traversers: &[usize],
    rtab: Option<&RangeTable>,
    rs: Option<&RangeSampler>,
    stream_seed: u64,
) -> ([f64; NUM_ACTIONS], [f64; NUM_ACTIONS], bool) {
    let scfg = ctx.scfg;
    let mut tab = Tables {
        // Pre-reserve past the early doubling churn (rehash copies showed up on the big river
        // solves); ~3 MB up front, reclaimed at solve end.
        rows: FxHashMap::with_capacity_and_hasher(1 << 14, Default::default()),
        leaf: FxHashMap::with_capacity_and_hasher(1 << 13, Default::default()),
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
            deal_opponent_holes(&mut world, hero, hrng, scfg, rtab, rs);
        }
        if scfg.repair_deck {
            prof_scope!(crate::prof::P_REPAIR);
            // Rebuild the undealt deck tail AFTER the deal (seed + known set both depend on the sampled
            // holes). A SEPARATE Rng stream (deterministic per-world seed) so the solve's iteration `rng`
            // is unperturbed and worlds stay CRN-stable across processes/arms.
            let sd = repair_deck_seed(&world, hero);
            let mut drng = Rng::seed_from_u64(sd);
            playout::reshuffle_undealt(&mut world, &mut drng);
        }
        let traverser = traversers[(t as usize) % traversers.len()];
        cfv(&world, traverser, 0, ctx, &mut tab, &mut rng, t, root_nd);
    }
    // Return the ROOT rows raw (unnormalized) so the ensemble can SUM strat_sum across streams.
    let root_row = tab.rows.get(&root_key).copied().unwrap_or(ROW_ZERO);
    let ss = root_row.ss;
    let has_avg = ss.iter().sum::<f64>() > 1e-12;
    (ss, root_row.reg, has_avg)
}

/// The full CFR solve MINUS the extraction: the ensemble-merged ROOT rows
/// `(strategy_sum, regrets, any-avg-mass)` that `solve` reads its served policy off. Pure code
/// motion from `solve` (behavior byte-identical); `pub(crate)` so the R5a golden test can assert
/// the exact extraction semantics (averaged vs final-iterate) against the raw rows.
pub(crate) fn solve_rows<const N: usize>(
    root: &EnvStateN<N>,
    hero: usize,
    bp: &Blueprint,
    pcfg: &PlayoutConfig,
    scfg: &SubgameConfig,
    public_seed: u64,
) -> ([f64; NUM_ACTIONS], [f64; NUM_ACTIONS], bool) {
    prof_scope!(crate::prof::P_SOLVE);
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
        // STAGE-1 — root-only un-mask (0 default => byte-identical; GTO path only).
        root_unmask: scfg.root_unmask,
    };
    // OPT-2: the root digest lanes are a pure function of the root's public board/street/history —
    // invariant across worlds AND ensemble streams — so they are built once here. The extraction key
    // is the root infoset digest (the digest twin of the legacy `infoset_key(root, hero)`; the root
    // state's spr/hist_len are world-invariant, so this is valid for every world).
    let root_nd = root_dig(root);
    let root_key = iset_dig(&root_nd, root, hero);
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
    // Q3/R1 RANGE TABLE: build the per-seat range table ONCE (invariant across worlds AND across
    // ensemble streams -- a pure blueprint read of the fixed root + its recorded public history).
    // range_reach_full => the R1 blueprint-reach product (fold-now / uniform fallback on replay
    // failure); range_aware only => the Q3 fold-now proxy; neither => None (uniform deal).
    let rtab = build_rtab(root, hero, bp, pcfg, scfg);
    // OPT-4: build the exact-mixture sampler once per solve (pure precompute over rtab).
    let rs = rtab.as_ref().map(|rt| RangeSampler::build(rt, root, hero));

    // Sum the per-stream ROOT strategy_sum (mass-weighted average of N independent CFR averages == variance
    // reduction) and root regrets (for the no-avg fallback). Single stream => stream 0 with seed exactly
    // `public_seed` => byte/stat-identical to the pre-ensemble binary.
    let n_streams = scfg.ensemble_streams.max(1) as usize;
    let (sum_ss, sum_rr, any_avg): ([f64; NUM_ACTIONS], [f64; NUM_ACTIONS], bool) = if n_streams <= 1 {
        let (ss, rr, has) = solve_stream(
            root, hero, &ctx, &root_nd, root_key, &traversers, rtab.as_ref(), rs.as_ref(),
            public_seed,
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
                solve_stream(
                    root, hero, &ctx, &root_nd, root_key, &traversers, rtab.as_ref(),
                    rs.as_ref(), seed,
                )
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
    (sum_ss, sum_rr, any_avg)
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
    let mut legal_root = abstracted_legal(root, hero, pcfg);
    // STAGE-1 ROOT UN-MASK: widen the EXTRACTION mask identically to the depth-0 widening inside
    // `cfv` — the served child must not be zeroed at extraction (0 default => byte-identical).
    if scfg.root_unmask != 0 {
        let full = core::legal_action_mask(root, hero as i32);
        for a in 0..NUM_ACTIONS {
            if scfg.root_unmask >> a & 1 == 1 && full[a] {
                legal_root[a] = true;
            }
        }
    }
    let mut probs = [0.0f64; NUM_ACTIONS];
    if !any_legal(&legal_root) {
        return probs;
    }
    let (sum_ss, sum_rr, any_avg) = solve_rows(root, hero, bp, pcfg, scfg, public_seed);

    // Root strategy = SUMMED strategy_sum (masked, normalized) when any stream had mass, else regret-match
    // fallback on the SUMMED regrets. Identical extraction shape to the single-stream path.
    //
    // R5a — FINAL-ITERATE SERVING (scfg.serve_final_iterate): skip the averaged branch entirely and
    // serve the regret-match of the root's cumulative (CFR+-clamped) regrets AFTER the last iteration
    // — the CFR's current strategy at iteration K, which is EXACTLY the expression the no-average
    // fallback below already computes. Default OFF => the averaged extraction runs byte-identically.
    let mut used_avg = false;
    if !scfg.serve_final_iterate && any_avg {
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
            // R5a serve_final_iterate is likewise a GTO-path knob: solve_exploit's extraction below is
            // HARD-CODED to the certified averaged serve (it never reads scfg.serve_final_iterate),
            // and the exploit FFI does not expose the knob (lib.rs hard-codes it false).
            leaf_biased: false,
            leaf_bias_mult: 5.0,
            // FIREWALL: root_unmask (Stage-1 river overbet / off-tree root widening) is a GTO-path
            // knob; force 0 here regardless of scfg (defense in depth — the exploit FFI hard-codes
            // it 0 too), so the certified exploit tree can never be widened at the root.
            root_unmask: 0,
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
    // Q3/R1 RANGE TABLE (a pure blueprint read of the root + its recorded public history — like
    // range_aware, reach-consistent ranges are GTO-consistent DATA entering only the world sampler;
    // the frozen-read exploit math is untouched, firewall-safe).
    let rtab = build_rtab(root, hero, bp, pcfg, scfg);
    // OPT-4: the exploit path shares the WORLD SAMPLER speedup (distribution-identical; the
    // frozen-read exploit math is untouched — same firewall rationale as range_aware itself).
    let rs = rtab.as_ref().map(|rt| RangeSampler::build(rt, root, hero));
    let iters = scfg.max_iterations.max(0);
    for t in 0..iters {
        let mut world = *root;
        {
            let hrng: &mut Rng = hole_rng.as_mut().unwrap_or(&mut rng);
            deal_opponent_holes(&mut world, hero, hrng, scfg, rtab.as_ref(), rs.as_ref());
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
