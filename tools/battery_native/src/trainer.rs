//! Native 6-max NLHE CFR+ blueprint trainer — Workstream-F's largest remaining green-field.
//!
//! WHY THIS MODULE: the deployed Pluribus search path serves a tabular blueprint
//! (`pokerbot/search/tabular_blueprint_adapter.py`, `dict[bytes16 -> [f64;11]]`). The CFR+ math that
//! produces it already exists and is tested — `subgame.rs` has regret-matching+ (`subgame.rs:91-115`),
//! CFR+ linear averaging (`subgame.rs:448-453`), strategy-sum accumulation, opponent sampling and the
//! recursion shape; `leduc.rs` has the exploitability-gated CFR+ trainer with checkpoint/resume. The
//! genuine gap is plumbing (per the plan, `artifacts/native_cfr_trainer_plan.md`):
//!   1. a PERSISTENT global table surviving millions of traversals (subgame's `Tables` is per-solve);
//!   2. an OUTER deal-sampling loop (subgame loops opponent holes at a FIXED root);
//!   3. serialization into the `{[u8;16] -> [f64;11]}` blueprint format.
//!
//! This module supplies exactly those three pieces, built ON TOP of `core.rs`'s public API (`init_hand`,
//! `apply_action`, `sample_chance`, `is_chance_node`, `compute_returns`) — the bit-identity triad
//! (`core.rs`/`np_engine.py`/`jax_engine.py`) is NOT touched. It does NOT reinvent the CFR+ math: the
//! regret-matching+ / linear-averaging / opponent-sampling primitives are the same shapes as
//! `subgame.rs` and `leduc.rs`.
//!
//! ALGORITHM: external-sampling MCCFR (CFR+). For the chosen traverser seat, branch ALL abstracted-legal
//! actions and do the CFR+ regret update; at every opponent seat sample ONE action from the
//! regret-matching+ strategy; advance chance via `core::sample_chance` (the board runout is baked into
//! each deal root's shuffled deck, so chance is sampled, matching ES-MCCFR). The infoset key is
//! `fingerprint_fast_depth` — BYTE-IDENTICAL to the serve key (`TabularBlueprintAdapter` looks up the
//! same key), so a trained snapshot loads and serves with zero train/serve skew.
//!
//! EQUIVALENCE is STATISTICAL (CI-overlap), NOT bit-identity: a fast native xoshiro RNG (`playout::Rng`)
//! replaces numpy's PCG64 for the deck shuffle + action sampling. This is the documented `marg4` rule
//! (xoshiro vs PCG64 is the bit-identity dead-end); the acceptance gate is CI-overlap, same as `lbr.rs`.
//!
//! 6-max ES-MCCFR has no Nash-exploitability guarantee (>2 players); this is the GTO FLOOR the
//! exploitation overlay sits on, not a Nash claim.

use rayon::prelude::*;
// FxHashMap: rustc's fast non-cryptographic hasher (FxHash) in place of std's SipHash. The CFR+ hot
// path is dominated by `cum_regret`/`strat_sum` lookups on small fixed-width [u8;16] keys, where
// SipHash's per-lookup mixing is the bottleneck. FxHash hashes the SAME keys to a (different but
// equally well-distributed) bucket, so it changes ONLY the internal bucket layout — never the key set,
// the [f64;11] values, the iteration order we rely on, or `snapshot()`/`serialize()` output (the latter
// sorts keys). It carries zero transitive deps. `FxHashMap<K,V>` is a `HashMap<K,V,FxBuildHasher>`; we
// alias it locally so the rest of the module reads exactly as before.
use rustc_hash::FxHashMap as HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::core::{self, EnvStateN, MAX_BOARD, NUM_ACTIONS, NUM_PLAYERS, SEAT_FOLDED, SEAT_LIVE};
// `fingerprint::CHIP_UNIT` is f64 (100.0) — used for the bb reward division exactly as subgame.rs does.
// `core::CHIP_UNIT` is i32 (100) — used for chip-amount arithmetic (start stack).
use crate::fingerprint::{
    fingerprint_fast_depth_memo, fingerprint_fast_depth_postflop_bucket_memo, FpMemo, CHIP_UNIT,
};
// Perfect-hash 7-card evaluator (phevaluator port) — used only by the phe showdown settle.
#[cfg(feature = "phe_showdown")]
use crate::fingerprint::evaluate_cards;
// FrozenEMDLookup is the SAME shared (hole,board)->EMD-bucket blob used by the serve / LBR path. When
// `TrainerCfg.emd_frozen` is `Some`, the trainer routes postflop bucket through this lookup INSTEAD of
// `compute_postflop_bucket` (made-hand rank) — producing byte-identical postflop keys to the LBR (which
// already routes through the same lookup via `lbr.rs`). When `None`, the trainer is byte-identical to
// the no-EMD shipped build.
use crate::emd::FrozenEMDLookup;
use crate::playout::Rng;

use std::cell::RefCell;
use std::time::Instant;

const ZEROS: [f64; NUM_ACTIONS] = [0.0; NUM_ACTIONS];

/// TEST-ONLY observability for the DCFR/RBP prune hook (CHANGE B). Records, across a traversal, how many
/// actions were actually PRUNED and on which streets, plus how many prune-ELIGIBLE candidates were instead
/// EXPLORED because they led directly to a terminal (Pluribus RULE 1b). Used by the prune-rule unit tests
/// to assert: (1a) no prune ever fires on the river or later; (1b) a terminal-leading action is never
/// pruned; (3) only warmed deeply-negative actions are pruned. Compiled out entirely in non-test builds
/// (the `record_prune_*` calls are `#[cfg(test)]` no-ops), so the production hot path is unchanged.
#[cfg(test)]
#[derive(Default, Clone, Copy)]
struct PruneProbe {
    pruned_total: u64,
    pruned_on_river_or_later: u64,
    terminal_leading_explored: u64,
    terminal_leading_pruned: u64,
}
#[cfg(test)]
thread_local! {
    static PRUNE_PROBE: RefCell<PruneProbe> = RefCell::new(PruneProbe::default());
}
#[cfg(test)]
#[inline]
fn prune_probe_reset() {
    PRUNE_PROBE.with(|p| *p.borrow_mut() = PruneProbe::default());
}
#[cfg(test)]
#[inline]
fn prune_probe_get() -> PruneProbe {
    PRUNE_PROBE.with(|p| *p.borrow())
}
/// Record a prune-eligible action's disposition at a hero node on `street`: `pruned` = it was skipped,
/// `terminal_leading` = its child was terminal (RULE 1b applies). Test-only.
#[cfg(test)]
#[inline]
fn record_prune_event(street: i32, pruned: bool, terminal_leading: bool) {
    PRUNE_PROBE.with(|p| {
        let mut pp = p.borrow_mut();
        if pruned {
            pp.pruned_total += 1;
            if street >= core::STREET_RIVER {
                pp.pruned_on_river_or_later += 1;
            }
            if terminal_leading {
                pp.terminal_leading_pruned += 1;
            }
        } else if terminal_leading {
            pp.terminal_leading_explored += 1;
        }
    });
}
#[cfg(not(test))]
#[inline(always)]
fn record_prune_event(_street: i32, _pruned: bool, _terminal_leading: bool) {}

/// Number of shards the global `NlheTables` is partitioned into (write-time partition / sharded-self
/// refactor, WF5 2026-06-28). Fixed compile-time power-of-two so `shard_of` is a cheap mask. This is a
/// PURE PERFORMANCE knob: `serialize` global-sorts across all shards (so the on-disk NLCFR001 blob is
/// shard-count-INVARIANT and checkpoint-portable). The merge produces the SAME key set as the old serial
/// fold and is STATISTICALLY EQUIVALENT to it (see `accumulate_local_into` / `merge_partials` — the
/// per-task chunk grouping sums each key as a tree rather than a strict per-deal left fold, so the result
/// differs from the serial fold only in the last few mantissa bits: measured max abs diff ~6e-14 at
/// nd=200, far inside the compare_trainer_states atol=1e-9 gate; NOT bit-identical in general). 128
/// covers full merge parallelism on a 128-vCPU box; the merge is `par_iter_mut` over the shards.
const N_SHARDS: usize = 128;

type ShardMap = HashMap<[u8; 16], [f64; NUM_ACTIONS]>;

/// Persistent-table precision for the cum_regret rows (feature `f32_regret`). f32 halves the regret-row
/// footprint (88B -> 44B) to relieve the 128-thread memory-traffic contention the metal attribution cert
/// identified, WITHOUT touching strat_sum (which stays f64: under Linear-CFR its O(T^2) growth to ~1.3e13
/// at 62k iters would swamp an f32 accumulator and floor exploitability — the one demonstrated failure
/// mode per the f32-quality-gate verdict). cum_regret is bounded + CFR+ floored (max abs ~2.2e7 projected
/// at 62k, 9 orders below f32 limits) so f32 is safe here. The within-iteration accumulation still happens
/// in f64 (see `radd`) and the on-disk NLCFR001 blob stays f64 (`widen_r` on serialize / `narrow_r` on
/// deserialize) so checkpoints are cross-precision-portable. Default OFF => `RegretT = f64` => the whole
/// module is BYTE-IDENTICAL to the shipped f64 trainer (radd/widen_r/narrow_r reduce to exact identities).
#[cfg(feature = "f32_regret")]
type RegretT = f32;
#[cfg(not(feature = "f32_regret"))]
type RegretT = f64;

/// Persistent cum_regret map (value precision = `RegretT`). Distinct from `ShardMap` (f64) which still
/// holds strat_sum and every within-iteration `LocalDelta`.
type RegretMap = HashMap<[u8; 16], [RegretT; NUM_ACTIONS]>;
const ZEROS_R: [RegretT; NUM_ACTIONS] = [0.0; NUM_ACTIONS];

/// Widen a persistent regret row to f64 for regret-matching / serialization. Identity (plain copy) when
/// `RegretT = f64`, so the f64 build is byte-identical at every read site.
#[inline]
fn widen_r(r: &[RegretT; NUM_ACTIONS]) -> [f64; NUM_ACTIONS] {
    let mut o = [0.0f64; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        o[a] = r[a] as f64;
    }
    o
}

/// Narrow an f64 (on-disk) regret row to the persistent precision on deserialize. Identity when
/// `RegretT = f64`; lossless for an f64 that was itself produced by widening an f32 (round-trip exact).
#[inline]
fn narrow_r(r: &[f64; NUM_ACTIONS]) -> [RegretT; NUM_ACTIONS] {
    let mut o: [RegretT; NUM_ACTIONS] = ZEROS_R;
    for a in 0..NUM_ACTIONS {
        o[a] = r[a] as RegretT;
    }
    o
}

/// f64-accumulate-then-round-store: add an f64 increment to a persistent regret cell, summing in f64 and
/// narrowing to `RegretT` ONCE. When `RegretT = f64` this is exactly `acc + x` (byte-identical f64 build);
/// under f32 it removes intra-accumulation error so only the single store rounds (the verdict's mitigated
/// design). Used by BOTH merge paths AND the serial-fold reference test so the sharded merge and the
/// serial fold stay BIT-IDENTICAL to each other at either precision (same operand order, same narrowing).
#[inline]
fn radd(acc: RegretT, x: f64) -> RegretT {
    ((acc as f64) + x) as RegretT
}

/// Deterministic, portable shard index for a 16-byte fingerprint key. Uses `FxHasher` with explicit
/// `write(&key[..])` (NOT `key.hash()`, whose `Hash for [u8;16]` impl may emit a length prefix and is
/// std-version-dependent) so the byte stream hashed is stable across builds/machines. FxHasher has fixed
/// multiplicative constants and no RandomState, so it is fully deterministic. The result is masked to
/// `N_SHARDS` (a power of two). Only within-process consistency is required (the serialized blob is
/// global-sorted and shard-count-free), so this never affects checkpoint portability.
//
// FORBIDDEN: per-shard sort then concat in `write_map_sharded`. The blob's key order MUST be ONE global
// ascending sort over all keys across all shards (matching the pre-refactor `keys.sort()`), else the
// NLCFR001 blob diverges from recorded goldens and S3 resume breaks.
#[inline]
fn shard_of(key: &[u8; 16]) -> usize {
    use std::hash::Hasher;
    let mut h = rustc_hash::FxHasher::default();
    h.write(&key[..]);
    (h.finish() as usize) & (N_SHARDS - 1)
}

thread_local! {
    /// Per-rayon-thread, process-persistent blake2b-digest memo for the CFR+ hot loop. A pure-function
    /// cache (fingerprint message -> blake2b digest); see the comment at its use site in
    /// `train_iteration_n`. Persisting it across iterations (not per-`par_iter`) keeps the cache warm for
    /// the whole run, so the blake2b compression runs ~once per distinct infoset message instead of once
    /// per node visit. It only ever holds values produced by the real `blake2b16`, so it is byte-identity
    /// neutral. Capacity hint sized to the ~60-70K distinct infoset messages a full 6-max run produces.
    static FP_MEMO: RefCell<FpMemo> =
        RefCell::new(FpMemo::with_capacity_and_hasher(1 << 17, Default::default()));
}

/// Hard per-thread capacity cap for FP_MEMO — a MODE-AGNOSTIC memory bound (NOT env-gated). FP_MEMO is a
/// PURE message->blake2b-digest cache (see `fingerprint::digest_msg_memoed`: a hit returns exactly the
/// digest a miss recomputes via `blake2b16(b"", &msg)`), so CLEARING it can only cost recompute — it can
/// NEVER change any emitted digest or any regret/strat value. Under summary / made-hand keys the distinct-
/// message count equals the trained row count: the LARGEST such run, S1000 (made-hand K1000), asymptotes
/// ~296K rows, S200 = 99,390, and the deployed EMD-K160 blueprint fewer still — all >7x under this cap, so
/// summary / production runs NEVER trip it and stay BYTE-IDENTICAL. Under POKERBOT_FULLHIST_KEY the
/// distinct-message count tracks the billions-scale infoset table and would otherwise duplicate the whole
/// message table PER rayon worker (the H200 OOM at 83.3M rows / 18.4 GiB). 2^21 = 2,097,152 entries
/// * ~150 B ~= 315 MB per thread instead of ~8-12 GB.
const FP_MEMO_CAP: usize = 1 << 21; // 2,097,152

/// Per-deal starting stack (in chips) and ante — fixed-depth 100bb, ante 0 (the deployed blueprint's
/// regime; depth randomization is a future env-gated knob, not in this floor trainer). 100 BB at
/// `core::CHIP_UNIT` (100) chips/BB = 10_000 chips.
const START_STACK_CHIPS: i32 = 100 * core::CHIP_UNIT;
const ANTE_CHIPS: i32 = 0;

/// Immutable trainer config: the abstraction parameters that determine the fingerprint KEY SPACE. Baked
/// in at construction (no `POKERBOT_BETTING_ABSTRACTION` env race) so the trained key space and the
/// serve key space are identical by construction. `depth_edges` empty => the non-depth blueprint.
#[derive(Clone)]
pub struct TrainerCfg {
    /// Postflop card-abstraction bucket count (must equal serve's `num_buckets`, e.g. 200).
    pub num_buckets: i32,
    /// Depth-bucket edges for a depth-aware preflop key; empty for the non-depth blueprint.
    pub depth_edges: Vec<f64>,
    /// Allowed action set preflop / postflop (action abstraction), ANDed into the legal mask. Must match
    /// `build_allowed_action_mask((4,5,6))` / `((0,1,3,4,5))` on the serve side.
    pub allowed_pre: [bool; NUM_ACTIONS],
    pub allowed_post: [bool; NUM_ACTIONS],
    /// Deals fanned out per `train_iteration` (rayon outer loop). Each deal trains all `n_seats`
    /// traverser seats.
    pub n_deals_per_iter: usize,
    /// Base RNG seed; each iteration derives `base_seed ^ t` and each deal derives a sub-seed from it.
    pub base_seed: u64,
    /// Number of seats at the table (full-ring). Default 6 (the deployed blueprint regime); 7 is the
    /// MTT tournament-anchor root. `train_iteration` dispatches on this to the matching const-generic
    /// monomorphization (`EnvStateN<N>` / `init_hand_n::<N>` / `fingerprint_fast_depth::<N>`). The N=6
    /// path is BYTE-IDENTICAL to the pre-generalization 6-seat trainer (same key, same recursion); N=7
    /// keys distinctly via the fingerprint's generic seat encoding. Only 6 and 7 are wired (any other
    /// value falls back to 6 — there is no trained blueprint for other widths). The action dim stays 11
    /// (seat-independent).
    pub n_seats: usize,
    /// Regret-Based Pruning floor (Brown-Sandholm partial pruning), in the trainer's bb regret units.
    /// At a HERO node, an action whose start-of-iteration cumulative regret is `<= prune_threshold`
    /// (deeply negative) is NOT recursed into: it carries 0 strategy mass (regret_match floors negative
    /// regret to 0 on read), so its exact counterfactual value barely moves the regret update for the
    /// positive actions. We assign the pruned action a CONSERVATIVE bound (the node EV over the explored
    /// actions) that keeps its regret <= 0 (so it stays pruned), and SKIP its entire subtree — which
    /// also stops firing the subtree's (dominant) showdown evals, so RBP MULTIPLIES with the phe swap.
    /// Textbook same-convergence-fewer-traversals (values statistically equivalent, NOT byte-identical;
    /// gate by LBR/self-play CI-overlap). `f64::NEG_INFINITY` (or any value no regret reaches) DISABLES
    /// pruning => byte-identical to the no-RBP trainer. Pruning is gated OFF until regrets warm: an
    /// action is only pruned once its regret has actually fallen below the floor.
    pub prune_threshold: f64,
    /// LEVER 9 — parallel (rayon tree-reduce) post-sweep merge of the per-deal partials. The default
    /// `false` keeps the BYTE-IDENTICAL serial merge (partials folded into `self` in deal order). `true`
    /// tree-reduces the partials in parallel, which changes the CROSS-PARTIAL float SUM ORDER (the merged
    /// regret/strat are intermediate sums feeding regret-matching) → STATISTICALLY equivalent, NOT byte-
    /// identical (gate by LBR/self-play CI-overlap, not SHA). Recovers most of the serial Amdahl tail at
    /// large `n_deals` (the reason deals/s flatlines). The capacity reserve + floor are unaffected.
    pub fast_merge: bool,
    /// EMD frozen `(hole,board) -> bucket` lookup consumed by the trainer's postflop fingerprint. `None`
    /// (the default) keeps the made-hand postflop fingerprint path → BYTE-IDENTICAL to the no-EMD shipped
    /// trainer. `Some(lookup)` routes postflop bucket through `lookup.bucket(...)` for flop/turn (falls
    /// back to made-hand for preflop/river/misses), producing postflop keys BIT-IDENTICAL to the LBR serve
    /// path (`lbr.rs::TabularPolicy::probs` consults the same `FrozenEMDLookup`). This is the wire that
    /// closes the train↔serve postflop key skew when the EMD pipeline is active. The lookup is wrapped
    /// in `Arc` so rayon workers share a single read-only map (cheap: hash probe, no copy on clone).
    pub emd_frozen: Option<Arc<FrozenEMDLookup>>,
    /// Batched, memory-level-parallel (MLP) traversal width. `0` or `1` => the BYTE-IDENTICAL recursive
    /// `one_traversal` fold (the shipped default). `>= 2` => step `B = batch_deals` independent
    /// deal-traversals in lockstep so their `cum_regret.get()` probes issue back-to-back and the CPU's
    /// out-of-order engine overlaps the (DRAM-bound) hashmap-probe latencies — a microbench measured
    /// ~1.4-2.2x per-thread recovery in the 128-thread contention regime. The batched path is
    /// BYTE-IDENTICAL to the recursive path: it only interleaves across DIFFERENT deals (independent rng),
    /// never the N heroes of one deal, folds each deal's `LocalDelta` into the task accumulator in
    /// ASCENDING deal-index order, and replicates every float op (regret_match, strat_sum, hero EV, regret
    /// write) verbatim. RBP (a finite `prune_threshold`) is NOT supported batched and FALLS BACK to the
    /// recursive path. Default `0` so every shipped path is unchanged.
    pub batch_deals: usize,
    /// CHANGE A — DYNAMIC WORK-STEALING deal scheduler (throughput; quality-SAFE). `false` (the default,
    /// also overridable to ON via env `POKERBOT_WORKSTEAL=1`) keeps the static
    /// `into_par_iter().with_min_len().fold()` rayon chunking — BYTE-IDENTICAL to today. `true` replaces
    /// that with a persistent `AtomicUsize` deal counter: exactly `rayon::current_num_threads()` workers
    /// are spawned (one per physical core; honors RAYON_NUM_THREADS) and each loops
    /// `d = next.fetch_add(1)` pulling deals DYNAMICALLY, so the measured ~100x per-deal work variance
    /// (preflop fold vs deep multi-street) and the ~8x mean-nodes-per-deal growth during training
    /// rebalance — fixing the load imbalance the static `~1.5 tasks/thread` chunking leaves on the table
    /// (rayon cannot steal mid-chunk once a deep deal starts). Each worker accumulates into its OWN
    /// per-worker sharded `NlheTables` (exactly like the static fold's per-task `acc`), and the per-deal
    /// RNG seed stays a PURE function of `(iter_seed, deal_index d)` — INDEPENDENT of which worker claims
    /// the deal or the claim order — so deal `d` produces an IDENTICAL traversal regardless of scheduling.
    /// The ONLY thing that differs vs the default path is the deal->worker assignment and therefore the
    /// cross-partial merge order (float non-associativity) => STATISTICALLY EQUIVALENT (gate by
    /// LBR/self-play CI-overlap, NOT a byte gate — the same standard as `fast_merge`). `merge_partials` /
    /// `floor_cum_regret` are UNCHANGED. RBP and the batched MLP path compose with it; the profile twin
    /// keeps the static path (work_steal is a pure-throughput scheduler, not a measurement change).
    pub work_steal: bool,
    /// CHANGE B — DISCOUNTED-CFR (DCFR, Brown & Sandholm 2019) signed-regret mode that ACTIVATES the
    /// otherwise-inert `prune_threshold` Regret-Based Pruning hook. `false` (the default, also overridable
    /// to ON via env `POKERBOT_DCFR=1`) keeps the shipped CFR+ positive-floor path (`floor_cum_regret`
    /// clamps regret >= 0 each iteration) — BYTE-IDENTICAL to today, and the prune hook stays INERT (no
    /// read-regret is ever < 0 so a finite `prune_threshold` prunes zero subtrees). `true` keeps SIGNED
    /// regret (negatives allowed) and applies the DCFR discount in the per-iteration pass that the CFR+
    /// path uses for `floor_cum_regret`: positive cum_regret *= t^alpha/(t^alpha+1), negative cum_regret
    /// *= t^beta/(t^beta+1), strat_sum *= (t/(t+1))^gamma — with (alpha,beta,gamma)=(1.5,0,2), matching
    /// `pokerbot/training/shared_hogwild_table.dcfr_weights`. `regret_match` already reads `max(0,.)` so
    /// the strategy naturally uses only positive regret. With signed regret the `prune_threshold` hook
    /// becomes ACTIVE; the trainer applies the 3 Pluribus correctness rules (never prune the last betting
    /// round / terminal-leading actions; decide prune-vs-explore PER-ITERATION with ~5% explore so a
    /// deeply-negative action can REVIVE; only prune a WARMED action whose regret actually fell below the
    /// floor). This is a CANDIDATE that must pass an LBR-at-convergence cert before it could replace CFR+;
    /// no default is flipped. `dcfr_prune_threshold` supplies the (deeply-negative) RBP floor when this is
    /// ON; `dcfr_explore_frac` the fraction of iterations that explore all actions (no pruning).
    pub dcfr: bool,
    /// Regret-Based-Pruning floor used by the DCFR mode (the value handed to the `prune_threshold` hook
    /// when `dcfr` is ON, in bb regret units). Ignored when `dcfr` is OFF (then `prune_threshold` governs
    /// directly, and stays inert under the CFR+ floor). Pluribus used a deeply-negative bound relative to
    /// the regret scale; the default `-300_000.0` bb is ~Pluribus's −300M chip floor at this trainer's
    /// 100-chip CHIP_UNIT (the showdown chip-delta divided by CHIP_UNIT lands regret in bb units).
    pub dcfr_prune_threshold: f64,
    /// Fraction of iterations (in `[0,1]`) the DCFR mode EXPLORES every action (NO pruning) so a
    /// deeply-negative action can revive (Pluribus rule 2). The per-iteration prune/explore decision is a
    /// DETERMINISTIC function of the iteration seed (reproducible across resume), NOT a fresh RNG. Default
    /// `0.05` (Pluribus's ~5%). Ignored when `dcfr` is OFF.
    pub dcfr_explore_frac: f64,
    /// Blueprint-v2 river OCHS abstraction (`river_ochs.rs`). `None` (the default) keeps the river on
    /// the made-hand mirror bucket — BYTE-IDENTICAL to the shipped trainer. `Some(table)` routes RIVER
    /// (5-board-card) postflop buckets through the OCHS nearest-centroid id instead (flop/turn stay on
    /// `emd_frozen`; preflop unchanged). River OCHS bucket ids intentionally share the `b"B"` message
    /// space with flop/turn ids — the street byte in the betting state disambiguates — so the key FORMAT
    /// is unchanged and only the river-row SEMANTICS differ. A blueprint trained with this MUST be served
    /// with the same table (`POKERBOT_RIVER_OCHS`), else serve HITS wrong-semantics river rows (the stamp
    /// guard enforces this). Wrapped in `Arc` so rayon workers share one read-only table.
    pub river_ochs: Option<Arc<crate::river_ochs::RiverOchs>>,
}

/// The persistent, durable trainer state — the genuine gap #1. Survives every traversal; merged
/// (hogwild-style, lock-free) after each rayon sweep. Keyed by the 16-byte fingerprint so a `snapshot()`
/// IS the `TabularBlueprintAdapter` table.
///
/// LEVER 8 NOTE (fused single-map tried + REVERTED): co-locating regret+strat in one map was expected
/// to halve per-node table probes. It does NOT here: the trainer's READ/WRITE split reads regret from
/// the IMMUTABLE shared `read` snapshot but writes deltas to a per-worker `local` map, and a HERO node
/// touches only the regret side while an OPPONENT node touches only the strat side — so there is no
/// per-node same-key double-lookup to fuse away. A measured fused-map build (with `has_regret/has_strat`
/// presence flags to keep serialization byte-identical) was byte-for-byte identical but ~5% SLOWER (the
/// extra presence bytes + the OR-merge outweighed the non-existent lookup saving). Kept the two-map
/// layout.
/// One shard of the global table: the cum_regret + strat_sum rows whose key hashes to this shard.
#[derive(Default, Clone)]
pub struct Shard {
    /// cum_regret precision is `RegretT` (f64 by default; f32 under feature `f32_regret`). strat_sum
    /// stays f64 unconditionally (swamping-safety, see `RegretT`).
    pub cum_regret: RegretMap,
    pub strat_sum: ShardMap,
}

/// Per-deal hot accumulator. UNSHARDED (a single map pair) so the ~190 touched infosets per deal are
/// written without the per-key shard index on the hottest path; the deal's rows are partitioned into
/// shards ONCE at the end of the traversal (see `collect_partials_n`). Field names match `Shard` so the
/// `one_traversal` write sites need zero textual change.
#[derive(Default)]
struct LocalDelta {
    // f64 (ShardMap), NOT RegretT: the within-iteration regret accumulation stays f64. Only the
    // PERSISTENT `Shard.cum_regret` narrows to RegretT, rounded ONCE at the `accumulate_local_into`
    // store (radd). This is the verdict's "accumulate in f64, round-store at the persistent write".
    cum_regret: ShardMap,
    strat_sum: ShardMap,
}

/// The global CFR+ table, partitioned into `N_SHARDS` shards by `shard_of(key)`. Sharding makes the
/// per-iteration partial merge parallel-over-shards (each shard worker owns `shards[s]` exclusively and
/// reads the pre-grouped `partial.shards[s]` — no per-key contention, no N× filter). `iters_done` stays
/// a top-level `u64` (lib.rs:898 reads it directly).
pub struct NlheTables {
    pub shards: Box<[Shard; N_SHARDS]>,
    /// Iterations completed (for resume). The next `train_iteration` continues at `iters_done+1`.
    pub iters_done: u64,
}

impl Default for NlheTables {
    fn default() -> Self {
        // array::from_fn (len 128 > 32) — cannot derive Default for [Shard; 128].
        NlheTables {
            shards: Box::new(std::array::from_fn(|_| Shard::default())),
            iters_done: 0,
        }
    }
}

impl Clone for NlheTables {
    fn clone(&self) -> Self {
        NlheTables {
            shards: Box::new((*self.shards).clone()),
            iters_done: self.iters_done,
        }
    }
}

/// Lightweight wall-clock phase timings for one `train_iteration`.
///
/// This is intentionally coarse and opt-in: it times the rayon traversal/collect phase, the post-sweep
/// partial merge, and the CFR+ floor pass without adding counters inside the recursive hot path.
#[derive(Debug, Clone, Copy)]
pub struct TrainIterProfile {
    pub iters_done: u64,
    pub n_deals: usize,
    pub rayon_threads: usize,
    pub fast_merge: bool,
    pub partials: usize,
    pub partial_cum_rows: usize,
    pub partial_strat_rows: usize,
    pub cum_regret_rows: usize,
    pub strat_sum_rows: usize,
    pub traverse_ms: f64,
    pub merge_ms: f64,
    pub floor_ms: f64,
    pub total_ms: f64,
    pub stats: TraverseStats,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct TraverseStats {
    pub nodes: u64,
    pub terminal_nodes: u64,
    pub chance_nodes: u64,
    pub decision_nodes: u64,
    pub hero_nodes: u64,
    pub opponent_nodes: u64,
    pub no_legal_nodes: u64,
    pub legal_actions_total: u64,
    pub hero_branches: u64,
    pub opponent_samples: u64,
    pub apply_calls: u64,
    pub terminal_settles: u64,
    pub fingerprint_calls: u64,
    pub fp_memo_new_entries: u64,
    pub emd_bucket_queries: u64,
    pub emd_bucket_hits: u64,
    pub emd_bucket_misses: u64,
    pub emd_bucket_fallbacks: u64,
    pub emd_fp_memo_hits: u64,
    pub emd_fp_memo_misses: u64,
    pub max_depth: u64,
}

impl TraverseStats {
    fn merge(&mut self, other: &TraverseStats) {
        self.nodes += other.nodes;
        self.terminal_nodes += other.terminal_nodes;
        self.chance_nodes += other.chance_nodes;
        self.decision_nodes += other.decision_nodes;
        self.hero_nodes += other.hero_nodes;
        self.opponent_nodes += other.opponent_nodes;
        self.no_legal_nodes += other.no_legal_nodes;
        self.legal_actions_total += other.legal_actions_total;
        self.hero_branches += other.hero_branches;
        self.opponent_samples += other.opponent_samples;
        self.apply_calls += other.apply_calls;
        self.terminal_settles += other.terminal_settles;
        self.fingerprint_calls += other.fingerprint_calls;
        self.fp_memo_new_entries += other.fp_memo_new_entries;
        self.emd_bucket_queries += other.emd_bucket_queries;
        self.emd_bucket_hits += other.emd_bucket_hits;
        self.emd_bucket_misses += other.emd_bucket_misses;
        self.emd_bucket_fallbacks += other.emd_bucket_fallbacks;
        self.emd_fp_memo_hits += other.emd_fp_memo_hits;
        self.emd_fp_memo_misses += other.emd_fp_memo_misses;
        self.max_depth = self.max_depth.max(other.max_depth);
    }
}

impl NlheTables {
    pub fn new() -> Self {
        NlheTables::default()
    }

    /// Number of distinct infosets with a strategy-sum row (== the served blueprint size).
    pub fn n_strat_infosets(&self) -> usize {
        self.total_strat_len()
    }

    /// Number of distinct infosets with a cum_regret row. Under v3 `snapshot_avg` (no postflop
    /// strat_sum) THIS is the growing table — every RAM backstop must watch
    /// `max(n_strat_infosets, n_cum_infosets)`, not strat alone (strat is preflop-bounded ~60M
    /// under snapshot_avg while cum keeps growing).
    pub fn n_cum_infosets(&self) -> usize {
        self.total_cum_len()
    }

    /// Total cum_regret rows across all shards.
    fn total_cum_len(&self) -> usize {
        self.shards.iter().map(|s| s.cum_regret.len()).sum()
    }

    /// Total strat_sum rows across all shards.
    fn total_strat_len(&self) -> usize {
        self.shards.iter().map(|s| s.strat_sum.len()).sum()
    }

    /// Iterate every cum_regret (key, row) across all shards (order = shard order then per-shard hash
    /// order; callers that need determinism must sort — `serialize` does).
    fn iter_all_cum(&self) -> impl Iterator<Item = (&[u8; 16], &[RegretT; NUM_ACTIONS])> {
        self.shards.iter().flat_map(|s| s.cum_regret.iter())
    }

    /// Iterate every strat_sum (key, row) across all shards.
    fn iter_all_strat(&self) -> impl Iterator<Item = (&[u8; 16], &[f64; NUM_ACTIONS])> {
        self.shards.iter().flat_map(|s| s.strat_sum.iter())
    }
}

/// Insert a (cum_regret, strat_sum) pair for `key` into the correct shard of `shards`. Used by the hogwild
/// table's `snapshot_tables` to materialize a shared table into the fork-merge `NlheTables` shape (the
/// regret row narrows f64 -> RegretT exactly as `read_map_into_shards` does on deserialize, so the produced
/// table is byte-format-identical to a fork-merge / checkpoint one). `pub(crate)` so the `hogwild` module
/// can reach it without exposing the shard internals publicly. Available unconditionally (so the f64 build
/// without the hogwild feature still compiles it as dead-but-valid code is undesirable — gate it on the
/// feature to avoid an unused-fn warning in the default build).
#[cfg(feature = "hogwild")]
pub(crate) fn shard_insert_snapshot_filtered(
    shards: &mut [Shard; N_SHARDS],
    key: &[u8; 16],
    reg: Option<&[f64; NUM_ACTIONS]>,
    strat: Option<&[f64; NUM_ACTIONS]>,
) {
    let s = shard_of(key);
    if let Some(reg) = reg {
        shards[s].cum_regret.insert(*key, narrow_r(reg));
    }
    if let Some(strat) = strat {
        shards[s].strat_sum.insert(*key, *strat);
    }
}

/// Visit every (key, cum_regret-widened-to-f64, strat_sum) row of `tab` in shard order, calling `f`. Used
/// by the hogwild table's `load_from_tables` to seed the shared table from a resumed checkpoint. The cum
/// row is widened to f64 (identity in the f64 build) so the hogwild table — which stores f64 cells — gets
/// the on-disk-equivalent value. `pub(crate)`; gated on the feature to avoid an unused-fn warning.
#[cfg(feature = "hogwild")]
pub(crate) fn for_each_table_row<F>(tab: &NlheTables, mut f: F) -> Result<(), String>
where
    F: FnMut(&[u8; 16], &[f64; NUM_ACTIONS], &[f64; NUM_ACTIONS]) -> Result<(), String>,
{
    for sh in tab.shards.iter() {
        for (key, reg) in sh.cum_regret.iter() {
            let widened = widen_r(reg);
            let strat = sh.strat_sum.get(key).copied().unwrap_or(ZEROS);
            f(key, &widened, &strat)?;
        }
        // A strat_sum row whose key has NO cum_regret row (possible after CFR+ flooring drops a regret row
        // — though in practice they co-occur) still needs to be loaded so the average strategy is complete.
        for (key, strat) in sh.strat_sum.iter() {
            if !sh.cum_regret.contains_key(key) {
                f(key, &ZEROS, strat)?;
            }
        }
    }
    Ok(())
}

/// Fold `src` shard into `dst` shard by `+=` on co-keyed rows (cum_regret AND strat_sum). The CFR+ floor
/// `max(r,0)` is applied ONCE PER ITERATION in `train_iteration` (after all partials are merged), not
/// per-partial — applying it per-partial is order-dependent and over-inflates regret (mirrors
/// leduc.rs:350-357's single-pass floor). The caller (`merge_partials`) folds partials in DEAL ORDER so
/// the per-key `+=` sum order is identical to the pre-refactor serial fold (bit-identical).
fn merge_shard(dst: &mut Shard, src: &Shard) {
    for (k, dr) in &src.cum_regret {
        let e = dst.cum_regret.entry(*k).or_insert(ZEROS_R);
        for a in 0..NUM_ACTIONS {
            e[a] = radd(e[a], dr[a] as f64);
        }
    }
    for (k, ds) in &src.strat_sum {
        let e = dst.strat_sum.entry(*k).or_insert(ZEROS);
        for a in 0..NUM_ACTIONS {
            e[a] += ds[a];
        }
    }
}

/// Fold one deal's unsharded `LocalDelta` into a per-TASK sharded accumulator `acc` (routing each key
/// to `acc.shards[shard_of(key)]` and `+=`-ing the row). Used by the rayon `fold` in
/// `collect_partials_n`: each rayon task owns ONE `acc` and folds all the deals in its work-chunk into
/// it, so the number of partials handed to `merge_partials` is the number of rayon TASKS (≈ a small
/// multiple of the thread count), NOT `n_deals`. This (a) removes the per-deal `Box<[Shard;128]>`
/// allocation + the `Vec<NlheTables>`-of-`n_deals` collect (the allocator-pressure the perf refuter
/// flagged), and (b) makes the downstream parallel-over-shards merge cheap and SCALABLE (sources =
/// n_tasks, not n_deals — so merge_ms barely grows with n_deals: ~1.4ms at 200, ~3.4ms at 1024, vs the
/// serial fold's 2.4ms / 7.5ms). rayon emits the fold accumulators as CONTIGUOUS ASCENDING deal
/// sub-ranges in index order. EQUIVALENCE (measured, NOT bit-identical in general): each key's final
/// value is summed as a TREE — per-task chunk subtotals then combined — whereas the old serial fold is a
/// strict per-deal left fold; `((a+b)+c)+d` (serial) != `(a+b)+(c+d)` (tree) in IEEE-754, so the two
/// differ in the last few mantissa bits. The KEY SET is identical and the max abs diff is ~6e-14 at
/// nd=200 (verified by an old-vs-new iters=1 A/B) / ~0 at nd=64 — all far inside the
/// `compare_trainer_states` atol=1e-9 production gate. (An earlier comment claimed "bit-identical at
/// 8/12 threads"; that was a coincidence of the nd=64 smoke shape, corrected after the throughput A/B.)
#[inline]
fn accumulate_local_into(acc: &mut NlheTables, local: LocalDelta) {
    for (k, row) in local.cum_regret {
        let e = acc.shards[shard_of(&k)].cum_regret.entry(k).or_insert(ZEROS_R);
        for a in 0..NUM_ACTIONS {
            e[a] = radd(e[a], row[a]);
        }
    }
    for (k, row) in local.strat_sum {
        let e = acc.shards[shard_of(&k)].strat_sum.entry(k).or_insert(ZEROS);
        for a in 0..NUM_ACTIONS {
            e[a] += row[a];
        }
    }
}

// --------------------------------------------------------------------------------------------------
// masks / strategy (same shapes as subgame.rs:71-115 — positive-regret matching over the abstracted
// legal set, uniform fallback)
// --------------------------------------------------------------------------------------------------

/// M-RR (menu re-raise thinning) + turn/river first-raise thinning — the blueprint-v4 action-
/// abstraction discipline (design `blueprint_pluribus_grade_design_v2.md` §4.2.2; Pluribus P-supp
/// pp.12-13 rounds-3-4 rule). ENV-GATED `POKERBOT_MRR=1`, read ONCE per process via OnceLock (same
/// convention as `POKERBOT_FULLHIST_KEY` / `POKERBOT_WORKSTEAL` / `POKERBOT_DCFR`). Default OFF =>
/// `abstracted_legal` returns the legacy pre/post-split mask, BYTE-IDENTICAL to before (because
/// `abstracted_legal_mrr(.., false)` is the exact old body) — so all trainer goldens are unchanged with
/// the env unset.
///
/// TRAIN/SERVE SCOPE (verified, WF2 cold-review): this `abstracted_legal` (`cfg: &TrainerCfg`) is the
/// TRAINER's, used ONLY on the training path. The serve/search path uses a SEPARATE function,
/// `subgame::abstracted_legal` (`pcfg: &PlayoutConfig`, subgame.rs:240), which this env does NOT touch.
/// So `POKERBOT_MRR` thins ONLY training, and serve stays byte-identical whether the env is set or not
/// (that is exactly what Stage 0 needs — it only runs `ab_train`). CONSEQUENCE FOR STAGE 2: the design's
/// serve keep-lists MUST separately mirror this rule in `subgame::abstracted_legal` (promoted to an
/// explicit `InfosetCfg` field in Stage 1). Do NOT assume serve inherits the thinning from this knob —
/// a v4 blueprint trained under thinned menus but served on the full menu is the train/serve action-
/// abstraction skew §4.2.2 exists to prevent.
fn mrr_enabled() -> bool {
    use std::sync::OnceLock;
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| {
        std::env::var("POKERBOT_MRR")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// M-RR re-raise keep-list (POSTFLOP, when FACING a live bet): {FOLD, CHECK, CALL, ALL_IN, 1x-pot} =
/// Pluribus rounds-3-4 re-raise menu {1x pot, all-in} (<=2 sizes). `BET_BUCKET_START + 4` is the 1.0x-
/// pot bucket (cf. `lbr::POT_BET_INDEX`). Mirrors `scripts/bpv4_size_census.py::allowed_actions(mrr=True)`.
fn mrr_reraise_keep() -> &'static [bool; NUM_ACTIONS] {
    use std::sync::OnceLock;
    static M: OnceLock<[bool; NUM_ACTIONS]> = OnceLock::new();
    M.get_or_init(|| allowed_mask(&[4]))
}

/// Turn/river FIRST-raise keep-list: {FOLD, CHECK, CALL, ALL_IN, 1/2-pot, 1x-pot} = Pluribus rounds-3-4
/// first-raise menu {1/2, 1x pot, all-in} (<=3 sizes) = keep-list [1,4] + jam (design §4.2.2 v2 reversal
/// of "keep richer"). The FLOP's first raise keeps the full `cfg.allowed_post` menu (Pluribus's flop is
/// "more coarse" than round 1 but NOT as thin as turn/river). `BET_BUCKET_START + 1` = 1/2-pot
/// (`lbr::HALF_POT_INDEX`).
fn mrr_tr_first_keep() -> &'static [bool; NUM_ACTIONS] {
    use std::sync::OnceLock;
    static M: OnceLock<[bool; NUM_ACTIONS]> = OnceLock::new();
    M.get_or_init(|| allowed_mask(&[1, 4]))
}

#[inline]
fn abstracted_legal<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    cfg: &TrainerCfg,
) -> [bool; NUM_ACTIONS] {
    abstracted_legal_mrr(s, seat, cfg, mrr_enabled())
}

/// The `abstracted_legal` body with M-RR as an explicit parameter (so unit tests can exercise both
/// arms without touching the process-global env OnceLock). `mrr == false` is the EXACT legacy body
/// (pre/post split), preserving byte-identity when the env is unset.
#[inline]
fn abstracted_legal_mrr<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    cfg: &TrainerCfg,
    mrr: bool,
) -> [bool; NUM_ACTIONS] {
    let full = core::legal_action_mask(s, seat as i32);
    // Preflop: legacy menu, UNTOUCHED (served direct, its fineness is load-bearing — design §4.2.2).
    // Postflop under M-RR: choose the thinned menu for this node. F/X/C/AI are in EVERY keep-list, so
    // the masked set is never empty (the same CHECK-fallback invariant the census relies on). `seat` is
    // the acting seat (== `s.action_seat` at every call site — `legal_action_mask` above is computed for
    // it), so `facing_bet` == "a live bet exists this street" == "any raise here is a RE-raise".
    // NOTE (WF2 census-parity): the keep-lists are applied DIRECTLY (not intersected with
    // `cfg.allowed_post`). Identical to the census `legal AND post AND keep` for the deployed post-list
    // [0,1,3,4,5] since both kept bet buckets (id4=½, id7=1x) ARE in it. If a future menu config drops
    // bucket 1 or 4 from `allowed_post`, intersect the keep-lists with it here (Stage-1 `InfosetCfg`).
    let allowed: &[bool; NUM_ACTIONS] = if s.street == 0 {
        &cfg.allowed_pre
    } else if mrr {
        let facing_bet = s.current_bet > s.committed_street[seat];
        if facing_bet {
            mrr_reraise_keep()
        } else if s.street >= 2 {
            mrr_tr_first_keep()
        } else {
            &cfg.allowed_post
        }
    } else {
        &cfg.allowed_post
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
/// (Identical to `subgame::regret_match` / the leduc refresh.)
///
/// LEVER 5 (branchless+SIMD regret_match) — TRIED AND REVERTED. A branchless `pos = regrets.max(0)*legalf`
/// + horizontal-sum + reciprocal-multiply form autovectorized to AVX2 and was statistically equivalent on
/// the strat-checksum / self-play-MAE fingerprints (within the trainer's own seed-to-seed noise floor).
/// BUT the low-bit float reorder shifted a MARGINAL native-LBR convergence smoke gate
/// (`tests/test_native_nlhe_cfr.py::test_smoke_blueprint_below_uniform_random`, trained-LBR vs uniform at
/// 200it/64deals, which the test itself notes is ~±130 bb/100 noisy) across its threshold — the scalar
/// (byte-identical-to-original) build PASSES it, the SIMD build FAILED it. Since the SIMD kernel saved
/// ~0% wall here (regret_match is ~1.4 ns/call and the showdown collapse from levers 2-3 moved the
/// bottleneck off the float kernels entirely), the risk/reward was clearly negative: reverted to the
/// byte-identical scalar form. (The autovectorizable form is preserved in the optimization notes if a
/// future profile ever makes this kernel a top band.)
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
    uniform_over_legal(legal)
}

#[inline]
fn uniform_over_legal(legal: &[bool; NUM_ACTIONS]) -> [f64; NUM_ACTIONS] {
    let nl = legal.iter().filter(|&&x| x).count().max(1) as f64;
    let mut out = [0.0f64; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        out[a] = if legal[a] { 1.0 / nl } else { 0.0 };
    }
    out
}

/// Compact list of legal action indices: `idx[0..n]` are the legal action ids (ascending). Built ONCE
/// per decision node (after `abstracted_legal`) and threaded through `sample_action_compact`, the hero
/// branch loop, and the opponent strat_sum accumulation — so the `[bool;11]` mask is scanned once, not
/// re-scanned at each of the ~3 highest-count call sites. Exact-int: the order is the same ascending
/// legal order the old per-site `for a in 0..11 { if legal[a] }` scans produced.
#[inline]
fn compact_legal(legal: &[bool; NUM_ACTIONS]) -> ([usize; NUM_ACTIONS], usize) {
    let mut idx: [usize; NUM_ACTIONS] = [0; NUM_ACTIONS];
    let mut n = 0usize;
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            idx[n] = a;
            n += 1;
        }
    }
    (idx, n)
}

/// EMD-aware postflop fingerprint, the trainer-side twin of `Lbr6Harness`'s EMD routing in `lbr.rs`.
///
/// When `cfg.emd_frozen` is `Some(lookup)`, postflop bucket comes from `lookup.bucket(...)` for flop/turn
/// (made-hand rank fallback for preflop/river/misses). When `None`, this falls through to the made-hand
/// `fingerprint_fast_depth_memo` path → BYTE-IDENTICAL to the no-EMD shipped trainer.
///
/// The blake2b digest is still routed through `FpMemo` — the message bytes differ from the made-hand
/// path only in the postflop bucket integer (the bucket id from the frozen lookup), so the memo amortizes
/// the same way it does for the made-hand path (one blake2b per distinct infoset message, then FxHash
/// lookups forever after).
///
/// The EMD lookup itself is a hash probe (`FxHashMap<u64, u16>::get`). `FrozenEMDLookup::bucket` invokes
/// `canonical_frozen_key`, which iterates 24 suit permutations and does packed-u64 assembly — measurable
/// per-call cost vs the made-hand `evaluate_cards`. We wrap it with a per-rayon-thread memo keyed by
/// a PACKED `u64` of the canonical-sorted `(hole[2], board[5])` card tuple (each card id 0..51 packs
/// into 6 bits, so 8 cards pack into 48 bits of a `u64`); on a miss we run `lookup.bucket` once and
/// cache the resulting `Option<i32>` bucket id. After warmup every reachable postflop (hole,board) is
/// cached and the per-node cost collapses to a single FxHash probe on an 8-byte key (no canonical-key
/// recompute, no lookup probe). Memo persists across iterations (same pattern as `FP_MEMO`).
pub(crate) type EmdBucketMemo = HashMap<(usize, u64), Option<i32>>;
thread_local! {
    /// Per-rayon-thread memo for `(packed sorted cards) -> EMD bucket id`. The packed key is 8 bytes
    /// (u64), so a `FxHash` probe is ~3-4x cheaper than the `[i32; 7]` key the earlier variant used.
    /// It is paired with the lookup object's address so distinct frozen tables loaded in one process do
    /// not share card-only cache entries.
    /// Distinct reachable postflop (hole,board) combos in a 6-max run are bounded (a few hundred K
    /// at most), so the map warms within the first few iterations and stays hot for the entire run.
    /// ~16 B/entry (u64 key + Option<i32> val) * ~250 K keys ~= 4 MB/thread; well within rayon worker
    /// memory.
    static EMD_BUCKET_MEMO: RefCell<EmdBucketMemo> =
        RefCell::new(EmdBucketMemo::with_capacity_and_hasher(1 << 19, Default::default()));
}
#[inline]
fn emd_bucket_cache_key(
    lookup: &FrozenEMDLookup,
    hole: &[i32; 2],
    board: &[i32],
) -> ((usize, u64), i32, i32) {
    // Pack the canonical-sorted (hole, board) into a u64. Cards are 0..51 (6 bits each, fits 8 cards
    // in 48 bits). Sort hole ascending + sort valid board cards ascending — the bucket id is
    // INVARIANT under both permutations because `canonical_frozen_key` already canonicalizes over
    // suit permutations AND sorts the board internally. -1 (invalid board pad) packs as 0x3F (63).
    let mut h0 = hole[0];
    let mut h1 = hole[1];
    if h1 < h0 {
        let t = h0;
        h0 = h1;
        h1 = t;
    }
    let mut bs = [0i32; 5];
    let mut bn = 0usize;
    for &c in board {
        if c >= 0 {
            bs[bn] = c;
            bn += 1;
        }
    }
    if bn >= 2 {
        let b = &mut bs[..bn];
        b.sort_unstable();
    }
    // Pack: 6 bits each, low-to-high: h0, h1, b0, b1, b2, b3, b4, pad.
    let pack = |c: i32| -> u64 {
        if c < 0 {
            0x3F
        } else {
            (c as u64) & 0x3F
        }
    };
    let mut cards_key: u64 = pack(h0);
    cards_key |= pack(h1) << 6;
    cards_key |= pack(bs[0]) << 12;
    cards_key |= pack(bs[1]) << 18;
    cards_key |= pack(bs[2]) << 24;
    cards_key |= pack(bs[3]) << 30;
    cards_key |= pack(bs[4]) << 36;
    let key = (lookup as *const FrozenEMDLookup as usize, cards_key);
    // Upper bits are unused (48/64 used). No collision risk in the reachable (hole, board) space —
    // every card id is in 0..51, so each 6-bit slot is fully populated.
    (key, h0, h1)
}

#[inline]
fn emd_bucket_memoed(lookup: &FrozenEMDLookup, hole: &[i32; 2], board: &[i32]) -> Option<i32> {
    let (key, h0, h1) = emd_bucket_cache_key(lookup, hole, board);
    EMD_BUCKET_MEMO.with(|m| {
        let map = &mut *m.borrow_mut();
        if let Some(&cached) = map.get(&key) {
            return cached;
        }
        // Miss: lookup.bucket (canonical_frozen_key + 15M-entry probe) runs once per distinct
        // reachable (hole,board), amortized across all subsequent visits.
        let v = lookup.bucket(&[h0, h1], board);
        map.insert(key, v);
        v
    })
}

/// A/B ATTRIBUTION stub mode (feature `ab_stub`, default OFF). POKERBOT_AB_STUB neutralizes ONE phase of the
/// profile-twin traverse so the per-node cost delta vs the unstubbed baseline attributes that phase (the big-box
/// attribution protocol's STEP-2 subtractive harness). 1=emd_const (skip the 152MB frozen-lookup probe -> fixed
/// bucket; SUBTRACTIVE, has counter drift), 4=emd_nomemo (bypass the memo -> a REAL lookup every postflop node;
/// ZERO-drift, measures the probe's per-call cost), 2=regret_zero (skip the cum_regret get), 3=showdown_const
/// (skip the terminal showdown eval). Read ONCE per process. When the feature is OFF this is a const 0 -> every
/// stub branch is dead code -> zero production overhead.
#[cfg(feature = "ab_stub")]
#[inline]
fn ab_stub_mode() -> u8 {
    use std::sync::OnceLock;
    static M: OnceLock<u8> = OnceLock::new();
    *M.get_or_init(|| match std::env::var("POKERBOT_AB_STUB").ok().as_deref() {
        Some("emd_const") => 1,
        Some("regret_zero") => 2,
        Some("showdown_const") => 3,
        Some("emd_nomemo") => 4,
        _ => 0,
    })
}
#[cfg(not(feature = "ab_stub"))]
#[inline(always)]
fn ab_stub_mode() -> u8 {
    0
}

/// CHANGE A runtime toggle: `cfg.work_steal` OR env `POKERBOT_WORKSTEAL=1` (matching the env-var toggle
/// convention used by `POKERBOT_HOGWILD` / `POKERBOT_AB_STUB`). Read ONCE per process (the scheduler choice
/// is a whole-run property; an `OnceLock` keeps the per-iteration check free and stable across resume).
/// Default `cfg.work_steal=false` + unset env => `false` => the static `par_iter().fold()` path (byte-
/// identical to today). The env override is OR-ed with the cfg field so either enables it.
fn work_steal_enabled(cfg: &TrainerCfg) -> bool {
    use std::sync::OnceLock;
    static ENV: OnceLock<bool> = OnceLock::new();
    let env_on = *ENV.get_or_init(|| {
        std::env::var("POKERBOT_WORKSTEAL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    });
    cfg.work_steal || env_on
}

/// CHANGE B runtime toggle: `cfg.dcfr` OR env `POKERBOT_DCFR=1` (same convention as above). Read once per
/// process. Default OFF => the CFR+ positive-floor path is unchanged (byte-identical) and the prune hook
/// stays inert.
fn dcfr_enabled(cfg: &TrainerCfg) -> bool {
    use std::sync::OnceLock;
    static ENV: OnceLock<bool> = OnceLock::new();
    let env_on = *ENV.get_or_init(|| {
        std::env::var("POKERBOT_DCFR")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    });
    cfg.dcfr || env_on
}

/// blueprint-v4 **v3 trainer schedule** (Pluribus-faithful LCFR + hard negative-regret pruning +
/// snapshot averaging — amendment doc `artifacts/blueprint_v4_amendment_v3.md`; primary source
/// `artifacts/pluribus_blueprint_spec_full.md` §1). ALL knobs default-OFF => the trainer is
/// BYTE-IDENTICAL to the shipped CFR+ path (pinned goldens: ab_train 20it/64nd sha256
/// `69ba34fb…` base config, `74a3bafa…` fullhist+MRR config — recorded pre-change in the v3
/// session, re-verified after). Env-gated (read ONCE per process, same OnceLock convention as
/// `POKERBOT_MRR`); unit tests bypass the env via `NlheTables::train_iteration_v3(&cfg, &V3Cfg{..})`.
///
/// The three Pluribus trainer deltas this encodes (all ITERATION-denominated — Pluribus's schedule
/// is wall-clock-denominated, which does not transfer across hardware; the mapping rationale is in
/// the amendment doc §2):
///   1. **LCFR discounting** (`lcfr`, `lcfr_d` = D, `lcfr_w` = W): every D iterations for the first
///      W iterations, ALL regrets and average-strategy rows are multiplied in place by
///      (t/D)/((t/D)+1) — Pluribus P-supp p.14 with minutes -> iterations. In `lcfr` mode regrets
///      are SIGNED (no CFR+ positive floor; floored at `regret_floor` instead) and the
///      average-strategy accumulation is UNWEIGHTED (w=1, not w=t): linear weighting comes from the
///      periodic discount during the LCFR phase, exactly as in Pluribus — CFR+'s w=t linear
///      averaging on top of LCFR discounting would double-apply linearity.
///   2. **Hard negative-regret pruning** (`prune_c` = C in bb, `prune_start`, `prune_explore`):
///      from iteration `prune_start`, ~95% of iterations (per-ITERATION coin, deterministic in the
///      iteration seed) skip traverser actions whose regret <= C, never on the final street and
///      never for terminal-leading actions (the existing RULE 1a/1b/2/3 machinery); ~5% explore
///      fully so pruned actions can revive. `regret_floor` (default C*31/30, mirroring Pluribus's
///      -3.1e8 floor vs -3.0e8 threshold) bounds regret below so unpruning stays reachable.
///   3. **Snapshot averaging** (`snapshot_avg`): postflop strat_sum rows are NOT accumulated
///      (in-memory average is kept for the PREFLOP round only, as Pluribus keeps it for round 1
///      only); the postflop blueprint is the offline average of periodic CURRENT-strategy snapshots
///      (`serialize_current_strategy` at the checkpoint cadence + the `snap_avg` offline tool).
///      This removes the postflop strat_sum map — at H200@20k that map held 347.7M rows vs
///      cum_regret's 184.1M, and the M-RR census bounds PREFLOP strat rows at ~60M — i.e. most of
///      the 400M-row/256GB wall was the postflop average this mode eliminates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V3Cfg {
    /// Pluribus linear-CFR trainer mode: signed regrets + LCFR discount schedule + unweighted
    /// strat accumulation. `false` => CFR+ (positive floor, w=t) — byte-identical legacy.
    pub lcfr: bool,
    /// blueprint-v4 ARM-2 mode (`artifacts/blueprint_v4_arm2_amendment.md`): SIGNED regrets with
    /// the deep `regret_floor` + hard negative-regret pruning, but **NO LCFR discount events and
    /// NO averaging change** — strat accumulation stays CFR+'s w=t linear weighting. This is the
    /// v3-closeout-mandated decomposition: pruning's coverage benefit (v3 postflop miss 61.3% vs
    /// CFR+ 71.2% at 20k) WITHOUT LCFR's shape damage (the discount decays correctly-trained early
    /// opening regrets, so v3 rebuilt the limp-trap FASTER than CFR+ — closeout §3). Ignored when
    /// `lcfr` is true (LCFR mode already implies signed regrets; the two are mutually exclusive at
    /// the launcher). No in-traversal float op changes under this flag (regret_matching already
    /// reads `max(0,·)` and the floor is a post-merge pass), so the batched-path guard needs no new
    /// term: a finite prune threshold already forces the recursive path, and before `prune_start`
    /// the batched float ops are the verbatim legacy ones.
    pub signed_regrets: bool,
    /// LCFR discount interval D in ITERATIONS (Pluribus: 10 minutes). 0 disables discount events
    /// (mode still keeps signed regrets when `lcfr` is true).
    pub lcfr_d: u64,
    /// LCFR discount window W in ITERATIONS (Pluribus: 400 minutes): no discount events after W.
    pub lcfr_w: u64,
    /// Hard negative-regret prune threshold C, in the trainer's bb regret units (Pluribus:
    /// -3.0e8 in THEIR chip units — see the amendment doc for the calibration that produced our
    /// default). NEG_INFINITY = pruning off. Only consulted when `lcfr` is true (CFR+ keeps
    /// regrets >= 0 so a finite C would be inert anyway).
    pub prune_c: f64,
    /// First iteration at which pruning may fire (Pluribus: minute 200 of 400 = half the LCFR
    /// window). Regrets must warm before deeply-negative is meaningful.
    pub prune_start: u64,
    /// Fraction of iterations that EXPLORE fully (no pruning) so pruned actions can revive
    /// (Pluribus: 5%).
    pub prune_explore: f64,
    /// Floor on stored regret in `lcfr` mode (Pluribus: -3.1e8 vs C=-3.0e8, i.e. C*31/30). Keeps
    /// pruned actions unprune-able and bounds the negative tail. NEG_INFINITY = no floor.
    pub regret_floor: f64,
    /// Pluribus rounds-2-4 snapshot averaging: skip postflop strat_sum accumulation (round-1
    /// average stays in memory); postflop blueprint = offline average of current-strategy
    /// snapshots.
    pub snapshot_avg: bool,
}

impl Default for V3Cfg {
    /// The INERT config: every field off => `train_iteration_v3(cfg, &V3Cfg::default())` is
    /// byte-identical to the legacy `train_iteration`.
    fn default() -> Self {
        V3Cfg {
            lcfr: false,
            signed_regrets: false,
            lcfr_d: 0,
            lcfr_w: 0,
            prune_c: f64::NEG_INFINITY,
            prune_start: 0,
            prune_explore: 0.05,
            regret_floor: f64::NEG_INFINITY,
            snapshot_avg: false,
        }
    }
}

impl V3Cfg {
    /// True when any v3 behavior is enabled (used by the batched-MLP fallback guard: the batched
    /// path replicates the LEGACY float ops verbatim and does not support the v3 modes).
    pub fn active(&self) -> bool {
        self.lcfr || self.signed_regrets || self.snapshot_avg
    }

    /// Read the v3 schedule from the environment (the launcher/ab_train surface):
    ///   POKERBOT_LCFR=1           enable the Pluribus LCFR trainer mode
    ///   POKERBOT_SIGNED_REGRETS=1 arm-2 mode: signed regrets + floor + pruning, NO discount,
    ///                             w=t linear averaging kept (mutually exclusive with LCFR)
    ///   POKERBOT_LCFR_D           discount interval D iters (default 250)
    ///   POKERBOT_LCFR_W           discount window W iters (default 10000)
    ///   POKERBOT_PRUNE_C          hard prune threshold C in bb (default off; try the calibrated
    ///                             default from the amendment doc)
    ///   POKERBOT_PRUNE_START      first prune iteration (default W/2, Pluribus's 200/400 ratio)
    ///   POKERBOT_PRUNE_EXPLORE    explore fraction (default 0.05)
    ///   POKERBOT_REGRET_FLOOR     regret floor in bb (default C*31/30 when C finite)
    ///   POKERBOT_SNAPSHOT_AVG=1   enable no-postflop-strat_sum + snapshot averaging
    pub fn from_env() -> V3Cfg {
        fn envf(name: &str) -> Option<f64> {
            std::env::var(name).ok().and_then(|s| s.parse().ok())
        }
        fn envu(name: &str) -> Option<u64> {
            std::env::var(name).ok().and_then(|s| s.parse().ok())
        }
        fn envb(name: &str) -> bool {
            std::env::var(name)
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        }
        let lcfr = envb("POKERBOT_LCFR");
        let signed_regrets = envb("POKERBOT_SIGNED_REGRETS");
        let lcfr_d = envu("POKERBOT_LCFR_D").unwrap_or(250);
        let lcfr_w = envu("POKERBOT_LCFR_W").unwrap_or(10_000);
        let prune_c = envf("POKERBOT_PRUNE_C").unwrap_or(f64::NEG_INFINITY);
        let prune_start = envu("POKERBOT_PRUNE_START").unwrap_or(lcfr_w / 2);
        let prune_explore = envf("POKERBOT_PRUNE_EXPLORE").unwrap_or(0.05);
        let regret_floor = envf("POKERBOT_REGRET_FLOOR").unwrap_or(if prune_c.is_finite() {
            prune_c * (31.0 / 30.0)
        } else {
            f64::NEG_INFINITY
        });
        V3Cfg {
            lcfr,
            signed_regrets,
            lcfr_d,
            lcfr_w,
            prune_c,
            prune_start,
            prune_explore,
            regret_floor,
            snapshot_avg: envb("POKERBOT_SNAPSHOT_AVG"),
        }
    }
}

/// Process-wide v3 schedule from env, read once (same OnceLock convention as `mrr_enabled`).
/// Default (no env set) is `V3Cfg::default()` == fully inert.
fn v3_env() -> &'static V3Cfg {
    use std::sync::OnceLock;
    static V3: OnceLock<V3Cfg> = OnceLock::new();
    V3.get_or_init(V3Cfg::from_env)
}

/// Per-iteration v3 traversal context threaded through `one_traversal` (a Copy 2-bool struct so the
/// hot-path cost is nil). `V3Iter::default()` (both false) preserves the legacy float ops exactly:
/// `unweighted_strat=false` selects `w = t as f64` (the identical expression the old code computed)
/// and `strat_pre_only=false` never skips a strat_sum write.
#[derive(Clone, Copy, Default)]
pub(crate) struct V3Iter {
    /// LCFR mode: average-strategy increments are unweighted (w=1); linear weighting comes from the
    /// periodic discount instead of CFR+'s w=t.
    pub unweighted_strat: bool,
    /// snapshot_avg mode: accumulate strat_sum ONLY at preflop (street 0) nodes.
    pub strat_pre_only: bool,
}

/// DCFR (Discounted CFR, Brown & Sandholm 2019) discount factors for iteration `t` (>= 1). Returns
/// `(pos, neg, strat)`: positive cum_regret is multiplied by `t^alpha/(t^alpha+1)`, negative by
/// `t^beta/(t^beta+1)`, and strat_sum by `(t/(t+1))^gamma`. (alpha,beta,gamma)=(1.5,0,2) — the
/// empirically-best Pluribus setting; BIT-for-BIT the same arithmetic as
/// `pokerbot/training/shared_hogwild_table.dcfr_weights` so the native DCFR mode and the Python reference
/// agree. `beta=0` => `t^0=1` => `neg = 1/2` (negative regret is halved each step). Only invoked when
/// `dcfr` is ON, so the CFR+ default path never touches this.
#[inline]
fn dcfr_weights(t: u64) -> (f64, f64, f64) {
    const ALPHA: f64 = 1.5;
    const BETA: f64 = 0.0;
    const GAMMA: f64 = 2.0;
    let tf = (t as f64).max(1.0);
    let ta = tf.powf(ALPHA);
    let tb = tf.powf(BETA);
    let pos = ta / (ta + 1.0);
    let neg = tb / (tb + 1.0);
    let strat = (tf / (tf + 1.0)).powf(GAMMA);
    (pos, neg, strat)
}

/// Pluribus RULE 2 (per-iteration prune-vs-explore decision): explore ALL actions (NO pruning) in
/// ~`explore_frac` of iterations so a deeply-negative action can revive; prune in the rest. The decision
/// is a DETERMINISTIC function of `iter_seed` (so it is reproducible and resume-stable — NOT a fresh RNG
/// that would diverge after a checkpoint restore). We mix the seed with a fixed odd constant and compare
/// the top-53-bit mantissa-scaled fraction against `explore_frac`. Returns `true` when this iteration
/// should EXPLORE (no pruning). `explore_frac <= 0` => never explore (always prune-eligible);
/// `explore_frac >= 1` => always explore (never prune).
#[inline]
fn dcfr_is_explore_iter(iter_seed: u64, explore_frac: f64) -> bool {
    if explore_frac <= 0.0 {
        return false;
    }
    if explore_frac >= 1.0 {
        return true;
    }
    // SplitMix64-style finalizer on the iteration seed -> a well-distributed u64, then the top 53 bits
    // map to a uniform f64 in [0,1). Deterministic given iter_seed.
    let mut z = iter_seed.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    let u = ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
    u < explore_frac
}

/// Pluribus RULE 1a (never prune the LAST betting round) as a shared predicate used by BOTH the
/// `one_traversal` and `one_traversal_profile` hero hooks (so the production guard and its test are the
/// same code). Returns `true` iff a hero node on `street` may be a prune SITE this iteration: pruning is
/// enabled (`eff_prune` finite) AND the node is BEFORE the river (`street < STREET_RIVER` — street 3 is the
/// final betting round, whose values feed every upstream regret and must always be exact). The further
/// per-action RULE 1b (never prune a TERMINAL-leading action) and RULE 3 (only prune a WARMED, deeply-
/// negative action) are applied inside the loop at each candidate.
#[inline]
fn dcfr_prune_node_allowed(street: i32, eff_prune: f64) -> bool {
    eff_prune.is_finite() && street < core::STREET_RIVER
}

/// The EFFECTIVE Regret-Based-Pruning threshold to use for the iteration with seed `iter_seed`.
///
/// CFR+ default (`dcfr` OFF): returns `cfg.prune_threshold` UNCHANGED — NEG_INFINITY in every shipped cfg
/// (disabling pruning) and inert even if finite (the positive floor keeps read-regret >= 0). The whole
/// function reduces to the identity for the byte-identical default path.
///
/// DCFR mode (`dcfr` ON): on a ~5% explore-iteration (Pluribus rule 2, decided deterministically from the
/// seed) returns NEG_INFINITY so EVERY action is re-explored (a deeply-negative action can revive); on the
/// other ~95% returns the deeply-negative `cfg.dcfr_prune_threshold` so RBP fires. Decided ONCE per
/// iteration (not per action).
///
/// v3 LCFR mode (`v3.lcfr` ON, `dcfr` OFF) or arm-2 signed-regrets mode (`v3.signed_regrets` ON):
/// the SAME per-iteration explore coin, but with the v3 hard prune constants — from iteration
/// `v3.prune_start`, non-explore iterations prune at `v3.prune_c` (finite => hard pruning active;
/// regrets are signed in both modes so it actually fires). Before `prune_start`, or with
/// `prune_c = NEG_INFINITY`, pruning stays off. DCFR takes precedence if both are (mis)configured on.
#[inline]
fn effective_prune_threshold(cfg: &TrainerCfg, iter_seed: u64, t: u64, v3: &V3Cfg) -> f64 {
    if dcfr_enabled(cfg) {
        return if dcfr_is_explore_iter(iter_seed, cfg.dcfr_explore_frac) {
            f64::NEG_INFINITY
        } else {
            cfg.dcfr_prune_threshold
        };
    }
    if (v3.lcfr || v3.signed_regrets) && v3.prune_c.is_finite() && t >= v3.prune_start.max(1) {
        return if dcfr_is_explore_iter(iter_seed, v3.prune_explore) {
            f64::NEG_INFINITY
        } else {
            v3.prune_c
        };
    }
    cfg.prune_threshold
}

#[inline]
fn emd_bucket_memoed_profile(
    lookup: &FrozenEMDLookup,
    hole: &[i32; 2],
    board: &[i32],
    stats: &mut TraverseStats,
) -> Option<i32> {
    let (key, h0, h1) = emd_bucket_cache_key(lookup, hole, board);
    let stub = ab_stub_mode();
    if stub == 1 {
        // emd_const: skip the 152MB frozen-table probe entirely (subtractive S_emd; assigns a fixed bucket).
        stats.emd_bucket_misses += 1;
        return Some(0);
    }
    EMD_BUCKET_MEMO.with(|m| {
        let map = &mut *m.borrow_mut();
        if stub != 4 {
            if let Some(&cached) = map.get(&key) {
                stats.emd_bucket_hits += 1;
                return cached;
            }
        }
        // stub==4 (emd_nomemo): bypass the memo cache -> a REAL lookup.bucket() every call (ZERO-drift,
        // identical bucket) to isolate the frozen-table per-probe cost without any abstraction change.
        stats.emd_bucket_misses += 1;
        let v = lookup.bucket(&[h0, h1], board);
        if stub != 4 {
            map.insert(key, v);
        }
        v
    })
}
#[inline]
pub fn fingerprint_fast_depth_memo_emd<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    num_buckets: i32,
    depth_edges: &[f64],
    emd_frozen: Option<&FrozenEMDLookup>,
    river_ochs: Option<&crate::river_ochs::RiverOchs>,
    memo: &mut FpMemo,
) -> [u8; 16] {
    let nb = s.board.iter().filter(|&&c| c >= 0).count();
    // Blueprint-v2: river OCHS bucket (5 board cards) when a table is configured. Checked FIRST so
    // the river never silently falls through to the made-hand mirror when OCHS is active (the two
    // share bucket-id space; a fall-through would write wrong-semantics rows). `None` (default) is
    // byte-identical to the shipped trainer.
    if let Some(table) = river_ochs {
        if nb == 5 && s.holes[seat][0] >= 0 && s.holes[seat][1] >= 0 {
            if let Some(bucket) =
                crate::river_ochs::river_ochs_bucket_memoed(table, &s.holes[seat], &s.board)
            {
                return fingerprint_fast_depth_postflop_bucket_memo(s, seat, bucket, memo);
            }
        }
    }
    if let Some(lookup) = emd_frozen {
        // Try the frozen EMD bucket for flop/turn postflop. `lookup.bucket` returns `None` for
        // preflop / river / malformed spots — those go through the made-hand path (river is
        // made-hand-equivalent by construction; see `emd::strength_histogram`).
        if nb >= 3 && nb <= 4 && s.holes[seat][0] >= 0 && s.holes[seat][1] >= 0 {
            if let Some(bucket) = emd_bucket_memoed(lookup, &s.holes[seat], &s.board) {
                return fingerprint_fast_depth_postflop_bucket_memo(s, seat, bucket, memo);
            }
        }
    }
    // Made-hand path (also used when EMD is None, when the spot is preflop/river, or when the frozen
    // table doesn't cover the spot). For preflop the made-hand and EMD branches coincide on the
    // weva/depth/betting bytes (bucket < 0 => preflop branch of `fingerprint_message`); for river the
    // bucket id is the made-hand rank percentile — the EMD path would have produced a single-spike
    // histogram = made-hand percentile anyway (see `emd.rs::strength_histogram` river branch).
    fingerprint_fast_depth_memo(s, seat, num_buckets, depth_edges, memo)
}

#[inline]
fn fingerprint_fast_depth_memo_emd_profile<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    num_buckets: i32,
    depth_edges: &[f64],
    emd_frozen: Option<&FrozenEMDLookup>,
    memo: &mut FpMemo,
    river_ochs: Option<&crate::river_ochs::RiverOchs>,
    stats: &mut TraverseStats,
) -> [u8; 16] {
    stats.fingerprint_calls += 1;
    if let Some(table) = river_ochs {
        let nb = s.board.iter().filter(|&&c| c >= 0).count();
        if nb == 5 && s.holes[seat][0] >= 0 && s.holes[seat][1] >= 0 {
            if let Some(bucket) =
                crate::river_ochs::river_ochs_bucket_memoed(table, &s.holes[seat], &s.board)
            {
                return fingerprint_fast_depth_postflop_bucket_memo(s, seat, bucket, memo);
            }
        }
    }
    if let Some(lookup) = emd_frozen {
        let nb = s.board.iter().filter(|&&c| c >= 0).count();
        if nb >= 3 && nb <= 4 && s.holes[seat][0] >= 0 && s.holes[seat][1] >= 0 {
            stats.emd_bucket_queries += 1;
            if let Some(bucket) = emd_bucket_memoed_profile(lookup, &s.holes[seat], &s.board, stats)
            {
                let before = memo.len();
                let out = fingerprint_fast_depth_postflop_bucket_memo(s, seat, bucket, memo);
                if memo.len() == before {
                    stats.emd_fp_memo_hits += 1;
                } else {
                    stats.emd_fp_memo_misses += 1;
                }
                return out;
            }
            stats.emd_bucket_fallbacks += 1;
        }
    }
    fingerprint_fast_depth_memo(s, seat, num_buckets, depth_edges, memo)
}

/// Categorical inverse-CDF sample over a PRE-EXTRACTED compact legal list `idx[0..n]` given `strat` and
/// `u in [0,1)`. Byte-identical sampling to the old `sample_action` (same inverse-CDF, same `u`, same
/// cumulative order over the same ascending legal indices), just without re-scanning the `[bool;11]`
/// mask (the caller already built `idx`/`n`). (Identical decision to `subgame::sample_action`.)
#[inline]
fn sample_action_compact(
    strat: &[f64; NUM_ACTIONS],
    idx: &[usize; NUM_ACTIONS],
    n: usize,
    u: f64,
) -> usize {
    if n == 0 {
        return 0;
    }
    let mut s = 0.0;
    for i in 0..n {
        s += strat[idx[i]];
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

/// Categorical inverse-CDF sample of an action over the legal set given `strat` and u in [0,1).
/// (Identical to `subgame::sample_action`.) Retained for the test-only serve path / public wrapper;
/// the hot trainer loop uses `sample_action_compact` with a pre-extracted index list.
fn sample_action(strat: &[f64; NUM_ACTIONS], legal: &[bool; NUM_ACTIONS], u: f64) -> usize {
    let (idx, n) = compact_legal(legal);
    sample_action_compact(strat, &idx, n, u)
}

// --------------------------------------------------------------------------------------------------
// deal_root — genuine gap #2: a fresh N-seat deal root each traversal (shuffle 52, deal via init_hand_n)
// --------------------------------------------------------------------------------------------------

/// Shuffle a 52-card deck (Fisher-Yates over the xoshiro `Rng`) and deal a fresh N-seat root via
/// `core::init_hand_n::<N>`. Button rotates with the deal index so positions are covered uniformly. All
/// N seats are live (the deployed full-ring blueprint); occupancy/short-handed mixing is a future env
/// knob. The N=6 monomorphization is BYTE-IDENTICAL to the pre-generalization 6-seat `deal_root`
/// (`init_hand_n::<6>` is exactly what `init_hand` aliases, same button-mod, same all-true occupancy).
/// N=7 is the MTT full-ring tournament anchor (7 seats dealt from the same 52-card deck — 14 holes + up
/// to 5 board = 19 cards, well within 52).
fn deal_root<const N: usize>(rng: &mut Rng, deal_idx: u64) -> EnvStateN<N> {
    let mut deck: [i32; 52] = [0; 52];
    for (i, c) in deck.iter_mut().enumerate() {
        *c = i as i32;
    }
    // Fisher-Yates over the whole deck.
    for i in (1..52).rev() {
        let j = rng.next_below((i + 1) as u64) as usize;
        deck.swap(i, j);
    }
    let button = (deal_idx % N as u64) as i32;
    let stacks = [START_STACK_CHIPS; N];
    let occupancy = [true; N];
    // rng_key only seeds the engine's internal Threefry board-deal RNG; chance is dealt from `deck`
    // (sample_chance reads s.deck), so the runout is fully determined by the shuffled deck above. A
    // fixed rng_key keeps init_hand_n deterministic given the deck.
    core::init_hand_n::<N>(button, &stacks, &occupancy, ANTE_CHIPS, &deck, [0u32, 0u32])
}

// --------------------------------------------------------------------------------------------------
// one_traversal — genuine gap #1: the ES-MCCFR recursion accumulating into a passed-in NlheTables
// (adapted from subgame.rs:389-455 `cfv`, but writing into the persistent table, not a per-solve one)
// --------------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------------
// Per-deal hot kernel #1 — multiway showdown eval (cpu128 verdict: the only absolute-throughput lever).
//
// The trainer's terminal settle scores each showdown with a 7-card hand evaluator. `core::compute_returns`
// runs the brute-force 21-combo `evaluate_seven` (`score_five` over every 5-of-7 subset) — measured at
// ~2937 ns/multiway-call and ~47% of single-thread node time (perf_baseline_profile_nodes.txt). NOTE the
// "showdown collapse" language elsewhere in this file is ASPIRATIONAL: RBP is shipped INERT (the CFR+
// positive floor keeps read-regrets >= 0, so a finite prune_threshold prunes ZERO subtrees), so the
// showdown eval actually runs on every multiway terminal. The crate already carries a perfect-hash 7-card
// evaluator, `fingerprint::evaluate_cards` (a phevaluator port, already pub + used by the EMD/bucketer
// path) — ~4 table loads, tens of ns, no enumeration. `compute_returns_phe` swaps ONLY the per-seat
// scoring to that lookup; every chip of the fold / side-pot-level / award / odd-chip math reuses the
// `pub(crate)` `core::*` helpers unchanged. The swap is PURE-INTEGER CHIP-IDENTITY (the per-seat [i32;N]
// chip-delta is byte-equal to `core::compute_returns`, proven by the 20000-trial
// `showdown_returns_matches_compute_returns` gate): there is NO float reduction anywhere in the
// showdown->award path, which is structurally WHY it is safe where the reverted AVX2 regret_match (a
// non-associative f64 sum) was not. `core::compute_returns` / `core::evaluate_seven` stay BYTE-IDENTICAL
// for serve/LBR/AIVAT (which compare the raw base-15 score magnitude); the swap is TRAINER-LOCAL and gated
// by `feature = "phe_showdown"` (default ON) so the byte-identical scalar path is a one-rebuild A/B
// fallback. This is the algorithmic lever the cpu128 probe pointed at — NOT portable-SIMD (the perfect-hash
// lookup strictly dominates SIMD-ing the brute force and needs no nightly toolchain).
// --------------------------------------------------------------------------------------------------

/// PHE per-seat showdown score, rank-order-equivalent to `core::player_showdown_score` (higher = better).
///
/// `fingerprint::evaluate_cards` returns phevaluator's 1 (best) .. 7462 (worst); map it to a POSITIVE
/// 1..7462 via `7463 - rank` so HIGHER = BETTER, matching `score_five`'s sense and — critically —
/// `core::award_layer`'s guards: ineligible seats carry the sentinel `-1` and a winner needs
/// `scores[i] == best && best >= 0`. A bare negation (range -7462..-1) would make `best >= 0` never true
/// (nobody wins; the pot vanishes; output is non-zero-sum) and collide royal-flush rank 1 -> -1 with the
/// ineligible sentinel. The trainer's multiway settle is ALWAYS reached with a FULL 5-card board
/// (`run_out` fills the board to `MAX_BOARD` before `STREET_TERMINAL`; `sample_chance` reaches terminal
/// only after the river), and the chip-identity gate likewise always uses 5 board cards — so feed
/// `evaluate_cards` the real 7 cards. Never feed it the card-0 padding `core::player_showdown_score` uses
/// for invalid slots: on a partial board that diverges from the padded-7 ordering, and a duplicated rank
/// can index phe's DP table out of bounds. The `debug_assert` pins the full-board invariant.
#[cfg(feature = "phe_showdown")]
#[inline]
fn player_showdown_score_phe(holes: &[i32; 2], board: &[i32; MAX_BOARD]) -> i32 {
    let mut cards = [0i32; 7];
    cards[0] = holes[0];
    cards[1] = holes[1];
    let mut n = 2usize;
    for i in 0..MAX_BOARD {
        if board[i] >= 0 {
            cards[n] = board[i];
            n += 1;
        }
    }
    debug_assert_eq!(
        n, 7,
        "phe showdown settle requires a full 5-card board; got {n} cards (holes={holes:?} board={board:?})"
    );
    7463 - evaluate_cards(&cards[..n])
}

/// Chip-identical copy of `core::compute_returns` with the ONLY change being the per-seat scorer
/// (`player_showdown_score` -> `player_showdown_score_phe`). Every other line — fold-win branch, side-pot
/// level loop, `cap_contributions` / `award_layer` / `eligible_to_win` (the `pub(crate)` `core::*`
/// helpers) — is reused unchanged, so the chip-identity surface is exactly the one swapped scorer.
/// Returns the per-seat net chip delta (sums to zero), byte-equal to `core::compute_returns`.
#[cfg(feature = "phe_showdown")]
fn compute_returns_phe<const N: usize>(
    holes: &[[i32; 2]; N],
    board: &[i32; MAX_BOARD],
    seat_status: &[i8; N],
    committed_total: &[i32; N],
    button: i32,
    n_showdown: i32,
) -> [i32; N] {
    if n_showdown <= 1 {
        // fold_win: winner = first eligible (argmax of live as int) — identical to core::compute_returns.
        let live = core::eligible_to_win(seat_status);
        let mut winner = 0usize;
        for i in 0..N {
            if live[i] {
                winner = i;
                break;
            }
        }
        let total: i32 = committed_total.iter().sum();
        let mut out = [0i32; N];
        for i in 0..N {
            out[i] = -committed_total[i];
        }
        out[winner] += total;
        return out;
    }

    // showdown — identical to core::compute_returns except the per-seat scorer below.
    let eligible_mask = core::eligible_to_win(seat_status);
    let mut scores = [-1i32; N];
    for s in 0..N {
        if eligible_mask[s] {
            scores[s] = player_showdown_score_phe(&holes[s], board);
        }
    }

    let capped = core::cap_contributions(committed_total);
    let mut excess = [0i32; N];
    for i in 0..N {
        excess[i] = committed_total[i] - capped[i];
    }
    let mut winnings = [0i32; N];

    let mut positive = [0i32; N];
    for i in 0..N {
        positive[i] = if capped[i] > 0 { capped[i] } else { -1 };
    }
    let mut uniq_vec: Vec<i32> = positive.to_vec();
    uniq_vec.sort_unstable();
    uniq_vec.dedup();
    let mut levels = [-1i32; N];
    let take = uniq_vec.len().min(N);
    for i in 0..take {
        levels[i] = uniq_vec[i];
    }
    levels.sort_unstable();
    for i in 0..N {
        if levels[i] < 0 {
            levels[i] = -1;
        }
    }

    let mut prev_level: i32 = 0;
    for li in 0..N {
        let level = levels[li];
        let valid = level >= 0;
        let layer_size = if valid { level - prev_level } else { 0 };
        let mut n_contrib = 0i32;
        let mut contributing = [false; N];
        for i in 0..N {
            contributing[i] = capped[i] >= level && valid;
            if contributing[i] {
                n_contrib += 1;
            }
        }
        let pot_amt = layer_size * n_contrib;
        let mut eligible = [false; N];
        for i in 0..N {
            eligible[i] = (seat_status[i] == core::SEAT_LIVE || seat_status[i] == core::SEAT_ALLIN)
                && contributing[i];
        }
        let win = core::award_layer(pot_amt, &eligible, &scores, button);
        for i in 0..N {
            winnings[i] += win[i];
        }
        if valid {
            prev_level = level;
        }
    }

    let mut out = [0i32; N];
    for i in 0..N {
        out[i] = winnings[i] + excess[i] - committed_total[i];
    }
    out
}

/// Trainer-local terminal settle. The fold-win branch is inlined (no eval); the multiway branch routes
/// through the perfect-hash `compute_returns_phe` (feature `phe_showdown`, default ON) — chip-identical to
/// `core::compute_returns` per the `showdown_returns_matches_compute_returns` gate — falling back to the
/// byte-identical 21-combo `core::compute_returns` when the feature is off. This is the trainer's #1
/// per-deal compute lever; `core::compute_returns` is left untouched for serve/LBR/AIVAT.
#[inline]
fn showdown_returns<const N: usize>(
    holes: &[[i32; 2]; N],
    board: &[i32; MAX_BOARD],
    seat_status: &[i8; N],
    committed_total: &[i32; N],
    button: i32,
    n_showdown: i32,
) -> [i32; N] {
    if n_showdown <= 1 {
        // fold-win: scores are unused; identical to core::compute_returns' fold branch.
        let mut winner = 0usize;
        for i in 0..N {
            if seat_status[i] == core::SEAT_LIVE || seat_status[i] == core::SEAT_ALLIN {
                winner = i;
                break;
            }
        }
        let total: i32 = committed_total.iter().sum();
        let mut out = [0i32; N];
        for i in 0..N {
            out[i] = -committed_total[i];
        }
        out[winner] += total;
        return out;
    }
    #[cfg(feature = "phe_showdown")]
    {
        compute_returns_phe(holes, board, seat_status, committed_total, button, n_showdown)
    }
    #[cfg(not(feature = "phe_showdown"))]
    {
        core::compute_returns(holes, board, seat_status, committed_total, button, n_showdown)
    }
}

/// Trainer-local `apply_action`: the SHARED, byte-identical betting state machine via
/// `core::apply_action_with`, with the IN-APPLY terminal settle (SITE 1) routed through the perfect-hash
/// `compute_returns_phe` (feature `phe_showdown`, default ON) instead of the scalar 21-combo
/// `core::compute_returns`. SITE 1 = the multiway all-in run-out terminals reached INSIDE `apply_action`
/// (`close_street` -> `run_out` then `env.done -> env.rewards`); fold-win terminals take the cheap
/// `n <= 1` branch (no eval) in both. Chip-identical to the scalar engine (gate:
/// `apply_action_with_phe_matches_scalar_byte_for_byte` over random legal lines, biased to all-in
/// showdowns), so the trainer output is byte-unchanged — this is a pure per-terminal speed lever and
/// COMPOUNDS with the SITE 2 phe at the `terminal_return` leaf (`showdown_returns`). `core::apply_action`
/// (= `apply_action_with(.., compute_returns)`) is left untouched for serve/LBR/np_engine-parity.
/// Feature OFF falls back to the scalar `core::apply_action` (the A/B baseline).
#[cfg(all(feature = "phe_showdown", feature = "phe_site1"))]
#[inline]
fn trainer_apply<const N: usize>(env: &EnvStateN<N>, action: usize) -> EnvStateN<N> {
    core::apply_action_with(env, action, compute_returns_phe::<N>)
}
// Site-1 scalar (site-2 phe stays on) = the pre-site-1 baseline; the A/B arm + a kill-switch.
#[cfg(all(feature = "phe_showdown", not(feature = "phe_site1")))]
#[inline]
fn trainer_apply<const N: usize>(env: &EnvStateN<N>, action: usize) -> EnvStateN<N> {
    core::apply_action(env, action)
}
#[cfg(not(feature = "phe_showdown"))]
#[inline]
fn trainer_apply<const N: usize>(env: &EnvStateN<N>, action: usize) -> EnvStateN<N> {
    core::apply_action(env, action)
}

/// Exact terminal return for `hero` (in bb) at a settled state. Counts the seats still eligible for the
/// pot (SEAT_LIVE or SEAT_ALLIN) as the showdown size and calls the trainer-local `showdown_returns`
/// (phe fast eval, chip-identical to `core::compute_returns`) — so a checked-down/called showdown yields
/// a REAL value, never the stale `env.rewards`. (Used by the post-chance terminal + the no-actor/no-legal
/// defensive leaves; FIX 1.) `core::compute_returns` is left for serve/LBR; this is the trainer hot path.
#[inline]
fn terminal_return<const N: usize>(env: &EnvStateN<N>, hero: usize) -> f64 {
    if ab_stub_mode() == 3 {
        return 0.0; // showdown_const: skip the terminal (site-2) showdown eval (subtractive attribution stub)
    }
    let n_show = env
        .seat_status
        .iter()
        .filter(|&&s| s == core::SEAT_LIVE || s == core::SEAT_ALLIN)
        .count() as i32;
    let rets = showdown_returns(
        &env.holes,
        &env.board,
        &env.seat_status,
        &env.committed_total,
        env.button,
        n_show,
    );
    rets[hero] as f64 / CHIP_UNIT
}

/// External-sampling MCCFR recursion for `hero` (the traverser). Branches all abstracted-legal actions
/// at hero nodes (CFR+ regret update + linear-averaged strat-sum, weight `t`), samples ONE action at
/// opponent nodes, advances chance via `sample_chance`, and reads terminal rewards from the engine.
///
/// READ/WRITE SPLIT (the perf+concurrency fix): regret-matching READS `cum_regret` from `read` — the
/// IMMUTABLE start-of-iteration snapshot shared `&` across all rayon workers (never cloned per deal) —
/// and WRITES only the touched regret / strat-sum DELTAS into `local`, which starts EMPTY so `local` IS
/// the per-worker delta directly (no full-table clone, no `delta(read, local)` diff): per-deal cost drops
/// from O(table size) to O(touched infosets), which is what eliminates the per-iteration blowup that
/// presented as a hang on long runs.
///
/// This makes the read side EXACTLY the documented contract (`train_iteration`: "the READ side of each
/// traversal sees the table as of the START of this iteration"). The old code seeded each deal's `local`
/// with a clone of `read` and read regrets back out of `local`, so the 6 hero traversals WITHIN one deal
/// were NOT independent — a later hero saw the regret deltas an earlier hero had just written for a shared
/// infoset. That intra-deal cross-read was a deviation from textbook ES-MCCFR (within an iteration the
/// strategy is frozen at the start-of-iteration table); reading from the immutable `read` snapshot here
/// restores that. The convergence / LBR-exploitability / checkpoint-resume gates all hold under the new
/// (correct) read semantics. Returns hero's counterfactual value (in bb).
///
/// `eff_prune` is the EFFECTIVE Regret-Based-Pruning threshold for THIS iteration. The default (CFR+) path
/// passes `cfg.prune_threshold` verbatim, so the shipped behavior is BYTE-IDENTICAL (the prune hook is
/// inert under the positive floor — no read-regret is ever < 0). DCFR mode passes `cfg.dcfr_prune_threshold`
/// on prune-iterations and `NEG_INFINITY` on the ~5% explore-iterations (Pluribus rule 2), so the prune
/// decision is made PER ITERATION (not per action) by the caller.
fn one_traversal<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    cfg: &TrainerCfg,
    read: &NlheTables,
    local: &mut LocalDelta,
    rng: &mut Rng,
    t: u64,
    memo: &mut FpMemo,
    eff_prune: f64,
    v3i: V3Iter,
) -> f64 {
    if env.done {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    if core::is_chance_node(env) {
        let child = core::sample_chance(env);
        return one_traversal(&child, hero, cfg, read, local, rng, t, memo, eff_prune, v3i);
    }
    // Post-chance terminal: `sample_chance` advances a checked/called river to STREET_TERMINAL with
    // action_seat=-1, done=FALSE, and STALE rewards=[0,...] (the engine fills `rewards` only on the
    // fold/all-in settle paths, not after the runout). If we fell through to the `actor_i<0` leaf below
    // we'd return env.rewards[hero]==0 for EVERY checked-down/called showdown — training against EV=0.
    // Detect it here and compute the real showdown returns directly. (FIX 1)
    if env.street >= core::STREET_TERMINAL {
        return terminal_return(env, hero);
    }
    let actor_i = env.action_seat;
    if actor_i < 0 {
        // No actor and not a chance/terminal node: an unexpected defensive leaf (mirrors subgame.rs's
        // `actor_i < 0` guard). Use the engine's exact returns rather than the possibly-stale
        // `env.rewards` so a stray no-actor state still yields a correct showdown value.
        return terminal_return(env, hero);
    }
    let actor = actor_i as usize;
    let legal = abstracted_legal(env, actor, cfg);
    if !any_legal(&legal) {
        return terminal_return(env, hero);
    }
    let key = fingerprint_fast_depth_memo_emd(
        env,
        actor,
        cfg.num_buckets,
        &cfg.depth_edges,
        cfg.emd_frozen.as_deref(),
        cfg.river_ochs.as_deref(),
        memo,
    );

    // Compact legal-index list, built ONCE per node and reused by the strategy sample / hero branch /
    // strat_sum accumulation below — the `[bool;11]` mask is scanned here a single time instead of at
    // each of those call sites. (Lever 4: exact-int loop restructure; same ascending legal order.)
    let (lidx, ln) = compact_legal(&legal);

    // Read the start-of-iteration regret row from the IMMUTABLE shared `read` snapshot (`ZEROS_R` if
    // absent), widening to f64 for regret-matching. Read once and reused by both branches below. The
    // probe loads `RegretT`-width bytes (44B under f32_regret vs 88B f64) — this halved read is the
    // memory-traffic relief; `widen_r` is a no-op copy in the f64 build.
    let regrets = widen_r(read.shards[shard_of(&key)].cum_regret.get(&key).unwrap_or(&ZEROS_R));

    if actor != hero {
        // Opponent node: regret-matching+ strategy, sample ONE action (external sampling). Regrets are
        // read from the IMMUTABLE iteration snapshot `read` (not `local`) — `local` holds only this
        // worker's deltas and is never the read source.
        let strat = regret_match(&regrets, &legal);
        // CFR+ linear averaging (OpenSpiel kSimple): accumulate the CURRENT PLAYER's strategy at the
        // OPPONENT's node, weighted by the iteration `w = t`, BEFORE sampling. `train_iteration` runs
        // `one_traversal` for ALL 6 heroes per deal, so every seat is the opponent in the other heroes'
        // traversals → each infoset gets the correct external-sampling visitation weight. (FIX 2)
        // Written as a DELTA into `local` (starts empty) — `+=` accumulates the strat-sum mass this
        // worker adds; the iteration merge sums all workers' deltas into the persistent table.
        // LEVER 6(b) (FMA strat_sum AXPY) — TRIED AND REVERTED with levers 5/6(a): same finding (stat-
        // equiv but flipped the marginal LBR smoke, ~0% wall). Scalar byte-identical-to-original form.
        //
        // v3 (default-off): `strat_pre_only` (snapshot_avg mode) skips the postflop strat_sum write
        // entirely — the round-1 (preflop) average stays in memory, rounds 2-4 are served from
        // offline-averaged snapshots (Pluribus P-supp p.15). `unweighted_strat` (LCFR mode) uses w=1:
        // linear weighting comes from the periodic LCFR discount, not CFR+'s w=t. With `V3Iter::
        // default()` this block computes EXACTLY the legacy `w = t as f64` accumulation.
        if !v3i.strat_pre_only || env.street == 0 {
            let w = if v3i.unweighted_strat { 1.0 } else { t as f64 };
            let ss = local.strat_sum.entry(key).or_insert(ZEROS);
            for a in 0..NUM_ACTIONS {
                if legal[a] {
                    ss[a] += w * strat[a];
                }
            }
        }
        let u = rng.next_f64();
        let a = sample_action_compact(&strat, &lidx, ln, u);
        let child = trainer_apply(env, a);
        return one_traversal(&child, hero, cfg, read, local, rng, t, memo, eff_prune, v3i);
    }

    // Hero node: branch ALL abstracted-legal actions, CFR+ regret update. Regrets are read from the
    // IMMUTABLE iteration snapshot `read` (start-of-iteration strategy, read once above), never `local`.
    let strat = regret_match(&regrets, &legal);
    let mut av = [0.0f64; NUM_ACTIONS];
    // LEVER 7 — Regret-Based Pruning (Brown & Sandholm 2019), ACTIVATED by DCFR mode (CHANGE B). An
    // action whose start-of-iteration cumulative regret is `<= eff_prune` (deeply negative) AND which
    // carries 0 strategy mass (`strat[a] == 0.0`, i.e. regret-match already zeroed it) has its entire
    // subtree — and that subtree's dominant showdown evals — skipped; its `av[a]` is backfilled to `ev`
    // (regret delta 0 => frozen => stays pruned this iteration). `eff_prune = NEG_INFINITY` DISABLES it:
    // the `eff_prune.is_finite()` gate below makes that path BYTE-IDENTICAL to the no-RBP trainer (no
    // `pruned` array, no extra checks). The CFR+ default path passes `cfg.prune_threshold` (NEG_INFINITY
    // in every shipped cfg, AND inert even if finite because the positive floor keeps read-regret >= 0);
    // DCFR mode keeps SIGNED regret so a finite `eff_prune` actually fires.
    //
    // PLURIBUS CORRECTNESS RULES enforced here:
    //   RULE 1a (never prune the LAST betting round): when `env.street >= STREET_RIVER` (street 3 = the
    //     final betting round) pruning is disabled for THIS node (`prune_here`), so river values — which
    //     feed every upstream regret — are always computed exactly.
    //   RULE 1b (never prune an action that leads DIRECTLY to a terminal payoff): a candidate is applied
    //     and only pruned if its child is NON-terminal. A terminal-leading action has no deep subtree to
    //     skip (its value is one cheap settle), so this loses no speed while protecting the value estimate
    //     that Pruning-at-a-terminal would corrupt.
    //   RULE 2 (per-ITERATION prune/explore decision): the caller already chose prune-vs-explore for the
    //     whole iteration via `dcfr_is_explore_iter` (explore-iters pass NEG_INFINITY here), so a
    //     deeply-negative action is fully re-explored ~5% of iterations and can REVIVE. (Not decided
    //     per-action — cheaper, Pluribus.)
    //   RULE 3 (only prune a WARMED action): `strat[a] == 0.0 && regrets[a] <= eff_prune` requires the
    //     action's accumulated regret to have actually fallen below the (deeply-negative) floor — i.e. it
    //     warmed up and lost — before it is pruned (the hook's original note).
    let prune_here = dcfr_prune_node_allowed(env.street, eff_prune); // RULE 1a
    if prune_here {
        let mut pruned = [false; NUM_ACTIONS];
        for i in 0..ln {
            let a = lidx[i];
            if strat[a] == 0.0 && regrets[a] <= eff_prune {
                // RULE 3 warmed + deeply-negative. Apply the action to check RULE 1b before committing to
                // the prune (a terminal child is cheap and must NOT be pruned).
                let child = trainer_apply(env, a);
                let terminal_leading = child.done || child.street >= core::STREET_TERMINAL;
                if terminal_leading {
                    // RULE 1b: action leads DIRECTLY to a terminal — never prune; evaluate its cheap leaf.
                    record_prune_event(env.street, false, true);
                    av[a] =
                        one_traversal(&child, hero, cfg, read, local, rng, t, memo, eff_prune, v3i);
                } else {
                    record_prune_event(env.street, true, false);
                    pruned[a] = true; // skip the (non-terminal) subtree + its showdowns; backfilled below.
                }
                continue;
            }
            let child = trainer_apply(env, a);
            av[a] = one_traversal(&child, hero, cfg, read, local, rng, t, memo, eff_prune, v3i);
        }
        // ev over the explored actions (pruned lanes have strat==0 so contribute nothing yet). Scalar
        // form, matching the main (RBP-off) path's byte-identical math.
        let mut ev = 0.0;
        for a in 0..NUM_ACTIONS {
            ev += strat[a] * av[a];
        }
        for a in 0..NUM_ACTIONS {
            if pruned[a] {
                av[a] = ev; // frozen-regret backfill (regret delta 0 => stays pruned)
            }
        }
        {
            let r = local.cum_regret.entry(key).or_insert(ZEROS);
            for a in 0..NUM_ACTIONS {
                if legal[a] {
                    r[a] += av[a] - ev;
                }
            }
        }
        return ev;
    }
    // RBP disabled (default, or river under DCFR per RULE 1a) — the zero-overhead path: branch all legal
    // actions, no prune machinery.
    for i in 0..ln {
        let a = lidx[i];
        let child = trainer_apply(env, a);
        av[a] = one_traversal(&child, hero, cfg, read, local, rng, t, memo, eff_prune, v3i);
    }
    // LEVER 6(a) (FMA hero EV dot-product + regret delta) — TRIED AND REVERTED with lever 5: the
    // `mul_add`/masked-11-wide form was statistically equivalent but tipped the same marginal LBR
    // convergence smoke and saved ~0% wall (these kernels are tiny post-showdown-collapse). Scalar,
    // byte-identical-to-original form retained.
    let mut ev = 0.0;
    for a in 0..NUM_ACTIONS {
        ev += strat[a] * av[a];
    }
    // CFR+ cumulative regret: PLAIN sum here — the positive floor `max(r,0)` is applied ONCE per
    // iteration in `train_iteration` (after merge), not per-partial/per-visit. regret_match() floors
    // per-action on READ, so a transiently-negative cumulant is harmless and the on-policy strategy is
    // unaffected; flooring per-visit over-inflates regret (order-dependent). Mirrors leduc.rs. (FIX 3)
    {
        let r = local.cum_regret.entry(key).or_insert(ZEROS);
        for a in 0..NUM_ACTIONS {
            if legal[a] {
                r[a] += av[a] - ev;
            }
        }
    }
    // NOTE: strat_sum is NOT accumulated at the hero node. The average-strategy estimator accumulates
    // the current player's strategy at the OPPONENT's node (see the `actor != hero` branch above, FIX 2)
    // — accumulating only in the hero branch biases the fixed point at every non-root infoset.
    ev
}

// --------------------------------------------------------------------------------------------------
// HOGWILD traversal (feature `hogwild`) — the SAME ES-MCCFR recursion as `one_traversal`, but reading
// regret-matching strategy from, and writing regret/strat deltas DIRECTLY INTO, the ONE shared
// open-addressing table (`hogwild::SharedHogwildTable`) in place. No `read` snapshot, no `local` delta, no
// per-iteration merge or floor barrier: all workers race on the shared table (the accepted hogwild noise)
// and the CFR+ floor is applied per regret add (`SharedHogwildTable::add_regret` clamps >= 0).
//
// DUPLICATION NOTE (intentional, called out in the report): this is a near-mirror of `one_traversal` — the
// chance/terminal/leaf/legal-mask/fingerprint structure and every float op (regret_match, the opponent
// strat_sum weight loop, the hero av/ev/regret-delta loops) are identical; the ONLY changes are (a) the
// strategy is read via `table.slot_for(&key)` -> `table.regret_row(slot)` instead of from `read.shards`,
// and (b) writes go to `table.add_strat` / `table.add_regret` instead of `local`. A trait seam over the
// fork-merge `one_traversal` was rejected to keep that hot path provably byte-identical (firewall
// requirement); the cost is this one duplicated recursion, gated entirely behind `feature = "hogwild"`.
// RBP (`cfg.prune_threshold` finite) is NOT supported on the hogwild path (it falls through to branching
// every legal action — same as the default `prune_threshold = NEG_INFINITY`).
#[cfg(feature = "hogwild")]
fn one_traversal_hogwild<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    cfg: &TrainerCfg,
    table: &crate::hogwild::SharedHogwildTable,
    rng: &mut Rng,
    t: u64,
    memo: &mut FpMemo,
) -> Result<f64, String> {
    if env.done {
        return Ok(env.rewards[hero] as f64 / CHIP_UNIT);
    }
    if core::is_chance_node(env) {
        let child = core::sample_chance(env);
        return one_traversal_hogwild(&child, hero, cfg, table, rng, t, memo);
    }
    if env.street >= core::STREET_TERMINAL {
        return Ok(terminal_return(env, hero));
    }
    let actor_i = env.action_seat;
    if actor_i < 0 {
        return Ok(terminal_return(env, hero));
    }
    let actor = actor_i as usize;
    let legal = abstracted_legal(env, actor, cfg);
    if !any_legal(&legal) {
        return Ok(terminal_return(env, hero));
    }
    let key = fingerprint_fast_depth_memo_emd(
        env,
        actor,
        cfg.num_buckets,
        &cfg.depth_edges,
        cfg.emd_frozen.as_deref(),
        cfg.river_ochs.as_deref(),
        memo,
    );
    let (lidx, ln) = compact_legal(&legal);

    // Claim (or find) the shared slot for this infoset, then read the (racy) regret row for
    // regret-matching. `slot_for` is the ONLY place a first-visit claim can fail (capacity overflow) — a
    // hard Err that propagates all the way out (never silent corruption).
    let slot = table.slot_for(&key)?;
    let regrets = table.regret_row(slot);

    if actor != hero {
        // Opponent node: regret-matching+ strategy, accumulate the linear-weighted strat-sum into the
        // shared table, then sample ONE action. Same math/order as `one_traversal`'s opponent branch.
        let strat = regret_match(&regrets, &legal);
        let w = t as f64;
        let mut contrib = [0.0f64; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            if legal[a] {
                contrib[a] = w * strat[a];
            }
        }
        table.add_strat(slot, &legal, &contrib);
        let u = rng.next_f64();
        let a = sample_action_compact(&strat, &lidx, ln, u);
        let child = trainer_apply(env, a);
        return one_traversal_hogwild(&child, hero, cfg, table, rng, t, memo);
    }

    // Hero node: branch ALL abstracted-legal actions, CFR+ regret update written into the shared table.
    let strat = regret_match(&regrets, &legal);
    let mut av = [0.0f64; NUM_ACTIONS];
    for i in 0..ln {
        let a = lidx[i];
        let child = trainer_apply(env, a);
        av[a] = one_traversal_hogwild(&child, hero, cfg, table, rng, t, memo)?;
    }
    let mut ev = 0.0;
    for a in 0..NUM_ACTIONS {
        ev += strat[a] * av[a];
    }
    // CFR+ regret delta = av - ev on legal lanes; `add_regret` floors the post-add value >= 0 per element
    // (the per-add floor replaces the fork-merge per-iteration `floor_cum_regret` barrier).
    let mut delta = [0.0f64; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            delta[a] = av[a] - ev;
        }
    }
    table.add_regret(slot, &legal, &delta);
    Ok(ev)
}

// --------------------------------------------------------------------------------------------------
// BATCHED, memory-level-parallel (MLP) traversal — BYTE-IDENTICAL to the recursive `one_traversal`.
//
// MECHANISM (NOT prefetch — explicit `_mm_prefetch` was measured INERT): the recursion is converted to an
// explicit frame stack so a single deal-traversal can be SUSPENDED at each actor-node regret read and
// RESUMED. `B = cfg.batch_deals` INDEPENDENT deals are stepped in lockstep: every round, all B traversals
// advance to their next regret-read (Phase A), then their B `cum_regret.get()` probes issue back-to-back
// in a tight loop (Phase B, the MLP — the CPU's out-of-order engine overlaps the DRAM-bound hashmap
// latencies), then each consumes its regret row and advances one step (Phase C).
//
// BYTE-IDENTITY: only DIFFERENT deals (independent rng) are interleaved — within ONE deal the N heroes
// run strictly sequentially (hero h finishes ALL its rng draws in DFS order before hero h+1 starts), the
// single per-deal `rng` is shared across heroes exactly as the sequential loop does, every float op
// (`regret_match`, the opponent strat_sum write loop, the hero `av`/`ev`/regret-write loops) is the SAME
// code with the SAME iteration bounds/order, and each deal's `LocalDelta` is folded into the task
// accumulator via `accumulate_local_into` in ASCENDING deal-index order. RBP (finite `prune_threshold`)
// is NOT implemented here — `collect_partials_n` routes finite-threshold cfgs to the recursive fold.
// --------------------------------------------------------------------------------------------------

/// An explicit-stack frame for one steppable deal-traversal. `Node` carries an env to (re)process from
/// scratch (exactly the head of `one_traversal`); `Hero` is the suspended state of a hero decision node
/// that is branching its abstracted-legal actions one child at a time.
enum Frame<const N: usize> {
    /// Process this env: terminal/chance/leaf produce a `ret`; an actor node yields a pending regret read.
    Node(EnvStateN<N>),
    /// A hero decision node mid-branch. `env` is the node, `key` its fingerprint, `strat` the
    /// regret-matched strategy (frozen at node entry), `legal`/`lidx`/`ln` the abstracted-legal set,
    /// `av` the per-action counterfactual values accumulated so far, `i` the index of the NEXT child to
    /// launch (so the child just completed is action `lidx[i]`).
    Hero {
        env: EnvStateN<N>,
        key: [u8; 16],
        strat: [f64; NUM_ACTIONS],
        legal: [bool; NUM_ACTIONS],
        lidx: [usize; NUM_ACTIONS],
        ln: usize,
        av: [f64; NUM_ACTIONS],
        i: usize,
    },
}

/// One in-flight deal-traversal: owns its rng, per-deal `LocalDelta`, the current `hero` seat being
/// traversed, the explicit frame stack, the last child's return value `ret`, the deal root (re-pushed
/// when advancing to the next hero), the pending fingerprint key awaiting a regret read, and a `done`
/// flag (set once all N heroes' traversals have completed).
struct DealTrav<const N: usize> {
    rng: Rng,
    local: LocalDelta,
    hero: usize,
    stack: Vec<Frame<N>>,
    ret: f64,
    root: EnvStateN<N>,
    /// Some(key) when the trav is SUSPENDED at an actor node awaiting `cum_regret.get(&key)`.
    pending_key: Option<[u8; 16]>,
    done: bool,
}

impl<const N: usize> DealTrav<N> {
    /// Initialize a traversal for deal `d`: derive the deal seed EXACTLY as `collect_partials_n` does
    /// (`iter_seed + d*0x100000001b3 rotl 17`), seed the rng, deal the root (the FIRST rng draws —
    /// Fisher-Yates + init, with the button rotating on `d`), and start hero 0.
    fn new(d: u64, iter_seed: u64) -> Self {
        let deal_seed = iter_seed
            .wrapping_add(d.wrapping_mul(0x100000001b3))
            .rotate_left(17);
        let mut rng = Rng::seed_from_u64(deal_seed);
        let root = deal_root::<N>(&mut rng, d);
        DealTrav {
            rng,
            local: LocalDelta::default(),
            hero: 0,
            stack: vec![Frame::Node(root)],
            ret: 0.0,
            root,
            pending_key: None,
            done: false,
        }
    }
}

/// PHASE A — advance one trav until it either (a) suspends at an actor node awaiting a regret read
/// (`pending_key` set), or (b) finishes (all N heroes done -> `done`). Processes every frame that does
/// NOT need a fresh regret read: terminal / chance / no-actor leaves, chance sampling, and Hero RESUME
/// (consuming a completed child, launching the next, or finalizing). Mirrors `one_traversal` exactly.
#[inline]
fn advance_to_probe<const N: usize>(
    trav: &mut DealTrav<N>,
    cfg: &TrainerCfg,
    memo: &mut FpMemo,
) {
    loop {
        let frame = match trav.stack.last() {
            Some(f) => f,
            None => {
                // Hero `trav.hero`'s traversal is complete (its root frame produced `trav.ret`). Advance
                // to the next hero with the SAME rng (do NOT reseed) — exactly the sequential
                // `for hero in 0..N` loop, which threads one rng through all heroes of a deal.
                trav.hero += 1;
                if trav.hero < N {
                    trav.stack.push(Frame::Node(trav.root));
                    continue;
                } else {
                    trav.done = true;
                    return;
                }
            }
        };
        match frame {
            Frame::Node(env) => {
                // Head of `one_traversal`: resolve the cheap (no-regret-read) cases here; an actor node
                // computes its fingerprint key, sets pending_key and YIELDS.
                if env.done {
                    trav.ret = env.rewards[trav.hero] as f64 / CHIP_UNIT;
                    trav.stack.pop();
                    continue;
                }
                if core::is_chance_node(env) {
                    let child = core::sample_chance(env);
                    // Tail recursion into the chance child: replace this frame.
                    *trav.stack.last_mut().unwrap() = Frame::Node(child);
                    continue;
                }
                if env.street >= core::STREET_TERMINAL {
                    trav.ret = terminal_return(env, trav.hero);
                    trav.stack.pop();
                    continue;
                }
                let actor_i = env.action_seat;
                if actor_i < 0 {
                    trav.ret = terminal_return(env, trav.hero);
                    trav.stack.pop();
                    continue;
                }
                let actor = actor_i as usize;
                let legal = abstracted_legal(env, actor, cfg);
                if !any_legal(&legal) {
                    trav.ret = terminal_return(env, trav.hero);
                    trav.stack.pop();
                    continue;
                }
                let key = fingerprint_fast_depth_memo_emd(
                    env,
                    actor,
                    cfg.num_buckets,
                    &cfg.depth_edges,
                    cfg.emd_frozen.as_deref(),
                    cfg.river_ochs.as_deref(),
                    memo,
                );
                // Suspend: the regret read for `key` is issued in Phase B (the MLP), consumed in Phase C.
                trav.pending_key = Some(key);
                return;
            }
            Frame::Hero { .. } => {
                // RESUME a hero node after the child for action `lidx[i]` returned `trav.ret`. Mutate the
                // top frame in place: store av[lidx[i]], advance i. Take the action to launch next (if any)
                // OR the finalize inputs (strat/av/legal/key) by value so the `&mut` borrow ends before we
                // push/pop.
                let ret_child = trav.ret;
                enum Step {
                    Launch(usize), // push Node(child) for action id
                    Finalize {
                        strat: [f64; NUM_ACTIONS],
                        av: [f64; NUM_ACTIONS],
                        legal: [bool; NUM_ACTIONS],
                        key: [u8; 16],
                    },
                }
                let env_for_apply;
                let step;
                {
                    let top = trav.stack.last_mut().unwrap();
                    let Frame::Hero { env, key, strat, legal, lidx, ln, av, i } = top else {
                        unreachable!("Frame::Hero")
                    };
                    let a_done = lidx[*i];
                    av[a_done] = ret_child;
                    *i += 1;
                    if *i < *ln {
                        let a_next = lidx[*i];
                        env_for_apply = *env;
                        step = Step::Launch(a_next);
                    } else {
                        env_for_apply = *env;
                        step = Step::Finalize {
                            strat: *strat,
                            av: *av,
                            legal: *legal,
                            key: *key,
                        };
                    }
                }
                match step {
                    Step::Launch(a_next) => {
                        // Launch the next child branch (same ascending legal order as the recursion's
                        // `for i in 0..ln { let a=lidx[i]; ... }`).
                        let child = trainer_apply(&env_for_apply, a_next);
                        trav.stack.push(Frame::Node(child));
                        continue;
                    }
                    Step::Finalize { strat, av, legal, key } => {
                        // ev = sum_a strat[a]*av[a] over a=0..NUM_ACTIONS — VERBATIM from one_traversal.
                        let mut ev = 0.0;
                        for a in 0..NUM_ACTIONS {
                            ev += strat[a] * av[a];
                        }
                        {
                            let r = trav.local.cum_regret.entry(key).or_insert(ZEROS);
                            for a in 0..NUM_ACTIONS {
                                if legal[a] {
                                    r[a] += av[a] - ev;
                                }
                            }
                        }
                        trav.ret = ev;
                        trav.stack.pop();
                        continue;
                    }
                }
            }
        }
    }
}

/// PHASE C — consume the regret row read for the suspended trav and advance ONE step past the actor node.
/// The trav's top frame is `Node(env)` at the actor node whose key was stashed in `pending_key`. The key
/// is REUSED (not recomputed) so there is no extra fingerprint/memo touch. Replicates the opponent / hero
/// branches of `one_traversal` EXACTLY (recomputing legal/lidx from `env` exactly as the recursion does,
/// same strat_sum write, same rng draw + sample for opponents, same Hero-frame launch for the hero).
#[inline]
fn consume<const N: usize>(
    trav: &mut DealTrav<N>,
    regrets: &[f64; NUM_ACTIONS],
    cfg: &TrainerCfg,
    t: u64,
) {
    let key = trav.pending_key.take().expect("consume without a pending key");
    // The actor node env is on top.
    let env = match trav.stack.last() {
        Some(Frame::Node(env)) => *env,
        _ => unreachable!("consume expects a Node actor on top"),
    };
    let actor = env.action_seat as usize;
    let legal = abstracted_legal(&env, actor, cfg);
    let (lidx, ln) = compact_legal(&legal);

    if actor != trav.hero {
        // Opponent node — VERBATIM from one_traversal's `actor != hero` branch (strat_sum write loop
        // BEFORE the rng draw, same w = t as f64, same sample).
        let strat = regret_match(regrets, &legal);
        let w = t as f64;
        {
            let ss = trav.local.strat_sum.entry(key).or_insert(ZEROS);
            for a in 0..NUM_ACTIONS {
                if legal[a] {
                    ss[a] += w * strat[a];
                }
            }
        }
        let u = trav.rng.next_f64();
        let a = sample_action_compact(&strat, &lidx, ln, u);
        let child = trainer_apply(&env, a);
        // Tail recursion: replace the actor Node with the child Node.
        *trav.stack.last_mut().unwrap() = Frame::Node(child);
        return;
    }

    // Hero node — VERBATIM from one_traversal's hero branch (RBP-off path; finite prune falls back to the
    // recursive scheduler upstream). Build the Hero frame and launch the first child branch (lidx[0]).
    let strat = regret_match(regrets, &legal);
    // ln >= 1 here (any_legal was true at suspend), so lidx[0] is valid.
    let a0 = lidx[0];
    let child0 = trainer_apply(&env, a0);
    *trav.stack.last_mut().unwrap() = Frame::Hero {
        env,
        key,
        strat,
        legal,
        lidx,
        ln,
        av: [0.0f64; NUM_ACTIONS],
        i: 0,
    };
    trav.stack.push(Frame::Node(child0));
}

/// BATCH DRIVER — run a contiguous batch of `travs` (one `DealTrav` per deal, ascending deal index) in
/// lockstep until ALL are `done`, then fold each deal's `LocalDelta` into `acc` via `accumulate_local_into`
/// in ASCENDING deal-index order (the `travs` slice is already in deal order). Per round: Phase A advances
/// every not-done trav to its next regret probe; Phase B gathers the B `cum_regret.get()` reads back-to-
/// back (the MLP); Phase C consumes each read and steps one node past the actor.
#[inline]
fn run_batch<const N: usize>(
    mut travs: Vec<DealTrav<N>>,
    cfg: &TrainerCfg,
    read: &NlheTables,
    t: u64,
    memo: &mut FpMemo,
    acc: &mut NlheTables,
) {
    let b = travs.len();
    // Scratch for the gathered regret rows (one per active trav this round), reused across rounds.
    let mut regrets: Vec<[f64; NUM_ACTIONS]> = vec![ZEROS; b];
    loop {
        // PHASE A: advance every not-yet-done trav to its next regret probe (or to done).
        let mut any_active = false;
        for trav in travs.iter_mut() {
            if trav.done {
                continue;
            }
            advance_to_probe::<N>(trav, cfg, memo);
            if trav.pending_key.is_some() {
                any_active = true;
            }
        }
        if !any_active {
            // No trav has a pending probe: all are done.
            break;
        }
        // PHASE B: issue the B independent cum_regret reads back-to-back (the memory-level parallelism).
        for (j, trav) in travs.iter().enumerate() {
            if let Some(key) = trav.pending_key {
                regrets[j] =
                    widen_r(read.shards[shard_of(&key)].cum_regret.get(&key).unwrap_or(&ZEROS_R));
            }
        }
        // PHASE C: consume each read and step one node past the actor.
        for (j, trav) in travs.iter_mut().enumerate() {
            if trav.pending_key.is_some() {
                consume::<N>(trav, &regrets[j], cfg, t);
            }
        }
    }
    // Fold each deal's LocalDelta into the task accumulator in ASCENDING deal-index order (travs is in
    // deal order) — IDENTICAL fold order to the sequential `accumulate_local_into` per deal.
    for trav in travs.into_iter() {
        accumulate_local_into(acc, trav.local);
    }
}

// --------------------------------------------------------------------------------------------------
// train_iteration — genuine gap #2: rayon deal-sampling outer loop with thread-local accumulation
// --------------------------------------------------------------------------------------------------

fn one_traversal_profile<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    cfg: &TrainerCfg,
    read: &NlheTables,
    local: &mut LocalDelta,
    rng: &mut Rng,
    t: u64,
    memo: &mut FpMemo,
    stats: &mut TraverseStats,
    depth: u64,
    eff_prune: f64,
    v3i: V3Iter,
) -> f64 {
    stats.nodes += 1;
    stats.max_depth = stats.max_depth.max(depth);
    if env.done {
        stats.terminal_nodes += 1;
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    if core::is_chance_node(env) {
        stats.chance_nodes += 1;
        let child = core::sample_chance(env);
        return one_traversal_profile(
            &child,
            hero,
            cfg,
            read,
            local,
            rng,
            t,
            memo,
            stats,
            depth + 1,
            eff_prune,
            v3i,
        );
    }
    if env.street >= core::STREET_TERMINAL {
        stats.terminal_nodes += 1;
        stats.terminal_settles += 1;
        return terminal_return(env, hero);
    }
    let actor_i = env.action_seat;
    if actor_i < 0 {
        stats.terminal_nodes += 1;
        stats.terminal_settles += 1;
        return terminal_return(env, hero);
    }
    let actor = actor_i as usize;
    let legal = abstracted_legal(env, actor, cfg);
    if !any_legal(&legal) {
        stats.no_legal_nodes += 1;
        stats.terminal_nodes += 1;
        stats.terminal_settles += 1;
        return terminal_return(env, hero);
    }
    stats.decision_nodes += 1;
    let key = fingerprint_fast_depth_memo_emd_profile(
        env,
        actor,
        cfg.num_buckets,
        &cfg.depth_edges,
        cfg.emd_frozen.as_deref(),
        memo,
        cfg.river_ochs.as_deref(),
        stats,
    );

    let (lidx, ln) = compact_legal(&legal);
    stats.legal_actions_total += ln as u64;
    let regrets = if ab_stub_mode() == 2 {
        ZEROS // regret_zero: skip the cum_regret hashmap probe (subtractive attribution stub)
    } else {
        widen_r(read.shards[shard_of(&key)].cum_regret.get(&key).unwrap_or(&ZEROS_R))
    };

    if actor != hero {
        stats.opponent_nodes += 1;
        let strat = regret_match(&regrets, &legal);
        // v3 twin of the `one_traversal` strat site (default-off => legacy `w = t as f64` exactly).
        if !v3i.strat_pre_only || env.street == 0 {
            let w = if v3i.unweighted_strat { 1.0 } else { t as f64 };
            let ss = local.strat_sum.entry(key).or_insert(ZEROS);
            for a in 0..NUM_ACTIONS {
                if legal[a] {
                    ss[a] += w * strat[a];
                }
            }
        }
        let u = rng.next_f64();
        let a = sample_action_compact(&strat, &lidx, ln, u);
        stats.opponent_samples += 1;
        stats.apply_calls += 1;
        let child = trainer_apply(env, a);
        return one_traversal_profile(
            &child,
            hero,
            cfg,
            read,
            local,
            rng,
            t,
            memo,
            stats,
            depth + 1,
            eff_prune,
            v3i,
        );
    }

    stats.hero_nodes += 1;
    let strat = regret_match(&regrets, &legal);
    let mut av = [0.0f64; NUM_ACTIONS];
    // Same Pluribus-RBP hook as `one_traversal` (rules 1a/1b/2/3) — see that function for the rationale.
    let prune_here = dcfr_prune_node_allowed(env.street, eff_prune); // RULE 1a
    if prune_here {
        let mut pruned = [false; NUM_ACTIONS];
        for i in 0..ln {
            let a = lidx[i];
            if strat[a] == 0.0 && regrets[a] <= eff_prune {
                // RULE 3 warmed + deeply-negative; apply to check RULE 1b (never prune a terminal child).
                stats.apply_calls += 1;
                let child = trainer_apply(env, a);
                if child.done || child.street >= core::STREET_TERMINAL {
                    stats.hero_branches += 1;
                    av[a] = one_traversal_profile(
                        &child,
                        hero,
                        cfg,
                        read,
                        local,
                        rng,
                        t,
                        memo,
                        stats,
                        depth + 1,
                        eff_prune,
                        v3i,
                    );
                } else {
                    pruned[a] = true;
                }
                continue;
            }
            stats.hero_branches += 1;
            stats.apply_calls += 1;
            let child = trainer_apply(env, a);
            av[a] = one_traversal_profile(
                &child,
                hero,
                cfg,
                read,
                local,
                rng,
                t,
                memo,
                stats,
                depth + 1,
                eff_prune,
                v3i,
            );
        }
        let mut ev = 0.0;
        for a in 0..NUM_ACTIONS {
            ev += strat[a] * av[a];
        }
        for a in 0..NUM_ACTIONS {
            if pruned[a] {
                av[a] = ev;
            }
        }
        {
            let r = local.cum_regret.entry(key).or_insert(ZEROS);
            for a in 0..NUM_ACTIONS {
                if legal[a] {
                    r[a] += av[a] - ev;
                }
            }
        }
        return ev;
    }
    for i in 0..ln {
        let a = lidx[i];
        stats.hero_branches += 1;
        stats.apply_calls += 1;
        let child = trainer_apply(env, a);
        av[a] = one_traversal_profile(
            &child,
            hero,
            cfg,
            read,
            local,
            rng,
            t,
            memo,
            stats,
            depth + 1,
            eff_prune,
            v3i,
        );
    }
    let mut ev = 0.0;
    for a in 0..NUM_ACTIONS {
        ev += strat[a] * av[a];
    }
    {
        let r = local.cum_regret.entry(key).or_insert(ZEROS);
        for a in 0..NUM_ACTIONS {
            if legal[a] {
                r[a] += av[a] - ev;
            }
        }
    }
    ev
}

impl NlheTables {
    /// Run ONE CFR+ deal-sampling iteration, dispatching on `cfg.n_seats` to the matching const-generic
    /// monomorphization. 6 (the deployed full-ring blueprint) routes through `train_iteration_n::<6>`,
    /// which is BYTE-IDENTICAL to the pre-generalization 6-seat trainer (same deal, key, recursion, RNG
    /// schedule). 7 is the MTT tournament-anchor full ring. Any other width falls back to 6 (there is no
    /// trained blueprint keyed for other seat counts, and the engine/fingerprint are only smoke-validated
    /// at N in {6,7}). Mutates `self` in place so a checkpoint can be taken between iterations.
    pub fn train_iteration(&mut self, cfg: &TrainerCfg) {
        self.train_iteration_v3(cfg, v3_env());
    }

    /// `train_iteration` with the v3 schedule passed EXPLICITLY (unit tests / harnesses that must not
    /// touch the process-global env OnceLock). `train_iteration(cfg)` == `train_iteration_v3(cfg,
    /// v3_env())`, and `train_iteration_v3(cfg, &V3Cfg::default())` is byte-identical to the legacy
    /// trainer (pinned ab_train goldens).
    pub fn train_iteration_v3(&mut self, cfg: &TrainerCfg, v3: &V3Cfg) {
        match cfg.n_seats {
            7 => self.train_iteration_n_v3::<7>(cfg, v3),
            // 6 is the default + the deployed path; any unrecognized width also routes here so the
            // trainer never silently produces keys for a seat count nothing serves.
            _ => self.train_iteration_n_v3::<6>(cfg, v3),
        }
    }

    /// Run one CFR+ iteration and return coarse phase timings for local profiling / on-box heartbeats.
    /// The returned counters are observational only; the training semantics are identical to
    /// `train_iteration`.
    pub fn train_iteration_profile(&mut self, cfg: &TrainerCfg) -> TrainIterProfile {
        match cfg.n_seats {
            7 => self.train_iteration_n_profile::<7>(cfg),
            // 6 is the default + the deployed path; any unrecognized width also routes here so the
            // trainer never silently produces keys for a seat count nothing serves.
            _ => self.train_iteration_n_profile::<6>(cfg),
        }
    }

    /// The const-generic CFR+ iteration. Fan `cfg.n_deals_per_iter` deals out over rayon; each worker
    /// deals a fresh N-seat root and runs `one_traversal::<N>` for ALL `N` traverser seats into a
    /// THREAD-LOCAL `NlheTables`; the per-thread tables are merged by `+=` (CFR+ floor applied at merge).
    /// The READ side of each traversal sees the table as of the START of this iteration (snapshot), so
    /// deals within an iteration are independent — the merge is the lock-free hogwild sync.
    ///
    /// The N=6 instantiation is byte-identical to the original 6-seat `train_iteration`: same `t`/seed
    /// derivation, same `deal_root::<6>` (== old `deal_root`), same `for hero in 0..6`, same merge+floor.
    /// `NlheTables` itself is seat-independent (keyed by the 16-byte fingerprint), so no per-N table type
    /// is needed — only the deal/recursion width changes.
    pub fn train_iteration_n<const N: usize>(&mut self, cfg: &TrainerCfg) {
        self.train_iteration_n_v3::<N>(cfg, v3_env())
    }

    pub fn train_iteration_n_v3<const N: usize>(&mut self, cfg: &TrainerCfg, v3: &V3Cfg) {
        let t = self.iters_done + 1;
        let iter_seed = cfg.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);
        // EFFECTIVE per-iteration prune threshold (CHANGE B / Pluribus rule 2 — the prune/explore choice is
        // made ONCE per iteration here, not per action). CFR+ default: `cfg.prune_threshold` verbatim
        // (NEG_INFINITY in every shipped cfg; inert even if finite because the positive floor keeps
        // read-regret >= 0) => BYTE-IDENTICAL. DCFR: on a ~95% prune-iteration use the deeply-negative
        // `dcfr_prune_threshold`; on a ~5% explore-iteration use NEG_INFINITY so every action is re-
        // explored and a pruned action can REVIVE. The explore decision is deterministic in `iter_seed`.
        // v3 LCFR mode: same coin with `v3.prune_c` / `v3.prune_explore` from `v3.prune_start`.
        let eff_prune = effective_prune_threshold(cfg, iter_seed, t, v3);
        let v3i = V3Iter {
            unweighted_strat: v3.lcfr,
            strat_pre_only: v3.snapshot_avg,
        };
        let read: &NlheTables = self;
        let partials = Self::collect_partials_n::<N>(cfg, read, t, iter_seed, eff_prune, v3i);
        self.merge_partials(cfg, partials);
        if dcfr_enabled(cfg) {
            // DCFR (CHANGE B): keep SIGNED regret and DISCOUNT it (no positive floor) so the prune hook's
            // negative-regret signal survives. Replaces `floor_cum_regret` for this iteration.
            self.apply_dcfr_discount(t);
        } else if v3.lcfr || v3.signed_regrets {
            // v3 LCFR (Pluribus): keep SIGNED regret, clamp at `regret_floor`, and on schedule apply the
            // (t/D)/((t/D)+1) discount to ALL regrets and average-strategy rows. Arm-2 signed-regrets
            // mode reaches the same pass but `apply_lcfr` fires NO discount events (`event` requires
            // `v3.lcfr`), so it is floor-only: signed regrets survive, pruning stays meaningful, and
            // the strat accumulation upstream kept CFR+'s w=t.
            self.apply_lcfr(t, v3);
        } else {
            // CFR+ default: positive-regret floor (byte-identical to today).
            self.floor_cum_regret();
        }
        self.iters_done = t;
    }

    fn collect_partials_n<const N: usize>(
        cfg: &TrainerCfg,
        read: &NlheTables,
        t: u64,
        iter_seed: u64,
        eff_prune: f64,
        v3i: V3Iter,
    ) -> Vec<NlheTables> {
        // WORK-STEALING PATH (CHANGE A): when enabled (cfg.work_steal OR env POKERBOT_WORKSTEAL=1), pull
        // deals dynamically off a shared atomic counter across exactly `current_num_threads()` workers so
        // the deep-deal tail rebalances. Composes with RBP (eff_prune is threaded through). Independent of
        // the batched MLP path — work-steal uses the recursive `one_traversal`; the two are mutually
        // exclusive here (work-steal takes precedence; a future merge could batch within a worker).
        if work_steal_enabled(cfg) {
            return Self::collect_partials_n_worksteal::<N>(cfg, read, t, iter_seed, eff_prune, v3i);
        }
        // BATCHED MLP PATH: when `cfg.batch_deals >= 2` AND RBP is OFF (the production regime), step
        // `B` independent deals in lockstep so their cum_regret probes overlap (see `run_batch`). The
        // rayon SPLITTING is IDENTICAL to the recursive path (same `into_par_iter().with_min_len()`), so
        // the leaf-job boundaries — and therefore the per-task partial grouping that `merge_partials`
        // folds — are the same; within each leaf job the deals are buffered into batches of B and each
        // batch's per-deal `LocalDelta`s are folded into the task accumulator in ASCENDING deal order
        // (`run_batch`), exactly as the recursive `.fold` does one deal at a time. Result: BYTE-IDENTICAL
        // merged table. A finite effective prune threshold (RBP on, e.g. DCFR mode) is NOT supported
        // batched and falls through to the recursive fold below. Likewise the v3 modes (LCFR unweighted
        // strat / snapshot_avg pre-only strat) are NOT replicated in the batched state machine — any
        // active v3 iter falls back to the recursive path (`batched_falls_back_under_v3` pins this).
        if cfg.batch_deals >= 2
            && !eff_prune.is_finite()
            && !v3i.unweighted_strat
            && !v3i.strat_pre_only
        {
            return Self::collect_partials_n_batched::<N>(cfg, read, t, iter_seed);
        }
        // Each rayon worker borrows a PROCESS-PERSISTENT, thread-local blake2b-digest memo (`FP_MEMO`).
        // Unlike a `map_init` cache (which resets every `par_iter`, i.e. every iteration), the
        // thread-local persists across ALL iterations of the whole run, so after a brief warmup the hit
        // rate approaches 100% and the dominant per-node cost — blake2b on the fingerprint message, ~95%
        // of `fingerprint_fast_depth` — collapses to a ~15x-cheaper FxHash lookup. The memo is a PURE
        // function cache (message -> digest): blake2b is config-independent and a given message always
        // maps to the same digest, so it never changes any value the trainer produces. The blueprint key
        // set, [f64;11] rows, merge, and serialization are all byte-identical to the non-memo path (the
        // in-suite `dispatch_n6_matches_explicit_n6_byte_for_byte` guard proves this).
        (0..cfg.n_deals_per_iter)
            .into_par_iter()
            // STAGE 2 / LEVER (R1 chunked scheduling): with_min_len groups contiguous deals per
            // rayon worker so the thread-local `FP_MEMO` (blake2b cache, ~95% of fingerprint cost) and
            // the per-thread `local` accumulator's page locality survive across tasks instead of being
            // torn down at every single-deal boundary. PURE SCHEDULING change — byte-identical
            // outputs (same `iter_seed` derivation, same traversal order, same merge semantics) —
            // the design's "byte-identical by construction" lever (artifacts/rust_search_parallelism_design.md §1a,
            // artifacts/_wf_outputs/roadmap.md STAGE 2). SIZE-AWARE: the design's fixed `64` is calibrated
            // for a 128-vCPU c6i.2xlarge spot box (256/64 = 4 chunks minimum). On smaller dev boxes
            // (e.g. 12 cores) `n_deals/64` chunks is BELOW the core count and starves the pool — we
            // saw 0.68s → 1.20s wall (1.76x SLOWER) at the local box. Floor at `n_deals/cores*4` so we
            // always emit >= the work-stealing fan-out; cap at the design's 64 for the production
            // c6i.2xlarge. Below `n_deals=128` we fall back to per-deal scheduling (the
            // per-deal-amortized overhead is small at that size anyway).
            .with_min_len(deal_chunk_min_len(
                cfg.n_deals_per_iter,
                rayon::current_num_threads(),
            ))
            // PER-TASK SHARDED FOLD (WF5 hybrid): each rayon task folds all the deals in its work-chunk
            // into ONE sharded accumulator (`accumulate_local_into`), so `collect` yields ~one partial
            // PER TASK (a small multiple of the thread count), not one per deal. The per-deal `local`
            // stays a cheap unsharded map (only the few-per-task `acc` is sharded). Downstream
            // `merge_partials` then folds these FEW sharded accumulators parallel-over-shards.
            .fold(NlheTables::default, |mut acc, d| {
                let deal_seed = iter_seed
                    .wrapping_add((d as u64).wrapping_mul(0x100000001b3))
                    .rotate_left(17);
                let mut rng = Rng::seed_from_u64(deal_seed);
                let root = deal_root::<N>(&mut rng, d as u64);
                // Per-deal DELTA accumulator: starts EMPTY, so `local` IS this deal's delta —
                // `one_traversal` reads strategy from the shared `read` snapshot and `+=`-writes only the
                // touched regret/strat-sum rows here. Folded into the per-task `acc` immediately after.
                let mut local = LocalDelta::default();
                FP_MEMO.with(|cell| {
                    let memo = &mut *cell.borrow_mut();
                    for hero in 0..N {
                        one_traversal::<N>(
                            &root, hero, cfg, read, &mut local, &mut rng, t, memo, eff_prune, v3i,
                        );
                    }
                });
                accumulate_local_into(&mut acc, local);
                acc
            })
            .collect()
    }

    /// CHANGE A — DYNAMIC WORK-STEALING deal collector. Replaces the static `into_par_iter().fold()`
    /// chunking with a persistent shared `AtomicUsize` deal counter so threads pull deals DYNAMICALLY and
    /// the measured ~100x per-deal work-variance + ~8x mean-growth tail rebalances (rayon cannot steal
    /// mid-chunk once a deep deal starts; here a thread that finishes a cheap deal immediately claims the
    /// next, so no worker idles while another grinds a deep multi-street deal).
    ///
    /// DESIGN:
    ///   * `next_deal: AtomicUsize` starts at 0. Exactly `rayon::current_num_threads()` workers are spawned
    ///     via `rayon::broadcast` (one task per rayon worker thread = one per physical core when
    ///     RAYON_NUM_THREADS is set to the core count, per the SMT-negative verdict — NO hardcoded count).
    ///   * Each worker loops `let d = next_deal.fetch_add(1, Relaxed); if d >= n_deals { break; }` and
    ///     traverses deal `d` (ALL N hero seats, exactly the static fold's body) into its OWN per-worker
    ///     sharded `NlheTables` accumulator (same sharded accumulation as the static path's per-task `acc`).
    ///   * The per-worker partials are collected (one `NlheTables` per worker) and handed to the UNCHANGED
    ///     `merge_partials` — same downstream merge + floor/discount.
    ///
    /// RNG-SEED-STAYS-PER-DEAL (statistical-equivalence proof): the per-deal seed is
    /// `iter_seed + d*0x100000001b3 rotl 17` — a PURE function of `(iter_seed, d)`, IDENTICAL to the static
    /// fold's derivation and INDEPENDENT of which worker claims `d` or the claim order. So deal `d` deals
    /// the same root, runs the same N-hero traversal, and produces the same `LocalDelta` regardless of
    /// scheduling. The ONLY difference vs the static path is the deal->worker assignment and therefore the
    /// CROSS-PARTIAL merge order (IEEE-754 non-associativity) => the result is STATISTICALLY EQUIVALENT,
    /// NOT byte-identical — the same standard as `fast_merge` (gate by LBR/self-play CI-overlap, not SHA).
    /// `accumulate_local_into` per deal within a worker keeps the per-worker fold deterministic; only the
    /// set of deals a given worker accumulates is scheduler-dependent.
    ///
    /// DATA-RACE FREEDOM: the only shared mutable state is `next_deal` (an `AtomicUsize`, `fetch_add` is
    /// atomic — each deal index handed to exactly one worker). The `read` snapshot is shared `&` (read-only).
    /// Every worker writes ONLY its own owned `NlheTables` (moved out via the returned partial). No `unsafe`,
    /// no aliased `&mut`.
    fn collect_partials_n_worksteal<const N: usize>(
        cfg: &TrainerCfg,
        read: &NlheTables,
        t: u64,
        iter_seed: u64,
        eff_prune: f64,
        v3i: V3Iter,
    ) -> Vec<NlheTables> {
        let n_deals = cfg.n_deals_per_iter;
        let next_deal = AtomicUsize::new(0);
        // One task per rayon worker thread; each returns its own per-worker partial. `broadcast` runs the
        // closure on every thread in the current pool and collects the results in thread-index order.
        rayon::broadcast(|_ctx| {
            let mut acc = NlheTables::default();
            FP_MEMO.with(|cell| {
                let memo = &mut *cell.borrow_mut();
                // MEMORY BOUND (byte-identity neutral, mode-agnostic): FP_MEMO is a pure
                // message->blake2b-digest cache, so clearing it only forces recompute of the SAME
                // digest — it can never change a regret/strat value or an emitted digest. Summary /
                // production runs stay far below FP_MEMO_CAP and NEVER clear (byte-identical); only
                // POKERBOT_FULLHIST_KEY runs, whose distinct-message count tracks the billions-scale
                // infoset table, ever trip it — bounding this worker's memo to ~FP_MEMO_CAP entries
                // (~315 MB) instead of duplicating the whole message table per rayon worker (the
                // H200 83.3M-row OOM). Checked once per worker per iteration (a `broadcast` task);
                // per-iteration regrowth is only a few thousand entries, so the memo stays ~cap.
                if memo.len() > FP_MEMO_CAP {
                    memo.clear();
                }
                loop {
                    // Claim the next deal index atomically. Relaxed is sufficient: the only invariant is
                    // that each index is handed out exactly once (fetch_add guarantees that); there is no
                    // happens-before dependency on other workers' table writes (each owns its own acc).
                    let d = next_deal.fetch_add(1, Ordering::Relaxed);
                    if d >= n_deals {
                        break;
                    }
                    let deal_seed = iter_seed
                        .wrapping_add((d as u64).wrapping_mul(0x100000001b3))
                        .rotate_left(17);
                    let mut rng = Rng::seed_from_u64(deal_seed);
                    let root = deal_root::<N>(&mut rng, d as u64);
                    let mut local = LocalDelta::default();
                    for hero in 0..N {
                        one_traversal::<N>(
                            &root, hero, cfg, read, &mut local, &mut rng, t, memo, eff_prune, v3i,
                        );
                    }
                    accumulate_local_into(&mut acc, local);
                }
            });
            acc
        })
    }

    /// Batched (MLP) twin of `collect_partials_n`. The rayon pipeline is IDENTICAL
    /// (`into_par_iter().with_min_len().fold(...).collect()`) so leaf-job boundaries match the recursive
    /// path exactly. Each leaf job's fold accumulator carries `(NlheTables, Vec<DealTrav<N>>)`: deals are
    /// buffered into the Vec and flushed in batches of B (`run_batch`, which folds each deal's LocalDelta
    /// into the NlheTables in ASCENDING deal order); a post-fold `.map` flushes the leaf job's tail batch.
    /// Same `deal_seed` derivation, same per-deal traversal, same ascending-order `accumulate_local_into`
    /// — so the per-task partial (and the merged table) is BYTE-IDENTICAL to the recursive fold.
    fn collect_partials_n_batched<const N: usize>(
        cfg: &TrainerCfg,
        read: &NlheTables,
        t: u64,
        iter_seed: u64,
    ) -> Vec<NlheTables> {
        let b = cfg.batch_deals.max(2);
        (0..cfg.n_deals_per_iter)
            .into_par_iter()
            .with_min_len(deal_chunk_min_len(
                cfg.n_deals_per_iter,
                rayon::current_num_threads(),
            ))
            // Accumulator: (per-task partial, pending-deal buffer). The buffer collects up to B deals
            // (ASCENDING within a leaf job, since the fold visits deal indices in order) and flushes a
            // full batch through `run_batch`.
            .fold(
                || (NlheTables::default(), Vec::<DealTrav<N>>::with_capacity(b)),
                |(mut acc, mut buf), d| {
                    buf.push(DealTrav::<N>::new(d as u64, iter_seed));
                    if buf.len() >= b {
                        let batch = std::mem::take(&mut buf);
                        FP_MEMO.with(|cell| {
                            let memo = &mut *cell.borrow_mut();
                            run_batch::<N>(batch, cfg, read, t, memo, &mut acc);
                        });
                    }
                    (acc, buf)
                },
            )
            // Flush each leaf job's tail (the <B remaining deals) into its partial, then drop the buffer.
            .map(|(mut acc, buf)| {
                if !buf.is_empty() {
                    FP_MEMO.with(|cell| {
                        let memo = &mut *cell.borrow_mut();
                        run_batch::<N>(buf, cfg, read, t, memo, &mut acc);
                    });
                }
                acc
            })
            .collect()
    }

    fn collect_partials_n_profile<const N: usize>(
        cfg: &TrainerCfg,
        read: &NlheTables,
        t: u64,
        iter_seed: u64,
        eff_prune: f64,
        v3i: V3Iter,
    ) -> (Vec<NlheTables>, TraverseStats) {
        // The profile twin keeps the STATIC scheduler regardless of `work_steal` (the dynamic scheduler is
        // a pure-throughput change, not a measurement change; the profile counters must reflect a stable,
        // comparable per-node traversal). DCFR's `eff_prune` IS honored so the profile reflects RBP firing.
        let pairs: Vec<(NlheTables, TraverseStats)> = (0..cfg.n_deals_per_iter)
            .into_par_iter()
            .with_min_len(deal_chunk_min_len(
                cfg.n_deals_per_iter,
                rayon::current_num_threads(),
            ))
            // PER-TASK SHARDED FOLD (profile twin of collect_partials_n): fold each chunk's deals into
            // one (sharded accumulator, accumulated stats) pair.
            .fold(
                || (NlheTables::default(), TraverseStats::default()),
                |(mut acc, mut acc_stats), d| {
                    let deal_seed = iter_seed
                        .wrapping_add((d as u64).wrapping_mul(0x100000001b3))
                        .rotate_left(17);
                    let mut rng = Rng::seed_from_u64(deal_seed);
                    let root = deal_root::<N>(&mut rng, d as u64);
                    let mut local = LocalDelta::default();
                    FP_MEMO.with(|cell| {
                        let memo = &mut *cell.borrow_mut();
                        let before = memo.len();
                        for hero in 0..N {
                            one_traversal_profile::<N>(
                                &root, hero, cfg, read, &mut local, &mut rng, t, memo, &mut acc_stats,
                                0, eff_prune, v3i,
                            );
                        }
                        acc_stats.fp_memo_new_entries += memo.len().saturating_sub(before) as u64;
                    });
                    accumulate_local_into(&mut acc, local);
                    (acc, acc_stats)
                },
            )
            .collect();

        let mut partials = Vec::with_capacity(pairs.len());
        let mut total = TraverseStats::default();
        for (partial, stats) in pairs {
            total.merge(&stats);
            partials.push(partial);
        }
        (partials, total)
    }

    fn merge_partials(&mut self, _cfg: &TrainerCfg, partials: Vec<NlheTables>) {
        // SHARDED PARALLEL MERGE (write-time partition, WF5 2026-06-28). Each rayon worker owns ONE
        // shard `dst = self.shards[s]` exclusively (par_iter_mut gives disjoint &mut) and folds every
        // partial's PRE-GROUPED shard-s sub-table `p.shards[s]` into it. No per-key contention (disjoint
        // shards), no N× filter (partials were sharded at write time in `split_local_into_shards`).
        //
        // ORDER / EQUIVALENCE: a key `k` lives in exactly one shard `s = shard_of(k)`, and inside that
        // shard the `for p in &partials` loop folds the partials (the per-rayon-TASK accumulators from
        // `collect_partials_n`'s `fold`) in Vec order. This produces the SAME KEY SET as the pre-refactor
        // serial fold and is STATISTICALLY EQUIVALENT to it, but NOT bit-identical in general: the old
        // path is a strict per-deal left fold `((d0+d1)+d2)+…`, whereas here each key is summed as a TREE
        // (per-task chunk subtotals, then combined), and `((a+b)+c)+d != (a+b)+(c+d)` in IEEE-754. Old-vs-
        // new iters=1 A/B (nd=200): identical keys, max abs diff 5.68e-14 — far inside the
        // `compare_trainer_states` atol=1e-9 production gate; full multi-iter runs then diverge into
        // distinct-but-equally-valid CFR sample paths via normal chaotic regret amplification.
        // `_cfg.fast_merge` is ignored (the sharded merge IS the parallel path; tree-reduce is retired).
        self.shards.par_iter_mut().enumerate().for_each(|(s, dst)| {
            for p in &partials {
                merge_shard(dst, &p.shards[s]);
            }
        });
    }

    fn floor_cum_regret(&mut self) {
        // CFR+ positive-regret floor: applied ONCE, after all partials are merged for this iteration
        // (not per-partial in `merge_shard`, not per-visit in `one_traversal`). The `max(x,0)` per
        // element is order-independent, so this is byte-identical regardless of merge path AND safe to
        // run parallel-over-shards (each shard's rows are disjoint).
        self.shards.par_iter_mut().for_each(|sh| {
            for row in sh.cum_regret.values_mut() {
                for a in 0..NUM_ACTIONS {
                    if row[a] < 0.0 {
                        row[a] = 0.0;
                    }
                }
            }
        });
    }

    /// CHANGE B — DCFR (Discounted CFR, Brown & Sandholm 2019) per-iteration discount, applied ONCE after
    /// the merge IN PLACE OF `floor_cum_regret` when `dcfr` is ON. Unlike the CFR+ floor this KEEPS SIGNED
    /// regret (negatives survive) so the prune hook's deeply-negative signal is real, and DISCOUNTS:
    ///   * positive cum_regret *= t^alpha/(t^alpha+1)
    ///   * negative cum_regret *= t^beta/(t^beta+1)   (beta=0 => factor 0.5 => halve negative regret)
    ///   * strat_sum           *= (t/(t+1))^gamma
    /// with (alpha,beta,gamma)=(1.5,0,2) (Pluribus), matching `shared_hogwild_table.apply_dcfr_discount`.
    ///
    /// Like `floor_cum_regret` the per-element op is the SAME regardless of merge order (each row is scaled
    /// independently) so it is safe parallel-over-shards. The discount is computed in f64 and narrowed back
    /// to `RegretT` once (the `radd`/`narrow_r` posture). A defensive clamp to ±`REG_CLIP` mirrors the
    /// Python reference's 2.2 fix: for very large `t` the factors approach a limit and float noise in the
    /// tail could flip a regret across zero, after which the wrong (pos vs neg) factor would apply on every
    /// subsequent discount — the clamp bounds that runaway. `strat_sum` is also clamped (down-weighting is
    /// conservative; the clamp only ever fires on a bug). NEVER reached on the CFR+ default path.
    fn apply_dcfr_discount(&mut self, t: u64) {
        const REG_CLIP: f64 = 1.0e9;
        const STRAT_CLIP: f64 = 1.0e9;
        let (pos, neg, strat_f) = dcfr_weights(t);
        self.shards.par_iter_mut().for_each(|sh| {
            for row in sh.cum_regret.values_mut() {
                for a in 0..NUM_ACTIONS {
                    let r = row[a] as f64;
                    let factor = if r > 0.0 { pos } else { neg };
                    let mut v = r * factor;
                    if v > REG_CLIP {
                        v = REG_CLIP;
                    } else if v < -REG_CLIP {
                        v = -REG_CLIP;
                    }
                    row[a] = v as RegretT;
                }
            }
            for row in sh.strat_sum.values_mut() {
                for a in 0..NUM_ACTIONS {
                    let mut v = row[a] * strat_f;
                    if v > STRAT_CLIP {
                        v = STRAT_CLIP;
                    } else if v < -STRAT_CLIP {
                        v = -STRAT_CLIP;
                    }
                    row[a] = v;
                }
            }
        });
    }

    /// v3 LCFR (Pluribus P-supp p.14, iteration-denominated) — the per-iteration regret pass used IN
    /// PLACE OF `floor_cum_regret` when `v3.lcfr` is ON. Two things happen here:
    ///   1. **Signed-regret floor**: every cum_regret entry is clamped BELOW at `v3.regret_floor`
    ///      (Pluribus: -3.1e8 vs prune C=-3.0e8 — "easier to unprune actions that were initially
    ///      pruned but later improved" + overflow protection). NO positive floor — negatives survive,
    ///      which is what makes the hard prune threshold meaningful.
    ///   2. **LCFR discount event** (only when `t % D == 0 && t <= W && D > 0`): ALL cum_regret entries
    ///      (both signs, the SAME factor — this is LCFR, not DCFR's sign-split) and ALL strat_sum
    ///      entries are multiplied by k/(k+1) where k = t/D. Successive events give an increment added
    ///      in discount-window j the asymptotic weight j/(n+1) — i.e. linear weighting in window index,
    ///      the periodic-in-place-multiply approximation of Linear CFR that Pluribus used (their D =
    ///      10 wall-minutes, W = 400 wall-minutes; ours is iteration-denominated because wall-clock
    ///      schedules do not transfer across hardware — amendment doc §2 has the mapping).
    /// Per-element ops are order-independent (scale + clamp), so parallel-over-shards is safe, exactly
    /// like `floor_cum_regret` / `apply_dcfr_discount`. Reached when `v3.lcfr` (floor + scheduled
    /// discount) OR `v3.signed_regrets` (arm 2: floor-only — `event` below requires `v3.lcfr`).
    fn apply_lcfr(&mut self, t: u64, v3: &V3Cfg) {
        // Discount events require `v3.lcfr` explicitly: in arm-2 `signed_regrets` mode this pass is
        // FLOOR-ONLY even though `lcfr_d`/`lcfr_w` carry their env defaults (250/10000) — the
        // whole point of arm 2 is pruning WITHOUT the LCFR discount's early-regret decay.
        let event = v3.lcfr && v3.lcfr_d > 0 && t <= v3.lcfr_w && t % v3.lcfr_d == 0;
        let factor = if event {
            let k = (t / v3.lcfr_d) as f64;
            k / (k + 1.0)
        } else {
            1.0
        };
        let floor = v3.regret_floor;
        self.shards.par_iter_mut().for_each(|sh| {
            for row in sh.cum_regret.values_mut() {
                for a in 0..NUM_ACTIONS {
                    let mut r = row[a] as f64;
                    if event {
                        r *= factor;
                    }
                    if r < floor {
                        r = floor;
                    }
                    row[a] = r as RegretT;
                }
            }
            if event {
                for row in sh.strat_sum.values_mut() {
                    for a in 0..NUM_ACTIONS {
                        row[a] *= factor;
                    }
                }
            }
        });
    }

    fn train_iteration_n_profile<const N: usize>(&mut self, cfg: &TrainerCfg) -> TrainIterProfile {
        let total_start = Instant::now();
        let t = self.iters_done + 1;
        let iter_seed = cfg.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);

        // Snapshot the read-side table for this iteration's traversals, shared `&` across ALL rayon
        // workers (cloned ONCE here, never per deal). Regret-matching reads the start-of-iteration
        // strategy from this immutable borrow; workers write only their deltas into a fresh empty
        // `local`. Borrow `self` immutably for the snapshot so no per-iteration full clone is needed.
        let read: &NlheTables = self;
        let v3 = v3_env();
        let eff_prune = effective_prune_threshold(cfg, iter_seed, t, v3);
        let v3i = V3Iter {
            unweighted_strat: v3.lcfr,
            strat_pre_only: v3.snapshot_avg,
        };

        let traverse_start = Instant::now();
        let (partials, stats) =
            Self::collect_partials_n_profile::<N>(cfg, read, t, iter_seed, eff_prune, v3i);
        let traverse_ms = traverse_start.elapsed().as_secs_f64() * 1000.0;

        let partials_len = partials.len();
        let partial_cum_rows: usize = partials.iter().map(|p| p.total_cum_len()).sum();
        let partial_strat_rows: usize = partials.iter().map(|p| p.total_strat_len()).sum();

        let merge_start = Instant::now();
        self.merge_partials(cfg, partials);
        let merge_ms = merge_start.elapsed().as_secs_f64() * 1000.0;

        // The `floor_ms` slot times the per-iteration regret pass: the CFR+ floor by default, OR the DCFR
        // discount when `dcfr` is ON (CHANGE B), OR the v3 LCFR floor+discount. Same accounting either way.
        let floor_start = Instant::now();
        if dcfr_enabled(cfg) {
            self.apply_dcfr_discount(t);
        } else if v3.lcfr || v3.signed_regrets {
            self.apply_lcfr(t, v3);
        } else {
            self.floor_cum_regret();
        }
        let floor_ms = floor_start.elapsed().as_secs_f64() * 1000.0;

        self.iters_done = t;
        TrainIterProfile {
            iters_done: t,
            n_deals: cfg.n_deals_per_iter,
            rayon_threads: rayon::current_num_threads(),
            fast_merge: cfg.fast_merge,
            partials: partials_len,
            partial_cum_rows,
            partial_strat_rows,
            cum_regret_rows: self.total_cum_len(),
            strat_sum_rows: self.total_strat_len(),
            traverse_ms,
            merge_ms,
            floor_ms,
            total_ms: total_start.elapsed().as_secs_f64() * 1000.0,
            stats,
        }
    }
}

// --------------------------------------------------------------------------------------------------
// HOGWILD driver (feature `hogwild`) — the per-iteration loop with NO fork/merge/floor barrier.
//
// One iteration = fan `cfg.n_deals_per_iter` deals over rayon; each deal runs `one_traversal_hogwild` for
// ALL N traverser seats, reading+writing the ONE shared table in place. There is no per-task accumulator,
// no merge_partials, and no floor_cum_regret — the CFR+ floor is applied on each regret add, and the
// shared table IS the global table. The only cross-thread synchronization is the per-slot first-visit claim
// lock inside `slot_for` (untouched after warmup). A capacity-overflow error from any deal is captured and
// surfaced as the iteration's result (the run must size up `hogwild_capacity`).
//
// READ SEMANTICS DIFFERENCE vs fork-merge (documented, expected): the fork-merge path freezes the read-side
// strategy at the START of the iteration (immutable `read` snapshot). The hogwild path reads the LIVE
// shared table, so a worker sees other workers' (and its own earlier) deltas mid-iteration. This is the
// defining property of hogwild MCCFR (Open Pure CFR / Pluribus) and the accepted noise; it is NOT bit-
// identical to fork-merge and is gated by LBR/self-play CI-overlap, not SHA.
#[cfg(feature = "hogwild")]
pub fn train_iteration_hogwild<const N: usize>(
    table: &crate::hogwild::SharedHogwildTable,
    cfg: &TrainerCfg,
    t: u64,
) -> Result<(), String> {
    use std::sync::atomic::{AtomicBool, Ordering};
    let iter_seed = cfg.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);
    // First-error capture: rayon `for_each` cannot early-return a Result, so we stash the first overflow
    // message in a Mutex and set a flag; remaining deals short-circuit cheaply.
    let failed = AtomicBool::new(false);
    let err_msg: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
    (0..cfg.n_deals_per_iter)
        .into_par_iter()
        .with_min_len(deal_chunk_min_len(
            cfg.n_deals_per_iter,
            rayon::current_num_threads(),
        ))
        .for_each(|d| {
            if failed.load(Ordering::Relaxed) {
                return;
            }
            let deal_seed = iter_seed
                .wrapping_add((d as u64).wrapping_mul(0x100000001b3))
                .rotate_left(17);
            let mut rng = Rng::seed_from_u64(deal_seed);
            let root = deal_root::<N>(&mut rng, d as u64);
            FP_MEMO.with(|cell| {
                let memo = &mut *cell.borrow_mut();
                for hero in 0..N {
                    if failed.load(Ordering::Relaxed) {
                        return;
                    }
                    if let Err(e) =
                        one_traversal_hogwild::<N>(&root, hero, cfg, table, &mut rng, t, memo)
                    {
                        if !failed.swap(true, Ordering::Relaxed) {
                            *err_msg.lock().unwrap() = Some(e);
                        }
                        return;
                    }
                }
            });
        });
    if failed.load(Ordering::Relaxed) {
        return Err(err_msg
            .lock()
            .unwrap()
            .take()
            .unwrap_or_else(|| "hogwild iteration failed".to_string()));
    }
    Ok(())
}

/// Dispatch the hogwild iteration on `cfg.n_seats` (6 default / 7 MTT anchor), mirroring
/// `train_iteration`'s dispatch. Any other width routes to 6.
#[cfg(feature = "hogwild")]
pub fn train_iteration_hogwild_dispatch(
    table: &crate::hogwild::SharedHogwildTable,
    cfg: &TrainerCfg,
    t: u64,
) -> Result<(), String> {
    match cfg.n_seats {
        7 => train_iteration_hogwild::<7>(table, cfg, t),
        _ => train_iteration_hogwild::<6>(table, cfg, t),
    }
}

pub fn deal_chunk_min_len(n_deals: usize, rayon_threads: usize) -> usize {
    if n_deals < 128 {
        return 1;
    }
    let target_chunks = rayon_threads.max(1).saturating_mul(4);
    if n_deals <= target_chunks {
        return 1;
    }
    (n_deals / target_chunks).clamp(1, 64)
}

// (The former `delta(before, after)` helper is gone: workers now accumulate into an EMPTY `local`, so
// `local` IS the delta directly — no after-minus-before diff against the read snapshot is needed.)

// --------------------------------------------------------------------------------------------------
// snapshot — the TabularBlueprintAdapter table (row-normalised average strategy)
// --------------------------------------------------------------------------------------------------

/// Average strategy = `strat_sum` row-normalised (uniform fallback for an all-zero row). This IS the
/// `dict[bytes16 -> [f64;11]]` the `TabularBlueprintAdapter` loads. Mirrors `leduc::average_strategy`.
pub fn snapshot(tab: &NlheTables) -> HashMap<[u8; 16], [f64; NUM_ACTIONS]> {
    // `with_capacity_and_hasher` (not `with_capacity`, which is RandomState-only): same capacity hint,
    // FxHash builder. Output map is identical in contents to the std-hasher version.
    let mut avg = HashMap::with_capacity_and_hasher(tab.total_strat_len(), Default::default());
    for (k, strat) in tab.iter_all_strat() {
        let total: f64 = strat.iter().sum();
        let mut row = *strat;
        if total > 0.0 {
            for a in 0..NUM_ACTIONS {
                row[a] /= total;
            }
        }
        avg.insert(*k, row);
    }
    avg
}

// --------------------------------------------------------------------------------------------------
// serialize / deserialize — genuine gap #3 (versioned, length-prefixed, 16-byte keys, magic NLCFR001)
// modelled on leduc::serialize (leduc.rs:607-696), adapted to fixed-width [u8;16] keys.
// --------------------------------------------------------------------------------------------------

pub fn serialize(tab: &NlheTables) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"NLCFR001");
    out.extend_from_slice(&tab.iters_done.to_le_bytes());
    // cum_regret rows WIDEN to f64 on the wire so the NLCFR001 blob is f64-width regardless of the
    // `f32_regret` feature: an f32-trained checkpoint is byte-format-compatible with the f64 build, and
    // the served strategy comes from strat_sum (f64) — the on-disk cum is only used to RESUME training,
    // where the f32->f64->f32 round-trip is exact. iter_all_strat is already f64.
    write_map_sharded(
        &mut out,
        tab.total_cum_len(),
        tab.iter_all_cum().map(|(k, r)| (k, widen_r(r))),
    );
    write_map_sharded(
        &mut out,
        tab.total_strat_len(),
        tab.iter_all_strat().map(|(k, r)| (k, *r)),
    );
    out
}

/// Emit a sharded map as the IDENTICAL on-disk form the old un-sharded `write_map` produced: a u64
/// length prefix, then every (16-byte key, 11 f64 LE) row in ONE GLOBAL ascending key order. We gather
/// the (key, row) pairs across ALL shards and do a single `sort_by(key)` — the same lexicographic order
/// as the old `keys.sort()` on the flat map — so the NLCFR001 blob is byte-identical regardless of
/// `N_SHARDS` (and thus checkpoint-portable). NEVER per-shard-sort-then-concat (would scramble the order).
fn write_map_sharded<'a>(
    out: &mut Vec<u8>,
    total_len: usize,
    all: impl Iterator<Item = (&'a [u8; 16], [f64; NUM_ACTIONS])>,
) {
    out.extend_from_slice(&(total_len as u64).to_le_bytes());
    // Owned f64 rows (cum_regret is widened by the caller; strat_sum copied) so the on-disk form is
    // f64-width for BOTH maps regardless of the in-memory cum precision.
    let mut pairs: Vec<(&[u8; 16], [f64; NUM_ACTIONS])> = all.collect();
    debug_assert_eq!(pairs.len(), total_len);
    pairs.sort_by(|a, b| a.0.cmp(b.0));
    for (key, row) in pairs {
        out.extend_from_slice(&key[..]); // fixed 16-byte key (no length prefix needed)
        for &v in &row {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

fn read_u64(b: &[u8], p: &mut usize) -> Option<u64> {
    if *p + 8 > b.len() {
        return None;
    }
    let v = u64::from_le_bytes(b[*p..*p + 8].try_into().ok()?);
    *p += 8;
    Some(v)
}

fn read_f64(b: &[u8], p: &mut usize) -> Option<f64> {
    if *p + 8 > b.len() {
        return None;
    }
    let v = f64::from_le_bytes(b[*p..*p + 8].try_into().ok()?);
    *p += 8;
    Some(v)
}

/// Which sub-map of a `Shard` a deserialized row belongs to.
enum MapKind {
    Regret,
    Strat,
}

/// Read one length-prefixed map's rows from the blob and route each key into its shard (`shard_of`),
/// inserting into the `Regret` or `Strat` sub-map per `which`. The on-disk row stream is the global
/// sorted form `write_map_sharded` produced; sharding on load is the inverse of the gather-and-sort on
/// save, so a blob saved at any `N_SHARDS` loads correctly at the current `N_SHARDS`.
fn read_map_into_shards(
    b: &[u8],
    p: &mut usize,
    shards: &mut [Shard; N_SHARDS],
    which: MapKind,
) -> Option<()> {
    let n = read_u64(b, p)? as usize;
    for _ in 0..n {
        if *p + 16 > b.len() {
            return None;
        }
        let mut key = [0u8; 16];
        key.copy_from_slice(&b[*p..*p + 16]);
        *p += 16;
        let mut row = [0.0f64; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            row[a] = read_f64(b, p)?;
        }
        let s = shard_of(&key);
        match which {
            // cum_regret narrows f64 (on-disk) -> RegretT (in-memory); strat_sum stays f64.
            MapKind::Regret => {
                shards[s].cum_regret.insert(key, narrow_r(&row));
            }
            MapKind::Strat => {
                shards[s].strat_sum.insert(key, row);
            }
        };
    }
    Some(())
}

/// Reconstruct `NlheTables` from a blob. Returns None on a bad magic/version/truncation (fail-closed).
pub fn deserialize(b: &[u8]) -> Option<NlheTables> {
    if b.len() < 16 || &b[0..8] != b"NLCFR001" {
        return None;
    }
    let mut p = 8usize;
    let iters_done = read_u64(b, &mut p)?;
    let mut t = NlheTables::default();
    read_map_into_shards(b, &mut p, &mut t.shards, MapKind::Regret)?;
    read_map_into_shards(b, &mut p, &mut t.shards, MapKind::Strat)?;
    t.iters_done = iters_done;
    Some(t)
}

// --------------------------------------------------------------------------------------------------
// blueprint-v4 v3 — CURRENT-STRATEGY SNAPSHOTS (Pluribus rounds-2-4 snapshot averaging) + the
// streaming offline averager, and the ON-POLICY COVERAGE probe (first-class metric per the Stage-0
// verdict §3.1: H200@20k missed 71.7% of its own postflop lookups — runs must be sized by coverage,
// and the trainer must REPORT it at checkpoint cadence).
// --------------------------------------------------------------------------------------------------

pub const SNAP_MAGIC: &[u8; 8] = b"NLSNAP01";
const SNAP_ROW_BYTES: usize = 16 + 4 * NUM_ACTIONS; // key16 + 11 f32 probs
const NLCFR_ROW_BYTES: u64 = 16 + 8 * NUM_ACTIONS as u64; // key16 + 11 f64

/// Serialize the CURRENT strategy — the regret-matched positive part of every cum_regret row — as an
/// `NLSNAP01` blob: magic, iters_done u64, n_rows u64, then (16-byte key, 11 x f32 LE probs) in ONE
/// global ascending key order (same sort discipline as `serialize`, so streams k-way-merge). Rows are
/// probabilities in [0,1]: positive regrets normalized over all 11 lanes (illegal lanes carry regret 0
/// by construction, so this equals regret-matching over the legal set whenever any positive mass
/// exists). A row with NO positive mass is stored ALL-ZERO — the current strategy there is
/// uniform-over-legal, which cannot be reconstructed from the key alone (the legal mask needs the
/// state); the offline averager SKIPS zero rows (serve falls back on a miss, same as an absent row).
/// f32 is ample for probabilities and halves the snapshot's disk/upload footprint (60 B/row).
pub fn serialize_current_strategy(tab: &NlheTables) -> Vec<u8> {
    let n = tab.total_cum_len();
    let mut out = Vec::with_capacity(24 + n * SNAP_ROW_BYTES);
    out.extend_from_slice(SNAP_MAGIC);
    out.extend_from_slice(&tab.iters_done.to_le_bytes());
    out.extend_from_slice(&(n as u64).to_le_bytes());
    let mut pairs: Vec<(&[u8; 16], [f32; NUM_ACTIONS])> = tab
        .iter_all_cum()
        .map(|(k, r)| {
            let rr = widen_r(r);
            let mut pos = [0.0f64; NUM_ACTIONS];
            let mut s = 0.0;
            for a in 0..NUM_ACTIONS {
                let v = rr[a].max(0.0);
                pos[a] = v;
                s += v;
            }
            let mut row = [0.0f32; NUM_ACTIONS];
            if s > 0.0 {
                for a in 0..NUM_ACTIONS {
                    row[a] = (pos[a] / s) as f32;
                }
            }
            (k, row)
        })
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(b.0));
    for (k, row) in pairs {
        out.extend_from_slice(&k[..]);
        for &v in &row {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Streaming reader for an `NLSNAP01` blob (any `Read`; the offline tool wraps `BufReader<File>`,
/// tests wrap `Cursor<Vec<u8>>`). Rows come back in the blob's global ascending key order; O(1) memory.
pub struct SnapshotStream<R: std::io::Read> {
    pub iters_done: u64,
    pub n_rows: u64,
    read_rows: u64,
    rdr: R,
}

impl<R: std::io::Read> SnapshotStream<R> {
    pub fn open(mut rdr: R) -> Result<Self, String> {
        let mut hdr = [0u8; 24];
        rdr.read_exact(&mut hdr).map_err(|e| format!("snapshot header: {e}"))?;
        if &hdr[0..8] != SNAP_MAGIC {
            return Err("bad snapshot magic (want NLSNAP01)".into());
        }
        let iters_done = u64::from_le_bytes(hdr[8..16].try_into().unwrap());
        let n_rows = u64::from_le_bytes(hdr[16..24].try_into().unwrap());
        Ok(SnapshotStream { iters_done, n_rows, read_rows: 0, rdr })
    }

    #[allow(clippy::type_complexity)]
    pub fn next_row(&mut self) -> Result<Option<([u8; 16], [f32; NUM_ACTIONS])>, String> {
        if self.read_rows >= self.n_rows {
            return Ok(None);
        }
        let mut buf = [0u8; SNAP_ROW_BYTES];
        self.rdr
            .read_exact(&mut buf)
            .map_err(|e| format!("snapshot row {}: {e}", self.read_rows))?;
        self.read_rows += 1;
        let mut key = [0u8; 16];
        key.copy_from_slice(&buf[0..16]);
        let mut row = [0.0f32; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            row[a] = f32::from_le_bytes(buf[16 + 4 * a..20 + 4 * a].try_into().unwrap());
        }
        Ok(Some((key, row)))
    }
}

/// Streaming reader for the STRAT section of an `NLCFR001` checkpoint (skips the cum_regret section
/// with a relative seek, so a 100 GB blob is never loaded). Under `snapshot_avg` training the strat
/// section holds ONLY the preflop (round-1) in-memory average — exactly the rows Pluribus serves from
/// the true average.
pub struct NlcfrStratStream<R: std::io::Read + std::io::Seek> {
    pub iters_done: u64,
    pub n_rows: u64,
    read_rows: u64,
    rdr: R,
}

impl<R: std::io::Read + std::io::Seek> NlcfrStratStream<R> {
    pub fn open(mut rdr: R) -> Result<Self, String> {
        let mut hdr = [0u8; 24];
        rdr.read_exact(&mut hdr).map_err(|e| format!("nlcfr header: {e}"))?;
        if &hdr[0..8] != b"NLCFR001" {
            return Err("bad checkpoint magic (want NLCFR001)".into());
        }
        let iters_done = u64::from_le_bytes(hdr[8..16].try_into().unwrap());
        let cum_len = u64::from_le_bytes(hdr[16..24].try_into().unwrap());
        rdr.seek(std::io::SeekFrom::Current((cum_len * NLCFR_ROW_BYTES) as i64))
            .map_err(|e| format!("nlcfr cum-section seek: {e}"))?;
        let mut nb = [0u8; 8];
        rdr.read_exact(&mut nb).map_err(|e| format!("nlcfr strat len: {e}"))?;
        let n_rows = u64::from_le_bytes(nb);
        Ok(NlcfrStratStream { iters_done, n_rows, read_rows: 0, rdr })
    }

    #[allow(clippy::type_complexity)]
    pub fn next_row(&mut self) -> Result<Option<([u8; 16], [f64; NUM_ACTIONS])>, String> {
        if self.read_rows >= self.n_rows {
            return Ok(None);
        }
        let mut buf = [0u8; 16 + 8 * NUM_ACTIONS];
        self.rdr
            .read_exact(&mut buf)
            .map_err(|e| format!("nlcfr strat row {}: {e}", self.read_rows))?;
        self.read_rows += 1;
        let mut key = [0u8; 16];
        key.copy_from_slice(&buf[0..16]);
        let mut row = [0.0f64; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            row[a] = f64::from_le_bytes(buf[16 + 8 * a..24 + 8 * a].try_into().unwrap());
        }
        Ok(Some((key, row)))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SnapAvgStats {
    pub rows_out: u64,
    pub rows_from_ckpt: u64,
    pub rows_from_snaps: u64,
    /// Keys whose every snapshot row was all-zero (no positive regret in any snapshot) — dropped.
    pub zero_rows_skipped: u64,
}

/// Offline snapshot averaging (Pluribus P-supp p.15 / Jackson 2016): k-way streaming merge of sorted
/// `NLSNAP01` snapshot streams + (optionally) the final checkpoint's strat section, writing an
/// `NLCFR001` blob whose cum section is EMPTY and whose strat section is the final blueprint:
///   * a key present in the CHECKPOINT strat stream (the preflop round-1 true average) is emitted
///     verbatim from the checkpoint — the in-memory average takes precedence over snapshots;
///   * any other key is the UNIFORM (per-Pluribus) mean of its non-zero snapshot rows, over the
///     snapshots in which it appears with positive mass (a lazily-allocated row simply doesn't exist
///     in earlier snapshots; an all-zero row means "no positive regret yet" — both are skipped);
///   * keys with only zero rows are dropped (serve treats them as a miss and falls back, which is
///     strictly more honest than serving a fabricated uniform row without the legal mask).
/// Output rows are normalized probabilities; downstream `snapshot()` row-normalization is then the
/// identity, so every existing NLCFR001 consumer (extractors, probes, the mmap index) reads it
/// unchanged. O(1) memory in the row count.
pub fn average_snapshots<R1, R2, W>(
    snaps: &mut [SnapshotStream<R1>],
    mut ckpt: Option<&mut NlcfrStratStream<R2>>,
    out: &mut W,
    iters_done: u64,
) -> Result<SnapAvgStats, String>
where
    R1: std::io::Read,
    R2: std::io::Read + std::io::Seek,
    W: std::io::Write + std::io::Seek,
{
    use std::io::SeekFrom;
    let mut stats = SnapAvgStats::default();
    // Header with a strat-count placeholder (fixed up by a final seek — the count is only known after
    // the merge). Cum section: length 0 (the final blueprint carries no regrets).
    out.write_all(b"NLCFR001").map_err(|e| e.to_string())?;
    out.write_all(&iters_done.to_le_bytes()).map_err(|e| e.to_string())?;
    out.write_all(&0u64.to_le_bytes()).map_err(|e| e.to_string())?; // cum len = 0
    let strat_len_pos = out.stream_position().map_err(|e| e.to_string())?;
    out.write_all(&0u64.to_le_bytes()).map_err(|e| e.to_string())?; // strat len placeholder

    // Stream heads. Ascending-order violations => corrupt input => hard error.
    let mut heads: Vec<Option<([u8; 16], [f32; NUM_ACTIONS])>> = Vec::with_capacity(snaps.len());
    for s in snaps.iter_mut() {
        heads.push(s.next_row()?);
    }
    let mut ckpt_head: Option<([u8; 16], [f64; NUM_ACTIONS])> = match ckpt.as_mut() {
        Some(c) => c.next_row()?,
        None => None,
    };
    let mut last_key: Option<[u8; 16]> = None;

    loop {
        // Minimum key across all live heads.
        let mut min_key: Option<[u8; 16]> = ckpt_head.as_ref().map(|(k, _)| *k);
        for h in heads.iter().flatten() {
            min_key = Some(match min_key {
                Some(m) if m <= h.0 => m,
                _ => h.0,
            });
        }
        let Some(key) = min_key else { break };
        if let Some(lk) = last_key {
            if key <= lk {
                return Err("input streams not strictly ascending (corrupt snapshot/ckpt)".into());
            }
        }
        last_key = Some(key);

        // Gather + advance every snapshot head at `key`.
        let mut acc = [0.0f64; NUM_ACTIONS];
        let mut n_present = 0u64;
        for (i, h) in heads.iter_mut().enumerate() {
            if let Some((k, row)) = h {
                if *k == key {
                    let s: f32 = row.iter().sum();
                    if s > 0.0 {
                        for a in 0..NUM_ACTIONS {
                            acc[a] += row[a] as f64;
                        }
                        n_present += 1;
                    }
                    *h = snaps[i].next_row()?;
                }
            }
        }
        // Checkpoint (preflop average) takes precedence at `key`.
        if let Some((k, row)) = &ckpt_head {
            if *k == key {
                out.write_all(&key[..]).map_err(|e| e.to_string())?;
                for &v in row {
                    out.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
                }
                stats.rows_out += 1;
                stats.rows_from_ckpt += 1;
                ckpt_head = match ckpt.as_mut() {
                    Some(c) => c.next_row()?,
                    None => None,
                };
                continue;
            }
        }
        if n_present == 0 {
            stats.zero_rows_skipped += 1;
            continue;
        }
        out.write_all(&key[..]).map_err(|e| e.to_string())?;
        for a in 0..NUM_ACTIONS {
            let v = acc[a] / n_present as f64;
            out.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        stats.rows_out += 1;
        stats.rows_from_snaps += 1;
    }

    // Fix up the strat-count placeholder.
    let end = out.stream_position().map_err(|e| e.to_string())?;
    out.seek(SeekFrom::Start(strat_len_pos)).map_err(|e| e.to_string())?;
    out.write_all(&stats.rows_out.to_le_bytes()).map_err(|e| e.to_string())?;
    out.seek(SeekFrom::Start(end)).map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())?;
    Ok(stats)
}

// --------------------------------------------------------------------------------------------------
// ON-POLICY COVERAGE probe — the d57d020 diagnostic made cheap enough for checkpoint cadence.
// --------------------------------------------------------------------------------------------------

/// Per-street on-policy coverage from `coverage_probe`. Streets index 0..=3 = preflop/flop/turn/river.
#[derive(Debug, Default, Clone, Copy)]
pub struct CoverageStats {
    pub hands: u64,
    pub decisions: [u64; 4],
    pub misses: [u64; 4],
    /// Hands that reached a FULL 5-card board with ZERO postflop decision nodes — the preflop-all-in
    /// runout share (the Stage-0 verdict's postflop-starvation signature: H200@20k had 65% here).
    pub preflop_allin_runouts: u64,
}

impl CoverageStats {
    pub fn miss_rate(&self, street: usize) -> f64 {
        if self.decisions[street] == 0 {
            0.0
        } else {
            self.misses[street] as f64 / self.decisions[street] as f64
        }
    }
    /// Aggregate postflop (flop+turn+river) on-policy miss rate — the pre-registered coverage-gate
    /// quantity.
    pub fn postflop_miss_rate(&self) -> f64 {
        let d: u64 = self.decisions[1..4].iter().sum();
        let m: u64 = self.misses[1..4].iter().sum();
        if d == 0 {
            0.0
        } else {
            m as f64 / d as f64
        }
    }
    pub fn preflop_allin_runout_rate(&self) -> f64 {
        if self.hands == 0 {
            0.0
        } else {
            self.preflop_allin_runouts as f64 / self.hands as f64
        }
    }
}

/// ON-POLICY coverage probe: play `n_hands` blueprint-vs-blueprint hands where EVERY seat plays the
/// CURRENT strategy (regret-matched cum_regret — the table snapshots serve; uniform-over-legal on a
/// miss) and count, per street, how many of the strategy's OWN lookups hit the trained table. This is
/// the `bpv4_onpolicy_coverage.py` (d57d020) diagnostic moved into the trainer so it can run at
/// checkpoint cadence for ~ms cost (read-only on the table; the RNG is derived from `(seed, hand)`
/// and never touches the training RNG schedule). Uses the SAME `abstracted_legal` (incl. the
/// process-global M-RR env) and fingerprint dispatch as training, so a miss here is a genuine
/// on-policy train-table miss, not a key-space artifact. NOTE the reference table is `cum_regret`
/// (current strategy): under `snapshot_avg` there IS no postflop strat_sum, and the current-strategy
/// table is what snapshot averaging serves — the Python diagnostic measured average-strategy (strat)
/// coverage instead; at H200@20k the two row sets differ (184M vs 348M rows) so absolute levels are
/// not directly comparable, trends and gates must be registered against THIS metric.
pub fn coverage_probe(
    tab: &NlheTables,
    cfg: &TrainerCfg,
    n_hands: u64,
    seed: u64,
) -> CoverageStats {
    let mut stats = CoverageStats::default();
    let mut memo: FpMemo = FpMemo::default();
    for h in 0..n_hands {
        // Same per-deal seed shape as training (pure function of (seed, h)), distinct stream.
        let deal_seed = seed
            .wrapping_add(h.wrapping_mul(0x100000001b3))
            .rotate_left(17);
        let mut rng = Rng::seed_from_u64(deal_seed);
        let mut env = deal_root::<{ NUM_PLAYERS }>(&mut rng, h);
        let mut had_postflop_decision = false;
        loop {
            if env.done {
                break;
            }
            if core::is_chance_node(&env) {
                env = core::sample_chance(&env);
                continue;
            }
            if env.street >= core::STREET_TERMINAL || env.action_seat < 0 {
                break;
            }
            let actor = env.action_seat as usize;
            let legal = abstracted_legal(&env, actor, cfg);
            if !any_legal(&legal) {
                break;
            }
            let key = fingerprint_fast_depth_memo_emd(
                &env,
                actor,
                cfg.num_buckets,
                &cfg.depth_edges,
                cfg.emd_frozen.as_deref(),
                cfg.river_ochs.as_deref(),
                &mut memo,
            );
            let st = (env.street.clamp(0, 3)) as usize;
            stats.decisions[st] += 1;
            if st > 0 {
                had_postflop_decision = true;
            }
            let row = tab.shards[shard_of(&key)].cum_regret.get(&key);
            let strat = match row {
                Some(r) => regret_match(&widen_r(r), &legal),
                None => {
                    stats.misses[st] += 1;
                    uniform_over_legal(&legal)
                }
            };
            let a = sample_action(&strat, &legal, rng.next_f64());
            env = trainer_apply(&env, a);
        }
        stats.hands += 1;
        let nb = env.board.iter().filter(|&&c| c >= 0).count();
        if !had_postflop_decision && nb == 5 {
            stats.preflop_allin_runouts += 1;
        }
    }
    stats
}

// --------------------------------------------------------------------------------------------------
// REGRET-SCALE stats — the hard-prune-C calibration instrument (log10 histogram pass, O(rows), no
// allocation proportional to the table). Drives the POKERBOT_PRUNE_C default derivation.
// --------------------------------------------------------------------------------------------------

/// Percentile summary of the stored regret values (negative tail by |value|, plus the positive tail).
#[derive(Debug, Default, Clone, Copy)]
pub struct RegStats {
    pub rows: u64,
    pub neg_vals: u64,
    pub pos_vals: u64,
    /// Negative-tail |value| percentiles: p50, p90, p99, p99.9 and the maximum |negative|.
    pub neg_p50: f64,
    pub neg_p90: f64,
    pub neg_p99: f64,
    pub neg_p999: f64,
    pub neg_max: f64,
    pub pos_p99: f64,
    pub pos_max: f64,
}

/// One pass over every cum_regret value, binning |v| into a fixed log10 histogram (1e-9..1e12, 0.05
/// decades/bin) per sign; percentiles are read off the histogram (bin upper edge — ~12% resolution,
/// ample for calibrating an order-of-magnitude prune threshold).
pub fn regret_stats(tab: &NlheTables) -> RegStats {
    const BINS: usize = 420; // 21 decades * 20 bins
    const LO: f64 = -9.0; // log10 lower edge
    let mut neg = [0u64; BINS];
    let mut pos = [0u64; BINS];
    let mut st = RegStats::default();
    for (_, row) in tab.iter_all_cum() {
        st.rows += 1;
        for a in 0..NUM_ACTIONS {
            let v = row[a] as f64;
            if v == 0.0 {
                continue;
            }
            let mag = v.abs();
            let b = (((mag.log10() - LO) * 20.0) as isize).clamp(0, BINS as isize - 1) as usize;
            if v < 0.0 {
                neg[b] += 1;
                st.neg_vals += 1;
                if mag > st.neg_max {
                    st.neg_max = mag;
                }
            } else {
                pos[b] += 1;
                st.pos_vals += 1;
                if mag > st.pos_max {
                    st.pos_max = mag;
                }
            }
        }
    }
    let pctl = |hist: &[u64; BINS], total: u64, q: f64| -> f64 {
        if total == 0 {
            return 0.0;
        }
        let target = (q * total as f64).ceil() as u64;
        let mut cum = 0u64;
        for (b, &c) in hist.iter().enumerate() {
            cum += c;
            if cum >= target {
                return 10f64.powf(LO + (b as f64 + 1.0) / 20.0);
            }
        }
        10f64.powf(LO + BINS as f64 / 20.0)
    };
    st.neg_p50 = pctl(&neg, st.neg_vals, 0.50);
    st.neg_p90 = pctl(&neg, st.neg_vals, 0.90);
    st.neg_p99 = pctl(&neg, st.neg_vals, 0.99);
    st.neg_p999 = pctl(&neg, st.neg_vals, 0.999);
    st.pos_p99 = pctl(&pos, st.pos_vals, 0.99);
    st
}

// --------------------------------------------------------------------------------------------------
// convenience entry: build a default-abstraction config matching the deployed serve
// --------------------------------------------------------------------------------------------------

/// `build_allowed_action_mask((4,5,6))` / `((0,1,3,4,5))` as native arrays: fold/check/call/all-in
/// always allowed, plus the listed bet-size bucket indices (BET_BUCKET_START + i). Mirrors
/// `pokerbot/training/sd_cfr_nlhe.build_allowed_action_mask` so the trained key space == the serve key
/// space (`pokerbot/eval/lbr_native._allowed_masks`).
pub fn allowed_mask(bet_bucket_keep: &[usize]) -> [bool; NUM_ACTIONS] {
    let mut m = [false; NUM_ACTIONS];
    m[core::FOLD] = true;
    m[core::CHECK] = true;
    m[core::CALL] = true;
    m[core::ALL_IN] = true;
    for &i in bet_bucket_keep {
        let a = core::BET_BUCKET_START + i;
        if a < NUM_ACTIONS {
            m[a] = true;
        }
    }
    m
}

// --------------------------------------------------------------------------------------------------
// serve_action — pick an action from a trained snapshot at a live decision node (any seat count N).
// Mirrors the serve-side lookup: fingerprint the state -> snapshot row -> renormalize over the FULL
// abstracted-legal set -> inverse-CDF sample. Uniform-over-legal fallback on a snapshot miss. Used by
// the native self-play behavioral eval (the 7-max LBR is concrete-6 in lbr.rs; this is the documented
// best-effort 7-max sanity, NOT a Nash-exploitability bound).
// --------------------------------------------------------------------------------------------------

/// Sample an action for `seat` at a live decision node of `env` from the row-normalized average-strategy
/// `snapshot` (the `snapshot()` output). Generic over seat count `N`. Snapshot miss => uniform over the
/// abstracted-legal set. Returns a FULL-mask-legal action index. (Test-only: drives the native 7-max
/// behavioral self-play sanity; the production serve path is `TabularBlueprintAdapter` in Python.)
#[cfg(test)]
fn serve_action<const N: usize>(
    env: &EnvStateN<N>,
    seat: usize,
    cfg: &TrainerCfg,
    snapshot: &HashMap<[u8; 16], [f64; NUM_ACTIONS]>,
    rng: &mut Rng,
) -> usize {
    let legal = abstracted_legal::<N>(env, seat, cfg);
    if !any_legal(&legal) {
        return core::FOLD;
    }
    // Test-only serve path: the non-memo fingerprint is fine here (not on the trainer hot loop).
    let key =
        crate::fingerprint::fingerprint_fast_depth(env, seat, cfg.num_buckets, &cfg.depth_edges);
    let row = match snapshot.get(&key) {
        Some(r) => regret_match_like_renorm(r, &legal),
        None => uniform_over_legal(&legal),
    };
    sample_action(&row, &legal, rng.next_f64())
}

/// Renormalize a snapshot row over the legal set (zeroing illegal mass); uniform fallback if no legal
/// mass survives. (The snapshot is already a distribution; this just guards an action-abstraction or
/// legality mismatch at serve time, exactly as `TabularBlueprintAdapter` legal-renorms.) Test-only.
#[cfg(test)]
#[inline]
fn regret_match_like_renorm(
    row: &[f64; NUM_ACTIONS],
    legal: &[bool; NUM_ACTIONS],
) -> [f64; NUM_ACTIONS] {
    let mut out = [0.0f64; NUM_ACTIONS];
    let mut s = 0.0;
    for a in 0..NUM_ACTIONS {
        if legal[a] && row[a] > 0.0 {
            out[a] = row[a];
            s += row[a];
        }
    }
    if s > 0.0 {
        for a in 0..NUM_ACTIONS {
            out[a] /= s;
        }
        out
    } else {
        uniform_over_legal(legal)
    }
}

/// Play ONE N-seat self-play hand from `root`, with each seat acting per `seat_policy` (a closure that
/// returns the chosen FULL-legal action for a (state, seat)). Advances chance via `sample_chance` and
/// settles via `compute_returns` at the post-runout terminal (FIX-1 path). Returns per-seat chip returns
/// (sums to 0 by engine zero-sum). Generic over N — the 7-max behavioral sanity entry point. Test-only.
#[cfg(test)]
fn play_hand<const N: usize, F>(mut env: EnvStateN<N>, mut seat_policy: F) -> [i32; N]
where
    F: FnMut(&EnvStateN<N>, usize) -> usize,
{
    let mut guard = 0;
    loop {
        guard += 1;
        if guard > 8192 {
            break; // defensive — a real hand always terminates well before this
        }
        if env.done {
            return env.rewards;
        }
        if core::is_chance_node(&env) {
            env = core::sample_chance(&env);
            continue;
        }
        if env.street >= core::STREET_TERMINAL {
            break; // post-runout showdown terminal (stale env.rewards) — settle below.
        }
        let actor_i = env.action_seat;
        if actor_i < 0 {
            break;
        }
        let a = seat_policy(&env, actor_i as usize);
        env = core::apply_action(&env, a);
    }
    // Showdown / no-actor terminal: settle via the engine's exact returns (the FIX-1 path).
    let n_show = env
        .seat_status
        .iter()
        .filter(|&&s| s == core::SEAT_LIVE || s == core::SEAT_ALLIN)
        .count() as i32;
    core::compute_returns(
        &env.holes,
        &env.board,
        &env.seat_status,
        &env.committed_total,
        env.button,
        n_show,
    )
}

// Dev-only public wrappers for the equivalence harness (examples/equiv_check.rs). Not on any hot path
// and not part of the serve surface; they just let an external A/B bench build a self-play fingerprint
// of the trained blueprint to gate the relaxed (statistical-equiv) levers. #[doc(hidden)] so they do
// not pollute the public API docs.
#[doc(hidden)]
pub fn deal_root_pub<const N: usize>(rng: &mut Rng, deal_idx: u64) -> EnvStateN<N> {
    deal_root::<N>(rng, deal_idx)
}

#[doc(hidden)]
pub fn abstracted_legal_pub<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    cfg: &TrainerCfg,
) -> [bool; NUM_ACTIONS] {
    abstracted_legal::<N>(s, seat, cfg)
}

#[doc(hidden)]
pub fn sample_action_pub(strat: &[f64; NUM_ACTIONS], legal: &[bool; NUM_ACTIONS], u: f64) -> usize {
    sample_action(strat, legal, u)
}

#[doc(hidden)]
pub fn uniform_over_legal_pub(legal: &[bool; NUM_ACTIONS]) -> [f64; NUM_ACTIONS] {
    uniform_over_legal(legal)
}

// `seat_status` helper note: a seat that folded preflop is SEAT_FOLDED; SEAT_LIVE for live seats. Used
// only implicitly via the engine; kept here so a future short-handed mix can reference the consts.
#[allow(dead_code)]
const _ASSERT_SEAT_CONSTS: (i8, i8) = (SEAT_LIVE, SEAT_FOLDED);

// type-check helper: the generic engine fns are called at N = NUM_PLAYERS (6). EnvState = EnvStateN<6>.
#[allow(dead_code)]
fn _assert_n6(_s: &EnvStateN<NUM_PLAYERS>) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg(n_deals: usize) -> TrainerCfg {
        TrainerCfg {
            num_buckets: 200,
            depth_edges: vec![],
            allowed_pre: allowed_mask(&[4, 5, 6]),
            allowed_post: allowed_mask(&[0, 1, 3, 4, 5]),
            n_deals_per_iter: n_deals,
            base_seed: 0xC0FFEE,
            n_seats: 6,
            // RBP disabled in the default test cfg so dispatch_n6 stays a clean self-consistency check.
            prune_threshold: f64::NEG_INFINITY,
            fast_merge: false,
            emd_frozen: None,
            batch_deals: 0,
            work_steal: false,
            dcfr: false,
            dcfr_prune_threshold: -300_000.0,
            dcfr_explore_frac: 0.05,
            river_ochs: None,
        }
    }

    fn test_cfg_7(n_deals: usize) -> TrainerCfg {
        TrainerCfg {
            num_buckets: 200,
            depth_edges: vec![],
            allowed_pre: allowed_mask(&[4, 5, 6]),
            allowed_post: allowed_mask(&[0, 1, 3, 4, 5]),
            n_deals_per_iter: n_deals,
            base_seed: 0xC0FFEE,
            n_seats: 7,
            prune_threshold: f64::NEG_INFINITY,
            fast_merge: false,
            emd_frozen: None,
            batch_deals: 0,
            work_steal: false,
            dcfr: false,
            dcfr_prune_threshold: -300_000.0,
            dcfr_explore_frac: 0.05,
            river_ochs: None,
        }
    }

    /// M-RR (menu re-raise thinning) + turn/river first-raise thinning — exercises `abstracted_legal_mrr`
    /// directly with `mrr=true`/`mrr=false` (NOT the process-global env OnceLock), on engine-built nodes.
    /// Node construction mirrors the `scripted_3bet_root()` idiom (src/tests.rs): identity-deck 100bb
    /// 6-max hand, button=0 => SB=1, BB=2, UTG=3; UTG opens 1x-pot, folds to BB who 3-bets, UTG calls,
    /// BB checks the option; flop deals with seats 2 & 3 live (BB=seat 2 first to act). `abstracted_legal`
    /// = `full_legal AND allowed`, allowed per design v2 §4.2.2 (spec §3 table). "aggressive bet-size" =
    /// the bet-fraction buckets `BET_BUCKET_START..ALL_IN` (ids 3..=9); ALL_IN (id 10) is in EVERY
    /// keep-list, so it is checked separately (never a thinning target).
    #[test]
    fn test_mrr_menu_thinning() {
        let f = core::FOLD;
        let x = core::CHECK;
        let c = core::CALL;
        let ai = core::ALL_IN;
        let bbs = core::BET_BUCKET_START;
        let cfg = test_cfg(64);

        // (1) Keep-mask helpers exactly.
        let rr = mrr_reraise_keep();
        let trf = mrr_tr_first_keep();
        for a in 0..NUM_ACTIONS {
            let want_rr = a == f || a == x || a == c || a == ai || a == bbs + 4;
            assert_eq!(rr[a], want_rr, "mrr_reraise_keep bit {a}");
            let want_trf =
                a == f || a == x || a == c || a == ai || a == bbs + 1 || a == bbs + 4;
            assert_eq!(trf[a], want_trf, "mrr_tr_first_keep bit {a}");
        }

        // Legacy postflop menu recomputed directly (spec §5 assertion 5 oracle).
        let legacy = |s: &EnvStateN<6>, seat: usize, allowed: &[bool; NUM_ACTIONS]| {
            let full = core::legal_action_mask(s, seat as i32);
            let mut m = [false; NUM_ACTIONS];
            for a in 0..NUM_ACTIONS {
                m[a] = full[a] && allowed[a];
            }
            m
        };

        // Build the flop root (identity-deck 3-bet-then-call), seats 2 & 3 live, BB=seat 2 to act.
        let stacks = [100 * core::BB_CHIPS; 6];
        let occ = [true; 6];
        let mut deck = [0i32; 52];
        for (i, d) in deck.iter_mut().enumerate() {
            *d = i as i32;
        }
        let pre_root = core::init_hand_n::<6>(0, &stacks, &occ, 0, &deck, [0u32, 0u32]);

        // (preflop, extra) both arms == legacy allowed_pre; MRR leaves preflop untouched (ON == OFF).
        assert_eq!(pre_root.street, 0, "pre_root is preflop");
        let pre_off = abstracted_legal_mrr(&pre_root, pre_root.action_seat as usize, &cfg, false);
        let pre_on = abstracted_legal_mrr(&pre_root, pre_root.action_seat as usize, &cfg, true);
        assert_eq!(
            pre_off,
            legacy(&pre_root, pre_root.action_seat as usize, &cfg.allowed_pre),
            "preflop OFF == legacy allowed_pre"
        );
        assert_eq!(pre_on, pre_off, "preflop untouched by MRR");

        let script: [(i32, usize); 8] = [
            (3, bbs + 4), // UTG opens 1x-pot
            (4, f),
            (5, f),
            (0, f),
            (1, f),
            (2, bbs + 5), // BB 3-bets 1.5x-pot
            (3, c),       // UTG calls
            (2, x),       // BB checks the option; preflop closes
        ];
        let mut env = pre_root;
        for (seat, action) in script {
            assert!(!core::is_chance_node(&env), "unexpected chance node mid-script");
            assert_eq!(env.action_seat, seat, "script actor mismatch");
            env = core::apply_action(&env, action);
        }
        while core::is_chance_node(&env) {
            env = core::sample_chance(&env);
        }
        assert_eq!(env.street, core::STREET_FLOP, "flop root");
        assert_eq!(env.action_seat, 2, "BB (seat 2) first to act on the flop");
        let flop_first = env;

        // (2) Flop FIRST raise (seat 2, not facing a bet): MRR ON == MRR OFF (flop first-raise unthinned).
        let s2 = flop_first.action_seat as usize;
        assert!(
            flop_first.current_bet <= flop_first.committed_street[s2],
            "flop first-to-act is not facing a live bet"
        );
        let ff_on = abstracted_legal_mrr(&flop_first, s2, &cfg, true);
        let ff_off = abstracted_legal_mrr(&flop_first, s2, &cfg, false);
        assert_eq!(ff_on, ff_off, "flop first raise unchanged by MRR");
        // (5) OFF == legacy allowed_post at the flop first-raise node.
        assert_eq!(
            ff_off,
            legacy(&flop_first, s2, &cfg.allowed_post),
            "flop first-raise OFF == legacy allowed_post"
        );

        // (3) Flop RE-raise: seat 2 bets 1x-pot, seat 3 acts facing the live bet.
        let facing = core::apply_action(&flop_first, bbs + 4);
        let s3 = facing.action_seat as usize;
        assert_eq!(s3, 3, "seat 3 acts facing the flop bet");
        assert!(
            facing.current_bet > facing.committed_street[s3],
            "seat 3 is facing a live bet (a raise here is a re-raise)"
        );
        let rr_on = abstracted_legal_mrr(&facing, s3, &cfg, true);
        let rr_off = abstracted_legal_mrr(&facing, s3, &cfg, false);
        // MRR-ON aggressive bet-size ids ⊆ {bbs+4}.
        for a in bbs..ai {
            if rr_on[a] {
                assert_eq!(a, bbs + 4, "MRR-ON re-raise aggressive bet id must be 1x-pot only");
            }
        }
        // Behavior change: at least one aggressive bet-size id present OFF is absent ON.
        let mut changed = false;
        for a in bbs..ai {
            if rr_off[a] && !rr_on[a] {
                changed = true;
            }
        }
        assert!(changed, "MRR must remove at least one aggressive re-raise size on the flop");
        assert!(rr_on[f] && rr_off[f], "FOLD present in both arms at re-raise");
        assert!(rr_on[c] && rr_off[c], "CALL present in both arms at re-raise");
        // (5) OFF == legacy allowed_post at the flop re-raise node.
        assert_eq!(
            rr_off,
            legacy(&facing, s3, &cfg.allowed_post),
            "flop re-raise OFF == legacy allowed_post"
        );

        // (4) Turn FIRST raise: flop check/check, sample the turn card, seat 2 first to act.
        let mut turn = core::apply_action(&flop_first, x); // seat 2 checks
        turn = core::apply_action(&turn, x); // seat 3 checks — flop closes
        while core::is_chance_node(&turn) {
            turn = core::sample_chance(&turn);
        }
        assert_eq!(turn.street, 2, "turn street == 2");
        let ts = turn.action_seat as usize;
        assert!(
            turn.current_bet <= turn.committed_street[ts],
            "turn first-to-act is not facing a live bet"
        );
        let tr_on = abstracted_legal_mrr(&turn, ts, &cfg, true);
        let tr_off = abstracted_legal_mrr(&turn, ts, &cfg, false);
        // MRR-ON aggressive bet-size ids ⊆ {bbs+1, bbs+4}.
        for a in bbs..ai {
            if tr_on[a] {
                assert!(
                    a == bbs + 1 || a == bbs + 4,
                    "MRR-ON turn first-raise aggressive bet id must be 1/2- or 1x-pot only, got {a}"
                );
            }
        }
        // ⅓ (bbs+0), ¾ (bbs+3), 1½ (bbs+5) present OFF must be absent ON (design v2 turn/river thinning).
        for a in [bbs, bbs + 3, bbs + 5] {
            if tr_off[a] {
                assert!(!tr_on[a], "MRR turn first-raise must drop bet id {a}");
            }
        }
        // At least one of them was actually legal OFF (so the check above bit on real behavior change).
        assert!(
            tr_off[bbs] || tr_off[bbs + 3] || tr_off[bbs + 5],
            "premise: some thinned turn bet size must be engine-legal OFF"
        );
        // (5) OFF == legacy allowed_post at the turn first-raise node.
        assert_eq!(
            tr_off,
            legacy(&turn, ts, &cfg.allowed_post),
            "turn first-raise OFF == legacy allowed_post"
        );
    }

    #[test]
    fn deal_root_is_a_valid_6seat_root() {
        let mut rng = Rng::seed_from_u64(1);
        let root = deal_root::<NUM_PLAYERS>(&mut rng, 0);
        assert!(!root.done);
        // All 6 seats live, each dealt 2 distinct hole cards in 0..52.
        let mut seen = [false; 52];
        for seat in 0..NUM_PLAYERS {
            assert_eq!(root.seat_status[seat], SEAT_LIVE);
            for k in 0..2 {
                let c = root.holes[seat][k];
                assert!((0..52).contains(&c), "bad hole card {c}");
                assert!(!seen[c as usize], "duplicate card {c}");
                seen[c as usize] = true;
            }
        }
        // A live actor exists preflop.
        assert!(root.action_seat >= 0);
    }

    #[test]
    fn deal_chunk_min_len_keeps_c6a32xlarge_fanout() {
        assert_eq!(deal_chunk_min_len(200, 128), 1);
        assert_eq!(deal_chunk_min_len(200, 16), 3);
        assert_eq!(deal_chunk_min_len(64, 128), 1);
        assert_eq!(deal_chunk_min_len(50000, 128), 64);
    }

    #[test]
    fn emd_bucket_memo_is_scoped_to_lookup() {
        let hole = [0, 5];
        let board = [10, 15, 22, -1, -1];
        let key = crate::emd::canonical_frozen_key(&hole, &board).expect("canonical flop key");

        let mut rows_a = HashMap::with_capacity_and_hasher(1, Default::default());
        rows_a.insert(key, 3);
        let lookup_a = FrozenEMDLookup {
            n_buckets: 10,
            n_bins: 20,
            n_runouts: 400,
            rows: rows_a,
        };

        let mut rows_b = HashMap::with_capacity_and_hasher(1, Default::default());
        rows_b.insert(key, 7);
        let lookup_b = FrozenEMDLookup {
            n_buckets: 10,
            n_bins: 20,
            n_runouts: 400,
            rows: rows_b,
        };

        assert_eq!(lookup_a.bucket(&hole, &board), Some(3));
        assert_eq!(emd_bucket_memoed(&lookup_a, &hole, &board), Some(3));

        // Same cards, different frozen table: the memo must not return lookup_a's cached bucket.
        assert_eq!(lookup_b.bucket(&hole, &board), Some(7));
        assert_eq!(emd_bucket_memoed(&lookup_b, &hole, &board), Some(7));

        let reversed_hole = [5, 0];
        let permuted_board = [22, 10, 15, -1, -1];
        assert_eq!(
            emd_bucket_memoed(&lookup_b, &reversed_hole, &permuted_board),
            Some(7)
        );
    }

    #[test]
    fn memoized_postflop_bucket_digest_matches_direct_digest() {
        let mut rng = Rng::seed_from_u64(99);
        let mut env = deal_root::<NUM_PLAYERS>(&mut rng, 0);
        env.street = core::STREET_FLOP;
        env.board = [10, 15, 22, -1, -1];
        let bucket = 42;

        let direct = crate::fingerprint::fingerprint_fast_depth_postflop_bucket(&env, 0, bucket);
        let mut memo = FpMemo::with_capacity_and_hasher(4, Default::default());
        let memoed = crate::fingerprint::fingerprint_fast_depth_postflop_bucket_memo(
            &env, 0, bucket, &mut memo,
        );
        assert_eq!(memoed, direct);
        assert_eq!(memo.len(), 1);

        let memoed_again = crate::fingerprint::fingerprint_fast_depth_postflop_bucket_memo(
            &env, 0, bucket, &mut memo,
        );
        assert_eq!(memoed_again, direct);
        assert_eq!(memo.len(), 1);
    }

    /// FIX 1 regression: a checked-down river reaches the engine's STREET_TERMINAL via `sample_chance`
    /// with stale `rewards=[0,...]`. Walking a real deal to that terminal (always CHECK/CALL, never bet/
    /// fold → all 6 seats see showdown) MUST yield a NONZERO showdown value from the trainer's terminal
    /// leaf, and it must equal a direct `core::compute_returns`. Before FIX 1 the trainer returned
    /// `env.rewards[hero] == 0` here, training every showdown against EV=0. The existing `train_smoke`
    /// only checks row normalization and would not have caught this.
    #[test]
    fn showdown_leaf_returns_nonzero_ev() {
        // Walk to a checked-down terminal from a deterministic deal: at each decision node pick CHECK if
        // legal, else CALL (never fold, never bet) so nobody is eliminated and the board runs out.
        let mut rng = Rng::seed_from_u64(42);
        let mut env = deal_root::<NUM_PLAYERS>(&mut rng, 0);
        let mut guard = 0;
        loop {
            assert!(guard < 4096, "walk did not terminate");
            guard += 1;
            if env.done || env.street >= core::STREET_TERMINAL {
                break;
            }
            if core::is_chance_node(&env) {
                env = core::sample_chance(&env);
                continue;
            }
            let seat = env.action_seat;
            assert!(
                seat >= 0,
                "non-chance, non-terminal node must have an actor"
            );
            let full = core::legal_action_mask(&env, seat);
            let a = if full[core::CHECK] {
                core::CHECK
            } else {
                core::CALL // facing a bet (blinds preflop) — flat-call, never fold
            };
            assert!(full[a], "chosen action {a} must be legal");
            env = core::apply_action(&env, a);
        }

        // The walk must land on the post-chance showdown terminal (street advanced past the river), with
        // every seat still eligible (nobody folded) — exactly the FIX 1 state with stale env.rewards.
        assert!(
            env.street >= core::STREET_TERMINAL,
            "expected a STREET_TERMINAL showdown"
        );
        let n_show = env
            .seat_status
            .iter()
            .filter(|&&s| s == core::SEAT_LIVE || s == core::SEAT_ALLIN)
            .count() as i32;
        assert_eq!(
            n_show, NUM_PLAYERS as i32,
            "all 6 seats should reach showdown"
        );

        // Direct engine returns (ground truth) — a 6-way all-in/checked pot is zero-sum and NOT all-zero.
        let direct = core::compute_returns(
            &env.holes,
            &env.board,
            &env.seat_status,
            &env.committed_total,
            env.button,
            n_show,
        );
        let sum: i32 = direct.iter().sum();
        assert_eq!(sum, 0, "chip-EV showdown must be zero-sum");
        assert!(
            direct.iter().any(|&r| r != 0),
            "showdown returns must not be all-zero"
        );

        // The trainer's terminal leaf MUST match the engine for every seat, and be nonzero somewhere.
        let mut any_nonzero = false;
        for hero in 0..NUM_PLAYERS {
            let leaf = terminal_return(&env, hero);
            let expect = direct[hero] as f64 / CHIP_UNIT;
            assert!(
                (leaf - expect).abs() < 1e-12,
                "leaf {leaf} != compute_returns {expect}"
            );
            if leaf != 0.0 {
                any_nonzero = true;
            }
        }
        assert!(
            any_nonzero,
            "FIX 1: showdown leaf returned 0 for every seat (the EV=0 bug)"
        );
    }

    /// LEVER 2 gate: the trainer-local phe `showdown_returns` must produce CHIP-IDENTICAL output to
    /// `core::compute_returns` over many random N-way showdown boards — fold-wins AND multiway showdowns,
    /// across a range of side-pot (all-in) structures. If the phe 7-card ordering ever disagreed with
    /// `score_five`'s 5-card-best ordering on eligible seats, a winner set / award would diverge and this
    /// fails. This is the exact-int proof that swapping the evaluator does not change a single chip.
    #[test]
    fn showdown_returns_matches_compute_returns() {
        const N: usize = NUM_PLAYERS;
        let mut rng = Rng::seed_from_u64(0xD0E5_5110);
        let mut checked_showdowns = 0u32;
        let mut checked_sidepots = 0u32;
        for _trial in 0..20000u64 {
            // Shuffle a full deck; deal N holes + a 5-card board (all distinct).
            let mut deck: [i32; 52] = [0; 52];
            for (i, c) in deck.iter_mut().enumerate() {
                *c = i as i32;
            }
            for i in (1..52).rev() {
                let j = rng.next_below((i + 1) as u64) as usize;
                deck.swap(i, j);
            }
            let mut holes = [[0i32; 2]; N];
            let mut p = 0usize;
            for seat in 0..N {
                holes[seat] = [deck[p], deck[p + 1]];
                p += 2;
            }
            let mut board = [-1i32; MAX_BOARD];
            for b in 0..MAX_BOARD {
                board[b] = deck[p];
                p += 1;
            }
            // Random seat statuses: each seat is LIVE, ALLIN, or FOLDED. Force at least 1 eligible.
            let mut seat_status = [core::SEAT_FOLDED; N];
            let mut committed = [0i32; N];
            let mut n_elig = 0;
            for seat in 0..N {
                let r = rng.next_below(3);
                seat_status[seat] = match r {
                    0 => core::SEAT_FOLDED,
                    1 => core::SEAT_LIVE,
                    _ => core::SEAT_ALLIN,
                };
                // Random committed chips (varied so side pots form). Folded seats may still have
                // committed dead money (folded after betting), which compute_returns handles.
                committed[seat] = (rng.next_below(20) as i32) * 50;
                if seat_status[seat] == core::SEAT_LIVE || seat_status[seat] == core::SEAT_ALLIN {
                    n_elig += 1;
                }
            }
            if n_elig == 0 {
                seat_status[0] = core::SEAT_LIVE; // ensure a valid pot owner
            }
            // committed must be >=0 and at least the eligible seats commit something for a real pot.
            // Make sure total committed > 0 so awards are non-degenerate.
            let total: i32 = committed.iter().sum();
            if total == 0 {
                committed[0] = 100;
            }
            let button = (rng.next_below(N as u64)) as i32;
            let n_show = seat_status
                .iter()
                .filter(|&&s| s == core::SEAT_LIVE || s == core::SEAT_ALLIN)
                .count() as i32;

            let want =
                core::compute_returns(&holes, &board, &seat_status, &committed, button, n_show);
            let got = showdown_returns(&holes, &board, &seat_status, &committed, button, n_show);
            assert_eq!(want, got, "showdown_returns diverged from core::compute_returns: holes={holes:?} board={board:?} status={seat_status:?} committed={committed:?} button={button} n_show={n_show}");
            if n_show >= 2 {
                checked_showdowns += 1;
                // distinct committed among eligible => a side pot is exercised
                let mut seen = std::collections::HashSet::new();
                let mut sidepot = false;
                for seat in 0..N {
                    if (seat_status[seat] == core::SEAT_LIVE
                        || seat_status[seat] == core::SEAT_ALLIN)
                        && committed[seat] > 0
                        && !seen.insert(committed[seat])
                    {
                        // a duplicate level
                    }
                    if seat_status[seat] == core::SEAT_ALLIN {
                        sidepot = true;
                    }
                }
                if sidepot {
                    checked_sidepots += 1;
                }
            }
        }
        assert!(
            checked_showdowns > 1000,
            "too few multiway showdowns exercised ({checked_showdowns})"
        );
        assert!(
            checked_sidepots > 100,
            "too few side-pot showdowns exercised ({checked_sidepots})"
        );
    }

    /// SITE-1 chip-identity gate. The trainer's IN-APPLY settle now routes through
    /// `core::apply_action_with(.., compute_returns_phe)` (the multiway all-in run-out terminals reached
    /// inside `apply_action`). This walks random legal lines BIASED to all-in (so the multiway in-apply
    /// run-out settle is exercised, not just the cheap fold-win `n<=1` branch) and asserts the phe-settled
    /// `EnvStateN` is BYTE-EQUAL to the scalar `core::apply_action` at every step. It catches a settle-arg
    /// WIRING bug that the self-consistent `dispatch_n6` gate cannot (both its paths share `trainer_apply`);
    /// the scorer chip-identity itself is proven by `showdown_returns_matches_compute_returns`.
    #[cfg(feature = "phe_showdown")]
    #[test]
    fn apply_action_with_phe_matches_scalar_byte_for_byte() {
        const N: usize = NUM_PLAYERS;
        let mut rng = Rng::seed_from_u64(0xA571_5176_C0FF_EE01);
        let mut terminals = 0u64;
        let mut multiway_showdowns = 0u64;
        for d in 0..30000u64 {
            let mut env = deal_root::<N>(&mut rng, d);
            for _step in 0..80 {
                if core::is_chance_node(&env) {
                    env = core::sample_chance(&env);
                    continue;
                }
                if env.done || env.action_seat < 0 {
                    break;
                }
                let actor = env.action_seat;
                let legal = core::legal_action_mask(&env, actor);
                let mut choices = [0usize; core::NUM_ACTIONS];
                let mut nc = 0usize;
                for a in 0..core::NUM_ACTIONS {
                    if legal[a] {
                        choices[nc] = a;
                        nc += 1;
                    }
                }
                if nc == 0 {
                    break;
                }
                // Bias 1/2 of moves to ALL_IN when legal -> drives multiway all-in run-out terminals,
                // the EXACT case the in-apply (site-1) settle must evaluate.
                let a = if legal[core::ALL_IN] && (rng.next_below(2) == 0) {
                    core::ALL_IN
                } else {
                    choices[rng.next_below(nc as u64) as usize]
                };
                let scalar = core::apply_action(&env, a);
                let phe = core::apply_action_with(&env, a, compute_returns_phe::<N>);
                assert_eq!(
                    scalar.rewards, phe.rewards,
                    "site-1 reward mismatch: deal {d} action {a}"
                );
                assert!(scalar == phe, "site-1 EnvStateN mismatch: deal {d} action {a}");
                if scalar.done {
                    terminals += 1;
                    let n_show = scalar
                        .seat_status
                        .iter()
                        .filter(|&&s| s == core::SEAT_LIVE || s == core::SEAT_ALLIN)
                        .count();
                    if n_show > 1 {
                        multiway_showdowns += 1;
                    }
                }
                env = scalar;
            }
        }
        assert!(terminals > 5000, "too few terminals exercised ({terminals})");
        assert!(
            multiway_showdowns > 200,
            "too few multiway in-apply showdowns exercised ({multiway_showdowns})"
        );
    }

    /// FOCUSED split-pot / odd-chip landmine (the one explicit-correctness ask): a board that PLAYS
    /// (royal flush on board) forces every eligible seat to TIE, so the whole pot is split — the exact
    /// case where a PHE score-direction or tie-equality error would silently corrupt awards
    /// (`award_layer` picks winners by `scores[i] == best`, then routes the odd chip by seat). Assert the
    /// phe `showdown_returns` is chip-equal to `core::compute_returns` on an even split, an odd-chip
    /// remainder (4 tied winners, indivisible pot, every button), and a side-pot tie.
    #[test]
    fn showdown_returns_split_pot_ties_match_core() {
        const N: usize = NUM_PLAYERS;
        // Royal flush in suit 0 (card id = rank*4 + suit): ranks 8..12 are the nut 5-card hand ON the
        // board, so no two hole cards can beat it -> every eligible seat ties.
        let board: [i32; MAX_BOARD] = [8 * 4, 9 * 4, 10 * 4, 11 * 4, 12 * 4];
        // Holes: low cards (0..11) disjoint from the board and from each other -> cannot improve on the
        // board's royal.
        let mut holes = [[0i32; 2]; N];
        for seat in 0..N {
            holes[seat] = [(seat as i32) * 2, (seat as i32) * 2 + 1];
        }

        // (a) even split: all N live, equal contributions, pot divisible by N -> no remainder.
        let all_live = [core::SEAT_LIVE; N];
        for button in 0..N as i32 {
            let committed = [600i32; N];
            let want = core::compute_returns(&holes, &board, &all_live, &committed, button, N as i32);
            let got = showdown_returns(&holes, &board, &all_live, &committed, button, N as i32);
            assert_eq!(want, got, "even-split tie diverged (button={button})");
            assert_eq!(want.iter().sum::<i32>(), 0, "even split not zero-sum");
        }

        // (b) odd-chip remainder: 4 live (tie) + 2 folded (dead money), all committed 101 -> pot 606,
        // 4 winners, base 151, remainder 2 -> the odd chip goes to the first eligible seat after the
        // button. Sweep the button so odd_chip_seat is exercised at every offset.
        {
            let mut status = [core::SEAT_LIVE; N];
            status[4] = core::SEAT_FOLDED;
            status[5] = core::SEAT_FOLDED;
            let committed = [101i32; N];
            for button in 0..N as i32 {
                let want =
                    core::compute_returns(&holes, &board, &status, &committed, button, 4);
                let got = showdown_returns(&holes, &board, &status, &committed, button, 4);
                assert_eq!(want, got, "odd-chip tie diverged (button={button})");
                assert_eq!(want.iter().sum::<i32>(), 0, "odd-chip split not zero-sum");
            }
        }

        // (c) side-pot tie: seat 0 all-in short -> a main pot (all tied) + a side pot (the rest, tied).
        {
            let mut status = [core::SEAT_LIVE; N];
            status[0] = core::SEAT_ALLIN;
            let mut committed = [800i32; N];
            committed[0] = 300;
            let button = 2i32;
            let want = core::compute_returns(&holes, &board, &status, &committed, button, N as i32);
            let got = showdown_returns(&holes, &board, &status, &committed, button, N as i32);
            assert_eq!(want, got, "side-pot tie diverged");
            assert_eq!(want.iter().sum::<i32>(), 0, "side-pot split not zero-sum");
        }
    }

    /// Micro-A/B for the per-eval kernel speedup (NOT a correctness gate — that is
    /// `showdown_returns_matches_compute_returns`). Run:
    /// `cargo test --release -- --ignored phe_showdown_eval_speedup --nocapture`.
    /// `#[ignore]` so the normal suite stays fast/deterministic; release so the timing is meaningful.
    #[cfg(feature = "phe_showdown")]
    #[test]
    #[ignore]
    fn phe_showdown_eval_speedup() {
        use std::time::Instant;
        let mut rng = Rng::seed_from_u64(0x5EED_F00D);
        const K: usize = 4096;
        let mut hands = [[0i32; 7]; K];
        for hand in hands.iter_mut() {
            let mut deck: [i32; 52] = [0; 52];
            for (i, c) in deck.iter_mut().enumerate() {
                *c = i as i32;
            }
            for i in (1..52).rev() {
                let j = rng.next_below((i + 1) as u64) as usize;
                deck.swap(i, j);
            }
            hand.copy_from_slice(&deck[..7]);
        }
        const REPS: usize = 64;
        let t0 = Instant::now();
        let mut acc_a: i64 = 0;
        for _ in 0..REPS {
            for h in &hands {
                acc_a = acc_a.wrapping_add(core::evaluate_seven(h) as i64);
            }
        }
        let dt_a = t0.elapsed();
        let t1 = Instant::now();
        let mut acc_b: i64 = 0;
        for _ in 0..REPS {
            for h in &hands {
                acc_b = acc_b.wrapping_add(evaluate_cards(h) as i64);
            }
        }
        let dt_b = t1.elapsed();
        let calls = (K * REPS) as f64;
        let ns_a = dt_a.as_nanos() as f64 / calls;
        let ns_b = dt_b.as_nanos() as f64 / calls;
        println!("evaluate_seven (21-combo brute force): {ns_a:.1} ns/call");
        println!("evaluate_cards (perfect-hash lookup):  {ns_b:.1} ns/call");
        println!(
            "per-eval speedup: {:.1}x   (checksum acc_a={acc_a} acc_b={acc_b})",
            ns_a / ns_b
        );
        assert!(
            ns_b * 2.0 < ns_a,
            "phe eval not >=2x faster: {ns_b:.1} vs {ns_a:.1} ns/call"
        );
    }

    #[test]
    fn train_smoke_accumulates_and_normalizes() {
        let cfg = test_cfg(8);
        let mut tab = NlheTables::new();
        for _ in 0..5 {
            tab.train_iteration(&cfg);
        }
        assert_eq!(tab.iters_done, 5);
        assert!(tab.n_strat_infosets() > 0, "no infosets trained");
        // Every snapshot row is a valid distribution (sums to ~1 or is all-zero -> fallback handled).
        let snap = snapshot(&tab);
        for (_, row) in &snap {
            let s: f64 = row.iter().sum();
            assert!(
                s == 0.0 || (s - 1.0).abs() < 1e-9,
                "row not normalized: sum={s}"
            );
            for &v in row {
                assert!(v >= -1e-12, "negative prob {v}");
            }
        }
    }

    #[test]
    fn train_profile_matches_plain_serial_byte_for_byte() {
        let cfg = test_cfg(4);
        let mut profiled = NlheTables::new();
        let mut plain = NlheTables::new();

        let p = profiled.train_iteration_profile(&cfg);
        plain.train_iteration(&cfg);

        assert_eq!(serialize(&profiled), serialize(&plain));
        assert_eq!(p.iters_done, 1);
        assert_eq!(p.n_deals, 4);
        assert_eq!(p.partials, 4);
        assert!(!p.fast_merge);
        assert_eq!(p.cum_regret_rows, profiled.total_cum_len());
        assert_eq!(p.strat_sum_rows, profiled.total_strat_len());
        assert!(p.total_ms >= 0.0);
        assert!(p.traverse_ms >= 0.0);
        assert!(p.merge_ms >= 0.0);
        assert!(p.floor_ms >= 0.0);
        assert!(p.stats.nodes > 0);
        assert_eq!(p.stats.fingerprint_calls, p.stats.decision_nodes);
        assert_eq!(
            p.stats.apply_calls,
            p.stats.hero_branches + p.stats.opponent_samples
        );
    }

    /// WF5 sharded-merge BIT-IDENTITY gate. The sharded parallel `merge_partials` (parallel-over-shards,
    /// folding partials in deal order within each shard) must produce cum_regret + strat_sum rows that
    /// are BIT-IDENTICAL (f64.to_bits equality, not merely atol) to an INDEPENDENT serial fold of the
    /// same partials into a flat map in deal order. Proof: each key lives in exactly one shard and is
    /// summed `partial[0]+partial[1]+...` in deal order in BOTH paths => same IEEE-754 += operand order
    /// => identical bits. Exercises n_deals ∈ {4,64,200} × NUM_PLAYERS ∈ {6,7}.
    #[test]
    fn sharded_merge_bit_identical_to_serial_fold() {
        for &n_deals in &[4usize, 64, 200] {
            for n_seats in [6usize, 7] {
                let mut cfg = test_cfg(n_deals);
                cfg.n_seats = n_seats;
                let read = NlheTables::new();
                let t = 1u64;
                let iter_seed = cfg.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);
                let ep = cfg.prune_threshold;
                let partials = match n_seats {
                    7 => NlheTables::collect_partials_n::<7>(&cfg, &read, t, iter_seed, ep, V3Iter::default()),
                    _ => NlheTables::collect_partials_n::<6>(&cfg, &read, t, iter_seed, ep, V3Iter::default()),
                };

                // Sharded merge (the real default path).
                let mut sharded = NlheTables::new();
                sharded.merge_partials(&cfg, partials.clone());
                sharded.floor_cum_regret();

                // Independent serial reference: flat FxHashMap, deal-order +=, then floor.
                let mut ref_cum: RegretMap = HashMap::default();
                let mut ref_strat: ShardMap = HashMap::default();
                for p in &partials {
                    for sh in p.shards.iter() {
                        for (k, row) in &sh.cum_regret {
                            let e = ref_cum.entry(*k).or_insert(ZEROS_R);
                            for a in 0..NUM_ACTIONS {
                                // SAME f64-accumulate-then-round-store (radd) as the merge paths, so the
                                // serial reference stays BIT-IDENTICAL to the sharded merge at EITHER
                                // precision (both fold partials in deal order, narrowing identically).
                                e[a] = radd(e[a], row[a] as f64);
                            }
                        }
                        for (k, row) in &sh.strat_sum {
                            let e = ref_strat.entry(*k).or_insert(ZEROS);
                            for a in 0..NUM_ACTIONS {
                                e[a] += row[a];
                            }
                        }
                    }
                }
                for row in ref_cum.values_mut() {
                    for a in 0..NUM_ACTIONS {
                        if row[a] < 0.0 {
                            row[a] = 0.0;
                        }
                    }
                }

                assert_eq!(
                    sharded.total_cum_len(),
                    ref_cum.len(),
                    "cum key-count n_deals={n_deals} n_seats={n_seats}"
                );
                assert_eq!(
                    sharded.total_strat_len(),
                    ref_strat.len(),
                    "strat key-count n_deals={n_deals} n_seats={n_seats}"
                );
                // Separate closures: RegretT::to_bits is u32 under f32_regret, u64 otherwise; strat is
                // always f64 (u64). Each is only compared against its own kind.
                let cbits = |r: &[RegretT; NUM_ACTIONS]| r.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                let sbits = |r: &[f64; NUM_ACTIONS]| r.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                for (k, row) in &ref_cum {
                    let got = sharded.shards[shard_of(k)]
                        .cum_regret
                        .get(k)
                        .expect("sharded missing cum key");
                    assert_eq!(
                        cbits(got),
                        cbits(row),
                        "cum row bits differ n_deals={n_deals} n_seats={n_seats}"
                    );
                }
                for (k, row) in &ref_strat {
                    let got = sharded.shards[shard_of(k)]
                        .strat_sum
                        .get(k)
                        .expect("sharded missing strat key");
                    assert_eq!(
                        sbits(got),
                        sbits(row),
                        "strat row bits differ n_deals={n_deals} n_seats={n_seats}"
                    );
                }
            }
        }
    }

    /// BATCHED-MLP BYTE-IDENTITY gate. The lockstep batched traversal (`batch_deals=8`) must produce a
    /// merged table BYTE-IDENTICAL (f64.to_bits / RegretT::to_bits equality, every cum_regret AND
    /// strat_sum row) to the recursive path (`batch_deals=1`), for the SAME iter_seed/t. The batched path
    /// only interleaves DIFFERENT deals (independent rng), runs the N heroes of a deal strictly
    /// sequentially, folds each deal's LocalDelta in ascending deal order, and replicates every float op
    /// verbatim — so the only thing it changes is the ORDER memory probes issue, never a value. Exercises
    /// n_deals ∈ {4,64,200} × NUM_PLAYERS ∈ {6,7}. Mirrors `sharded_merge_bit_identical_to_serial_fold`'s
    /// to_bits row comparison.
    #[test]
    fn batched_traversal_byte_identical_to_recursive() {
        for &n_deals in &[4usize, 64, 200] {
            for n_seats in [6usize, 7] {
                // Recursive reference (batch_deals=1).
                let mut cfg_rec = test_cfg(n_deals);
                cfg_rec.n_seats = n_seats;
                cfg_rec.batch_deals = 1;
                // Batched (batch_deals=8).
                let mut cfg_bat = test_cfg(n_deals);
                cfg_bat.n_seats = n_seats;
                cfg_bat.batch_deals = 8;

                let read = NlheTables::new();
                let t = 1u64;
                let iter_seed = cfg_rec.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);

                let ep_rec = cfg_rec.prune_threshold;
                let ep_bat = cfg_bat.prune_threshold;
                let (rec_partials, bat_partials) = match n_seats {
                    7 => (
                        NlheTables::collect_partials_n::<7>(&cfg_rec, &read, t, iter_seed, ep_rec, V3Iter::default()),
                        NlheTables::collect_partials_n::<7>(&cfg_bat, &read, t, iter_seed, ep_bat, V3Iter::default()),
                    ),
                    _ => (
                        NlheTables::collect_partials_n::<6>(&cfg_rec, &read, t, iter_seed, ep_rec, V3Iter::default()),
                        NlheTables::collect_partials_n::<6>(&cfg_bat, &read, t, iter_seed, ep_bat, V3Iter::default()),
                    ),
                };

                let mut rec = NlheTables::new();
                rec.merge_partials(&cfg_rec, rec_partials);
                rec.floor_cum_regret();
                let mut bat = NlheTables::new();
                bat.merge_partials(&cfg_bat, bat_partials);
                bat.floor_cum_regret();

                assert_eq!(
                    bat.total_cum_len(),
                    rec.total_cum_len(),
                    "cum key-count n_deals={n_deals} n_seats={n_seats}"
                );
                assert_eq!(
                    bat.total_strat_len(),
                    rec.total_strat_len(),
                    "strat key-count n_deals={n_deals} n_seats={n_seats}"
                );
                let cbits =
                    |r: &[RegretT; NUM_ACTIONS]| r.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                let sbits = |r: &[f64; NUM_ACTIONS]| r.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                for (k, row) in rec.iter_all_cum() {
                    let got = bat.shards[shard_of(k)]
                        .cum_regret
                        .get(k)
                        .expect("batched missing cum key");
                    assert_eq!(
                        cbits(got),
                        cbits(row),
                        "cum row bits differ n_deals={n_deals} n_seats={n_seats}"
                    );
                }
                for (k, row) in rec.iter_all_strat() {
                    let got = bat.shards[shard_of(k)]
                        .strat_sum
                        .get(k)
                        .expect("batched missing strat key");
                    assert_eq!(
                        sbits(got),
                        sbits(row),
                        "strat row bits differ n_deals={n_deals} n_seats={n_seats}"
                    );
                }
            }
        }
    }

    /// A finite `prune_threshold` (RBP on) MUST route batched cfgs back to the recursive fold (batched RBP
    /// is intentionally unimplemented). Proves the fallback: with RBP on, batch_deals=8 and the recursive
    /// path produce a byte-identical merged table (because the batched scheduler is bypassed entirely).
    #[test]
    fn batched_with_rbp_falls_back_to_recursive() {
        let n_deals = 64usize;
        let mut cfg_rec = test_cfg(n_deals);
        cfg_rec.prune_threshold = -1e6; // finite => RBP "on" (prunes nothing under the CFR+ floor, but the gate fires)
        cfg_rec.batch_deals = 1;
        let mut cfg_bat = cfg_rec.clone();
        cfg_bat.batch_deals = 8; // must be IGNORED because prune_threshold.is_finite()

        let read = NlheTables::new();
        let t = 1u64;
        let iter_seed = cfg_rec.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);
        let ep_rec = cfg_rec.prune_threshold;
        let ep_bat = cfg_bat.prune_threshold;
        let rec_partials = NlheTables::collect_partials_n::<6>(&cfg_rec, &read, t, iter_seed, ep_rec, V3Iter::default());
        let bat_partials = NlheTables::collect_partials_n::<6>(&cfg_bat, &read, t, iter_seed, ep_bat, V3Iter::default());

        let mut rec = NlheTables::new();
        rec.merge_partials(&cfg_rec, rec_partials);
        rec.floor_cum_regret();
        let mut bat = NlheTables::new();
        bat.merge_partials(&cfg_bat, bat_partials);
        bat.floor_cum_regret();
        assert_eq!(
            serialize(&rec),
            serialize(&bat),
            "RBP-on batch_deals=8 must equal recursive (fallback failed)"
        );
    }

    /// WF5 deserialize-routing + serialize-determinism gate. A trained table's blob, after a
    /// deserialize -> re-serialize round trip, is byte-identical AND every key lands in shard_of(key).
    #[test]
    fn deserialize_routes_keys_to_shards_and_reserializes_identically() {
        let cfg = test_cfg(64);
        let mut a = NlheTables::new();
        for _ in 0..3 {
            a.train_iteration(&cfg);
        }
        let blob = serialize(&a);
        let b = deserialize(&blob).expect("deserialize");
        // Round-trip byte-identity (the NLCFR001 blob is shard-count-invariant via global sort).
        assert_eq!(serialize(&b), blob, "re-serialize not byte-identical");
        // Every key sits in its shard_of(key) bucket (no cross-shard leakage on load).
        for (s, sh) in b.shards.iter().enumerate() {
            for k in sh.cum_regret.keys() {
                assert_eq!(shard_of(k), s, "cum key in wrong shard after deserialize");
            }
            for k in sh.strat_sum.keys() {
                assert_eq!(shard_of(k), s, "strat key in wrong shard after deserialize");
            }
        }
    }

    /// On-disk NLCFR001 row width is f64 (8 bytes/elem) for BOTH maps regardless of the in-memory
    /// cum_regret precision. Guarantees an f32_regret-trained checkpoint is byte-format-compatible with
    /// the f64 build (cross-precision-portable resume). Passes in both builds.
    #[test]
    fn on_disk_blob_width_is_f64_regardless_of_regret_precision() {
        let cfg = test_cfg(64);
        let mut a = NlheTables::new();
        for _ in 0..3 {
            a.train_iteration(&cfg);
        }
        let n_cum = a.total_cum_len();
        let n_strat = a.total_strat_len();
        let blob = serialize(&a);
        // magic(8) + iters_done(8) + cum[len(8) + n*(16 key + 11*8)] + strat[len(8) + n*(16 + 11*8)]
        let row_bytes = 16 + NUM_ACTIONS * 8;
        let expect = 8 + 8 + (8 + n_cum * row_bytes) + (8 + n_strat * row_bytes);
        assert_eq!(
            blob.len(),
            expect,
            "on-disk row width must be f64 (cross-precision-portable blob)"
        );
    }

    /// f32_regret build only: RegretT really is 4 bytes, and the persistent cum rows survive a
    /// serialize(f64 on disk) -> deserialize(narrow to f32) round trip exactly (the on-disk f64 was made
    /// by widening an f32, so narrowing back is lossless), with the re-serialized blob byte-stable.
    #[cfg(feature = "f32_regret")]
    #[test]
    fn f32_regret_rows_are_f32_and_roundtrip_through_f64_disk() {
        assert_eq!(
            std::mem::size_of::<RegretT>(),
            4,
            "f32_regret feature must make RegretT f32"
        );
        let cfg = test_cfg(64);
        let mut a = NlheTables::new();
        for _ in 0..4 {
            a.train_iteration(&cfg);
        }
        let blob = serialize(&a);
        let b = deserialize(&blob).expect("deserialize");
        assert_eq!(serialize(&b), blob, "re-serialized f32 checkpoint not byte-stable");
        // Every persistent f32 cum row equals its own widen->narrow image (f32 idempotent).
        for sh in b.shards.iter() {
            for row in sh.cum_regret.values() {
                assert_eq!(
                    *row,
                    narrow_r(&widen_r(row)),
                    "f32 regret row not narrow/widen-stable"
                );
            }
        }
    }

    #[test]
    fn default_train_path_avoids_profile_instrumentation() {
        let src = include_str!("trainer.rs");
        let dispatch_start = src
            .find("pub fn train_iteration(&mut self, cfg: &TrainerCfg)")
            .expect("train_iteration source");
        let profile_start = src
            .find("/// Run one CFR+ iteration and return coarse phase timings")
            .expect("train_iteration_profile source");
        let dispatch = &src[dispatch_start..profile_start];
        // v3: `train_iteration` == `train_iteration_v3(cfg, v3_env())` (env default = inert), and the
        // seat dispatch lives in `train_iteration_v3`. The guard's intent is unchanged: the default
        // path must reach the PLAIN per-seat iteration, never the profile twin.
        assert!(dispatch.contains("self.train_iteration_v3(cfg, v3_env())"));
        assert!(dispatch.contains("self.train_iteration_n_v3::<7>(cfg, v3)"));
        assert!(dispatch.contains("self.train_iteration_n_v3::<6>(cfg, v3)"));
        assert!(!dispatch.contains("train_iteration_profile"));
        assert!(!dispatch.contains("train_iteration_n_profile"));

        let plain_start = src
            .find("pub fn train_iteration_n<const N: usize>(&mut self, cfg: &TrainerCfg)")
            .expect("train_iteration_n source");
        let profile_impl_start = src
            .find("fn train_iteration_n_profile<const N: usize>")
            .expect("train_iteration_n_profile source");
        let plain_impl = &src[plain_start..profile_impl_start];
        assert!(plain_impl.contains("collect_partials_n"));
        assert!(plain_impl.contains("merge_partials"));
        assert!(plain_impl.contains("floor_cum_regret"));
        assert!(!plain_impl.contains("Instant::now"));
        assert!(!plain_impl.contains("elapsed()"));
        assert!(!plain_impl.contains("partial_cum_rows"));
        assert!(!plain_impl.contains("partial_strat_rows"));
        assert!(!plain_impl.contains("train_iteration_n_profile"));
    }

    #[test]
    fn fast_merge_profile_preserves_snapshot_invariants() {
        let mut cfg = test_cfg(8);
        cfg.fast_merge = true;
        let mut tab = NlheTables::new();

        let p = tab.train_iteration_profile(&cfg);
        tab.train_iteration(&cfg);

        assert!(p.fast_merge);
        assert_eq!(tab.iters_done, 2);
        assert!(tab.n_strat_infosets() > 0);
        let snap = snapshot(&tab);
        for (_, row) in &snap {
            let s: f64 = row.iter().sum();
            assert!(
                s == 0.0 || (s - 1.0).abs() < 1e-9,
                "row not normalized: sum={s}"
            );
            for &v in row {
                assert!(v >= -1e-12, "negative prob {v}");
            }
        }
    }

    /// 7-max generalization: `deal_root::<7>` deals a valid 7-seat root (7 live seats, 14 distinct
    /// holes in 0..52, a live preflop actor). Mirrors `deal_root_is_a_valid_6seat_root` at N=7.
    #[test]
    fn deal_root_is_a_valid_7seat_root() {
        let mut rng = Rng::seed_from_u64(1);
        let root = deal_root::<7>(&mut rng, 0);
        assert!(!root.done);
        let mut seen = [false; 52];
        for seat in 0..7 {
            assert_eq!(root.seat_status[seat], SEAT_LIVE);
            for k in 0..2 {
                let c = root.holes[seat][k];
                assert!((0..52).contains(&c), "bad hole card {c}");
                assert!(!seen[c as usize], "duplicate card {c}");
                seen[c as usize] = true;
            }
        }
        assert!(root.action_seat >= 0);
    }

    /// 7-max training smoke: `n_seats=7` dispatches to `train_iteration_n::<7>`, accumulates a
    /// non-empty table, and every snapshot row is a valid (non-negative, sums to ~1 or 0) distribution.
    /// The seat width MUST reach the fingerprint key: the 7-max preflop key space encodes 7 position
    /// enums (pos_by_offset::<7> spans 0..7) where 6-max spans 0..6, so the two tables key into
    /// LARGELY DISJOINT preflop infoset spaces. We assert the key overlap is small (most 7-seat keys
    /// are NOT present in the 6-seat table), which would be impossible if the seat count collapsed to a
    /// 6-seat key (then the spaces would coincide). This is the robust seat-reaches-key check; raw
    /// infoset COUNTS are not monotone in seat width at small sample sizes (7 seats see flops together
    /// more often, shifting the preflop/postflop mix), so a count inequality is not a valid invariant.
    #[test]
    fn train_smoke_7seat_normalizes_and_keys_distinctly() {
        let cfg6 = test_cfg(8);
        let cfg7 = test_cfg_7(8);
        let mut tab6 = NlheTables::new();
        let mut tab7 = NlheTables::new();
        for _ in 0..5 {
            tab6.train_iteration(&cfg6);
            tab7.train_iteration(&cfg7);
        }
        assert_eq!(tab7.iters_done, 5);
        assert!(tab7.n_strat_infosets() > 0, "no 7-seat infosets trained");
        let snap6 = snapshot(&tab6);
        let snap7 = snapshot(&tab7);
        for (_, row) in &snap7 {
            let s: f64 = row.iter().sum();
            assert!(
                s == 0.0 || (s - 1.0).abs() < 1e-9,
                "7-seat row not normalized: sum={s}"
            );
            for &v in row {
                assert!(v >= -1e-12, "negative prob {v}");
            }
        }
        // Seat width reaches the key: most 7-seat keys are absent from the 6-seat table. A large overlap
        // would mean the seat count never entered the fingerprint (key-space collapse). `snapshot` keys
        // are exactly the strat-sum (served) key set of each table.
        let overlap = snap7.keys().filter(|k| snap6.contains_key(*k)).count();
        let frac = overlap as f64 / snap7.len() as f64;
        assert!(
            frac < 0.5,
            "7-seat/6-seat key overlap {overlap}/{} = {frac:.2} too high — seat width did not reach the key",
            snap7.len()
        );
    }

    /// The 6-seat default path is byte-stable: `train_iteration` (n_seats=6) must produce EXACTLY the
    /// same serialized blob as the explicit const-generic `train_iteration_n::<6>` — i.e. the runtime
    /// dispatch on n_seats=6 is the same computation as the direct N=6 monomorphization (no skew from
    /// the generalization). This is the golden-preservation guard for the deployed 6-max path.
    #[test]
    fn dispatch_n6_matches_explicit_n6_byte_for_byte() {
        let cfg = test_cfg(8);
        let mut via_dispatch = NlheTables::new();
        let mut via_explicit = NlheTables::new();
        for _ in 0..4 {
            via_dispatch.train_iteration(&cfg); // n_seats=6 -> dispatch -> train_iteration_n::<6>
            via_explicit.train_iteration_n::<6>(&cfg); // direct
        }
        assert_eq!(
            serialize(&via_dispatch),
            serialize(&via_explicit),
            "n_seats=6 dispatch diverged from explicit train_iteration_n::<6>"
        );
    }

    #[test]
    fn checkpoint_resume_continues_iteration_count() {
        let cfg = test_cfg(4);
        let mut a = NlheTables::new();
        for _ in 0..3 {
            a.train_iteration(&cfg);
        }
        let blob = serialize(&a);
        let mut b = deserialize(&blob).expect("deserialize");
        assert_eq!(b.iters_done, 3);
        assert_eq!(b.total_cum_len(), a.total_cum_len());
        assert_eq!(b.total_strat_len(), a.total_strat_len());
        // A serialize round-trip must reproduce the EXACT blob (serialization byte-identity).
        assert_eq!(
            serialize(&b),
            blob,
            "deserialize->serialize round-trip not byte-identical"
        );
        // Resume continues the count.
        b.train_iteration(&cfg);
        assert_eq!(b.iters_done, 4);
    }

    #[test]
    fn deserialize_rejects_garbage() {
        assert!(deserialize(b"not a checkpoint").is_none());
        assert!(deserialize(b"").is_none());
        assert!(deserialize(b"NLCFR001").is_none()); // magic but truncated (no iters_done)
    }

    // ----------------------------------------------------------------------------------------------
    // 7-max BEHAVIORAL eval (best-effort). The native LBR (`lbr.rs`) is concrete-6 (`BeliefSet=
    // [_;NUM_PLAYERS]`, EnvState pinned N=6), so a full 7-max exploitability LOWER BOUND is a large
    // lift and is left as documented remaining work. These two sanities exercise the 7-seat trained
    // blueprint end-to-end through the real const-generic engine instead:
    //   (1) chip conservation — every 7-seat self-play hand is zero-sum (engine guarantee, but it also
    //       proves serve_action drives a legal, terminating hand at N=7);
    //   (2) blueprint vs fold-all — when 6 seats always fold and 1 plays the blueprint, the blueprint
    //       seat must NET POSITIVE over many rotated deals (it scoops the surrendered blinds and never
    //       does anything actively losing). A degenerate/garbage policy (e.g. always all-in into folds
    //       still wins here, but a policy that FOLDS its own blinds back would not) — so this is a
    //       floor sanity that the served policy is not pathologically self-defeating at N=7.
    // ----------------------------------------------------------------------------------------------

    /// (1) 7-seat self-play chip conservation: every hand's per-seat returns sum to 0, across many
    /// rotated deals, with the trained blueprint serving ALL seats. Proves `serve_action::<7>` +
    /// `play_hand::<7>` run a legal, terminating, zero-sum 7-max hand.
    #[test]
    fn sevenmax_selfplay_chip_conservation() {
        let cfg = test_cfg_7(16);
        let mut tab = NlheTables::new();
        for _ in 0..20 {
            tab.train_iteration(&cfg);
        }
        let snap = snapshot(&tab);
        let mut rng = Rng::seed_from_u64(0xBADC0DE);
        for d in 0..400u64 {
            let mut deal_rng = Rng::seed_from_u64(0x51EED ^ d.wrapping_mul(0x9E3779B97F4A7C15));
            let root = deal_root::<7>(&mut deal_rng, d);
            let rets = play_hand::<7, _>(root, |env, seat| {
                serve_action::<7>(env, seat, &cfg, &snap, &mut rng)
            });
            let sum: i32 = rets.iter().sum();
            assert_eq!(
                sum, 0,
                "7-seat hand {d} not chip-conserving: returns={rets:?}"
            );
        }
    }

    /// (2) 7-seat blueprint vs fold-all: seat S plays the blueprint, the other 6 always fold. The
    /// blueprint seat must net STRICTLY POSITIVE chips summed over all 7 button rotations × many deals
    /// (it collects the dead blinds every hand it is not itself in the blinds folding for free, and the
    /// engine never forces it to surrender). This is the behavioral floor: the served 7-max policy is
    /// not pathologically self-defeating (it beats the always-fold baseline), the documented stand-in
    /// for the (concrete-6, not-yet-generalized) native LBR lower bound.
    #[test]
    fn sevenmax_blueprint_beats_fold_all() {
        let cfg = test_cfg_7(16);
        let mut tab = NlheTables::new();
        for _ in 0..40 {
            tab.train_iteration(&cfg);
        }
        let snap = snapshot(&tab);
        let mut rng = Rng::seed_from_u64(0x7E57);
        // Average the blueprint seat's chip result over all 7 hero seats × many deals so blind position
        // is balanced (otherwise the hero sitting in the BB every hand would skew negative).
        let mut total: i64 = 0;
        let n_deals = 300u64;
        for hero in 0..7usize {
            for d in 0..n_deals {
                let mut deal_rng = Rng::seed_from_u64(
                    0xF01D ^ (d ^ (hero as u64) << 40).wrapping_mul(0x100000001b3),
                );
                let root = deal_root::<7>(&mut deal_rng, d);
                let rets = play_hand::<7, _>(root, |env, seat| {
                    if seat == hero {
                        serve_action::<7>(env, seat, &cfg, &snap, &mut rng)
                    } else {
                        // Fold if folding is legal, else check (cannot fold when nothing to call).
                        let legal = core::legal_action_mask(env, seat as i32);
                        if legal[core::FOLD] {
                            core::FOLD
                        } else if legal[core::CHECK] {
                            core::CHECK
                        } else {
                            core::CALL
                        }
                    }
                });
                total += rets[hero] as i64;
            }
        }
        assert!(
            total > 0,
            "7-max blueprint did not beat fold-all baseline (net {total} chips over {} hands) — \
             served policy is self-defeating at N=7",
            7 * n_deals
        );
    }

    // ==============================================================================================
    // CHANGE A — DYNAMIC WORK-STEALING deal scheduler tests
    // ==============================================================================================

    /// Mean per-row absolute L1 difference between two row-normalized snapshots over their SHARED key set
    /// (sum |p-q| over actions, averaged across shared keys). 0 == identical strategies; small == close.
    fn snapshot_strategy_l1(
        a: &HashMap<[u8; 16], [f64; NUM_ACTIONS]>,
        b: &HashMap<[u8; 16], [f64; NUM_ACTIONS]>,
    ) -> (f64, usize) {
        let mut total = 0.0;
        let mut shared = 0usize;
        for (k, ra) in a {
            if let Some(rb) = b.get(k) {
                let mut l1 = 0.0;
                for x in 0..NUM_ACTIONS {
                    l1 += (ra[x] - rb[x]).abs();
                }
                total += l1;
                shared += 1;
            }
        }
        if shared == 0 {
            (0.0, 0)
        } else {
            (total / shared as f64, shared)
        }
    }

    /// (a) CHANGE A statistical-equivalence: the work-stealing scheduler is NOT byte-identical to the
    /// static one (the deal->worker assignment changes the cross-partial merge order, IEEE-754 non-
    /// associative), but it must be STATISTICALLY EQUIVALENT.
    ///   * At iteration 1 (BEFORE any chaotic divergence) the infoset SET is IDENTICAL — the per-deal RNG
    ///     seed is a pure function of (iter_seed, d), so the same deals are traversed and the same infosets
    ///     are reached regardless of which worker runs each deal.
    ///   * After 12 iterations the set is NEAR-identical (a handful of infosets differ because the last-bit
    ///     merge-order divergence flips a few sampled opponent actions, reaching slightly different
    ///     downstream nodes — normal chaotic CFR amplification, the same posture as `fast_merge`) and the
    ///     mean strategy-L1 over the shared keys is LOW. The authoritative gate is an LBR/self-play
    ///     CI-overlap at convergence; this is the equivalence sanity.
    #[test]
    fn work_steal_matches_static_infoset_set_and_low_l1() {
        // (1) Iteration-1 infoset SET is IDENTICAL (no divergence yet).
        {
            let mut cfg_s = test_cfg(200);
            cfg_s.work_steal = false;
            let mut cfg_w = test_cfg(200);
            cfg_w.work_steal = true;
            let mut ts = NlheTables::new();
            let mut tw = NlheTables::new();
            ts.train_iteration(&cfg_s);
            tw.train_iteration(&cfg_w);
            let ss = snapshot(&ts);
            let sw = snapshot(&tw);
            assert_eq!(
                ss.len(),
                sw.len(),
                "iter-1 infoset count differs ({} vs {}) — work_steal traverses different deals",
                ss.len(),
                sw.len()
            );
            for k in ss.keys() {
                assert!(
                    sw.contains_key(k),
                    "iter-1 work_steal missing an infoset present in the static run"
                );
            }
        }

        // (2) After 12 iterations: NEAR-identical set + LOW strategy-L1.
        let mut cfg_static = test_cfg(200);
        cfg_static.work_steal = false;
        let mut cfg_steal = test_cfg(200);
        cfg_steal.work_steal = true;
        let mut tab_static = NlheTables::new();
        let mut tab_steal = NlheTables::new();
        for _ in 0..12 {
            tab_static.train_iteration(&cfg_static);
            tab_steal.train_iteration(&cfg_steal);
        }
        let snap_static = snapshot(&tab_static);
        let snap_steal = snapshot(&tab_steal);

        // Near-identical count: within a tiny fraction (chaotic divergence touches only a few infosets).
        let dn = (snap_static.len() as i64 - snap_steal.len() as i64).unsigned_abs();
        let rel = dn as f64 / snap_static.len().max(1) as f64;
        assert!(
            rel < 0.005,
            "work_steal infoset count diverged too far ({} vs {}, rel {rel:.4})",
            snap_static.len(),
            snap_steal.len()
        );

        // LOW strategy-L1 over the shared keys (each row L1 in [0,2]; a tiny mean == essentially the same
        // strategy after 12 iters).
        let (mean_l1, shared) = snapshot_strategy_l1(&snap_static, &snap_steal);
        assert!(shared > 100, "too few shared infosets to be meaningful ({shared})");
        assert!(
            mean_l1 < 0.10,
            "work_steal strategy-L1 too high vs static ({mean_l1:.4} over {shared} infosets) — not \
             statistically equivalent"
        );
    }

    /// CHANGE A: the work-steal collector hands out EVERY deal index exactly once (no skips, no dupes), so
    /// it covers the same deals as the static fold. We verify via the infoset count being nonzero and the
    /// iters/count matching a re-run (determinism of the *set*, not the bits). A skipped deal would drop
    /// infosets; a double-counted deal would (under the atomic) be impossible. Single-iter check keeps it
    /// before chaotic divergence so the counts are tight.
    #[test]
    fn work_steal_covers_all_deals_once() {
        let read = NlheTables::new();
        let mut cfg = test_cfg(256);
        cfg.work_steal = true;
        let t = 1u64;
        let iter_seed = cfg.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);
        let ep = cfg.prune_threshold;
        let partials = NlheTables::collect_partials_n_worksteal::<6>(&cfg, &read, t, iter_seed, ep, V3Iter::default());
        // One partial per rayon worker thread.
        assert!(!partials.is_empty(), "work-steal produced no partials");
        // Sum the strat rows across partials -> at least one infoset trained (deals were actually run).
        let total_strat: usize = partials.iter().map(|p| p.total_strat_len()).sum();
        assert!(total_strat > 0, "work-steal trained no infosets (deals not run?)");

        // Re-run: the merged infoset SET is scheduler-order-invariant (deals are a pure fn of the seed), so
        // a second work-steal collect merges to the same key count.
        let p2 = NlheTables::collect_partials_n_worksteal::<6>(&cfg, &read, t, iter_seed, ep, V3Iter::default());
        let mut m1 = NlheTables::new();
        m1.merge_partials(&cfg, partials);
        m1.floor_cum_regret();
        let mut m2 = NlheTables::new();
        m2.merge_partials(&cfg, p2);
        m2.floor_cum_regret();
        assert_eq!(
            m1.total_strat_len(),
            m2.total_strat_len(),
            "work-steal infoset count not deal-order-invariant across re-runs"
        );
    }

    // ==============================================================================================
    // CHANGE B — DCFR signed-regret + Pluribus-RBP pruning tests
    // ==============================================================================================

    /// Seed every cum_regret row so RBP actually fires: the "passive" lanes that are legal at most hero
    /// nodes (CHECK, CALL, ALL_IN) get a strong POSITIVE regret, and the others (FOLD + every bet bucket)
    /// get a deeply-NEGATIVE regret below any test threshold. regret_match then gives the positive lanes
    /// all the strategy mass (so the negative lanes have strat==0) while the negative lanes satisfy
    /// `regrets[a] <= eff_prune` — i.e. they are prune-eligible (RULE 3). At least one positive lane is
    /// legal at essentially every reachable hero node, so the negative lanes are reached as prune
    /// candidates. (All-negative seeding does NOT work: regret_match falls back to UNIFORM, giving every
    /// lane nonzero mass, so nothing is ever pruned.)
    fn seed_one_positive_rest_negative(tab: &mut NlheTables) {
        let positive_lanes = [core::CHECK, core::CALL, core::ALL_IN];
        for sh in tab.shards.iter_mut() {
            for row in sh.cum_regret.values_mut() {
                for a in 0..NUM_ACTIONS {
                    row[a] = if positive_lanes.contains(&a) {
                        1.0e6 as RegretT
                    } else {
                        -1.0e9 as RegretT
                    };
                }
            }
        }
    }

    /// CHANGE B (b): DCFR mode keeps SIGNED regret (negatives ALLOWED — the CFR+ positive floor is NOT
    /// applied) and the discount is applied. We train a few DCFR iterations and assert (1) at least one
    /// persistent cum_regret cell is NEGATIVE (impossible under the CFR+ floor, which clamps >= 0), and
    /// (2) the strat_sum is finite/non-NaN and snapshot rows still normalize.
    #[test]
    fn dcfr_keeps_signed_regret_and_applies_discount() {
        let mut cfg = test_cfg(200);
        cfg.dcfr = true;
        cfg.dcfr_prune_threshold = -300_000.0;
        cfg.dcfr_explore_frac = 0.05;
        let mut tab = NlheTables::new();
        for _ in 0..15 {
            tab.train_iteration(&cfg);
        }
        // (1) SIGNED regret survives: at least one cum_regret cell is < 0 (the CFR+ floor would forbid it).
        let mut any_negative = false;
        let mut all_finite = true;
        for sh in tab.shards.iter() {
            for row in sh.cum_regret.values() {
                for a in 0..NUM_ACTIONS {
                    let v = row[a] as f64;
                    if v < 0.0 {
                        any_negative = true;
                    }
                    if !v.is_finite() {
                        all_finite = false;
                    }
                }
            }
        }
        assert!(
            any_negative,
            "DCFR produced no negative regret — the signed-regret path is not active (floor still on?)"
        );
        assert!(all_finite, "DCFR regret went non-finite (NaN/Inf) — discount/clamp bug");

        // (2) strat_sum finite + snapshot still a valid distribution.
        let snap = snapshot(&tab);
        assert!(snap.len() > 0, "DCFR trained no infosets");
        for (_, row) in &snap {
            let s: f64 = row.iter().sum();
            assert!(s.is_finite(), "DCFR snapshot row non-finite");
            assert!(
                s == 0.0 || (s - 1.0).abs() < 1e-9,
                "DCFR snapshot row not normalized: sum={s}"
            );
        }
    }

    /// CHANGE B: the DCFR discount factors match the Pluribus / Python `dcfr_weights` reference exactly
    /// (alpha=1.5, beta=0, gamma=2): pos = t^1.5/(t^1.5+1), neg = 1/2 (t^0=1), strat = (t/(t+1))^2.
    #[test]
    fn dcfr_weights_match_pluribus_reference() {
        for &t in &[1u64, 2, 10, 100, 62000] {
            let (pos, neg, strat) = dcfr_weights(t);
            let tf = t as f64;
            let ta = tf.powf(1.5);
            let want_pos = ta / (ta + 1.0);
            let want_neg = 1.0 / 2.0; // t^0 = 1 -> 1/(1+1)
            let want_strat = (tf / (tf + 1.0)).powf(2.0);
            assert!((pos - want_pos).abs() < 1e-12, "pos factor wrong at t={t}");
            assert!((neg - want_neg).abs() < 1e-12, "neg factor wrong at t={t}");
            assert!((strat - want_strat).abs() < 1e-12, "strat factor wrong at t={t}");
            // Sanity: positive discount in (0,1), strat down-weight in (0,1), neg halves.
            assert!(pos > 0.0 && pos < 1.0);
            assert!(strat > 0.0 && strat < 1.0);
        }
    }

    /// CHANGE B (c) RULE 1a: pruning NEVER fires on the LAST betting round. The shared
    /// `dcfr_prune_node_allowed` predicate (used by BOTH production hooks) must reject the river and any
    /// later street regardless of how deeply-negative the threshold is, and accept earlier streets.
    #[test]
    fn rbp_never_prunes_last_betting_round() {
        let ep = -300_000.0; // finite, deeply negative
        // Preflop/flop/turn (streets 0,1,2) are prune SITES; river (3) and terminal (4) are NOT.
        assert!(dcfr_prune_node_allowed(core::STREET_PREFLOP, ep));
        assert!(dcfr_prune_node_allowed(core::STREET_FLOP, ep));
        assert!(dcfr_prune_node_allowed(2 /* turn */, ep));
        assert!(
            !dcfr_prune_node_allowed(core::STREET_RIVER, ep),
            "RULE 1a violated: river is a prune site"
        );
        assert!(
            !dcfr_prune_node_allowed(core::STREET_TERMINAL, ep),
            "RULE 1a violated: terminal street is a prune site"
        );
        // NEG_INFINITY (pruning disabled) rejects every street (byte-identical no-RBP path).
        for s in 0..=4 {
            assert!(!dcfr_prune_node_allowed(s, f64::NEG_INFINITY));
        }
    }

    /// CHANGE B (c) end-to-end RULES 1a + 1b: drive a full traversal with a DEEPLY-NEGATIVE seeded regret
    /// snapshot (every action prune-eligible) and a finite `eff_prune`, then assert via the test prune-probe
    /// that (1a) ZERO prunes fired on the river or later, and (1b) ZERO terminal-leading actions were
    /// pruned (they were explored instead). Also asserts pruning DID fire somewhere (the mechanism is live).
    #[test]
    fn rbp_pruning_respects_terminal_and_river_guards_e2e() {
        let cfg = test_cfg(64);
        // Build a read snapshot whose every cum_regret cell is deeply NEGATIVE so regret_match zeroes the
        // strategy (strat[a]==0) and regrets[a] <= eff_prune for every action -> all prune-eligible (RULE 3
        // satisfied) at every hero node. Train a few CFR+ iters to populate the key set, then negate.
        let mut read = NlheTables::new();
        for _ in 0..4 {
            read.train_iteration(&cfg);
        }
        // Seed a snapshot that ACTUALLY prunes: regret_match prunes action `a` only when `strat[a]==0`,
        // which needs SOME positive-regret action to absorb the mass (all-negative -> uniform fallback ->
        // nothing has 0 mass -> nothing prunes). So set ONE lane (index 0) strongly POSITIVE and every
        // other lane deeply NEGATIVE: regret_match gives lane 0 all the mass, the rest strat[a]==0 and
        // regrets[a] <= eff_prune => prune-eligible (RULE 3 met).
        seed_one_positive_rest_negative(&mut read);

        let eff_prune = -300_000.0_f64; // finite => RBP active on streets < river
        let t = 5u64;
        let iter_seed = cfg.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);
        prune_probe_reset();
        // Run a batch of deals single-threaded so the thread-local probe is observed on this thread.
        FP_MEMO.with(|cell| {
            let memo = &mut *cell.borrow_mut();
            for d in 0..64u64 {
                let deal_seed = iter_seed
                    .wrapping_add(d.wrapping_mul(0x100000001b3))
                    .rotate_left(17);
                let mut rng = Rng::seed_from_u64(deal_seed);
                let root = deal_root::<6>(&mut rng, d);
                let mut local = LocalDelta::default();
                for hero in 0..6 {
                    one_traversal::<6>(
                        &root, hero, &cfg, &read, &mut local, &mut rng, t, memo, eff_prune,
                        V3Iter::default(),
                    );
                }
            }
        });
        let probe = prune_probe_get();
        // The mechanism must actually fire (else the guards are vacuously satisfied).
        assert!(
            probe.pruned_total > 0,
            "RBP pruned nothing — cannot validate the guards (seed/threshold too weak?)"
        );
        // RULE 1a: not a single prune on the river or later.
        assert_eq!(
            probe.pruned_on_river_or_later, 0,
            "RULE 1a violated: {} prunes fired on the river or later",
            probe.pruned_on_river_or_later
        );
        // RULE 1b: not a single terminal-leading action pruned.
        assert_eq!(
            probe.terminal_leading_pruned, 0,
            "RULE 1b violated: {} terminal-leading actions pruned",
            probe.terminal_leading_pruned
        );
        // RULE 1b positive evidence: terminal-leading actions WERE encountered and explored.
        assert!(
            probe.terminal_leading_explored > 0,
            "no terminal-leading prune-eligible actions exercised — RULE 1b not actually tested"
        );
    }

    /// CHANGE B (d) RULE 2: on an EXPLORE iteration (the ~5%) the effective prune threshold is
    /// NEG_INFINITY so ALL actions are visited (no pruning), letting a deeply-negative action revive; on a
    /// prune iteration it is the deeply-negative `dcfr_prune_threshold`. The explore decision is a
    /// DETERMINISTIC function of the iteration seed (reproducible across resume). We assert both the
    /// per-iteration threshold selection and that, over many iterations, the explore fraction is ~5%.
    #[test]
    fn dcfr_explore_iterations_visit_all_actions() {
        let mut cfg = test_cfg(8);
        cfg.dcfr = true;
        cfg.dcfr_prune_threshold = -300_000.0;
        cfg.dcfr_explore_frac = 0.05;

        // Per-iteration: an explore iter -> NEG_INFINITY (prune disabled, all actions visited); a prune
        // iter -> the finite deeply-negative threshold. The decision is deterministic in iter_seed.
        let mut explore_count = 0u64;
        let n = 20000u64;
        for t in 1..=n {
            let iter_seed = cfg.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);
            let ep = effective_prune_threshold(&cfg, iter_seed, t, &V3Cfg::default());
            let is_explore = dcfr_is_explore_iter(iter_seed, cfg.dcfr_explore_frac);
            if is_explore {
                explore_count += 1;
                assert!(
                    ep == f64::NEG_INFINITY,
                    "explore iteration must disable pruning (NEG_INFINITY), got {ep}"
                );
            } else {
                assert!(
                    ep.is_finite() && ep == cfg.dcfr_prune_threshold,
                    "prune iteration must use the deeply-negative dcfr_prune_threshold, got {ep}"
                );
            }
        }
        // ~5% explore (Pluribus). Generous band around 0.05 for the finite sample.
        let frac = explore_count as f64 / n as f64;
        assert!(
            (frac - 0.05).abs() < 0.01,
            "explore fraction {frac:.4} not ~5% over {n} iters (deterministic hash mis-calibrated?)"
        );

        // DETERMINISM / resume-stability: the same seed yields the same decision every call.
        for t in [1u64, 7, 42, 1000, 62000] {
            let iter_seed = cfg.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);
            let a = dcfr_is_explore_iter(iter_seed, cfg.dcfr_explore_frac);
            let b = dcfr_is_explore_iter(iter_seed, cfg.dcfr_explore_frac);
            assert_eq!(a, b, "explore decision not deterministic for a fixed seed");
        }
    }

    /// CHANGE B end-to-end RULE 2: an explore iteration actually visits every action (the prune-probe sees
    /// ZERO prunes that iteration) while a prune iteration prunes some. Find one of each by seed and run a
    /// traversal with the deeply-negative seeded regret snapshot.
    #[test]
    fn dcfr_explore_iter_prunes_nothing_e2e() {
        let mut cfg = test_cfg(64);
        cfg.dcfr = true;
        cfg.dcfr_prune_threshold = -300_000.0;
        cfg.dcfr_explore_frac = 0.05;

        let mut read = NlheTables::new();
        for _ in 0..4 {
            read.train_iteration(&{
                let mut c = cfg.clone();
                c.dcfr = false; // populate keys with a plain CFR+ warmup
                c
            });
        }
        seed_one_positive_rest_negative(&mut read);

        // Find an explore-iteration t and a prune-iteration t by their deterministic seeds.
        let mut explore_t = None;
        let mut prune_t = None;
        for t in 1u64..5000 {
            let iter_seed = cfg.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);
            if dcfr_is_explore_iter(iter_seed, cfg.dcfr_explore_frac) {
                explore_t.get_or_insert(t);
            } else {
                prune_t.get_or_insert(t);
            }
            if explore_t.is_some() && prune_t.is_some() {
                break;
            }
        }
        let explore_t = explore_t.expect("found no explore iteration in 5000");
        let prune_t = prune_t.expect("found no prune iteration in 5000");

        let run = |t: u64| -> PruneProbe {
            let iter_seed = cfg.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);
            let eff_prune = effective_prune_threshold(&cfg, iter_seed, t, &V3Cfg::default());
            prune_probe_reset();
            FP_MEMO.with(|cell| {
                let memo = &mut *cell.borrow_mut();
                for d in 0..64u64 {
                    let deal_seed = iter_seed
                        .wrapping_add(d.wrapping_mul(0x100000001b3))
                        .rotate_left(17);
                    let mut rng = Rng::seed_from_u64(deal_seed);
                    let root = deal_root::<6>(&mut rng, d);
                    let mut local = LocalDelta::default();
                    for hero in 0..6 {
                        one_traversal::<6>(
                            &root, hero, &cfg, &read, &mut local, &mut rng, t, memo, eff_prune,
                            V3Iter::default(),
                        );
                    }
                }
            });
            prune_probe_get()
        };

        let explore_probe = run(explore_t);
        let prune_probe = run(prune_t);
        assert_eq!(
            explore_probe.pruned_total, 0,
            "RULE 2 violated: explore iteration t={explore_t} pruned {} actions (must visit all)",
            explore_probe.pruned_total
        );
        assert!(
            prune_probe.pruned_total > 0,
            "prune iteration t={prune_t} pruned nothing — cannot contrast with explore"
        );
    }

    // ==============================================================================================
    // blueprint-v4 v3 trainer-schedule tests (LCFR + hard prune + snapshot averaging + coverage).
    // Every knob is exercised through the EXPLICIT `V3Cfg`/`V3Iter` parameters (never the process-
    // global env OnceLock, which other tests must not be poisoned by). Byte-identity-when-off is
    // pinned three ways: (a) `v3_default_is_inert_byte_identical` below (train_iteration_v3(default)
    // == the dispatcher path in-process), (b) the pre/post-change ab_train golden SHAs recorded in
    // the v3 session (69ba34fb... base / 74a3bafa... fullhist+MRR), (c) the pre-existing byte-gates
    // (dispatch_n6/batched/sharded-merge) which all still pass.
    // ==============================================================================================

    /// Golden inertness: the v3-parameterized iteration with `V3Cfg::default()` serializes byte-for-
    /// byte identically to the legacy `train_iteration` dispatcher (which reads the -- unset -- env).
    #[test]
    fn v3_default_is_inert_byte_identical() {
        let cfg = test_cfg(64);
        let mut a = NlheTables::new();
        let mut b = NlheTables::new();
        let v3 = V3Cfg::default();
        for _ in 0..4 {
            a.train_iteration(&cfg);
            b.train_iteration_v3(&cfg, &v3);
        }
        assert_eq!(serialize(&a), serialize(&b), "V3Cfg::default() not inert");
    }

    /// `apply_lcfr` semantics on a hand-built table: (1) a discount iteration multiplies BOTH signs
    /// of cum_regret and every strat_sum lane by k/(k+1); (2) a non-discount iteration leaves values
    /// untouched except the floor clamp; (3) the floor clamps below; (4) after `lcfr_w` no discount.
    #[test]
    fn lcfr_discount_and_floor_semantics() {
        let key1 = [1u8; 16];
        let key2 = [2u8; 16];
        let mut tab = NlheTables::new();
        {
            let sh = &mut tab.shards[shard_of(&key1)];
            let mut r = ZEROS_R;
            r[0] = 10.0 as RegretT;
            r[1] = -4.0 as RegretT;
            sh.cum_regret.insert(key1, r);
        }
        {
            let sh = &mut tab.shards[shard_of(&key2)];
            let mut ss = ZEROS;
            ss[2] = 6.0;
            sh.strat_sum.insert(key2, ss);
        }
        let v3 = V3Cfg {
            lcfr: true,
            lcfr_d: 10,
            lcfr_w: 30,
            regret_floor: -3.0,
            ..V3Cfg::default()
        };
        // t=20 => k = 2 => factor 2/3, applied to +10 / -4 / strat 6; then floor(-3) clamps -4*2/3
        // = -2.6667 (above the floor, no clamp).
        tab.apply_lcfr(20, &v3);
        let r1 = widen_r(tab.shards[shard_of(&key1)].cum_regret.get(&key1).unwrap());
        assert!((r1[0] - 10.0 * (2.0 / 3.0)).abs() < 1e-12, "pos regret discount: {}", r1[0]);
        assert!((r1[1] - (-4.0 * (2.0 / 3.0))).abs() < 1e-12, "neg regret discount: {}", r1[1]);
        let s2 = *tab.shards[shard_of(&key2)].strat_sum.get(&key2).unwrap();
        assert!((s2[2] - 6.0 * (2.0 / 3.0)).abs() < 1e-12, "strat discount: {}", s2[2]);
        // t=21: NOT a discount event; only the floor applies. Push a value below the floor first.
        {
            let sh = &mut tab.shards[shard_of(&key1)];
            let r = sh.cum_regret.get_mut(&key1).unwrap();
            r[3] = -100.0 as RegretT;
        }
        let before_pos = r1[0];
        tab.apply_lcfr(21, &v3);
        let r1b = widen_r(tab.shards[shard_of(&key1)].cum_regret.get(&key1).unwrap());
        assert_eq!(r1b[0], before_pos, "non-event iteration must not discount");
        assert_eq!(r1b[3], -3.0, "floor must clamp -100 to regret_floor");
        let s2b = *tab.shards[shard_of(&key2)].strat_sum.get(&key2).unwrap();
        assert!((s2b[2] - 6.0 * (2.0 / 3.0)).abs() < 1e-12, "strat untouched on non-event");
        // t=40 > lcfr_w=30: no discount even though 40 % 10 == 0.
        tab.apply_lcfr(40, &v3);
        let r1c = widen_r(tab.shards[shard_of(&key1)].cum_regret.get(&key1).unwrap());
        assert_eq!(r1c[0], before_pos, "discount fired past lcfr_w");
    }

    /// LCFR mode keeps SIGNED regrets (vs CFR+'s positive floor) and accumulates the average strategy
    /// UNWEIGHTED: from an identical (empty) start at iteration t=5, the traversals are identical
    /// (strategy reads see the same regrets), so the LCFR arm's strat_sum row must equal the CFR+
    /// arm's row divided by w=t=5, and the CFR+ regrets must be the positive part of the LCFR ones.
    #[test]
    fn lcfr_signed_regret_and_unweighted_strat() {
        let cfg = test_cfg(64);
        let mut classic = NlheTables::new();
        let mut lin = NlheTables::new();
        classic.iters_done = 4; // next iteration is t=5
        lin.iters_done = 4;
        let v3 = V3Cfg {
            lcfr: true,
            lcfr_d: 0, // no discount events -- isolates the weighting + sign behavior
            ..V3Cfg::default()
        };
        classic.train_iteration_v3(&cfg, &V3Cfg::default());
        lin.train_iteration_v3(&cfg, &v3);
        // strat_sum: classic w=5, lcfr w=1 -- same visits, same per-visit probs.
        let mut strat_rows = 0usize;
        for (k, row) in lin.iter_all_strat() {
            let c = classic.shards[shard_of(k)]
                .strat_sum
                .get(k)
                .expect("strat key set must match");
            for a in 0..NUM_ACTIONS {
                assert!(
                    (c[a] - 5.0 * row[a]).abs() <= 1e-9 * c[a].abs().max(1.0),
                    "strat weight mismatch at lane {a}: classic {} vs 5*lcfr {}",
                    c[a],
                    5.0 * row[a]
                );
            }
            strat_rows += 1;
        }
        assert!(strat_rows > 0, "no strat rows trained");
        assert_eq!(strat_rows, classic.total_strat_len(), "strat key sets differ");
        // regrets: identical pre-floor sums => classic == max(lcfr, 0); and some negative survived.
        let mut saw_negative = false;
        for (k, row) in lin.iter_all_cum() {
            let widened = widen_r(row);
            let c = widen_r(
                classic.shards[shard_of(k)]
                    .cum_regret
                    .get(k)
                    .expect("cum key set must match"),
            );
            for a in 0..NUM_ACTIONS {
                if widened[a] < 0.0 {
                    saw_negative = true;
                }
                assert_eq!(c[a], widened[a].max(0.0), "classic must be the positive part");
            }
        }
        assert!(saw_negative, "LCFR mode floored negatives away (should keep signed regret)");
    }

    /// Arm-2 `signed_regrets` mode (blueprint_v4_arm2_amendment.md): signed regrets like LCFR mode,
    /// but the average-strategy accumulation KEEPS CFR+'s w=t linear weighting. From an identical
    /// (empty) start at t=5 the single iteration's traversals coincide (regret-matching reads
    /// `max(0,·)` of an empty table either way), so the strat_sum table must be BYTE-IDENTICAL to
    /// the classic CFR+ run (SAME w=t — contrast `lcfr_signed_regret_and_unweighted_strat`, where
    /// the LCFR row is classic/5), while the cum table keeps signed values whose positive part
    /// equals the classic table. NOTE: beyond one iteration the two arms genuinely diverge — signed
    /// regret is vanilla-CFR regret memory (`max(0, raw+inc) != max(0, max(0,raw)+inc)`); an action
    /// must dig out of its negative hole to revive. That memory is exactly what makes the hard prune
    /// threshold meaningful, so a multi-iteration byte-compare is deliberately NOT asserted.
    #[test]
    fn signed_regrets_keeps_wt_strat_and_signed_cum() {
        let cfg = test_cfg(64);
        let mut classic = NlheTables::new();
        let mut signed = NlheTables::new();
        classic.iters_done = 4; // next iteration is t=5 (w = t = 5 on both arms)
        signed.iters_done = 4;
        let v3 = V3Cfg { signed_regrets: true, ..V3Cfg::default() };
        classic.train_iteration_v3(&cfg, &V3Cfg::default());
        signed.train_iteration_v3(&cfg, &v3);
        // strat_sum: w=t on BOTH arms + identical traversals => byte-identical rows.
        assert_eq!(
            classic.total_strat_len(),
            signed.total_strat_len(),
            "strat key sets differ"
        );
        for (k, row) in signed.iter_all_strat() {
            let c = classic.shards[shard_of(k)]
                .strat_sum
                .get(k)
                .expect("strat key set must match");
            for a in 0..NUM_ACTIONS {
                assert_eq!(c[a], row[a], "w=t strat mismatch at lane {a}");
            }
        }
        // cum: classic must be the positive part; some negative must survive (no CFR+ floor).
        let mut saw_negative = false;
        for (k, row) in signed.iter_all_cum() {
            let widened = widen_r(row);
            let c = widen_r(
                classic.shards[shard_of(k)]
                    .cum_regret
                    .get(k)
                    .expect("cum key set must match"),
            );
            for a in 0..NUM_ACTIONS {
                if widened[a] < 0.0 {
                    saw_negative = true;
                }
                assert_eq!(c[a], widened[a].max(0.0), "classic must be the positive part");
            }
        }
        assert!(saw_negative, "signed_regrets mode floored negatives away");
    }

    /// Arm-2 `signed_regrets` mode fires NO LCFR discount events even when `lcfr_d`/`lcfr_w` carry
    /// live values (they default to 250/10000 from the env reader): `apply_lcfr` must be FLOOR-ONLY.
    #[test]
    fn signed_regrets_never_discounts() {
        let key1 = [1u8; 16];
        let key2 = [2u8; 16];
        let mut tab = NlheTables::new();
        {
            let sh = &mut tab.shards[shard_of(&key1)];
            let mut r = ZEROS_R;
            r[0] = 10.0 as RegretT;
            r[1] = -4.0 as RegretT;
            r[3] = -100.0 as RegretT;
            sh.cum_regret.insert(key1, r);
        }
        {
            let sh = &mut tab.shards[shard_of(&key2)];
            let mut ss = ZEROS;
            ss[2] = 6.0;
            sh.strat_sum.insert(key2, ss);
        }
        let v3 = V3Cfg {
            signed_regrets: true,
            lcfr_d: 10, // would be a discount event at t=20 under lcfr mode
            lcfr_w: 30,
            regret_floor: -3.0,
            ..V3Cfg::default()
        };
        tab.apply_lcfr(20, &v3); // t % lcfr_d == 0 and t <= lcfr_w — MUST still not discount
        let r1 = widen_r(tab.shards[shard_of(&key1)].cum_regret.get(&key1).unwrap());
        assert_eq!(r1[0], 10.0, "signed_regrets discounted a positive regret");
        assert_eq!(r1[1], -3.0, "floor must clamp -4 to regret_floor");
        assert_eq!(r1[3], -3.0, "floor must clamp -100 to regret_floor");
        let s2 = *tab.shards[shard_of(&key2)].strat_sum.get(&key2).unwrap();
        assert_eq!(s2[2], 6.0, "signed_regrets discounted a strat_sum row");
    }

    /// v3 hard pruning fires under the LCFR schedule and honors the Pluribus rules: never on the
    /// river, never a terminal-leading action, only from `prune_start`, and the floor bounds stored
    /// regret. Mirrors `rbp_never_prunes_last_betting_round` but drives the v3 (non-DCFR) path.
    #[test]
    fn v3_hard_prune_fires_with_pluribus_rules() {
        let mut cfg = test_cfg(48);
        cfg.base_seed = 0xB0B0;
        // A LENIENT threshold so a short run can reach it: C = -0.05 bb. (Production C comes from the
        // amendment-doc calibration; the rules under test are threshold-magnitude-independent.)
        let v3 = V3Cfg {
            lcfr: true,
            lcfr_d: 0,
            prune_c: -0.05,
            prune_start: 3,
            prune_explore: 0.0, // every eligible iteration prunes => deterministic assertion
            regret_floor: -0.05 * (31.0 / 30.0),
            ..V3Cfg::default()
        };
        let mut tab = NlheTables::new();
        // Warm 2 iterations (below prune_start: eff_prune must stay -inf).
        for t in 1..=2u64 {
            let iter_seed = cfg.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);
            assert_eq!(
                effective_prune_threshold(&cfg, iter_seed, t, &v3),
                f64::NEG_INFINITY,
                "pruning before prune_start"
            );
            tab.train_iteration_v3(&cfg, &v3);
        }
        // From t=3 the threshold is C on every iteration (explore_frac 0). Warm a few more so
        // regrets actually cross C.
        for _ in 0..6 {
            tab.train_iteration_v3(&cfg, &v3);
        }
        // NB: train_iteration_v3 fans over rayon; PRUNE_PROBE is per-thread. Run one more iteration
        // single-threaded through one_traversal directly so the probe observes it.
        let t = tab.iters_done + 1;
        let iter_seed = cfg.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);
        let eff_prune = effective_prune_threshold(&cfg, iter_seed, t, &v3);
        assert_eq!(eff_prune, v3.prune_c, "prune threshold must be C from prune_start");
        prune_probe_reset();
        let read = &tab;
        FP_MEMO.with(|cell| {
            let memo = &mut *cell.borrow_mut();
            for d in 0..48u64 {
                let deal_seed = iter_seed
                    .wrapping_add(d.wrapping_mul(0x100000001b3))
                    .rotate_left(17);
                let mut rng = Rng::seed_from_u64(deal_seed);
                let root = deal_root::<6>(&mut rng, d);
                let mut local = LocalDelta::default();
                for hero in 0..6 {
                    one_traversal::<6>(
                        &root,
                        hero,
                        &cfg,
                        read,
                        &mut local,
                        &mut rng,
                        t,
                        memo,
                        eff_prune,
                        V3Iter { unweighted_strat: true, strat_pre_only: false },
                    );
                }
            }
        });
        let probe = prune_probe_get();
        assert!(probe.pruned_total > 0, "v3 hard prune never fired (C=-0.05, 8 warm iters)");
        assert_eq!(probe.pruned_on_river_or_later, 0, "RULE 1a violated: pruned on river");
        assert_eq!(probe.terminal_leading_pruned, 0, "RULE 1b violated: pruned terminal-leading");
        // Floor: every stored regret must be >= regret_floor after the iteration passes.
        for (_, row) in tab.iter_all_cum() {
            for a in 0..NUM_ACTIONS {
                assert!(
                    (row[a] as f64) >= v3.regret_floor - 1e-12,
                    "regret {} below floor {}",
                    row[a],
                    v3.regret_floor
                );
            }
        }
    }

    /// `effective_prune_threshold` v3 paths: off before prune_start, C on prune iterations,
    /// NEG_INFINITY on explore iterations, DCFR precedence, inert when lcfr off.
    #[test]
    fn effective_prune_threshold_v3_paths() {
        let cfg = test_cfg(8);
        let v3 = V3Cfg {
            lcfr: true,
            prune_c: -1000.0,
            prune_start: 100,
            prune_explore: 1.0, // always explore
            ..V3Cfg::default()
        };
        assert_eq!(
            effective_prune_threshold(&cfg, 7, 50, &v3),
            f64::NEG_INFINITY,
            "before prune_start"
        );
        assert_eq!(
            effective_prune_threshold(&cfg, 7, 200, &v3),
            f64::NEG_INFINITY,
            "explore_frac=1 must always explore"
        );
        let v3p = V3Cfg { prune_explore: 0.0, ..v3 };
        assert_eq!(effective_prune_threshold(&cfg, 7, 200, &v3p), -1000.0, "prune iteration");
        let v3off = V3Cfg { lcfr: false, ..v3p };
        assert_eq!(
            effective_prune_threshold(&cfg, 7, 200, &v3off),
            cfg.prune_threshold,
            "lcfr off => legacy threshold"
        );
        // Arm-2: signed_regrets alone (lcfr off) must activate the SAME v3 prune constants.
        let v3sr = V3Cfg { signed_regrets: true, ..v3off };
        assert_eq!(
            effective_prune_threshold(&cfg, 7, 200, &v3sr),
            -1000.0,
            "signed_regrets must activate hard pruning"
        );
        assert_eq!(
            effective_prune_threshold(&cfg, 7, 50, &v3sr),
            f64::NEG_INFINITY,
            "signed_regrets before prune_start"
        );
        let mut cfg_dcfr = test_cfg(8);
        cfg_dcfr.dcfr = true;
        cfg_dcfr.dcfr_explore_frac = 0.0;
        assert_eq!(
            effective_prune_threshold(&cfg_dcfr, 7, 200, &v3p),
            cfg_dcfr.dcfr_prune_threshold,
            "DCFR takes precedence over v3"
        );
    }

    /// snapshot_avg mode: cum_regret must be BYTE-IDENTICAL to the default run (strat_sum is never
    /// read during training), the strat_sum row set must shrink (postflop rows skipped), and every
    /// retained row must be byte-equal to the default run's row for the same key (preflop rows are
    /// accumulated identically).
    #[test]
    fn snapshot_avg_drops_postflop_strat_keeps_cum_identical() {
        let cfg = test_cfg(128);
        let mut full = NlheTables::new();
        let mut snap = NlheTables::new();
        let v3 = V3Cfg { snapshot_avg: true, ..V3Cfg::default() };
        for _ in 0..4 {
            full.train_iteration_v3(&cfg, &V3Cfg::default());
            snap.train_iteration_v3(&cfg, &v3);
        }
        // cum sections byte-identical.
        let sf = serialize(&full);
        let ss = serialize(&snap);
        let cum_len = u64::from_le_bytes(sf[16..24].try_into().unwrap());
        assert_eq!(
            u64::from_le_bytes(ss[16..24].try_into().unwrap()),
            cum_len,
            "cum row counts differ"
        );
        let cum_end = 24 + (cum_len as usize) * (16 + 8 * NUM_ACTIONS);
        assert_eq!(sf[..cum_end], ss[..cum_end], "cum sections differ (strat skip leaked)");
        // strat: strictly smaller, and retained rows byte-equal.
        assert!(
            snap.total_strat_len() < full.total_strat_len(),
            "snapshot_avg dropped nothing ({} vs {})",
            snap.total_strat_len(),
            full.total_strat_len()
        );
        assert!(snap.total_strat_len() > 0, "snapshot_avg kept no preflop rows");
        for (k, row) in snap.iter_all_strat() {
            let f = full.shards[shard_of(k)]
                .strat_sum
                .get(k)
                .expect("retained strat key missing from full run");
            assert_eq!(row, f, "retained (preflop) strat row differs from full run");
        }
    }

    /// `serialize_current_strategy` roundtrip: header sane, rows globally ascending, count == cum
    /// rows, and each row equals the positive-part-normalized cum row (zero row iff no positive mass).
    #[test]
    fn snapshot_current_strategy_roundtrip() {
        let cfg = test_cfg(64);
        let mut tab = NlheTables::new();
        for _ in 0..3 {
            tab.train_iteration_v3(&cfg, &V3Cfg::default());
        }
        let blob = serialize_current_strategy(&tab);
        let mut rd = SnapshotStream::open(std::io::Cursor::new(&blob)).expect("open snapshot");
        assert_eq!(rd.iters_done, 3);
        assert_eq!(rd.n_rows as usize, tab.total_cum_len());
        let mut prev: Option<[u8; 16]> = None;
        let mut n = 0u64;
        while let Some((k, row)) = rd.next_row().expect("row") {
            if let Some(pk) = prev {
                assert!(k > pk, "snapshot keys not strictly ascending");
            }
            prev = Some(k);
            let cum = widen_r(tab.shards[shard_of(&k)].cum_regret.get(&k).expect("cum row"));
            let s: f64 = cum.iter().map(|v| v.max(0.0)).sum();
            if s > 0.0 {
                for a in 0..NUM_ACTIONS {
                    let want = (cum[a].max(0.0) / s) as f32;
                    assert!(
                        (row[a] - want).abs() <= 1e-6,
                        "prob mismatch lane {a}: {} vs {}",
                        row[a],
                        want
                    );
                }
            } else {
                assert!(row.iter().all(|&v| v == 0.0), "no-positive-mass row must be all-zero");
            }
            n += 1;
        }
        assert_eq!(n, rd.n_rows);
    }

    /// Offline snapshot averaging: hand-built snapshots + checkpoint exercise the merge rules --
    /// uniform mean over present non-zero rows, checkpoint (round-1 average) precedence, zero-row
    /// skipping -- and the output must be a loadable NLCFR001 blob with an empty cum section.
    #[test]
    fn snap_avg_streaming_merge_rules() {
        use std::io::Cursor;
        let ka = [1u8; 16]; // in both snapshots
        let kb = [2u8; 16]; // snapshot 1 only
        let kc = [3u8; 16]; // zero in both snapshots => dropped
        let kd = [4u8; 16]; // in ckpt AND snapshot 2 => ckpt wins
        let mk_snap = |iters: u64, rows: Vec<([u8; 16], [f32; NUM_ACTIONS])>| -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(SNAP_MAGIC);
            out.extend_from_slice(&iters.to_le_bytes());
            out.extend_from_slice(&(rows.len() as u64).to_le_bytes());
            for (k, row) in rows {
                out.extend_from_slice(&k[..]);
                for &v in &row {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            out
        };
        let row = |vals: &[(usize, f32)]| -> [f32; NUM_ACTIONS] {
            let mut r = [0.0f32; NUM_ACTIONS];
            for &(i, v) in vals {
                r[i] = v;
            }
            r
        };
        let s1 = mk_snap(
            10,
            vec![(ka, row(&[(0, 1.0)])), (kb, row(&[(1, 0.5), (2, 0.5)])), (kc, row(&[]))],
        );
        let s2 = mk_snap(
            20,
            vec![(ka, row(&[(0, 0.5), (1, 0.5)])), (kc, row(&[])), (kd, row(&[(2, 1.0)]))],
        );
        // Checkpoint: cum EMPTY (irrelevant), strat holds kd (the "preflop average" row).
        let mut ckpt_tab = NlheTables::new();
        {
            let sh = &mut ckpt_tab.shards[shard_of(&kd)];
            let mut r = ZEROS;
            r[5] = 42.0;
            sh.strat_sum.insert(kd, r);
        }
        ckpt_tab.iters_done = 20;
        let ckpt_blob = serialize(&ckpt_tab);
        let mut snaps = vec![
            SnapshotStream::open(Cursor::new(&s1)).unwrap(),
            SnapshotStream::open(Cursor::new(&s2)).unwrap(),
        ];
        let mut ckpt = NlcfrStratStream::open(Cursor::new(&ckpt_blob)).unwrap();
        let mut out = Cursor::new(Vec::<u8>::new());
        let stats = average_snapshots(&mut snaps, Some(&mut ckpt), &mut out, 20).expect("average");
        assert_eq!(stats.rows_out, 3, "ka, kb, kd");
        assert_eq!(stats.rows_from_ckpt, 1);
        assert_eq!(stats.rows_from_snaps, 2);
        assert_eq!(stats.zero_rows_skipped, 1, "kc dropped");
        let tab = deserialize(&out.into_inner()).expect("output must be loadable NLCFR001");
        assert_eq!(tab.iters_done, 20);
        assert_eq!(tab.total_cum_len(), 0, "final blueprint carries no regrets");
        assert_eq!(tab.total_strat_len(), 3);
        let ra = tab.shards[shard_of(&ka)].strat_sum.get(&ka).unwrap();
        assert!((ra[0] - 0.75).abs() < 1e-6 && (ra[1] - 0.25).abs() < 1e-6, "mean of snapshots");
        let rb = tab.shards[shard_of(&kb)].strat_sum.get(&kb).unwrap();
        assert!((rb[1] - 0.5).abs() < 1e-6, "single-snapshot mean is the row itself");
        let rdd = tab.shards[shard_of(&kd)].strat_sum.get(&kd).unwrap();
        assert_eq!(rdd[5], 42.0, "checkpoint strat row must win over the snapshot");
    }

    /// Coverage probe: an EMPTY table misses every lookup; training reduces the preflop miss rate;
    /// the probe is deterministic in (table, seed) and never mutates the table.
    #[test]
    fn coverage_probe_empty_then_trained() {
        let cfg = test_cfg(64);
        let empty = NlheTables::new();
        let c0 = coverage_probe(&empty, &cfg, 64, 0xC0FFEE);
        assert_eq!(c0.hands, 64);
        assert!(c0.decisions[0] > 0, "no preflop decisions probed");
        assert_eq!(c0.misses, c0.decisions, "empty table must miss everything");
        let mut tab = NlheTables::new();
        for _ in 0..20 {
            tab.train_iteration_v3(&cfg, &V3Cfg::default());
        }
        let before = serialize(&tab);
        let c1 = coverage_probe(&tab, &cfg, 64, 0xC0FFEE);
        let c2 = coverage_probe(&tab, &cfg, 64, 0xC0FFEE);
        assert_eq!(serialize(&tab), before, "probe mutated the table");
        assert_eq!(
            (c1.decisions, c1.misses, c1.preflop_allin_runouts),
            (c2.decisions, c2.misses, c2.preflop_allin_runouts),
            "probe not deterministic"
        );
        assert!(
            c1.miss_rate(0) < 1.0,
            "trained table still misses 100% preflop on-policy ({}/{})",
            c1.misses[0],
            c1.decisions[0]
        );
    }

    /// The batched-MLP path must FALL BACK to the recursive path when any v3 iter mode is active
    /// (the batched state machine replicates only the legacy float ops). Proven by byte-equality of
    /// the batched-cfg run against the recursive run under the same v3 config.
    #[test]
    fn batched_falls_back_under_v3() {
        // Single-iteration collect_partials comparison from an EMPTY read table (the proven
        // stability envelope of `batched_with_rbp_falls_back_to_recursive`): multi-iteration
        // whole-train comparisons amplify rayon fold-boundary low-bit reordering into visible
        // divergence (chaotic regret amplification), which is scheduling noise, not a fallback
        // failure. Here both configs must take the RECURSIVE path (v3 active => batched fallback),
        // so the merged tables serialize byte-identically.
        let n_deals = 64usize;
        let cfg_rec = test_cfg(n_deals);
        let mut cfg_bat = test_cfg(n_deals);
        cfg_bat.batch_deals = 8; // must be IGNORED because v3i is active
        let v3i = V3Iter { unweighted_strat: false, strat_pre_only: true };
        let read = NlheTables::new();
        let t = 1u64;
        let iter_seed = cfg_rec.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);
        let rec_partials = NlheTables::collect_partials_n::<6>(
            &cfg_rec,
            &read,
            t,
            iter_seed,
            cfg_rec.prune_threshold,
            v3i,
        );
        let bat_partials = NlheTables::collect_partials_n::<6>(
            &cfg_bat,
            &read,
            t,
            iter_seed,
            cfg_bat.prune_threshold,
            v3i,
        );
        let mut rec = NlheTables::new();
        rec.merge_partials(&cfg_rec, rec_partials);
        rec.floor_cum_regret();
        let mut bat = NlheTables::new();
        bat.merge_partials(&cfg_bat, bat_partials);
        bat.floor_cum_regret();
        assert_eq!(
            serialize(&rec),
            serialize(&bat),
            "v3-active batch_deals=8 must equal recursive (fallback failed)"
        );
        // And the v3 mode itself demonstrably did something: strat rows were thinned vs a full run.
        let full_partials = NlheTables::collect_partials_n::<6>(
            &cfg_rec,
            &read,
            t,
            iter_seed,
            cfg_rec.prune_threshold,
            V3Iter::default(),
        );
        let mut full = NlheTables::new();
        full.merge_partials(&cfg_rec, full_partials);
        full.floor_cum_regret();
        assert!(
            rec.total_strat_len() < full.total_strat_len(),
            "strat_pre_only dropped nothing in the fallback path"
        );
    }

    /// `regret_stats` histogram-percentile sanity on a hand-built table (the C-calibration tool).
    #[test]
    fn regret_stats_percentiles_sane() {
        let mut tab = NlheTables::new();
        for i in 0..100u8 {
            let mut k = [0u8; 16];
            k[0] = i;
            let mut r = ZEROS_R;
            r[0] = -((i as f64) + 1.0) as RegretT; // negatives -1..-100
            r[1] = 5.0 as RegretT;
            tab.shards[shard_of(&k)].cum_regret.insert(k, r);
        }
        let st = regret_stats(&tab);
        assert_eq!(st.rows, 100);
        assert_eq!(st.neg_vals, 100);
        assert_eq!(st.pos_vals, 100);
        assert_eq!(st.neg_max, 100.0);
        // Histogram-resolution (~12%) checks around the true percentiles (p50 ~ 50, p90 ~ 90).
        assert!(st.neg_p50 > 35.0 && st.neg_p50 < 70.0, "neg_p50 {}", st.neg_p50);
        assert!(st.neg_p90 > 70.0 && st.neg_p90 < 120.0, "neg_p90 {}", st.neg_p90);
        assert!(st.pos_p99 > 3.5 && st.pos_p99 < 7.0, "pos_p99 {}", st.pos_p99);
    }
}

