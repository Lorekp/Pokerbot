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

#[inline]
fn abstracted_legal<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    cfg: &TrainerCfg,
) -> [bool; NUM_ACTIONS] {
    let full = core::legal_action_mask(s, seat as i32);
    let allowed = if s.street == 0 {
        &cfg.allowed_pre
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
    memo: &mut FpMemo,
) -> [u8; 16] {
    if let Some(lookup) = emd_frozen {
        // Try the frozen EMD bucket for flop/turn postflop. `lookup.bucket` returns `None` for
        // preflop / river / malformed spots — those go through the made-hand path (river is
        // made-hand-equivalent by construction; see `emd::strength_histogram`).
        let nb = s.board.iter().filter(|&&c| c >= 0).count();
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
    stats: &mut TraverseStats,
) -> [u8; 16] {
    stats.fingerprint_calls += 1;
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
fn one_traversal<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    cfg: &TrainerCfg,
    read: &NlheTables,
    local: &mut LocalDelta,
    rng: &mut Rng,
    t: u64,
    memo: &mut FpMemo,
) -> f64 {
    if env.done {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    if core::is_chance_node(env) {
        let child = core::sample_chance(env);
        return one_traversal(&child, hero, cfg, read, local, rng, t, memo);
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
        let w = t as f64;
        {
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
        return one_traversal(&child, hero, cfg, read, local, rng, t, memo);
    }

    // Hero node: branch ALL abstracted-legal actions, CFR+ regret update. Regrets are read from the
    // IMMUTABLE iteration snapshot `read` (start-of-iteration strategy, read once above), never `local`.
    let strat = regret_match(&regrets, &legal);
    let mut av = [0.0f64; NUM_ACTIONS];
    // LEVER 7 — Regret-Based Pruning (Brown-Sandholm). An action whose start-of-iteration cumulative
    // regret is `<= prune_threshold` (and which therefore carries 0 strategy mass) would have its entire
    // subtree — and that subtree's dominant showdown evals — skipped; its `av[a]` is backfilled to `ev`
    // (regret delta 0 => frozen => stays pruned). `prune_threshold = NEG_INFINITY` (the default + every
    // shipped cfg) DISABLES it: the `prune.is_finite()` gate below makes that path BYTE-IDENTICAL to the
    // no-RBP trainer (no `pruned` array, no extra checks).
    //
    // ARCHITECTURAL NOTE (why RBP is shipped present-but-OFF): textbook RBP keys on DEEPLY NEGATIVE
    // cumulative regret. This trainer applies the CFR+ positive floor `max(r,0)` to the PERSISTENT
    // `cum_regret` table once per iteration (FIX 3, mirrors leduc.rs — load-bearing, prevents regret
    // over-inflation). So `read.cum_regret[key][a]` is ALWAYS `>= 0` at read time: the negative-regret
    // signal RBP needs is destroyed by the floor, and a measured threshold sweep (C from -1e6 down to
    // -10 bb) prunes ZERO subtrees (no read-regret is ever < 0). Making RBP fire would require either
    // removing the load-bearing CFR+ floor or maintaining a separate un-floored shadow-regret signal —
    // both larger, fixed-point-risk changes outside this pass's safe scope. The mechanism is verified
    // correct (disabled == byte-identical; the gate + backfill are sound) and left wired for a future
    // shadow-regret variant, but is NOT enabled because it cannot deliver speedup against the CFR+ floor.
    let prune = cfg.prune_threshold;
    if prune.is_finite() {
        let mut pruned = [false; NUM_ACTIONS];
        for i in 0..ln {
            let a = lidx[i];
            if strat[a] == 0.0 && regrets[a] <= prune {
                pruned[a] = true;
                continue; // skip the subtree (and its showdowns); av[a] backfilled to ev below.
            }
            let child = trainer_apply(env, a);
            av[a] = one_traversal(&child, hero, cfg, read, local, rng, t, memo);
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
    // RBP disabled (default) — the zero-overhead path: branch all legal actions, no prune machinery.
    for i in 0..ln {
        let a = lidx[i];
        let child = trainer_apply(env, a);
        av[a] = one_traversal(&child, hero, cfg, read, local, rng, t, memo);
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
        let w = t as f64;
        {
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
        );
    }

    stats.hero_nodes += 1;
    let strat = regret_match(&regrets, &legal);
    let mut av = [0.0f64; NUM_ACTIONS];
    let prune = cfg.prune_threshold;
    if prune.is_finite() {
        let mut pruned = [false; NUM_ACTIONS];
        for i in 0..ln {
            let a = lidx[i];
            if strat[a] == 0.0 && regrets[a] <= prune {
                pruned[a] = true;
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
        match cfg.n_seats {
            7 => self.train_iteration_n::<7>(cfg),
            // 6 is the default + the deployed path; any unrecognized width also routes here so the
            // trainer never silently produces keys for a seat count nothing serves.
            _ => self.train_iteration_n::<6>(cfg),
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
        let t = self.iters_done + 1;
        let iter_seed = cfg.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);
        let read: &NlheTables = self;
        let partials = Self::collect_partials_n::<N>(cfg, read, t, iter_seed);
        self.merge_partials(cfg, partials);
        self.floor_cum_regret();
        self.iters_done = t;
    }

    fn collect_partials_n<const N: usize>(
        cfg: &TrainerCfg,
        read: &NlheTables,
        t: u64,
        iter_seed: u64,
    ) -> Vec<NlheTables> {
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
                        one_traversal::<N>(&root, hero, cfg, read, &mut local, &mut rng, t, memo);
                    }
                });
                accumulate_local_into(&mut acc, local);
                acc
            })
            .collect()
    }

    fn collect_partials_n_profile<const N: usize>(
        cfg: &TrainerCfg,
        read: &NlheTables,
        t: u64,
        iter_seed: u64,
    ) -> (Vec<NlheTables>, TraverseStats) {
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
                                0,
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

    fn train_iteration_n_profile<const N: usize>(&mut self, cfg: &TrainerCfg) -> TrainIterProfile {
        let total_start = Instant::now();
        let t = self.iters_done + 1;
        let iter_seed = cfg.base_seed ^ t.wrapping_mul(0x9E3779B97F4A7C15);

        // Snapshot the read-side table for this iteration's traversals, shared `&` across ALL rayon
        // workers (cloned ONCE here, never per deal). Regret-matching reads the start-of-iteration
        // strategy from this immutable borrow; workers write only their deltas into a fresh empty
        // `local`. Borrow `self` immutably for the snapshot so no per-iteration full clone is needed.
        let read: &NlheTables = self;

        let traverse_start = Instant::now();
        let (partials, stats) = Self::collect_partials_n_profile::<N>(cfg, read, t, iter_seed);
        let traverse_ms = traverse_start.elapsed().as_secs_f64() * 1000.0;

        let partials_len = partials.len();
        let partial_cum_rows: usize = partials.iter().map(|p| p.total_cum_len()).sum();
        let partial_strat_rows: usize = partials.iter().map(|p| p.total_strat_len()).sum();

        let merge_start = Instant::now();
        self.merge_partials(cfg, partials);
        let merge_ms = merge_start.elapsed().as_secs_f64() * 1000.0;

        let floor_start = Instant::now();
        self.floor_cum_regret();
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
        }
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
                let partials = match n_seats {
                    7 => NlheTables::collect_partials_n::<7>(&cfg, &read, t, iter_seed),
                    _ => NlheTables::collect_partials_n::<6>(&cfg, &read, t, iter_seed),
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
        assert!(dispatch.contains("self.train_iteration_n::<7>(cfg)"));
        assert!(dispatch.contains("self.train_iteration_n::<6>(cfg)"));
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
}
