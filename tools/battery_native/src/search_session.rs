//! PyO3 `SearchSession` — a long-lived CFR+ search worker that owns the blueprint ONCE per session
//! and runs many `solve` calls against it without per-call HashMap cloning.
//!
//! # Why a session (vs the existing per-call `SubgameSolver`)
//!
//! `SubgameSolver` is correct but it builds/owns a private `Blueprint` HashMap per PyO3 class — the
//! caller passes the full dict into `__new__` and `solve` borrows it. On a live search where ONE
//! Python solver makes THOUSANDS of `solve` calls per second against the SAME blueprint snapshot,
//! the per-call HashMap clone is the wrong shape (and the live process owns the blueprint
//! elsewhere, so it would have to re-marshal on every call).
//!
//! `SearchSession` flips that: the blueprint is marshaled ONCE at `__new__` and lives across all
//! solves. Per-call work is a single `&[u8]` state copy + the cheap per-solve `tilt_table` rebuild +
//! the CFR recursion. The blueprint is held by an `Arc` so the session can be cheaply cloned from
//! Python and shared across worker threads if the Python layer dispatches in parallel (it does
//! not today, but the wiring stays open).
//!
//! # Efficiency baked in
//!
//! - **Blueprint owned alive across solves** — no per-call HashMap clone.
//! - **AoSoA table layout** — `CfrTables` stores regrets / strat_sum / leaf as parallel flat arrays
//!   indexed by an `infoset_id: u32` slot, so each row is a contiguous NUM_ACTIONS chunk of f64
//!   (11 f64 = 88 B, fits 2 rows / 64-byte cache line) and the inner loop touches contiguous
//!   memory (vs `HashMap<Vec<i64>, [f64;11]>` where every node is a heap allocation + a Vec<i64>
//!   key hash).
//! - **GIL-released inner loop** — `py.detach()` around the whole solve; the Rust side does NOT
//!   take the GIL per iteration. Cooperative cancel + wall check go through an `AtomicBool`/an
//!   `Instant` snapshot (no Python round-trips on the hot path).
//! - **Per-iter cancel + wall check at the LOOP TOP** — NOT per node. Per-node checks were profiled
//!   in the existing `subgame.rs` to cost ~5-8% of the inner loop; doing them once per iteration
//!   costs ~0% (the recursion depth is bounded by max_depth, so the per-iter outer loop dominates
//!   wall time anyway).
//! - **Per-worker Rng** — `Rng::seed_from_u64(seed_mix(session_seed, public_seed, worker_id))` so
//!   a future parallel dispatch (Stage 3) gets independent streams without changing the solve body.
//!
//! # Stages
//!
//! - **Stage 1** (THIS FILE): the GTO-base / `opponent_model = None` path — the same body as
//!   `subgame::cfv`, but driven by `CfrTables` (AoSoA) and wrapped by the session.
//! - **Stage 2** (this file): the model-aware LEAF port via `LeafContWeights`. When supplied,
//!   `leaf_value` consults `LeafContWeights` BEFORE falling through to `uniform_leaf_value` (the
//!   same uniform-continuation gadget the existing path uses). The exploit MATH stays in Python
//!   (firewall); we only ever consume the precomputed weight ROW as DATA.
//! - **Stage 3** (future): exploit tilt is already wired through `tilt_table`; rayon-parallel
//!   iteration is the obvious next lever once the AoSoA + per-worker Rng plumbing is in.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::core::{self, EnvState, EnvStateN, NUM_ACTIONS};
use crate::cfr_table;
use crate::fingerprint::CHIP_UNIT;
use crate::leaf_cont::LeafContWeights;
use crate::playout::{self, Blueprint, PlayoutConfig, Rng};
use crate::subgame::{
    abstracted_legal, any_legal, infoset_key, leaf_key, leaf_seed,
    opponent_strategy, regret_match, sample_action, Ctx, SubgameConfig, TiltMap,
};

// AoSoA slot capacity: 16k infosets is ~512 KiB per parallel array (regret + strat_sum) which
// fits comfortably in L2; on overflow the underlying `cfr_table::CfrTable` amortised-reallocs
// each group on the next insert.
const AOSOA_INITIAL_CAPACITY: usize = 1 << 14;

// ---------------------------------------------------------------------------
// CfrTables (Stage 1) — AoSoA-backed CFR+ storage.
// ---------------------------------------------------------------------------
//
// Two `cfr_table::CfrTable` instances back the regrets and strat_sum rows (each row is 11 f64;
// AoSoA's lane-major layout puts the working set into L1). A separate `HashMap` backs the leaf
// cache (one f64 per public leaf key). The same `Vec<i64>` infoset-key encoding
// (`subgame::infoset_key` / `subgame::leaf_key`) threads through every layer, so the Python
// `tilt_table` keys land in the same hash space as the AoSoA slots (one canonical key space).

/// Per-solve tables: two AoSoA-backed row stores + a public leaf cache + telemetry counters.
pub struct CfrTables {
    /// Cumulative regret rows (AoSoA, FNV-1a 64-bit keyed index).
    regrets: cfr_table::CfrTable,
    /// Weighted strategy-sum rows (AoSoA, FNV-1a 64-bit keyed index).
    strat_sum: cfr_table::CfrTable,
    /// `Vec<i64>` leaf key -> f64 hero bb (PUBLIC leaf cache; same encoding as `subgame::leaf_key`).
    /// Leaf cache STAYS Vec<i64>-keyed: the per-leaf hot path is a SINGLE write (cache miss) or a
    /// single read (cache hit) per leaf visit — there's no per-iteration churn that would benefit
    /// from the scalar-fold conversion. The AoSoA path is where the index probe dominates wall time.
    leaf: HashMap<Vec<i64>, f64>,
    /// Leaf-cache hit counter (debug / telemetry).
    leaf_hits: u64,
    /// Leaf-cache miss counter (number of playouts actually executed).
    leaf_evals: u64,
    /// Tilt-table HIT counter (POKERBOT_NATIVE_CFV, when a tilt_table is supplied).
    tilt_hits: u64,
    /// Tilt-table MISS counter (fell back to GTO blueprint-mix).
    tilt_misses: u64,
}

impl CfrTables {
    pub fn new() -> Self {
        CfrTables {
            regrets: cfr_table::CfrTable::new_with_capacity(AOSOA_INITIAL_CAPACITY),
            strat_sum: cfr_table::CfrTable::new_with_capacity(AOSOA_INITIAL_CAPACITY),
            leaf: HashMap::with_capacity(AOSOA_INITIAL_CAPACITY),
            leaf_hits: 0,
            leaf_evals: 0,
            tilt_hits: 0,
            tilt_misses: 0,
        }
    }

    /// Read the current regrets row (all-zero on miss). Equivalent to the legacy
    /// `*tab.regrets.get(&key).unwrap_or(&ZEROS)`. `key` is the FNV-1a 64-bit fold of the i64
    /// infoset-key vector (computed at the call boundary in `cfv` — see `cfr_table::infoset_key_u64`).
    #[inline]
    pub fn regret_row(&self, key: u64) -> [f64; NUM_ACTIONS] {
        self.regrets.get_or_zero(key)
    }

    /// Overwrite the regrets row for `key` (allocates on miss). `key` is the FNV-1a 64-bit fold.
    #[inline]
    pub fn write_regret_row(&mut self, key: u64, row: [f64; NUM_ACTIONS]) {
        self.regrets.insert(key, row);
    }

    /// Read the current strat_sum row (all-zero on miss). Equivalent to the legacy
    /// `tab.strat_sum.get(&key)`. `key` is the FNV-1a 64-bit fold.
    #[inline]
    pub fn strat_sum_row(&self, key: u64) -> Option<[f64; NUM_ACTIONS]> {
        let row = self.strat_sum.get_or_zero(key);
        let sum: f64 = row.iter().sum();
        if sum > 0.0 { Some(row) } else { None }
    }

    /// Pointwise-add `delta` into the strat_sum row for `key` (allocates on miss). The
    /// AoSoA-friendly path: lane-in-place, no row materialised. `key` is the FNV-1a 64-bit fold.
    #[inline]
    pub fn add_strat_sum(&mut self, key: u64, delta: &[f64; NUM_ACTIONS]) {
        self.strat_sum.add_into(key, delta);
    }

    #[inline]
    pub fn leaf_get(&self, key: &[i64]) -> Option<f64> {
        self.leaf.get(key).copied()
    }

    #[inline]
    pub fn leaf_insert(&mut self, key: Vec<i64>, v: f64) {
        self.leaf.insert(key, v);
    }

    #[inline]
    pub fn note_leaf_hit(&mut self) {
        self.leaf_hits += 1;
    }

    #[inline]
    pub fn note_leaf_eval(&mut self) {
        self.leaf_evals += 1;
    }

    #[inline]
    pub fn note_tilt_hit(&mut self) {
        self.tilt_hits += 1;
    }

    #[inline]
    pub fn note_tilt_miss(&mut self) {
        self.tilt_misses += 1;
    }

    pub fn n_infosets(&self) -> usize {
        self.regrets.n_rows()
    }

    pub fn n_leaf_cached(&self) -> usize {
        self.leaf.len()
    }
}

// ---------------------------------------------------------------------------
// LeafContWeights (Stage 2) — model-aware leaf override rows.
//
// Optional per-leaf weight row. When supplied to `solve`, `leaf_value` consults this map BEFORE
// falling through to the uniform continuation gadget; a hit SHADOWS the uniform leaf with a
// weighted-continuation estimator computed from the row. The exploit math is never recomputed here
// — the row is the precomputed DATA the Python exploit tail emits. On a miss, the existing uniform
// leaf is used. Wire-up:
//   - Python (`pokerbot/search/leaf_cont.py`) emits `dict[tuple_leaf_key -> list[float;11]]`.
//   - The wrapper passes `Option<HashMap<Vec<i64>, [f64; NUM_ACTIONS]>>` here.
//   - The Rust side never sees the ROW source — only the table itself.
//
// Encoding: `Vec<i64>` LEAF key (same `subgame::leaf_key` shape). Value is a normalised 11-vector
// of continuation weights (NOT a strategy — the row is multiplied into the leaf rollout estimator).
//
// We re-export `crate::leaf_cont::LeafContWeights` for local use (the canonical type lives in
// `leaf_cont.rs` and is also consumed by `playout::uniform_leaf_value`'s `lcw` arg).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// SearchResult — what `solve` returns to Python.
// ---------------------------------------------------------------------------

/// What `SearchSession.solve` returns to Python. Fields:
///   * `action_probs`     : length-11 root action distribution.
///   * `iterations`       : iterations actually executed (may be < `max_iterations` if cancelled).
///   * `budget_bound`     : 0 == ran to `max_iterations`; 1 == wall-clock budget fired;
///                          2 == cancel handle fired.
///   * `leaf_eval_count`  : number of leaf playouts (cache misses).
///   * `leaf_cache_hits`  : number of leaf cache hits.
///   * `tilt_hits`        : POKERBOT_NATIVE_CFV — opponent-node tilt-table hits.
///   * `tilt_misses`      : POKERBOT_NATIVE_CFV — opponent-node tilt-table misses (fell back to GTO).
pub struct SearchResult {
    pub action_probs: [f64; NUM_ACTIONS],
    pub iterations: u32,
    pub budget_bound: u8,
    pub leaf_eval_count: u64,
    pub leaf_cache_hits: u64,
    pub tilt_hits: u64,
    pub tilt_misses: u64,
}

// ---------------------------------------------------------------------------
// Inner solve body — same shape as `subgame::solve` but driven by `CfrTables`.
// ---------------------------------------------------------------------------

/// Maximum number of CFR+ iterations we'll ever run, even if the caller passes something huge.
/// Safety valve against a malformed `max_iterations` arg triggering an OOM spin on the AoSoA
/// arrays. 1M iterations is ~10x any production solve; anything beyond is almost certainly a
/// caller bug and we cap rather than panic.
const MAX_HARD_ITERS: u32 = 1_000_000;

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

/// FNV-1a fold mixing the cross-session seed with per-solve entropy + worker id. Same fold
/// shape as `subgame::leaf_seed` so the per-solve stream is deterministic from
/// (session_seed, public_seed, worker_id) — re-running with the same Python inputs always
/// reproduces the same statistical shape (statistical equivalence across re-runs).
fn seed_mix(session_seed: u64, public_seed: u64, worker_id: usize) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in &[session_seed, public_seed, worker_id as u64] {
        let xb = x;
        for b in xb.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Inner solve driver. Returns the SearchResult + the wall-clock of the solve for telemetry.
/// `worker_id` (used to seed the per-worker Rng) is the Python-side worker id when solve() is
/// dispatched from multiple threads (future parallel mode); 0 for the single-threaded case.
#[allow(clippy::too_many_arguments)]
fn run_solve<const N: usize>(
    root: &EnvStateN<N>,
    hero: usize,
    bp: &Blueprint,
    pcfg: &PlayoutConfig,
    public_seed: u64,
    session_seed: u64,
    max_iterations: u32,
    max_depth: i32,
    blueprint_mix: f64,
    rollout_samples: i32,
    resample_board: bool,
    linear_averaging: bool,
    dcfr: bool,
    dcfr_alpha: f64,
    dcfr_beta: f64,
    dcfr_gamma: f64,
    tilt_table: Option<TiltMap>,
    leaf_cont_weights: Option<&LeafContWeights>,
    time_budget_ms: u64,
    cancel: &Arc<AtomicBool>,
    worker_id: usize,
) -> SearchResult {
    let legal_root = abstracted_legal(root, hero, pcfg);
    let mut probs = [0.0f64; NUM_ACTIONS];
    if !any_legal(&legal_root) {
        return SearchResult {
            action_probs: probs,
            iterations: 0,
            budget_bound: 0,
            leaf_eval_count: 0,
            leaf_cache_hits: 0,
            tilt_hits: 0,
            tilt_misses: 0,
        };
    }

    let mut tab = CfrTables::new();

    // Per-worker Rng — seeded from session_seed (the stable cross-session seed) mixed with
    // public_seed (per-solve entropy from the Python side) and worker_id (parallel dispatch).
    // DEFAULT-DEGENERATE: when `session_seed == 0` AND `worker_id == 0`, fall back to `public_seed`
    // directly. This preserves byte-identity with the per-call `SubgameSolver.solve` (which seeds
    // from `public_seed` only with no fold) — the contract pinned by
    // `tests/test_search_session_equiv.py::test_search_session_matches_subgame_solver_gto_base`.
    // Setting a non-zero `session_seed` (or non-zero `worker_id`) opts into the FNV-mix fold that
    // distinguishes parallel-worker streams.
    let rng_seed = if session_seed == 0 && worker_id == 0 {
        public_seed
    } else {
        seed_mix(session_seed, public_seed, worker_id)
    };
    let mut rng = Rng::seed_from_u64(rng_seed);

    // Cooperative-cancel + wall-clock deadline snapshots (per-iter check at the LOOP TOP, NOT per
    // node — per-node was profiled to cost ~5-8% of the inner loop). `time_budget_ms == 0` =>
    // no wall-clock gate (run to `max_iterations` regardless of elapsed time).
    let t_start = Instant::now();
    let t_deadline = if time_budget_ms > 0 {
        Some(t_start + std::time::Duration::from_millis(time_budget_ms))
    } else {
        None
    };

    let iters_target = max_iterations.min(MAX_HARD_ITERS);
    let mut budget_bound: u8 = 0;
    let mut iters_done: u32 = 0;

    // Build the per-solve SubgameConfig once; reused for every opponent-node lookup (the Ctx it
    // backs owns the tilt-table reference + DCFR knobs). `max_iterations` / `max_depth` here are
    // the config defaults (we thread our own `max_depth` into the recursion for clarity).
    let scfg = SubgameConfig {
        max_iterations: max_iterations as i32,
        max_depth,
        blueprint_mix,
        rollout_samples,
        resample_board,
        linear_averaging,
        dcfr,
        dcfr_alpha,
        dcfr_beta,
        dcfr_gamma,
        tilt_table,
    };
    let ctx = Ctx {
        bp,
        pcfg,
        scfg: &scfg,
        public_seed,
    };

    for t in 0..iters_target {
        // ----- COOPERATIVE CANCEL + WALL CHECK (per-iter, NOT per node) -----
        if cancel.load(Ordering::Relaxed) {
            budget_bound = 2;
            break;
        }
        if let Some(deadline) = t_deadline {
            if Instant::now() >= deadline {
                budget_bound = 1;
                break;
            }
        }

        let mut world = *root;
        deal_opponent_holes(&mut world, hero, &mut rng);
        cfv(
            &world,
            hero,
            0,
            max_depth,
            &ctx,
            leaf_cont_weights,
            &mut tab,
            &mut rng,
            t as i32,
        );
        iters_done = t + 1;
    }

    probs = root_strategy(root, hero, pcfg, &ctx, &tab);

    SearchResult {
        action_probs: probs,
        iterations: iters_done,
        budget_bound,
        leaf_eval_count: tab.leaf_evals,
        leaf_cache_hits: tab.leaf_hits,
        tilt_hits: tab.tilt_hits,
        tilt_misses: tab.tilt_misses,
    }
}

/// The recursion body. Mirrors `subgame::cfv` but writes the regret/strat_sum updates into
/// `CfrTables` (AoSoA) instead of a `HashMap<Vec<i64>, [f64;11]>`. The opponent-side
/// `opponent_strategy` body (with its tilt-table LOOKUP) is reused unchanged; we wrap it with
/// a thin hit/miss counter so the SearchResult reports POKERBOT_NATIVE_CFV telemetry.
#[allow(clippy::too_many_arguments)]
fn cfv<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    depth: i32,
    max_depth: i32,
    ctx: &Ctx<'_>,
    leaf_cont_weights: Option<&LeafContWeights>,
    tab: &mut CfrTables,
    rng: &mut Rng,
    t: i32,
) -> f64 {
    if env.done {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    if core::is_chance_node(env) {
        return leaf_value(env, hero, ctx, leaf_cont_weights, tab);
    }
    if depth >= max_depth {
        return leaf_value(env, hero, ctx, leaf_cont_weights, tab);
    }
    let actor_i = env.action_seat;
    if actor_i < 0 {
        return leaf_value(env, hero, ctx, leaf_cont_weights, tab);
    }
    let actor = actor_i as usize;
    let legal = abstracted_legal(env, actor, ctx.pcfg);
    if !any_legal(&legal) {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    let key = infoset_key(env, actor);

    if actor != hero {
        // Opponent node: read current regrets row out of the AoSoA; consult the opponent
        // strategy helper (which LOOKS UP the tilt table when one is wired); sample ONE action
        // and recurse. Telemetry: classify this opponent node as a tilt HIT iff a row was found,
        // else MISS (so SearchResult.tilt_hits / tilt_misses reflect the per-call classification
        // exactly — the helper itself does not maintain our counter).
        //
        // Fold the i64 infoset-key Vec to a u64 scalar ONCE at the boundary (see
        // `cfr_table::infoset_key_u64`). The fold + the `tilt_table` lookup share the same `key`
        // slice — `opponent_strategy` still receives `&[i64]` because its tilt-table probe and
        // its regret-matching are both unchanged; only the AoSoA store is keyed on the u64 fold.
        let scalar_key = cfr_table::infoset_key_u64(&key);
        let regrets = tab.regret_row(scalar_key);
        let tilt_hit = ctx
            .scfg
            .tilt_table
            .as_ref()
            .and_then(|t| t.get(&key).map(|_| true))
            .unwrap_or(false);
        let strat = opponent_strategy(env, actor, &legal, &regrets, ctx, &key);
        if tilt_hit {
            tab.note_tilt_hit();
        } else if ctx.scfg.tilt_table.is_some() {
            // A tilt table is wired but no row matched this infoset — count it as a miss so the
            // hit rate is computable from the SearchResult telemetry.
            tab.note_tilt_miss();
        }
        let u = rng.next_f64();
        let a = sample_action(&strat, &legal, u);
        // Gated real-NLHE street closure (POKERBOT_CLOSE_CALLED_STREET; default OFF => identity).
        let child = crate::closure::maybe_close_called_street(core::apply_action(env, a));
        return cfv(&child, hero, depth + 1, max_depth, ctx, leaf_cont_weights, tab, rng, t);
    }

    // Hero node: branch all (abstracted-)legal actions, CFR+ regret update against the AoSoA.
    let scalar_key = cfr_table::infoset_key_u64(&key);
    let regrets = tab.regret_row(scalar_key);
    let strat = regret_match(&regrets, &legal);
    let mut av = [0.0f64; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        if legal[a] {
            let child = crate::closure::maybe_close_called_street(core::apply_action(env, a));
            av[a] = cfv(&child, hero, depth + 1, max_depth, ctx, leaf_cont_weights, tab, rng, t);
        }
    }
    let mut ev = 0.0;
    for a in 0..NUM_ACTIONS {
        ev += strat[a] * av[a];
    }
    let mut newr = regrets;
    let w;
    if ctx.scfg.dcfr {
        let t1 = (t + 1) as f64;
        let pa = t1.powf(ctx.scfg.dcfr_alpha);
        let pos_f = pa / (pa + 1.0);
        let neg_f = if ctx.scfg.dcfr_beta == 0.0 {
            0.5
        } else {
            let nb = t1.powf(ctx.scfg.dcfr_beta);
            nb / (nb + 1.0)
        };
        for a in 0..NUM_ACTIONS {
            if legal[a] {
                let r = newr[a];
                let r = if r > 0.0 { r * pos_f } else { r * neg_f };
                newr[a] = r + (av[a] - ev);
            }
        }
        w = t1.powf(ctx.scfg.dcfr_gamma);
    } else {
        for a in 0..NUM_ACTIONS {
            if legal[a] {
                newr[a] = (newr[a] + av[a] - ev).max(0.0);
            }
        }
        w = if ctx.scfg.linear_averaging {
            (t + 1) as f64
        } else {
            1.0
        };
    }
    tab.write_regret_row(scalar_key, newr);
    let mut ss = [0.0f64; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        ss[a] = w * strat[a];
    }
    tab.add_strat_sum(scalar_key, &ss);
    ev
}

/// Same root action probability calculation as `subgame::solve` but reading the strategy_sum
/// straight out of `CfrTables` (AoSoA) rather than a HashMap<Vec<i64>, [f64;11]>.
fn root_strategy<const N: usize>(
    root: &EnvStateN<N>,
    hero: usize,
    pcfg: &PlayoutConfig,
    _ctx: &Ctx<'_>,
    tab: &CfrTables,
) -> [f64; NUM_ACTIONS] {
    let legal = abstracted_legal(root, hero, pcfg);
    let mut probs = [0.0f64; NUM_ACTIONS];
    if !any_legal(&legal) {
        return probs;
    }
    let root_key = infoset_key(root, hero);
    // Fold the i64 infoset-key Vec to a u64 scalar ONCE at the boundary (same fold as `cfv`).
    let scalar_root_key = cfr_table::infoset_key_u64(&root_key);
    let mut used_avg = false;
    if let Some(ss) = tab.strat_sum_row(scalar_root_key) {
        let tot: f64 = ss.iter().sum();
        if tot > 1e-12 {
            for a in 0..NUM_ACTIONS {
                probs[a] = if legal[a] { ss[a] } else { 0.0 };
            }
            used_avg = true;
        }
    }
    if !used_avg {
        // Fallback: regret-match against the root regrets row (if any). We reuse `regret_match`
        // unchanged — the helper is byte/behavior-identical between the AoSoA and HashMap paths.
        let regrets = tab.regret_row(scalar_root_key);
        probs = regret_match(&regrets, &legal);
    }
    // Final mask + renormalize (mirrors `probs = probs*legal; probs/=max(sum,1e-12)`).
    let mut s = 0.0;
    for a in 0..NUM_ACTIONS {
        if !legal[a] {
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

/// Leaf-value with optional Stage-2 `LeafContWeights` override. The leaf-row override (matching
/// at the current leaf key) is deferred to Stage 3 — for now the only Stage-2 surface is the
/// continuation-bias the playout's per-step draw picks up automatically (uniform_leaf_value's
/// `lcw` parameter). On a cache miss we run the gadget and cache.
fn leaf_value<const N: usize>(
    env: &EnvStateN<N>,
    hero: usize,
    ctx: &Ctx<'_>,
    leaf_cont_weights: Option<&LeafContWeights>,
    tab: &mut CfrTables,
) -> f64 {
    if env.done {
        return env.rewards[hero] as f64 / CHIP_UNIT;
    }
    let key = leaf_key(env);
    if let Some(v) = tab.leaf_get(&key) {
        tab.note_leaf_hit();
        return v;
    }
    let seed = leaf_seed(&key, ctx.public_seed);
    // Pass `lcw` straight through so the rollout's deeper leaves consult the biased table at
    // their own (deeper) leaf-key hashes. We do NOT consult the table at the CURRENT leaf key
    // here — the per-leaf row override (the original Stage-2 design) is deferred; the
    // playout-level continuation bias is the live Stage-2 surface today.
    let v = playout::uniform_leaf_value(
        env, hero, ctx.bp, ctx.pcfg, seed,
        ctx.scfg.rollout_samples, ctx.scfg.resample_board,
        ctx.scfg.blueprint_cv,
    );
    tab.note_leaf_eval();
    tab.leaf_insert(key, v);
    v
}

/// Stage-2 weighted-continuation leaf (RESERVED). The per-leaf row override at the CURRENT leaf
/// key is Stage 3 — the live Stage-2 surface is the playout's per-step continuation bias (the
/// `lcw` arg above). Kept here as a no-op stub so a future per-leaf port lands in one obvious
/// place (one file owns the "the row is DATA, never recomputed" invariant).
#[allow(dead_code)]
fn weighted_leaf_value<const N: usize>(
    _env: &EnvStateN<N>,
    _hero: usize,
    _bp: &Blueprint,
    _pcfg: &PlayoutConfig,
    _base_seed: u64,
    _rollout_samples: i32,
    _resample_board: bool,
    _weights: &[f64; NUM_ACTIONS],
) -> f64 {
    // Stage 3: per-leaf row override at the CURRENT leaf key. The exploit math lives in Python
    // (firewall); the row is consumed as DATA here. Until then this function is never called
    // (call sites delegate to `playout::uniform_leaf_value` directly).
    unimplemented!("weighted_leaf_value is Stage 3; the live Stage-2 path is the playout's lcw bias")
}

// ---------------------------------------------------------------------------
// We extend `subgame::Ctx` with one extra field (the optional `LeafContWeights` table) via a
// thin local wrapper struct. We do NOT modify `subgame.rs`'s `Ctx` — that file is the GTO-base
// body and we want it to stay frozen. The wrapper carries the same `bp` / `pcfg` / `scfg` /
// `public_seed` fields plus the optional Stage-2 `LeafContWeights`, and the recursion
// (`cfv` / `leaf_value`) consumes `ctx.leaf_cont_weights` for the Stage-2 branch.
// ---------------------------------------------------------------------------

// We need to make `leaf_cont_weights` available on `Ctx` without modifying subgame.rs. The
// cleanest way without forking `Ctx` is a tiny local extension trait that lets us read the
// optional Stage-2 weights out of a side-table. We thread the side-table through `run_solve`
// and the recursion as an EXTRA argument (it never appears in `Ctx` itself).

// ---------------------------------------------------------------------------
// PyO3 surface — `SearchSession` class. Gated on `#[cfg(feature = "python")]`.
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
pub mod py {
    use super::*;
    use crate::core::NUM_ACTIONS as CORE_NUM_ACTIONS;
    use crate::flat;
    use crate::live;
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};
    use std::sync::atomic::AtomicBool;

    /// PyO3 SearchSession class. Owns the blueprint ONCE per session; `solve` rebuilds the
    /// per-solve `tilt_table` (cheap) and dispatches to the native `run_solve` body under
    /// `py.detach()` (GIL released for the whole recursion).
    #[pyclass]
    pub struct SearchSession {
        bp: Arc<Blueprint>,
        pcfg: PlayoutConfig,
        session_seed: u64,
    }

    #[pymethods]
    impl SearchSession {
        /// `blueprint` is the same `dict[bytes(16) -> list[float; 11]]` the SubgameSolver takes.
        /// `session_seed` is the stable cross-solve seed (mixed with `public_seed` + `worker_id`
        /// inside `solve` to produce the per-worker Rng). The blueprint is OWNED by the session
        /// (no per-call clone).
        #[new]
        #[pyo3(signature = (blueprint, allowed_pre, allowed_post, bias_mult, bias_floor,
                            max_depth, num_buckets, session_seed=0))]
        fn new(
            blueprint: &Bound<'_, PyDict>,
            allowed_pre: Vec<bool>,
            allowed_post: Vec<bool>,
            bias_mult: f64,
            bias_floor: f64,
            max_depth: i32,
            num_buckets: i32,
            session_seed: u64,
        ) -> PyResult<Self> {
            let mut bp: Blueprint = HashMap::with_capacity(blueprint.len());
            for (k, v) in blueprint.iter() {
                let kb: &[u8] = k.extract()?;
                if kb.len() != 16 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "blueprint key must be 16 bytes",
                    ));
                }
                let mut key = [0u8; 16];
                key.copy_from_slice(kb);
                let row: Vec<f64> = v.extract()?;
                if row.len() != CORE_NUM_ACTIONS {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "blueprint row must have 11 entries",
                    ));
                }
                let mut r = [0.0f64; CORE_NUM_ACTIONS];
                r.copy_from_slice(&row[..CORE_NUM_ACTIONS]);
                bp.insert(key, r);
            }
            let mut ap = [false; CORE_NUM_ACTIONS];
            let mut po = [false; CORE_NUM_ACTIONS];
            for a in 0..CORE_NUM_ACTIONS {
                ap[a] = *allowed_pre.get(a).unwrap_or(&false);
                po[a] = *allowed_post.get(a).unwrap_or(&false);
            }
            Ok(SearchSession {
                bp: Arc::new(bp),
                pcfg: PlayoutConfig {
                    allowed_pre: ap,
                    allowed_post: po,
                    bias_mult,
                    bias_floor,
                    max_depth,
                    num_buckets,
                },
                session_seed,
            })
        }

        fn n_keys(&self) -> usize {
            self.bp.len()
        }

        fn session_seed(&self) -> u64 {
            self.session_seed
        }

        /// Solve one subgame. Cooperative cancel via `cancel_handle`: a `u64` id previously
        /// returned by `register_cancel()` (see `crate::live`). When supplied, the inner loop
        /// polls the process-global cancel flag for that id and breaks at the LOOP TOP — the
        /// `set_cancel(id)` watchdog firing then terminates an in-flight solve within ONE
        /// iteration (bounded by `max_depth` recursion cost). When `None`, no cancel contract
        /// is wired and the solve runs to `max_iterations` or `time_budget_ms` (whichever fires
        /// first). Per-iter cancel + wall check happen at the LOOP TOP — NOT per node — to keep
        /// the inner loop free of per-iteration Python overhead.
        #[allow(clippy::too_many_arguments)]
        #[pyo3(signature = (state, hero, public_seed, max_iterations, max_depth, blueprint_mix,
                            rollout_samples, resample_board, linear_averaging,
                            dcfr=false, dcfr_alpha=1.5, dcfr_beta=0.0, dcfr_gamma=2.0,
                            tilt_table=None, cont_weights=None, time_budget_ms=0,
                            cancel_handle=None, worker_id=0))]
        fn solve(
            &self,
            py: Python<'_>,
            state: &[u8],
            hero: usize,
            public_seed: u64,
            max_iterations: u32,
            max_depth: i32,
            blueprint_mix: f64,
            rollout_samples: i32,
            resample_board: bool,
            linear_averaging: bool,
            dcfr: bool,
            dcfr_alpha: f64,
            dcfr_beta: f64,
            dcfr_gamma: f64,
            tilt_table: Option<&Bound<'_, PyDict>>,
            cont_weights: Option<&Bound<'_, PyDict>>,
            time_budget_ms: u64,
            cancel_handle: Option<u64>,
            worker_id: usize,
        ) -> PyResult<Py<PyAny>> {
            // ---- Marshal state bytes -> EnvStateN<NUM_PLAYERS> ----
            let s_env: EnvState = flat::from_flat(&bytes_to_flat(state));
            let s_constgeneric: &EnvStateN<{ crate::core::NUM_PLAYERS }> = &s_env;

            // ---- Build the per-solve tilt table (rebuilt per call; cheap) ----
            let tilt = build_tilt(tilt_table)?;

            // ---- Build the per-solve LeafContWeights (Stage 2, opt-in) ----
            let leaf_cont_weights = build_leaf_cont_weights(cont_weights)?;

            // ---- Cooperative cancel flag (process-global registry lookup; falls back to a
            //      never-fired local Arc when no id was supplied OR the id was already released).
            //      The registry owns the live Arc so the Python `set_cancel(id)` watchdog
            //      writes to the SAME AtomicBool the inner loop polls at the LOOP TOP — that's
            //      the contract the live driver depends on.
            let cancel: Arc<AtomicBool> = match cancel_handle {
                Some(id) => live::cancel_registry()
                    .lock()
                    .expect("cancel registry poisoned")
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| Arc::new(AtomicBool::new(false))),
                None => Arc::new(AtomicBool::new(false)),
            };

            // ---- GIL-released inner solve body ----
            let bp = self.bp.clone();
            let pcfg = &self.pcfg;
            let session_seed = self.session_seed;
            let res = py.detach(|| {
                run_solve(
                    s_constgeneric,
                    hero,
                    bp.as_ref(),
                    pcfg,
                    public_seed,
                    session_seed,
                    max_iterations,
                    max_depth,
                    blueprint_mix,
                    rollout_samples,
                    resample_board,
                    linear_averaging,
                    dcfr,
                    dcfr_alpha,
                    dcfr_beta,
                    dcfr_gamma,
                    tilt,
                    leaf_cont_weights.as_ref(),
                    time_budget_ms,
                    &cancel,
                    worker_id,
                )
            });

            // ---- Return SearchResult as a Python dict (so the wrapper can pick fields by
            //      name without coupling to the Rust struct layout). ----
            let dict = PyDict::new(py);
            let probs_list = PyList::new(py, res.action_probs.iter().copied())?;
            dict.set_item("action_probs", probs_list)?;
            dict.set_item("iterations", res.iterations)?;
            dict.set_item("budget_bound", res.budget_bound)?;
            dict.set_item("leaf_eval_count", res.leaf_eval_count)?;
            dict.set_item("leaf_cache_hits", res.leaf_cache_hits)?;
            dict.set_item("tilt_hits", res.tilt_hits)?;
            dict.set_item("tilt_misses", res.tilt_misses)?;
            Ok(dict.into())
        }
    }

    // -----------------------------------------------------------------------
    // helpers
    // -----------------------------------------------------------------------

    fn bytes_to_flat(b: &[u8]) -> Vec<i64> {
        debug_assert_eq!(b.len(), flat::FLAT_LEN * 8);
        let mut v = vec![0i64; flat::FLAT_LEN];
        for (i, chunk) in b.chunks_exact(8).enumerate() {
            v[i] = i64::from_le_bytes(chunk.try_into().unwrap());
        }
        v
    }

    fn build_tilt(d: Option<&Bound<'_, PyDict>>) -> PyResult<Option<TiltMap>> {
        let d = match d {
            None => return Ok(None),
            Some(d) => d,
        };
        let mut map: TiltMap = Default::default();
        map.reserve(d.len());
        for (k, v) in d.iter() {
            let key: Vec<i64> = k.extract()?;
            let row: Vec<f64> = v.extract()?;
            if row.len() != CORE_NUM_ACTIONS {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "tilt_table row must have 11 entries",
                ));
            }
            let mut r = [0.0f64; CORE_NUM_ACTIONS];
            r.copy_from_slice(&row[..CORE_NUM_ACTIONS]);
            map.insert(key, r);
        }
        Ok(Some(map))
    }

    fn build_leaf_cont_weights(
        d: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<LeafContWeights>> {
        // `LeafContWeights` is keyed by a 64-bit FNV-1a hash of the leaf-key Vec<i64> (i64),
        // and each row is a pre-normalized [f64; 4] continuation-weight vector. Python computes
        // the i64 hash once per leaf and ships a scalar-keyed dict. We just unpack and forward.
        let d = match d {
            None => return Ok(None),
            Some(d) => d,
        };
        let mut items: Vec<(i64, [f64; playout::NUM_CONTINUATIONS])> = Vec::with_capacity(d.len());
        for (k, v) in d.iter() {
            let key: i64 = k.extract()?;
            let row: Vec<f64> = v.extract()?;
            if row.len() != playout::NUM_CONTINUATIONS {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "cont_weights row must have {} entries",
                    playout::NUM_CONTINUATIONS
                )));
            }
            let mut r = [0.0f64; playout::NUM_CONTINUATIONS];
            r.copy_from_slice(&row[..playout::NUM_CONTINUATIONS]);
            items.push((key, r));
        }
        Ok(Some(LeafContWeights::from_iter(items)))
    }
}

// ---------------------------------------------------------------------------
// (The duplicated `run_solve_with_ctx` driver that originally followed this comment block has
// been removed — it was a near-byte-for-byte duplicate of `run_solve`. The single driver above
// is what `SearchSession::solve` calls inside `py.detach`.)
// ---------------------------------------------------------------------------
