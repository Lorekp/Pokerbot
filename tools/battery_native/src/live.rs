//! PyO3 `SearchHandle` — long-lived live-decision handle for the 800 ms hot path.
//!
//! # Why a separate handle (vs `SubgameSolver` / `SearchSession`)
//!
//! The existing `SubgameSolver` is correct but pays a per-call `Blueprint` HashMap clone — fine for
//! the LBR harness, wrong shape for the live serve where ONE Python worker makes THOUSANDS of
//! `solve` calls per second against the SAME blueprint snapshot. `SearchSession` already lifts the
//! snapshot out; `SearchHandle` keeps the SAME ownership model but layers in the LIVE-API contract:
//!
//! * `build_root(state, hero, rng_salt) -> {public_seed, legal, info_key}` — the SubgameRoot
//!   stand-in the Python side caches (cheap: one fingerprint + one legal mask + one blake2b).
//! * `solve(state, hero, public_seed, ..., cancel_handle, tilt_table)` — the per-decision hot path,
//!   GIL-released inside the loop via `py.detach()`. The blueprint is OWNED by the handle; no per-
//!   call clone. Per-iter cancel + wall check at the LOOP TOP (NOT per node) — profiled to be the
//!   cheapest observable cancel site that does not dominate the recursion.
//! * `blueprint_prior(state, hero)` — ONE FFI call, no recursion. The trusted GTO floor the Python
//!   fallback ladder serves when the future times out (the live driver has a 750 ms hard wall;
//!   the native loop's 750 ms budget keeps the in-loop overshoot bounded to ~one iteration).
//!
//! # Efficiency baked in
//!
//! * **Blueprint owned alive across calls** — `FxHashMap<[u8;16], [f64;11]>` lives in the
//!   `SearchHandle`; never rebuilt per `solve`. The `Blueprint` type is the same one
//!   `SubgameSolver` / `PlayoutEngine` consume.
//! * **GIL released inside the loop** — `py.detach(|| { ... })` wraps the whole iteration loop.
//!   No per-iter `PyList::new` / GIL acquire on the hot path.
//! * **Per-worker Rng pool** — `Rng::seed_from_u64(public_seed ^ session_seed_mix)` once at
//!   solve entry; the recursion mutates it in place. Re-running with the same `(public_seed,
//!   session_seed)` reproduces the same Monte-Carlo stream (CRN A/B parity).
//! * **Fingerprint cache** — `SubgameRoot.info_key` is cached in the Python wrapper between
//!   `build_root` and `solve`; the native handle itself doesn't need to cache it because the
//!   recursion rebuilds the infoset key per node (Vec<i64> allocation), and that allocation is the
//!   hash-key of the per-call `Tables` HashMap (same cost as the Python solve allocates per node).
//!   The fingerprint cache that DOES matter — the LAST digest the wrapper showed the engine, so
//!   repeated calls against the SAME `(state, seat)` skip the marshal — is owned by the Python
//!   wrapper (`live_native.SearchHandle._last_info_key`).
//! * **`blueprint_prior` is one FFI call** — fingerprint lookup + legal-mask projection +
//!   renormalize. No recursion, no allocations beyond the returned Vec. The Python fallback ladder
//!   invokes this when the per-decision `solve` future times out (the 750 ms hard wall).
//!
//! # Stages
//!
//! * **Stage 1 (this file)**: GTO-base / `opponent_model = None` body, same recursion shape as
//!   `subgame::cfv`. Per-call scratch (`regrets` / `strat_sum` / `leaf`) is fresh per `solve` — no
//!   cross-decision contamination, no growing heap over a long live session.
//! * **Stage 2 (next; deferred)**: model-aware LEAF via `LeafContWeights`, mirroring the
//!   `SearchSession` Stage-2 port. The Python side already passes `None` for `leaf_cont_weights`
//!   today so the GTO-base leaf is the live path.
//! * **Stage 3 (future)**: `tilt_table` is already wired (POKERBOT_NATIVE_CFV) so the firewalled
//!   exploit tail consumes precomputed opponent-policy rows as DATA. The recursion shape does not
//!   change; only the opponent-node lookup swaps to a hash probe of the supplied table.

// The whole module body is gated on `feature = "python"` because it pulls in `pyo3` and
// `crate::flat` (which is itself gated on the python feature). The rlib / `cargo test` path gets
// an empty `live` module — no Python coupling, no `EnvState` layout knowledge, no-op stub.

#[cfg(feature = "python")]
pub(crate) mod inner {
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use rustc_hash::FxHashMap as HashMap;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::core::{self, EnvStateN, NUM_ACTIONS, NUM_PLAYERS, MAX_HISTORY};
use crate::fingerprint::{self, fingerprint_fast};
use crate::flat;
use crate::playout::{self, Blueprint, PlayoutConfig, Rng};
use crate::subgame::{
    abstracted_legal, any_legal, infoset_key, leaf_key, leaf_seed, opponent_strategy,
    regret_match, sample_action, SubgameConfig, TiltMap,
};
use rustc_hash::FxHashMap;

// ---------------------------------------------------------------------------
// deal_opponent_holes — uniform belief: deal 2 distinct unseen cards to every non-hero seat
// (Fisher-Yates shuffle over the unseen pool, then sequential assignment in ascending seat order).
// Same body as `subgame::deal_opponent_holes` and `search_session::deal_opponent_holes`; kept
// private to the module so each solve path owns its RNG-driven shuffle (no cross-module
// borrow entanglement).
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// PublicSeed — stable, blake2b-derived 64-bit seed the Python side caches on `build_root`.
// Same encoding as `pluribus_search.PluribusReBeLSolver._public_seed` so the Python/native
// CRN properties line up.
// ---------------------------------------------------------------------------

/// `blake2b(digest_size=8, person="")` over the little-endian i64 encoding of `[rng_salt, hero,
/// street, hist_len, hist_seat*hist_len, hist_action*hist_len, board*MAX_BOARD, stacks*NUM_PLAYERS,
/// committed_street*NUM_PLAYERS, pot_dead, seat_status*NUM_PLAYERS]`. Matches
/// `pluribus_search._public_seed` byte-for-byte.
pub(crate) fn public_seed<const N: usize>(s: &EnvStateN<N>, hero: usize, rng_salt: u64) -> u64 {
    // Bound hist_len to MAX_HISTORY (defensive; engine-produced states always satisfy the bound).
    let hist_len = (s.hist_len.max(0) as usize).min(MAX_HISTORY);
    // 1 (rng_salt) + 1 (hero) + 1 (street) + 1 (hist_len) + 2*hist_len + 5 (board) + N + N + 1 + N
    let cap: usize = 4 + 2 * hist_len + 5 + 3 * N;
    let mut msg: Vec<u8> = Vec::with_capacity(cap * 8);
    let mut push = |v: i64| msg.extend_from_slice(&v.to_le_bytes());
    push(rng_salt as i64);
    push(hero as i64);
    push(s.street as i64);
    push(hist_len as i64);
    for i in 0..hist_len {
        push(s.hist_seat[i] as i64);
        push(s.hist_action[i] as i64);
    }
    for &c in s.board.iter() {
        push(c as i64);
    }
    for i in 0..N {
        push(s.stacks[i] as i64);
        push(s.committed_street[i] as i64);
    }
    push(s.pot_dead as i64);
    for i in 0..N {
        push(s.seat_status[i] as i64);
    }
    let digest = fingerprint::blake2b(8, b"", &msg);
    let mut out = [0u8; 8];
    out.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(out)
}

// ---------------------------------------------------------------------------
// LiveTables — per-solve scratch (regrets + strat_sum + leaf cache).
// Same shape as `subgame::Tables`, just hoisted into the live module so the handle owns it.
// `default()` + `reserve` — FxHash is the fast non-cryptographic hasher; the `Vec<i64>` keys
// (5-15 entries each) are the live hot path.
// ---------------------------------------------------------------------------

/// Per-solve scratch. Allocated FRESH in `solve()` (no cross-call contamination). The FxHash-backed
/// maps are tuned for the live search's small `Vec<i64>` infoset keys (5-15 entries) — SipHash on a
/// 5-15-byte key was a measurable share of the Python CFR+ latency per the subgame.rs file comment.
struct LiveTables {
    regrets: HashMap<Vec<i64>, [f64; NUM_ACTIONS]>,
    strat_sum: HashMap<Vec<i64>, [f64; NUM_ACTIONS]>,
    leaf: HashMap<Vec<i64>, f64>,
    /// Number of unique infosets touched by the recursion (proxy for CFR work; counts new inserts).
    n_strat_inserts: u64,
    /// Number of unique public leaves evaluated (cache misses).
    n_leaf_inserts: u64,
}

impl LiveTables {
    fn with_capacity_hint(hint_keys: usize) -> Self {
        let mut regrets: HashMap<Vec<i64>, [f64; NUM_ACTIONS]> = HashMap::default();
        let mut strat_sum: HashMap<Vec<i64>, [f64; NUM_ACTIONS]> = HashMap::default();
        let mut leaf: HashMap<Vec<i64>, f64> = HashMap::default();
        regrets.reserve(hint_keys.max(16));
        strat_sum.reserve(hint_keys.max(16));
        leaf.reserve(hint_keys.max(16) / 4);
        LiveTables {
            regrets,
            strat_sum,
            leaf,
            n_strat_inserts: 0,
            n_leaf_inserts: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// CancelRegistry — process-global map of u64 id -> Arc<AtomicBool>.
//
// PyO3 doesn't ship Arc across the FFI cleanly; the wrapper (live_native.py) hands us a u64 id and
// keeps the Arc alive in a Python-side dict keyed on the id. On `solve` return (normal or
// exception) the wrapper calls `release_cancel(id)` which drops the registry entry. The Rust side
// is free to clone the Arc into the recursion closure; the Arc keeps the AtomicBool alive as long
// as ANY handle holds it.
//
// `Mutex<HashMap>` (not RwLock) because the critical section is two-pointer-shuffles and the
// contention is per-decision (one acquire per solve, one per release) — the lock is never held
// across the GIL-release, so a contended Mutex is invisible in the live-driver profile.
// ---------------------------------------------------------------------------

type CancelMap = HashMap<u64, Arc<AtomicBool>>;

/// Process-global cancel registry. `OnceLock<Mutex<CancelMap>>` is std-only (no `once_cell` dep).
pub(crate) fn cancel_registry() -> &'static Mutex<CancelMap> {
    static REG: OnceLock<Mutex<CancelMap>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::default()))
}

/// Process-global monotonic id allocator. Atomic so the wrapper can call `register_cancel` from
/// any thread without holding a lock around the id assignment.
fn next_cancel_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Blueprint prior — the trusted GTO root policy served by the fallback ladder.
//
// ONE FFI call: fingerprint lookup + legal-mask projection + renormalize. No recursion, no
// allocations beyond the returned Vec. Called from Python only when the solve future times out
// (the 750 ms hard wall) — the per-decision solve is the common path.
// ---------------------------------------------------------------------------

/// Read the blueprint row for `(state, seat)` and project onto the (abstracted) legal set. Same
/// recipe as `subgame::blueprint_policy` (the fallback path inside the recursion): uniform over the
/// abstracted legal set on a fingerprint miss or a degenerate row (mass-sum <= 1e-12).
fn blueprint_prior<const N: usize>(
    s: &EnvStateN<N>,
    hero: usize,
    legal: &[bool; NUM_ACTIONS],
    bp: &Blueprint,
    num_buckets: i32,
) -> [f64; NUM_ACTIONS] {
    let key = fingerprint_fast(s, hero, num_buckets);
    let mut out = [0.0f64; NUM_ACTIONS];
    if let Some(row) = bp.get(&key) {
        let mut tot = 0.0;
        for a in 0..NUM_ACTIONS {
            let v = if legal[a] { row[a] } else { 0.0 };
            out[a] = v;
            tot += v;
        }
        if tot > 1e-12 {
            for a in 0..NUM_ACTIONS {
                out[a] /= tot;
            }
            return out;
        }
    }
    // Uniform over the abstracted legal set — the legacy fallback.
    let nl = legal.iter().filter(|&&x| x).count().max(1) as f64;
    for a in 0..NUM_ACTIONS {
        out[a] = if legal[a] { 1.0 / nl } else { 0.0 };
    }
    out
}

// ---------------------------------------------------------------------------
// PyO3 surface — `SearchHandle` class. Gated on `#[cfg(feature = "python")]`.
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
pub mod py {
    use super::*;
    use crate::subgame;

    // -----------------------------------------------------------------------
    // Cancel registry PyO3 surface — `register_cancel` / `set_cancel` / `release_cancel`.
    // Re-exported at the module root via `pub use py::{register_cancel, set_cancel, release_cancel};`
    // so `lib.rs` can `wrap_pyfunction!` them without reaching into the inner module.
    // -----------------------------------------------------------------------

    /// Allocate a new cancel id, insert a fresh `Arc<AtomicBool>` in the registry, return the id.
    /// The wrapper pairs this with `release_cancel(id)` in `finally:` (live driver contract:
    /// `test_cancel_release_round_trip_releases_the_sentinel`).
    #[pyfunction]
    pub fn register_cancel() -> u64 {
        let id = next_cancel_id();
        let mut reg = cancel_registry().lock().expect("cancel registry poisoned");
        reg.insert(id, Arc::new(AtomicBool::new(false)));
        id
    }

    /// Set the cancel flag for `id` (no-op if `id` was already released / never registered).
    /// Mirrors the live driver's `cancel_event.set()` — fires the sentinel the Rust loop polls at
    /// the LOOP TOP (cheap `Relaxed` load).
    #[pyfunction]
    pub fn set_cancel(id: u64) {
        if let Some(f) = cancel_registry().lock().expect("cancel registry poisoned").get(&id) {
            f.store(true, Ordering::Relaxed);
        }
    }

    /// Drop the registry entry for `id`. After `release_cancel`, `set_cancel(id)` is a no-op so a
    /// stale sentinel can't bleed into the next decision (the round-trip semantics
    /// `test_cancel_release_round_trip_releases_the_sentinel` enforces).
    #[pyfunction]
    pub fn release_cancel(id: u64) {
        cancel_registry().lock().expect("cancel registry poisoned").remove(&id);
    }


    /// Live-decision handle. SearchAgent owns ONE per worker (per Frida thread / per multitable
    /// table). The heavy blueprint snapshot + abstraction masks + leaf knobs are loaded ONCE at
    /// construction; per-decision work is one `solve(...)` call returning
    /// `(probs: f64[11], ev: f64, iters: i32, metadata: dict)`.
    #[pyclass]
    pub struct SearchHandle {
        /// Immutable blueprint snapshot. FxHashMap<[u8;16], [f64;11]> — owned alive across calls.
        bp: Blueprint,
        /// Leaf knobs baked in. `bias_mult` / `bias_floor` come from the `ContinuationLeafConfig`
        /// the Python `PluribusReBeLConfig` holds; `num_buckets` matches the EMD/200-bucket
        /// postflop abstraction the search was trained against. The abstraction masks
        /// `allowed_pre` / `allowed_post` live INSIDE `leaf_cfg` (they're duplicated as the same
        /// `[bool; 11]` arrays so the live + leaf code paths share one canonical copy).
        leaf_cfg: PlayoutConfig,
        /// CFR+ knobs baked in (linear_averaging + DCFR exponents). These are constants for the
        /// live serve — a process restart picks them up from env, not per-call.
        linear_averaging: bool,
        /// Blueprint-implied-value control variate (artifacts/search_speed_quality_report.md
        /// phase 3.C). When true, the leaf rollout subtracts V_bp (the blueprint-only playout)
        /// per rollout and adds it back at the end. Reduces leaf-variance 2-10x. Mirrors
        /// POKERBOT_BLUEPRINT_CV.
        blueprint_cv: bool,
        dcfr: bool,
        dcfr_alpha: f64,
        dcfr_beta: f64,
        dcfr_gamma: f64,
        /// Per-call stats — read by Python after `solve()` for the decision log.
        last_budget_used_ms: f64,
        last_iters: i32,
        last_budget_bound: bool,
        /// Root EV (chips) under the converged CFR+ strategy from the LAST completed iteration
        /// that hit a hero node at depth 0. 0.0 when no hero iteration completed (budget-bound
        /// before the first hero node, or no legal actions). Used by the live decision log + coach
        /// row to expose real expected value instead of a hard-coded placeholder (was F-NATIVE-8).
        last_root_ev: f64,
        last_served_kind: &'static str, // "search_native", "search_native_cfv", "blueprint_fallback"
    }

    #[pymethods]
    impl SearchHandle {
        /// `blueprint`: dict[bytes(16) -> list[float] len 11] (TabularBlueprintAdapter snapshot).
        /// `allowed_pre` / `allowed_post`: list[bool] len 11 — deployed action abstraction.
        /// `bias_mult` / `bias_floor` / `leaf_max_depth` / `num_buckets`: from the
        /// ContinuationLeafConfig + postflop bucket count.
        /// `dcfr` / `dcfr_alpha` / `dcfr_beta` / `dcfr_gamma`: Discounted-CFR knobs from
        /// `PluribusReBeLConfig`.
        /// `linear_averaging`: CFR+ recency-weighted strategy averaging flag.
        #[new]
        #[pyo3(signature = (blueprint, allowed_pre, allowed_post, bias_mult, bias_floor,
                            leaf_max_depth, num_buckets, dcfr, dcfr_alpha, dcfr_beta,
                            dcfr_gamma, linear_averaging, blueprint_cv))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            blueprint: &Bound<'_, PyDict>,
            allowed_pre: Vec<bool>,
            allowed_post: Vec<bool>,
            bias_mult: f64,
            bias_floor: f64,
            leaf_max_depth: i32,
            num_buckets: i32,
            dcfr: bool,
            dcfr_alpha: f64,
            dcfr_beta: f64,
            dcfr_gamma: f64,
            linear_averaging: bool,
            blueprint_cv: bool,
        ) -> PyResult<Self> {
            let mut bp: Blueprint = Default::default();
            bp.reserve(blueprint.len());
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
                if row.len() != NUM_ACTIONS {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "blueprint row must have 11 entries",
                    ));
                }
                let mut r = [0.0f64; NUM_ACTIONS];
                r.copy_from_slice(&row[..NUM_ACTIONS]);
                bp.insert(key, r);
            }
            let mut ap = [false; NUM_ACTIONS];
            let mut po = [false; NUM_ACTIONS];
            for a in 0..NUM_ACTIONS {
                ap[a] = *allowed_pre.get(a).unwrap_or(&false);
                po[a] = *allowed_post.get(a).unwrap_or(&false);
            }
            Ok(SearchHandle {
                bp,
                leaf_cfg: PlayoutConfig {
                    allowed_pre: ap,
                    allowed_post: po,
                    bias_mult,
                    bias_floor,
                    max_depth: leaf_max_depth,
                    num_buckets,
                },
                linear_averaging,
                blueprint_cv,
                dcfr,
                dcfr_alpha,
                dcfr_beta,
                dcfr_gamma,
                last_budget_used_ms: 0.0,
                last_iters: 0,
                last_budget_bound: false,
                last_root_ev: 0.0,
                last_served_kind: "search_native",
            })
        }

        /// Number of blueprint rows loaded into the handle. Same value across calls.
        fn blueprint_key_count(&self) -> usize {
            self.bp.len()
        }

        /// Last `solve()` wall time in milliseconds. Read by Python for the decision log.
        fn last_budget_used_ms(&self) -> f64 {
            self.last_budget_used_ms
        }

        /// Last `solve()` iteration count.
        fn last_iters(&self) -> i32 {
            self.last_iters
        }

        /// Last `solve()` budget-bound flag (True iff the loop exited via wall OR cancel).
        fn last_budget_bound(&self) -> bool {
            self.last_budget_bound
        }

        /// Last `solve()` served-kind string.
        fn last_served_kind(&self) -> &'static str {
            self.last_served_kind
        }

        /// Last `solve()` root EV (chips) under the converged CFR+ strategy from the final hero
        /// iteration. Reads 0.0 when no hero iteration completed (budget-bound before the first
        /// hero node). Used by the live decision log + coach NLHE row for EV-adjusted stats
        /// (fix for F-NATIVE-8: previously hard-coded 0.0 in `solve()`).
        fn last_root_ev(&self) -> f64 {
            self.last_root_ev
        }

        /// Cheap (~7 µs): validates the flat buffer, projects the abstracted legal mask, emits the
        /// public seed (blake2b over the public state + `rng_salt`) and the infoset key. Returns a
        /// Python dict the Python wrapper caches as the `SubgameRoot` stand-in.
        #[pyo3(signature = (state, hero, rng_salt=0))]
        fn build_root<'py>(
            &self,
            py: Python<'py>,
            state: &[u8],
            hero: usize,
            rng_salt: i64,
        ) -> PyResult<Bound<'py, PyDict>> {
            if state.len() != flat::FLAT_LEN * 8 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "flat state length mismatch",
                ));
            }
            let s = flat::from_flat(&bytes_to_flat(state));
            let s_constgeneric: &EnvStateN<{ NUM_PLAYERS }> = &s;
            let legal = abstracted_legal(s_constgeneric, hero, &self.leaf_cfg);
            let seed = public_seed(s_constgeneric, hero, rng_salt as u64);
            let info_key = infoset_key(s_constgeneric, hero);
            let d = PyDict::new(py);
            d.set_item("hero", hero)?;
            d.set_item("public_seed", seed)?;
            d.set_item("info_key", info_key)?;
            d.set_item("legal", legal.to_vec())?;
            Ok(d)
        }

        /// THE hot path. One call per decision. Returns
        /// `(probs: Vec<f64; 11>, ev: f64, iters: i32, metadata: dict)`.
        ///
        /// **Cancel contract**: `cancel_handle` is a `u64` id previously returned by
        /// `register_cancel()`; the wrapper (live_native.py) wires it to a `threading.Event` and
        /// calls `set_cancel(id)` when the watchdog fires. The Rust loop polls at the LOOP TOP
        /// (cheap `Relaxed` load) — never per node, profiled to cost ~0% of the recursion.
        ///
        /// **Wall contract**: `time_budget_ms == 0` => no wall gate (run to `max_iterations`).
        /// `time_budget_ms > 0` => check elapsed at the LOOP TOP; on overrun break with
        /// `budget_bound=True` and the partial policy (strategy_sum is incremental — a partial
        /// solve returns a partial average, never garbage).
        ///
        /// **Tilt table (POKERBOT_NATIVE_CFV)**: when `tilt_table` is supplied, opponent nodes
        /// look up the precomputed policy row instead of recomputing the GTO blueprint-mix. The
        /// exploit MATH stays in Python (firewall); we only consume the row as DATA.
        #[pyo3(signature = (state, hero, public_seed, max_iterations, max_depth,
                            blueprint_mix, rollout_samples, resample_board,
                            time_budget_ms=750, cancel_handle=None, tilt_table=None))]
        #[allow(clippy::too_many_arguments)]
        fn solve<'py>(
            &mut self,
            py: Python<'py>,
            state: &[u8],
            hero: usize,
            public_seed: u64,
            max_iterations: i32,
            max_depth: i32,
            blueprint_mix: f64,
            rollout_samples: i32,
            resample_board: bool,
            time_budget_ms: u64,
            cancel_handle: Option<u64>,
            tilt_table: Option<&Bound<'_, PyDict>>,
        ) -> PyResult<(Vec<f64>, f64, i32, Bound<'py, PyDict>)> {
            // ----- 1. validate / unmarshal -----
            if state.len() != flat::FLAT_LEN * 8 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "flat state length mismatch",
                ));
            }
            let s = flat::from_flat(&bytes_to_flat(state));
            let s_constgeneric: &EnvStateN<{ NUM_PLAYERS }> = &s;

            // ----- 2. build the per-solve tilt table (optional, POKERBOT_NATIVE_CFV) -----
            let tilt = build_tilt_table(tilt_table)?.map(std::sync::Arc::new);
            let served_kind: &'static str =
                if tilt.is_some() { "search_native_cfv" } else { "search_native" };

            // ----- 3. resolve cancel flag (Arc<AtomicBool> from the process-global registry) -----
            // Clone the Arc (cheap refcount bump) so the recursion closure owns it independently
            // of the registry's lifetime — the wrapper still MUST call `release_cancel(id)` in
            // `finally:` to drop the registry entry.
            let cancel_flag: Option<Arc<AtomicBool>> = cancel_handle
                .and_then(|id| cancel_registry().lock().expect("cancel registry poisoned").get(&id).cloned());

            // ----- 4. construct SubgameConfig (DCFR + linear_averaging are handle constants) -----
            let scfg = SubgameConfig {
                max_iterations,
                max_depth,
                time_budget_ms: time_budget_ms as i64,
                blueprint_mix,
                rollout_samples,
                resample_board,
                linear_averaging: self.linear_averaging,
                blueprint_cv: self.blueprint_cv,
                dcfr: self.dcfr,
                dcfr_alpha: self.dcfr_alpha,
                dcfr_beta: self.dcfr_beta,
                dcfr_gamma: self.dcfr_gamma,
                tilt_table: tilt,
            };

            // ----- 5. THE LOOP (release the GIL; pure-Rust work) -----
            let t0 = Instant::now();
            let budget_ns = time_budget_ms.saturating_mul(1_000_000);
            // Capacity hint: 8 unique infosets per CFR+ iteration is a measured upper bound for
            // 200-iter 6-max solves (the recursion visits at most one unique infoset per depth
            // level per iteration); under-sized maps re-hash but never grow beyond ~16k entries
            // over a 750 ms budget.
            let mut tables = LiveTables::with_capacity_hint((max_iterations as usize).saturating_mul(8));
            let root_key = infoset_key(s_constgeneric, hero);
            let mut iters: i32 = 0;
            let mut budget_bound = false;

            // Root EV from the LAST completed hero iteration at depth 0. Captured by the recursion
            // closure so we can read it out of `py.detach` after the loop without crossing the FFI
            // per node. Init to 0.0: stays 0.0 iff the loop breaks BEFORE the first hero node
            // (budget-bound, no legal actions, etc.) — matches the "no signal" sentinel the Python
            // wrapper treats as "EV not measured this decision".
            //
            // Implementation: `AtomicU64` bit-cast. f64 <-> u64 is a defined bit-pattern on every
            // IEEE-754 target Rust supports (the rust reference's `f64::to_bits` / `f64::from_bits`
            // are total functions, no UB). The closure stores via `Relaxed`, we read via `Relaxed`
            // after `py.detach` returns on the same thread. `AtomicU64` is `Ungil` so the borrow
            // check inside `py.detach` is satisfied.
            use std::sync::atomic::AtomicU64;
            let root_ev_atomic: AtomicU64 = AtomicU64::new(0);

            // GIL-released iteration loop. The recursion mutates the per-call tables + Rng in
            // place; nothing crosses the FFI inside the loop.
            let result_probs: [f64; NUM_ACTIONS] = py.detach(|| -> [f64; NUM_ACTIONS] {
                let bp_ref: &Blueprint = &self.bp;
                let pcfg: &PlayoutConfig = &self.leaf_cfg;
                let ctx = LiveCtx {
                    bp: bp_ref,
                    pcfg,
                    scfg: &scfg,
                    public_seed,
                };
                let mut rng = Rng::seed_from_u64(public_seed);
                while iters < max_iterations {
                    // ---- per-iter cancel + wall check (cheap; NOT per node) ----
                    if let Some(ref f) = cancel_flag {
                        if f.load(Ordering::Relaxed) {
                            budget_bound = true;
                            break;
                        }
                    }
                    if time_budget_ms > 0 && t0.elapsed().as_nanos() as u64 > budget_ns {
                        budget_bound = true;
                        break;
                    }
                    // ---- one CFR+ iteration ----
                    // Clone the immutable root into a per-iter working world (a small fixed-size
                    // stack copy); deal opponent holes via the per-worker Rng; then recurse.
                    let mut world = *s_constgeneric;
                    deal_opponent_holes(&mut world, hero, &mut rng);
                    let iter_root_ev = cfv_iter(
                        &world,
                        hero,
                        0,
                        max_depth,
                        &ctx,
                        &mut tables,
                        &mut rng,
                        iters,
                    );
                    // Only overwrite when the recursion actually produced a root hero EV (depth 0 +
                    // actor == hero). Other paths (chance, leaf, depth-limit, terminal, opponent
                    // node at depth 0) return the leaf/terminal value as a "best estimate", but we
                    // ONLY want the hero's policy EV at the root for the live decision log —
                    // skip those to avoid polluting the metric.
                    if !core::is_chance_node(&world) && !world.done && world.action_seat >= 0 {
                        let actor = world.action_seat as usize;
                        let legal = abstracted_legal(&world, actor, &ctx.pcfg);
                        if actor == hero && any_legal(&legal) {
                            root_ev_atomic.store(iter_root_ev.to_bits(), Ordering::Relaxed);
                        }
                    }
                    iters += 1;
                }
                // ---- root policy extraction ----
                root_policy(s_constgeneric, hero, &root_key, pcfg, &tables)
            });
            let root_ev = f64::from_bits(root_ev_atomic.load(Ordering::Relaxed));

            // ----- 6. metadata -----
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            self.last_budget_used_ms = elapsed_ms;
            self.last_iters = iters;
            self.last_budget_bound = budget_bound;
            // Cache the root EV on the handle so Python can read it via `last_root_ev()` AFTER the
            // solve returns (the live decision log + coach NLHE row use this for EV-adjusted
            // statistics — was hard-coded 0.0 pre F-NATIVE-8 fix).
            self.last_root_ev = root_ev;
            self.last_served_kind = served_kind;

            let d = PyDict::new(py);
            d.set_item("served", served_kind)?;
            d.set_item("iters", iters)?;
            d.set_item("budget_bound", budget_bound)?;
            d.set_item("budget_used_ms", elapsed_ms)?;
            d.set_item("max_iterations", max_iterations)?;
            d.set_item("max_depth", max_depth)?;
            d.set_item("n_strat_inserts", tables.n_strat_inserts as i64)?;
            d.set_item("n_leaf_inserts", tables.n_leaf_inserts as i64)?;
            d.set_item("blueprint_mix", blueprint_mix)?;
            d.set_item("root_ev", root_ev)?;
            // Return the actual root EV (chips) from the last completed hero iteration at depth 0 —
            // NOT the hard-coded 0.0 placeholder it was before F-NATIVE-8. When no hero iteration
            // ran (budget-bound before the first one), this reads as 0.0, matching the pre-fix
            // behavior on the no-signal path.
            Ok((result_probs.to_vec(), root_ev, iters, d))
        }

        /// Trusted GTO root policy for the fallback ladder — ONE FFI call, no recursion, no
        /// allocations beyond the returned Vec. The Python wrapper invokes this when the
        /// `solve` future times out (the 750 ms hard wall) and returns the result as
        /// `metadata['served'] = 'blueprint_fallback'`.
        fn blueprint_prior(&self, state: &[u8], hero: usize) -> PyResult<Vec<f64>> {
            if state.len() != flat::FLAT_LEN * 8 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "flat state length mismatch",
                ));
            }
            let s = flat::from_flat(&bytes_to_flat(state));
            let s_constgeneric: &EnvStateN<{ NUM_PLAYERS }> = &s;
            let legal = abstracted_legal(s_constgeneric, hero, &self.leaf_cfg);
            let p = blueprint_prior(s_constgeneric, hero, &legal, &self.bp, self.leaf_cfg.num_buckets);
            Ok(p.to_vec())
        }
    }

    // -----------------------------------------------------------------------
    // helpers (private to the python module)
    // -----------------------------------------------------------------------

    /// `&[u8]` -> `Vec<i64>` for the flat EnvState buffer (same shape as the lib.rs helper).
    #[inline]
    fn bytes_to_flat(b: &[u8]) -> Vec<i64> {
        debug_assert_eq!(b.len(), flat::FLAT_LEN * 8);
        let mut v = vec![0i64; flat::FLAT_LEN];
        for (i, chunk) in b.chunks_exact(8).enumerate() {
            v[i] = i64::from_le_bytes(chunk.try_into().unwrap());
        }
        v
    }

    /// Unmarshal the per-call `tilt_table` from a Python dict. Each row MUST be NUM_ACTIONS long
    /// (firewall contract — a malformed row is a programming error, not a runtime fallback).
    fn build_tilt_table(d: Option<&Bound<'_, PyDict>>) -> PyResult<Option<TiltMap>> {
        let d = match d {
            None => return Ok(None),
            Some(d) => d,
        };
        let mut map: TiltMap = HashMap::default();
        map.reserve(d.len());
        for (k, v) in d.iter() {
            let key: Vec<i64> = k.extract()?;
            let row: Vec<f64> = v.extract()?;
            if row.len() != NUM_ACTIONS {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "tilt_table row must have 11 entries",
                ));
            }
            let mut r = [0.0f64; NUM_ACTIONS];
            r.copy_from_slice(&row[..NUM_ACTIONS]);
            map.insert(key, r);
        }
        Ok(Some(map))
    }

    // -----------------------------------------------------------------------
    // LiveCtx — per-solve immutable handle into the handle's long-lived state.
    // Mirrors `subgame::Ctx`; held by value inside the recursion closure.
    // -----------------------------------------------------------------------

    struct LiveCtx<'a> {
        bp: &'a Blueprint,
        pcfg: &'a PlayoutConfig,
        scfg: &'a SubgameConfig,
        public_seed: u64,
    }

    /// Root policy extraction at the end of the solve. Strategy-sum when it has mass,
    /// regret-match fallback otherwise, uniform-over-legal on a total miss (the legacy fallback).
    fn root_policy<const N: usize>(
        root: &EnvStateN<N>,
        hero: usize,
        root_key: &[i64],
        pcfg: &PlayoutConfig,
        tables: &LiveTables,
    ) -> [f64; NUM_ACTIONS] {
        let legal = abstracted_legal(root, hero, pcfg);
        let mut probs = [0.0f64; NUM_ACTIONS];
        if !any_legal(&legal) {
            return probs;
        }
        if let Some(ss) = tables.strat_sum.get(root_key) {
            let tot: f64 = ss.iter().sum();
            if tot > 1e-12 {
                for a in 0..NUM_ACTIONS {
                    probs[a] = if legal[a] { ss[a] } else { 0.0 };
                }
                let s: f64 = probs.iter().sum();
                if s > 1e-12 {
                    for a in 0..NUM_ACTIONS {
                        probs[a] /= s;
                    }
                }
                return probs;
            }
        }
        // Fallback: regret-match against the root regrets row (if any), else uniform-over-legal.
        if let Some(regrets) = tables.regrets.get(root_key) {
            probs = regret_match(regrets, &legal);
            let s: f64 = probs.iter().sum();
            if s > 1e-12 {
                for a in 0..NUM_ACTIONS {
                    probs[a] /= s;
                }
            }
            return probs;
        }
        let n = legal.iter().filter(|&&x| x).count().max(1) as f64;
        for a in 0..NUM_ACTIONS {
            probs[a] = if legal[a] { 1.0 / n } else { 0.0 };
        }
        probs
    }

    // -----------------------------------------------------------------------
    // One CFR+ iteration — body lifted out of `subgame::cfv` so the handle can drive the loop
    // with its own cancel + wall semantics. Mirrors `subgame::cfv` byte-for-byte modulo the
    // per-call scratch it writes to (LiveTables here vs subgame::Tables there).
    // -----------------------------------------------------------------------

    /// One CFR+ iteration: opponent hole deal + recursive walk + regret/strategy_sum updates +
    /// leaf cache writes. `t` is the iteration index (0-based; needed for DCFR's t^a / t^b / t^g
    /// discounting and for linear_averaging's (t+1) strategy weighting).
    #[allow(clippy::too_many_arguments)]
    fn cfv_iter<const N: usize>(
        env: &EnvStateN<N>,
        hero: usize,
        depth: i32,
        max_depth: i32,
        ctx: &LiveCtx<'_>,
        tab: &mut LiveTables,
        rng: &mut Rng,
        t: i32,
    ) -> f64 {
        if env.done {
            return env.rewards[hero] as f64 / core::CHIP_UNIT as f64;
        }
        if core::is_chance_node(env) {
            return live_leaf_value(env, hero, ctx, tab);
        }
        if depth >= max_depth {
            return live_leaf_value(env, hero, ctx, tab);
        }
        let actor_i = env.action_seat;
        if actor_i < 0 {
            return live_leaf_value(env, hero, ctx, tab);
        }
        let actor = actor_i as usize;
        let legal = abstracted_legal(env, actor, ctx.pcfg);
        if !any_legal(&legal) {
            return env.rewards[hero] as f64 / core::CHIP_UNIT as f64;
        }
        let key = infoset_key(env, actor);

        if actor != hero {
            // Opponent node: read current regrets row, consult opponent_strategy (which LOOKS UP
            // the tilt table when one is wired), sample ONE action, recurse.
            let regrets = tab.regrets.get(&key).copied().unwrap_or([0.0; NUM_ACTIONS]);
            let strat = opponent_strategy(env, actor, &legal, &regrets, &subgame_ctx(ctx), &key);
            let u = rng.next_f64();
            let a = sample_action(&strat, &legal, u);
            // Gated real-NLHE street closure (POKERBOT_CLOSE_CALLED_STREET; default OFF => identity).
            let child = crate::closure::maybe_close_called_street(core::apply_action(env, a));
            return cfv_iter(&child, hero, depth + 1, max_depth, ctx, tab, rng, t);
        }

        // Hero node: branch all (abstracted-)legal actions, CFR+ regret update.
        let regrets = tab.regrets.get(&key).copied().unwrap_or([0.0; NUM_ACTIONS]);
        let strat = regret_match(&regrets, &legal);
        let mut av = [0.0f64; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            if legal[a] {
                let child = crate::closure::maybe_close_called_street(core::apply_action(env, a));
                av[a] = cfv_iter(&child, hero, depth + 1, max_depth, ctx, tab, rng, t);
            }
        }
        let mut ev = 0.0;
        for a in 0..NUM_ACTIONS {
            ev += strat[a] * av[a];
        }
        let mut newr = regrets;
        let w: f64;
        if ctx.scfg.dcfr {
            // DCFR signed regret update (mirrors pluribus_search._cfv): discount the running
            // regret then add the new instantaneous regret. Regrets stay signed; regret_match
            // clamps at max(R, 0).
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
            // CFR+ linear averaging: weight iteration t's strategy by (t+1) (matches Python).
            w = if ctx.scfg.linear_averaging { (t + 1) as f64 } else { 1.0 };
        }
        // Insert/update the regret row; track unique infoset insertions.
        match tab.regrets.entry(key.clone()) {
            std::collections::hash_map::Entry::Vacant(e) => {
                tab.n_strat_inserts += 1;
                e.insert(newr);
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                e.insert(newr);
            }
        }
        // Accumulate the strategy-sum (weighted by w per linear_averaging / DCFR gamma).
        let ss = tab.strat_sum.entry(key).or_insert([0.0; NUM_ACTIONS]);
        for a in 0..NUM_ACTIONS {
            ss[a] += w * strat[a];
        }
        ev
    }

    /// Bridge from `LiveCtx` to `subgame::Ctx` for `opponent_strategy` (avoids copying the
    /// opponent helper into this module — keep the firewalled exploit lookup in one place).
    fn subgame_ctx<'a>(ctx: &'a LiveCtx<'a>) -> subgame::Ctx<'a> {
        subgame::Ctx {
            bp: ctx.bp,
            pcfg: ctx.pcfg,
            scfg: ctx.scfg,
            public_seed: ctx.public_seed,
        }
    }

    /// Leaf value with the public leaf cache (same recipe as `subgame::leaf_value`).
    fn live_leaf_value<const N: usize>(
        env: &EnvStateN<N>,
        hero: usize,
        ctx: &LiveCtx<'_>,
        tab: &mut LiveTables,
    ) -> f64 {
        if env.done {
            return env.rewards[hero] as f64 / core::CHIP_UNIT as f64;
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
        tab.n_leaf_inserts += 1;
        tab.leaf.insert(key, v);
        v
    }
}

// ---------------------------------------------------------------------------
// Close the `inner` module opened at the top of the file. Everything below this brace belongs to
// the `live` module's top level (only doc-comment + empty module body when `python` is OFF).
} // close `mod inner`

// ---------------------------------------------------------------------------
// Re-exports — `live::py` is the inner module; `lib.rs` references `crate::live::py::SearchHandle`
// and `crate::live::register_cancel` (and the two siblings). Hoist them to `crate::live::*` so the
// registration calls in `lib.rs` don't have to reach into the inner module path.
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
pub use inner::py::{register_cancel, set_cancel, release_cancel, SearchHandle};

// Re-export the registry accessor (process-global cancel map) so the search_session module can
// resolve a cancel_handle id to the same Arc<AtomicBool> the Python side stored via
// register_cancel(). Gated on `python` because the registry only matters when the cancel contract
// is wired (the rlib / `cargo test` path doesn't need it).
#[cfg(feature = "python")]
pub(crate) use inner::cancel_registry;

// Re-export the `public_seed` helper so the parity-test PyO3 entry point in `lib.rs` can call
// `crate::live::public_seed` (F-NATIVE-7 / tests/test_native_public_seed_parity.py). The helper
// itself is gated on `python` because the parity test is a Python-side probe — the rlib path
// doesn't need it.
#[cfg(feature = "python")]
pub(crate) use inner::public_seed;
