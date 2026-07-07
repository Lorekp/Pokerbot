//! `battery_native` — bit-identical native core for the exploitation battery hot path.
//!
//! The `core` module is pure Rust (testable with `cargo test`, no Python). The PyO3 bindings below are
//! compiled only with `--features python` (the cdylib loaded by `pokerbot/search/native_solver.py`).
//!
//! State is marshaled as a single flat `Vec<i64>` (fixed offset layout, see FLAT_LEN below) to avoid
//! PyO3's 12-element tuple limit and to keep marshaling a single contiguous copy. The wrapper
//! (`native_solver.py`) builds the flat buffer from a numpy EnvState once per call; everything between
//! is native.

pub mod cfr_table;
pub mod core;
pub mod emd;
pub mod fingerprint;
// Lock-free (hogwild) shared regret/strategy table for the NLHE CFR+ trainer. Gated entirely behind
// `feature = "hogwild"` (default OFF) so the default build is byte-identical to the fork-merge trainer.
#[cfg(feature = "hogwild")]
pub mod hogwild;
pub mod icm;
pub mod lbr;
pub mod leduc;
pub mod nnet;
pub mod phe_tables;
pub mod playout;
pub mod river_ochs;
pub mod subgame;
pub mod trainer;

#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
pub mod flat {
    use crate::core::{EnvState, MAX_BOARD, MAX_HISTORY, NUM_PLAYERS};

    // Offset layout into the flat i64 buffer. Order mirrors EnvState._fields.
    pub const O_HOLES: usize = 0; // 12
    pub const O_BOARD: usize = O_HOLES + 12; // 5
    pub const O_SEAT_STATUS: usize = O_BOARD + MAX_BOARD; // 6
    pub const O_STACKS: usize = O_SEAT_STATUS + NUM_PLAYERS; // 6
    pub const O_COMMITTED_STREET: usize = O_STACKS + NUM_PLAYERS; // 6
    pub const O_COMMITTED_TOTAL: usize = O_COMMITTED_STREET + NUM_PLAYERS; // 6
    pub const O_STREET: usize = O_COMMITTED_TOTAL + NUM_PLAYERS; // 1
    pub const O_POT_DEAD: usize = O_STREET + 1;
    pub const O_CURRENT_BET: usize = O_POT_DEAD + 1;
    pub const O_MIN_RAISE: usize = O_CURRENT_BET + 1;
    pub const O_BUTTON: usize = O_MIN_RAISE + 1;
    pub const O_ACTION_SEAT: usize = O_BUTTON + 1;
    pub const O_LAST_AGGRESSOR: usize = O_ACTION_SEAT + 1;
    pub const O_ACTED: usize = O_LAST_AGGRESSOR + 1; // 6
    pub const O_LAST_RAISE_FULL: usize = O_ACTED + NUM_PLAYERS; // 1
    pub const O_DECK: usize = O_LAST_RAISE_FULL + 1; // 52
    pub const O_DECK_PTR: usize = O_DECK + 52;
    pub const O_RNG_KEY: usize = O_DECK_PTR + 1; // 2
    pub const O_HIST_SEAT: usize = O_RNG_KEY + 2; // 64
    pub const O_HIST_ACTION: usize = O_HIST_SEAT + MAX_HISTORY; // 64
    pub const O_HIST_AMOUNT: usize = O_HIST_ACTION + MAX_HISTORY; // 64
    pub const O_HIST_LEN: usize = O_HIST_AMOUNT + MAX_HISTORY;
    pub const O_DONE: usize = O_HIST_LEN + 1;
    pub const O_REWARDS: usize = O_DONE + 1; // 6
    pub const O_PENDING_DEAL: usize = O_REWARDS + NUM_PLAYERS;
    pub const FLAT_LEN: usize = O_PENDING_DEAL + 1;

    pub fn from_flat(b: &[i64]) -> EnvState {
        let mut holes = [[0i32; 2]; NUM_PLAYERS];
        for i in 0..NUM_PLAYERS {
            holes[i][0] = b[O_HOLES + i * 2] as i32;
            holes[i][1] = b[O_HOLES + i * 2 + 1] as i32;
        }
        let mut board = [0i32; MAX_BOARD];
        for i in 0..MAX_BOARD {
            board[i] = b[O_BOARD + i] as i32;
        }
        let mut seat_status = [0i8; NUM_PLAYERS];
        let mut stacks = [0i32; NUM_PLAYERS];
        let mut committed_street = [0i32; NUM_PLAYERS];
        let mut committed_total = [0i32; NUM_PLAYERS];
        let mut acted = [false; NUM_PLAYERS];
        let mut rewards = [0i32; NUM_PLAYERS];
        for i in 0..NUM_PLAYERS {
            seat_status[i] = b[O_SEAT_STATUS + i] as i8;
            stacks[i] = b[O_STACKS + i] as i32;
            committed_street[i] = b[O_COMMITTED_STREET + i] as i32;
            committed_total[i] = b[O_COMMITTED_TOTAL + i] as i32;
            acted[i] = b[O_ACTED + i] != 0;
            rewards[i] = b[O_REWARDS + i] as i32;
        }
        let mut deck = [0i32; 52];
        for i in 0..52 {
            deck[i] = b[O_DECK + i] as i32;
        }
        let rng_key = [b[O_RNG_KEY] as u32, b[O_RNG_KEY + 1] as u32];
        let mut hist_seat = [0i32; MAX_HISTORY];
        let mut hist_action = [0i32; MAX_HISTORY];
        let mut hist_amount = [0i32; MAX_HISTORY];
        for i in 0..MAX_HISTORY {
            hist_seat[i] = b[O_HIST_SEAT + i] as i32;
            hist_action[i] = b[O_HIST_ACTION + i] as i32;
            hist_amount[i] = b[O_HIST_AMOUNT + i] as i32;
        }
        EnvState {
            holes,
            board,
            seat_status,
            stacks,
            committed_street,
            committed_total,
            street: b[O_STREET] as i32,
            pot_dead: b[O_POT_DEAD] as i32,
            current_bet: b[O_CURRENT_BET] as i32,
            min_raise: b[O_MIN_RAISE] as i32,
            button: b[O_BUTTON] as i32,
            action_seat: b[O_ACTION_SEAT] as i32,
            last_aggressor: b[O_LAST_AGGRESSOR] as i32,
            acted,
            last_raise_full: b[O_LAST_RAISE_FULL] != 0,
            deck,
            deck_ptr: b[O_DECK_PTR] as i32,
            rng_key,
            hist_seat,
            hist_action,
            hist_amount,
            hist_len: b[O_HIST_LEN] as i32,
            done: b[O_DONE] != 0,
            rewards,
            pending_deal: b[O_PENDING_DEAL] != 0,
        }
    }

    pub fn to_flat(s: &EnvState) -> Vec<i64> {
        let mut b = vec![0i64; FLAT_LEN];
        for i in 0..NUM_PLAYERS {
            b[O_HOLES + i * 2] = s.holes[i][0] as i64;
            b[O_HOLES + i * 2 + 1] = s.holes[i][1] as i64;
            b[O_SEAT_STATUS + i] = s.seat_status[i] as i64;
            b[O_STACKS + i] = s.stacks[i] as i64;
            b[O_COMMITTED_STREET + i] = s.committed_street[i] as i64;
            b[O_COMMITTED_TOTAL + i] = s.committed_total[i] as i64;
            b[O_ACTED + i] = s.acted[i] as i64;
            b[O_REWARDS + i] = s.rewards[i] as i64;
        }
        for i in 0..MAX_BOARD {
            b[O_BOARD + i] = s.board[i] as i64;
        }
        for i in 0..52 {
            b[O_DECK + i] = s.deck[i] as i64;
        }
        for i in 0..MAX_HISTORY {
            b[O_HIST_SEAT + i] = s.hist_seat[i] as i64;
            b[O_HIST_ACTION + i] = s.hist_action[i] as i64;
            b[O_HIST_AMOUNT + i] = s.hist_amount[i] as i64;
        }
        b[O_STREET] = s.street as i64;
        b[O_POT_DEAD] = s.pot_dead as i64;
        b[O_CURRENT_BET] = s.current_bet as i64;
        b[O_MIN_RAISE] = s.min_raise as i64;
        b[O_BUTTON] = s.button as i64;
        b[O_ACTION_SEAT] = s.action_seat as i64;
        b[O_LAST_AGGRESSOR] = s.last_aggressor as i64;
        b[O_LAST_RAISE_FULL] = s.last_raise_full as i64;
        b[O_DECK_PTR] = s.deck_ptr as i64;
        b[O_RNG_KEY] = s.rng_key[0] as i64;
        b[O_RNG_KEY + 1] = s.rng_key[1] as i64;
        b[O_HIST_LEN] = s.hist_len as i64;
        b[O_DONE] = s.done as i64;
        b[O_PENDING_DEAL] = s.pending_deal as i64;
        b
    }
}

#[cfg(feature = "python")]
mod py {
    use crate::core::{self, EnvState, MAX_BOARD, NUM_ACTIONS, NUM_PLAYERS};
    use crate::fingerprint;
    use crate::flat;
    use crate::lbr::{self, LbrCfg, LbrPolicy, TabularPolicy};
    use crate::nnet::Net;
    use crate::playout::{self, Blueprint, PlayoutConfig};
    use crate::subgame::{self, SubgameConfig};
    use crate::trainer as nlhe_trainer;
    use pyo3::prelude::*;
    use pyo3::types::{PyBytes, PyDict};
    use std::collections::HashMap;
    use std::fmt::Write as _;
    use std::sync::Arc;

    // The state buffer is marshaled as raw little-endian i64 bytes (numpy `arr.tobytes()` -> here,
    // and a returned `bytes` -> `np.frombuffer(..., np.int64)` on the Python side). This is a single
    // C-speed memcpy each way, far cheaper than per-element list iteration.
    #[inline]
    fn bytes_to_flat(b: &[u8]) -> Vec<i64> {
        debug_assert_eq!(b.len(), flat::FLAT_LEN * 8);
        let mut v = vec![0i64; flat::FLAT_LEN];
        for (i, chunk) in b.chunks_exact(8).enumerate() {
            v[i] = i64::from_le_bytes(chunk.try_into().unwrap());
        }
        v
    }

    #[inline]
    fn flat_to_bytes(v: &[i64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 8);
        for &x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    /// `apply_action_np(state_bytes, action)` -> new flat state as bytes.
    #[pyfunction]
    fn apply_action<'py>(py: Python<'py>, state: &[u8], action: usize) -> Bound<'py, PyBytes> {
        let s = flat::from_flat(&bytes_to_flat(state));
        let out = flat::to_flat(&core::apply_action(&s, action));
        PyBytes::new(py, &flat_to_bytes(&out))
    }

    /// `sample_chance_np(state_bytes)` -> new flat state as bytes.
    #[pyfunction]
    fn sample_chance<'py>(py: Python<'py>, state: &[u8]) -> Bound<'py, PyBytes> {
        let s = flat::from_flat(&bytes_to_flat(state));
        let out = flat::to_flat(&core::sample_chance(&s));
        PyBytes::new(py, &flat_to_bytes(&out))
    }

    /// `init_hand(button, stacks, occupancy, ante, deck, rng_key)` -> flat root state as bytes.
    /// Byte-identical to `jax_engine.init_hand(rng_key, button, stacks, occupancy, ante)` WHEN given
    /// jax's permuted `deck` (52 ints) and the jax state's `rng_key` (2 u32). `stacks`/`occupancy` are
    /// length-NUM_PLAYERS; `occupancy` non-zero => SEAT_LIVE. The Threefry deck permutation is supplied
    /// by the caller (the cross-check feeds jax's own deck), not replicated natively.
    #[pyfunction]
    #[pyo3(signature = (button, stacks, occupancy, ante, deck, rng_key))]
    fn init_hand<'py>(
        py: Python<'py>,
        button: i32,
        stacks: Vec<i32>,
        occupancy: Vec<i32>,
        ante: i32,
        deck: Vec<i32>,
        rng_key: Vec<u32>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if stacks.len() != NUM_PLAYERS || occupancy.len() != NUM_PLAYERS {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "stacks and occupancy must have length NUM_PLAYERS",
            ));
        }
        if deck.len() != 52 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "deck must have length 52",
            ));
        }
        if rng_key.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rng_key must have length 2",
            ));
        }
        let mut st = [0i32; NUM_PLAYERS];
        let mut occ = [false; NUM_PLAYERS];
        for i in 0..NUM_PLAYERS {
            st[i] = stacks[i];
            occ[i] = occupancy[i] != 0;
        }
        let mut dk = [0i32; 52];
        dk.copy_from_slice(&deck[..52]);
        let key = [rng_key[0], rng_key[1]];
        let out = flat::to_flat(&core::init_hand(button, &st, &occ, ante, &dk, key));
        Ok(PyBytes::new(py, &flat_to_bytes(&out)))
    }

    /// `legal_action_mask_np(state_bytes, seat)` -> list[bool] (len NUM_ACTIONS).
    #[pyfunction]
    #[pyo3(signature = (state, seat=-1))]
    fn legal_action_mask(state: &[u8], seat: i32) -> Vec<bool> {
        let s = flat::from_flat(&bytes_to_flat(state));
        core::legal_action_mask(&s, seat).to_vec()
    }

    /// `is_chance_node_np(state_bytes)`.
    #[pyfunction]
    fn is_chance_node(state: &[u8]) -> bool {
        core::is_chance_node(&flat::from_flat(&bytes_to_flat(state)))
    }

    /// `evaluate_seven(cards)` -> int32 score. cards = 7 card ids.
    #[pyfunction]
    fn evaluate_seven(cards: Vec<i32>) -> i32 {
        let mut c = [0i32; 7];
        c.copy_from_slice(&cards[..7]);
        core::evaluate_seven(&c)
    }

    /// `compute_returns_np` -> list[i32] (len NUM_PLAYERS).
    #[pyfunction]
    fn compute_returns(
        holes: Vec<i32>,
        board: Vec<i32>,
        seat_status: Vec<i8>,
        committed_total: Vec<i32>,
        button: i32,
        n_showdown: i32,
    ) -> Vec<i32> {
        let mut h = [[0i32; 2]; NUM_PLAYERS];
        for i in 0..NUM_PLAYERS {
            h[i][0] = holes[i * 2];
            h[i][1] = holes[i * 2 + 1];
        }
        let mut b = [0i32; MAX_BOARD];
        b.copy_from_slice(&board[..MAX_BOARD]);
        let mut ss = [0i8; NUM_PLAYERS];
        ss.copy_from_slice(&seat_status[..NUM_PLAYERS]);
        let mut ct = [0i32; NUM_PLAYERS];
        ct.copy_from_slice(&committed_total[..NUM_PLAYERS]);
        core::compute_returns(&h, &b, &ss, &ct, button, n_showdown).to_vec()
    }

    /// `fingerprint_fast(state_bytes, seat, num_buckets)` -> 16-byte digest. Byte-identical to the
    /// Python `pokerbot.engine.jax_weva_fast.fingerprint_fast` for the bounded eval config.
    #[pyfunction]
    fn fingerprint_fast<'py>(
        py: Python<'py>,
        state: &[u8],
        seat: usize,
        num_buckets: i32,
    ) -> Bound<'py, PyBytes> {
        let s = flat::from_flat(&bytes_to_flat(state));
        let d = fingerprint::fingerprint_fast(&s, seat, num_buckets);
        PyBytes::new(py, &d)
    }

    /// `fingerprint_fast_depth(state_bytes, seat, num_buckets, depth_edges)` -> 16-byte digest with the
    /// depth byte hashed into the preflop key (empty `depth_edges` => identical to `fingerprint_fast`).
    #[pyfunction]
    fn fingerprint_fast_depth<'py>(
        py: Python<'py>,
        state: &[u8],
        seat: usize,
        num_buckets: i32,
        depth_edges: Vec<f64>,
    ) -> Bound<'py, PyBytes> {
        let s = flat::from_flat(&bytes_to_flat(state));
        let d = fingerprint::fingerprint_fast_depth(&s, seat, num_buckets, &depth_edges);
        PyBytes::new(py, &d)
    }

    /// `multiway_equity_river(lbr0, lbr1, opp_flat, board5)` -> P(LBR best, ties shared) at a full board
    /// (deterministic; cross-check entry for the LBR showdown equity).
    #[pyfunction]
    fn multiway_equity_river(lbr0: i32, lbr1: i32, opp_flat: Vec<i32>, board: Vec<i32>) -> f64 {
        let opps: Vec<(i32, i32)> = opp_flat.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        lbr::multiway_equity_river_test((lbr0, lbr1), &opps, &board)
    }

    /// `eval_seven_phe(cards)` -> phevaluator 1..7462 rank for 5..7 card ids (cross-check helper).
    #[pyfunction]
    fn eval_seven_phe(cards: Vec<i32>) -> i32 {
        fingerprint::evaluate_cards(&cards)
    }

    /// Capability probe for the Python fail-closed gates (`native_solver.py` /
    /// `fingerprint_native.py`): which serve-abstraction regimes this binary can key correctly.
    /// A binary built before the EMD port lacks this symbol entirely, so `hasattr` alone already
    /// fails closed on stale .pyds.
    #[pyfunction]
    fn native_caps() -> Vec<String> {
        vec![
            "emd_fingerprint".to_string(),
            "emd_subgame".to_string(),
            "emd_rollout".to_string(),
            "emd_lbr".to_string(),
            // blueprint-v3: fingerprint.rs carries the env-gated (POKERBOT_PREFLOP_HIST_KEY=1)
            // 3-byte preflop history key; native_solver fails closed on binaries without this.
            "preflop_hist_key".to_string(),
            // blueprint-v4 Stage 0: fingerprint.rs carries the env-gated (POKERBOT_FULLHIST_KEY=1)
            // full action-history key segment on every street (design doc §7.1; default-off
            // byte-identical). Stale .pyds without this cap must not serve a fullhist checkpoint.
            "full_hist_key".to_string(),
        ]
    }

    /// Resolve the optional EMD keying ctor args into the shared (once-per-process) lookup config.
    fn emd_key_cfg(
        emd_frozen_path: Option<String>,
        emd_miss_made_hand: bool,
    ) -> PyResult<Option<crate::playout::EmdKeyCfg>> {
        match emd_frozen_path {
            None => Ok(None),
            Some(p) => {
                let lookup = crate::emd::load_shared(&p)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                Ok(Some(crate::playout::EmdKeyCfg {
                    lookup,
                    miss_made_hand: emd_miss_made_hand,
                }))
            }
        }
    }

    /// `fingerprint_fast_emd(state_bytes, seat, num_buckets, depth_edges, emd_frozen_path,
    /// emd_miss_made_hand)` -> 16-byte digest under the EMD frozen-lookup abstraction. Byte-parity
    /// target: `jax_weva_fast.fingerprint_fast` under `POKERBOT_POSTFLOP_MODE=emd` (the deployed
    /// EMD-K160 regime). The blob loads once per process (shared cache). Used by the parity gate
    /// (`scripts/probe_native_emd_parity.py`) and any EMD-mode key crosscheck.
    #[pyfunction]
    #[pyo3(signature = (state, seat, num_buckets, depth_edges, emd_frozen_path, emd_miss_made_hand=true))]
    fn fingerprint_fast_emd<'py>(
        py: Python<'py>,
        state: &[u8],
        seat: usize,
        num_buckets: i32,
        depth_edges: Vec<f64>,
        emd_frozen_path: String,
        emd_miss_made_hand: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let lookup = crate::emd::load_shared(&emd_frozen_path)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let s = flat::from_flat(&bytes_to_flat(state));
        let d = fingerprint::fingerprint_fast_depth_emd(
            &s,
            seat,
            num_buckets,
            &depth_edges,
            &lookup,
            emd_miss_made_hand,
        );
        Ok(PyBytes::new(py, &d))
    }

    /// `subgame_ctx_key_id(state, actor)` -> the all_contexts() index (0..21) the native exploit path
    /// recomputes from public state, or -1 on an out-of-range street / no-texture state. Cross-check
    /// entry asserting the Rust `ctx_key_id` port equals `list(all_contexts()).index(
    /// _context_key_from_env(env, actor))` — the firewall KEY contract.
    #[pyfunction]
    fn subgame_ctx_key_id(state: &[u8], actor: usize) -> i32 {
        let s = flat::from_flat(&bytes_to_flat(state));
        match subgame::ctx_key_id_pub(&s, actor) {
            Some(id) => id as i32,
            None => -1,
        }
    }

    /// Native uniform-mode continuation leaf engine. Holds the blueprint + config ONCE per worker;
    /// `leaf_value` runs the rollout for a depth-limit state with no per-step Python marshaling.
    #[pyclass]
    struct PlayoutEngine {
        bp: Blueprint,
        cfg: PlayoutConfig,
    }

    #[pymethods]
    impl PlayoutEngine {
        /// blueprint: dict[bytes(16) -> sequence[float] len 11]; abstraction masks len 11.
        /// `emd_frozen_path`/`emd_miss_made_hand` (kwargs, default None/true): EMD frozen-lookup
        /// keying for the EMD-K160 serve regime; omitted => the made-hand key, byte-identical.
        #[new]
        #[pyo3(signature = (blueprint, allowed_pre, allowed_post, bias_mult, bias_floor, max_depth, num_buckets, emd_frozen_path=None, emd_miss_made_hand=true))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            blueprint: &Bound<'_, PyDict>,
            allowed_pre: Vec<bool>,
            allowed_post: Vec<bool>,
            bias_mult: f64,
            bias_floor: f64,
            max_depth: i32,
            num_buckets: i32,
            emd_frozen_path: Option<String>,
            emd_miss_made_hand: bool,
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
            Ok(PlayoutEngine {
                bp,
                cfg: PlayoutConfig {
                    allowed_pre: ap,
                    allowed_post: po,
                    bias_mult,
                    bias_floor,
                    max_depth,
                    num_buckets,
                    emd: emd_key_cfg(emd_frozen_path, emd_miss_made_hand)?,
                },
            })
        }

        fn n_keys(&self) -> usize {
            self.bp.len()
        }

        /// Uniform-mode continuation leaf value (hero bb) for a depth-limit state.
        fn leaf_value(
            &self,
            state: &[u8],
            hero: usize,
            base_seed: u64,
            rollout_samples: i32,
            resample_board: bool,
        ) -> f64 {
            let s = flat::from_flat(&bytes_to_flat(state));
            playout::uniform_leaf_value(
                &s,
                hero,
                &self.bp,
                &self.cfg,
                base_seed,
                rollout_samples,
                resample_board,
            )
        }
    }

    /// Native depth-limited CFR subgame solver (GTO-base / opponent_model=None path). Holds the
    /// blueprint + abstraction config ONCE; `solve` runs the whole CFR loop (hole sampling, recursion,
    /// regret-matching, blueprint mix, continuation-gadget leaf) natively and returns the root action
    /// probabilities — no per-node Python marshaling. See `subgame.rs`.
    #[pyclass]
    struct SubgameSolver {
        bp: Blueprint,
        cfg: PlayoutConfig,
    }

    #[pymethods]
    impl SubgameSolver {
        /// Same construction as `PlayoutEngine`: blueprint dict + abstraction masks + leaf knobs.
        /// `leaf_max_depth` bounds the per-leaf rollout (PlayoutConfig.max_depth); the CFR
        /// betting-recursion depth is passed per-call to `solve`.
        /// `emd_frozen_path`/`emd_miss_made_hand` (kwargs, default None/true): EMD frozen-lookup
        /// keying — applies to BOTH `solve` and `solve_exploit` blueprint lookups (the exploit
        /// path's frozen-table firewall knobs are untouched; only the key abstraction changes).
        #[new]
        #[pyo3(signature = (blueprint, allowed_pre, allowed_post, bias_mult, bias_floor, leaf_max_depth, num_buckets, emd_frozen_path=None, emd_miss_made_hand=true))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            blueprint: &Bound<'_, PyDict>,
            allowed_pre: Vec<bool>,
            allowed_post: Vec<bool>,
            bias_mult: f64,
            bias_floor: f64,
            leaf_max_depth: i32,
            num_buckets: i32,
            emd_frozen_path: Option<String>,
            emd_miss_made_hand: bool,
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
            Ok(SubgameSolver {
                bp,
                cfg: PlayoutConfig {
                    allowed_pre: ap,
                    allowed_post: po,
                    bias_mult,
                    bias_floor,
                    max_depth: leaf_max_depth,
                    num_buckets,
                    emd: emd_key_cfg(emd_frozen_path, emd_miss_made_hand)?,
                },
            })
        }

        fn n_keys(&self) -> usize {
            self.bp.len()
        }

        /// Solve the depth-limited subgame rooted at `state` for `hero`; returns 11 root action probs.
        #[allow(clippy::too_many_arguments)]
        fn solve(
            &self,
            state: &[u8],
            hero: usize,
            public_seed: u64,
            max_iterations: i32,
            max_depth: i32,
            blueprint_mix: f64,
            rollout_samples: i32,
            resample_board: bool,
            linear_averaging: bool,
            repair_deck: bool,
            two_sided: bool,
            decouple_sampler: bool,
            range_aware: bool,
            range_uniform_floor: f64,
            range_reach_full: bool,
            ensemble_streams: i32,
            leaf_cache_worlds: i32,
            avg_skip_frac: f64,
            leaf_biased_continuations: bool,
            leaf_bias_mult: f64,
            serve_final_iterate: bool,
            root_unmask: u32,
        ) -> Vec<f64> {
            let s = flat::from_flat(&bytes_to_flat(state));
            let scfg = SubgameConfig {
                max_iterations,
                max_depth,
                blueprint_mix,
                rollout_samples,
                resample_board,
                linear_averaging,
                repair_deck,
                two_sided,
                decouple_sampler,
                range_aware,
                range_uniform_floor,
                // R1 reach-consistent opponent root ranges (default OFF at the Python caller => the
                // legacy fold-now / uniform deal, byte-identical).
                range_reach_full,
                ensemble_streams,
                // BUG-1 leaf hole-world integration (1 = legacy first-touch freeze) + BUG-5 delayed
                // averaging (0.0 = legacy accumulate-from-t=0). Both default no-op at the Python caller.
                leaf_cache_worlds,
                avg_skip_frac,
                // R2 — Pluribus biased continuation leaves (default OFF at the Python caller => the
                // legacy uniform leaf, byte-identical).
                leaf_biased_continuations,
                leaf_bias_mult,
                // R5a — final-iterate serving (default OFF at the Python caller => the averaged
                // extraction, byte-identical).
                serve_final_iterate,
                // STAGE-1 — river overbet / off-tree ROOT un-mask bitmask (root-only, depth==0,
                // full-legal-ANDed; 0 default at the Python caller => byte-identical).
                root_unmask,
            };
            subgame::solve(&s, hero, &self.bp, &self.cfg, &scfg, public_seed).to_vec()
        }

        /// PROBE/DEBUG (R1, never on the solve hot path): the per-seat opponent-range weights the
        /// world sampler would draw from at this root. `mode` is "fold_now" (the Q3 proxy),
        /// "reach" (the R1 blueprint-reach product) or "reach_strict" (reach with NO fold-now
        /// fallback — returns None whenever the history replay fails, exposing the fallback path).
        /// Returns a seat-major flattened `N*52*52` weight vector (upper triangle c0<c1 populated,
        /// pre-floor / pre-normalization), or None when no table applies. Used by the offline
        /// sanity + log-loss validation of the reach-range plumbing.
        fn range_weights(&self, state: &[u8], hero: usize, mode: &str) -> Option<Vec<f64>> {
            let s = flat::from_flat(&bytes_to_flat(state));
            let (range_aware, range_reach_full) = match mode {
                "fold_now" => (true, false),
                "reach" => (true, true),
                "reach_strict" => (false, true),
                _ => return None,
            };
            let scfg = SubgameConfig {
                max_iterations: 0,
                max_depth: 0,
                blueprint_mix: 1.0,
                rollout_samples: 1,
                resample_board: false,
                linear_averaging: true,
                repair_deck: false,
                two_sided: false,
                decouple_sampler: false,
                range_aware,
                range_uniform_floor: 0.25,
                range_reach_full,
                ensemble_streams: 1,
                leaf_cache_worlds: 1,
                avg_skip_frac: 0.0,
                leaf_biased_continuations: false,
                leaf_bias_mult: 5.0,
                serve_final_iterate: false,
                root_unmask: 0,
            };
            subgame::build_rtab(&s, hero, &self.bp, &self.cfg, &scfg).map(|rt| rt.to_flat())
        }

        /// EXPLOIT solve: same CFR loop as `solve`, but opponent nodes apply a PRECOMPUTED pure-frozen
        /// deviation (built in Python by `exploit_table.build_frozen_table` — the exploit MATH stays
        /// there) and the root per-action CFVs are exposed for the AIVAT baseline. The frozen table is
        /// passed as parallel lists `(keys_actor:u8, keys_ctx:u16, keys_mask:u16)` + `vals` (flat
        /// `len*11` f64) reconstructed into a `FrozenTable`. `rnr` is the length-6 per-actor scalar
        /// (NaN = overwrite-with-frozen). Returns `(root_probs[11], root_q[11])` (root_q all-zeros when
        /// `want_root_q` is false). The ctor is UNCHANGED (the table is per-call, not baked), so the
        /// model=None engine cache key is shared/untouched.
        #[allow(clippy::too_many_arguments)]
        fn solve_exploit(
            &self,
            state: &[u8],
            hero: usize,
            public_seed: u64,
            max_iterations: i32,
            max_depth: i32,
            blueprint_mix: f64,
            rollout_samples: i32,
            resample_board: bool,
            linear_averaging: bool,
            keys_actor: Vec<u8>,
            keys_ctx: Vec<u16>,
            keys_mask: Vec<u16>,
            vals: Vec<f64>,
            rnr: Vec<f64>,
            frozen_read_response: bool,
            rnr_restrict: bool,
            want_root_q: bool,
            repair_deck: bool,
            decouple_sampler: bool,
            range_aware: bool,
            range_uniform_floor: f64,
            range_reach_full: bool,
        ) -> (Vec<f64>, Vec<f64>) {
            let s = flat::from_flat(&bytes_to_flat(state));
            let scfg = SubgameConfig {
                max_iterations,
                max_depth,
                blueprint_mix,
                rollout_samples,
                resample_board,
                linear_averaging,
                repair_deck,
                // FIREWALL: two_sided is NEVER exposed on the exploit path (one-sided BR to the modeled
                // opponent is the intended behaviour). Hard-code false so a caller mistake can't flip it.
                two_sided: false,
                // Q2/Q3/R1 world-sampling levers apply to the exploit path too (they change ONLY the
                // sampled opponent worlds, never the frozen-read exploit math -> firewall-safe).
                decouple_sampler,
                range_aware,
                range_uniform_floor,
                range_reach_full,
                // ENSEMBLE: the exploit path is left single-stream for now (solve_exploit ignores this
                // field entirely). Firewalled follow-up once the GTO ensemble is validated.
                ensemble_streams: 1,
                // FIREWALL: leaf_cache_worlds / avg_skip_frac are GTO-path knobs, NOT exposed on the
                // exploit FFI — hard-code the no-op values so solve_exploit is byte-identical to the
                // certified exploit path regardless of what a caller sets on the GTO side (subgame.rs
                // additionally forces them in the exploit Ctx, defense in depth).
                leaf_cache_worlds: 1,
                avg_skip_frac: 0.0,
                // FIREWALL: R2 biased continuation leaves are a GTO-path knob, NOT exposed on the
                // exploit FFI — hard-code OFF (subgame.rs additionally forces leaf_biased=false in the
                // exploit Ctx, defense in depth) so the certified exploit leaf is untouched.
                leaf_biased_continuations: false,
                leaf_bias_mult: 5.0,
                // FIREWALL: R5a final-iterate serving is a GTO-path knob, NOT exposed on the exploit
                // FFI — hard-code OFF (subgame.rs solve_exploit's extraction never reads the field
                // either, defense in depth) so the certified averaged exploit serve is untouched.
                serve_final_iterate: false,
                // FIREWALL: root_unmask (Stage-1 river overbet / off-tree root widening) is a
                // GTO-path knob, NOT exposed on the exploit FFI — hard-code 0 (subgame.rs
                // additionally forces it 0 in the exploit Ctx, defense in depth) so the certified
                // exploit tree can never be widened at the root.
                root_unmask: 0,
            };
            // Reconstruct the FrozenTable from the parallel key lists + flat vals (chunked by 11).
            let n = keys_actor
                .len()
                .min(keys_ctx.len())
                .min(keys_mask.len())
                .min(vals.len() / NUM_ACTIONS);
            let mut frozen: subgame::FrozenTable = HashMap::with_capacity(n);
            for i in 0..n {
                let mut row = [0.0f64; NUM_ACTIONS];
                row.copy_from_slice(&vals[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS]);
                frozen.insert((keys_actor[i], keys_ctx[i], keys_mask[i]), row);
            }
            // rnr -> [f64;6] (pad missing with NaN = overwrite-with-frozen sentinel).
            let mut rnr6 = [f64::NAN; 6];
            for i in 0..6.min(rnr.len()) {
                rnr6[i] = rnr[i];
            }
            let (probs, root_q) = subgame::solve_exploit(
                &s,
                hero,
                &self.bp,
                &self.cfg,
                &scfg,
                public_seed,
                &frozen,
                &rnr6,
                rnr_restrict,
                frozen_read_response,
                want_root_q,
            );
            (probs.to_vec(), root_q.to_vec())
        }
    }

    // -----------------------------------------------------------------------
    // Native LBR exploitability harness (tabular blueprint / calibration policies).
    // -----------------------------------------------------------------------
    /// Result of a native LBR run. `per_seat` is a list of (seat, n_hands, bb_per_100, std_bb_per_hand).
    #[pyclass(skip_from_py_object)]
    #[derive(Clone)]
    struct Lbr6Out {
        #[pyo3(get)]
        ev_bb_per_hand: f64,
        #[pyo3(get)]
        stderr_bb_per_hand: f64,
        #[pyo3(get)]
        hands: usize,
        #[pyo3(get)]
        per_seat: Vec<(usize, u64, f64, f64)>,
    }

    fn roots_from_blob(blob: &[u8], n: usize) -> Vec<EnvState> {
        let stride = flat::FLAT_LEN * 8;
        let mut out = Vec::with_capacity(n);
        for h in 0..n {
            let window = &blob[h * stride..(h + 1) * stride];
            out.push(flat::from_flat(&bytes_to_flat(window)));
        }
        out
    }

    fn pack_outcome(o: lbr::LbrOutcome) -> Lbr6Out {
        let per_seat = o
            .per_seat
            .iter()
            .enumerate()
            .filter(|(_, a)| a.n > 0)
            .map(|(seat, a)| {
                let nf = a.n as f64;
                let bb100 = a.sum / nf * 100.0;
                let std = if a.n > 1 {
                    ((a.sumsq - a.sum * a.sum / nf) / (nf - 1.0))
                        .max(0.0)
                        .sqrt()
                } else {
                    0.0
                };
                (seat, a.n, bb100, std)
            })
            .collect();
        Lbr6Out {
            ev_bb_per_hand: o.ev_bb_per_hand,
            stderr_bb_per_hand: o.stderr_bb_per_hand,
            hands: o.hands,
            per_seat,
        }
    }

    fn allowed_arr(v: &[bool]) -> [bool; NUM_ACTIONS] {
        let mut a = [false; NUM_ACTIONS];
        for i in 0..NUM_ACTIONS {
            a[i] = *v.get(i).unwrap_or(&false);
        }
        a
    }

    /// Holds the blueprint snapshot ONCE; `lbr_tabular` runs the whole native LBR (engine + belief +
    /// equity + fingerprint serve) across rayon workers, returning the exploitability lower bound.
    #[pyclass]
    struct Lbr6Harness {
        bp: Blueprint,
        allowed_pre: [bool; NUM_ACTIONS],
        allowed_post: [bool; NUM_ACTIONS],
        num_buckets: i32,
        depth_edges: Vec<f64>,
        emd: Option<crate::playout::EmdKeyCfg>,
    }

    #[pymethods]
    impl Lbr6Harness {
        /// blueprint: dict[bytes(16) -> seq[float] len 11] (may be EMPTY for calibration modes).
        /// `depth_edges`: depth-bucket edges for a DEPTH-BUCKETED blueprint (e.g. [50,90,140]); pass an
        /// empty list for a non-depth blueprint -- the preflop key then omits the depth byte.
        /// `emd_frozen_path`/`emd_miss_made_hand` (kwargs, default None/true): EMD frozen-lookup
        /// keying for an EMD-keyed blueprint; omitted => the made-hand key, byte-identical.
        #[new]
        #[pyo3(signature = (blueprint, allowed_pre, allowed_post, num_buckets, depth_edges, emd_frozen_path=None, emd_miss_made_hand=true))]
        fn new(
            blueprint: &Bound<'_, PyDict>,
            allowed_pre: Vec<bool>,
            allowed_post: Vec<bool>,
            num_buckets: i32,
            depth_edges: Vec<f64>,
            emd_frozen_path: Option<String>,
            emd_miss_made_hand: bool,
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
                if row.len() != NUM_ACTIONS {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "blueprint row must have 11 entries",
                    ));
                }
                let mut r = [0.0f64; NUM_ACTIONS];
                r.copy_from_slice(&row[..NUM_ACTIONS]);
                bp.insert(key, r);
            }
            Ok(Lbr6Harness {
                bp,
                allowed_pre: allowed_arr(&allowed_pre),
                allowed_post: allowed_arr(&allowed_post),
                num_buckets,
                depth_edges,
                emd: emd_key_cfg(emd_frozen_path, emd_miss_made_hand)?,
            })
        }

        /// Load (shared, once/process) the frozen EMD lookup and key ALL subsequent blueprint
        /// lookups with it (`pokerbot/eval/lbr_native.py` calls this when it is handed an
        /// `emd_frozen_path`). Returns the lookup meta `(n_buckets, n_bins, n_runouts, rows)` —
        /// the wrapper cross-checks `n_buckets` against the active abstraction. `miss_made_hand`
        /// defaults to the mirror rule (made-hand on frozen miss — the trainer/serve semantics).
        #[pyo3(signature = (path, miss_made_hand=true))]
        fn set_emd_frozen(
            &mut self,
            path: String,
            miss_made_hand: bool,
        ) -> PyResult<(usize, usize, usize, usize)> {
            let lookup = crate::emd::load_shared(&path)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            let meta = (
                lookup.n_buckets,
                lookup.n_bins,
                lookup.n_runouts,
                lookup.rows.len(),
            );
            self.emd = Some(crate::playout::EmdKeyCfg {
                lookup,
                miss_made_hand,
            });
            Ok(meta)
        }

        fn n_keys(&self) -> usize {
            self.bp.len()
        }

        /// Serve the tabular policy for one (state, seat) -- the blueprint/calibration cross-check entry.
        /// `policy_mode`: 0 blueprint, 1 fold, 2 nit, 3 uniform, 4 calling-station.
        fn serve(&self, state: &[u8], seat: usize, policy_mode: u8) -> Vec<f64> {
            let s = flat::from_flat(&bytes_to_flat(state));
            let policy = TabularPolicy {
                bp: self.bp.clone(),
                allowed_pre: self.allowed_pre,
                allowed_post: self.allowed_post,
                num_buckets: self.num_buckets,
                depth_edges: self.depth_edges.clone(),
                emd: self.emd.clone(),
                mode: policy_mode,
            };
            policy.probs(&s, seat).to_vec()
        }

        /// Run the LBR. `roots` is a contiguous int64-LE blob of `lbr_seats.len()` flat EnvStates.
        /// `policy_mode`: 0 blueprint, 1 always-fold, 2 nit, 3 uniform, 4 calling-station
        /// (1-4 are harness calibration policies).
        #[allow(clippy::too_many_arguments)]
        fn lbr_tabular(
            &self,
            py: Python<'_>,
            roots: &[u8],
            lbr_seats: Vec<usize>,
            seeds: Vec<u64>,
            n_seats: usize,
            track_ranges: bool,
            range_cap: usize,
            eq_samples: i32,
            eq_runouts: i32,
            settle_samples: i32,
            max_depth: i32,
            bet_candidates: Vec<usize>,
            value_only_raises: bool,
            policy_mode: u8,
        ) -> PyResult<Lbr6Out> {
            let n = lbr_seats.len();
            if seeds.len() != n {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "lbr_seats and seeds must have equal length",
                ));
            }
            if roots.len() != n * flat::FLAT_LEN * 8 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "roots blob length != n_hands * FLAT_LEN * 8",
                ));
            }
            // Clone the blueprint handle into the policy. The HashMap clone is the one non-trivial copy,
            // but it is per-call (not per-hand) and keeps the policy self-contained for the rayon closure.
            let policy = TabularPolicy {
                bp: self.bp.clone(),
                allowed_pre: self.allowed_pre,
                allowed_post: self.allowed_post,
                num_buckets: self.num_buckets,
                depth_edges: self.depth_edges.clone(),
                emd: self.emd.clone(),
                mode: policy_mode,
            };
            let cfg = LbrCfg {
                n_seats,
                track_ranges,
                range_cap,
                eq_samples,
                eq_runouts,
                settle_samples,
                max_depth,
                bet_candidates,
                value_only_raises,
            };
            let states = roots_from_blob(roots, n);
            let outcome = py.detach(|| lbr::run(&states, &lbr_seats, &seeds, &policy, &cfg));
            Ok(pack_outcome(outcome))
        }
    }

    /// Native HU/6-max LBR serving the trained neural average-strategy net (the CFR validator). Holds
    /// the small MLP weights ONCE; `lbr_neural` runs the same native LBR as `Lbr6Harness` but with the
    /// net as the policy (no fingerprint/abstraction -- the net sees raw cards).
    #[pyclass]
    struct LbrNeuralHarness {
        net: Net,
    }

    #[pymethods]
    impl LbrNeuralHarness {
        /// 3 dense layers as flat f32 arrays (kernel row-major in-major, bias len out) + dims + the
        /// checkpoint's info_suits flag.
        #[allow(clippy::too_many_arguments)]
        #[new]
        fn new(
            w0: Vec<f32>,
            b0: Vec<f32>,
            w1: Vec<f32>,
            b1: Vec<f32>,
            w2: Vec<f32>,
            b2: Vec<f32>,
            din: Vec<usize>,
            dout: Vec<usize>,
            info_suits: bool,
        ) -> PyResult<Self> {
            if din.len() != 3 || dout.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "din/dout must have length 3",
                ));
            }
            let check = |w: &[f32], b: &[f32], i: usize, o: usize, l: usize| -> PyResult<()> {
                if w.len() != i * o || b.len() != o {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "layer {l}: expected w={}*{}, b={}; got w={}, b={}",
                        i,
                        o,
                        o,
                        w.len(),
                        b.len()
                    )));
                }
                Ok(())
            };
            check(&w0, &b0, din[0], dout[0], 0)?;
            check(&w1, &b1, din[1], dout[1], 1)?;
            check(&w2, &b2, din[2], dout[2], 2)?;
            Ok(LbrNeuralHarness {
                net: Net::new(
                    w0,
                    b0,
                    w1,
                    b1,
                    w2,
                    b2,
                    [din[0], din[1], din[2]],
                    [dout[0], dout[1], dout[2]],
                    info_suits,
                ),
            })
        }

        /// Serve the net's action distribution for one (state, seat) -- the golden cross-check entry.
        fn serve(&self, state: &[u8], seat: usize) -> Vec<f64> {
            let s = flat::from_flat(&bytes_to_flat(state));
            self.net.probs(&s, seat).to_vec()
        }

        #[allow(clippy::too_many_arguments)]
        fn lbr_neural(
            &self,
            py: Python<'_>,
            roots: &[u8],
            lbr_seats: Vec<usize>,
            seeds: Vec<u64>,
            n_seats: usize,
            track_ranges: bool,
            range_cap: usize,
            eq_samples: i32,
            eq_runouts: i32,
            settle_samples: i32,
            max_depth: i32,
            bet_candidates: Vec<usize>,
            value_only_raises: bool,
        ) -> PyResult<Lbr6Out> {
            let n = lbr_seats.len();
            if seeds.len() != n || roots.len() != n * flat::FLAT_LEN * 8 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "roots/seeds length mismatch",
                ));
            }
            let cfg = LbrCfg {
                n_seats,
                track_ranges,
                range_cap,
                eq_samples,
                eq_runouts,
                settle_samples,
                max_depth,
                bet_candidates,
                value_only_raises,
            };
            let states = roots_from_blob(roots, n);
            let outcome = py.detach(|| lbr::run(&states, &lbr_seats, &seeds, &self.net, &cfg));
            Ok(pack_outcome(outcome))
        }
    }

    // -----------------------------------------------------------------------
    // Native NLHE CFR+ trainer used by the cloud checkpoint/serve pipeline.
    // -----------------------------------------------------------------------
    #[pyclass]
    struct NlheCFRTrainer {
        tab: nlhe_trainer::NlheTables,
        cfg: nlhe_trainer::TrainerCfg,
        emd_stamp: String,
        river_stamp: String,
        // HOGWILD runtime state (feature `hogwild`). `Some` when the trainer was constructed with the
        // hogwild path enabled (constructor `hogwild=True` OR env `POKERBOT_HOGWILD=1`): all `train` calls
        // then read+write the shared lock-free table in place (no fork/merge/floor barrier). When `None`
        // (the default), the trainer is the unchanged fork-merge path on `tab`. `iters_done` is tracked on
        // the shared state because the durable `tab` is only materialized on save/snapshot.
        #[cfg(feature = "hogwild")]
        hog: Option<HogState>,
    }

    #[cfg(feature = "hogwild")]
    struct HogState {
        table: crate::hogwild::SharedHogwildTable,
        iters_done: u64,
    }

    #[pymethods]
    impl NlheCFRTrainer {
        #[new]
        #[pyo3(signature = (num_buckets, depth_edges, n_deals_per_iter, n_seats=6, prune_threshold=f64::NEG_INFINITY, fast_merge=false, emd_frozen_path=None, batch_deals=0, hogwild=false, hogwild_capacity=1_048_576, work_steal=false, dcfr=false, dcfr_prune_threshold=-300_000.0, dcfr_explore_frac=0.05, river_ochs_path=None))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            num_buckets: i32,
            depth_edges: Vec<f64>,
            n_deals_per_iter: usize,
            n_seats: usize,
            prune_threshold: f64,
            fast_merge: bool,
            emd_frozen_path: Option<String>,
            batch_deals: usize,
            hogwild: bool,
            hogwild_capacity: usize,
            // CHANGE A — dynamic work-stealing deal scheduler (default OFF; also env POKERBOT_WORKSTEAL=1).
            work_steal: bool,
            // CHANGE B — DCFR signed-regret + active RBP pruning (default OFF; also env POKERBOT_DCFR=1).
            dcfr: bool,
            dcfr_prune_threshold: f64,
            dcfr_explore_frac: f64,
            // Blueprint-v2 river OCHS table (`river_ochs.rs`); None (default) = made-hand mirror river,
            // byte-identical to the shipped trainer.
            river_ochs_path: Option<String>,
        ) -> PyResult<Self> {
            let (emd_frozen, emd_stamp) = match emd_frozen_path {
                Some(path) => {
                    let lookup = crate::emd::FrozenEMDLookup::load_path(&path)
                        .map_err(pyo3::exceptions::PyValueError::new_err)?;
                    if lookup.n_buckets as i32 != num_buckets {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "EMD frozen K mismatch: lookup has {}, trainer requested {}",
                            lookup.n_buckets, num_buckets
                        )));
                    }
                    let stamp = format!(
                        "path={} k={} n_bins={} n_runouts={} rows={}",
                        path,
                        lookup.n_buckets,
                        lookup.n_bins,
                        lookup.n_runouts,
                        lookup.rows.len()
                    );
                    (Some(Arc::new(lookup)), stamp)
                }
                None => (None, "none".to_string()),
            };
            let (river_ochs, river_stamp) = match river_ochs_path {
                Some(path) => {
                    let table = crate::river_ochs::RiverOchs::load_path(&path)
                        .map_err(pyo3::exceptions::PyValueError::new_err)?;
                    let stamp = format!("path={} k={} n_clusters={}", path, table.n_buckets, 8);
                    (Some(Arc::new(table)), stamp)
                }
                None => (None, "none".to_string()),
            };
            // HOGWILD enable = explicit constructor flag OR env `POKERBOT_HOGWILD=1` (matching the env-var
            // toggle convention used elsewhere, e.g. POKERBOT_AB_STUB). Only honored when the crate was
            // built with `--features hogwild`; without the feature the path does not exist and the flag is
            // a hard error (so a caller never silently gets the fork-merge path when they asked for hogwild).
            let _hogwild_requested = hogwild
                || std::env::var("POKERBOT_HOGWILD")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
            #[cfg(not(feature = "hogwild"))]
            if _hogwild_requested {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "hogwild requested but battery_native was not built with --features hogwild",
                ));
            }
            let _ = hogwild_capacity; // silence unused warning when the feature is off

            Ok(Self {
                tab: nlhe_trainer::NlheTables::new(),
                cfg: nlhe_trainer::TrainerCfg {
                    num_buckets,
                    depth_edges,
                    // Arm-3 menu envs (absent => legacy, byte-identical): POKERBOT_PRE_KEEP /
                    // POKERBOT_POST_KEEP; the trainer also reads POKERBOT_PRE_RR_KEEP /
                    // POKERBOT_FLOP_RR_KEEP inside abstracted_legal (trainer.rs docs).
                    allowed_pre: nlhe_trainer::keep_mask_from_env("POKERBOT_PRE_KEEP", &[4, 5, 6]),
                    allowed_post: nlhe_trainer::keep_mask_from_env(
                        "POKERBOT_POST_KEEP",
                        &[0, 1, 3, 4, 5],
                    ),
                    n_deals_per_iter,
                    base_seed: 0xC0FFEE,
                    n_seats,
                    prune_threshold,
                    fast_merge,
                    emd_frozen,
                    batch_deals,
                    work_steal,
                    dcfr,
                    dcfr_prune_threshold,
                    dcfr_explore_frac,
                    river_ochs,
                },
                emd_stamp,
                river_stamp,
                #[cfg(feature = "hogwild")]
                hog: if _hogwild_requested {
                    Some(HogState {
                        table: crate::hogwild::SharedHogwildTable::new(hogwild_capacity),
                        iters_done: 0,
                    })
                } else {
                    None
                },
            })
        }

        fn iters_done(&self) -> u64 {
            #[cfg(feature = "hogwild")]
            if let Some(h) = &self.hog {
                return h.iters_done;
            }
            self.tab.iters_done
        }

        fn n_infosets(&self) -> usize {
            #[cfg(feature = "hogwild")]
            if let Some(h) = &self.hog {
                // O(capacity) scan of the shared table's occupancy — matches the served blueprint size
                // (one strat row per claimed infoset). Cheap relative to a train step; only called for
                // telemetry.
                return h.table.n_occupied();
            }
            self.tab.n_strat_infosets()
        }

        /// blueprint-v4 v3: cum_regret row count. Under POKERBOT_SNAPSHOT_AVG the strat map is
        /// preflop-bounded while cum keeps growing, so RAM backstops must watch
        /// max(n_infosets, n_cum_infosets). (Hogwild path: the shared table has one row kind, so
        /// n_infosets already covers it — return the same count.)
        fn n_cum_infosets(&self) -> usize {
            #[cfg(feature = "hogwild")]
            if let Some(h) = &self.hog {
                return h.table.n_occupied();
            }
            self.tab.n_cum_infosets()
        }

        /// True if this trainer instance is running the hogwild (shared-table) path. False = fork-merge.
        fn is_hogwild(&self) -> bool {
            #[cfg(feature = "hogwild")]
            {
                return self.hog.is_some();
            }
            #[cfg(not(feature = "hogwild"))]
            false
        }

        fn emd_stamp(&self) -> String {
            self.emd_stamp.clone()
        }

        /// Provenance stamp of the river OCHS table this trainer keys the river with ("none" =
        /// the made-hand mirror river of the shipped trainer).
        fn river_ochs_stamp(&self) -> String {
            self.river_stamp.clone()
        }

        fn train(&mut self, py: Python<'_>, iterations: u64) -> PyResult<()> {
            #[cfg(feature = "hogwild")]
            if let Some(h) = &mut self.hog {
                let cfg = &self.cfg;
                let table = &h.table;
                let start = h.iters_done;
                let res = py.detach(|| {
                    for k in 0..iterations {
                        let t = start + k + 1;
                        nlhe_trainer::train_iteration_hogwild_dispatch(table, cfg, t)?;
                    }
                    Ok::<(), String>(())
                });
                match res {
                    Ok(()) => {
                        h.iters_done += iterations;
                        return Ok(());
                    }
                    Err(e) => return Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
                }
            }
            py.detach(|| {
                for _ in 0..iterations {
                    self.tab.train_iteration(&self.cfg);
                }
            });
            Ok(())
        }

        /// Train for `iterations` and return JSON phase timings for each iteration.
        ///
        /// Profiling is intentionally coarse and opt-in so normal training keeps the same code path and
        /// output semantics. Each element reports traversal/collect, merge, and CFR+ floor wall time.
        fn train_profile_json(&mut self, py: Python<'_>, iterations: u64) -> String {
            py.detach(|| {
                let mut out = String::from("[");
                for i in 0..iterations {
                    let p = self.tab.train_iteration_profile(&self.cfg);
                    if i > 0 {
                        out.push(',');
                    }
                    let _ = write!(
                        out,
                        "{{\"iters_done\":{},\"n_deals\":{},\"rayon_threads\":{},\"fast_merge\":{},\
                          \"partials\":{},\"partial_cum_rows\":{},\"partial_strat_rows\":{},\
                          \"cum_regret_rows\":{},\"strat_sum_rows\":{},\"traverse_ms\":{:.6},\
                          \"merge_ms\":{:.6},\"floor_ms\":{:.6},\"total_ms\":{:.6},\
                          \"nodes\":{},\"terminal_nodes\":{},\"chance_nodes\":{},\
                          \"decision_nodes\":{},\"hero_nodes\":{},\"opponent_nodes\":{},\
                          \"no_legal_nodes\":{},\"legal_actions_total\":{},\
                          \"hero_branches\":{},\"opponent_samples\":{},\"apply_calls\":{},\
                          \"terminal_settles\":{},\"fingerprint_calls\":{},\
                          \"fp_memo_new_entries\":{},\"emd_bucket_queries\":{},\
                          \"emd_bucket_hits\":{},\"emd_bucket_misses\":{},\
                          \"emd_bucket_fallbacks\":{},\"emd_fp_memo_hits\":{},\
                          \"emd_fp_memo_misses\":{},\"max_depth\":{}}}",
                        p.iters_done,
                        p.n_deals,
                        p.rayon_threads,
                        p.fast_merge,
                        p.partials,
                        p.partial_cum_rows,
                        p.partial_strat_rows,
                        p.cum_regret_rows,
                        p.strat_sum_rows,
                        p.traverse_ms,
                        p.merge_ms,
                        p.floor_ms,
                        p.total_ms,
                        p.stats.nodes,
                        p.stats.terminal_nodes,
                        p.stats.chance_nodes,
                        p.stats.decision_nodes,
                        p.stats.hero_nodes,
                        p.stats.opponent_nodes,
                        p.stats.no_legal_nodes,
                        p.stats.legal_actions_total,
                        p.stats.hero_branches,
                        p.stats.opponent_samples,
                        p.stats.apply_calls,
                        p.stats.terminal_settles,
                        p.stats.fingerprint_calls,
                        p.stats.fp_memo_new_entries,
                        p.stats.emd_bucket_queries,
                        p.stats.emd_bucket_hits,
                        p.stats.emd_bucket_misses,
                        p.stats.emd_bucket_fallbacks,
                        p.stats.emd_fp_memo_hits,
                        p.stats.emd_fp_memo_misses,
                        p.stats.max_depth,
                    );
                }
                out.push(']');
                out
            })
        }

        fn save_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
            // Hogwild: materialize the shared table into the fork-merge `NlheTables` shape FIRST, then use
            // the UNCHANGED `serialize` — so a hogwild checkpoint is byte-format-identical to a fork-merge
            // one (same NLCFR001 blob layout) and resumes on either path / serves identically.
            #[cfg(feature = "hogwild")]
            if let Some(h) = &self.hog {
                let tab = h.table.snapshot_tables(h.iters_done);
                return PyBytes::new(py, &nlhe_trainer::serialize(&tab));
            }
            PyBytes::new(py, &nlhe_trainer::serialize(&self.tab))
        }

        /// Serialize the checkpoint DIRECTLY to `path` (Rust file write under `py.detach()`), returning
        /// the number of bytes written. Byte-identical to writing `save_bytes()` to disk, but WITHOUT the
        /// intermediate ~58 GB PyBytes copy — the driver's per-checkpoint pause and RAM peak both drop by
        /// one full blob (checkpoint-speed ops, 2026-07-06). The caller keeps `.tmp`+rename atomicity and
        /// the marker/upload handoff; this only replaces `blob = save_bytes(); f.write(blob)` with a single
        /// in-Rust serialize+write. Hogwild materializes to the fork-merge shape first (same as save_bytes).
        fn save_bytes_to_file(&self, py: Python<'_>, path: &str) -> PyResult<usize> {
            #[cfg(feature = "hogwild")]
            if let Some(h) = &self.hog {
                let tab = h.table.snapshot_tables(h.iters_done);
                return py
                    .detach(|| nlhe_trainer::serialize_to_file(&tab, path))
                    .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()));
            }
            let tab = &self.tab;
            py.detach(|| nlhe_trainer::serialize_to_file(tab, path))
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        // ---- DELTA CHECKPOINTS (checkpoint-speed R2, 2026-07-06) --------------------------------------
        // Fork-merge-only (same posture as the v3 snapshot modes): the delta dirty-set is populated in the
        // fork-merge `merge_partials{,_clamped}` path. Arm ONCE at startup; then per checkpoint the driver
        // calls `save_delta_to_file` (or `save_bytes_to_file` for a full base) and `reset_dirty`.

        /// Arm delta-checkpoint dirty tracking (idempotent). Errors on the hogwild path (delta mode is
        /// fork-merge only). A straight run that never arms pays nothing and is byte-identical.
        fn arm_delta_tracking(&mut self) -> PyResult<()> {
            #[cfg(feature = "hogwild")]
            if self.hog.is_some() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "arm_delta_tracking: unsupported on the hogwild path",
                ));
            }
            self.tab.arm_delta_tracking();
            Ok(())
        }

        /// Serialize the DIRTY rows since the last checkpoint as an `NLDLT001` delta straight to `path`
        /// (Rust write under `py.detach()`), rooted at `base_iter` with sequence `seq`. Returns the byte
        /// count, or None if delta tracking is unarmed OR the window touched the whole table
        /// (`dirty_is_full` after a discount pass) — in either case the caller must write a full base
        /// instead. The delta does NOT reset the dirty set (the driver calls `reset_dirty` only after the
        /// upload handoff, so a failed/partial upload leaves the window intact for the next attempt).
        fn save_delta_to_file(
            &self,
            py: Python<'_>,
            path: &str,
            base_iter: u64,
            seq: u64,
        ) -> PyResult<Option<usize>> {
            #[cfg(feature = "hogwild")]
            if self.hog.is_some() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "save_delta_to_file: unsupported on the hogwild path",
                ));
            }
            if !self.tab.delta_armed() || self.tab.dirty_is_full() {
                return Ok(None);
            }
            let tab = &self.tab;
            py.detach(|| nlhe_trainer::serialize_delta_to_file(tab, path, base_iter, seq))
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        /// Apply an `NLDLT001` delta blob onto the in-memory table (resume path). Chain-verified: the
        /// delta's base_iter must equal `base_iter` and its seq must be `prev_seq + 1` and its iters_done
        /// must advance. Returns `(iters_done, seq)` on success, None on any mismatch (fail-closed).
        fn apply_delta(
            &mut self,
            blob: &[u8],
            base_iter: u64,
            prev_seq: u64,
        ) -> PyResult<Option<(u64, u64)>> {
            #[cfg(feature = "hogwild")]
            if self.hog.is_some() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "apply_delta: unsupported on the hogwild path",
                ));
            }
            Ok(nlhe_trainer::apply_delta_checked(
                &mut self.tab,
                blob,
                base_iter,
                prev_seq,
            ))
        }

        /// Read the `(base_iter, iters_done, seq)` header of a delta blob without applying it (cheap;
        /// 32-byte parse). None on bad magic / truncation. Lets the driver validate chain contiguity
        /// before downloading a full delta.
        #[staticmethod]
        fn delta_header(blob: &[u8]) -> Option<(u64, u64, u64)> {
            nlhe_trainer::delta_header(blob)
        }

        /// Clear the dirty set at a checkpoint boundary (retains capacity). No-op when unarmed. Called
        /// AFTER a delta/base checkpoint's rows are captured AND handed to the uploader.
        fn reset_dirty(&mut self) {
            self.tab.reset_dirty();
        }

        /// `(dirty_cum_rows, dirty_strat_rows)` accumulated since the last `reset_dirty` — telemetry so
        /// the driver can log delta sparsity and decide base-vs-delta. (0,0) when unarmed.
        fn dirty_counts(&self) -> (usize, usize) {
            self.tab.dirty_counts()
        }

        /// blueprint-v4 v3 (snapshot averaging): serialize the CURRENT strategy (positive-part-
        /// normalized cum_regret) as an NLSNAP01 blob — the rounds-2-4 snapshot the offline
        /// `snap_avg` averager consumes (`trainer::serialize_current_strategy`; Pluribus P-supp
        /// p.15). The cloud driver uploads one next to each checkpoint when POKERBOT_SNAPSHOT_AVG=1.
        /// Not supported on the hogwild path (the v3 modes are fork-merge-only).
        fn save_current_strategy_bytes<'py>(
            &self,
            py: Python<'py>,
        ) -> PyResult<Bound<'py, PyBytes>> {
            #[cfg(feature = "hogwild")]
            if self.hog.is_some() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "save_current_strategy_bytes: unsupported on the hogwild path",
                ));
            }
            Ok(PyBytes::new(py, &nlhe_trainer::serialize_current_strategy(&self.tab)))
        }

        /// blueprint-v4 v3 (coverage as a first-class metric): ON-POLICY coverage probe — play
        /// `n_hands` self-play hands on the CURRENT strategy and count, per street, how many of the
        /// strategy's own lookups hit the trained table (`trainer::coverage_probe`; the Stage-0
        /// §3.1 on-policy diagnostic at checkpoint cadence). Returns
        /// `(hands, decisions[4], misses[4], preflop_allin_runouts)`, streets ordered
        /// preflop/flop/turn/river. Read-only; deterministic in (table, seed).
        #[pyo3(signature = (n_hands=1024, seed=0xC0FFEE))]
        fn coverage(
            &self,
            py: Python<'_>,
            n_hands: u64,
            seed: u64,
        ) -> PyResult<(u64, Vec<u64>, Vec<u64>, u64)> {
            #[cfg(feature = "hogwild")]
            if self.hog.is_some() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "coverage: unsupported on the hogwild path",
                ));
            }
            let (tab, cfg) = (&self.tab, &self.cfg);
            let c = py.detach(|| nlhe_trainer::coverage_probe(tab, cfg, n_hands, seed));
            Ok((c.hands, c.decisions.to_vec(), c.misses.to_vec(), c.preflop_allin_runouts))
        }

        /// blueprint-v4 v3 (hard-prune-C calibration): one-pass log-histogram percentiles of the
        /// stored regrets (`trainer::regret_stats`). Returns
        /// `(rows, neg_vals, pos_vals, neg_p50, neg_p90, neg_p99, neg_p999, neg_max, pos_p99,
        /// pos_max)` — negative-tail magnitudes in bb. Read-only.
        #[allow(clippy::type_complexity)]
        fn regret_stats(
            &self,
            py: Python<'_>,
        ) -> PyResult<(u64, u64, u64, f64, f64, f64, f64, f64, f64, f64)> {
            #[cfg(feature = "hogwild")]
            if self.hog.is_some() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "regret_stats: unsupported on the hogwild path",
                ));
            }
            let tab = &self.tab;
            let s = py.detach(|| nlhe_trainer::regret_stats(tab));
            Ok((
                s.rows, s.neg_vals, s.pos_vals, s.neg_p50, s.neg_p90, s.neg_p99, s.neg_p999,
                s.neg_max, s.pos_p99, s.pos_max,
            ))
        }

        fn load_bytes(&mut self, py: Python<'_>, blob: &[u8]) -> PyResult<bool> {
            // RESUME-SPEED path (2026-07-06): parallel hash-partition + pre-sized sharded insert under
            // `py.detach()` (GIL released) — semantically identical to the serial `deserialize` (same
            // shards/rows/iters_done; re-serializes byte-for-byte identically) but the deserialize is no
            // longer a single-threaded rehash-storm. The blob is already fully in RAM here (Python owns
            // the `bytes`); `load_bytes_from_file` avoids even that copy.
            let parsed = py.detach(|| nlhe_trainer::deserialize_parallel(blob));
            self.install_loaded_tables(parsed)
        }

        /// Resume DIRECTLY from a local checkpoint file: memory-map the `.nlcfr` file and parallel-
        /// deserialize it without materializing a Python `bytes` copy of the whole blob first. Pairs with
        /// the download path that already stages the checkpoint on the local ckpt volume — the driver can
        /// hand this the on-disk path instead of `download -> read() -> load_bytes(blob)` (which builds a
        /// full ~55-100 GB Python bytes object and holds the GIL). Returns True on a successful load, False
        /// on a bad-magic/truncated blob (same fail-closed contract as `load_bytes`).
        fn load_bytes_from_file(&mut self, py: Python<'_>, path: &str) -> PyResult<bool> {
            let parsed = py
                .detach(|| -> std::io::Result<Option<nlhe_trainer::NlheTables>> {
                    let f = std::fs::File::open(path)?;
                    // SAFETY: the checkpoint file is written once + renamed atomically and is not mutated
                    // while we hold it open for read; the mapping is read-only and dropped before return.
                    let mmap = unsafe { memmap2::Mmap::map(&f)? };
                    Ok(nlhe_trainer::deserialize_parallel(&mmap))
                })
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
            self.install_loaded_tables(parsed)
        }

        fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let out = PyDict::new(py);
            #[cfg(feature = "hogwild")]
            if let Some(h) = &self.hog {
                let tab = h.table.snapshot_tables(h.iters_done);
                for (key, row) in nlhe_trainer::snapshot(&tab) {
                    out.set_item(PyBytes::new(py, &key), row.to_vec())?;
                }
                return Ok(out);
            }
            for (key, row) in nlhe_trainer::snapshot(&self.tab) {
                out.set_item(PyBytes::new(py, &key), row.to_vec())?;
            }
            Ok(out)
        }
    }

    // Non-Python helpers on the trainer (a plain `impl`, kept OUT of `#[pymethods]` because it takes the
    // internal `NlheTables` type). Shared by `load_bytes` and `load_bytes_from_file`.
    impl NlheCFRTrainer {
        /// Install a freshly-deserialized table (or None on a bad/truncated blob) into the trainer,
        /// preserving the hogwild-vs-fork-merge branch that both load entrypoints previously duplicated.
        /// Returns Ok(false) when `parsed` is None (fail-closed: the caller reports "not resumable").
        fn install_loaded_tables(
            &mut self,
            parsed: Option<nlhe_trainer::NlheTables>,
        ) -> PyResult<bool> {
            match parsed {
                Some(tab) => {
                    #[cfg(feature = "hogwild")]
                    if let Some(h) = &mut self.hog {
                        // Resume a hogwild run from a checkpoint: rebuild a fresh shared table (capacity
                        // preserved) and load every row into it. A capacity-overflow here is a hard error.
                        let cap = h.table.capacity();
                        let table = crate::hogwild::SharedHogwildTable::new(cap);
                        table
                            .load_from_tables(&tab)
                            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
                        h.table = table;
                        h.iters_done = tab.iters_done;
                        return Ok(true);
                    }
                    self.tab = tab;
                    Ok(true)
                }
                None => Ok(false),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Native Leduc CFR+ trainer (Workstream F: the "native Rust trainer" acceptance, tabular path).
    // The whole train -> exact infoset-BR exploitability -> checkpoint/resume loop runs natively.
    // Equivalence-gated vs the Python reference (leduc_tabular_cfr.py) by tests/test_native_leduc_cfr.py.
    // -----------------------------------------------------------------------
    use crate::leduc;

    /// Durable native Leduc CFR+ trainer. Holds the regret/strategy tables; `train` continues from the
    /// last iteration, `exploitability` returns the EXACT infoset-BR number (mbb), and `save_bytes`/
    /// `load_bytes` give atomic checkpoint/resume (Python writes the bytes to disk via os.replace).
    #[pyclass]
    struct LeducCFRTrainer {
        tab: leduc::Tables,
    }

    #[pymethods]
    impl LeducCFRTrainer {
        #[new]
        fn new() -> Self {
            LeducCFRTrainer {
                tab: leduc::Tables::default(),
            }
        }

        /// Run `iterations` CFR+ sweeps (continuing from where the trainer left off).
        fn train(&mut self, py: Python<'_>, iterations: u64) {
            py.detach(|| leduc::train(&mut self.tab, iterations));
        }

        /// Iterations completed so far.
        fn iters_done(&self) -> u64 {
            self.tab.iters_done
        }

        /// Exact information-set BR exploitability (mbb) of the current average strategy:
        /// (expl_mbb, u0, u1, br0, br1). Matches `leduc_tabular_cfr.infoset_exploitability_mbb`.
        fn exploitability(&self, py: Python<'_>) -> (f64, f64, f64, f64, f64) {
            py.detach(|| {
                let avg = leduc::average_strategy(&self.tab);
                leduc::infoset_exploitability_mbb(&avg)
            })
        }

        /// Serve the average strategy at one (own_card, public, round, to_call, raises, hist, seat).
        /// Cross-check entry: returns the 3-action distribution for the matching infoset (uniform-over-
        /// legal if unseen). `hist` is the action sequence (ints in 0..3).
        #[pyo3(signature = (seat, own_card, public, round, to_call, raises, hist))]
        fn serve(
            &self,
            seat: i64,
            own_card: i64,
            public: i64,
            round: i64,
            to_call: i64,
            raises: i64,
            hist: Vec<i64>,
        ) -> Vec<f64> {
            let mut key: Vec<i64> = vec![
                seat,
                own_card,
                public,
                round,
                to_call,
                raises,
                hist.len() as i64,
            ];
            key.extend_from_slice(&hist);
            let avg = leduc::average_strategy(&self.tab);
            match avg.get(&key) {
                Some(row) => row.to_vec(),
                None => {
                    // uniform over the 2 (preflop) / variable legal -- caller masks; return uniform-3.
                    vec![1.0 / 3.0; leduc::NUM_ACTIONS]
                }
            }
        }

        /// Atomic-checkpoint payload: a versioned byte blob of the regret/strategy tables. Python writes
        /// it to a temp file and `os.replace`s it onto the checkpoint path (atomic on the same volume).
        fn save_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
            PyBytes::new(py, &leduc::serialize(&self.tab))
        }

        /// Resume from a `save_bytes` blob: rebuilds the tables and re-derives the per-sweep strategy so
        /// continuing the solve reproduces a straight-through run. Returns True on success.
        fn load_bytes(&mut self, blob: &[u8]) -> bool {
            match leduc::deserialize(blob) {
                Some(t) => {
                    self.tab = t;
                    true
                }
                None => false,
            }
        }
    }

    /// One-shot: train from scratch for `iterations` and return exploitability — the gate entry point.
    #[pyfunction]
    fn leduc_solve_exploitability(py: Python<'_>, iterations: u64) -> (f64, f64, f64, f64, f64) {
        py.detach(|| leduc::solve_and_exploitability(iterations))
    }

    /// `icm_equity(stacks, payouts)` -> per-seat $-equity (Malmuth-Harville). Pool-conserving.
    /// Cross-checked against `pokerbot.tournament.icm.icm_equity` by `tests/test_native_icm_equiv.py`.
    #[pyfunction]
    fn icm_equity(stacks: Vec<f64>, payouts: Vec<f64>) -> Vec<f64> {
        crate::icm::icm_equity(&stacks, &payouts)
    }

    /// `bubble_factor(stacks, payouts, hero, villain)` -> pairwise ICM risk premium (inf if hero
    /// cannot gain). Mirrors `pokerbot.tournament.icm.bubble_factor`.
    #[pyfunction]
    fn bubble_factor(stacks: Vec<f64>, payouts: Vec<f64>, hero: usize, villain: usize) -> f64 {
        crate::icm::bubble_factor(&stacks, &payouts, hero, villain)
    }

    /// `icm_delta(pre_stacks, chip_delta, payouts)` -> per-seat $-equity change (busts clamp at 0).
    /// Mirrors `pokerbot.tournament.icm.icm_delta`.
    #[pyfunction]
    fn icm_delta(pre_stacks: Vec<f64>, chip_delta: Vec<f64>, payouts: Vec<f64>) -> Vec<f64> {
        crate::icm::icm_delta(&pre_stacks, &chip_delta, &payouts)
    }

    // -----------------------------------------------------------------------
    // EMD postflop bucketer (potential-aware abstraction) — Workstream F.
    // Cross-checked by tests/test_native_emd_equiv.py:
    //   * emd_bucket_from_hist is BIT-IDENTICAL to numpy's argmin-of-L1-of-CDF;
    //   * emd_strength_histogram is STATISTICALLY EQUIVALENT (fast native RNG, not a PCG64 replay);
    //   * emd_fit is calibration-equivalent (independent init, deterministic per seed).
    // -----------------------------------------------------------------------

    /// `emd_strength_histogram(hole, board, n_bins, n_runouts, seed)` -> normalized length-`n_bins`
    /// strength-percentile histogram. `hole` must have length 2 (extra entries are ignored).
    #[pyfunction]
    fn emd_strength_histogram(
        hole: Vec<i32>,
        board: Vec<i32>,
        n_bins: usize,
        n_runouts: usize,
        seed: u64,
    ) -> PyResult<Vec<f64>> {
        if hole.len() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "hole must have 2 cards",
            ));
        }
        let h = [hole[0], hole[1]];
        Ok(crate::emd::strength_histogram(
            &h, &board, n_bins, n_runouts, seed,
        ))
    }

    /// `emd_bucket_from_hist(hist, centroids)` -> argmin centroid index (numpy-identical keep-first).
    #[pyfunction]
    fn emd_bucket_from_hist(hist: Vec<f64>, centroids: Vec<Vec<f64>>) -> usize {
        crate::emd::emd_bucket_from_hist(&hist, &centroids)
    }

    /// `emd_fit(histograms, num_buckets, iters, seed)` -> `num_buckets` centroids (k-means under EMD).
    #[pyfunction]
    fn emd_fit(
        histograms: Vec<Vec<f64>>,
        num_buckets: usize,
        iters: usize,
        seed: u64,
    ) -> Vec<Vec<f64>> {
        crate::emd::emd_fit(&histograms, num_buckets, iters, seed)
    }

    /// `emd_mean_inertia(histograms, centroids)` -> mean within-cluster EMD (clustering quality scalar).
    #[pyfunction]
    fn emd_mean_inertia(histograms: Vec<Vec<f64>>, centroids: Vec<Vec<f64>>) -> f64 {
        crate::emd::mean_inertia(&histograms, &centroids)
    }

    /// `emd_strength_histograms_batch(spots, n_bins, n_runouts, seed_base)` -> per-spot histograms,
    /// rayon-parallel, deterministic per index. Each spot is `(hole[2], board)`.
    #[pyfunction]
    fn emd_strength_histograms_batch(
        py: Python<'_>,
        spots: Vec<(Vec<i32>, Vec<i32>)>,
        n_bins: usize,
        n_runouts: usize,
        seed_base: u64,
    ) -> PyResult<Vec<Vec<f64>>> {
        let mut conv: Vec<([i32; 2], Vec<i32>)> = Vec::with_capacity(spots.len());
        for (hole, board) in spots {
            if hole.len() < 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "each spot hole must have 2 cards",
                ));
            }
            conv.push(([hole[0], hole[1]], board));
        }
        let out = py
            .detach(|| crate::emd::strength_histograms_batch(&conv, n_bins, n_runouts, seed_base));
        Ok(out)
    }

    /// Frozen flop/turn `(hole,board)->EMD bucket` lookup loaded from an EMDFRZN1 blob.
    #[pyclass]
    struct EmdFrozenLookup {
        inner: crate::emd::FrozenEMDLookup,
    }

    #[pymethods]
    impl EmdFrozenLookup {
        #[staticmethod]
        fn load(path: String) -> PyResult<Self> {
            let inner = crate::emd::FrozenEMDLookup::load_path(&path)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok(Self { inner })
        }

        fn bucket(&self, hole: Vec<i32>, board: Vec<i32>) -> i32 {
            self.inner.bucket(&hole, &board).unwrap_or(-1)
        }

        fn meta(&self) -> (usize, usize, usize, usize) {
            (
                self.inner.n_buckets,
                self.inner.n_bins,
                self.inner.n_runouts,
                self.inner.rows.len(),
            )
        }
    }

    /// `emd_frozen_key(hole, board)` -> canonical u64 key for tiny lookup smoke tests.
    #[pyfunction]
    fn emd_frozen_key(hole: Vec<i32>, board: Vec<i32>) -> Option<u64> {
        crate::emd::canonical_frozen_key(&hole, &board)
    }

    /// Build the full EMDFRZN1 frozen lookup. Heavy; callers should run this only on approved hardware.
    #[pyfunction]
    fn emd_frozen_build(
        py: Python<'_>,
        centroids: Vec<Vec<f64>>,
        out_path: String,
        n_bins: usize,
        n_runouts: usize,
    ) -> PyResult<(u64, String)> {
        py.detach(|| {
            crate::emd::build_frozen_lookup(&centroids, &out_path, n_bins, n_runouts)
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)
        })
    }

    /// Flat buffer length + engine dims, for the wrapper's startup self-check.
    #[pyfunction]
    fn engine_dims() -> (usize, usize, usize, usize, usize) {
        (
            NUM_PLAYERS,
            core::NUM_ACTIONS,
            MAX_BOARD,
            core::MAX_HISTORY,
            flat::FLAT_LEN,
        )
    }

    #[pyfunction]
    fn rayon_num_threads() -> usize {
        rayon::current_num_threads()
    }

    #[pymodule]
    fn battery_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(apply_action, m)?)?;
        m.add_function(wrap_pyfunction!(sample_chance, m)?)?;
        m.add_function(wrap_pyfunction!(init_hand, m)?)?;
        m.add_function(wrap_pyfunction!(legal_action_mask, m)?)?;
        m.add_function(wrap_pyfunction!(is_chance_node, m)?)?;
        m.add_function(wrap_pyfunction!(evaluate_seven, m)?)?;
        m.add_function(wrap_pyfunction!(compute_returns, m)?)?;
        m.add_function(wrap_pyfunction!(engine_dims, m)?)?;
        m.add_function(wrap_pyfunction!(rayon_num_threads, m)?)?;
        m.add_function(wrap_pyfunction!(fingerprint_fast, m)?)?;
        m.add_function(wrap_pyfunction!(fingerprint_fast_depth, m)?)?;
        m.add_function(wrap_pyfunction!(fingerprint_fast_emd, m)?)?;
        m.add_function(wrap_pyfunction!(native_caps, m)?)?;
        m.add_function(wrap_pyfunction!(eval_seven_phe, m)?)?;
        m.add_function(wrap_pyfunction!(subgame_ctx_key_id, m)?)?;
        m.add_function(wrap_pyfunction!(multiway_equity_river, m)?)?;
        m.add_function(wrap_pyfunction!(leduc_solve_exploitability, m)?)?;
        m.add_function(wrap_pyfunction!(icm_equity, m)?)?;
        m.add_function(wrap_pyfunction!(bubble_factor, m)?)?;
        m.add_function(wrap_pyfunction!(icm_delta, m)?)?;
        m.add_function(wrap_pyfunction!(emd_strength_histogram, m)?)?;
        m.add_function(wrap_pyfunction!(emd_bucket_from_hist, m)?)?;
        m.add_function(wrap_pyfunction!(emd_fit, m)?)?;
        m.add_function(wrap_pyfunction!(emd_mean_inertia, m)?)?;
        m.add_function(wrap_pyfunction!(emd_strength_histograms_batch, m)?)?;
        m.add_function(wrap_pyfunction!(emd_frozen_key, m)?)?;
        m.add_function(wrap_pyfunction!(emd_frozen_build, m)?)?;
        m.add_class::<EmdFrozenLookup>()?;
        m.add_class::<PlayoutEngine>()?;
        m.add_class::<SubgameSolver>()?;
        m.add_class::<Lbr6Harness>()?;
        m.add_class::<LbrNeuralHarness>()?;
        m.add_class::<Lbr6Out>()?;
        m.add_class::<NlheCFRTrainer>()?;
        m.add_class::<LeducCFRTrainer>()?;
        Ok(())
    }
}
