//! `battery_native` — bit-identical native core for the exploitation battery hot path.
//!
//! The `core` module is pure Rust (testable with `cargo test`, no Python). The PyO3 bindings below are
//! compiled only with `--features python` (the cdylib loaded by `pokerbot/search/native_solver.py`).
//!
//! State is marshaled as a single flat `Vec<i64>` (fixed offset layout, see FLAT_LEN below) to avoid
//! PyO3's 12-element tuple limit and to keep marshaling a single contiguous copy. The wrapper
//! (`native_solver.py`) builds the flat buffer from a numpy EnvState once per call; everything between
//! is native.

pub mod core;
pub mod emd;
pub mod fingerprint;
pub mod icm;
pub mod leduc;
pub mod lbr;
pub mod nnet;
pub mod phe_tables;
pub mod playout;
pub mod subgame;

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
    use pyo3::prelude::*;
    use pyo3::types::{PyBytes, PyDict};
    use std::collections::HashMap;

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
            return Err(pyo3::exceptions::PyValueError::new_err("deck must have length 52"));
        }
        if rng_key.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("rng_key must have length 2"));
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
        #[new]
        fn new(
            blueprint: &Bound<'_, PyDict>,
            allowed_pre: Vec<bool>,
            allowed_post: Vec<bool>,
            bias_mult: f64,
            bias_floor: f64,
            max_depth: i32,
            num_buckets: i32,
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
        #[new]
        fn new(
            blueprint: &Bound<'_, PyDict>,
            allowed_pre: Vec<bool>,
            allowed_post: Vec<bool>,
            bias_mult: f64,
            bias_floor: f64,
            leaf_max_depth: i32,
            num_buckets: i32,
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
        ) -> Vec<f64> {
            let s = flat::from_flat(&bytes_to_flat(state));
            let scfg = SubgameConfig {
                max_iterations,
                max_depth,
                blueprint_mix,
                rollout_samples,
                resample_board,
                linear_averaging,
            };
            subgame::solve(&s, hero, &self.bp, &self.cfg, &scfg, public_seed).to_vec()
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
                    ((a.sumsq - a.sum * a.sum / nf) / (nf - 1.0)).max(0.0).sqrt()
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
    }

    #[pymethods]
    impl Lbr6Harness {
        /// blueprint: dict[bytes(16) -> seq[float] len 11] (may be EMPTY for calibration modes).
        /// `depth_edges`: depth-bucket edges for a DEPTH-BUCKETED blueprint (e.g. [50,90,140]); pass an
        /// empty list for a non-depth blueprint -- the preflop key then omits the depth byte.
        #[new]
        fn new(
            blueprint: &Bound<'_, PyDict>,
            allowed_pre: Vec<bool>,
            allowed_post: Vec<bool>,
            num_buckets: i32,
            depth_edges: Vec<f64>,
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
            })
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
            w0: Vec<f32>, b0: Vec<f32>,
            w1: Vec<f32>, b1: Vec<f32>,
            w2: Vec<f32>, b2: Vec<f32>,
            din: Vec<usize>, dout: Vec<usize>,
            info_suits: bool,
        ) -> PyResult<Self> {
            if din.len() != 3 || dout.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err("din/dout must have length 3"));
            }
            let check = |w: &[f32], b: &[f32], i: usize, o: usize, l: usize| -> PyResult<()> {
                if w.len() != i * o || b.len() != o {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "layer {l}: expected w={}*{}, b={}; got w={}, b={}",
                        i, o, o, w.len(), b.len()
                    )));
                }
                Ok(())
            };
            check(&w0, &b0, din[0], dout[0], 0)?;
            check(&w1, &b1, din[1], dout[1], 1)?;
            check(&w2, &b2, din[2], dout[2], 2)?;
            Ok(LbrNeuralHarness {
                net: Net::new(w0, b0, w1, b1, w2, b2, [din[0], din[1], din[2]],
                              [dout[0], dout[1], dout[2]], info_suits),
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
                n_seats, track_ranges, range_cap, eq_samples, eq_runouts, settle_samples,
                max_depth, bet_candidates, value_only_raises,
            };
            let states = roots_from_blob(roots, n);
            let outcome = py.detach(|| lbr::run(&states, &lbr_seats, &seeds, &self.net, &cfg));
            Ok(pack_outcome(outcome))
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
            LeducCFRTrainer { tab: leduc::Tables::default() }
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
            let mut key: Vec<i64> =
                vec![seat, own_card, public, round, to_call, raises, hist.len() as i64];
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
            return Err(pyo3::exceptions::PyValueError::new_err("hole must have 2 cards"));
        }
        let h = [hole[0], hole[1]];
        Ok(crate::emd::strength_histogram(&h, &board, n_bins, n_runouts, seed))
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
                return Err(pyo3::exceptions::PyValueError::new_err("each spot hole must have 2 cards"));
            }
            conv.push(([hole[0], hole[1]], board));
        }
        let out = py.detach(|| {
            crate::emd::strength_histograms_batch(&conv, n_bins, n_runouts, seed_base)
        });
        Ok(out)
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
        m.add_function(wrap_pyfunction!(fingerprint_fast, m)?)?;
        m.add_function(wrap_pyfunction!(fingerprint_fast_depth, m)?)?;
        m.add_function(wrap_pyfunction!(eval_seven_phe, m)?)?;
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
        m.add_class::<PlayoutEngine>()?;
        m.add_class::<SubgameSolver>()?;
        m.add_class::<Lbr6Harness>()?;
        m.add_class::<LbrNeuralHarness>()?;
        m.add_class::<Lbr6Out>()?;
        m.add_class::<LeducCFRTrainer>()?;
        Ok(())
    }
}
