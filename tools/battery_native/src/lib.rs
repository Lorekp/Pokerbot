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
    use crate::core::{self, MAX_BOARD, NUM_PLAYERS};
    use crate::flat;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

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
        m.add_function(wrap_pyfunction!(legal_action_mask, m)?)?;
        m.add_function(wrap_pyfunction!(is_chance_node, m)?)?;
        m.add_function(wrap_pyfunction!(evaluate_seven, m)?)?;
        m.add_function(wrap_pyfunction!(compute_returns, m)?)?;
        m.add_function(wrap_pyfunction!(engine_dims, m)?)?;
        Ok(())
    }
}
