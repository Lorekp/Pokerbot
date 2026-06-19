//! `fingerprint_native` — bit-identical native port of the blueprint infoset fingerprint hot path.
//!
//! `core` (+ `phe`, `blake2b`, `tables`) is pure Rust (testable with `cargo test`, no Python). The
//! PyO3 bindings below compile only with `--features python` (the `extension-module` cdylib loaded by
//! `pokerbot/encoding/fingerprint_native.py`). Output is byte-for-byte identical to
//! `pokerbot.engine.jax_weva_fast.fingerprint_fast(state, seat)` in betting-abstraction mode.

pub mod blake2b;
pub mod core;
pub mod phe;
pub mod tables;

#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
mod py {
    use crate::core::{self, FlatState, FpConfig};
    use pyo3::buffer::PyBuffer;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    /// Borrow a read-only C-contiguous i32 numpy array as `&[i32]` (zero-copy).
    unsafe fn ro_i32<'py>(
        obj: &Bound<'py, PyAny>,
        min_len: usize,
        name: &str,
    ) -> PyResult<(&'py [i32], PyBuffer<i32>)> {
        let buf = PyBuffer::<i32>::get(obj)?;
        if !buf.is_c_contiguous() {
            return Err(PyValueError::new_err(format!("{name} must be C-contiguous int32")));
        }
        if buf.item_count() < min_len {
            return Err(PyValueError::new_err(format!(
                "{name} length {} < expected >= {}",
                buf.item_count(),
                min_len
            )));
        }
        let ptr = buf.buf_ptr() as *const i32;
        let s = std::slice::from_raw_parts(ptr, buf.item_count());
        Ok((s, buf))
    }

    /// Compute the 16-byte fingerprint for one (flat-state, seat). All EnvState fields are passed as
    /// plain C-contiguous int32 arrays + scalars (the Python wrapper does the cheap extraction). The
    /// `weva_table` is the (52,52,6) i32 default table flattened row-major.
    #[pyfunction]
    #[pyo3(signature = (
        holes, board, seat_status, stacks, committed_street,
        street, pot_dead, current_bet, button, last_aggressor,
        hist_seat, hist_action, hist_len, seat,
        weva_table, preflop_lossless, preflop_lossless_betting,
        postflop_buckets_enabled, postflop_num_buckets,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fingerprint<'py>(
        py: Python<'py>,
        holes: &Bound<'py, PyAny>,
        board: &Bound<'py, PyAny>,
        seat_status: &Bound<'py, PyAny>,
        stacks: &Bound<'py, PyAny>,
        committed_street: &Bound<'py, PyAny>,
        street: i32,
        pot_dead: i32,
        current_bet: i32,
        button: i32,
        last_aggressor: i32,
        hist_seat: &Bound<'py, PyAny>,
        hist_action: &Bound<'py, PyAny>,
        hist_len: i32,
        seat: usize,
        weva_table: &Bound<'py, PyAny>,
        preflop_lossless: bool,
        preflop_lossless_betting: bool,
        postflop_buckets_enabled: bool,
        postflop_num_buckets: i32,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let (holes_s, _h) = unsafe { ro_i32(holes, core::NUM_PLAYERS * 2, "holes")? };
        let (board_s, _b) = unsafe { ro_i32(board, 1, "board")? };
        let (status_s, _ss) = unsafe { ro_i32(seat_status, core::NUM_PLAYERS, "seat_status")? };
        let (stacks_s, _st) = unsafe { ro_i32(stacks, core::NUM_PLAYERS, "stacks")? };
        let (commit_s, _cs) = unsafe { ro_i32(committed_street, core::NUM_PLAYERS, "committed_street")? };
        let hl = hist_len.max(0) as usize;
        let (hs_s, _hs) = unsafe { ro_i32(hist_seat, hl, "hist_seat")? };
        let (ha_s, _ha) = unsafe { ro_i32(hist_action, hl, "hist_action")? };
        let (wt_s, _wt) = unsafe { ro_i32(weva_table, 52 * 52 * 6, "weva_table")? };

        let state = FlatState {
            holes: holes_s,
            board: board_s,
            seat_status: status_s,
            stacks: stacks_s,
            committed_street: commit_s,
            street,
            pot_dead,
            current_bet,
            button,
            last_aggressor,
            hist_seat: hs_s,
            hist_action: ha_s,
            hist_len,
        };
        let cfg = FpConfig {
            preflop_lossless,
            preflop_lossless_betting,
            postflop_buckets_enabled,
            postflop_num_buckets,
        };
        let out = core::fingerprint(&state, seat, wt_s, &cfg);
        Ok(PyBytes::new(py, &out))
    }

    /// Batched fingerprint: `b` states stacked along axis 0 in each field, one `seat` per state.
    /// Returns 16*b packed bytes (state i's digest at [16*i .. 16*i+16]). Amortizes the per-call FFI
    /// crossing across the whole batch — the realizable speedup for the search/LBR sweep.
    ///
    /// Strides (per state): holes 12, board 5, seat_status/stacks/committed 6, hist_seat/hist_action
    /// = hist_cap (passed explicitly; the EnvState history capacity, e.g. 64).
    #[pyfunction]
    #[pyo3(signature = (
        b, holes, board, seat_status, stacks, committed_street,
        street, pot_dead, current_bet, button, last_aggressor,
        hist_seat, hist_action, hist_len, hist_cap, seats,
        weva_table, preflop_lossless, preflop_lossless_betting,
        postflop_buckets_enabled, postflop_num_buckets,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fingerprint_batch<'py>(
        py: Python<'py>,
        b: usize,
        holes: &Bound<'py, PyAny>,
        board: &Bound<'py, PyAny>,
        seat_status: &Bound<'py, PyAny>,
        stacks: &Bound<'py, PyAny>,
        committed_street: &Bound<'py, PyAny>,
        street: &Bound<'py, PyAny>,
        pot_dead: &Bound<'py, PyAny>,
        current_bet: &Bound<'py, PyAny>,
        button: &Bound<'py, PyAny>,
        last_aggressor: &Bound<'py, PyAny>,
        hist_seat: &Bound<'py, PyAny>,
        hist_action: &Bound<'py, PyAny>,
        hist_len: &Bound<'py, PyAny>,
        hist_cap: usize,
        seats: &Bound<'py, PyAny>,
        weva_table: &Bound<'py, PyAny>,
        preflop_lossless: bool,
        preflop_lossless_betting: bool,
        postflop_buckets_enabled: bool,
        postflop_num_buckets: i32,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let np = core::NUM_PLAYERS;
        let (holes_s, _h) = unsafe { ro_i32(holes, b * np * 2, "holes")? };
        let (board_s, _bd) = unsafe { ro_i32(board, b * 5, "board")? };
        let (status_s, _ss) = unsafe { ro_i32(seat_status, b * np, "seat_status")? };
        let (stacks_s, _st) = unsafe { ro_i32(stacks, b * np, "stacks")? };
        let (commit_s, _cs) = unsafe { ro_i32(committed_street, b * np, "committed_street")? };
        let (street_s, _str) = unsafe { ro_i32(street, b, "street")? };
        let (pot_s, _pot) = unsafe { ro_i32(pot_dead, b, "pot_dead")? };
        let (cur_s, _cur) = unsafe { ro_i32(current_bet, b, "current_bet")? };
        let (btn_s, _btn) = unsafe { ro_i32(button, b, "button")? };
        let (agg_s, _agg) = unsafe { ro_i32(last_aggressor, b, "last_aggressor")? };
        let (hs_s, _hs) = unsafe { ro_i32(hist_seat, b * hist_cap, "hist_seat")? };
        let (ha_s, _ha) = unsafe { ro_i32(hist_action, b * hist_cap, "hist_action")? };
        let (hl_s, _hl) = unsafe { ro_i32(hist_len, b, "hist_len")? };
        let (seats_s, _se) = unsafe { ro_i32(seats, b, "seats")? };
        let (wt_s, _wt) = unsafe { ro_i32(weva_table, 52 * 52 * 6, "weva_table")? };

        let cfg = FpConfig {
            preflop_lossless,
            preflop_lossless_betting,
            postflop_buckets_enabled,
            postflop_num_buckets,
        };
        let mut out = vec![0u8; b * 16];
        for i in 0..b {
            let state = FlatState {
                holes: &holes_s[i * np * 2..i * np * 2 + np * 2],
                board: &board_s[i * 5..i * 5 + 5],
                seat_status: &status_s[i * np..i * np + np],
                stacks: &stacks_s[i * np..i * np + np],
                committed_street: &commit_s[i * np..i * np + np],
                street: street_s[i],
                pot_dead: pot_s[i],
                current_bet: cur_s[i],
                button: btn_s[i],
                last_aggressor: agg_s[i],
                hist_seat: &hs_s[i * hist_cap..i * hist_cap + hist_cap],
                hist_action: &ha_s[i * hist_cap..i * hist_cap + hist_cap],
                hist_len: hl_s[i],
            };
            let fp = core::fingerprint(&state, seats_s[i] as usize, wt_s, &cfg);
            out[i * 16..i * 16 + 16].copy_from_slice(&fp);
        }
        Ok(PyBytes::new(py, &out))
    }

    /// Standalone phevaluator rank (test/verify hook). `cards` = encoder ids, 5..7 of them.
    #[pyfunction]
    fn evaluate_cards(cards: Vec<i32>) -> i32 {
        crate::phe::evaluate_cards(&cards)
    }

    /// Standalone postflop bucket (test/verify hook).
    #[pyfunction]
    fn postflop_bucket(hole: Vec<i32>, board: Vec<i32>, num_buckets: i32) -> i32 {
        let h = [hole[0], hole[1]];
        core::postflop_bucket(&h, &board, num_buckets)
    }

    /// Standalone blake2b digest (golden cross-check vs hashlib.blake2b).
    #[pyfunction]
    fn blake2b_digest<'py>(
        py: Python<'py>,
        digest_size: usize,
        person: &[u8],
        data: &[u8],
    ) -> Bound<'py, PyBytes> {
        let d = crate::blake2b::digest(digest_size, person, data);
        PyBytes::new(py, &d)
    }

    #[pymodule]
    fn fingerprint_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(fingerprint, m)?)?;
        m.add_function(wrap_pyfunction!(fingerprint_batch, m)?)?;
        m.add_function(wrap_pyfunction!(evaluate_cards, m)?)?;
        m.add_function(wrap_pyfunction!(postflop_bucket, m)?)?;
        m.add_function(wrap_pyfunction!(blake2b_digest, m)?)?;
        Ok(())
    }
}
