//! `aivat_native` — bit-identical native core for the AIVAT control variates.
//!
//! The `core` module is pure Rust (testable with `cargo test`, no Python). The PyO3 bindings below are
//! compiled only with `--features python` (the `extension-module` cdylib loaded by
//! `pokerbot/eval/aivat_native.py`). The exposed functions mirror the Python oracle API 1:1 and are
//! verified bit-identical to it via the golden vectors.

pub mod core;

#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
mod py {
    use crate::core;
    use pyo3::prelude::*;

    /// Bit-identical `aivat.decision_correction(probs, baseline_values, taken_idx)`.
    #[pyfunction]
    fn decision_correction(probs: Vec<f64>, baseline_values: Vec<f64>, taken_idx: isize) -> f64 {
        core::decision_correction(&probs, &baseline_values, taken_idx)
    }

    /// Bit-identical `aivat.chance_correction(expected_value, realized_value)`.
    #[pyfunction]
    fn chance_correction(expected_value: f64, realized_value: f64) -> f64 {
        core::chance_correction(expected_value, realized_value)
    }

    /// Per-reveal chance reduction `realized - float(np.mean(vals))` (the chance-term inner reduction).
    #[pyfunction]
    fn chance_reduce(realized: f64, vals: Vec<f64>) -> f64 {
        core::chance_reduce(realized, &vals)
    }

    /// `np.mean` over an f64 array, bit-identical (exposed for testing the pairwise reduction directly).
    #[pyfunction]
    fn np_mean(vals: Vec<f64>) -> f64 {
        if vals.is_empty() {
            f64::NAN
        } else {
            core::mean(&vals)
        }
    }

    /// `np.sum` over an f64 array, bit-identical (pairwise).
    #[pyfunction]
    fn np_sum(vals: Vec<f64>) -> f64 {
        core::pairwise_sum(&vals, 0, vals.len())
    }

    /// `AIVATAccumulator.add_hand` adjusted score for one hand.
    #[pyfunction]
    fn add_hand_adjusted(raw: f64, corrections: Vec<f64>, chance: Vec<f64>) -> f64 {
        core::add_hand_adjusted(raw, &corrections, &chance)
    }

    /// `AIVATAccumulator.summary()` over the raw + adjusted per-hand series.
    /// Returns (hands, raw_mean_bb, adj_mean_bb, raw_std, adj_std, variance_reduction).
    #[pyfunction]
    fn accumulator_summary(raw: Vec<f64>, adj: Vec<f64>) -> (usize, f64, f64, f64, f64, f64) {
        let s = core::accumulator_summary(&raw, &adj);
        (
            s.hands,
            s.raw_mean_bb,
            s.adj_mean_bb,
            s.raw_std,
            s.adj_std,
            s.variance_reduction,
        )
    }

    #[pymodule]
    fn aivat_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(decision_correction, m)?)?;
        m.add_function(wrap_pyfunction!(chance_correction, m)?)?;
        m.add_function(wrap_pyfunction!(chance_reduce, m)?)?;
        m.add_function(wrap_pyfunction!(np_mean, m)?)?;
        m.add_function(wrap_pyfunction!(np_sum, m)?)?;
        m.add_function(wrap_pyfunction!(add_hand_adjusted, m)?)?;
        m.add_function(wrap_pyfunction!(accumulator_summary, m)?)?;
        Ok(())
    }
}
