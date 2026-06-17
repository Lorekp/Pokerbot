//! `marg4_native` — bit-identical native kernel for the 4-max AoF marginal payoff precompute.
//!
//! The `core` module is pure Rust (testable with `cargo test`, no Python). The PyO3 bindings below
//! are compiled only with `--features python` (the `extension-module` cdylib loaded by
//! `aof_cfr/marg4_native.py`). The single entry point `precompute_payoffs` fills the `(15, n, 4)`
//! payoff tensor that `fp_4max_vec.precompute_payoffs` produces, deal-parallel via rayon, and is
//! byte-for-byte identical to the Python path (matched MT19937 board RNG + winner-equivalent eval7).

pub mod core;

#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
mod py {
    use crate::core;
    use pyo3::buffer::PyBuffer;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use rayon::prelude::*;

    /// Borrow a read-only C-contiguous i64 numpy array as `&[i64]` (zero-copy).
    unsafe fn ro_i64<'py>(
        obj: &Bound<'py, PyAny>,
        want_len: usize,
    ) -> PyResult<(&'py [i64], PyBuffer<i64>)> {
        let buf = PyBuffer::<i64>::get(obj)?;
        if !buf.is_c_contiguous() {
            return Err(PyValueError::new_err("holes array must be C-contiguous int64"));
        }
        if buf.item_count() != want_len {
            return Err(PyValueError::new_err(format!(
                "holes length {} != expected {}",
                buf.item_count(),
                want_len
            )));
        }
        let ptr = buf.buf_ptr() as *const i64;
        let s = std::slice::from_raw_parts(ptr, want_len);
        Ok((s, buf))
    }

    /// Borrow a writable C-contiguous f64 numpy array as `&mut [f64]` (zero-copy).
    unsafe fn rw_f64<'py>(
        obj: &Bound<'py, PyAny>,
        want_len: usize,
    ) -> PyResult<(&'py mut [f64], PyBuffer<f64>)> {
        let buf = PyBuffer::<f64>::get(obj)?;
        if buf.readonly() {
            return Err(PyValueError::new_err("output array must be writable"));
        }
        if !buf.is_c_contiguous() {
            return Err(PyValueError::new_err("output must be C-contiguous f64"));
        }
        if buf.item_count() != want_len {
            return Err(PyValueError::new_err(format!(
                "output length {} != expected {}",
                buf.item_count(),
                want_len
            )));
        }
        let ptr = buf.buf_ptr() as *mut f64;
        let s = std::slice::from_raw_parts_mut(ptr, want_len);
        Ok((s, buf))
    }

    /// Fill `out` (shape (15, n, 4), C-contiguous f64) with the per-(outcome, deal, seat) net-bb
    /// payoff. `holes` is (n, 8) int64 in seat order (UTG,BTN,SB,BB; two card ids per seat).
    ///
    /// Bit-identical to `fp_4max_vec.precompute_payoffs(holes, stack_bb, game_model=...)` with the
    /// flat showdown fee `fee_bb` (== pushfold_payoffs.flat_fee_default() at call time). `threads`
    /// is the rayon pool size (0 = rayon default = all cores); the per-deal work is independent.
    #[pyfunction]
    #[pyo3(signature = (holes, n_deals, stack_bb, game_model, fee_bb, out, threads = 0))]
    #[allow(clippy::too_many_arguments)]
    fn precompute_payoffs(
        py: Python<'_>,
        holes: &Bound<'_, PyAny>,
        n_deals: usize,
        stack_bb: f64,
        game_model: i32,
        fee_bb: f64,
        out: &Bound<'_, PyAny>,
        threads: usize,
    ) -> PyResult<()> {
        let outs = core::outcome_masks(); // 15 masks, _OUTCOMES order
        let n_out = outs.len();
        let (holes_s, _hbuf) = unsafe { ro_i64(holes, n_deals * 8)? };
        let (out_s, _obuf) = unsafe { rw_f64(out, n_out * n_deals * 4)? };

        // out layout is (n_out, n_deals, 4): out[oi*(n_deals*4) + d*4 + seat]. To parallelize over
        // deals (the independent axis) without aliasing, we compute each deal's (15,4) block into a
        // local buffer, then scatter into the strided output. The scatter is cheap (60 f64/deal).
        let run = |out_s: &mut [f64]| {
            // Build per-deal then scatter. Use rayon over deal indices with a per-deal local buffer;
            // scatter under a split of out_s is awkward with the (outcome-major) layout, so collect
            // into a deal-major temp then transpose once. For 25M-scale runs the chunk driver keeps
            // n_deals modest (~chunk size), so the temp is fine; we still parallelize the heavy MC.
            let blocks: Vec<[f64; 60]> = (0..n_deals)
                .into_par_iter()
                .map(|d| {
                    let mut h = [0i32; 8];
                    for k in 0..8 {
                        h[k] = holes_s[d * 8 + k] as i32;
                    }
                    let mut block = [0.0f64; 60]; // 15 outcomes * 4 seats
                    core::deal_payoffs(&h, stack_bb, game_model, fee_bb, &outs, &mut block);
                    block
                })
                .collect();
            // scatter deal-major blocks into outcome-major out_s
            for d in 0..n_deals {
                let block = &blocks[d];
                for oi in 0..n_out {
                    let base = oi * (n_deals * 4) + d * 4;
                    out_s[base..base + 4].copy_from_slice(&block[oi * 4..oi * 4 + 4]);
                }
            }
        };

        py.detach(|| {
            if threads > 0 {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|e| PyValueError::new_err(format!("rayon pool: {e}")));
                match pool {
                    Ok(p) => p.install(|| run(out_s)),
                    Err(_) => run(out_s),
                }
            } else {
                run(out_s);
            }
        });
        Ok(())
    }

    /// Standalone showdown shares for a single n-way subset (test/verify hook).
    #[pyfunction]
    fn showdown_shares(flat: Vec<i32>, n: usize) -> Vec<f64> {
        let mut shares = [0.0f64; 4];
        core::showdown_shares(&flat, n, &mut shares);
        shares[..n].to_vec()
    }

    /// First few MT19937 32-bit outputs for a seed (golden cross-check against CPython random).
    #[pyfunction]
    fn mt_first(seed: u32, k: usize) -> Vec<u32> {
        let mut rng = core::MT19937::new_from_u32(seed);
        (0..k).map(|_| rng.genrand_uint32()).collect()
    }

    #[pymodule]
    fn marg4_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(precompute_payoffs, m)?)?;
        m.add_function(wrap_pyfunction!(showdown_shares, m)?)?;
        m.add_function(wrap_pyfunction!(mt_first, m)?)?;
        Ok(())
    }
}
