//! `marg3_native` — bit-identical native core for the 3-max legal-marginal cube math.
//!
//! The `core` module is pure Rust (testable with `cargo test`, no Python). The PyO3 bindings below are
//! compiled only with `--features python` (the `extension-module` cdylib loaded by
//! `aof_cfr/marg3_native.py`). The kernel materializes the two contiguous cubes `w` and `prod = w*pc`
//! that `fp_br._legal_marginals_3max` would build via NumPy broadcasts; the Python caller then does the
//! `np.sum(..., axis=red)` reductions on those exact buffers, so the result is bit-identical (the
//! reductions, which are NumPy-SIMD-order-sensitive, are NEVER reimplemented here).

pub mod core;

#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
mod py {
    use crate::core::{self, Cubes, Key, Pushes, N, N3};
    use pyo3::buffer::PyBuffer;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use rayon::prelude::*;

    /// Borrow a read-only C-contiguous f64 numpy array as `&[f64]` (zero-copy). Errors if the array is
    /// not C-contiguous or not the expected length.
    unsafe fn ro_slice<'py>(
        py: Python<'py>,
        obj: &Bound<'py, PyAny>,
        want_len: usize,
    ) -> PyResult<(&'py [f64], PyBuffer<f64>)> {
        let buf = PyBuffer::<f64>::get(obj)?;
        if !buf.is_c_contiguous() {
            return Err(PyValueError::new_err("array must be C-contiguous f64"));
        }
        if buf.item_count() != want_len {
            return Err(PyValueError::new_err(format!(
                "array length {} != expected {}",
                buf.item_count(),
                want_len
            )));
        }
        let ptr = buf.buf_ptr() as *const f64;
        let s = std::slice::from_raw_parts(ptr, want_len);
        // The PyBuffer keeps the export alive; return it so the borrow outlives the call.
        let _ = py;
        Ok((s, buf))
    }

    /// Borrow a writable C-contiguous f64 numpy array as `&mut [f64]` (zero-copy).
    unsafe fn rw_slice<'py>(
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

    fn key_from_str(k: &str) -> PyResult<Key> {
        Ok(match k {
            "btn" => Key::Btn,
            "sb_f" => Key::SbF,
            "sb_a" => Key::SbA,
            "bb_fa" => Key::BbFa,
            "bb_af" => Key::BbAf,
            "bb_aa" => Key::BbAa,
            _ => return Err(PyValueError::new_err(format!("unknown marginal key {k}"))),
        })
    }

    /// Build the `w` and `prod = w*pc` cubes for one marginal `key`, in place, into the preallocated
    /// `out_w` / `out_prod` (length 169^3). The cube arguments are the relevant `StaticPayoffs` fields;
    /// unused cubes for a given key may be the same zero array (they are not read).
    ///
    /// `pushes` is the flat layout: [btn(169), psf(169), psa(169), pfa(169), paf(169), paa(169),
    /// btn_win_uncontested_after_sb_fold, sb_win_uncontested_after_btn_fold].
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn build_marginal_cubes(
        py: Python<'_>,
        key: &str,
        deal_w: &Bound<'_, PyAny>,
        btn_bb_call0: &Bound<'_, PyAny>,
        btn_sb_call0: &Bound<'_, PyAny>,
        three0: &Bound<'_, PyAny>,
        sb_bb_call1: &Bound<'_, PyAny>,
        btn_sb_call1: &Bound<'_, PyAny>,
        three1: &Bound<'_, PyAny>,
        sb_bb_call2: &Bound<'_, PyAny>,
        btn_bb_call2: &Bound<'_, PyAny>,
        three2: &Bound<'_, PyAny>,
        pushes: Vec<f64>,
        out_w: &Bound<'_, PyAny>,
        out_prod: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let k = key_from_str(key)?;
        if pushes.len() != 6 * N + 2 {
            return Err(PyValueError::new_err("pushes must have 6*169+2 entries"));
        }
        unsafe {
            let (dw, _h0) = ro_slice(py, deal_w, N3)?;
            let (c_bb0, _h1) = ro_slice(py, btn_bb_call0, N3)?;
            let (c_sb0, _h2) = ro_slice(py, btn_sb_call0, N3)?;
            let (c_t0, _h3) = ro_slice(py, three0, N3)?;
            let (c_sbbb1, _h4) = ro_slice(py, sb_bb_call1, N3)?;
            let (c_btsb1, _h5) = ro_slice(py, btn_sb_call1, N3)?;
            let (c_t1, _h6) = ro_slice(py, three1, N3)?;
            let (c_sbbb2, _h7) = ro_slice(py, sb_bb_call2, N3)?;
            let (c_btbb2, _h8) = ro_slice(py, btn_bb_call2, N3)?;
            let (c_t2, _h9) = ro_slice(py, three2, N3)?;
            let (ow, _hw) = rw_slice(out_w, N3)?;
            let (op, _hp) = rw_slice(out_prod, N3)?;

            let cubes = Cubes {
                deal_w: dw,
                btn_bb_call0: c_bb0,
                btn_sb_call0: c_sb0,
                three0: c_t0,
                sb_bb_call1: c_sbbb1,
                btn_sb_call1: c_btsb1,
                three1: c_t1,
                sb_bb_call2: c_sbbb2,
                btn_bb_call2: c_btbb2,
                three2: c_t2,
            };
            let p = Pushes {
                btn: &pushes[0..N],
                psf: &pushes[N..2 * N],
                psa: &pushes[2 * N..3 * N],
                pfa: &pushes[3 * N..4 * N],
                paf: &pushes[4 * N..5 * N],
                paa: &pushes[5 * N..6 * N],
                btn_win_uncontested_after_sb_fold: pushes[6 * N],
                sb_win_uncontested_after_btn_fold: pushes[6 * N + 1],
            };

            // Parallelize across the outer index i (each i writes a disjoint 169*169 contiguous block).
            // The GIL is released so other Python threads can run; the borrowed numpy buffers stay alive
            // for the duration via the PyBuffer handles above.
            let n2 = N * N;
            py.detach(|| {
                ow.par_chunks_mut(n2)
                    .zip(op.par_chunks_mut(n2))
                    .enumerate()
                    .for_each(|(i, (wblk, pblk))| {
                        // build_block writes rows [i, i+1) using GLOBAL offsets, but here the blocks are
                        // local; call a block-local kernel instead.
                        core::build_block_one(k, &cubes, &p, i, wblk, pblk);
                    });
            });
        }
        Ok(())
    }

    #[pymodule]
    fn marg3_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(build_marginal_cubes, m)?)?;
        Ok(())
    }
}
