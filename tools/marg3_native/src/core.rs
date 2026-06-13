//! Bit-identical native core for the 3-max legal-marginal cube math (`aof_cfr/fp_br.py::
//! _legal_marginals_3max`).
//!
//! The Python reference, per (seat, history) marginal, builds a (169,169,169) "push cell" cube via a
//! chain of `pushfold_payoffs.mix` calls, multiplies it by a deal-weight cube `w`, and reduces the two
//! non-kept axes with `np.sum`. The two reductions (`np.sum(w, axis=red)` and `np.sum(w*pc, axis=red)`)
//! are ORDER-SENSITIVE: NumPy's large-`n` reduction uses a SIMD-blocked pairwise order that cannot be
//! reproduced portably in scalar Rust. So this core does NOT reduce. Instead it materializes the two
//! contiguous cubes `w` and `prod = w*pc` BIT-IDENTICALLY (pure elementwise, the same per-element IEEE-754
//! operand order as NumPy's broadcasts), and the caller hands those exact contiguous buffers to
//! `np.sum(..., axis=red)` — which then yields a result identical to the original code (same bytes in,
//! same reduction out). The win is the fused, cache-blocked, multi-threaded elementwise pass, which
//! replaces NumPy's chain of seven 38 MB broadcast temporaries per `btn` marginal.
//!
//! Elementwise bit-identity is the easy half of the original problem: `(1-p)*uf + p*up` per element is a
//! deterministic IEEE-754 expression with the same operands NumPy uses, so the materialized cube matches
//! NumPy's broadcast result exactly (verified by `np.array_equal` in the Python harness).

pub const N: usize = 169;
pub const N2: usize = N * N; // 28561
pub const N3: usize = N * N * N; // 4_826_809

#[inline(always)]
fn mix(p: f64, u_fold: f64, u_push: f64) -> f64 {
    (1.0 - p) * u_fold + p * u_push
}

/// Which (seat, history) marginal to build. The numeric tag mirrors the six keys in
/// `_legal_marginals_3max`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Key {
    Btn,
    SbF,
    SbA,
    BbFa,
    BbAf,
    BbAa,
}

/// Inputs shared by every kernel. The cube slices are read-only views of the relevant
/// `StaticPayoffs` arrays (C-contiguous, length `N3`); only the cubes a given key needs are read.
pub struct Cubes<'a> {
    pub deal_w: &'a [f64],          // dw            (always)
    pub btn_bb_call0: &'a [f64],    // btn key
    pub btn_sb_call0: &'a [f64],    // btn key
    pub three0: &'a [f64],          // btn key
    pub sb_bb_call1: &'a [f64],     // sb_f key
    pub btn_sb_call1: &'a [f64],    // sb_a key
    pub three1: &'a [f64],          // sb_a key
    pub sb_bb_call2: &'a [f64],     // bb_fa key
    pub btn_bb_call2: &'a [f64],    // bb_af key
    pub three2: &'a [f64],          // bb_aa key
}

/// Per-seat 1-D push-probability vectors (length `N`) and the two scalar fold constants used by the
/// uncontested mixes. Broadcast meaning (matching the Python `[:,None,None]` etc.):
///   * `btn` varies on axis i, `sb*`/`psf`/`psa` on axis j, `bb*`/`pfa`/`paf`/`paa` on axis k.
pub struct Pushes<'a> {
    pub btn: &'a [f64],  // pbtn[i]
    pub psf: &'a [f64],  // sb_push[0] -> psf[j]
    pub psa: &'a [f64],  // sb_push[1] -> psa[j]
    pub pfa: &'a [f64],  // bb_push[0] -> pfa[k]
    pub paf: &'a [f64],  // bb_push[1] -> paf[k]
    pub paa: &'a [f64],  // bb_push[2] -> paa[k]
    pub btn_win_uncontested_after_sb_fold: f64, // btn key fold-branch constant
    pub sb_win_uncontested_after_btn_fold: f64, // sb_f key fold-branch constant
}

/// Build the contiguous `w` and `prod = w*pc` cubes for `key` into the FULL `out_w` / `out_prod`
/// buffers (both length `N3`), writing all 169 outer rows. Single-threaded reference used by tests.
/// The per-element expressions are transcribed verbatim from `_legal_marginals_3max`.
pub fn build_full(key: Key, cubes: &Cubes, p: &Pushes, out_w: &mut [f64], out_prod: &mut [f64]) {
    for i in 0..N {
        let woff = i * N2;
        build_block_one(
            key,
            cubes,
            p,
            i,
            &mut out_w[woff..woff + N2],
            &mut out_prod[woff..woff + N2],
        );
    }
}

/// Build a single outer-index slab `i` into LOCAL `out_w` / `out_prod` blocks of length `N2` (indexed
/// `j*N + k`). The cube reads use the GLOBAL offset `i*N2 + j*N + k`. This is the per-thread kernel.
#[inline]
pub fn build_block_one(
    key: Key,
    cubes: &Cubes,
    p: &Pushes,
    i: usize,
    out_w: &mut [f64],
    out_prod: &mut [f64],
) {
    let pbtn = p.btn[i];
    let base = i * N2;
    // Outputs are LOCAL blocks indexed `loff = j*N + k`; cube reads use GLOBAL `off = base + loff`.
    match key {
        Key::Btn => {
            let bwsf = p.btn_win_uncontested_after_sb_fold;
            for j in 0..N {
                let psa = p.psa[j];
                let row = j * N;
                for k in 0..N {
                    let loff = row + k;
                    let off = base + loff;
                    let uf1 = mix(p.paf[k], bwsf, cubes.btn_bb_call0[off]);
                    let uf2 = mix(p.paa[k], cubes.btn_sb_call0[off], cubes.three0[off]);
                    let pc = mix(psa, uf1, uf2);
                    let w = cubes.deal_w[off];
                    out_w[loff] = w;
                    out_prod[loff] = w * pc;
                }
            }
        }
        Key::SbF => {
            let swf = p.sb_win_uncontested_after_btn_fold;
            let wfac = 1.0 - pbtn; // deal_w * (1 - pbtn)
            for j in 0..N {
                let row = j * N;
                for k in 0..N {
                    let loff = row + k;
                    let off = base + loff;
                    let pc = mix(p.pfa[k], swf, cubes.sb_bb_call1[off]);
                    let w = cubes.deal_w[off] * wfac;
                    out_w[loff] = w;
                    out_prod[loff] = w * pc;
                }
            }
        }
        Key::SbA => {
            let wfac = pbtn; // deal_w * pbtn
            for j in 0..N {
                let row = j * N;
                for k in 0..N {
                    let loff = row + k;
                    let off = base + loff;
                    let pc = mix(p.paa[k], cubes.btn_sb_call1[off], cubes.three1[off]);
                    let w = cubes.deal_w[off] * wfac;
                    out_w[loff] = w;
                    out_prod[loff] = w * pc;
                }
            }
        }
        Key::BbFa => {
            // NumPy: deal_w * (1.0 - pbtn) * psf  ==  ((dw * (1-pbtn[i])) * psf[j]).
            let mbtn = 1.0 - pbtn;
            for j in 0..N {
                let psf = p.psf[j];
                let row = j * N;
                for k in 0..N {
                    let loff = row + k;
                    let off = base + loff;
                    let pc = cubes.sb_bb_call2[off];
                    let w = (cubes.deal_w[off] * mbtn) * psf;
                    out_w[loff] = w;
                    out_prod[loff] = w * pc;
                }
            }
        }
        Key::BbAf => {
            // NumPy: deal_w * pbtn * (1.0 - psa)  ==  ((dw * pbtn[i]) * (1-psa[j])).
            for j in 0..N {
                let mpsa = 1.0 - p.psa[j];
                let row = j * N;
                for k in 0..N {
                    let loff = row + k;
                    let off = base + loff;
                    let pc = cubes.btn_bb_call2[off];
                    let w = (cubes.deal_w[off] * pbtn) * mpsa;
                    out_w[loff] = w;
                    out_prod[loff] = w * pc;
                }
            }
        }
        Key::BbAa => {
            // NumPy: deal_w * pbtn * psa  ==  ((dw * pbtn[i]) * psa[j]).
            for j in 0..N {
                let psa = p.psa[j];
                let row = j * N;
                for k in 0..N {
                    let loff = row + k;
                    let off = base + loff;
                    let pc = cubes.three2[off];
                    let w = (cubes.deal_w[off] * pbtn) * psa;
                    out_w[loff] = w;
                    out_prod[loff] = w * pc;
                }
            }
        }
    }
}
