//! Bit-identical native core for the AIVAT control variates.
//!
//! Every reduction here reproduces NumPy's IEEE-754 f64 operation order EXACTLY — this is the hard,
//! non-negotiable constraint (the Python oracle is `pokerbot/eval/aivat.py` + `chance_term.py`). The
//! single trap is NumPy's PAIRWISE summation (`pairwise_sum` in numpy's `_core/src/umath/loops_utils`):
//! `np.sum` / `np.mean` are NOT naive left-to-right above a handful of elements. We replicate the exact
//! algorithm (8-way unrolled base case <=128, recursive split rounded to a multiple of 8 above) so
//! native output == Python output bit-for-bit (verified against golden vectors in tests).
//!
//! No allocation in the hot reductions; `#[inline]` on the leaves; the recursion is on indices into a
//! borrowed slice (cache-friendly sequential access, the same memory the caller already owns).

/// NumPy `pairwise_sum` over `a[off .. off+n]`, bit-identical to `np.add.reduce` / `np.sum`.
///
/// Mirrors numpy's C `pairwise_sum`:
///   * `n < 8`            -> naive left-to-right accumulation.
///   * `8 <= n <= 128`    -> 8 independent accumulators, 8-wide unrolled, combined as
///                           `((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))`, then the `n%8` tail naively.
///   * `n > 128`          -> split into `n2 = (n/2) rounded down to a multiple of 8`, recurse on
///                           `[off, off+n2)` and `[off+n2, off+n)`, return the sum of the two halves.
///
/// The blocksize (128) and the round-to-8 split are load-bearing for bit-identity; do not "simplify".
pub fn pairwise_sum(a: &[f64], off: usize, n: usize) -> f64 {
    if n < 8 {
        let mut s = 0.0_f64;
        for i in 0..n {
            s += a[off + i];
        }
        s
    } else if n <= 128 {
        // 8 partial sums, exactly as numpy's unrolled base case.
        let mut r0 = a[off];
        let mut r1 = a[off + 1];
        let mut r2 = a[off + 2];
        let mut r3 = a[off + 3];
        let mut r4 = a[off + 4];
        let mut r5 = a[off + 5];
        let mut r6 = a[off + 6];
        let mut r7 = a[off + 7];
        let mut i = 8usize;
        let limit = n - (n % 8);
        while i < limit {
            r0 += a[off + i];
            r1 += a[off + i + 1];
            r2 += a[off + i + 2];
            r3 += a[off + i + 3];
            r4 += a[off + i + 4];
            r5 += a[off + i + 5];
            r6 += a[off + i + 6];
            r7 += a[off + i + 7];
            i += 8;
        }
        let mut res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
        // Remaining n%8 tail, naive left-to-right (numpy does the same).
        while i < n {
            res += a[off + i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(a, off, n2) + pairwise_sum(a, off + n2, n - n2)
    }
}

/// `np.mean(a)` bit-identical: `pairwise_sum(a) / n`. Empty array policy is the caller's (numpy warns
/// and returns NaN; the chance term guards empties before calling, see `chance_reduce`).
#[inline]
pub fn mean(a: &[f64]) -> f64 {
    pairwise_sum(a, 0, a.len()) / (a.len() as f64)
}

/// `decision_correction(probs, baseline_values, taken_idx)` — bit-identical to `aivat.decision_correction`.
///
/// Python:  `p=asarray(probs,f64); bv=asarray(bv,f64); s=p.sum()`
///          `if s<=0 or not (0<=taken<len(bv)): return 0.0`
///          `p=p/s; return float(bv[taken] - float((p*bv).sum()))`
/// `p.sum()` and `(p*bv).sum()` are pairwise sums. We form the elementwise product into a scratch
/// buffer to match `(p*bv)` then reduce — identical op order.
pub fn decision_correction(probs: &[f64], bv: &[f64], taken_idx: isize) -> f64 {
    let s = pairwise_sum(probs, 0, probs.len());
    if !(s > 0.0) || taken_idx < 0 || taken_idx >= bv.len() as isize {
        return 0.0;
    }
    let n = probs.len();
    // p = probs / s ; then product (p*bv) elementwise, reduced pairwise.
    let mut prod = vec![0.0_f64; n.min(bv.len())];
    // numpy broadcasts p*bv over the shared length; both come from asarray so shapes match in the
    // oracle's usage. We multiply over min length defensively (oracle assumes equal length).
    for i in 0..prod.len() {
        let p = probs[i] / s;
        prod[i] = p * bv[i];
    }
    let dot = pairwise_sum(&prod, 0, prod.len());
    bv[taken_idx as usize] - dot
}

/// `chance_correction(expected, realized) = realized - expected` (bit-identical; trivial f64 subtract).
#[inline]
pub fn chance_correction(expected: f64, realized: f64) -> f64 {
    realized - expected
}

/// Per-reveal AIVAT chance reduction: `c_C = realized - float(np.mean(vals))`.
/// Mirrors `chance_term.chance_corrections_for_hand`'s arithmetic exactly: `expected = float(np.mean(vals))
/// if vals else realized`, then `realized - expected`. The `vals` are the precomputed gadget values over
/// the (enumerated) board complement; the RNG-driven gadget v stays in Python and feeds this.
#[inline]
pub fn chance_reduce(realized: f64, vals: &[f64]) -> f64 {
    let expected = if vals.is_empty() { realized } else { mean(vals) };
    realized - expected
}

/// `AIVATAccumulator.summary()` — bit-identical. `raw` and `adj` are the two per-hand series.
///
/// Python:
///   `rstd = float(raw.std(ddof=1)) if len(raw)>1 else 0.0`  (likewise adj)
///   `raw_mean = float(raw.mean()) if len(raw) else 0.0`
///   `variance_reduction = (1 - (astd/rstd)**2) if rstd>0 else 0.0`
///
/// NumPy's `ndarray.std(ddof=1)`: `m = sum(a)/n` (pairwise), then `sqrt( sum((a-m)^2)/(n-1) )` where the
/// inner sum is ALSO pairwise over the squared deviations. We materialize the deviation-squared buffer
/// to match numpy's reduction order exactly.
pub struct Summary {
    pub hands: usize,
    pub raw_mean_bb: f64,
    pub adj_mean_bb: f64,
    pub raw_std: f64,
    pub adj_std: f64,
    pub variance_reduction: f64,
}

fn np_mean_or_zero(a: &[f64]) -> f64 {
    if a.is_empty() {
        0.0
    } else {
        mean(a)
    }
}

fn np_std_ddof1(a: &[f64]) -> f64 {
    // numpy: var = mean of |a - a.mean()|**2 with N-ddof in the denominator; std = sqrt(var).
    let n = a.len();
    if n <= 1 {
        return 0.0;
    }
    let m = mean(a); // pairwise sum / n
    let mut dev2 = vec![0.0_f64; n];
    for i in 0..n {
        let d = a[i] - m;
        dev2[i] = d * d;
    }
    let ss = pairwise_sum(&dev2, 0, n);
    let var = ss / ((n - 1) as f64);
    var.sqrt()
}

pub fn accumulator_summary(raw: &[f64], adj: &[f64]) -> Summary {
    let rstd = np_std_ddof1(raw);
    let astd = np_std_ddof1(adj);
    let vr = if rstd > 0.0 {
        let ratio = astd / rstd;
        1.0 - ratio * ratio
    } else {
        0.0
    };
    Summary {
        hands: raw.len(),
        raw_mean_bb: np_mean_or_zero(raw),
        adj_mean_bb: np_mean_or_zero(adj),
        raw_std: rstd,
        adj_std: astd,
        variance_reduction: vr,
    }
}

/// CPython builtin `sum()` over an f64 iterable, bit-identical.
///
/// BIT-IDENTITY TRAP: CPython's `sum()` (>=3.12) does NOT do naive left-to-right addition for floats —
/// it uses NEUMAIER compensated summation (a running correction term `c`). `aivat.AIVATAccumulator.
/// add_hand` calls `float(sum(corrections))` / `float(sum(chance_corrections))`, so we MUST reproduce
/// Neumaier, not a plain fold (a plain fold is off by 1 ULP on ~8% of 3-element series). `sum()` starts
/// from int 0, which for floats is equivalent to `s=0.0, c=0.0`.
#[inline]
pub fn py_sum(items: &[f64]) -> f64 {
    let mut s = 0.0_f64;
    let mut c = 0.0_f64;
    for &x in items {
        let t = s + x;
        if s.abs() >= x.abs() {
            c += (s - t) + x;
        } else {
            c += (x - t) + s;
        }
        s = t;
    }
    s + c
}

/// `AIVATAccumulator.add_hand` adjusted-score: `adj = raw - (sum(corrections) + sum(chance))`.
/// `sum(...)` is CPython's Neumaier-compensated `py_sum`; the two sums are then combined with a plain
/// f64 `+=` (matching `c = float(sum(corr)); c += float(sum(chance))`). Empty list -> 0.0 (the oracle's
/// `... if corrections else 0.0`).
#[inline]
pub fn add_hand_adjusted(raw: f64, corrections: &[f64], chance: &[f64]) -> f64 {
    let mut c = 0.0_f64;
    if !corrections.is_empty() {
        c = py_sum(corrections);
    }
    if !chance.is_empty() {
        c += py_sum(chance);
    }
    raw - c
}
