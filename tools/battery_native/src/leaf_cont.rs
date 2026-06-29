//! `LeafContWeights` — the model-aware continuation leaf's data-only contract.
//!
//! Replaces the uniform `U{0, 4}` continuation draw in `playout::playout` with a biased draw driven by
//! a per-leaf 4-vector precomputed in Python (opponent-model read at the leaf). Pure data: the Rust
//! side never recomputes the model — it just LOOKS UP a `[f64; 4]` vector keyed by the leaf's i64 key
//! and does an inverse-CDF sample over it. Same float surface as the Python branch
//! (`np.cumsum(w)` + `np.searchsorted(cum, u, side="right")`), so the per-leaf continuation distribution
//! is statistically equivalent to Python's `cont_weights` branch — NOT a SHA match (the leaf already
//! uses a non-replay RNG; we just preserve the bias surface).
//!
//! EFFICIENCY:
//!   * zero-copy lookup: a single `HashMap::get` per leaf (amortized — same leaf-key the leaf cache uses).
//!   * pre-normalized: the constructor divides by `sum` ONCE per leaf at construction time, so the hot
//!     path is an inverse-CDF over a `s ≈ 1.0` vector (no runtime normalize, no `s<=1e-12` fallback).
//!   * no per-call Python rebuild: the PyO3 constructor copies the dict ONCE; the leaf reuses the
//!     stored `Arc<LeafContWeights>` across all rollouts in the worker.
//!   * FxHashMap-backed (matches the crate's existing key-table convention; small fixed-width keys).

use rustc_hash::FxHashMap;

use crate::playout::NUM_CONTINUATIONS;

/// Per-leaf continuation weights: leaf_key(i64) -> 4-vector (already pre-normalized at construction).
///
/// `i64` keys are the 64-bit FNV-1a hash of the leaf's `Vec<i64>` key (the SAME encoding
/// `subgame::leaf_key` emits, collapsed to a single `u64` so the table is a scalar-keyed HashMap).
/// Python computes the dict once on the master and ships it across — the lookup path is just
/// `weights.get(leaf_key)` followed by an inverse-CDF sample (a 3-compare scan over 4 entries).
#[derive(Clone, Debug)]
pub struct LeafContWeights {
    map: FxHashMap<i64, [f64; NUM_CONTINUATIONS]>,
}

/// Tiny scratch buffer for cumsum-in-place. Hot-path allocation-free.
#[inline]
fn sample_with(row: &[f64; NUM_CONTINUATIONS], u: f64) -> usize {
    // Pre-normalized row: sum is ~1.0. Build cumsum sequentially (matches numpy `np.cumsum` float
    // accumulation surface), then count the entries with `cum <= u` (numpy `searchsorted(side="right")`).
    // NUM_CONTINUATIONS=4 is tiny — unrolled is overkill, but the compiler unrolls the loop anyway.
    debug_assert!(NUM_CONTINUATIONS == 4, "sample_with assumes NUM_CONTINUATIONS=4");
    let mut cum = 0.0f64;
    let mut pick = 0usize;
    for i in 0..NUM_CONTINUATIONS {
        cum += row[i];
        if cum <= u {
            pick = i + 1;
        }
    }
    if pick > NUM_CONTINUATIONS - 1 {
        pick = NUM_CONTINUATIONS - 1;
    }
    pick
}

impl LeafContWeights {
    /// Build from a Python-side iterator of `(leaf_key_i64, [f64; 4])`. Pre-normalizes each row by
    /// dividing by `sum` (clamped at `1e-12` to avoid div-by-zero on an all-zero vector; such a row
    /// is replaced with a uniform `[0.25; 4]` so the inverse-CDF still draws a valid index).
    pub fn from_iter<I>(it: I) -> Self
    where
        I: IntoIterator<Item = (i64, [f64; NUM_CONTINUATIONS])>,
    {
        let mut map: FxHashMap<i64, [f64; NUM_CONTINUATIONS]> = FxHashMap::with_capacity_and_hasher(
            0,
            Default::default(),
        );
        for (k, row) in it {
            let mut s = 0.0f64;
            for &v in &row {
                // Match Python: `w = cont_weights.get(s)` returns a numpy float row that may carry
                // tiny negative noise (~1e-18) from upstream float math. Clamp negatives to 0 so a
                // residual -0.0 / -1e-17 can't poison the cumsum.
                let v = if v < 0.0 { 0.0 } else { v };
                s += v;
            }
            let row_n = if s > 1e-12 {
                let inv = 1.0 / s;
                [row[0] * inv, row[1] * inv, row[2] * inv, row[3] * inv]
            } else {
                // Degenerate (all-zero or near-zero) row — fall back to uniform over the 4 cells so
                // the inverse-CDF still picks a legal index.
                [0.25, 0.25, 0.25, 0.25]
            };
            map.insert(k, row_n);
        }
        LeafContWeights { map }
    }

    /// Look up the pre-normalized 4-vector for `leaf_key` (no allocation, returns `None` on miss).
    /// Hot path: `match weights.get_vec(leaf_key) { Some(row) => sample_with(row, u), None => uniform }`.
    #[inline]
    pub fn get_vec(&self, leaf_key: i64) -> Option<&[f64; NUM_CONTINUATIONS]> {
        self.map.get(&leaf_key)
    }

    /// Sample a continuation index for `leaf_key` using a `u in [0, 1)` uniform draw from the worker's
    /// RNG. Miss falls back to a uniform `U{0, 4}` so a sparse dict (or a `None` model) preserves the
    /// legacy read-blind behavior bit-for-bit (well, distribution-equivalent under the native RNG).
    #[inline]
    pub fn sample(&self, leaf_key: i64, u: f64) -> usize {
        match self.map.get(&leaf_key) {
            Some(row) => sample_with(row, u),
            None => {
                // Uniform fallback: choose the bin whose cumsum (`(i+1)/4`) first exceeds `u`.
                // Equivalent to `int(u * 4)` clamped to 3, but uses the same `cum <= u` accounting as
                // the biased path so the float surface is identical (no special-case jump).
                let mut cum = 0.0f64;
                let uniform_step = 1.0 / NUM_CONTINUATIONS as f64;
                let mut pick = 0usize;
                for i in 0..NUM_CONTINUATIONS {
                    cum += uniform_step;
                    if cum <= u {
                        pick = i + 1;
                    }
                }
                if pick > NUM_CONTINUATIONS - 1 {
                    pick = NUM_CONTINUATIONS - 1;
                }
                pick
            }
        }
    }

    /// Number of leaves with a non-uniform override (diagnostic / sanity gate).
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// 64-bit FNV-1a hash over a `Vec<i64>` leaf-key vector — the SAME per-leaf hash the existing
/// `subgame::leaf_seed` uses (minus the `public_seed` mix, which is per-solve and applied at the call
/// site). The Python gate computes the hash once per leaf and ships it as a scalar `i64` dict key;
/// Rust reconstructs the same `i64` from the same `EnvState` so lookups line up without a per-leaf
/// round trip.
pub fn leaf_key_i64(key: &[i64]) -> i64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in key {
        let xb = x as u64;
        for b in xb.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    // Reinterpret as i64 — hash maps don't care about signedness, and Python passes i64 over the wire.
    h as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_iter(items: Vec<(i64, [f64; 4])>) -> LeafContWeights {
        LeafContWeights::from_iter(items)
    }

    #[test]
    fn from_iter_pre_normalizes() {
        // sum=8 -> normalize to [0.125, 0.25, 0.125, 0.5].
        let w = build_iter(vec![(1, [1.0, 2.0, 1.0, 4.0])]);
        let row = w.get_vec(1).expect("row must exist");
        assert!((row[0] - 0.125).abs() < 1e-12);
        assert!((row[1] - 0.25).abs() < 1e-12);
        assert!((row[2] - 0.125).abs() < 1e-12);
        assert!((row[3] - 0.5).abs() < 1e-12);
        // Sum must be ~1.0 (pre-normalized).
        let s: f64 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "sum={} should be 1.0", s);
    }

    #[test]
    fn from_iter_clamps_negative_and_unifalls_zero() {
        // Negative noise clamped to 0; all-zero -> uniform fallback.
        let w = build_iter(vec![(2, [-1e-18, 0.0, 0.0, 0.0]), (3, [0.0, 0.0, 0.0, 0.0])]);
        let row2 = w.get_vec(2).unwrap();
        // The clamped vector is essentially [0, 0, 0, 0] -> sum 0 -> uniform fallback.
        assert_eq!(row2, &[0.25, 0.25, 0.25, 0.25]);
        assert_eq!(w.get_vec(3).unwrap(), &[0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn sample_with_is_right_side_searchsorted() {
        // [0.25, 0.25, 0.25, 0.25] uniform: u=0.0 -> 0 (cum[0]=0.25 > 0.0 nope, 0.5 nope, 0.75 nope,
        // 1.0 nope -> pick stays 0). Wait — `cum <= u` with cum=0.25>0.0 means pick stays 0. Then
        // cum=0.5>0.0 -> 0. cum=0.75>0.0 -> 0. cum=1.0>0.0 -> 0. pick=0. Correct.
        let row = [0.25, 0.25, 0.25, 0.25];
        assert_eq!(sample_with(&row, 0.0), 0);
        // u=0.25 -> cum=0.25 <= 0.25 -> pick=1; cum=0.5>0.25 -> 1; ... -> 1.
        assert_eq!(sample_with(&row, 0.25), 1);
        // u=0.99 -> cum=0.25, 0.5, 0.75, 1.0 all <= 0.99 -> pick=4 -> clamped to 3.
        assert_eq!(sample_with(&row, 0.99), 3);
    }

    #[test]
    fn sample_biased_matches_expected_distribution() {
        // w = [1, 0, 0, 0] normalized -> [1, 0, 0, 0]. ALL u in [0, 1) -> pick 0 (cum[0]=1>u always
        // unless u==1 which we never draw; sample_with clamps pick at NUM_CONTINUATIONS-1=3).
        let w = build_iter(vec![(10, [1.0, 0.0, 0.0, 0.0])]);
        for u_bits in 0..1000u64 {
            // Spread `u` across [0, 1) — bit-cast using a known-good float surface.
            let u = (u_bits as f64) / 1000.0;
            assert_eq!(w.sample(10, u), 0, "u={} should pick 0", u);
        }
    }

    #[test]
    fn sample_miss_uniform_fallback() {
        let w = build_iter(vec![(1, [1.0, 0.0, 0.0, 0.0])]);
        // No entry for key=999 -> uniform fallback: cum = 0.25, 0.5, 0.75, 1.0.
        assert_eq!(w.sample(999, 0.0), 0);
        assert_eq!(w.sample(999, 0.3), 1); // cum=0.25<=0.3 -> 1; cum=0.5>0.3 -> 1.
        assert_eq!(w.sample(999, 0.5), 2);
        assert_eq!(w.sample(999, 0.8), 3);
    }

    #[test]
    fn leaf_key_i64_deterministic_and_distinguishes_inputs() {
        let k1 = vec![0i64, 1, 2, 3];
        let k2 = vec![0i64, 1, 2, 4];
        let k3 = vec![0i64, 1, 2, 3];
        assert_eq!(leaf_key_i64(&k1), leaf_key_i64(&k3), "same input -> same hash");
        assert_ne!(leaf_key_i64(&k1), leaf_key_i64(&k2), "different input -> different hash");
    }
}