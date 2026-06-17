//! Pure-Rust bit-identity unit tests (run with `cargo test`, no Python required).
//!
//! These pin the EXACT f64 outputs that the NumPy oracle produced for a handful of vectors (captured
//! from `tools/aivat_native/golden/golden.json`, decoded from the little-endian hex). Equality is exact
//! `==` on f64 bit patterns. The full golden suite (hundreds of cases incl. real blueprint reveals) is
//! checked from Python in `pokerbot/eval/aivat_native.py::verify_against_golden`.

use crate::core::*;

/// Decode a little-endian f64 hex (as stored in golden.json) back to f64.
fn fh(hex: &str) -> f64 {
    let bytes = (0..8)
        .map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap())
        .collect::<Vec<u8>>();
    f64::from_le_bytes(bytes.try_into().unwrap())
}

#[test]
fn pairwise_matches_naive_below_8() {
    // n<8 is naive left-to-right; trivially equals a manual fold.
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut s = 0.0;
    for x in a {
        s += x;
    }
    assert_eq!(pairwise_sum(&a, 0, a.len()), s);
}

#[test]
fn pairwise_split_is_recursive_above_128() {
    // 200 elements: numpy splits at n2 = 100 -> 96 (rounded to mult of 8). Verify our split matches
    // the documented structure by reconstructing it independently.
    let a: Vec<f64> = (0..200).map(|i| (i as f64) * 0.5 - 17.0).collect();
    let n = a.len();
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    assert_eq!(n2, 96);
    let expect = pairwise_sum(&a, 0, n2) + pairwise_sum(&a, n2, n - n2);
    assert_eq!(pairwise_sum(&a, 0, n), expect);
}

#[test]
fn decision_correction_zero_when_probs_sum_nonpositive() {
    assert_eq!(decision_correction(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0], 1), 0.0);
}

#[test]
fn decision_correction_zero_when_idx_out_of_range() {
    assert_eq!(decision_correction(&[0.5, 0.5], &[1.0, 2.0], 5), 0.0);
}

#[test]
fn decision_correction_golden_simple() {
    // Golden case (golden.json[0]): probs=[0.2,0.3,0.5], bv=[1.0,-2.0,3.5], taken=0.
    // Mathematically ~ -0.35, but NumPy's p/s normalization + pairwise (p*bv).sum() rounds to
    // -0.3500000000000001. The native core must hit THIS exact bit pattern, not the textbook value.
    let probs = [fh("9a9999999999c93f"), fh("333333333333d33f"), fh("000000000000e03f")];
    let bv = [fh("000000000000f03f"), fh("00000000000000c0"), fh("0000000000000c40")];
    let out = decision_correction(&probs, &bv, 0);
    let golden = fh("686666666666d6bf"); // -0.3500000000000001
    assert_eq!(out, golden);
}

#[test]
fn chance_correction_is_realized_minus_expected() {
    assert_eq!(chance_correction(0.25, 1.5), 1.25);
    assert_eq!(chance_correction(-3.0, -3.0), 0.0);
}

#[test]
fn chance_reduce_empty_vals_is_zero() {
    assert_eq!(chance_reduce(1.25, &[]), 0.0);
}

#[test]
fn add_hand_adjusted_naive_sum() {
    // raw=3.0, corr=[0.5], chance=[0.25] -> adj = 3 - (0.5+0.25) = 2.25
    assert_eq!(add_hand_adjusted(3.0, &[0.5], &[0.25]), 2.25);
    // empties -> raw unchanged
    assert_eq!(add_hand_adjusted(7.0, &[], &[]), 7.0);
}

#[test]
fn py_sum_neumaier_matches_cpython() {
    // CPython sum() uses Neumaier compensation: this 3-elt list differs by 1 ULP from a naive fold.
    let corr = [
        fh("86aeb6fe1cb295bf"), // -0.021187260672077567
        fh("0b58d587a9fff23f"), //  1.1874175363532256
        fh("51333a445f4cd8bf"), // -0.37966138518168174
    ];
    // CPython sum(corr) == 0.7865688904994662 (NOT the naive 0.7865688904994663).
    let golden_sum = fh("f9e09785922be93f");
    assert_eq!(py_sum(&corr), golden_sum);
    // and the adjusted score raw - sum(corr), raw = -0.12697153429002567
    let raw = -0.12697153429002567_f64;
    let adj = add_hand_adjusted(raw, &corr, &[]);
    assert_eq!(adj, fh("206f0121b93bedbf")); // -0.9135404247894918
}

#[test]
fn accumulator_empty_is_zero() {
    let s = accumulator_summary(&[], &[]);
    assert_eq!(s.hands, 0);
    assert_eq!(s.raw_mean_bb, 0.0);
    assert_eq!(s.variance_reduction, 0.0);
}

#[test]
fn accumulator_single_hand_zero_std() {
    let s = accumulator_summary(&[3.0], &[2.25]);
    assert_eq!(s.hands, 1);
    assert_eq!(s.raw_mean_bb, 3.0);
    assert_eq!(s.raw_std, 0.0); // ddof=1, n<=1 -> 0.0
    assert_eq!(s.variance_reduction, 0.0);
}
