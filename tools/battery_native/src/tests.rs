//! Pure-Rust sanity tests (no Python). Bit-identity vs the Python oracle is proven by the golden
//! cross-check in `pokerbot/search/native_solver.py` / `tools/battery_native/crosscheck.py`.

use crate::core::*;

#[test]
fn straight_flush_beats_quads() {
    // royal flush: A K Q J T of clubs (suit 0). card = (rank-2)*4 + suit.
    let rf_ranks = [14, 13, 12, 11, 10];
    let suits0 = [0, 0, 0, 0, 0];
    let rf = score_five(&rf_ranks, &suits0);
    // quad aces + king kicker, mixed suits
    let quad_ranks = [14, 14, 14, 14, 13];
    let quad_suits = [0, 1, 2, 3, 0];
    let quad = score_five(&quad_ranks, &quad_suits);
    assert!(rf > quad, "straight flush {} should beat quads {}", rf, quad);
}

#[test]
fn wheel_is_five_high() {
    // A-2-3-4-5 straight (wheel) ranks as 5-high.
    let wheel = [14, 5, 4, 3, 2];
    let suits = [0, 1, 2, 3, 0];
    let sc = score_five(&wheel, &suits);
    // category 4 (straight), within = 5
    assert_eq!(sc, 4 * 15i32.pow(5) + 5);
}

#[test]
fn flush_beats_two_pair() {
    let flush_ranks = [14, 12, 9, 5, 3];
    let flush_suits = [1, 1, 1, 1, 1];
    let flush = score_five(&flush_ranks, &flush_suits);
    let tp_ranks = [14, 14, 13, 13, 2];
    let tp_suits = [0, 1, 0, 1, 2];
    let tp = score_five(&tp_ranks, &tp_suits);
    assert!(flush > tp);
}

#[test]
fn evaluate_seven_picks_best_five() {
    // 7 cards containing a flush; best-of-21 must find it.
    // clubs(suit0): A K Q 7 2  + 2 off-suit
    let cards = [
        (14 - 2) * 4,     // Ac
        (13 - 2) * 4,     // Kc
        (12 - 2) * 4,     // Qc
        (7 - 2) * 4,      // 7c
        (2 - 2) * 4,      // 2c
        (5 - 2) * 4 + 1,  // 5d
        (9 - 2) * 4 + 2,  // 9h
    ];
    let sc = evaluate_seven(&cards);
    // flush is category 5
    assert_eq!(sc / 15i32.pow(5), 5);
}

#[test]
fn threefry_split_deterministic() {
    let k = [0u32, 0u32];
    let a = split_consume_pub(k);
    let b = split_consume_pub(k);
    assert_eq!(a, b);
    // advancing twice differs from once
    let c = split_consume_pub(a);
    assert_ne!(a, c);
}

// expose split_consume for the test (it's private in core)
fn split_consume_pub(key: [u32; 2]) -> [u32; 2] {
    // re-derive via a deal: easier to just test sample_chance determinism instead.
    // Here we call the public sample_chance path indirectly is overkill; replicate inline.
    threefry_carry(key)
}

fn threefry_carry(key: [u32; 2]) -> [u32; 2] {
    // mirror core::split_consume (threefry2x32(key, 0, 0))
    crate::core::_test_split_consume(key)
}
