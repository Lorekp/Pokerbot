//! Bit-exact port of `phevaluator.evaluate_cards` (the C-backed perfect-hash 5/6/7-card evaluator).
//!
//! Card id == rank*4 + suit (rank 0-12, suit 0-3) — IDENTICAL to the encoder's id convention, so the
//! encoder ids the Python passes to `evaluate_cards(*ints)` need no remapping. Returns rank 1 (best)
//! .. 7462 (worst). Tables are vendored in `tables.rs` (perfect-hash, no float) -> exact by construction.

use crate::tables::{
    BINARIES_BY_ID, DP, FLUSH, NO_FLUSH_5, NO_FLUSH_6, NO_FLUSH_7, SUITBIT_BY_ID, SUITS,
};

/// hash_quinary (phevaluator.hash.hash_quinary): quinary has length 13.
fn hash_quinary(quinary: &[u8; 13], num_cards_in: usize) -> usize {
    let mut sum_numb: u32 = 0;
    let length = 13usize;
    let mut num_cards = num_cards_in;
    for (rank, &cnt) in quinary.iter().enumerate() {
        let cnt = cnt as usize;
        if cnt != 0 {
            sum_numb += DP[cnt][length - rank - 1][num_cards];
            num_cards -= cnt;
        }
    }
    sum_numb as usize
}

/// Evaluate 5..7 cards (encoder ids). Mirrors phevaluator.evaluator._evaluate_cards exactly.
pub fn evaluate_cards(cards: &[i32]) -> i32 {
    let hand_size = cards.len();
    debug_assert!((5..=7).contains(&hand_size));

    let mut suit_hash: usize = 0;
    for &c in cards {
        suit_hash += SUITBIT_BY_ID[c as usize] as usize;
    }
    let flush_suit: i32 = SUITS[suit_hash] as i32 - 1;

    if flush_suit != -1 {
        let mut hand_binary: usize = 0;
        for &c in cards {
            if (c % 4) == flush_suit {
                hand_binary |= BINARIES_BY_ID[c as usize] as usize;
            }
        }
        return FLUSH[hand_binary] as i32;
    }

    let mut quinary = [0u8; 13];
    for &c in cards {
        quinary[(c / 4) as usize] += 1;
    }
    let idx = hash_quinary(&quinary, hand_size);
    let val = match hand_size {
        5 => NO_FLUSH_5[idx],
        6 => NO_FLUSH_6[idx],
        7 => NO_FLUSH_7[idx],
        _ => unreachable!(),
    };
    val as i32
}
