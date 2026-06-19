//! Pure-Rust core of the blueprint infoset fingerprint — bit-identical to
//! `pokerbot.engine.jax_weva_fast.fingerprint_fast(state, seat)` in BETTING-ABSTRACTION mode.
//!
//! The Python hot path (`NLHEEncodeBundle._digest` with `_BETTING_ABSTRACTION`) hashes ONLY integer
//! features through blake2b(digest_size=16, person=b""):
//!   POSTFLOP (bucket>=0): b"B" + bucket(i32 le) + b"S" + betting_state(6 bytes)
//!   PREFLOP  (bucket<0):  b"w" + weva_id(i32 le) + (b"H" + history_key | b"S" + betting_state)
//! This module reproduces all of: the lossless preflop class, the WEVA table lookup (passed in from
//! Python), the bounded betting-state key, the lossless history key, the postflop made-hand bucket
//! (vendored phevaluator), and the blake2b assembly. No floats enter the digest -> exact by construction.

use crate::blake2b::Blake2b;
use crate::phe;

pub const NUM_PLAYERS: usize = 6;
pub const CHIP_UNIT: f64 = 100.0;
pub const PREFLOP_NUM_CLASSES: i32 = 169;

// Seat-status enums (jax_engine / side_pots_jax): EMPTY=0, FOLDED=1, LIVE=2, ALLIN=3.
pub const SEAT_EMPTY: i32 = 0;
pub const SEAT_FOLDED: i32 = 1;

// Position enum order used by nlhe_bridge._position_for_seat: BTN,SB,BB,UTG,HJ,CO indexed by offset.
const POS_BY_OFFSET: [i32; 6] = [3, 4, 5, 0, 1, 2];

// Bucket edges for the bounded betting-state key (jax_weva_fast).
const POT_EDGES: [f64; 7] = [3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 200.0];
const TOCALL_EDGES: [f64; 5] = [0.01, 0.34, 0.67, 1.01, 2.01];
const SPR_EDGES: [f64; 5] = [1.0, 2.5, 5.0, 10.0, 20.0];

#[inline]
fn bucket(x: f64, edges: &[f64]) -> u8 {
    let mut b: u8 = 0;
    for &e in edges {
        if x >= e {
            b += 1;
        } else {
            break;
        }
    }
    b
}

/// Lossless suit-isomorphic preflop class in [0,168] (weva_encoder.canonical_preflop_class).
pub fn canonical_preflop_class(c0: i32, c1: i32) -> i32 {
    let (r0, s0) = (c0 / 4, c0 % 4);
    let (r1, s1) = (c1 / 4, c1 % 4);
    if r0 == r1 {
        return r0;
    }
    let (hr, lr) = if r0 > r1 { (r0, r1) } else { (r1, r0) };
    let tri = hr * (hr - 1) / 2 + lr;
    if s0 == s1 {
        13 + tri
    } else {
        13 + 78 + tri
    }
}

/// Flat EnvState fields needed by the fingerprint (the Python wrapper extracts these as plain arrays).
pub struct FlatState<'a> {
    pub holes: &'a [i32],            // (6,2) flattened, row-major: holes[seat*2 + k]
    pub board: &'a [i32],           // (5,)
    pub seat_status: &'a [i32],     // (6,)
    pub stacks: &'a [i32],          // (6,)
    pub committed_street: &'a [i32],// (6,)
    pub street: i32,
    pub pot_dead: i32,
    pub current_bet: i32,
    pub button: i32,
    pub last_aggressor: i32,
    pub hist_seat: &'a [i32],       // (hist_len_capacity,)
    pub hist_action: &'a [i32],     // (hist_len_capacity,)
    pub hist_len: i32,
}

/// Bounded betting-state descriptor (jax_weva_fast.betting_abstraction_key) -> 6 bytes.
pub fn betting_abstraction_key(s: &FlatState, seat: usize) -> [u8; 6] {
    let committed_sum: i64 = s.committed_street.iter().map(|&x| x as i64).sum();
    let pot: i64 = s.pot_dead as i64 + committed_sum;
    let cur = s.current_bet as i64;
    let to_call = (cur - s.committed_street[seat] as i64).max(0);
    let street = s.street;
    let button = s.button;
    let last_agg = s.last_aggressor;
    let pot_bb = pot as f64 / CHIP_UNIT;
    // effective stack = min live stack (live = status not EMPTY/FOLDED)
    let mut eff_min: Option<i64> = None;
    for st in 0..s.stacks.len() {
        let status = s.seat_status[st];
        if status != SEAT_EMPTY && status != SEAT_FOLDED {
            let v = s.stacks[st] as i64;
            eff_min = Some(match eff_min {
                Some(m) => m.min(v),
                None => v,
            });
        }
    }
    let eff = eff_min.unwrap_or(0) as f64 / CHIP_UNIT;
    let pot_b = bucket(pot_bb, &POT_EDGES);
    let tocall_b = bucket(to_call as f64 / (pot.max(1)) as f64, &TOCALL_EDGES);
    let spr_b = bucket(eff / pot_bb.max(1e-6), &SPR_EDGES);
    let is_button = if seat as i32 == button { 1u8 } else { 0u8 };
    let facing_agg = if last_agg >= 0 && last_agg != seat as i32 {
        1u8
    } else {
        0u8
    };
    [
        (street & 7) as u8,
        pot_b,
        tocall_b,
        spr_b,
        is_button,
        facing_agg,
    ]
}

/// Lossless betting history key (jax_weva_fast.betting_history_key) -> variable bytes.
/// hl<=0 => b"\x00\x00"; else hl(u16 le) + seat[i16 le]* + act[i16 le]* (seat then action arrays).
pub fn betting_history_key(s: &FlatState) -> Vec<u8> {
    let hl = s.hist_len;
    if hl <= 0 {
        return vec![0u8, 0u8];
    }
    let hl = hl as usize;
    let mut out = Vec::with_capacity(2 + 4 * hl);
    out.extend_from_slice(&(hl as u16).to_le_bytes());
    // seat array as int16 little-endian, then action array as int16 little-endian.
    for i in 0..hl {
        out.extend_from_slice(&(s.hist_seat[i] as i16).to_le_bytes());
    }
    for i in 0..hl {
        out.extend_from_slice(&(s.hist_action[i] as i16).to_le_bytes());
    }
    out
}

/// Postflop made-hand bucket (postflop_bucketer.fast_made_hand_bucket via phevaluator):
/// rank 1..7462, bucket = min(num_buckets-1, (rank-1)*num_buckets/7462). -1 if <3 board cards.
pub fn postflop_bucket(hole: &[i32; 2], board: &[i32], num_buckets: i32) -> i32 {
    let mut cards: Vec<i32> = Vec::with_capacity(7);
    for &h in hole.iter() {
        if h < 0 {
            return -1;
        }
        cards.push(h);
    }
    let mut nboard = 0;
    for &c in board.iter() {
        if c >= 0 {
            cards.push(c);
            nboard += 1;
        }
    }
    if nboard < 3 {
        return -1;
    }
    let rank = phe::evaluate_cards(&cards);
    let bk = ((rank - 1) * num_buckets / 7462).min(num_buckets - 1);
    bk
}

/// Config knobs that are import-baked in Python (read from env at import). Passed in per-call so the
/// native path mirrors whatever regime the process is running.
pub struct FpConfig {
    pub preflop_lossless: bool,
    pub preflop_lossless_betting: bool,
    pub postflop_buckets_enabled: bool,
    pub postflop_num_buckets: i32,
}

/// Compute the 16-byte fingerprint, bit-identical to fingerprint_fast in betting-abstraction mode.
/// `weva_table` is the (52,52,6) i32 table (default_weva_bucket_table) flattened row-major.
pub fn fingerprint(
    s: &FlatState,
    seat: usize,
    weva_table: &[i32],
    cfg: &FpConfig,
) -> [u8; 16] {
    let button = s.button;
    let pos_off = ((seat as i32 - button).rem_euclid(NUM_PLAYERS as i32)) as usize;
    let pos_enum = POS_BY_OFFSET[pos_off] as usize;

    // weva_id = table[lo, hi, pos_enum] with lo<=hi of the hero's two hole ids (0 if either <0).
    let h0 = s.holes[seat * 2];
    let h1 = s.holes[seat * 2 + 1];
    let weva_id: i32 = if h0 < 0 || h1 < 0 {
        0
    } else {
        let (lo, hi) = if h0 <= h1 { (h0, h1) } else { (h1, h0) };
        weva_table[(lo as usize) * 52 * 6 + (hi as usize) * 6 + pos_enum]
    };

    // postflop bucket
    let hole = [h0, h1];
    let bucket_id: i32 = if cfg.postflop_buckets_enabled {
        postflop_bucket(&hole, s.board, cfg.postflop_num_buckets)
    } else {
        -1
    };

    let mut h = Blake2b::new(16, b"");
    if bucket_id >= 0 {
        h.update(b"B");
        h.update(&bucket_id.to_le_bytes());
        h.update(b"S");
        h.update(&betting_abstraction_key(s, seat));
    } else {
        h.update(b"w");
        h.update(&weva_id.to_le_bytes());
        if cfg.preflop_lossless && cfg.preflop_lossless_betting {
            h.update(b"H");
            h.update(&betting_history_key(s));
        } else {
            h.update(b"S");
            h.update(&betting_abstraction_key(s, seat));
        }
    }
    let d = h.finalize();
    let mut out = [0u8; 16];
    out.copy_from_slice(&d[..16]);
    out
}
