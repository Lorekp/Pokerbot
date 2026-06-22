//! Byte-identical native port of `pokerbot.engine.jax_weva_fast.fingerprint_fast` for the
//! bounded-abstraction eval config used by the Pluribus battery hero:
//!   POKERBOT_BETTING_ABSTRACTION=1, POKERBOT_POSTFLOP_BUCKETS=1, POKERBOT_POSTFLOP_MODE=fast,
//!   POKERBOT_POSTFLOP_NUM_BUCKETS=200, POKERBOT_PREFLOP_LOSSLESS=1, PREFLOP_LOSSLESS_BETTING=0.
//!
//! Under that config `fingerprint_fast` reduces to a blake2b(digest_size=16, person=b"") of:
//!   * POSTFLOP (>=3 board cards): b"B" + bucket_id(i32 LE) + b"S" + betting_state(6 bytes)
//!   * PREFLOP : b"w" + weva_id(i32 LE) + b"S" + betting_state(6 bytes)
//! where weva_id = pos_enum*169 + canonical_preflop_class, bucket_id = fast_made_hand_bucket
//! (phevaluator 1..7462 rank binned into 200), and betting_state is the 6-byte abstraction key.
//!
//! Every sub-routine is a faithful port; the acceptance gate is a fuzz cross-check against the live
//! Python `fingerprint_fast` (see `crosscheck_fingerprint.py`) before the native rollout is trusted.

use crate::core::{EnvStateN, NUM_PLAYERS};
use crate::phe_tables;

pub const PREFLOP_NUM_CLASSES: i32 = 169;
pub const CHIP_UNIT: f64 = 100.0; // matches jax_engine.CHIP_UNIT (BB_CHIPS)

// Seat-status sentinels (np_engine): 0 empty, 1 folded, 2 live, 3 all-in.
const SEAT_EMPTY: i8 = 0;
const SEAT_FOLDED: i8 = 1;

// _POS_BY_OFFSET = [3,4,5,0,1,2] (BTN, SB, BB, UTG, HJ, CO), indexed by (seat - button) % 6.
const POS_BY_OFFSET: [i32; NUM_PLAYERS] = [3, 4, 5, 0, 1, 2];

/// Generic seat-offset -> position enum, the single source for any seat count `N`.
///
/// For `N == NUM_PLAYERS` (6) this is BYTE-IDENTICAL to indexing `POS_BY_OFFSET` (the trained 6-max
/// preflop position encoding the deployed blueprint was solved against). For any other `N` (e.g. the
/// 7-seat tournament root) there is NO trained blueprint keyed by these position ids, so the exact
/// enum is irrelevant to served correctness — the solver's blueprint lookup misses and falls back to
/// the uniform-over-legal prior (the chip-EV base). We need only a deterministic, in-range mapping.
/// The natural generalization keeps the 6-max ordering shape: offset 0 = button = the highest position
/// id, then SB, BB, then UTG..(N-3) ascending — i.e. BTN=`N-1`, SB=`N-2`... wraps to UTG=0. Concretely
/// `pos = (N - 1 - offset + 1) % N`-style; we reproduce the 6-map exactly via the table for N=6 and use
/// the offset-rotated form otherwise.
#[inline]
fn pos_by_offset<const N: usize>(offset: usize) -> i32 {
    if N == NUM_PLAYERS {
        // Byte-identical 6-max path (the trained encoding); offset is already in 0..6.
        POS_BY_OFFSET[offset]
    } else {
        // Deterministic in-range generalization: BTN(offset 0)=N-1, SB(1)=N-2, BB(2)=N-3, then
        // UTG.. ascending from 0. Equivalent to the 6-map's shape ([3,4,5,0,1,2] = btn-relative late
        // positions high, early positions low) generalized to width N. Only ever hit when N != 6,
        // where no blueprint row exists for the key (uniform fallback), so this just must be stable.
        let o = (offset % N) as i32;
        if o <= 2 {
            (N as i32) - 1 - o
        } else {
            o - 3
        }
    }
}

// betting_abstraction_key bin edges (jax_weva_fast).
const POT_EDGES: [f64; 7] = [3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 200.0];
const TOCALL_EDGES: [f64; 5] = [0.01, 0.34, 0.67, 1.01, 2.01];
const SPR_EDGES: [f64; 5] = [1.0, 2.5, 5.0, 10.0, 20.0];

#[inline]
fn bucket_edge(x: f64, edges: &[f64]) -> u8 {
    // Python `_bucket`: count leading edges with x >= e (monotone), stop at first x < e.
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

/// Port of `canonical_preflop_class(c0, c1)`. card id = rank_ordinal*4 + suit.
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

/// Preflop weva_id under POKERBOT_PREFLOP_LOSSLESS: pos_enum*169 + canonical class.
/// Mirrors `_weva_bucket_from_holes` over the lossless table; returns 0 if any hole < 0.
fn weva_id_lossless(h0: i32, h1: i32, pos_enum: i32) -> i32 {
    if h0 < 0 || h1 < 0 {
        return 0;
    }
    let (lo, hi) = if h0 <= h1 { (h0, h1) } else { (h1, h0) };
    pos_enum * PREFLOP_NUM_CLASSES + canonical_preflop_class(lo, hi)
}

/// Port of `betting_abstraction_key(state, seat)` -> 6 bytes. Generic over seat count `N`
/// (the N=6 monomorphization is byte-identical to the pre-generic 6-seat code: same loop bounds,
/// same ops — only the loop upper bound `N` is a const, == 6 at the default instantiation).
pub fn betting_abstraction_key<const N: usize>(s: &EnvStateN<N>, seat: usize) -> [u8; 6] {
    let mut committed_sum: i64 = 0;
    for i in 0..N {
        committed_sum += s.committed_street[i] as i64;
    }
    let pot = s.pot_dead as i64 + committed_sum;
    let cur = s.current_bet as i64;
    let to_call = (cur - s.committed_street[seat] as i64).max(0);
    let street = s.street as i64;
    let button = s.button as i64;
    let last_agg = s.last_aggressor as i64;

    let pot_bb = pot as f64 / CHIP_UNIT;
    // Effective stack = min live stack (live = status not in {empty, folded}).
    let mut eff_chips: Option<i64> = None;
    for i in 0..N {
        let st = s.seat_status[i];
        if st != SEAT_EMPTY && st != SEAT_FOLDED {
            let v = s.stacks[i] as i64;
            eff_chips = Some(match eff_chips {
                Some(m) => m.min(v),
                None => v,
            });
        }
    }
    let eff = eff_chips.unwrap_or(0) as f64 / CHIP_UNIT;

    let pot_b = bucket_edge(pot_bb, &POT_EDGES);
    // to_call / max(pot, 1)
    let tocall_ratio = to_call as f64 / (pot.max(1)) as f64;
    let tocall_b = bucket_edge(tocall_ratio, &TOCALL_EDGES);
    // eff / max(pot_bb, 1e-6)
    let spr_ratio = eff / pot_bb.max(1e-6);
    let spr_b = bucket_edge(spr_ratio, &SPR_EDGES);
    let is_button: u8 = if seat as i64 == button { 1 } else { 0 };
    let facing_agg: u8 = if last_agg >= 0 && last_agg != seat as i64 {
        1
    } else {
        0
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

// ---------------------------------------------------------------------------
// phevaluator evaluate_cards (1..7462), byte-identical to the installed package.
// ---------------------------------------------------------------------------

/// Port of `hash_quinary(quinary, num_cards)`. quinary is 13 rank-counts; length is always 13.
fn hash_quinary(quinary: &[u8; 13], num_cards_in: i32) -> usize {
    let mut sum_numb: u32 = 0;
    let length: i32 = 13;
    let mut num_cards = num_cards_in;
    for rank in 0..13usize {
        let cnt = quinary[rank] as i32;
        if cnt != 0 {
            // DP[cnt][length - rank - 1][num_cards]
            let i = cnt as usize;
            let j = (length - rank as i32 - 1) as usize;
            let k = num_cards as usize;
            sum_numb += phe_tables::DP[i][j][k];
            num_cards -= cnt;
        }
    }
    sum_numb as usize
}

/// Port of `_evaluate_cards(*cards)` for 5..7 raw card ids (rank*4 + suit). Returns 1..7462.
pub fn evaluate_cards(cards: &[i32]) -> i32 {
    let hand_size = cards.len();
    let mut suit_hash: usize = 0;
    for &c in cards {
        suit_hash += phe_tables::SUITBIT_BY_ID[c as usize] as usize;
    }
    let flush_suit = phe_tables::SUITS[suit_hash] - 1;
    if flush_suit != -1 {
        let mut hand_binary: usize = 0;
        for &c in cards {
            if c % 4 == flush_suit {
                hand_binary |= phe_tables::BINARIES_BY_ID[c as usize] as usize;
            }
        }
        return phe_tables::FLUSH[hand_binary] as i32;
    }
    let mut quinary = [0u8; 13];
    for &c in cards {
        quinary[(c / 4) as usize] += 1;
    }
    let h = hash_quinary(&quinary, hand_size as i32);
    let v = match hand_size {
        5 => phe_tables::NO_FLUSH_5[h],
        6 => phe_tables::NO_FLUSH_6[h],
        7 => phe_tables::NO_FLUSH_7[h],
        _ => unreachable!("phe evaluate_cards supports 5..7 cards"),
    };
    v as i32
}

/// Port of `fast_made_hand_bucket((h0,h1), board, num_buckets)`. board are the up-to-5 board ids.
/// Returns -1 preflop (<3 board cards). num_buckets is 200 in the eval config.
pub fn fast_made_hand_bucket(h0: i32, h1: i32, board: &[i32], num_buckets: i32) -> i32 {
    // Collect valid board cards (>=0). Python sorts hole and board, but evaluate_cards is order-
    // invariant, so sorting is unnecessary for the rank. Keep only >=0.
    let mut cards: Vec<i32> = Vec::with_capacity(7);
    cards.push(h0);
    cards.push(h1);
    let mut nb = 0;
    for &c in board {
        if c >= 0 {
            cards.push(c);
            nb += 1;
        }
    }
    if nb < 3 || h0 < 0 || h1 < 0 {
        return -1;
    }
    let rank = evaluate_cards(&cards);
    let bucket = ((rank - 1) * num_buckets / 7462).min(num_buckets - 1);
    bucket
}

/// compute_postflop_bucket: -1 if <3 board cards or invalid holes, else fast_made_hand_bucket.
pub fn compute_postflop_bucket(h0: i32, h1: i32, board: &[i32], num_buckets: i32) -> i32 {
    let nb = board.iter().filter(|&&c| c >= 0).count();
    if nb < 3 {
        return -1;
    }
    if h0 < 0 || h1 < 0 {
        return -1;
    }
    fast_made_hand_bucket(h0, h1, board, num_buckets)
}

// ---------------------------------------------------------------------------
// BLAKE2b (RFC 7693) with digest_size + personalization, matching hashlib.blake2b.
// Only the path needed by fingerprint_fast: keyless, salt empty, person<=16 bytes, one update.
// ---------------------------------------------------------------------------

const BLAKE2B_IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

#[inline]
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

fn compress(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
    let mut m = [0u64; 16];
    for i in 0..16 {
        let mut w = 0u64;
        for j in 0..8 {
            w |= (block[i * 8 + j] as u64) << (8 * j);
        }
        m[i] = w;
    }
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&BLAKE2B_IV);
    v[12] ^= (t & 0xFFFF_FFFF_FFFF_FFFF) as u64;
    v[13] ^= (t >> 64) as u64;
    if last {
        v[14] ^= 0xFFFF_FFFF_FFFF_FFFF;
    }
    for r in 0..12 {
        let s = &SIGMA[r];
        g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// blake2b(digest_size, person) of a single message. Matches hashlib.blake2b for keyless,
/// no-salt, person<=16 bytes. `out_len` <= 64.
pub fn blake2b(out_len: usize, person: &[u8], data: &[u8]) -> Vec<u8> {
    let mut h = BLAKE2B_IV;
    // Parameter block: h[0] ^= 0x0101kknn where nn=digest_len, kk=keylen(0), fanout=depth=1.
    h[0] ^= 0x0101_0000 ^ (out_len as u64);
    // Personalization occupies param bytes 48..64 -> words h[6] (48..56) and h[7] (56..64).
    let mut pbytes = [0u8; 16];
    let plen = person.len().min(16);
    pbytes[..plen].copy_from_slice(&person[..plen]);
    let mut pw0 = 0u64;
    let mut pw1 = 0u64;
    for j in 0..8 {
        pw0 |= (pbytes[j] as u64) << (8 * j);
        pw1 |= (pbytes[8 + j] as u64) << (8 * j);
    }
    h[6] ^= pw0;
    h[7] ^= pw1;

    // Process message in 128-byte blocks. blake2b always processes at least one (final) block.
    let mut t: u128 = 0;
    let n = data.len();
    let full = if n == 0 { 0 } else { (n - 1) / 128 }; // number of NON-final full blocks
    let mut offset = 0;
    for _ in 0..full {
        let mut block = [0u8; 128];
        block.copy_from_slice(&data[offset..offset + 128]);
        t += 128;
        compress(&mut h, &block, t, false);
        offset += 128;
    }
    // Final block (may be empty -> all zero, t stays at total len).
    let mut block = [0u8; 128];
    let rem = n - offset;
    block[..rem].copy_from_slice(&data[offset..n]);
    t += rem as u128;
    compress(&mut h, &block, t, true);

    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        out.push((h[i / 8] >> (8 * (i % 8))) as u8);
    }
    out
}

// ---------------------------------------------------------------------------
// fingerprint_fast (bounded config) -> 16-byte digest
// ---------------------------------------------------------------------------

/// Byte-identical `fingerprint_fast(state, seat)` for the bounded eval config (16-byte key).
/// Generic over seat count `N` (N=6 monomorphization byte-identical to the pre-generic code).
pub fn fingerprint_fast<const N: usize>(s: &EnvStateN<N>, seat: usize, num_buckets: i32) -> [u8; 16] {
    fingerprint_fast_depth(s, seat, num_buckets, &[])
}

/// `effective_depth_bb`: min over non-folded seats (with positive total) of (stacks+committed_total),
/// in BB. Mirrors `jax_weva_fast.effective_depth_bb` including its fallbacks. Generic over `N`.
fn effective_depth_bb<const N: usize>(s: &EnvStateN<N>) -> f64 {
    let mut m: Option<i64> = None;
    for i in 0..N {
        let st = s.seat_status[i];
        if st != SEAT_EMPTY && st != SEAT_FOLDED {
            let total = s.stacks[i] as i64 + s.committed_total[i] as i64;
            if total > 0 {
                m = Some(match m {
                    Some(x) => x.min(total),
                    None => total,
                });
            }
        }
    }
    let chips = match m {
        Some(x) => x as f64,
        None => {
            // fallback: max over all seats' totals (empty seats are 0); else 100bb.
            let mut mx: i64 = 0;
            let mut any = false;
            for i in 0..N {
                let total = s.stacks[i] as i64 + s.committed_total[i] as i64;
                if !any || total > mx {
                    mx = total;
                    any = true;
                }
            }
            if any {
                mx as f64
            } else {
                100.0 * CHIP_UNIT
            }
        }
    };
    chips / CHIP_UNIT
}

/// Depth-aware fingerprint (`POKERBOT_DEPTH_BUCKETS=1`). When `depth_edges` is non-empty, the PREFLOP
/// key gains `b"D" + depth_bucket(1 byte)` between weva_id and the betting_state -- byte-identical to
/// `NLHEEncodeBundle._digest` with `_DEPTH_BUCKETS`. Pass `&[]` for the non-depth blueprint (unchanged).
pub fn fingerprint_fast_depth<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    num_buckets: i32,
    depth_edges: &[f64],
) -> [u8; 16] {
    let button = s.button;
    let offset = (((seat as i32 - button) % N as i32 + N as i32) % N as i32) as usize;
    let pos_enum = pos_by_offset::<N>(offset);
    let h0 = s.holes[seat][0];
    let h1 = s.holes[seat][1];
    let weva_id = weva_id_lossless(h0, h1, pos_enum);
    let bucket = compute_postflop_bucket(h0, h1, &s.board, num_buckets);
    let betting_state = betting_abstraction_key(s, seat);

    let mut msg: Vec<u8> = Vec::with_capacity(1 + 4 + 2 + 1 + 6);
    if bucket >= 0 {
        msg.push(b'B');
        msg.extend_from_slice(&bucket.to_le_bytes());
    } else {
        msg.push(b'w');
        msg.extend_from_slice(&weva_id.to_le_bytes());
        // Depth byte: preflop only, when depth bucketing is on.
        if !depth_edges.is_empty() {
            let dbucket = bucket_edge(effective_depth_bb(s), depth_edges);
            msg.push(b'D');
            msg.push(dbucket);
        }
    }
    msg.push(b'S');
    msg.extend_from_slice(&betting_state);

    let d = blake2b(16, b"", &msg);
    let mut out = [0u8; 16];
    out.copy_from_slice(&d);
    out
}
