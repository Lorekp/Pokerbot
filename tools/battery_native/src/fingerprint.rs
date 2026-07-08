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
use rustc_hash::FxHashMap as HashMap;

pub const PREFLOP_NUM_CLASSES: i32 = 169;
pub const CHIP_UNIT: f64 = 100.0; // matches jax_engine.CHIP_UNIT (BB_CHIPS)
pub type FpMemo = HashMap<Vec<u8>, [u8; 16]>;

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
    let h = blake2b_state(out_len, person, data);
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        out.push((h[i / 8] >> (8 * (i % 8))) as u8);
    }
    out
}

/// 16-byte fixed-output blake2b (R3, perf): identical state math to `blake2b(16, ..)` (both call
/// `blake2b_state`) but returns the digest in a stack array — no heap Vec per call. The trainer-hot
/// fingerprint paths emit ~3M digests/iter at nd=8192; this removes that allocation entirely.
#[inline]
pub fn blake2b16(person: &[u8], data: &[u8]) -> [u8; 16] {
    let h = blake2b_state(16, person, data);
    let mut out = [0u8; 16];
    for (i, o) in out.iter_mut().enumerate() {
        *o = (h[i / 8] >> (8 * (i % 8))) as u8;
    }
    out
}

/// The blake2b core shared by `blake2b` / `blake2b16` — byte-for-byte the pre-R3 `blake2b` body up
/// to (but excluding) the output serialization.
fn blake2b_state(out_len: usize, person: &[u8], data: &[u8]) -> [u64; 8] {
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
    h
}

// ---------------------------------------------------------------------------
// fingerprint_fast (bounded config) -> 16-byte digest
// ---------------------------------------------------------------------------

/// Byte-identical `fingerprint_fast(state, seat)` for the bounded eval config (16-byte key).
/// Generic over seat count `N` (N=6 monomorphization byte-identical to the pre-generic code).
pub fn fingerprint_fast<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    num_buckets: i32,
) -> [u8; 16] {
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
    let h0 = s.holes[seat][0];
    let h1 = s.holes[seat][1];
    let bucket = compute_postflop_bucket(h0, h1, &s.board, num_buckets);
    fingerprint_from_bucket(s, seat, bucket, depth_edges)
}

// ---------------------------------------------------------------------------
// EXPERIMENT (preflop mispricing diagnosis, 2026-07-03): opt-in preflop HISTORY key extension.
//
// Root-cause finding: the 6-byte betting_abstraction_key ALIASES strategically distinct preflop
// histories into one infoset (folded-to-hero == after-a-limper; BB-vs-UTG-open == BB-vs-BTN-open;
// facing-4bet-jam == facing-limp-trap-jam). The trained equilibrium of that aliased game is the
// ultra-tight/limpy/trap-heavy preflop (AA open-limps BTN 93%). This opt-in extension appends
// b"H" + [n_limps_before_first_raise, n_raises (cap 3), first_raiser_pos_enum + 1] to PREFLOP keys
// so the trainer can distinguish those histories. Gated by POKERBOT_PREFLOP_HIST_KEY=1 read ONCE;
// default OFF is byte-identical to the shipped key space (golden digest tests unaffected).
// TRAIN/SERVE SKEW WARNING: a checkpoint trained with this ON must be served with matching keying.
// SERVE WIRING (blueprint-v3 ship prep): the Python twin is jax_weva_fast::preflop_hist_bytes +
// the b"H" branch in jax_weva_encode.NLHEEncodeBundle._digest (parity-gated by
// scripts/probe_preflop_hist_key_parity.py + tests/test_preflop_hist_key_parity.py); the flag is
// stamped via active_abstraction_config()["preflop_hist_key"] and two-sided-guarded in
// lbr.check_blueprint_serve_env; native_caps() advertises "preflop_hist_key" so stale .pyds fail
// closed in native_solver. Keep all of those in lockstep with any change here.
// ---------------------------------------------------------------------------

pub fn preflop_hist_key_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("POKERBOT_PREFLOP_HIST_KEY").map(|v| v == "1").unwrap_or(false)
    })
}

// ---------------------------------------------------------------------------
// EXPERIMENT (blueprint-v4 Stage 0, 2026-07-04): opt-in FULL action-HISTORY key extension.
//
// Design of record: artifacts/blueprint_pluribus_grade_design.md §7.1 (Stage-0 causal mini-proof).
// The bp3 campaign proved the 6-byte betting-STATE summary key causally warps equilibria and that
// de-aliased PREFLOP keys alone are insufficient (the POSTFLOP rows stay aliased: limped vs raised
// pots collide, initiative/sequence invisible — bp3_gate_verdict.md, bp3_forced_response_probe.md).
// This opt-in extension appends, on EVERY street (both the preflop b"w" and postflop b"B" branches):
//     b"A" + street(1 byte) + one raw engine action id byte per hist_action[0..hist_len]
// i.e. exact in-abstraction action-history keying (perfect recall of abstracted actions — the
// Pluribus construction). NO street separators inside the byte run: the engine's abstract tree is
// DETERMINISTIC (fixed 100bb stacks, deterministic blinds/dealing order), so the (current-street,
// action-prefix) pair identifies the node's street boundaries unambiguously; the leading street
// byte pins the current street explicitly (per-entry streets are not stored in EnvStateN).
// The b"S"+betting6 tail is kept UNCHANGED (harmless redundancy; keeps the default-OFF layout
// golden-identical and keeps preflop_root_diag-style direct-key extraction reusable).
//
// Gated by POKERBOT_FULLHIST_KEY=1 read ONCE (OnceLock); default OFF is byte-identical to the
// shipped key space (golden digest tests unaffected).
// TRAIN/SERVE SKEW WARNING: a checkpoint trained with this ON must be served with matching keying
// (Python twin: jax_weva_fast.full_hist_bytes + the b"A" branch in
// jax_weva_encode.NLHEEncodeBundle._digest, parity-gated by scripts/probe_fullhist_key_parity.py +
// tests/test_fullhist_key_parity.py). native_caps() advertises "full_hist_key" so stale .pyds can
// fail closed. This is the Stage-0 EXPERIMENTAL key (the bp3 POKERBOT_PREFLOP_HIST_KEY pattern),
// NOT the Stage-1 infoset.rs module.
// ---------------------------------------------------------------------------

pub fn full_hist_key_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("POKERBOT_FULLHIST_KEY").map(|v| v == "1").unwrap_or(false)
    })
}

// ---------------------------------------------------------------------------
// EXPERIMENT (blueprint-v4 arm-4, 2026-07-08): opt-in POSTFLOP PREFLOP-SUMMARY key ("presum").
//
// Design of record: artifacts/bpv4_arm4_design_and_review.md §35 (amendment). Measurements proved
// the fullhist POSTFLOP key's carrying of the EXACT PREFLOP action-history prefix is what makes the
// reached postflop set unbounded/uncoverable (menu-thinning 1.29x and recall-bounding 1.003x both
// failed to bend the curve). `presum` fixes exactly that: keep preflop DECISIONS exact (fullhist),
// but key POSTFLOP infosets on a low-cardinality preflop POT-TYPE SUMMARY instead of the exact
// preflop action sequence; postflop actions stay EXACT. This is the only lever that bends the
// growth curve (6.25x @20k, sub-linear per the census).
//
// It ONLY takes effect when POKERBOT_FULLHIST_KEY is ALSO on (presum is a REFINEMENT of the
// fullhist A-segment, never a standalone key): under `full_hist_key_enabled() && presum_key_enabled()`
// AND a POSTFLOP node (s.street >= 1), the A-segment becomes
//     b"A" + street(1 byte) + [4 preflop-summary bytes] + [EXACT postflop action bytes]
// where the 4 summary bytes = (n_calls_before_first_raise cap3, n_raises cap3, opener_pos_enum+1,
// n_active) computed over ONLY hist_action[0..n_preflop] (+ n_active over the CURRENT state), and the
// postflop bytes = s.hist_action[i] for i in n_preflop..min(hist_len, MAX_HISTORY) (unchanged bytes).
// For PREFLOP nodes (street==0), and whenever presum is OFF, the A-segment is the CURRENT fullhist
// bytes UNCHANGED (byte-identical). The b"S"+betting6 tail is unchanged everywhere.
//
// Gated by POKERBOT_PRESUM_KEY=1 read ONCE (OnceLock); default OFF is byte-identical to the fullhist
// key space (and thus to the shipped key space when fullhist is also off). TRAIN/SERVE SKEW WARNING:
// a checkpoint trained with this ON must be served with matching keying (Python twin:
// jax_weva_fast.presum_hist_bytes + the presum branch in jax_weva_encode.NLHEEncodeBundle._digest,
// parity-gated by scripts/probe_fullhist_key_parity.py --presum + tests/test_fullhist_key_parity.py).
// native_caps() advertises "presum_key" so stale .pyds fail closed. Two-sided serve guard in
// lbr.check_blueprint_serve_env + stamp active_abstraction_config()["presum_key"].
// ---------------------------------------------------------------------------

pub fn presum_key_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("POKERBOT_PRESUM_KEY").map(|v| v == "1").unwrap_or(false)
    })
}

/// Minimal byte-sink abstraction so the message-assembly helpers emit IDENTICAL bytes into either a
/// heap `Vec<u8>` (serve paths, unchanged) or the stack `MsgBuf` (R3 trainer-hot paths). Method
/// names/semantics match `Vec`'s inherent ones, so the `Vec` impl is a pure passthrough.
trait PushBytes {
    fn push(&mut self, b: u8);
    fn extend_from_slice(&mut self, s: &[u8]);
}
impl PushBytes for Vec<u8> {
    #[inline]
    fn push(&mut self, b: u8) {
        Vec::push(self, b);
    }
    #[inline]
    fn extend_from_slice(&mut self, s: &[u8]) {
        Vec::extend_from_slice(self, s);
    }
}

/// Stack-allocated fingerprint message buffer (R3, perf, byte-identity-neutral): the trainer's hot
/// fingerprint paths assembled every message in a fresh heap `Vec<u8>` (~3M alloc+free/iter at
/// nd=8192, all under the process allocator's thread contention). Message length is statically
/// bounded: worst case = b"w" branch 1+4 + D 2 + H 4 + fullhist A 2+MAX_HISTORY(64) + S 1+6 = 84
/// bytes; 128 leaves headroom (an overflow would panic via the slice bound — impossible by the
/// bound above, and a panic is loud, never a silent truncation). Bytes emitted are IDENTICAL to the
/// Vec path (same PushBytes calls), so digests — and every trained value — are unchanged.
const MSG_MAX: usize = 128;
struct MsgBuf {
    buf: [u8; MSG_MAX],
    len: usize,
}
impl MsgBuf {
    #[inline]
    fn new() -> Self {
        MsgBuf { buf: [0; MSG_MAX], len: 0 }
    }
    #[inline]
    fn as_slice(&self) -> &[u8] {
        &self.buf[..self.len]
    }
}
impl PushBytes for MsgBuf {
    #[inline]
    fn push(&mut self, b: u8) {
        self.buf[self.len] = b;
        self.len += 1;
    }
    #[inline]
    fn extend_from_slice(&mut self, s: &[u8]) {
        self.buf[self.len..self.len + s.len()].copy_from_slice(s);
        self.len += s.len();
    }
}

/// Append the full-history segment: b"A" + current street byte + raw action-id bytes.
/// Caps at the engine's MAX_HISTORY (the engine itself stops recording past it — documented,
/// accepted aliasing of astronomically rare saturated tails).
#[inline]
fn push_full_hist_bytes<const N: usize>(s: &EnvStateN<N>, msg: &mut impl PushBytes) {
    msg.push(b'A');
    msg.push(s.street as u8);
    let len = (s.hist_len.max(0) as usize).min(crate::core::MAX_HISTORY);
    for i in 0..len {
        msg.push(s.hist_action[i] as u8);
    }
}

/// [n_calls_before_first_raise (cap 3), n_raises (cap 3), first_raiser_pos_enum + 1 (0 = none)].
/// Caller is preflop-only (street==0), so every history entry is a preflop action; blind posts are
/// not history entries (`init_hand_n` starts hist_len=0). Folds/checks are ignored.
pub fn preflop_hist_bytes<const N: usize>(s: &EnvStateN<N>) -> [u8; 3] {
    let mut n_calls: u8 = 0;
    let mut n_raises: u8 = 0;
    let mut opener_seat: i32 = -1;
    let len = (s.hist_len.max(0) as usize).min(crate::core::MAX_HISTORY);
    for i in 0..len {
        let a = s.hist_action[i];
        if a == crate::core::CALL as i32 {
            if n_raises == 0 && n_calls < 3 {
                n_calls += 1;
            }
        } else if a >= crate::core::BET_BUCKET_START as i32 && a <= crate::core::ALL_IN as i32 {
            if n_raises == 0 {
                opener_seat = s.hist_seat[i];
            }
            if n_raises < 3 {
                n_raises += 1;
            }
        }
    }
    let opener_pos: i32 = if opener_seat >= 0 {
        let offset = (((opener_seat - s.button) % N as i32 + N as i32) % N as i32) as usize;
        pos_by_offset::<N>(offset)
    } else {
        -1
    };
    [n_calls, n_raises, (opener_pos + 1) as u8]
}

// ---------------------------------------------------------------------------
// presum (blueprint-v4 arm-4) helpers.
// ---------------------------------------------------------------------------

/// n_active = count of seats with seat_status in {SEAT_LIVE, SEAT_ALLIN} in the CURRENT state.
/// Matches core::eligible_to_win / the Python census twin's `int(seat_status) in (0,1)` where the
/// census enum maps SEAT_LIVE/SEAT_ALLIN to those two values; here we compare the core.rs sentinels
/// directly (SEAT_LIVE=2, SEAT_ALLIN=3). Capped at u8 (N<=7, never overflows).
#[inline]
fn presum_n_active<const N: usize>(s: &EnvStateN<N>) -> u8 {
    let mut n: u8 = 0;
    for i in 0..N {
        let st = s.seat_status[i];
        if st == crate::core::SEAT_LIVE || st == crate::core::SEAT_ALLIN {
            n += 1;
        }
    }
    n
}

/// The 4-byte presum PREFLOP pot-type summary computed over ONLY the first `n_preflop` history
/// entries: [n_calls_before_first_raise (cap 3), n_raises (cap 3), opener_pos_enum + 1, n_active].
/// The first 3 bytes reuse `preflop_hist_bytes`' EXACT logic but restricted to hist_action[0..n_pre]
/// (a preflop-only prefix). The 4th byte is `presum_n_active` over the CURRENT state (post-fold).
fn preflop_summary_bytes<const N: usize>(s: &EnvStateN<N>, n_preflop: usize) -> [u8; 4] {
    let mut n_calls: u8 = 0;
    let mut n_raises: u8 = 0;
    let mut opener_seat: i32 = -1;
    let len = n_preflop.min(crate::core::MAX_HISTORY);
    for i in 0..len {
        let a = s.hist_action[i];
        if a == crate::core::CALL as i32 {
            if n_raises == 0 && n_calls < 3 {
                n_calls += 1;
            }
        } else if a >= crate::core::BET_BUCKET_START as i32 && a <= crate::core::ALL_IN as i32 {
            if n_raises == 0 {
                opener_seat = s.hist_seat[i];
            }
            if n_raises < 3 {
                n_raises += 1;
            }
        }
    }
    let opener_pos: i32 = if opener_seat >= 0 {
        let offset = (((opener_seat - s.button) % N as i32 + N as i32) % N as i32) as usize;
        pos_by_offset::<N>(offset)
    } else {
        -1
    };
    [n_calls, n_raises, (opener_pos + 1) as u8, presum_n_active(s)]
}

/// The presum PREFLOP pot-type of a state, exposed for the K-COV on-policy coverage instrument
/// (blueprint-v4 blocker D). Returns `(n_calls, n_raises, opener_pos_enum, n_active)` computed over
/// the reconstructed preflop prefix — the SAME classification `push_presum_hist_bytes` keys on, so
/// K-COV subclasses coverage in EXACTLY the pot-type buckets presum collapses. `n_raises == 0` is a
/// limped/walk pot (opener_pos = -1); `n_raises >= 1` is a RAISED ("opener-faced") pot — the region
/// the limp-trap starves. Valid on any state (it replays the preflop history through the real engine
/// via `n_preflop_boundary`; on a preflop state it summarizes the actions taken so far). Cost is
/// O(n_preflop) apply_action calls — call it ONCE per hand (the pot-type is fixed once preflop
/// closes) and reuse. Independent of the key space (pure history count), so it classifies correctly
/// whether or not presum/fullhist is enabled.
pub fn preflop_pot_type<const N: usize>(s: &EnvStateN<N>) -> (u8, u8, i8, u8) {
    let n_preflop = n_preflop_boundary(s);
    let [n_calls, n_raises, opener_plus1, n_active] = preflop_summary_bytes(s, n_preflop);
    (n_calls, n_raises, (opener_plus1 as i8) - 1, n_active)
}

/// Recover `n_preflop` = the number of leading `hist_action` entries that were PREFLOP decisions
/// (the count of actions taken before the preflop betting round first closed). Per-entry street is
/// NOT stored in EnvStateN, so this REPLAYS the recorded history through the REAL engine
/// (`core::apply_action`, which reuses `core::street_complete`) — zero divergence from the trainer's
/// own transitions. Cost: O(n_preflop) apply_action calls, only at postflop nodes.
///
/// Reconstruction: the starting stack of seat i is `stacks[i] + committed_total[i]` (undo every chip
/// this seat committed across all streets — with ante=0 the engine keeps committed_total == the sum
/// of every street commit, so this is the pre-blind stack). Re-init the hand root via
/// `core::init_hand_n` with the SAME button, those starting stacks, the SAME occupancy (seats that
/// were ever dealt in = status != SEAT_EMPTY), ante=0, a DUMMY deck [0..52], and the SAME rng_key.
/// Then replay hist_action[0..min(hist_len,MAX_HISTORY)] one at a time; STOP at the first action
/// after which the replayed state has `street >= 1` OR `pending_deal` is set (the preflop round just
/// closed — this happens BEFORE any board card is dealt, so the dummy deck is never drawn for the
/// board). Returns i+1 at that first close. If the loop never closes preflop (all recorded actions
/// are preflop, e.g. a preflop all-in fold-out that terminated), returns the full replayed length.
///
/// Debug builds assert the engine picks the SAME acting seat as the recorded `hist_seat[i]`
/// (a cheap zero-divergence check; release builds skip it).
fn n_preflop_boundary<const N: usize>(s: &EnvStateN<N>) -> usize {
    let hist_len = (s.hist_len.max(0) as usize).min(crate::core::MAX_HISTORY);
    if hist_len == 0 {
        return 0;
    }
    // Starting stacks = current stack + everything this seat put in the pot (all streets). ante=0 in
    // the trainer, so committed_total is exactly the reconstructable pre-blind commitment.
    let mut start_stacks = [0i32; N];
    let mut occupancy = [false; N];
    for i in 0..N {
        start_stacks[i] = s.stacks[i] + s.committed_total[i];
        // A seat was in the hand iff it is not EMPTY now (folded/live/all-in were all dealt in).
        occupancy[i] = s.seat_status[i] != crate::core::SEAT_EMPTY;
    }
    let dummy_deck: [i32; 52] = std::array::from_fn(|i| i as i32);
    let mut r = crate::core::init_hand_n::<N>(
        s.button,
        &start_stacks,
        &occupancy,
        0, // ante: trainer is ante-free (committed_total == committed reconstruction relies on this)
        &dummy_deck,
        s.rng_key,
    );
    for i in 0..hist_len {
        debug_assert!(
            r.action_seat == s.hist_seat[i],
            "presum n_preflop replay diverged: engine action_seat {} != recorded hist_seat {} at i={}",
            r.action_seat,
            s.hist_seat[i],
            i
        );
        let action = s.hist_action[i];
        // Guard against a corrupt/sentinel action id (shouldn't happen for recorded history).
        if !(0..crate::core::NUM_ACTIONS as i32).contains(&action) {
            return hist_len;
        }
        r = crate::core::apply_action::<N>(&r, action as usize);
        if r.street >= crate::core::STREET_FLOP || r.pending_deal {
            // The preflop round closed at this action (the FIRST street boundary, before any board
            // card is dealt — close_street sets pending_deal=true with street still 0, and a preflop
            // all-in run-out would set street>=1). n_preflop = the count of actions up to and incl. i.
            return i + 1;
        }
    }
    hist_len
}

/// Append the presum action-history segment for a POSTFLOP node: b"A" + current street byte +
/// [4 preflop-summary bytes] + [EXACT postflop action bytes]. Only called when
/// `full_hist_key_enabled() && presum_key_enabled()` AND `s.street >= 1`.
#[inline]
fn push_presum_hist_bytes<const N: usize>(s: &EnvStateN<N>, msg: &mut impl PushBytes) {
    msg.push(b'A');
    msg.push(s.street as u8);
    let n_preflop = n_preflop_boundary(s);
    msg.extend_from_slice(&preflop_summary_bytes(s, n_preflop));
    let len = (s.hist_len.max(0) as usize).min(crate::core::MAX_HISTORY);
    // EXACT postflop action bytes (unchanged raw bytes), i in n_preflop..hist_len.
    for i in n_preflop..len {
        msg.push(s.hist_action[i] as u8);
    }
}

/// The A-segment dispatcher: presum (postflop, both flags on) or the plain fullhist bytes. Callers
/// invoke this ONLY when `full_hist_key_enabled()` (the outer flag). When presum is off, or the node
/// is preflop (street==0), this is byte-identical to `push_full_hist_bytes` (the shipped fullhist
/// layout) — so the fullhist golden config stays byte-identical.
#[inline]
fn push_hist_bytes<const N: usize>(s: &EnvStateN<N>, msg: &mut impl PushBytes) {
    if presum_key_enabled() && s.street >= crate::core::STREET_FLOP {
        push_presum_hist_bytes(s, msg);
    } else {
        push_full_hist_bytes(s, msg);
    }
}

/// The digest for an ALREADY-RESOLVED postflop bucket id (or `bucket < 0` => the preflop `b"w"`
/// branch, including the depth byte when enabled). Pure code motion out of `fingerprint_fast_depth`
/// — the message assembly is byte-identical to the pre-refactor body (golden digest tests pin it).
/// This is the shared tail that lets the EMD keying substitute the frozen-lookup bucket while
/// reusing the exact made-hand byte layout.
pub fn fingerprint_from_bucket<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    bucket: i32,
    depth_edges: &[f64],
) -> [u8; 16] {
    let button = s.button;
    let offset = (((seat as i32 - button) % N as i32 + N as i32) % N as i32) as usize;
    let pos_enum = pos_by_offset::<N>(offset);
    let h0 = s.holes[seat][0];
    let h1 = s.holes[seat][1];
    let weva_id = weva_id_lossless(h0, h1, pos_enum);
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
        if preflop_hist_key_enabled() {
            msg.push(b'H');
            msg.extend_from_slice(&preflop_hist_bytes(s));
        }
    }
    // blueprint-v4 Stage 0: full action-history segment (both branches; see block comment above).
    if full_hist_key_enabled() {
        // presum (arm-4): postflop nodes get the pot-type-summary A-segment when POKERBOT_PRESUM_KEY
        // is also on; preflop nodes and presum-off are byte-identical to push_full_hist_bytes.
        push_hist_bytes(s, &mut msg);
    }
    msg.push(b'S');
    msg.extend_from_slice(&betting_state);

    let d = blake2b(16, b"", &msg);
    let mut out = [0u8; 16];
    out.copy_from_slice(&d);
    out
}

/// EMD-mode fingerprint for the SERVE fast paths (subgame / rollout / LBR / fingerprint export) —
/// the serve twin of the trainer's `fingerprint_fast_depth_memo_emd` (trainer.rs), which keyed the
/// promoted EMD-K160 blueprint and is therefore the ground truth for what the blueprint rows were
/// written under. Byte-parity target: `jax_weva_fast.fingerprint_fast` with
/// `POKERBOT_POSTFLOP_MODE=emd` (`jax_weva_encode.compute_postflop_bucket`):
///   * flop/turn (3-4 board cards): bucket = the FROZEN lookup id -> the postflop `b"B"` key;
///   * frozen MISS (river always — it is not in the blob — or an off-blob flop/turn spot):
///       - `miss_made_hand=true`  (POKERBOT_EMD_FROZEN_MIRROR_FALLBACK=1 / silent fallback /
///         fail-on-miss OFF — the deployed serve regime): made-hand bucket, exactly what the
///         trainer keyed the river with (the "mirror rule");
///       - `miss_made_hand=false` (fail-on-miss, no mirror): Python returns bucket=-1, so the
///         digest takes the preflop `b"w"` branch — replicated via `fingerprint_from_bucket(-1)`.
///   * preflop / invalid holes: identical to the made-hand path (the branches coincide).
pub fn fingerprint_fast_depth_emd<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    num_buckets: i32,
    depth_edges: &[f64],
    lookup: &crate::emd::FrozenEMDLookup,
    miss_made_hand: bool,
) -> [u8; 16] {
    let h0 = s.holes[seat][0];
    let h1 = s.holes[seat][1];
    let nb = s.board.iter().filter(|&&c| c >= 0).count();
    if nb >= 3 && h0 >= 0 && h1 >= 0 {
        if nb <= 4 {
            if let Some(bucket) = crate::emd::frozen_bucket_memoed(lookup, &s.holes[seat], &s.board)
            {
                return fingerprint_fast_depth_postflop_bucket(s, seat, bucket);
            }
        }
        // Frozen miss: river (never in the blob) or a true flop/turn blob miss.
        if miss_made_hand {
            return fingerprint_fast_depth(s, seat, num_buckets, depth_edges);
        }
        return fingerprint_from_bucket(s, seat, -1, depth_edges);
    }
    fingerprint_fast_depth(s, seat, num_buckets, depth_edges)
}

/// Digest a stack-assembled message with the FP memo (R3: slice-probed — the memo key type
/// `Vec<u8>` borrows as `[u8]`, so the HIT path allocates NOTHING; only a memo MISS pays one
/// `to_vec` for the inserted key, exactly the allocation the old Vec-assembly path paid on every
/// call).
#[inline]
fn digest_msg_slice_memoed(msg: &[u8], memo: &mut FpMemo) -> [u8; 16] {
    // FULLHIST BYPASS (R1, perf, byte-identity-neutral): under POKERBOT_FULLHIST_KEY the message
    // space tracks the billions-scale infoset table, so the memo's measured hit rate collapses
    // (~27% at nd=8192 iter 16, arm-2b bench 2026-07-06) while every miss pays a DRAM-cold probe +
    // an insert that grows the per-thread map by ~2.1M entries/iter (~300 MB/iter process-wide —
    // the paging driver the FP_MEMO_CAP clears only bound, not remove). blake2b16 on a <100-byte
    // message is CHEAPER than the cold probe, so skip the memo entirely: the memo is a PURE
    // message->digest cache (a hit returns exactly what a miss recomputes), so the emitted digest —
    // and every trained value — is byte-identical either way (measured 2.26x wall, util 0.53->0.89).
    // Summary-key runs (flag off) keep the memo: their distinct-message count is small and the hit
    // rate ~100%, where the memo is a measured win.
    if full_hist_key_enabled() {
        return blake2b16(b"", msg);
    }
    if let Some(digest) = memo.get(msg) {
        return *digest;
    }
    let out = blake2b16(b"", msg);
    memo.insert(msg.to_vec(), out);
    out
}

pub fn fingerprint_fast_depth_memo<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    num_buckets: i32,
    depth_edges: &[f64],
    memo: &mut FpMemo,
) -> [u8; 16] {
    let button = s.button;
    let offset = (((seat as i32 - button) % N as i32 + N as i32) % N as i32) as usize;
    let pos_enum = pos_by_offset::<N>(offset);
    let h0 = s.holes[seat][0];
    let h1 = s.holes[seat][1];
    let weva_id = weva_id_lossless(h0, h1, pos_enum);
    let bucket = compute_postflop_bucket(h0, h1, &s.board, num_buckets);
    let betting_state = betting_abstraction_key(s, seat);

    // R3: stack MsgBuf instead of a heap Vec — identical PushBytes sequence, identical bytes.
    let mut msg = MsgBuf::new();
    if bucket >= 0 {
        msg.push(b'B');
        msg.extend_from_slice(&bucket.to_le_bytes());
    } else {
        msg.push(b'w');
        msg.extend_from_slice(&weva_id.to_le_bytes());
        if !depth_edges.is_empty() {
            let dbucket = bucket_edge(effective_depth_bb(s), depth_edges);
            msg.push(b'D');
            msg.push(dbucket);
        }
        if preflop_hist_key_enabled() {
            msg.push(b'H');
            msg.extend_from_slice(&preflop_hist_bytes(s));
        }
    }
    // blueprint-v4 Stage 0: full action-history segment (both branches; see block comment above).
    if full_hist_key_enabled() {
        // presum (arm-4): postflop nodes get the pot-type-summary A-segment when POKERBOT_PRESUM_KEY
        // is also on; preflop nodes and presum-off are byte-identical to push_full_hist_bytes.
        push_hist_bytes(s, &mut msg);
    }
    msg.push(b'S');
    msg.extend_from_slice(&betting_state);
    digest_msg_slice_memoed(msg.as_slice(), memo)
}

pub fn fingerprint_fast_depth_postflop_bucket<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    bucket: i32,
) -> [u8; 16] {
    let betting_state = betting_abstraction_key(s, seat);
    let mut msg: Vec<u8> = Vec::with_capacity(1 + 4 + 1 + 6);
    msg.push(b'B');
    msg.extend_from_slice(&bucket.to_le_bytes());
    // blueprint-v4 Stage 0: full action-history segment (see block comment above).
    if full_hist_key_enabled() {
        // presum (arm-4): postflop nodes get the pot-type-summary A-segment when POKERBOT_PRESUM_KEY
        // is also on; preflop nodes and presum-off are byte-identical to push_full_hist_bytes.
        push_hist_bytes(s, &mut msg);
    }
    msg.push(b'S');
    msg.extend_from_slice(&betting_state);
    let d = blake2b(16, b"", &msg);
    let mut out = [0u8; 16];
    out.copy_from_slice(&d);
    out
}

pub fn fingerprint_fast_depth_postflop_bucket_memo<const N: usize>(
    s: &EnvStateN<N>,
    seat: usize,
    bucket: i32,
    memo: &mut FpMemo,
) -> [u8; 16] {
    let betting_state = betting_abstraction_key(s, seat);
    // R3: stack MsgBuf instead of a heap Vec — identical PushBytes sequence, identical bytes.
    let mut msg = MsgBuf::new();
    msg.push(b'B');
    msg.extend_from_slice(&bucket.to_le_bytes());
    // blueprint-v4 Stage 0: full action-history segment (see block comment above).
    if full_hist_key_enabled() {
        // presum (arm-4): postflop nodes get the pot-type-summary A-segment when POKERBOT_PRESUM_KEY
        // is also on; preflop nodes and presum-off are byte-identical to push_full_hist_bytes.
        push_hist_bytes(s, &mut msg);
    }
    msg.push(b'S');
    msg.extend_from_slice(&betting_state);
    digest_msg_slice_memoed(msg.as_slice(), memo)
}

// ---------------------------------------------------------------------------
// presum (blueprint-v4 arm-4) unit tests. These exercise the PURE keying helpers directly (no env
// OnceLock flip needed — the gates route to these functions, and byte-identity-off is proven by the
// ab_train golden digest + the fullhist parity gate). Random hands are generated by playing the REAL
// engine forward (never hand-built nodes), recording each action's ORIGIN street as an INDEPENDENT
// oracle for the true preflop boundary.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod presum_tests {
    use crate::core::{
        self, apply_action, init_hand_n, is_chance_node, legal_action_mask, sample_chance, EnvState,
        NUM_ACTIONS, NUM_PLAYERS,
    };

    /// Tiny deterministic xorshift RNG (test-local; not a serve path).
    struct TestRng(u64);
    impl TestRng {
        fn new(seed: u64) -> Self {
            TestRng(seed ^ 0x9E37_79B9_7F4A_7C15)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    fn shuffled_deck(rng: &mut TestRng) -> [i32; 52] {
        let mut deck: [i32; 52] = std::array::from_fn(|i| i as i32);
        for i in (1..52).rev() {
            let j = rng.below(i + 1);
            deck.swap(i, j);
        }
        deck
    }

    /// One recorded decision node: the state, the acting seat, and the ORIGIN street.
    struct Recorded {
        state: EnvState,
        origin_street: i32,
    }

    /// Play a full 6-max hand forward, recording every DECISION node (not chance nodes) with the
    /// street the action was taken FROM. The action policy is biased toward check/call (like the
    /// parity probe's) so hands reach POSTFLOP reliably (uniform-over-legal folds out preflop too
    /// often to exercise the postflop keying). `rng` drives the deck and the choices. Returns the
    /// decision nodes visited in engine apply order.
    fn play_hand(rng: &mut TestRng, button: i32) -> Vec<Recorded> {
        let deck = shuffled_deck(rng);
        let stacks = [100 * core::CHIP_UNIT; NUM_PLAYERS];
        let occ = [true; NUM_PLAYERS];
        let mut s = init_hand_n::<NUM_PLAYERS>(button, &stacks, &occ, 0, &deck, [0u32, 0u32]);
        let mut rec = Vec::new();
        let mut guard = 0;
        while !s.done && guard < 200 {
            guard += 1;
            if is_chance_node(&s) {
                s = sample_chance(&s);
                continue;
            }
            let seat = s.action_seat;
            if seat < 0 {
                break;
            }
            let mask = legal_action_mask(&s, seat);
            let legal: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a]).collect();
            if legal.is_empty() {
                break;
            }
            rec.push(Recorded {
                state: s,
                origin_street: s.street,
            });
            // Bias: mostly check/call (reach postflop + long histories), some sized bets/raises
            // (raise chains), a little jam, few folds.
            let sized: Vec<usize> = legal
                .iter()
                .copied()
                .filter(|&a| (core::BET_BUCKET_START..core::ALL_IN).contains(&a))
                .collect();
            let u = rng.below(1000);
            let a = if u < 90 && mask[core::FOLD] && legal.len() > 1 {
                core::FOLD
            } else if u < 640 && (mask[core::CHECK] || mask[core::CALL]) {
                if mask[core::CHECK] { core::CHECK } else { core::CALL }
            } else if u < 940 && !sized.is_empty() {
                sized[rng.below(sized.len())]
            } else if mask[core::ALL_IN] && rng.below(4) == 0 {
                core::ALL_IN
            } else {
                legal[rng.below(legal.len())]
            };
            s = apply_action(&s, a);
        }
        rec
    }

    /// GATE 2 (boundary correctness): the fingerprint's reconstructed n_preflop equals the TRUE
    /// preflop action count (independently: the number of leading recorded actions taken from a
    /// street==0 state) on every decision node of many random hands, all streets.
    #[test]
    fn presum_n_preflop_boundary_matches_true_count() {
        let mut rng = TestRng::new(0xABCD_1234);
        let mut nodes = 0usize;
        let mut postflop_nodes = 0usize;
        let mut hands = 0usize;
        // >= 50k random decision states across all streets (target from the gate spec).
        while nodes < 60_000 {
            let button = (hands % 6) as i32;
            let rec = play_hand(&mut rng, button);
            hands += 1;
            for node in &rec {
                nodes += 1;
                let s = &node.state;
                let got = super::n_preflop_boundary::<NUM_PLAYERS>(s);
                // Independent truth: among the FIRST hist_len recorded actions of this hand, how many
                // had origin_street == 0. The node's hist_len == its position in `rec` (chronological
                // apply order), so the true preflop boundary is the count of leading preflop-origin
                // actions among the first hist_len recorded nodes.
                let hist_len = (s.hist_len.max(0) as usize).min(core::MAX_HISTORY);
                let true_n_pre = rec
                    .iter()
                    .take(hist_len)
                    .filter(|r| r.origin_street == 0)
                    .count();
                assert_eq!(
                    got, true_n_pre,
                    "n_preflop mismatch: got {} true {} at street {} hist_len {} hand {}",
                    got, true_n_pre, s.street, hist_len, hands
                );
                if s.street >= core::STREET_FLOP {
                    postflop_nodes += 1;
                }
            }
        }
        assert!(nodes >= 50_000, "too few nodes: {}", nodes);
        assert!(postflop_nodes >= 5_000, "too few postflop nodes: {}", postflop_nodes);
    }

    /// The raw presum A-segment for a state (b"A" + street + 4 summary + postflop bytes), assembled
    /// via the same helper the digest uses.
    fn presum_a_segment(s: &EnvState) -> Vec<u8> {
        let mut v: Vec<u8> = Vec::new();
        super::push_presum_hist_bytes::<NUM_PLAYERS>(s, &mut v);
        v
    }

    /// The plain fullhist A-segment.
    fn fullhist_a_segment(s: &EnvState) -> Vec<u8> {
        let mut v: Vec<u8> = Vec::new();
        super::push_full_hist_bytes::<NUM_PLAYERS>(s, &mut v);
        v
    }

    /// GATE 3 (presum key well-formed): on postflop states the presum A-segment is
    /// b"A" + street + 4 summary bytes + postflop bytes, the 4 summary bytes match
    /// preflop_summary_bytes over the reconstructed n_preflop, and the postflop bytes equal the raw
    /// postflop tail of the fullhist segment.
    #[test]
    fn presum_a_segment_is_well_formed() {
        let mut rng = TestRng::new(0x55AA_9931);
        let mut checked = 0usize;
        let mut hands = 0usize;
        while checked < 4_000 && hands < 100_000 {
            let rec = play_hand(&mut rng, (hands % 6) as i32);
            hands += 1;
            for node in &rec {
                let s = &node.state;
                if s.street < core::STREET_FLOP {
                    continue;
                }
                checked += 1;
                let seg = presum_a_segment(s);
                assert!(seg.len() >= 6, "presum segment too short: {:?}", seg);
                assert_eq!(seg[0], b'A');
                assert_eq!(seg[1], s.street as u8);
                let n_pre = super::n_preflop_boundary::<NUM_PLAYERS>(s);
                let summary = super::preflop_summary_bytes::<NUM_PLAYERS>(s, n_pre);
                assert_eq!(&seg[2..6], &summary[..], "summary bytes mismatch");
                let n_active = super::presum_n_active::<NUM_PLAYERS>(s);
                assert_eq!(summary[3], n_active);
                let hist_len = (s.hist_len.max(0) as usize).min(core::MAX_HISTORY);
                let expect_tail: Vec<u8> =
                    (n_pre..hist_len).map(|i| s.hist_action[i] as u8).collect();
                assert_eq!(&seg[6..], &expect_tail[..], "postflop tail mismatch");
                // presum only rewrites the PREFLOP prefix; the postflop suffix is byte-identical to
                // the fullhist segment's postflop tail (fh = b"A" + street + hist[0..hist_len]).
                let fh = fullhist_a_segment(s);
                assert_eq!(
                    &seg[6..],
                    &fh[2 + n_pre..],
                    "presum postflop tail != fullhist postflop tail"
                );
            }
        }
        assert!(checked >= 4_000, "too few postflop states checked: {}", checked);
    }

    /// GATE 3 (collapse + distinctness): two postflop states differing ONLY in preflop bet SIZES
    /// (same pot-type summary, same postflop actions) collapse to the SAME presum A-segment; two
    /// differing in postflop actions do NOT.
    #[test]
    fn presum_collapses_preflop_sizes_not_postflop() {
        // Reach the flop with a single preflop open of a given SIZE (bucket), called by BB. Both
        // sizes yield the SAME pot-type summary (1 raise, opener UTG, same n_active) but DIFFERENT
        // raw preflop action ids — presum must collapse them; fullhist must not.
        fn reach_flop_with_open(open_action: usize) -> EnvState {
            let deck: [i32; 52] = std::array::from_fn(|i| i as i32);
            let stacks = [100 * core::CHIP_UNIT; NUM_PLAYERS];
            let occ = [true; NUM_PLAYERS];
            let mut s = init_hand_n::<NUM_PLAYERS>(0, &stacks, &occ, 0, &deck, [0u32, 0u32]);
            let mut guard = 0;
            let mut opened = false;
            while !s.done && guard < 60 {
                guard += 1;
                if is_chance_node(&s) {
                    s = sample_chance(&s);
                    continue;
                }
                let seat = s.action_seat;
                if seat < 0 {
                    break;
                }
                if s.street >= core::STREET_FLOP {
                    return s;
                }
                let mask = legal_action_mask(&s, seat);
                let act = if !opened && mask[open_action] {
                    opened = true;
                    open_action
                } else if mask[core::CALL] && s.current_bet > core::BB_CHIPS {
                    core::CALL
                } else if mask[core::FOLD] {
                    core::FOLD
                } else {
                    core::CHECK
                };
                s = apply_action(&s, act);
            }
            s
        }
        let a = reach_flop_with_open(core::BET_BUCKET_START + 4); // ~pot-sized open
        let b = reach_flop_with_open(core::BET_BUCKET_START + 6); // ~2x-sized open
        assert!(a.street >= core::STREET_FLOP && b.street >= core::STREET_FLOP);
        let seg_a = presum_a_segment(&a);
        let seg_b = presum_a_segment(&b);
        assert_eq!(
            seg_a, seg_b,
            "presum should COLLAPSE preflop bet sizes: {:?} vs {:?}",
            seg_a, seg_b
        );
        // Sanity: fullhist DIFFERS (distinct preflop histories) — the collapse is meaningful.
        assert_ne!(
            fullhist_a_segment(&a),
            fullhist_a_segment(&b),
            "test setup invalid: fullhist should differ on distinct preflop sizes"
        );
        // Differing POSTFLOP action must NOT collapse: apply a flop check on one line and a flop bet
        // on the other, then compare the resulting postflop decision nodes' presum segments.
        let seat_a = a.action_seat;
        let mask_a = legal_action_mask(&a, seat_a);
        let bet = (core::BET_BUCKET_START..core::ALL_IN)
            .find(|&x| mask_a[x])
            .expect("a flop bet must be legal");
        let a_check = apply_action(&a, if mask_a[core::CHECK] { core::CHECK } else { bet });
        let a_bet = apply_action(&a, bet);
        fn next_decision(mut s: EnvState) -> Option<EnvState> {
            let mut guard = 0;
            while !s.done && guard < 20 {
                guard += 1;
                if is_chance_node(&s) {
                    s = sample_chance(&s);
                    continue;
                }
                if s.action_seat < 0 {
                    return None;
                }
                return Some(s);
            }
            None
        }
        if let (Some(sc), Some(sb)) = (next_decision(a_check), next_decision(a_bet)) {
            if sc.street >= core::STREET_FLOP && sb.street >= core::STREET_FLOP {
                assert_ne!(
                    presum_a_segment(&sc),
                    presum_a_segment(&sb),
                    "presum must DISTINGUISH differing postflop actions"
                );
            }
        }
    }
}
