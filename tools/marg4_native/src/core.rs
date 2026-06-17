//! Bit-identical native kernel for the 4-max AoF marginal payoff precompute.
//!
//! Reproduces `aof_cfr/fp_4max_vec.py::precompute_payoffs` exactly:
//!   precompute_payoffs -> _terminal_returns (per outcome, per deal)
//!     -> _showdown_shares(alive_holes)  [400-board MC, native eval7, MT19937 Fisher-Yates]
//!
//! Two correctness pillars make the output BIT-IDENTICAL to the production Python path:
//!   1. RNG: a verbatim port of CPython's MT19937 (`init_by_array` seeding +
//!      `genrand_uint32`) and `random.Random._randbelow_with_getrandbits`, driving the same
//!      Fisher-Yates board sequence per showdown (seed = the same 131-rolling hash of the flat
//!      alive hole-card ids, masked to 32 bits).
//!   2. Evaluator: `evaluate_seven` is the same higher-is-better 7-card evaluator as the OMPEval
//!      `hu_table_native.eval7` used in production — verified to make identical winner/tie
//!      decisions over millions of random boards (the absolute rank value is irrelevant; only the
//!      per-board argmax + equality classes drive the pot-share arithmetic, and those match).
//!
//! The terminal chip model mirrors `exploit_mc._terminal_returns` + `pushfold_payoffs`:
//! alive (all-in) seats get `share*pot - stack`, folders get `-blind`, an uncontested survivor
//! takes the folded blinds. `pot = contested_pot_bb(k_alive, folded_blinds, stack, gm, fee)`.

// ---------------------------------------------------------------------------
// CPython MT19937 (verbatim) + random.Random._randbelow
// ---------------------------------------------------------------------------

const MT_N: usize = 624;
const MT_M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

pub struct MT19937 {
    mt: [u32; MT_N],
    mti: usize,
}

impl MT19937 {
    /// `random.Random(seed)` for a non-negative int seed < 2^32: CPython converts the seed to a
    /// one-element key array and calls `init_by_array`.
    pub fn new_from_u32(seed: u32) -> MT19937 {
        let mut s = MT19937 { mt: [0u32; MT_N], mti: MT_N + 1 };
        // CPython: seed 0 -> key = [0] (one zero element), any other -> the 32-bit limbs.
        let key: [u32; 1] = [seed];
        s.init_by_array(&key);
        s
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..MT_N {
            self.mt[i] = (1_812_433_253u32)
                .wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        self.mti = MT_N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let key_length = key.len();
        let mut i: usize = 1;
        let mut j: usize = 0;
        let mut k: usize = if MT_N > key_length { MT_N } else { key_length };
        while k > 0 {
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1_664_525u32)))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= MT_N {
                self.mt[0] = self.mt[MT_N - 1];
                i = 1;
            }
            if j >= key_length {
                j = 0;
            }
            k -= 1;
        }
        k = MT_N - 1;
        while k > 0 {
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1_566_083_941u32)))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= MT_N {
                self.mt[0] = self.mt[MT_N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x8000_0000;
    }

    #[inline]
    pub fn genrand_uint32(&mut self) -> u32 {
        if self.mti >= MT_N {
            let mag01: [u32; 2] = [0, MATRIX_A];
            let mut kk = 0;
            while kk < MT_N - MT_M {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + MT_M] ^ (y >> 1) ^ mag01[(y & 1) as usize];
                kk += 1;
            }
            while kk < MT_N - 1 {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                // C index: mt[kk + (M - N)] with kk >= N-M, so kk + M - N >= 0 (no underflow).
                let idx = kk + MT_M - MT_N;
                self.mt[kk] = self.mt[idx] ^ (y >> 1) ^ mag01[(y & 1) as usize];
                kk += 1;
            }
            let y = (self.mt[MT_N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[MT_N - 1] = self.mt[MT_M - 1] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `getrandbits(k)` for 1 <= k <= 32: top k bits of one genrand draw.
    #[inline]
    fn getrandbits(&mut self, k: u32) -> u32 {
        // CPython: r = genrand_uint32() >> (32 - k)
        self.genrand_uint32() >> (32 - k)
    }

}

// n.bit_length() for n>=1 == 32 - leading_zeros(n).
#[inline]
fn bit_length(n: u32) -> u32 {
    32 - n.leading_zeros()
}

impl MT19937 {
    /// `randrange(n)` == `_randbelow(n)` with the exact CPython bit_length(n).
    #[inline]
    pub fn randrange(&mut self, n: u32) -> u32 {
        debug_assert!(n >= 1);
        let k = bit_length(n);
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 7-card evaluator (higher-is-better; identical winner/tie decisions to OMPEval eval7)
// Ported from tools/battery_native/src/core.rs (score_five / evaluate_seven).
// ---------------------------------------------------------------------------

const STRAIGHT_WINDOWS: [[i32; 5]; 10] = [
    [14, 13, 12, 11, 10],
    [13, 12, 11, 10, 9],
    [12, 11, 10, 9, 8],
    [11, 10, 9, 8, 7],
    [10, 9, 8, 7, 6],
    [9, 8, 7, 6, 5],
    [8, 7, 6, 5, 4],
    [7, 6, 5, 4, 3],
    [6, 5, 4, 3, 2],
    [14, 5, 4, 3, 2],
];

const COMBOS: [[usize; 5]; 21] = [
    [0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2, 3, 6], [0, 1, 2, 4, 5], [0, 1, 2, 4, 6],
    [0, 1, 2, 5, 6], [0, 1, 3, 4, 5], [0, 1, 3, 4, 6], [0, 1, 3, 5, 6], [0, 1, 4, 5, 6],
    [0, 2, 3, 4, 5], [0, 2, 3, 4, 6], [0, 2, 3, 5, 6], [0, 2, 4, 5, 6], [0, 3, 4, 5, 6],
    [1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [1, 2, 3, 5, 6], [1, 2, 4, 5, 6], [1, 3, 4, 5, 6],
    [2, 3, 4, 5, 6],
];

const HIST_W: [i64; 5] = [15i64.pow(4), 15i64.pow(3), 15i64.pow(2), 15, 1];
const CAT_SCALE: i32 = 15i32.pow(5);

#[inline]
fn card_rank(card: i32) -> i32 {
    card / 4 + 2
}
#[inline]
fn card_suit(card: i32) -> i32 {
    card % 4
}

fn straight_high(ranks: &[i32; 5]) -> i32 {
    let mut best: i32 = 0;
    for window in STRAIGHT_WINDOWS.iter() {
        let mut matched = true;
        for i in 0..5 {
            let target = window[i];
            if !ranks.iter().any(|&r| r == target) {
                matched = false;
                break;
            }
        }
        let high = if window[0] == 14 {
            if window[1] == 5 {
                5
            } else {
                14
            }
        } else {
            window[0]
        };
        let high = if matched { high } else { 0 };
        if high > best {
            best = high;
        }
    }
    best
}

fn score_five(ranks_in: &[i32; 5], suits: &[i32; 5]) -> i32 {
    let mut ranks = *ranks_in;
    ranks.sort_unstable();
    ranks.reverse();

    let mut counts = [0i64; 15];
    for &r in ranks.iter() {
        counts[r as usize] += 1;
    }

    let is_flush = suits.iter().all(|&s| s == suits[0]);
    let s_high = straight_high(&ranks);
    let is_straight = s_high > 0;

    let mut count_vals = counts;
    count_vals.sort_unstable();
    count_vals.reverse();

    let mut cat = 0i32;
    if is_flush && is_straight {
        cat = 8;
    }
    if count_vals[0] == 4 && cat < 8 {
        cat = 7;
    }
    if count_vals[0] == 3 && count_vals[1] >= 2 && cat < 7 {
        cat = 6;
    }
    if is_flush && !is_straight && cat < 6 {
        cat = 5;
    }
    if is_straight && !is_flush && cat < 5 {
        cat = 4;
    }
    if count_vals[0] == 3 && count_vals[1] < 2 && cat < 4 {
        cat = 3;
    }
    if count_vals[0] == 2 && count_vals[1] == 2 && cat < 3 {
        cat = 2;
    }
    if count_vals[0] == 2 && count_vals[1] < 2 && cat < 2 {
        cat = 1;
    }

    let mut idx: [usize; 5] = [0, 1, 2, 3, 4];
    let key: [i64; 5] = [
        counts[ranks[0] as usize] * 15 + ranks[0] as i64,
        counts[ranks[1] as usize] * 15 + ranks[1] as i64,
        counts[ranks[2] as usize] * 15 + ranks[2] as i64,
        counts[ranks[3] as usize] * 15 + ranks[3] as i64,
        counts[ranks[4] as usize] * 15 + ranks[4] as i64,
    ];
    idx.sort_by(|&a, &b| key[a].cmp(&key[b]));
    let mut hist: i64 = 0;
    for (w, &ii) in idx.iter().rev().enumerate() {
        hist += ranks[ii] as i64 * HIST_W[w];
    }

    let within = if cat == 4 || cat == 8 {
        s_high as i64
    } else {
        hist
    };
    cat * CAT_SCALE + within as i32
}

#[inline]
pub fn evaluate_seven(cards: &[i32; 7]) -> i32 {
    let mut best = i32::MIN;
    for combo in COMBOS.iter() {
        let mut ranks = [0i32; 5];
        let mut suits = [0i32; 5];
        for i in 0..5 {
            let card = cards[combo[i]];
            ranks[i] = card_rank(card);
            suits[i] = card_suit(card);
        }
        let sc = score_five(&ranks, &suits);
        if sc > best {
            best = sc;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Showdown shares (exact reproduction of exploit_mc._showdown_shares)
// ---------------------------------------------------------------------------

pub const N_BOARDS: u32 = 400;

/// 131-rolling hash of the flat alive hole ids, masked to 32 bits (== exploit_mc._showdown_shares).
#[inline]
fn share_seed(flat: &[i32]) -> u32 {
    let mut seed: u32 = 0;
    for &c in flat {
        seed = seed.wrapping_mul(131).wrapping_add((c as u32).wrapping_add(1));
    }
    seed
}

/// Build the live deck for a showdown: ascending 0..51 minus the subset's hole cards (matches
/// build_live_deck, which removes ONLY the alive seats' cards — folded seats' cards return to the
/// deck, exactly as the production all-in equity convention does).
#[inline]
fn live_deck(flat: &[i32], out: &mut [i32; 52]) -> usize {
    let mut dead = [false; 52];
    for &c in flat {
        dead[c as usize] = true;
    }
    let mut nd = 0;
    for c in 0..52i32 {
        if !dead[c as usize] {
            out[nd] = c;
            nd += 1;
        }
    }
    nd
}

/// Pot shares for an n-way (n in 2..=4) showdown, bit-identical to `_showdown_shares`.
/// `flat` is the alive seats' hole ids in seat order (2*n entries). Result written to `shares[..n]`.
pub fn showdown_shares(flat: &[i32], n: usize, shares: &mut [f64; 4]) {
    for s in shares.iter_mut() {
        *s = 0.0;
    }
    let mut deck = [0i32; 52];
    let nd = live_deck(flat, &mut deck);
    if nd < 5 {
        let v = 1.0 / n as f64;
        for i in 0..n {
            shares[i] = v;
        }
        return;
    }
    let seed = share_seed(flat);
    let mut rng = MT19937::new_from_u32(seed);
    let mut wins = [0.0f64; 4];
    let mut work = [0i32; 52];
    let mut ranks = [0i32; 4];
    for _ in 0..N_BOARDS {
        // work = list(deck) each board (fresh copy; swaps never accumulate).
        work[..nd].copy_from_slice(&deck[..nd]);
        for bi in 0..5usize {
            let j = bi + rng.randrange((nd - bi) as u32) as usize;
            work.swap(bi, j);
        }
        let b0 = work[0];
        let b1 = work[1];
        let b2 = work[2];
        let b3 = work[3];
        let b4 = work[4];
        let mut best = i32::MIN;
        let mut nwin = 0i32;
        for pi in 0..n {
            let r = evaluate_seven(&[flat[2 * pi], flat[2 * pi + 1], b0, b1, b2, b3, b4]);
            ranks[pi] = r;
            if r > best {
                best = r;
                nwin = 1;
            } else if r == best {
                nwin += 1;
            }
        }
        let share = 1.0 / nwin as f64;
        for pi in 0..n {
            if ranks[pi] == best {
                wins[pi] += share;
            }
        }
    }
    let inv = 1.0 / N_BOARDS as f64;
    for i in 0..n {
        shares[i] = wins[i] * inv;
    }
}

// ---------------------------------------------------------------------------
// Terminal payoffs + the (15, 4) per-deal payoff tensor
// ---------------------------------------------------------------------------

// The 15 reachable outcomes as the `folded` seat-set bitmask (bit i set => seat i folded),
// in the SAME ORDER as fp_4max_vec._OUTCOMES (FFF first, then the AABB enumeration). The build
// here mirrors _outcomes() so payoff[oi] aligns with the Python tensor's row oi.
fn outcomes() -> Vec<u8> {
    // mirror: for a0 in FA: for a1 in FA: for a2 in FA: if FFF -> {0,1,2}; else for a3 in FA -> folded set
    let mut outs: Vec<u8> = Vec::with_capacity(15);
    let fa = [false, true]; // false='F', true='A'
    for &a0 in fa.iter() {
        for &a1 in fa.iter() {
            for &a2 in fa.iter() {
                if !a0 && !a1 && !a2 {
                    // FFF -> BB wins uncontested; folded {0,1,2}
                    outs.push(0b0111);
                    continue;
                }
                for &a3 in fa.iter() {
                    let acts = [a0, a1, a2, a3];
                    let mut mask = 0u8;
                    for (i, &a) in acts.iter().enumerate() {
                        if !a {
                            mask |= 1 << i;
                        }
                    }
                    outs.push(mask);
                }
            }
        }
    }
    outs
}

#[inline]
fn seat_blind(seat: usize, _stack_bb: f64) -> f64 {
    // seats = active_seats(4) = (UTG, BTN, SB, BB) -> blinds on SB(idx2)=0.5, BB(idx3)=1.0.
    // Matches exploit_mc._seat_blind via active_seats ordering.
    match seat {
        2 => 0.5,  // SB_POST
        3 => 1.0,  // BB_POST
        _ => 0.0,
    }
}

#[inline]
fn contested_pot(k_alive: usize, folded_blinds: f64, stack_bb: f64, gm: i32, fee: f64) -> f64 {
    if gm >= 2 {
        k_alive as f64 * stack_bb + folded_blinds - fee
    } else {
        // v1 legacy: + BLINDS_DEAD (1.5)
        k_alive as f64 * stack_bb + 1.5 - fee
    }
}

/// Per-deal payoff rows for all 15 outcomes. `holes` = 8 card ids [u0a,u0b,b0a,b0b,s0a,s0b,bb0a,bb0b]
/// (seat order UTG,BTN,SB,BB). Writes `out[oi*4 + seat]` for oi in 0..15.
pub fn deal_payoffs(holes: &[i32; 8], stack_bb: f64, gm: i32, fee: f64, outs: &[u8], out: &mut [f64]) {
    let blinds = [
        seat_blind(0, stack_bb),
        seat_blind(1, stack_bb),
        seat_blind(2, stack_bb),
        seat_blind(3, stack_bb),
    ];
    for (oi, &mask) in outs.iter().enumerate() {
        let row = &mut out[oi * 4..oi * 4 + 4];
        // init: every seat = -blind (folder default)
        for s in 0..4 {
            row[s] = -blinds[s];
        }
        // alive = seats with bit clear
        let mut alive = [0usize; 4];
        let mut na = 0;
        let mut folded_blinds = 0.0;
        for s in 0..4 {
            if mask & (1 << s) == 0 {
                alive[na] = s;
                na += 1;
            } else {
                folded_blinds += blinds[s];
            }
        }
        if na == 1 {
            row[alive[0]] = folded_blinds;
            continue;
        }
        let pot = contested_pot(na, folded_blinds, stack_bb, gm, fee);
        // flat alive holes in seat order
        let mut flat = [0i32; 8];
        for k in 0..na {
            flat[2 * k] = holes[2 * alive[k]];
            flat[2 * k + 1] = holes[2 * alive[k] + 1];
        }
        let mut shares = [0.0f64; 4];
        showdown_shares(&flat[..2 * na], na, &mut shares);
        for k in 0..na {
            row[alive[k]] = shares[k] * pot - stack_bb;
        }
    }
}

/// The 15 outcome masks (cached order matches fp_4max_vec._OUTCOMES).
pub fn outcome_masks() -> Vec<u8> {
    outcomes()
}
