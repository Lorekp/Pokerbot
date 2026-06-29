//! Bit-identical Rust port of `pokerbot/engine/np_engine.py` (the NLHE state machine + showdown eval)
//! and the JAX Threefry-2x32 deck-deal RNG.
//!
//! CORRECTNESS-CRITICAL: every transition here must produce results bit-identical to `np_engine.py`,
//! which is itself bit-identical to the JAX engine (`tests/test_np_engine_equiv.py`). The acceptance
//! gate is `tools/battery_native/golden/golden.json` (>=5000 random states + full random hands) and the
//! live cross-check in `pokerbot/search/native_solver.py`.
//!
//! Layout mirrors np_engine: 6 seats, 11 actions (0 fold, 1 check, 2 call, 3..9 pot-fraction bets,
//! 10 all-in), board up to 5 cards, history up to 64.

// ---- Constants (mirror jax_engine / np_engine verbatim) ----
pub const NUM_PLAYERS: usize = 6;
pub const NUM_ACTIONS: usize = 11;
pub const MAX_BOARD: usize = 5;
pub const MAX_HISTORY: usize = 64;
pub const CHIP_UNIT: i32 = 100;
pub const BB_CHIPS: i32 = 100;
pub const SB_CHIPS: i32 = CHIP_UNIT / 2; // 50
pub const STREET_PREFLOP: i32 = 0;

pub const FOLD: usize = 0;
pub const CHECK: usize = 1;
pub const CALL: usize = 2;
pub const BET_BUCKET_START: usize = 3;
pub const ALL_IN: usize = 10;

pub const STREET_FLOP: i32 = 1;
pub const STREET_RIVER: i32 = 3;
pub const STREET_TERMINAL: i32 = 4;

// Seat status sentinels.
pub const SEAT_EMPTY: i8 = 0;
pub const SEAT_FOLDED: i8 = 1;
pub const SEAT_LIVE: i8 = 2;
pub const SEAT_ALLIN: i8 = 3;

pub const N_BUCKETS: usize = 7;
// BET_SIZE_BUCKETS stored as f32 in np_engine (`BET_BUCKETS = np.asarray(..., dtype=np.float32)`).
// The f32 round-trip of each f64 literal is what the sizing arithmetic uses — reproduce exactly.
pub const BET_BUCKETS_F32: [f32; N_BUCKETS] = [
    0.333_333_34_f32, // f32(1/3)
    0.5_f32,
    0.66_f32,
    0.75_f32,
    1.0_f32,
    1.5_f32,
    2.0_f32,
];

// ---------------------------------------------------------------------------
// EnvStateN<N> (flat, Copy) — only the fields the engine transitions need.
//
// Const-generic over seat count `N`. The public `EnvState` alias pins `N = NUM_PLAYERS` (= 6) so
// every existing reference across the crate keeps the exact 6-seat layout and behavior (the alias
// monomorphizes to a struct byte-identical to the pre-refactor `EnvState`). A 7-seat (or any-N)
// instantiation is `EnvStateN<7>`; the state-machine functions below are generic over `N` so the
// same engine drives any seat count. Seat-sized fields are `[_; N]`; fixed-size fields
// (`board`/`deck`/`rng_key`/`hist_*`) are independent of seat count and stay as-is.
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnvStateN<const N: usize> {
    pub holes: [[i32; 2]; N],
    pub board: [i32; MAX_BOARD],
    pub seat_status: [i8; N],
    pub stacks: [i32; N],
    pub committed_street: [i32; N],
    pub committed_total: [i32; N],
    pub street: i32,
    pub pot_dead: i32,
    pub current_bet: i32,
    pub min_raise: i32,
    pub button: i32,
    pub action_seat: i32,
    pub last_aggressor: i32,
    pub acted: [bool; N],
    pub last_raise_full: bool,
    pub deck: [i32; 52],
    pub deck_ptr: i32,
    pub rng_key: [u32; 2],
    pub hist_seat: [i32; MAX_HISTORY],
    pub hist_action: [i32; MAX_HISTORY],
    pub hist_amount: [i32; MAX_HISTORY],
    pub hist_len: i32,
    pub done: bool,
    pub rewards: [i32; N],
    pub pending_deal: bool,
}

/// The default 6-seat state. ALIAS: every existing `EnvState` reference across the crate (lib.rs
/// flat marshaling + PyO3 exports, fingerprint.rs, playout.rs, subgame.rs, tests) resolves to this
/// and is unchanged. Public API surface (`EnvState`, `NUM_PLAYERS`) is preserved verbatim.
pub type EnvState = EnvStateN<NUM_PLAYERS>;

// ---------------------------------------------------------------------------
// Hand evaluation (port of hand_eval portion of np_engine.py)
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
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 5],
    [0, 1, 2, 3, 6],
    [0, 1, 2, 4, 5],
    [0, 1, 2, 4, 6],
    [0, 1, 2, 5, 6],
    [0, 1, 3, 4, 5],
    [0, 1, 3, 4, 6],
    [0, 1, 3, 5, 6],
    [0, 1, 4, 5, 6],
    [0, 2, 3, 4, 5],
    [0, 2, 3, 4, 6],
    [0, 2, 3, 5, 6],
    [0, 2, 4, 5, 6],
    [0, 3, 4, 5, 6],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 6],
    [1, 2, 3, 5, 6],
    [1, 2, 4, 5, 6],
    [1, 3, 4, 5, 6],
    [2, 3, 4, 5, 6],
];

// base-15 positional weights (i64-exact, matches _HIST_WEIGHTS).
const HIST_W: [i64; 5] = [15i64.pow(4), 15i64.pow(3), 15i64.pow(2), 15, 1];
const CAT_SCALE: i32 = 15i32.pow(5); // 759375

#[inline]
pub fn card_rank(card: i32) -> i32 {
    card / 4 + 2
}
#[inline]
pub fn card_suit(card: i32) -> i32 {
    card % 4
}

/// Mirror `_straight_high(ranks)`: ranks is a 5-elem slice (any order). Returns straight high (0 = none).
fn straight_high(ranks: &[i32; 5]) -> i32 {
    let mut best: i32 = 0;
    for w in 0..STRAIGHT_WINDOWS.len() {
        let window = &STRAIGHT_WINDOWS[w];
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

/// Bit-identical port of `score_five(ranks, suits)`.
/// `ranks`/`suits` are 5-card arrays (ranks 2..14, suits 0..3).
pub fn score_five(ranks_in: &[i32; 5], suits: &[i32; 5]) -> i32 {
    // ranks = np.sort(ranks)[::-1]  (descending)
    let mut ranks = *ranks_in;
    ranks.sort_unstable();
    ranks.reverse();

    // counts = bincount(ranks, minlength=15)[:15]
    let mut counts = [0i64; 15];
    for &r in ranks.iter() {
        counts[r as usize] += 1;
    }

    let is_flush = suits.iter().all(|&s| s == suits[0]);
    let s_high = straight_high(&ranks);
    let is_straight = s_high > 0;

    // count_vals = np.sort(counts)[::-1]  (descending)
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

    // within-category positional code:
    //   key = counts[ranks] * 15 + ranks
    //   ordered = ranks[ stable_argsort(key)[::-1] ]
    //   hist = sum(ordered * HIST_W)
    // np.argsort(kind="stable") is ascending-stable; [::-1] reverses. We replicate by building
    // (key, original_index) and sorting ascending-stably, then reversing.
    let mut idx: [usize; 5] = [0, 1, 2, 3, 4];
    let key: [i64; 5] = [
        counts[ranks[0] as usize] * 15 + ranks[0] as i64,
        counts[ranks[1] as usize] * 15 + ranks[1] as i64,
        counts[ranks[2] as usize] * 15 + ranks[2] as i64,
        counts[ranks[3] as usize] * 15 + ranks[3] as i64,
        counts[ranks[4] as usize] * 15 + ranks[4] as i64,
    ];
    // stable ascending sort of indices by key
    idx.sort_by(|&a, &b| key[a].cmp(&key[b])); // sort_by is stable
                                               // reversed -> descending; ties keep reversed-original order, matching np.argsort(stable)[::-1].
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

/// Bit-identical `evaluate_seven(cards)`: best score over the 21 5-card combos. `cards` = 7 card ids.
pub fn evaluate_seven(cards: &[i32; 7]) -> i32 {
    let mut best = i32::MIN;
    for c in 0..COMBOS.len() {
        let combo = &COMBOS[c];
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

/// `player_showdown_score(holes, board, board_valid)`: 7-card buffer = [hole0, hole1, board(masked)].
fn player_showdown_score(holes: &[i32; 2], board: &[i32; MAX_BOARD]) -> i32 {
    let mut seven = [0i32; 7];
    seven[0] = holes[0];
    seven[1] = holes[1];
    for i in 0..MAX_BOARD {
        seven[2 + i] = if board[i] >= 0 { board[i] } else { 0 };
    }
    evaluate_seven(&seven)
}

// ---------------------------------------------------------------------------
// Side pots / terminal returns (port of compute_returns_np)
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn eligible_to_win<const N: usize>(seat_status: &[i8; N]) -> [bool; N] {
    let mut e = [false; N];
    for i in 0..N {
        e[i] = seat_status[i] == SEAT_LIVE || seat_status[i] == SEAT_ALLIN;
    }
    e
}

/// `_cap_contributions`: cap each positive contribution at the second-highest (or highest if <2).
pub(crate) fn cap_contributions<const N: usize>(committed_total: &[i32; N]) -> [i32; N] {
    let mut positive = [0i32; N];
    for i in 0..N {
        positive[i] = if committed_total[i] > 0 {
            committed_total[i]
        } else {
            0
        };
    }
    // sorted_c = np.sort(positive)[::-1]  (descending)
    let mut sorted_c = positive;
    sorted_c.sort_unstable();
    sorted_c.reverse();
    let second = sorted_c[1];
    let n_pos = positive.iter().filter(|&&p| p > 0).count();
    // cap = second if n_pos>=2 else sorted_c[0]; cap = cap if n_pos>=1 else 0
    let mut cap = if n_pos >= 2 { second } else { sorted_c[0] };
    if n_pos < 1 {
        cap = 0;
    }
    let mut out = [0i32; N];
    for i in 0..N {
        out[i] = if positive[i] > 0 {
            positive[i].min(cap)
        } else {
            0
        };
    }
    out
}

/// `_odd_chip_seat(eligible, button)`: first eligible seat clockwise after the button.
fn odd_chip_seat<const N: usize>(eligible: &[bool; N], button: i32) -> i32 {
    let any = eligible.iter().any(|&e| e);
    if !any {
        return button;
    }
    // seats = (button+1+offsets) % N for offsets 0..N; idx = argmax(is_elig).
    // np.argmax returns the FIRST max (first True).
    for off in 0..N {
        let seat = ((button as i64 + 1 + off as i64).rem_euclid(N as i64)) as usize;
        if eligible[seat] {
            return seat as i32;
        }
    }
    button
}

/// `award_layer(amount, eligible, scores, button)`.
pub(crate) fn award_layer<const N: usize>(
    amount: i32,
    eligible: &[bool; N],
    scores: &[i32; N],
    button: i32,
) -> [i32; N] {
    // best = max(where(eligible, scores, -1))
    let mut best = i32::MIN;
    for i in 0..N {
        let v = if eligible[i] { scores[i] } else { -1 };
        if v > best {
            best = v;
        }
    }
    let mut winners = [false; N];
    for i in 0..N {
        winners[i] = eligible[i] && scores[i] == best && best >= 0;
    }
    let n_winners = (winners.iter().filter(|&&w| w).count() as i32).max(1);
    let base = amount / n_winners;
    let rem = amount - base * n_winners;
    let odd_seat = odd_chip_seat(&winners, button);
    let mut winnings = [0i32; N];
    for i in 0..N {
        winnings[i] = if winners[i] { base } else { 0 };
    }
    if rem > 0 && winners[odd_seat as usize] {
        winnings[odd_seat as usize] += rem;
    }
    winnings
}

/// Bit-identical `compute_returns_np`. Returns per-seat net chip delta (sums to zero).
pub fn compute_returns<const N: usize>(
    holes: &[[i32; 2]; N],
    board: &[i32; MAX_BOARD],
    seat_status: &[i8; N],
    committed_total: &[i32; N],
    button: i32,
    n_showdown: i32,
) -> [i32; N] {
    if n_showdown <= 1 {
        // fold_win: winner = first eligible (argmax of live as int)
        let live = eligible_to_win(seat_status);
        let mut winner = 0usize;
        for i in 0..N {
            if live[i] {
                winner = i;
                break;
            }
        }
        let total: i32 = committed_total.iter().sum();
        let mut out = [0i32; N];
        for i in 0..N {
            out[i] = -committed_total[i];
        }
        out[winner] += total;
        return out;
    }

    // showdown
    let eligible_mask = eligible_to_win(seat_status);
    let mut scores = [-1i32; N];
    for s in 0..N {
        if eligible_mask[s] {
            scores[s] = player_showdown_score(&holes[s], board);
        }
    }

    let capped = cap_contributions(committed_total);
    let mut excess = [0i32; N];
    for i in 0..N {
        excess[i] = committed_total[i] - capped[i];
    }
    let mut winnings = [0i32; N];

    // positive = where(capped>0, capped, -1); uniq sorted ascending; levels padded to MAX_POTS(-1);
    // re-sorted; clamp <0 -> -1. MAX_POTS == N (one pot level per seat is the worst case).
    let mut positive = [0i32; N];
    for i in 0..N {
        positive[i] = if capped[i] > 0 { capped[i] } else { -1 };
    }
    // np.unique => sorted ascending unique
    let mut uniq_vec: Vec<i32> = positive.to_vec();
    uniq_vec.sort_unstable();
    uniq_vec.dedup();
    let mut levels = [-1i32; N];
    let take = uniq_vec.len().min(N);
    for i in 0..take {
        levels[i] = uniq_vec[i];
    }
    levels.sort_unstable();
    for i in 0..N {
        if levels[i] < 0 {
            levels[i] = -1;
        }
    }

    let mut prev_level: i32 = 0;
    for li in 0..N {
        let level = levels[li];
        let valid = level >= 0;
        let layer_size = if valid { level - prev_level } else { 0 };
        let mut n_contrib = 0i32;
        let mut contributing = [false; N];
        for i in 0..N {
            contributing[i] = capped[i] >= level && valid;
            if contributing[i] {
                n_contrib += 1;
            }
        }
        let pot_amt = layer_size * n_contrib;
        let mut eligible = [false; N];
        for i in 0..N {
            eligible[i] =
                (seat_status[i] == SEAT_LIVE || seat_status[i] == SEAT_ALLIN) && contributing[i];
        }
        let win = award_layer(pot_amt, &eligible, &scores, button);
        for i in 0..N {
            winnings[i] += win[i];
        }
        if valid {
            prev_level = level;
        }
    }

    let mut out = [0i32; N];
    for i in 0..N {
        out[i] = winnings[i] + excess[i] - committed_total[i];
    }
    out
}

// ---------------------------------------------------------------------------
// Seat helpers
// ---------------------------------------------------------------------------

fn live_count<const N: usize>(seat_status: &[i8; N]) -> i32 {
    let mut c = 0;
    for &v in seat_status.iter() {
        if v == SEAT_LIVE || v == SEAT_ALLIN {
            c += 1;
        }
    }
    c
}

fn pot_for_sizing<const N: usize>(s: &EnvStateN<N>) -> i32 {
    let mut p = s.pot_dead;
    for i in 0..N {
        p += s.committed_street[i];
    }
    p
}

fn first_live_after<const N: usize>(seat_status: &[i8; N], from_seat: i32) -> i32 {
    let base = from_seat;
    for off in 1..=(N as i32) {
        let seat = ((base + off).rem_euclid(N as i32)) as usize;
        if seat_status[seat] == SEAT_LIVE {
            return seat as i32;
        }
    }
    -1
}

fn first_postflop_actor<const N: usize>(s: &EnvStateN<N>) -> i32 {
    // _nth_live_after(seat_status, button, 1) = first_live_after(button)
    first_live_after(&s.seat_status, s.button)
}

// ---------------------------------------------------------------------------
// Threefry-2x32 RNG (deck-deal key advance)
// ---------------------------------------------------------------------------

const ROT_BLOCKS: [[u32; 4]; 5] = [
    [13, 15, 26, 6],
    [17, 29, 16, 24],
    [13, 15, 26, 6],
    [17, 29, 16, 24],
    [13, 15, 26, 6],
];

#[inline]
fn rotl32(x: u32, r: u32) -> u32 {
    (x << r) | (x >> (32 - r))
}

fn threefry2x32(keypair: [u32; 2], count0: u32, count1: u32) -> (u32, u32) {
    let k0 = keypair[0];
    let k1 = keypair[1];
    let k2 = 0x1BD1_1BDA_u32 ^ k0 ^ k1;
    let ks = [k0, k1, k2];
    let mut x0 = count0.wrapping_add(ks[0]);
    let mut x1 = count1.wrapping_add(ks[1]);
    for block in 0..5usize {
        let rots = &ROT_BLOCKS[block];
        for i in 0..4usize {
            x0 = x0.wrapping_add(x1);
            x1 = rotl32(x1, rots[i]);
            x1 ^= x0;
        }
        x0 = x0.wrapping_add(ks[(block + 1) % 3]);
        x1 = x1
            .wrapping_add(ks[(block + 2) % 3])
            .wrapping_add((block + 1) as u32);
    }
    (x0, x1)
}

fn split_consume(key: [u32; 2]) -> [u32; 2] {
    let (o0, o1) = threefry2x32(key, 0, 0);
    [o0, o1]
}

/// Test-only re-export of `split_consume` (the deck-deal key advance).
pub fn _test_split_consume(key: [u32; 2]) -> [u32; 2] {
    split_consume(key)
}

// ---------------------------------------------------------------------------
// Street / chance
// ---------------------------------------------------------------------------

pub fn is_chance_node<const N: usize>(s: &EnvStateN<N>) -> bool {
    s.pending_deal && !s.done
}

fn n_board(board: &[i32; MAX_BOARD]) -> usize {
    board.iter().filter(|&&c| c >= 0).count()
}

/// `_deal_n_cards`: deal n_new board cards from deck, advance ptr, advance rng_key 3 times.
fn deal_n_cards<const N: usize>(s: &mut EnvStateN<N>, n_new: usize) {
    let mut ptr = s.deck_ptr as usize;
    let nb = n_board(&s.board);
    let mut key = s.rng_key;
    for i in 0..3 {
        key = split_consume(key);
        if i < n_new {
            s.board[nb + i] = s.deck[ptr];
            ptr += 1;
        }
    }
    s.deck_ptr = ptr as i32;
    s.rng_key = key;
    s.pending_deal = false;
}

/// `sample_chance_np`: deal next street after betting closed.
pub fn sample_chance<const N: usize>(s_in: &EnvStateN<N>) -> EnvStateN<N> {
    let mut s = *s_in;
    let new_street = s.street + 1;
    let n_new = if new_street == STREET_FLOP {
        3
    } else if new_street <= STREET_RIVER {
        1
    } else {
        0
    };
    s.street = new_street;
    if n_new > 0 {
        deal_n_cards(&mut s, n_new);
    } else {
        s.pending_deal = false;
    }
    let new_street2 = if new_street > STREET_RIVER {
        STREET_TERMINAL
    } else {
        s.street
    };
    let actor = if new_street2 >= STREET_TERMINAL {
        -1
    } else {
        first_postflop_actor(&s)
    };
    s.street = new_street2;
    s.action_seat = actor;
    s.committed_street = [0; N];
    s.current_bet = 0;
    s.min_raise = BB_CHIPS;
    s.acted = [false; N];
    s.last_aggressor = -1;
    s.last_raise_full = true;
    s
}

fn run_out<const N: usize>(s: &mut EnvStateN<N>) {
    let mut ptr = s.deck_ptr as usize;
    for _ in 0..MAX_BOARD {
        let nb = n_board(&s.board);
        if nb < MAX_BOARD {
            s.board[nb] = s.deck[ptr];
            ptr += 1;
        }
    }
    s.deck_ptr = ptr as i32;
    s.street = STREET_TERMINAL;
    s.pending_deal = false;
}

fn close_street<const N: usize>(s: &mut EnvStateN<N>) {
    let mut pot_dead = s.pot_dead;
    for i in 0..N {
        pot_dead += s.committed_street[i];
    }
    s.pot_dead = pot_dead;
    s.committed_street = [0; N];
    s.current_bet = 0;
    s.pending_deal = true;
    s.action_seat = -1;
    let actors = s.seat_status.iter().any(|&st| st == SEAT_LIVE);
    if !actors && s.street < STREET_RIVER {
        run_out(s);
    }
}

fn street_complete<const N: usize>(s: &EnvStateN<N>) -> bool {
    let cur = s.current_bet;
    let mut n_can_act = 0;
    let mut matched = true;
    let mut all_acted = true;
    for i in 0..N {
        if s.seat_status[i] == SEAT_LIVE {
            n_can_act += 1;
            if s.committed_street[i] < cur {
                matched = false;
            }
            if !s.acted[i] {
                all_acted = false;
            }
        }
    }
    let back_to_aggressor = s.action_seat == s.last_aggressor;
    let no_aggressor = s.last_aggressor < 0;
    (n_can_act == 0) || (matched && all_acted && (back_to_aggressor || no_aggressor))
}

// ---------------------------------------------------------------------------
// Betting
// ---------------------------------------------------------------------------

fn commit<const N: usize>(s: &mut EnvStateN<N>, seat: usize, amount_in: i32) {
    let amount = amount_in.min(s.stacks[seat]);
    s.stacks[seat] -= amount;
    s.committed_street[seat] += amount;
    s.committed_total[seat] += amount;
    if s.stacks[seat] <= 0 {
        s.seat_status[seat] = SEAT_ALLIN;
    }
}

/// f32-truncate bet target: `np.int32(np.float32(np.float32(pot) * frac))`.
#[inline]
fn bucket_prod(pot: i32, frac: f32) -> i32 {
    let p = (pot as f32) * frac;
    // np.int32(float32) truncates toward zero.
    p as i32
}

/// Bit-identical `legal_action_mask_np(state, seat)`.
pub fn legal_action_mask<const N: usize>(s: &EnvStateN<N>, seat_in: i32) -> [bool; NUM_ACTIONS] {
    let seat = if seat_in < 0 {
        s.action_seat as usize
    } else {
        seat_in as usize
    };
    let status = s.seat_status[seat];
    let stack = s.stacks[seat];
    let committed = s.committed_street[seat];
    let to_call = (s.current_bet - committed).max(0);
    let pot = pot_for_sizing(s);

    let mut mask = [false; NUM_ACTIONS];
    mask[FOLD] = to_call > 0;
    mask[CHECK] = to_call == 0;
    mask[CALL] = to_call > 0;

    let min_raise_amt = s.min_raise;
    let cur_bet = s.current_bet;
    let is_live = status == SEAT_LIVE;
    let reopen_ok = !s.acted[seat] || s.last_raise_full || to_call == 0;

    for i in 0..N_BUCKETS {
        let frac = BET_BUCKETS_F32[i];
        let prod = bucket_prod(pot, frac);
        let target_total = cur_bet.max(prod);
        let raise_by = target_total - committed;
        let bucket_ok = is_live
            && (stack > to_call)
            && (raise_by >= min_raise_amt)
            && (raise_by <= stack)
            && reopen_ok;
        mask[BET_BUCKET_START + i] = bucket_ok;
    }
    mask[ALL_IN] = is_live && stack > 0 && reopen_ok;
    if !is_live {
        for m in mask.iter_mut() {
            *m = false;
        }
    }
    if !mask.iter().any(|&m| m) {
        mask[CHECK] = true;
    }
    mask
}

fn action_amount<const N: usize>(s: &EnvStateN<N>, seat: usize, action: usize) -> i32 {
    let committed = s.committed_street[seat];
    let to_call = (s.current_bet - committed).max(0);
    let stack = s.stacks[seat];
    let pot = pot_for_sizing(s);

    if action == CALL {
        return to_call;
    }
    if action == ALL_IN {
        return stack;
    }
    if BET_BUCKET_START <= action && action < ALL_IN {
        let frac = BET_BUCKETS_F32[action - BET_BUCKET_START];
        let prod = bucket_prod(pot, frac);
        let target = s.current_bet.max(prod);
        let bucket_val = (target - committed).max(0);
        return to_call + bucket_val;
    }
    0
}

/// Bit-identical `apply_action_np(state, action)`.
pub fn apply_action<const N: usize>(s_in: &EnvStateN<N>, action: usize) -> EnvStateN<N> {
    apply_action_with(s_in, action, compute_returns::<N>)
}

/// `apply_action` generic over the terminal showdown SETTLE scorer, so the trainer can inject the
/// perfect-hash `compute_returns_phe` at the IN-APPLY settle (SITE 1 — the multiway all-in run-out
/// terminals reached inside `apply_action`) while serve / LBR / np_engine-parity keep the scalar
/// `core::compute_returns`. `apply_action` above is exactly this with `settle = compute_returns`, so it
/// stays byte-identical to the pre-refactor concrete body (the optimizer inlines the known fn; chip
/// values are unchanged). `settle` MUST be chip-identical to `compute_returns` (gates:
/// `apply_action_with_phe_matches_scalar_byte_for_byte` + `showdown_returns_matches_compute_returns`);
/// the only effect is the per-terminal eval speed.
pub fn apply_action_with<const N: usize, F>(s_in: &EnvStateN<N>, action: usize, settle: F) -> EnvStateN<N>
where
    F: Fn(&[[i32; 2]; N], &[i32; MAX_BOARD], &[i8; N], &[i32; N], i32, i32) -> [i32; N],
{
    let mut s = *s_in;
    let seat = s.action_seat as usize;
    let amount = action_amount(&s, seat, action);

    if action == FOLD {
        s.seat_status[seat] = SEAT_FOLDED;
    }

    let is_call = action == CALL;
    let is_aggressive = action >= BET_BUCKET_START || action == ALL_IN;
    if is_call || is_aggressive {
        commit(&mut s, seat, amount);
    }

    let prev_bet = s_in.current_bet; // note: np reads state.current_bet AFTER commit, but commit
                                     // never changes current_bet, so prev_bet == pre-action current_bet.
    let new_street_bet = s.committed_street[seat];
    let raised = is_aggressive && new_street_bet > prev_bet;
    let full_raise = raised && new_street_bet >= prev_bet + s.min_raise;

    s.acted[seat] = true;
    if full_raise {
        s.acted = [false; N];
        s.acted[seat] = true;
    }

    if raised {
        let raise_size = new_street_bet - prev_bet;
        s.current_bet = new_street_bet;
        s.min_raise = BB_CHIPS.max(raise_size);
        s.last_aggressor = seat as i32;
        s.last_raise_full = full_raise;
    }
    // else: current_bet / min_raise / last_aggressor / last_raise_full unchanged.

    let idx = s.hist_len as usize;
    if idx < MAX_HISTORY {
        s.hist_seat[idx] = seat as i32;
        s.hist_action[idx] = action as i32;
        s.hist_amount[idx] = amount;
    }
    s.hist_len = ((idx + 1).min(MAX_HISTORY)) as i32;

    let n_live = live_count(&s.seat_status);
    let mut terminal = n_live <= 1;

    if terminal {
        s.street = STREET_TERMINAL;
        s.pending_deal = false;
    } else if street_complete(&s) {
        close_street(&mut s);
    }

    terminal = terminal || s.street >= STREET_TERMINAL || n_live <= 1;

    let action_seat = if terminal || s.pending_deal {
        -1
    } else {
        first_live_after(&s.seat_status, s.action_seat)
    };
    s.action_seat = action_seat;
    s.done = terminal;

    if terminal {
        s.rewards = settle(
            &s.holes,
            &s.board,
            &s.seat_status,
            &s.committed_total,
            s.button,
            n_live,
        );
    } else {
        s.rewards = [0; N];
    }
    s
}

// ---------------------------------------------------------------------------
// Native root construction (port of jax_engine `_post_blinds` + `init_hand`, with the B2/B3 fixes).
//
// GENERIC over seat count `N` (2..=7), the SINGLE source for any-seat root construction. The default
// 6-seat path (`init_hand`/`post_blinds`, used by the PyO3 binding + the np_engine parity gate) is the
// `N = NUM_PLAYERS` monomorphization of `init_hand_n`/`post_blinds_n` and is BYTE-IDENTICAL to the
// pre-generic concrete code (same loop bounds, same ops, same field order). 7-seat is the additive
// `N = 7` instantiation — NO second concrete copy, NO change to the N=6 default. `ante=0` keeps
// `committed_total == committed_street` and `pot_dead == 0` (byte-identical to the no-ante engine).
// ---------------------------------------------------------------------------

/// Mirror `_nth_live_after(seat_status, from_seat, n)`: the n-th live seat clockwise after
/// `from_seat` (n >= 1). Bit-identical to the Python `fori_loop` of `_first_live_after`.
/// Generic over seat count `N` (`first_live_after` already is).
fn nth_live_after<const N: usize>(seat_status: &[i8; N], from_seat: i32, n: i32) -> i32 {
    let mut seat = from_seat;
    for _ in 0..n {
        seat = first_live_after(seat_status, seat);
    }
    seat
}

/// Port of `_post_blinds(stacks, seat_status, button, ante)` with the B2/B3 fixes, generic over the
/// seat count `N`. Mutates `stacks`/`seat_status` in place and returns
/// `(committed_street, committed_total, pot_dead, current_bet, min_raise)`.
///
/// HU (`n_live == 2`) blind posting: jax `_post_blinds` does NOT special-case HU — it ALWAYS posts
/// SB = `nth_live_after(button, 1)` and BB = `nth_live_after(button, 2)` for every N. We mirror jax
/// verbatim (bit-identity to jax is the binding constraint). With exactly 2 live seats the clockwise
/// walk makes the 1st-live-after-button the non-button seat (SB) and the 2nd-live-after wrap back to
/// the button (BB); preflop, the button then acts first (`action_seat = btn_first = SB` via the
/// `n_live <= 2` branch in `init_hand_n`), matching jax. No extra branch is needed for any N.
fn post_blinds_n<const N: usize>(
    stacks: &mut [i32; N],
    seat_status: &mut [i8; N],
    button: i32,
    ante: i32,
) -> ([i32; N], [i32; N], i32, i32, i32) {
    let sb = nth_live_after(seat_status, button, 1) as usize;
    let bb = nth_live_after(seat_status, button, 2) as usize;
    let mut committed = [0i32; N];
    let sb_live = (seat_status[sb] != SEAT_EMPTY) as i32;
    let bb_live = (seat_status[bb] != SEAT_EMPTY) as i32;
    // B2: clamp each blind to the poster's stack (no negative stack).
    let sb_paid = (SB_CHIPS * sb_live).min(stacks[sb].max(0));
    let bb_paid = (BB_CHIPS * bb_live).min(stacks[bb].max(0));
    stacks[sb] -= sb_paid;
    stacks[bb] -= bb_paid;
    committed[sb] = sb_paid;
    committed[bb] = bb_paid;
    // Per-player ante: dead money clamped to the post-blind stack, added to pot_dead + committed_total
    // (NOT committed_street). ante=0 -> committed_total == committed, pot_dead == 0 (byte-identical).
    let mut ante_each = [0i32; N];
    let mut committed_total = [0i32; N];
    let mut pot_dead = 0i32;
    for i in 0..N {
        let live_mask = (seat_status[i] != SEAT_EMPTY) as i32;
        ante_each[i] = ante.min(stacks[i].max(0)) * live_mask;
        stacks[i] -= ante_each[i];
        pot_dead += ante_each[i];
        committed_total[i] = committed[i] + ante_each[i];
    }
    // B2: a live seat that committed its entire stack (blind and/or ante) is ALL-IN.
    for i in 0..N {
        let live_mask = seat_status[i] != SEAT_EMPTY;
        if live_mask && stacks[i] == 0 {
            seat_status[i] = SEAT_ALLIN;
        }
    }
    // B3: standing bet is the largest street commitment, min raise stays a full BB.
    let mut current_bet = 0i32;
    for i in 0..N {
        if committed[i] > current_bet {
            current_bet = committed[i];
        }
    }
    let min_raise = BB_CHIPS;
    (committed, committed_total, pot_dead, current_bet, min_raise)
}

/// Native root construction generic over seat count `N` (2..=7), byte-identical to
/// `jax_engine.init_hand(rng_key, button, stacks, occupancy, ante, n_players=N)` WHEN GIVEN jax's
/// permuted deck (the Threefry permutation is not replicated; the caller supplies the already-permuted
/// `deck`, which is how the gate cross-checks byte-identity). Deck-as-input + per-player ante +
/// occupancy (`SEAT_EMPTY` for absent seats) + button rotation + HU button=SB are all handled by the
/// generic blind/deal walk (see `post_blinds_n`). The default 6-seat `init_hand` is the
/// `N = NUM_PLAYERS` instantiation.
pub fn init_hand_n<const N: usize>(
    button: i32,
    stacks_in: &[i32; N],
    occupancy: &[bool; N],
    ante: i32,
    deck: &[i32; 52],
    rng_key: [u32; 2],
) -> EnvStateN<N> {
    let mut seat_status = [SEAT_EMPTY; N];
    for i in 0..N {
        seat_status[i] = if occupancy[i] { SEAT_LIVE } else { SEAT_EMPTY };
    }
    let mut stacks = *stacks_in;

    let (committed, committed_total, pot_dead, current_bet, min_raise) =
        post_blinds_n(&mut stacks, &mut seat_status, button, ante);

    // Deal holes from the supplied deck exactly as the jax fori_loop does.
    let mut holes = [[-1i32; 2]; N];
    let mut p: usize = 0;
    for i in 0..N {
        let live = seat_status[i] != SEAT_EMPTY;
        if live {
            holes[i][0] = deck[p];
            holes[i][1] = deck[p + 1];
            p += 2;
        }
    }
    let deck_ptr = p as i32;

    let mut n_live = 0i32;
    for i in 0..N {
        if seat_status[i] == SEAT_LIVE {
            n_live += 1;
        }
    }
    let btn_first = nth_live_after(&seat_status, button, 1);
    let utg = nth_live_after(&seat_status, button, 3);
    let action_seat = if n_live <= 2 { btn_first } else { utg };

    EnvStateN {
        holes,
        board: [-1i32; MAX_BOARD],
        seat_status,
        stacks,
        committed_street: committed,
        committed_total,
        street: STREET_PREFLOP,
        pot_dead,
        current_bet,
        min_raise,
        button,
        action_seat,
        last_aggressor: -1,
        acted: [false; N],
        last_raise_full: true,
        deck: *deck,
        deck_ptr,
        rng_key,
        hist_seat: [-1i32; MAX_HISTORY],
        hist_action: [-1i32; MAX_HISTORY],
        hist_amount: [0i32; MAX_HISTORY],
        hist_len: 0,
        done: false,
        rewards: [0i32; N],
        pending_deal: false,
    }
}

/// Default 6-seat native root construction. THIN alias over the generic `init_hand_n::<NUM_PLAYERS>`
/// — the N=6 monomorphization is byte-identical to the pre-generic concrete code (same loop bounds,
/// same field order). Public API surface (`init_hand`, PyO3 binding) is preserved verbatim.
pub fn init_hand(
    button: i32,
    stacks_in: &[i32; NUM_PLAYERS],
    occupancy: &[bool; NUM_PLAYERS],
    ante: i32,
    deck: &[i32; 52],
    rng_key: [u32; 2],
) -> EnvState {
    init_hand_n::<NUM_PLAYERS>(button, stacks_in, occupancy, ante, deck, rng_key)
}
