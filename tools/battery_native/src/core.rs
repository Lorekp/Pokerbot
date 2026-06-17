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
pub const BB_CHIPS: i32 = 100;

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
// EnvState (flat, Copy) — only the fields the engine transitions need.
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnvState {
    pub holes: [[i32; 2]; NUM_PLAYERS],
    pub board: [i32; MAX_BOARD],
    pub seat_status: [i8; NUM_PLAYERS],
    pub stacks: [i32; NUM_PLAYERS],
    pub committed_street: [i32; NUM_PLAYERS],
    pub committed_total: [i32; NUM_PLAYERS],
    pub street: i32,
    pub pot_dead: i32,
    pub current_bet: i32,
    pub min_raise: i32,
    pub button: i32,
    pub action_seat: i32,
    pub last_aggressor: i32,
    pub acted: [bool; NUM_PLAYERS],
    pub last_raise_full: bool,
    pub deck: [i32; 52],
    pub deck_ptr: i32,
    pub rng_key: [u32; 2],
    pub hist_seat: [i32; MAX_HISTORY],
    pub hist_action: [i32; MAX_HISTORY],
    pub hist_amount: [i32; MAX_HISTORY],
    pub hist_len: i32,
    pub done: bool,
    pub rewards: [i32; NUM_PLAYERS],
    pub pending_deal: bool,
}

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
    [0, 1, 2, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2, 3, 6], [0, 1, 2, 4, 5], [0, 1, 2, 4, 6],
    [0, 1, 2, 5, 6], [0, 1, 3, 4, 5], [0, 1, 3, 4, 6], [0, 1, 3, 5, 6], [0, 1, 4, 5, 6],
    [0, 2, 3, 4, 5], [0, 2, 3, 4, 6], [0, 2, 3, 5, 6], [0, 2, 4, 5, 6], [0, 3, 4, 5, 6],
    [1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [1, 2, 3, 5, 6], [1, 2, 4, 5, 6], [1, 3, 4, 5, 6],
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
fn eligible_to_win(seat_status: &[i8; NUM_PLAYERS]) -> [bool; NUM_PLAYERS] {
    let mut e = [false; NUM_PLAYERS];
    for i in 0..NUM_PLAYERS {
        e[i] = seat_status[i] == SEAT_LIVE || seat_status[i] == SEAT_ALLIN;
    }
    e
}

/// `_cap_contributions`: cap each positive contribution at the second-highest (or highest if <2).
fn cap_contributions(committed_total: &[i32; NUM_PLAYERS]) -> [i32; NUM_PLAYERS] {
    let mut positive = [0i32; NUM_PLAYERS];
    for i in 0..NUM_PLAYERS {
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
    let mut out = [0i32; NUM_PLAYERS];
    for i in 0..NUM_PLAYERS {
        out[i] = if positive[i] > 0 {
            positive[i].min(cap)
        } else {
            0
        };
    }
    out
}

/// `_odd_chip_seat(eligible, button)`: first eligible seat clockwise after the button.
fn odd_chip_seat(eligible: &[bool; NUM_PLAYERS], button: i32) -> i32 {
    let any = eligible.iter().any(|&e| e);
    if !any {
        return button;
    }
    // seats = (button+1+offsets) % NUM_PLAYERS for offsets 0..NUM_PLAYERS; idx = argmax(is_elig).
    // np.argmax returns the FIRST max (first True).
    for off in 0..NUM_PLAYERS {
        let seat = ((button as i64 + 1 + off as i64).rem_euclid(NUM_PLAYERS as i64)) as usize;
        if eligible[seat] {
            return seat as i32;
        }
    }
    button
}

/// `award_layer(amount, eligible, scores, button)`.
fn award_layer(
    amount: i32,
    eligible: &[bool; NUM_PLAYERS],
    scores: &[i32; NUM_PLAYERS],
    button: i32,
) -> [i32; NUM_PLAYERS] {
    // best = max(where(eligible, scores, -1))
    let mut best = i32::MIN;
    for i in 0..NUM_PLAYERS {
        let v = if eligible[i] { scores[i] } else { -1 };
        if v > best {
            best = v;
        }
    }
    let mut winners = [false; NUM_PLAYERS];
    for i in 0..NUM_PLAYERS {
        winners[i] = eligible[i] && scores[i] == best && best >= 0;
    }
    let n_winners = (winners.iter().filter(|&&w| w).count() as i32).max(1);
    let base = amount / n_winners;
    let rem = amount - base * n_winners;
    let odd_seat = odd_chip_seat(&winners, button);
    let mut winnings = [0i32; NUM_PLAYERS];
    for i in 0..NUM_PLAYERS {
        winnings[i] = if winners[i] { base } else { 0 };
    }
    if rem > 0 && winners[odd_seat as usize] {
        winnings[odd_seat as usize] += rem;
    }
    winnings
}

/// Bit-identical `compute_returns_np`. Returns per-seat net chip delta (sums to zero).
pub fn compute_returns(
    holes: &[[i32; 2]; NUM_PLAYERS],
    board: &[i32; MAX_BOARD],
    seat_status: &[i8; NUM_PLAYERS],
    committed_total: &[i32; NUM_PLAYERS],
    button: i32,
    n_showdown: i32,
) -> [i32; NUM_PLAYERS] {
    if n_showdown <= 1 {
        // fold_win: winner = first eligible (argmax of live as int)
        let live = eligible_to_win(seat_status);
        let mut winner = 0usize;
        for i in 0..NUM_PLAYERS {
            if live[i] {
                winner = i;
                break;
            }
        }
        let total: i32 = committed_total.iter().sum();
        let mut out = [0i32; NUM_PLAYERS];
        for i in 0..NUM_PLAYERS {
            out[i] = -committed_total[i];
        }
        out[winner] += total;
        return out;
    }

    // showdown
    let eligible_mask = eligible_to_win(seat_status);
    let mut scores = [-1i32; NUM_PLAYERS];
    for s in 0..NUM_PLAYERS {
        if eligible_mask[s] {
            scores[s] = player_showdown_score(&holes[s], board);
        }
    }

    let capped = cap_contributions(committed_total);
    let mut excess = [0i32; NUM_PLAYERS];
    for i in 0..NUM_PLAYERS {
        excess[i] = committed_total[i] - capped[i];
    }
    let mut winnings = [0i32; NUM_PLAYERS];

    // positive = where(capped>0, capped, -1); uniq sorted ascending; levels padded to MAX_POTS(-1);
    // re-sorted; clamp <0 -> -1.
    let mut positive = [0i32; NUM_PLAYERS];
    for i in 0..NUM_PLAYERS {
        positive[i] = if capped[i] > 0 { capped[i] } else { -1 };
    }
    // np.unique => sorted ascending unique
    let mut uniq_vec: Vec<i32> = positive.to_vec();
    uniq_vec.sort_unstable();
    uniq_vec.dedup();
    const MAX_POTS: usize = NUM_PLAYERS;
    let mut levels = [-1i32; MAX_POTS];
    let take = uniq_vec.len().min(MAX_POTS);
    for i in 0..take {
        levels[i] = uniq_vec[i];
    }
    levels.sort_unstable();
    for i in 0..MAX_POTS {
        if levels[i] < 0 {
            levels[i] = -1;
        }
    }

    let mut prev_level: i32 = 0;
    for li in 0..MAX_POTS {
        let level = levels[li];
        let valid = level >= 0;
        let layer_size = if valid { level - prev_level } else { 0 };
        let mut n_contrib = 0i32;
        let mut contributing = [false; NUM_PLAYERS];
        for i in 0..NUM_PLAYERS {
            contributing[i] = capped[i] >= level && valid;
            if contributing[i] {
                n_contrib += 1;
            }
        }
        let pot_amt = layer_size * n_contrib;
        let mut eligible = [false; NUM_PLAYERS];
        for i in 0..NUM_PLAYERS {
            eligible[i] = (seat_status[i] == SEAT_LIVE || seat_status[i] == SEAT_ALLIN)
                && contributing[i];
        }
        let win = award_layer(pot_amt, &eligible, &scores, button);
        for i in 0..NUM_PLAYERS {
            winnings[i] += win[i];
        }
        if valid {
            prev_level = level;
        }
    }

    let mut out = [0i32; NUM_PLAYERS];
    for i in 0..NUM_PLAYERS {
        out[i] = winnings[i] + excess[i] - committed_total[i];
    }
    out
}

// ---------------------------------------------------------------------------
// Seat helpers
// ---------------------------------------------------------------------------

fn live_count(seat_status: &[i8; NUM_PLAYERS]) -> i32 {
    let mut c = 0;
    for &v in seat_status.iter() {
        if v == SEAT_LIVE || v == SEAT_ALLIN {
            c += 1;
        }
    }
    c
}

fn pot_for_sizing(s: &EnvState) -> i32 {
    let mut p = s.pot_dead;
    for i in 0..NUM_PLAYERS {
        p += s.committed_street[i];
    }
    p
}

fn first_live_after(seat_status: &[i8; NUM_PLAYERS], from_seat: i32) -> i32 {
    let base = from_seat;
    for off in 1..=(NUM_PLAYERS as i32) {
        let seat = ((base + off).rem_euclid(NUM_PLAYERS as i32)) as usize;
        if seat_status[seat] == SEAT_LIVE {
            return seat as i32;
        }
    }
    -1
}

fn first_postflop_actor(s: &EnvState) -> i32 {
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

pub fn is_chance_node(s: &EnvState) -> bool {
    s.pending_deal && !s.done
}

fn n_board(board: &[i32; MAX_BOARD]) -> usize {
    board.iter().filter(|&&c| c >= 0).count()
}

/// `_deal_n_cards`: deal n_new board cards from deck, advance ptr, advance rng_key 3 times.
fn deal_n_cards(s: &mut EnvState, n_new: usize) {
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
pub fn sample_chance(s_in: &EnvState) -> EnvState {
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
    s.committed_street = [0; NUM_PLAYERS];
    s.current_bet = 0;
    s.min_raise = BB_CHIPS;
    s.acted = [false; NUM_PLAYERS];
    s.last_aggressor = -1;
    s.last_raise_full = true;
    s
}

fn run_out(s: &mut EnvState) {
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

fn close_street(s: &mut EnvState) {
    let mut pot_dead = s.pot_dead;
    for i in 0..NUM_PLAYERS {
        pot_dead += s.committed_street[i];
    }
    s.pot_dead = pot_dead;
    s.committed_street = [0; NUM_PLAYERS];
    s.current_bet = 0;
    s.pending_deal = true;
    s.action_seat = -1;
    let actors = s.seat_status.iter().any(|&st| st == SEAT_LIVE);
    if !actors && s.street < STREET_RIVER {
        run_out(s);
    }
}

fn street_complete(s: &EnvState) -> bool {
    let cur = s.current_bet;
    let mut n_can_act = 0;
    let mut matched = true;
    let mut all_acted = true;
    for i in 0..NUM_PLAYERS {
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

fn commit(s: &mut EnvState, seat: usize, amount_in: i32) {
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
pub fn legal_action_mask(s: &EnvState, seat_in: i32) -> [bool; NUM_ACTIONS] {
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

fn action_amount(s: &EnvState, seat: usize, action: usize) -> i32 {
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
pub fn apply_action(s_in: &EnvState, action: usize) -> EnvState {
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
        s.acted = [false; NUM_PLAYERS];
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
        s.rewards = compute_returns(
            &s.holes,
            &s.board,
            &s.seat_status,
            &s.committed_total,
            s.button,
            n_live,
        );
    } else {
        s.rewards = [0; NUM_PLAYERS];
    }
    s
}
