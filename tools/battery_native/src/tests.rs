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

// ---------------------------------------------------------------------------
// Const-generic seat-count smoke: the engine state machine is generic over N.
// The 6-seat byte-identity is pinned by the Python gate (tests/test_battery_native.py);
// here we prove a non-default seat count (N=7) WORKS end-to-end: build a post-blinds
// EnvStateN<7> directly (mirroring jax_engine.init_hand's layout), drive a few
// apply_action::<7> / sample_chance::<7> steps to a terminal, and assert the
// compute_returns::<7> chip deltas sum to zero (chip conservation).
// ---------------------------------------------------------------------------

const SB_CHIPS: i32 = BB_CHIPS / 2; // 50, mirrors jax_engine.SB_CHIPS = CHIP_UNIT // 2

/// Build a 7-seat preflop post-blinds state equivalent to `init_hand` for button=0, all 7 seats
/// live, equal `stack_bb` BB stacks, a fixed 52-card deck (identity order), ante=0. Holes are dealt
/// from the top of the deck (2 per live seat, ascending seat order), board cleared.
fn init_hand_7(stack_bb: i32) -> EnvStateN<7> {
    let stack0 = stack_bb * BB_CHIPS;
    let mut stacks = [stack0; 7];
    let mut committed = [0i32; 7];
    let seat_status = [SEAT_LIVE; 7];

    // button = 0 => SB = seat 1, BB = seat 2, UTG (first to act) = seat 3 (n_live > 2).
    let button = 0i32;
    let sb = 1usize;
    let bb = 2usize;
    stacks[sb] -= SB_CHIPS;
    committed[sb] = SB_CHIPS;
    stacks[bb] -= BB_CHIPS;
    committed[bb] = BB_CHIPS;

    // Identity deck 0..52; deal 2 holes per seat off the top (14 cards), board cleared.
    let mut deck = [0i32; 52];
    for (i, d) in deck.iter_mut().enumerate() {
        *d = i as i32;
    }
    let mut holes = [[-1i32; 2]; 7];
    let mut p = 0usize;
    for seat in 0..7 {
        holes[seat][0] = deck[p];
        holes[seat][1] = deck[p + 1];
        p += 2;
    }
    let deck_ptr = p as i32; // next undealt card index (board deals continue from here)

    EnvStateN::<7> {
        holes,
        board: [-1i32; MAX_BOARD],
        seat_status,
        stacks,
        committed_street: committed,
        committed_total: committed,
        street: 0, // STREET_PREFLOP
        pot_dead: 0,
        current_bet: BB_CHIPS,
        min_raise: BB_CHIPS,
        button,
        action_seat: 3, // UTG
        last_aggressor: -1,
        acted: [false; 7],
        last_raise_full: true,
        deck,
        deck_ptr,
        rng_key: [0u32, 0u32],
        hist_seat: [-1i32; MAX_HISTORY],
        hist_action: [-1i32; MAX_HISTORY],
        hist_amount: [0i32; MAX_HISTORY],
        hist_len: 0,
        done: false,
        rewards: [0i32; 7],
        pending_deal: false,
    }
}

/// Drive `env` to a terminal by always taking the most passive legal action (CHECK > CALL > FOLD).
/// Resolves chance nodes via `sample_chance`. Returns the terminal state.
///
/// NOTE on this engine's terminal contract (mirrors `np_engine.apply_action_np` exactly): a river
/// check-down does NOT auto-finalize inside `apply_action`/`sample_chance`. The last river action
/// closes the street (`pending_deal=true`, `done=false`), then `sample_chance` advances `street` to
/// `STREET_TERMINAL` with `action_seat=-1` and `done` still false — the consumer is expected to call
/// `compute_returns` once it observes `street >= STREET_TERMINAL`. Only the fold-to-one-live and
/// all-in-runout lines set `done=true` from inside `apply_action`. So "terminal" here means
/// `done || street >= STREET_TERMINAL || action_seat < 0`.
fn play_passive_to_terminal(mut env: EnvStateN<7>) -> EnvStateN<7> {
    for _ in 0..400 {
        if env.done || env.street >= STREET_TERMINAL {
            break;
        }
        if is_chance_node(&env) {
            env = sample_chance(&env);
            continue;
        }
        let seat = env.action_seat;
        if seat < 0 {
            break;
        }
        let mask = legal_action_mask(&env, seat);
        // Prefer CHECK (1), else CALL (2), else FOLD (0); fall back to first legal.
        let action = if mask[CHECK] {
            CHECK
        } else if mask[CALL] {
            CALL
        } else if mask[FOLD] {
            FOLD
        } else {
            (0..mask.len()).find(|&a| mask[a]).unwrap_or(CHECK)
        };
        env = apply_action(&env, action);
    }
    env
}

#[test]
fn n7_instantiation_compiles_and_runs() {
    // Construct a 7-seat hand and confirm the generic struct/fields have width 7.
    let env = init_hand_7(100);
    assert_eq!(env.seat_status.len(), 7);
    assert_eq!(env.stacks.len(), 7);
    assert_eq!(env.holes.len(), 7);
    assert_eq!(env.rewards.len(), 7);
    assert_eq!(env.action_seat, 3); // UTG with button=0, 7 live
    // legal_action_mask::<7> must return a usable mask for the actor.
    let m = legal_action_mask(&env, env.action_seat);
    assert!(m.iter().any(|&b| b), "actor must have a legal action");
}

#[test]
fn n7_check_down_showdown_returns_sum_to_zero() {
    // Everyone passive => preflop all call/check to BB, then check the flop/turn/river down to a
    // multiway showdown. Drives sample_chance::<7> (3 streets) to the terminal river state.
    let env = play_passive_to_terminal(init_hand_7(100));
    // A check-down with everyone live reaches the river and shows down (street == terminal).
    assert_eq!(env.street, STREET_TERMINAL, "check-down must reach the showdown street");
    assert_eq!(env.action_seat, -1, "terminal state has no actor");
    // All 7 seats are still live at showdown (nobody folded).
    let n_showdown = env
        .seat_status
        .iter()
        .filter(|&&s| s == SEAT_LIVE || s == SEAT_ALLIN)
        .count() as i32;
    assert_eq!(n_showdown, 7);
    // The board is fully run out (5 cards) for a river showdown.
    assert_eq!(env.board.iter().filter(|&&c| c >= 0).count(), 5);
    // Per the engine's terminal contract, the consumer computes the showdown payout explicitly.
    let rewards = compute_returns::<7>(
        &env.holes,
        &env.board,
        &env.seat_status,
        &env.committed_total,
        env.button,
        n_showdown,
    );
    // Chip conservation: net deltas across all 7 seats sum to zero.
    let total: i64 = rewards.iter().map(|&r| r as i64).sum();
    assert_eq!(total, 0, "compute_returns::<7> must conserve chips (sum to zero)");
}

#[test]
fn n7_fold_around_returns_sum_to_zero() {
    // Drive a fold-heavy line by hand: UTG..button all fold, SB folds, BB wins uncontested.
    // This exercises the n_showdown <= 1 fold-win branch of compute_returns::<7>.
    let mut env = init_hand_7(100);
    // Fold every seat that is forced to act preflop until only BB remains.
    for _ in 0..400 {
        if env.done {
            break;
        }
        if is_chance_node(&env) {
            env = sample_chance(&env);
            continue;
        }
        let seat = env.action_seat;
        if seat < 0 {
            break;
        }
        let mask = legal_action_mask(&env, seat);
        // Fold whenever folding is legal (facing a bet); otherwise check.
        let action = if mask[FOLD] { FOLD } else { CHECK };
        env = apply_action(&env, action);
    }
    assert!(env.done, "fold-around hand must terminate");
    let total: i64 = env.rewards.iter().map(|&r| r as i64).sum();
    assert_eq!(total, 0, "fold-win compute_returns::<7> must conserve chips");
    // BB (seat 2) wins the dead small blind: net positive for exactly one seat.
    let winners = env.rewards.iter().filter(|&&r| r > 0).count();
    assert_eq!(winners, 1, "uncontested pot has exactly one net winner");
}
