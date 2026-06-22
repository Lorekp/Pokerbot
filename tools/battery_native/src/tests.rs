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

/// Build a 7-seat preflop root via the REAL generic `init_hand_n::<7>` (NOT a hand-built duplicate),
/// for button=0, all 7 seats live, equal `stack_bb` BB stacks, a fixed 52-card identity-order deck,
/// ante=0. This exercises the production deal+blinds+ante+occupancy+button path at N=7.
fn init_hand_7(stack_bb: i32) -> EnvStateN<7> {
    init_hand_7_ante(stack_bb, 0)
}

/// As `init_hand_7` but with a per-player `ante` (dead money -> pot_dead + committed_total).
fn init_hand_7_ante(stack_bb: i32, ante: i32) -> EnvStateN<7> {
    let stack0 = stack_bb * BB_CHIPS;
    let stacks = [stack0; 7];
    let occupancy = [true; 7];
    // Identity deck 0..52 (the supplied permutation; jax would supply its Threefry permutation).
    let mut deck = [0i32; 52];
    for (i, d) in deck.iter_mut().enumerate() {
        *d = i as i32;
    }
    init_hand_n::<7>(0, &stacks, &occupancy, ante, &deck, [0u32, 0u32])
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
fn play_passive_to_terminal(env: EnvStateN<7>) -> EnvStateN<7> {
    play_passive_to_terminal_n::<7>(env)
}

/// Generic-N version of `play_passive_to_terminal` (CHECK > CALL > FOLD), resolving chance nodes.
fn play_passive_to_terminal_n<const N: usize>(mut env: EnvStateN<N>) -> EnvStateN<N> {
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

#[test]
fn n7_root_positions_and_blinds_correct() {
    // 7-handed, button=0, all live: SB=1, BB=2, UTG=3 acts first. Blinds posted, ante=0.
    let env = init_hand_7(100);
    assert_eq!(env.button, 0);
    // committed_street: SB=50, BB=100, everyone else 0.
    assert_eq!(env.committed_street[1], SB_CHIPS, "SB posts 50");
    assert_eq!(env.committed_street[2], BB_CHIPS, "BB posts 100");
    for s in [0usize, 3, 4, 5, 6] {
        assert_eq!(env.committed_street[s], 0, "non-blind seat {s} posts nothing");
    }
    // Stacks reduced by the blind only (ante=0).
    assert_eq!(env.stacks[1], 100 * BB_CHIPS - SB_CHIPS);
    assert_eq!(env.stacks[2], 100 * BB_CHIPS - BB_CHIPS);
    assert_eq!(env.stacks[0], 100 * BB_CHIPS);
    // current_bet = max committed = BB; first actor = UTG = seat 3 (n_live > 2).
    assert_eq!(env.current_bet, BB_CHIPS);
    assert_eq!(env.min_raise, BB_CHIPS);
    assert_eq!(env.action_seat, 3, "UTG acts first 7-handed");
    // ante=0 byte-identity invariant: committed_total == committed_street, pot_dead == 0.
    assert_eq!(env.committed_total, env.committed_street);
    assert_eq!(env.pot_dead, 0);
    // 14 cards dealt (2 * 7 live), deck_ptr advanced.
    assert_eq!(env.deck_ptr, 14);
    for s in 0..7 {
        assert!(env.holes[s][0] >= 0 && env.holes[s][1] >= 0, "seat {s} dealt");
    }
}

#[test]
fn n7_root_button_rotation() {
    // Rotating the button rotates SB/BB/UTG by the same generic clockwise walk.
    // button=3, 7 live: SB=4, BB=5, UTG=6.
    let stacks = [100 * BB_CHIPS; 7];
    let occ = [true; 7];
    let mut deck = [0i32; 52];
    for (i, d) in deck.iter_mut().enumerate() {
        *d = i as i32;
    }
    let env = init_hand_n::<7>(3, &stacks, &occ, 0, &deck, [0, 0]);
    assert_eq!(env.committed_street[4], SB_CHIPS, "SB = btn+1 = seat 4");
    assert_eq!(env.committed_street[5], BB_CHIPS, "BB = btn+2 = seat 5");
    assert_eq!(env.action_seat, 6, "UTG = btn+3 = seat 6");
    // Wrapping button=6: SB=0, BB=1, UTG=2.
    let env2 = init_hand_n::<7>(6, &stacks, &occ, 0, &deck, [0, 0]);
    assert_eq!(env2.committed_street[0], SB_CHIPS);
    assert_eq!(env2.committed_street[1], BB_CHIPS);
    assert_eq!(env2.action_seat, 2);
}

#[test]
fn n7_ante_is_dead_money_and_conserves() {
    // Per-player ante goes to pot_dead + committed_total, NOT committed_street.
    let ante = 13i32;
    let env = init_hand_7_ante(100, ante);
    // committed_street is blinds-only (ante NOT here).
    assert_eq!(env.committed_street[1], SB_CHIPS);
    assert_eq!(env.committed_street[2], BB_CHIPS);
    assert_eq!(env.committed_street[3], 0);
    // committed_total = committed_street + ante for every live seat.
    for s in 0..7 {
        assert_eq!(
            env.committed_total[s],
            env.committed_street[s] + ante,
            "seat {s} committed_total includes its ante"
        );
    }
    // pot_dead == sum of antes (7 live seats).
    assert_eq!(env.pot_dead, ante * 7);
    // Stacks reduced by blind + ante.
    assert_eq!(env.stacks[1], 100 * BB_CHIPS - SB_CHIPS - ante);
    assert_eq!(env.stacks[2], 100 * BB_CHIPS - BB_CHIPS - ante);
    assert_eq!(env.stacks[0], 100 * BB_CHIPS - ante);
    // Chip conservation through a full check-down showdown with antes in the pot.
    let term = play_passive_to_terminal(env);
    assert_eq!(term.street, STREET_TERMINAL);
    let n_showdown = term
        .seat_status
        .iter()
        .filter(|&&s| s == SEAT_LIVE || s == SEAT_ALLIN)
        .count() as i32;
    let rewards = compute_returns::<7>(
        &term.holes,
        &term.board,
        &term.seat_status,
        &term.committed_total,
        term.button,
        n_showdown,
    );
    let total: i64 = rewards.iter().map(|&r| r as i64).sum();
    assert_eq!(total, 0, "ante pot must still conserve chips");
}

#[test]
fn n7_occupancy_empty_seats_skipped() {
    // 7-seat table with 4 empty seats (occupancy = [T,F,T,F,T,F,T]) => 4 live; empties are SEAT_EMPTY,
    // get no holes, and are skipped by the clockwise blind/UTG walk.
    let stacks = [100 * BB_CHIPS; 7];
    let occ = [true, false, true, false, true, false, true];
    let mut deck = [0i32; 52];
    for (i, d) in deck.iter_mut().enumerate() {
        *d = i as i32;
    }
    let env = init_hand_n::<7>(0, &stacks, &occ, 0, &deck, [0, 0]);
    // Empty seats stay SEAT_EMPTY with no holes.
    for s in [1usize, 3, 5] {
        assert_eq!(env.seat_status[s], SEAT_EMPTY, "seat {s} empty");
        assert_eq!(env.holes[s], [-1, -1], "empty seat {s} undealt");
    }
    let n_live = env.seat_status.iter().filter(|&&s| s == SEAT_LIVE).count();
    assert_eq!(n_live, 4);
    // button=0; live seats are 0,2,4,6. SB = 1st live after 0 = 2, BB = 2nd live after = 4,
    // UTG = 3rd live after = 6.
    assert_eq!(env.committed_street[2], SB_CHIPS, "SB = first live after button");
    assert_eq!(env.committed_street[4], BB_CHIPS, "BB = second live after button");
    assert_eq!(env.action_seat, 6, "UTG = third live after button (4 live)");
    // Only live seats dealt: deck_ptr = 2 * 4.
    assert_eq!(env.deck_ptr, 8);
    // Play to terminal; chips conserve over the live subset.
    let term = play_passive_to_terminal(env);
    let n_showdown = term
        .seat_status
        .iter()
        .filter(|&&s| s == SEAT_LIVE || s == SEAT_ALLIN)
        .count() as i32;
    let rewards = compute_returns::<7>(
        &term.holes,
        &term.board,
        &term.seat_status,
        &term.committed_total,
        term.button,
        n_showdown,
    );
    let total: i64 = rewards.iter().map(|&r| r as i64).sum();
    assert_eq!(total, 0, "occupancy-mixed table conserves chips");
}

#[test]
fn n2_heads_up_root_button_is_sb() {
    // HU root via the generic path at N=2. jax posts SB = nth_live_after(button,1),
    // BB = nth_live_after(button,2); with 2 live seats SB = the non-button seat, BB = button.
    // Preflop the button (= BB poster here) acts first via the n_live<=2 branch -> action_seat = btn.
    let stacks = [100 * BB_CHIPS; 2];
    let occ = [true; 2];
    let mut deck = [0i32; 52];
    for (i, d) in deck.iter_mut().enumerate() {
        *d = i as i32;
    }
    let env = init_hand_n::<2>(0, &stacks, &occ, 0, &deck, [0, 0]);
    // button=0: SB = nth_live_after(0,1) = seat 1, BB = nth_live_after(0,2) = seat 0 (wrap).
    assert_eq!(env.committed_street[1], SB_CHIPS, "HU SB = non-button seat");
    assert_eq!(env.committed_street[0], BB_CHIPS, "HU BB = button seat");
    assert_eq!(env.current_bet, BB_CHIPS);
    // n_live <= 2 => first actor is btn_first = first live after button = seat 1 (the SB).
    assert_eq!(env.action_seat, 1, "HU preflop SB acts first");
    assert_eq!(env.deck_ptr, 4, "HU deals 2 seats");
    // Full hand conserves chips.
    let term = play_passive_to_terminal_n::<2>(env);
    let n_showdown = term
        .seat_status
        .iter()
        .filter(|&&s| s == SEAT_LIVE || s == SEAT_ALLIN)
        .count() as i32;
    let rewards = compute_returns::<2>(
        &term.holes,
        &term.board,
        &term.seat_status,
        &term.committed_total,
        term.button,
        n_showdown,
    );
    let total: i64 = rewards.iter().map(|&r| r as i64).sum();
    assert_eq!(total, 0, "HU root conserves chips");
}

// ---------------------------------------------------------------------------
// Const-generic SUBGAME SOLVER (subgame.rs) at N=7 — the tournament-engine unlock.
//
// The engine state machine was already generic; this proves the depth-limited CFR SUBGAME SOLVE
// (the GTO/chip-EV base the tournament deep-stack serve uses) runs end-to-end at N=7 via the new
// EnvStateN<7> root, through the now-generic playout leaf + fingerprint blueprint lookup. With an
// EMPTY blueprint the policy falls back to uniform-over-legal (the pure chip-EV base), which is
// exactly the "chip-EV subgame solve completes 7-handed" acceptance item. We assert the solver
// returns a valid probability distribution supported only on the legal action set.
// ---------------------------------------------------------------------------
mod subgame_n7 {
    use super::*;
    use crate::playout::{Blueprint, PlayoutConfig};
    use crate::subgame::{self, SubgameConfig};
    use crate::core::{legal_action_mask, NUM_ACTIONS, BET_BUCKET_START, ALL_IN};
    use std::collections::HashMap;

    fn allowed_all() -> [bool; NUM_ACTIONS] {
        // Allow the full menu (fold/check/call + all 7 bet buckets + all-in) on every street.
        [true; NUM_ACTIONS]
    }

    fn pcfg() -> PlayoutConfig {
        PlayoutConfig {
            allowed_pre: allowed_all(),
            allowed_post: allowed_all(),
            bias_mult: 5.0,
            bias_floor: 0.0,
            max_depth: 24,        // per-leaf rollout depth
            num_buckets: 200,
        }
    }

    fn scfg() -> SubgameConfig {
        SubgameConfig {
            max_iterations: 32,   // small but real CFR budget — proves the loop runs, not just types
            max_depth: 8,         // betting-round CFR recursion depth
            blueprint_mix: 0.0,   // no blueprint -> pure regret-match (chip-EV base)
            rollout_samples: 2,
            resample_board: true,
            linear_averaging: true,
        }
    }

    fn assert_valid_root_probs(env: &EnvStateN<7>, hero: usize, probs: &[f64; NUM_ACTIONS]) {
        let legal = legal_action_mask::<7>(env, hero as i32);
        let mut s = 0.0;
        for a in 0..NUM_ACTIONS {
            assert!(probs[a] >= -1e-12, "prob[{a}] negative: {}", probs[a]);
            if !legal[a] {
                assert!(probs[a].abs() < 1e-9, "mass {} on illegal action {a}", probs[a]);
            }
            s += probs[a];
        }
        assert!((s - 1.0).abs() < 1e-6, "root probs must sum to 1, got {s}");
    }

    #[test]
    fn n7_subgame_solve_completes_and_returns_valid_distribution() {
        // 7-handed, 40bb deep, button=0 -> UTG (seat 3) is the first actor / hero.
        let env = init_hand_7(40);
        let hero = env.action_seat as usize;
        assert_eq!(hero, 3, "7-handed UTG is the first preflop actor");
        let bp: Blueprint = HashMap::new();    // no 7-seat blueprint -> uniform fallback = chip-EV base
        let pc = pcfg();
        let sc = scfg();
        let probs = subgame::solve::<7>(&env, hero, &bp, &pc, &sc, 0xC0FFEE);
        let arr: [f64; NUM_ACTIONS] = probs;
        assert_valid_root_probs(&env, hero, &arr);
        // The chip-EV base must put SOME mass on an aggressive/continuing action (not a degenerate
        // all-fold), confirming the regret recursion actually explored beyond the fold leaf.
        let aggressive: f64 = (BET_BUCKET_START..=ALL_IN).map(|a| arr[a]).sum::<f64>()
            + arr[crate::core::CALL] + arr[crate::core::CHECK];
        assert!(aggressive > 0.0, "chip-EV solve degenerated to pure fold");
    }

    #[test]
    fn n7_subgame_solve_deterministic_for_fixed_seed() {
        // Pure function of (public state, config, seed): same inputs -> identical root probs.
        let env = init_hand_7(25);
        let hero = env.action_seat as usize;
        let bp: Blueprint = HashMap::new();
        let pc = pcfg();
        let sc = scfg();
        let a = subgame::solve::<7>(&env, hero, &bp, &pc, &sc, 12345);
        let b = subgame::solve::<7>(&env, hero, &bp, &pc, &sc, 12345);
        assert_eq!(a, b, "fixed-seed 7-handed solve must be deterministic");
    }

    #[test]
    fn n7_subgame_solve_postflop_node() {
        // Drive the 7-handed root to a postflop decision (where the blueprint bucket path + leaf
        // rollout both engage), then solve — proves the generic fingerprint/leaf at a board state.
        let mut env = init_hand_7(40);
        // Everyone calls/checks to the flop via the passive driver, then stop at the first postflop
        // actor. Reuse the passive walk but break at street >= FLOP with a live actor.
        for _ in 0..64 {
            if env.done || env.street >= STREET_TERMINAL { break; }
            if crate::core::is_chance_node::<7>(&env) {
                env = crate::core::sample_chance::<7>(&env);
                if env.street >= STREET_FLOP && env.action_seat >= 0 { break; }
                continue;
            }
            let seat = env.action_seat;
            if seat < 0 { break; }
            let mask = legal_action_mask::<7>(&env, seat);
            let a = if mask[crate::core::CHECK] { crate::core::CHECK }
                    else if mask[crate::core::CALL] { crate::core::CALL }
                    else if mask[crate::core::FOLD] { crate::core::FOLD }
                    else { (0..NUM_ACTIONS).find(|&i| mask[i]).unwrap() };
            env = crate::core::apply_action::<7>(&env, a);
        }
        if env.street >= STREET_FLOP && !env.done && env.action_seat >= 0 {
            let hero = env.action_seat as usize;
            let bp: Blueprint = HashMap::new();
            let probs = subgame::solve::<7>(&env, hero, &bp, &pcfg(), &scfg(), 99);
            assert_valid_root_probs(&env, hero, &probs);
        }
    }
}
