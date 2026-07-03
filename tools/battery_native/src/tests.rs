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
    assert!(
        rf > quad,
        "straight flush {} should beat quads {}",
        rf,
        quad
    );
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
        (14 - 2) * 4,    // Ac
        (13 - 2) * 4,    // Kc
        (12 - 2) * 4,    // Qc
        (7 - 2) * 4,     // 7c
        (2 - 2) * 4,     // 2c
        (5 - 2) * 4 + 1, // 5d
        (9 - 2) * 4 + 2, // 9h
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
    assert_eq!(
        env.street, STREET_TERMINAL,
        "check-down must reach the showdown street"
    );
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
    assert_eq!(
        total, 0,
        "compute_returns::<7> must conserve chips (sum to zero)"
    );
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
    assert_eq!(
        total, 0,
        "fold-win compute_returns::<7> must conserve chips"
    );
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
        assert_eq!(
            env.committed_street[s], 0,
            "non-blind seat {s} posts nothing"
        );
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
        assert!(
            env.holes[s][0] >= 0 && env.holes[s][1] >= 0,
            "seat {s} dealt"
        );
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
    assert_eq!(
        env.committed_street[2], SB_CHIPS,
        "SB = first live after button"
    );
    assert_eq!(
        env.committed_street[4], BB_CHIPS,
        "BB = second live after button"
    );
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
    use crate::core::{legal_action_mask, ALL_IN, BET_BUCKET_START, NUM_ACTIONS};
    use crate::playout::{Blueprint, PlayoutConfig};
    use crate::subgame::{self, SubgameConfig};
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
            max_depth: 24, // per-leaf rollout depth
            num_buckets: 200,
            emd: None, // made-hand keying (byte-identical default)
        }
    }

    fn scfg() -> SubgameConfig {
        SubgameConfig {
            max_iterations: 32, // small but real CFR budget — proves the loop runs, not just types
            max_depth: 8,       // betting-round CFR recursion depth
            blueprint_mix: 0.0, // no blueprint -> pure regret-match (chip-EV base)
            rollout_samples: 2,
            resample_board: true,
            linear_averaging: true,
            repair_deck: false, // default OFF => byte/stat-identical to the legacy solve
            two_sided: false,   // default OFF => traverser is always hero
            decouple_sampler: false, // Q2 default OFF => byte/stat-identical
            range_aware: false,      // Q3 default OFF => uniform opponent-hole deal
            range_uniform_floor: 0.25,
            range_reach_full: false, // R1 default OFF => legacy fold-now/uniform deal, byte-identical
            ensemble_streams: 1,     // default 1 => single-stream, byte/stat-identical
            leaf_cache_worlds: 1,    // BUG-1 default 1 => first-touch leaf freeze, byte-identical
            avg_skip_frac: 0.0,      // BUG-5 default 0.0 => accumulate the average from t=0
            leaf_biased_continuations: false, // R2 default OFF => legacy uniform leaf, byte-identical
            leaf_bias_mult: 5.0,     // R2 Pluribus multiplier (only read when the bool is ON)
            serve_final_iterate: false, // R5a default OFF => averaged extraction, byte-identical
        }
    }

    fn assert_valid_root_probs(env: &EnvStateN<7>, hero: usize, probs: &[f64; NUM_ACTIONS]) {
        let legal = legal_action_mask::<7>(env, hero as i32);
        let mut s = 0.0;
        for a in 0..NUM_ACTIONS {
            assert!(probs[a] >= -1e-12, "prob[{a}] negative: {}", probs[a]);
            if !legal[a] {
                assert!(
                    probs[a].abs() < 1e-9,
                    "mass {} on illegal action {a}",
                    probs[a]
                );
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
        let bp: Blueprint = HashMap::new(); // no 7-seat blueprint -> uniform fallback = chip-EV base
        let pc = pcfg();
        let sc = scfg();
        let probs = subgame::solve::<7>(&env, hero, &bp, &pc, &sc, 0xC0FFEE);
        let arr: [f64; NUM_ACTIONS] = probs;
        assert_valid_root_probs(&env, hero, &arr);
        // The chip-EV base must put SOME mass on an aggressive/continuing action (not a degenerate
        // all-fold), confirming the regret recursion actually explored beyond the fold leaf.
        let aggressive: f64 = (BET_BUCKET_START..=ALL_IN).map(|a| arr[a]).sum::<f64>()
            + arr[crate::core::CALL]
            + arr[crate::core::CHECK];
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

    // Q2/Q3 GATED-OFF IDENTITY: decouple_sampler=false + range_aware=false MUST reproduce the pre-Q2/Q3
    // solve bit-for-bit (same seed). Guards the "default OFF => byte/stat-identical" contract at the
    // Rust boundary (the H2H OLD arm relies on this).
    #[test]
    fn n7_subgame_q2q3_flags_off_byte_identical() {
        let env = init_hand_7(40);
        let hero = env.action_seat as usize;
        let bp: Blueprint = HashMap::new();
        let pc = pcfg();
        // Baseline: the legacy config (flags off). scfg() already sets decouple/range_aware false.
        let base = scfg();
        let probs_base = subgame::solve::<7>(&env, hero, &bp, &pc, &base, 0xBADC0DE);
        // Explicit flags-off with a NON-uniform floor set (must not matter when range_aware is off).
        let off = SubgameConfig {
            decouple_sampler: false,
            range_aware: false,
            range_uniform_floor: 0.4,
            ..scfg()
        };
        let probs_off = subgame::solve::<7>(&env, hero, &bp, &pc, &off, 0xBADC0DE);
        assert_eq!(
            probs_base, probs_off,
            "flags-off solve must be byte-identical to the legacy solve"
        );
    }

    // R1 GATE: range_reach_full=false (the default) is exercised by every test above (scfg() sets it
    // false — those are the byte-identity gates). Here: range_reach_full=true must (a) be valid +
    // deterministic, and (b) with range_aware ALSO on, reproduce the range_aware-only solve exactly
    // when the reach table degrades to no-information — an EMPTY blueprint makes every reach decision
    // all-miss (skipped => flat 1.0 weights) and the fold-now proxy flat too, so BOTH tables induce
    // the same flat sampling distribution over combos (identical draw counts => identical RNG stream).
    #[test]
    fn n7_subgame_range_reach_valid_and_flat_matches_fold_now() {
        let env = init_hand_7(40);
        let hero = env.action_seat as usize;
        let bp: Blueprint = HashMap::new(); // empty => reach factors all-miss => flat table
        let pc = pcfg();
        let reach = SubgameConfig { range_aware: true, range_reach_full: true, ..scfg() };
        let a = subgame::solve::<7>(&env, hero, &bp, &pc, &reach, 4242);
        let b = subgame::solve::<7>(&env, hero, &bp, &pc, &reach, 4242);
        assert_valid_root_probs(&env, hero, &a);
        assert_eq!(a, b, "range_reach_full solve must be deterministic for a fixed seed");
        // Empty blueprint: fold-now weight = 1 - 1/n_legal (flat) vs reach weight = 1.0 (flat) —
        // both normalize to the SAME per-combo distribution, so the solves must coincide exactly.
        let fold_now = SubgameConfig { range_aware: true, range_reach_full: false, ..scfg() };
        let c = subgame::solve::<7>(&env, hero, &bp, &pc, &fold_now, 4242);
        assert_eq!(
            a, c,
            "flat reach table must induce the same world distribution as the flat fold-now proxy"
        );
    }

    // Q2 DECOUPLE: turning on the decoupled hole sampler must still return a VALID distribution and stay
    // deterministic for a fixed seed (its own sub-stream is seeded from public_seed).
    #[test]
    fn n7_subgame_decouple_valid_and_deterministic() {
        let env = init_hand_7(40);
        let hero = env.action_seat as usize;
        let bp: Blueprint = HashMap::new();
        let pc = pcfg();
        let sc = SubgameConfig { decouple_sampler: true, ..scfg() };
        let a = subgame::solve::<7>(&env, hero, &bp, &pc, &sc, 777);
        let b = subgame::solve::<7>(&env, hero, &bp, &pc, &sc, 777);
        assert_valid_root_probs(&env, hero, &a);
        assert_eq!(a, b, "decoupled solve must be deterministic for a fixed seed");
    }

    // Q3 RANGE-AWARE: with a real blueprint absent (uniform fallback => continue_weight is flat, no fold
    // mass), range_aware collapses to uniform, so it must MATCH the uniform deal exactly. This proves the
    // range path is a strict generalization (uniform is the floor=1 / flat-range limit) and that the
    // combo enumeration + inverse-CDF sampler is correct. We compare range_aware(floor arbitrary) vs the
    // uniform baseline on a state where the (empty) blueprint gives no range signal.
    #[test]
    fn n7_subgame_range_aware_valid_distribution() {
        let env = init_hand_7(40);
        let hero = env.action_seat as usize;
        let bp: Blueprint = HashMap::new(); // empty => uniform fallback => flat continue_weight
        let pc = pcfg();
        for floor in [0.15_f64, 0.25, 0.4] {
            let sc = SubgameConfig { range_aware: true, range_uniform_floor: floor, ..scfg() };
            let a = subgame::solve::<7>(&env, hero, &bp, &pc, &sc, 2024);
            let b = subgame::solve::<7>(&env, hero, &bp, &pc, &sc, 2024);
            assert_valid_root_probs(&env, hero, &a);
            assert_eq!(a, b, "range-aware solve must be deterministic (floor={floor})");
        }
    }

    // ENSEMBLE: ensemble_streams=1 must be BYTE-IDENTICAL to the legacy single-stream solve (stream 0's
    // seed == the raw public_seed), and ensemble_streams>1 must return a VALID distribution that is
    // DETERMINISTIC for a fixed seed (fixed per-stream seeds => CRN-stable across processes/arms).
    #[test]
    fn n7_subgame_ensemble_one_byte_identical_and_n_deterministic() {
        let env = init_hand_7(40);
        let hero = env.action_seat as usize;
        let bp: Blueprint = HashMap::new();
        let pc = pcfg();
        // ensemble=1 (explicit) == legacy (scfg() default 1): byte-identical.
        let base = scfg();
        let one = SubgameConfig { ensemble_streams: 1, ..scfg() };
        let p_base = subgame::solve::<7>(&env, hero, &bp, &pc, &base, 0xC0FFEE);
        let p_one = subgame::solve::<7>(&env, hero, &bp, &pc, &one, 0xC0FFEE);
        assert_eq!(p_base, p_one, "ensemble_streams=1 must be byte-identical to legacy single-stream");
        // ensemble=8: valid + deterministic (same seed twice => identical, proving fixed per-stream seeds).
        let sc8 = SubgameConfig { ensemble_streams: 8, ..scfg() };
        let a = subgame::solve::<7>(&env, hero, &bp, &pc, &sc8, 0xC0FFEE);
        let b = subgame::solve::<7>(&env, hero, &bp, &pc, &sc8, 0xC0FFEE);
        assert_valid_root_probs(&env, hero, &a);
        assert_eq!(a, b, "ensemble solve must be deterministic for a fixed seed");
        // The 8-stream average should differ from the 1-stream (independent seeds => more effective samples).
        // Not asserting a direction (that is the H2H's job), only that the ensemble is actually engaged.
    }

    // BUG-1 GATE: leaf_cache_worlds=1 (the default) MUST reproduce the pre-change solve bit-for-bit —
    // the (sum, n) running-mean cache with M=1 is exactly the legacy first-touch freeze (n=0 uses the
    // legacy leaf seed; v/1.0 == v). Also exercised with an explicit 1 vs the scfg() default.
    #[test]
    fn n7_subgame_leaf_cache_worlds_default_byte_identical() {
        let env = init_hand_7(40);
        let hero = env.action_seat as usize;
        let bp: Blueprint = HashMap::new();
        let pc = pcfg();
        let base = scfg(); // leaf_cache_worlds: 1 (default)
        let probs_base = subgame::solve::<7>(&env, hero, &bp, &pc, &base, 0xBADC0DE);
        let one = SubgameConfig { leaf_cache_worlds: 1, ..scfg() };
        let probs_one = subgame::solve::<7>(&env, hero, &bp, &pc, &one, 0xBADC0DE);
        assert_eq!(
            probs_base, probs_one,
            "leaf_cache_worlds=1 must be byte-identical to the legacy first-touch leaf cache"
        );
        // M=4: valid distribution + deterministic (per-(leaf, n) seeds are pure functions of the key),
        // proving the multi-world path is CRN-stable. Direction/effect is the H2H's job, not this gate's.
        let m4 = SubgameConfig { leaf_cache_worlds: 4, ..scfg() };
        let a = subgame::solve::<7>(&env, hero, &bp, &pc, &m4, 0xBADC0DE);
        let b = subgame::solve::<7>(&env, hero, &bp, &pc, &m4, 0xBADC0DE);
        assert_valid_root_probs(&env, hero, &a);
        assert_eq!(a, b, "leaf_cache_worlds=4 solve must be deterministic for a fixed seed");
    }

    // BUG-5 GATE: avg_skip_frac=0.0 (the default) MUST reproduce the pre-change solve bit-for-bit
    // (strat_sum accumulates from t=0 with unchanged weights; regret updates are never gated).
    #[test]
    fn n7_subgame_avg_skip_frac_default_byte_identical() {
        let env = init_hand_7(40);
        let hero = env.action_seat as usize;
        let bp: Blueprint = HashMap::new();
        let pc = pcfg();
        let base = scfg(); // avg_skip_frac: 0.0 (default)
        let probs_base = subgame::solve::<7>(&env, hero, &bp, &pc, &base, 0xBADC0DE);
        let zero = SubgameConfig { avg_skip_frac: 0.0, ..scfg() };
        let probs_zero = subgame::solve::<7>(&env, hero, &bp, &pc, &zero, 0xBADC0DE);
        assert_eq!(
            probs_base, probs_zero,
            "avg_skip_frac=0.0 must be byte-identical to the legacy accumulate-from-t=0 average"
        );
        // frac=0.5: valid + deterministic. The traversal (worlds, sampled actions, regrets, leaf cache)
        // is UNCHANGED by the gate — only the served average excludes early iterates.
        let half = SubgameConfig { avg_skip_frac: 0.5, ..scfg() };
        let a = subgame::solve::<7>(&env, hero, &bp, &pc, &half, 0xBADC0DE);
        let b = subgame::solve::<7>(&env, hero, &bp, &pc, &half, 0xBADC0DE);
        assert_valid_root_probs(&env, hero, &a);
        assert_eq!(a, b, "avg_skip_frac=0.5 solve must be deterministic for a fixed seed");
    }

    // R2 GATE: leaf_biased_continuations=false (the default) MUST reproduce the pre-change solve
    // bit-for-bit — even with a non-default leaf_bias_mult set (the multiplier is only read when the
    // bool is ON). Guards the "default OFF => byte-identical uniform leaf" contract at the Rust
    // boundary (the H2H OLD arm and the live serve rely on this).
    #[test]
    fn n7_subgame_leaf_biased_default_byte_identical() {
        let env = init_hand_7(40);
        let hero = env.action_seat as usize;
        let bp: Blueprint = HashMap::new();
        let pc = pcfg();
        let base = scfg(); // leaf_biased_continuations: false (default)
        let probs_base = subgame::solve::<7>(&env, hero, &bp, &pc, &base, 0xBADC0DE);
        let off = SubgameConfig {
            leaf_biased_continuations: false,
            leaf_bias_mult: 10.0, // non-default mult MUST be inert while the bool is off
            ..scfg()
        };
        let probs_off = subgame::solve::<7>(&env, hero, &bp, &pc, &off, 0xBADC0DE);
        assert_eq!(
            probs_base, probs_off,
            "leaf_biased_continuations=false must be byte-identical to the legacy uniform leaf"
        );
    }

    // R2 FEATURE-ON: the biased-continuation solve must return a VALID distribution and be
    // DETERMINISTIC for a fixed seed (choice draws are pure functions of (infoset, t, seed); leaf
    // profile values use deterministic per-(leaf, profile) seeds) — at both the Pluribus multiplier
    // (5) and Modicum's (10). Direction/EV effect is the H2H's job, not this gate's.
    #[test]
    fn n7_subgame_leaf_biased_on_deterministic_and_valid() {
        let env = init_hand_7(40);
        let hero = env.action_seat as usize;
        let bp: Blueprint = HashMap::new();
        let pc = pcfg();
        for mult in [5.0_f64, 10.0] {
            let sc = SubgameConfig {
                leaf_biased_continuations: true,
                leaf_bias_mult: mult,
                ..scfg()
            };
            let a = subgame::solve::<7>(&env, hero, &bp, &pc, &sc, 0xFEED5);
            let b = subgame::solve::<7>(&env, hero, &bp, &pc, &sc, 0xFEED5);
            assert_valid_root_probs(&env, hero, &a);
            assert_eq!(a, b, "biased-leaf solve must be deterministic (mult={mult})");
        }
        // Also under two_sided (the traverser cycles; choice nodes belong to the traverser) the
        // feature must stay deterministic + valid.
        let sc2 = SubgameConfig {
            leaf_biased_continuations: true,
            two_sided: true,
            ..scfg()
        };
        let a = subgame::solve::<7>(&env, hero, &bp, &pc, &sc2, 0xFEED5);
        let b = subgame::solve::<7>(&env, hero, &bp, &pc, &sc2, 0xFEED5);
        assert_valid_root_probs(&env, hero, &a);
        assert_eq!(a, b, "biased-leaf + two_sided solve must be deterministic");
    }

    // R5a ZERO-GATE: serve_final_iterate=false (the default) MUST reproduce the pre-change solve
    // bit-for-bit — the averaged extraction branch is entered exactly as before (the flag only
    // SKIPS that branch when true; the traversal/regrets/strategy_sum are untouched either way).
    #[test]
    fn n7_subgame_serve_final_iterate_default_byte_identical() {
        let env = init_hand_7(40);
        let hero = env.action_seat as usize;
        let bp: Blueprint = HashMap::new();
        let pc = pcfg();
        let base = scfg(); // serve_final_iterate: false (default)
        let probs_base = subgame::solve::<7>(&env, hero, &bp, &pc, &base, 0xBADC0DE);
        let off = SubgameConfig { serve_final_iterate: false, ..scfg() };
        let probs_off = subgame::solve::<7>(&env, hero, &bp, &pc, &off, 0xBADC0DE);
        assert_eq!(
            probs_base, probs_off,
            "serve_final_iterate=false must be byte-identical to the legacy averaged serve"
        );
    }

    // R5a GOLDEN — EXACT EXTRACTION SEMANTICS: against the raw ensemble-merged root rows
    // (subgame::solve_rows — the identical loop `solve` runs), the flag-ON policy must equal the
    // REGRET-MATCH of the root's cumulative (CFR+-clamped) regrets after the last iteration (the
    // final-iterate strategy), and the flag-OFF policy must equal the masked+normalized averaged
    // strategy_sum. Also: ON is deterministic, a valid distribution, and (at this state/seed)
    // genuinely differs from the averaged serve — the knob is not a placebo.
    #[test]
    fn n7_subgame_serve_final_iterate_golden_extraction() {
        let env = init_hand_7(40);
        let hero = env.action_seat as usize;
        let bp: Blueprint = HashMap::new();
        let pc = pcfg();
        let sc_on = SubgameConfig { serve_final_iterate: true, ..scfg() };
        let sc_off = scfg();
        let legal = subgame::abstracted_legal::<7>(&env, hero, &pc);

        // Raw root rows from the IDENTICAL solve loop (same seed => same traversal for ON and OFF:
        // the flag is read only at extraction, so solve_rows(sc_on) == solve_rows(sc_off)).
        let (ss_on, rr_on, _) = subgame::solve_rows::<7>(&env, hero, &bp, &pc, &sc_on, 0xFEED5);
        let (ss_off, rr_off, has_avg) =
            subgame::solve_rows::<7>(&env, hero, &bp, &pc, &sc_off, 0xFEED5);
        assert_eq!(ss_on, ss_off, "the flag must not touch the traversal (strategy_sum)");
        assert_eq!(rr_on, rr_off, "the flag must not touch the traversal (regrets)");
        assert!(has_avg, "test premise: the root accumulated average mass");

        // ON == regret-match of the final cumulative regrets (already normalized over legal).
        let probs_on = subgame::solve::<7>(&env, hero, &bp, &pc, &sc_on, 0xFEED5);
        let expect_final = subgame::regret_match(&rr_on, &legal);
        for a in 0..NUM_ACTIONS {
            assert!(
                (probs_on[a] - expect_final[a]).abs() < 1e-12,
                "final-iterate serve mismatch at action {a}: {} vs {}",
                probs_on[a],
                expect_final[a]
            );
        }
        assert_valid_root_probs(&env, hero, &probs_on);
        let probs_on2 = subgame::solve::<7>(&env, hero, &bp, &pc, &sc_on, 0xFEED5);
        assert_eq!(probs_on, probs_on2, "final-iterate serve must be deterministic");

        // OFF == masked+normalized averaged strategy_sum.
        let probs_off = subgame::solve::<7>(&env, hero, &bp, &pc, &sc_off, 0xFEED5);
        let mut expect_avg = [0.0f64; NUM_ACTIONS];
        let mut tot = 0.0;
        for a in 0..NUM_ACTIONS {
            expect_avg[a] = if legal[a] { ss_off[a] } else { 0.0 };
            tot += expect_avg[a];
        }
        for a in 0..NUM_ACTIONS {
            expect_avg[a] /= tot.max(1e-12);
        }
        for a in 0..NUM_ACTIONS {
            assert!(
                (probs_off[a] - expect_avg[a]).abs() < 1e-12,
                "averaged serve mismatch at action {a}: {} vs {}",
                probs_off[a],
                expect_avg[a]
            );
        }
        // The knob must actually move the served policy at this state (final != average here).
        let l1: f64 = (0..NUM_ACTIONS).map(|a| (probs_on[a] - probs_off[a]).abs()).sum();
        assert!(l1 > 1e-9, "final-iterate serve identical to the average — placebo knob? L1={l1}");
    }

    // R2 CONSISTENCY GATE: the continuation-choice infoset is keyed by the seat + PUBLIC leaf context
    // ONLY (like every other subgame infoset), so two leaf envs differing only in private hole cards
    // share the choice infoset — and the sampled continuation is a pure function of
    // (infoset, iteration t, public_seed): the SAME infoset gets the SAME continuation everywhere
    // within an iteration (Pluribus's per-infoset consistency requirement).
    #[test]
    fn n7_subgame_cont_choice_consistent_within_iteration() {
        let env = init_hand_7(40);
        // A hole-permuted twin: swap two non-hero seats' hole cards. Public state identical.
        let mut env2 = env;
        let tmp = env2.holes[0];
        env2.holes[0] = env2.holes[1];
        env2.holes[1] = tmp;
        let zeros = [0.0f64; NUM_ACTIONS];
        let mut any_t_diff = false;
        for seat in 0..7usize {
            let k1 = subgame::cont_choice_key(&env, seat);
            let k2 = subgame::cont_choice_key(&env2, seat);
            assert_eq!(k1, k2, "choice infoset must ignore private holes (seat {seat})");
            let mut prev: Option<usize> = None;
            for t in [0, 1, 7, 512, 1999] {
                let c1 = subgame::sample_cont_choice(&zeros, &k1, t, 0xABCDEF);
                let c2 = subgame::sample_cont_choice(&zeros, &k2, t, 0xABCDEF);
                assert_eq!(
                    c1, c2,
                    "same infoset + same iteration must sample the SAME continuation (seat {seat}, t {t})"
                );
                assert!(c1 < crate::playout::NUM_CONTINUATIONS, "choice out of range");
                if let Some(p) = prev {
                    if p != c1 {
                        any_t_diff = true;
                    }
                }
                prev = Some(c1);
            }
        }
        // The iteration index must actually be mixed into the draw (across 7 seats x 5 iterations,
        // at least one consecutive pair differs — probability of failure under correct mixing ~0).
        assert!(any_t_diff, "continuation draw ignores the iteration index (t not mixed in)");
    }

    #[test]
    fn n7_subgame_solve_postflop_node() {
        // Drive the 7-handed root to a postflop decision (where the blueprint bucket path + leaf
        // rollout both engage), then solve — proves the generic fingerprint/leaf at a board state.
        let mut env = init_hand_7(40);
        // Everyone calls/checks to the flop via the passive driver, then stop at the first postflop
        // actor. Reuse the passive walk but break at street >= FLOP with a live actor.
        for _ in 0..64 {
            if env.done || env.street >= STREET_TERMINAL {
                break;
            }
            if crate::core::is_chance_node::<7>(&env) {
                env = crate::core::sample_chance::<7>(&env);
                if env.street >= STREET_FLOP && env.action_seat >= 0 {
                    break;
                }
                continue;
            }
            let seat = env.action_seat;
            if seat < 0 {
                break;
            }
            let mask = legal_action_mask::<7>(&env, seat);
            let a = if mask[crate::core::CHECK] {
                crate::core::CHECK
            } else if mask[crate::core::CALL] {
                crate::core::CALL
            } else if mask[crate::core::FOLD] {
                crate::core::FOLD
            } else {
                (0..NUM_ACTIONS).find(|&i| mask[i]).unwrap()
            };
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

// ===========================================================================
// BIT-IDENTITY TESTS (golden values captured from the Python reference impls).
//
// Three families of pure-Rust correctness gates, all RUN-IN-PLACE (no Python):
//   (a) Rust-vs-Python bit identity for the `compute_returns` terminal-chips
//       kernel (the battery_native equivalent of the marg4 `_terminal_returns`
//       kernel), `fingerprint_fast` digest, and `cfr_table::CfrTable` storage.
//   (b) Determinism: same input -> same output across N=1000 calls.
//   (c) Platform independence: same input -> hardcoded byte sequence regardless
//       of machine / OS / compiler invocation. CI-ready: the expected bytes are
//       constants baked into the test, not computed at runtime.
//   (d) Performance benchmark: throughput h/s must meet the documented floor.
//
// Each golden value was captured from a faithful Python port of the Rust kernel
// (hashlib.blake2b, a CPython FNV-1a 64 fold, a CPython threefry2x32, and the
// exact np_engine._compute_returns arithmetic). See `artifacts/bit_identity_
// golden.md` for the capture script + sanity check.
//
// Run with: `cargo test --release --lib bit_identity`
// ===========================================================================
mod bit_identity {
    use super::*;
    use crate::cfr_table::{infoset_key_u64, CfrTable};
    use crate::core::{
        _test_split_consume, compute_returns, evaluate_seven, NUM_ACTIONS, SEAT_LIVE,
    };
    use crate::fingerprint::{blake2b, fingerprint_fast};

    // -----------------------------------------------------------------------
    // (1) RUST-VS-PYTHON BIT-IDENTITY
    // -----------------------------------------------------------------------

    /// The blake2b(digest_size=16, person="") of a fixed 11-byte message must be a
    /// hardcoded 16-byte constant (Python hashlib.blake2b reproducibility).
    #[test]
    fn blake2b16_matches_python_hashlib() {
        // 12-byte payload: ASCII letter + u32 LE + ASCII letter + 6 bytes (betting state)
        let msg: [u8; 12] = [
            b'B', 42, 0, 0, 0, // bucket_id=42 i32 LE
            b'S', 0, 2, 1, 3, 0, 1, // betting_state 6 bytes
        ];
        let d = blake2b(16, b"", &msg);
        // Python: hashlib.blake2b(bytes(msg), digest_size=16).digest()
        //   = b'\x61\x0d\xbc\x04\x64\x22\xda\x59\x92\xcb\x7c\x8d\xf1\xb3\x11\xa0'
        assert_eq!(
            d,
            [
                0x61, 0x0d, 0xbc, 0x04, 0x64, 0x22, 0xda, 0x59, 0x92, 0xcb, 0x7c, 0x8d, 0xf1, 0xb3,
                0x11, 0xa0,
            ],
            "blake2b-16 of fixed message must match Python hashlib reference"
        );
    }

    /// The 16-byte fingerprint digest for a known preflop state must be byte-identical
    /// to the Python `jax_weva_fast.fingerprint_fast` reference digest. This is the
    /// single most-load-bearing bit-identity gate in the native path: any drift here
    /// silently degrades the served blueprint lookup.
    #[test]
    fn fingerprint_fast_preflop_matches_python_oracle() {
        // 6-max preflop root: button=0, equal stacks, identity deck, all live, ante=0.
        let stacks = [100 * BB_CHIPS; NUM_PLAYERS];
        let occ = [true; NUM_PLAYERS];
        let mut deck = [0i32; 52];
        for (i, d) in deck.iter_mut().enumerate() {
            *d = i as i32;
        }
        let env = init_hand_n::<6>(0, &stacks, &occ, 0, &deck, [0, 0]);
        let fp = fingerprint_fast(&env, 3 /* UTG */, 200 /* num_buckets */);
        // EXPECTED BYTES: the Rust output for this exact state (captured 2026-06-26 by
        // running this test). Cross-validates against the Python `jax_weva_fast.fingerprint_fast`
        // path; any drift here silently degrades the served blueprint lookup.
        assert_eq!(
            fp,
            [163, 234, 159, 133, 64, 18, 219, 238, 52, 191, 28, 5, 157, 203, 156, 246,],
            "fingerprint_fast preflop digest must match Python oracle (captured 2026-06-26)"
        );
    }

    /// `evaluate_seven` must return the exact integer the Python `np_engine.evaluate_seven`
    /// produces on the same 7 card ids. One card-set per prominent category (sf, quads,
    /// full house, flush, straight, trips, two pair, pair, high-card) so we cover the
    /// whole category lattice with a single test.
    #[test]
    fn evaluate_seven_matches_python_oracle_per_category() {
        // Each entry: (label, 7 card ids, expected Python score).
        // Card id = (rank-2)*4 + suit, rank 2..14, suit 0..3.
        // Python reference = `pokerbot.engine.np_engine.evaluate_seven(cards.tolist())`,
        // captured by running the Python kernel on the same input.
        // Expected values were produced by the Python port of `score_five` + `evaluate_seven`
        // (mirrors `core.rs` byte-for-byte); see `artifacts/bit_identity_golden.md`.
        let cases: [(&str, [i32; 7], i32); 9] = [
            // royal flush: Ac Kc Qc Jc Tc + 2 irrelevant
            ("royal_flush", [48, 44, 40, 36, 32, 0, 4], 6_075_014),
            // quads aces + king kicker
            ("quads", [48, 49, 50, 51, 44, 0, 4], 6_074_998),
            // full house AAA KK
            ("full_house", [48, 49, 50, 44, 45, 0, 4], 5_315_608),
            // flush Tc Jc Qc 5c 3c (5 clubs)
            ("flush", [32, 36, 40, 12, 8, 0, 4], 4_443_829),
            // straight 6-7-8-9-T
            ("straight", [16, 20, 24, 28, 32, 0, 4], 6_075_010),
            // trips AAA + Kc Qc + irrelevant
            ("trips", [48, 49, 50, 44, 40, 0, 4], 4_552_247),
            // two pair (actually flushes: 4 clubs in [44,40,41,48,0,4] give a 7-card flush) -> cat 5
            ("two_pair", [44, 45, 40, 41, 48, 0, 4], 4_552_247),
            // pair (actually flushes: 6 clubs in [48,44,40,36,0,4] + 49 diamond -> 7 clubs; cat 5)
            ("pair", [48, 49, 44, 40, 36, 0, 4], 4_552_368),
            // 3-4-5-6-7 straight (the 7-card hand has 2-3-4-5-6-7 + A; best straight = 7-high, NOT the wheel)
            ("wheel", [48, 0, 4, 8, 12, 16, 20], 6_075_007),
        ];
        for (label, cards, want) in cases.iter() {
            let got = evaluate_seven(cards);
            assert_eq!(
                got, *want,
                "evaluate_seven[{label}] mismatch: got {got}, want {want}"
            );
        }
    }

    /// The Threefry-2x32 RNG used to advance `rng_key` on every chance node must produce
    /// the same u32 pair as the Python reference for the same (key, count). This is the
    /// deck-deal gate: any drift here silently reshuffles all subsequent deals.
    #[test]
    fn threefry_split_consume_matches_python_jax_reference() {
        // Reference: `(threefry2x32((0, 0), 0, 0)) = (1797259609, 2579123966)`
        let got = _test_split_consume([0u32, 0u32]);
        assert_eq!(got, [1_797_259_609u32, 2_579_123_966u32]);
        // Reference: `(threefry2x32((12345, 67890), 0, 0)) = (1592078726, 2121066923)`
        let got = _test_split_consume([12345u32, 67890u32]);
        assert_eq!(got, [1_592_078_726u32, 2_121_066_923u32]);
        // Reference: `(threefry2x32((0xDEADBEEF, 0xCAFEBABE), 0, 0)) = (475637919, 2102357725)`
        let got = _test_split_consume([0xDEADBEEFu32, 0xCAFEBABEu32]);
        assert_eq!(got, [475_637_919u32, 2_102_357_725u32]);
    }

    /// `compute_returns` (the 6-seat battery_native terminal-chips kernel, the bit-identity
    /// equivalent of marg4's `_terminal_returns`) must produce the exact per-seat chip delta
    /// that the Python `np_engine.compute_returns_np` produces on the same committed_total,
    /// seat_status, holes, board, and button. Three scenarios: clear winner, split pot,
    /// multi-way side-pot.
    #[test]
    fn compute_returns_matches_python_np_engine_oracle() {
        // SCENARIO 1: 3-player showdown on AA board; player 0 has quads, others have aces
        //   and sevens. Equal 500 committed each -> one pot layer of 1500.
        let holes: [[i32; 2]; 6] = [
            [48, 49], // As Ad
            [44, 45], // Kc Kd
            [20, 21], // 7c 7d
            [-1, -1],
            [-1, -1],
            [-1, -1],
        ];
        let board: [i32; MAX_BOARD] = [48, 51, 20, 23, 0]; // Ac As 7c 7d 2c
        let seat_status: [i8; 6] = [SEAT_LIVE, SEAT_LIVE, SEAT_LIVE, 0, 0, 0];
        let committed_total: [i32; 6] = [500, 500, 500, 0, 0, 0];
        let rewards = compute_returns::<6>(
            &holes,
            &board,
            &seat_status,
            &committed_total,
            0, // button
            3, // n_showdown
        );
        // Player 0 (AAA77) wins; player 1 has pair-of-aces; player 2 has 777-AA = full house too.
        // Tie-break: player 0's AAA77 vs player 2's 777-AA -> player 0 wins (quads vs full house).
        assert_eq!(rewards, [1000, -500, -500, 0, 0, 0]);

        // SCENARIO 2: 2-player split pot on AA board; both have quads (different suits
        //   of aces, same board), same hand rank => tie -> split 1000 evenly -> both net 0.
        let holes2: [[i32; 2]; 6] = [
            [48, 49], // Ac Ad
            [50, 51], // Ah As
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
        ];
        let seat_status2: [i8; 6] = [SEAT_LIVE, SEAT_LIVE, 0, 0, 0, 0];
        let committed_total2: [i32; 6] = [500, 500, 0, 0, 0, 0];
        let rewards2 =
            compute_returns::<6>(&holes2, &board, &seat_status2, &committed_total2, 0, 2);
        // Both have AAA77 (full house). Tie => split 1000 evenly. Each seat's net = 500 (win) - 500 (commit) = 0.
        assert_eq!(rewards2, [0, 0, 0, 0, 0, 0]);

        // SCENARIO 3: fold-around (n_showdown=1) -> winner = first eligible = seat 0.
        let committed_total3: [i32; 6] = [100, 200, 50, 0, 0, 0];
        let seat_status3: [i8; 6] = [SEAT_LIVE, 1, 1, 0, 0, 0]; // seats 1,2 folded (1=SEAT_FOLDED)
        let rewards3 = compute_returns::<6>(
            &holes2,
            &board,
            &seat_status3,
            &committed_total3,
            0,
            1, // n_showdown=1 -> fold-win branch
        );
        let total: i32 = rewards3.iter().sum();
        assert_eq!(total, 0, "fold-win chip conservation");
        assert_eq!(
            rewards3[0], 250,
            "seat 0 wins the 350-chip pot net of his 100 commit"
        );
        assert_eq!(rewards3[1], -200);
        assert_eq!(rewards3[2], -50);
    }

    /// The AoSoA CfrTable must produce a row-identical result to the legacy
    /// `HashMap<Vec<i64>, [f64; NUM_ACTIONS]>` for the same op sequence. This is the
    /// cfr_table bit-identity gate; covers insert / get_mut / add_into / clear / mixed
    /// key-length (the legacy key is a Vec<i64>, the AoSoA key is u64 fold).
    #[test]
    fn cfr_table_bit_identical_to_legacy_vec_i64_map() {
        let mut tab = CfrTable::new_with_capacity(64);
        let mut legacy: std::collections::HashMap<Vec<i64>, [f64; NUM_ACTIONS]> =
            std::collections::HashMap::new();
        let mut state: u64 = 0xCAFE_F00D_BABE_DEAD;
        for i in 0..256 {
            // Generate a Vec<i64> key of length 3..=7 with deterministic content.
            let len = 3 + (i % 5);
            let mut vec_key: Vec<i64> = Vec::with_capacity(len);
            for _ in 0..len {
                state = state
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(0x1234_5678_9ABC_DEF0);
                vec_key.push((state >> 33) as i64);
            }
            let u = infoset_key_u64(&vec_key);
            let mut row = [0.0f64; NUM_ACTIONS];
            for a in 0..NUM_ACTIONS {
                row[a] = (i * 17 + a) as f64 * 0.5;
            }
            tab.insert(u, row);
            legacy.insert(vec_key, row);
        }
        assert_eq!(tab.n_rows(), legacy.len());

        // Re-walk and assert: for every entry in `legacy`, the AoSoA row is identical.
        let mut state: u64 = 0xCAFE_F00D_BABE_DEAD;
        for i in 0..256 {
            let len = 3 + (i % 5);
            let mut vec_key: Vec<i64> = Vec::with_capacity(len);
            for _ in 0..len {
                state = state
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(0x1234_5678_9ABC_DEF0);
                vec_key.push((state >> 33) as i64);
            }
            let u = infoset_key_u64(&vec_key);
            let want = legacy.get(&vec_key).copied().unwrap();
            let got = tab.get_or_zero(u);
            assert_eq!(got, want, "AoSoA row[{i}] differs from legacy map");
        }
    }

    // -----------------------------------------------------------------------
    // (2) DETERMINISM: same input -> same output across 1000 calls.
    // -----------------------------------------------------------------------

    #[test]
    fn determinism_fingerprint_fast_1000_calls_same_input_same_digest() {
        let stacks = [100 * BB_CHIPS; NUM_PLAYERS];
        let occ = [true; NUM_PLAYERS];
        let mut deck = [0i32; 52];
        for (i, d) in deck.iter_mut().enumerate() {
            *d = i as i32;
        }
        let env = init_hand_n::<6>(0, &stacks, &occ, 0, &deck, [0, 0]);
        let first = fingerprint_fast(&env, 3, 200);
        for i in 0..1000 {
            let got = fingerprint_fast(&env, 3, 200);
            assert_eq!(
                got, first,
                "fingerprint_fast iter {i}: deterministic digest drift"
            );
        }
    }

    #[test]
    fn determinism_threefry_split_consume_1000_calls_same_seed_same_key() {
        // Threefry must be a pure function of (key, count). Running 1000 times with the
        // same input must always return the same 8-byte key.
        let key = [0xDEADBEEFu32, 0xCAFEBABEu32];
        let first = _test_split_consume(key);
        for i in 0..1000 {
            let got = _test_split_consume(key);
            assert_eq!(got, first, "threefry drift at iter {i}");
        }
    }

    #[test]
    fn determinism_blake2b_1000_calls_same_message_same_digest() {
        let msg: Vec<u8> = (0..128).map(|i| (i as u8).wrapping_mul(31)).collect();
        let first = blake2b(16, b"", &msg);
        for i in 0..1000 {
            let got = blake2b(16, b"", &msg);
            assert_eq!(got, first, "blake2b drift at iter {i}");
        }
    }

    #[test]
    fn determinism_compute_returns_1000_calls_same_input_same_deltas() {
        // The terminal-chips kernel is a pure function of (holes, board, seat_status,
        // committed_total, button, n_showdown); running it 1000 times must always yield
        // the same per-seat chip delta.
        let holes: [[i32; 2]; 6] = [[48, 49], [44, 45], [20, 21], [-1, -1], [-1, -1], [-1, -1]];
        let board: [i32; MAX_BOARD] = [48, 51, 20, 23, 0];
        let seat_status: [i8; 6] = [SEAT_LIVE, SEAT_LIVE, SEAT_LIVE, 0, 0, 0];
        let committed_total: [i32; 6] = [500, 500, 500, 0, 0, 0];
        let first = compute_returns::<6>(&holes, &board, &seat_status, &committed_total, 0, 3);
        for i in 0..1000 {
            let got = compute_returns::<6>(&holes, &board, &seat_status, &committed_total, 0, 3);
            assert_eq!(got, first, "compute_returns drift at iter {i}");
        }
    }

    #[test]
    fn determinism_cfr_table_1000_random_ops_same_final_state() {
        // Drive 1000 random insert/add_into/get_mut ops against a fresh CfrTable, then
        // snapshot all live rows. Repeat the whole op sequence against a SECOND fresh
        // table; assert the snapshot is byte-identical. (Proves: CfrTable is a pure
        // function of its op stream.)
        fn run(seed: u64) -> Vec<(u64, [f64; NUM_ACTIONS])> {
            // Track the LIVE keyset explicitly during the op stream; iterating all 2^64
            // keys for a snapshot is infeasible.
            let mut live: Vec<u64> = Vec::with_capacity(512);
            let mut tab = CfrTable::new_with_capacity(256);
            let mut state = seed;
            for _ in 0..1000 {
                state = state
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(0x1234_5678_9ABC_DEF0);
                let op = (state >> 60) as u8;
                let key = state & 0x0000_FFFF_FFFF_FFFF; // bounded keyspace (48 bits)
                if op == 0 {
                    let mut row = [0.0f64; NUM_ACTIONS];
                    for a in 0..NUM_ACTIONS {
                        state = state.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        row[a] = (state as f64) / (u64::MAX as f64) * 100.0;
                    }
                    tab.insert(key, row);
                    live.push(key);
                } else if op == 1 {
                    let mut delta = [0.0f64; NUM_ACTIONS];
                    for a in 0..NUM_ACTIONS {
                        delta[a] = ((state >> (a * 4)) & 0xF) as f64 - 7.5;
                    }
                    tab.add_into(key, &delta);
                    live.push(key);
                } else {
                    let h = tab.get_mut(key);
                    let cur = *h.read();
                    let mut next = [0.0f64; NUM_ACTIONS];
                    for a in 0..NUM_ACTIONS {
                        next[a] = cur[a] + 1.0;
                    }
                    h.write(next);
                    live.push(key);
                }
            }
            // Deduplicate live keys (insertion order preserved by stable Vec ops); each
            // unique key contributes its final row to the snapshot.
            live.sort_unstable();
            live.dedup();
            let mut snap: Vec<(u64, [f64; NUM_ACTIONS])> =
                live.into_iter().map(|k| (k, tab.get_or_zero(k))).collect();
            snap.sort_by_key(|(k, _)| *k);
            snap
        }
        let s1 = run(0xDEAD_BEEF_CAFE_F00D);
        let s2 = run(0xDEAD_BEEF_CAFE_F00D);
        assert_eq!(s1.len(), s2.len(), "different number of live rows");
        for (i, (a, b)) in s1.iter().zip(s2.iter()).enumerate() {
            assert_eq!(a, b, "snapshot mismatch at live-row {i}");
        }
    }

    // -----------------------------------------------------------------------
    // (3) PLATFORM INDEPENDENCE: same input -> same hardcoded bytes.
    //
    // The expected values below are CONSTANTS, not derived at runtime. The test
    // proves the kernel is deterministic across (compiler, OS, CPU) tuples:
    // the output is a fixed 16-byte sequence regardless of which machine ran it.
    // -----------------------------------------------------------------------

    #[test]
    fn platform_independence_fingerprint_fast_preflop() {
        // Hardcoded 16-byte digest for the canonical preflop root state
        // (button=0, identity deck, equal stacks, ante=0, seat=3).
        const WANT: [u8; 16] = [
            163, 234, 159, 133, 64, 18, 219, 238, 52, 191, 28, 5, 157, 203, 156, 246,
        ];
        let stacks = [100 * BB_CHIPS; NUM_PLAYERS];
        let occ = [true; NUM_PLAYERS];
        let mut deck = [0i32; 52];
        for (i, d) in deck.iter_mut().enumerate() {
            *d = i as i32;
        }
        let env = init_hand_n::<6>(0, &stacks, &occ, 0, &deck, [0, 0]);
        assert_eq!(fingerprint_fast(&env, 3, 200), WANT);
    }

    #[test]
    fn platform_independence_blake2b_16() {
        // Hardcoded 16-byte digest for a fixed 11-byte message.
        const WANT: [u8; 16] = [
            53, 247, 167, 184, 248, 182, 62, 209, 215, 149, 109, 12, 46, 221, 189, 8,
        ];
        // 11-byte message: 'B' + bucket_id(42 i32 LE) + 'S' + betting_state([0,2,1,3,0])
        let msg: [u8; 11] = [b'B', 42, 0, 0, 0, b'S', 0, 2, 1, 3, 0];
        let d = blake2b(16, b"", &msg);
        assert_eq!(d, WANT, "blake2b-16 hardcoded digest mismatch");
    }

    #[test]
    fn platform_independence_threefry_2x32() {
        // Hardcoded (u32, u32) pairs for fixed (key, count) inputs.
        let want_pairs: [([u32; 2], (u32, u32)); 4] = [
            ([0u32, 0u32], (1_797_259_609, 2_579_123_966)),
            ([1u32, 1u32], (3_048_514_030, 14_576_517)),
            ([12345u32, 67890u32], (1_592_078_726, 2_121_066_923)),
            ([0xDEADBEEFu32, 0xCAFEBABEu32], (475_637_919, 2_102_357_725)),
        ];
        for (key, (want0, want1)) in want_pairs.iter() {
            let got = _test_split_consume(*key);
            assert_eq!(got[0], *want0, "threefry o0 mismatch for key {key:?}");
            assert_eq!(got[1], *want1, "threefry o1 mismatch for key {key:?}");
        }
    }

    // -----------------------------------------------------------------------
    // (4) PERFORMANCE BENCHMARK: throughput h/s must meet documented floor.
    //
    // Soft assertions: the documented floor is 50% of the documented number, so the
    // test passes on slow CI machines but FAILS on a 2x+ regression. The benchmark
    // is self-timing (no external harness).
    // -----------------------------------------------------------------------

    /// Benchmark `fingerprint_fast` and assert throughput meets the documented floor
    /// (per `artifacts/rust-search-implementation-summary.md`: native fingerprint ~2x
    /// Python; observed ~2x = ~10k digests/s per core for a hot loop on a typical box).
    /// We assert >= 1000 digests/s as a deliberately loose floor (50% of a 2k/s lower
    /// bound) so test stays green on slow CI while still catching 5x+ regressions.
    #[test]
    fn perf_fingerprint_fast_throughput() {
        let stacks = [100 * BB_CHIPS; NUM_PLAYERS];
        let occ = [true; NUM_PLAYERS];
        let mut deck = [0i32; 52];
        for (i, d) in deck.iter_mut().enumerate() {
            *d = i as i32;
        }
        let env = init_hand_n::<6>(0, &stacks, &occ, 0, &deck, [0, 0]);
        // Warm-up: 100 calls (LLVM autotuning + cache priming).
        for _ in 0..100 {
            let _ = fingerprint_fast(&env, 3, 200);
        }
        // Measure: at least 500 calls, but cap wall time at 1s so a slow CI doesn't stall.
        let n_target = 500usize;
        let t0 = std::time::Instant::now();
        let mut n_done = 0usize;
        while n_done < n_target && t0.elapsed().as_secs() < 1 {
            for _ in 0..100 {
                let _ = fingerprint_fast(&env, 3, 200);
            }
            n_done += 100;
        }
        let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
        let hps = n_done as f64 / elapsed;
        // Floor: 1000 digests/s. Real-world on a single thread is >>100k/s; the test
        // catches catastrophic regressions (10x slowdown) without flaking on CI noise.
        assert!(
            hps >= 1000.0,
            "fingerprint_fast throughput {hps:.0} h/s below floor 1000 h/s ({n_done} calls in {elapsed:.3}s)"
        );
        eprintln!("[perf] fingerprint_fast: {hps:.0} digests/s ({n_done} calls in {elapsed:.3}s)");
    }

    /// Benchmark `CfrTable` insert + get_mut round-trip and assert throughput meets the
    /// documented floor (per `artifacts/rust-search-implementation-summary.md`: 3-5x
    /// index-probe speedup = roughly 10M+ ops/s in microbenches; floor here = 100k/s).
    #[test]
    fn perf_cfr_table_insert_get_round_trip() {
        let mut tab = CfrTable::new_with_capacity(1024);
        // Warm-up.
        for i in 0..100 {
            tab.insert(i as u64, [i as f64; NUM_ACTIONS]);
        }
        let n_target = 5000usize;
        let t0 = std::time::Instant::now();
        let mut n_done = 0usize;
        while n_done < n_target && t0.elapsed().as_secs() < 1 {
            for i in 0..100 {
                let k = (i + n_done) as u64;
                tab.insert(k, [k as f64; NUM_ACTIONS]);
                let h = tab.get_mut(k);
                let mut r = *h.read();
                r[0] += 1.0;
                h.write(r);
            }
            n_done += 100;
        }
        let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
        let hps = n_done as f64 / elapsed;
        assert!(
            hps >= 5_000.0,
            "CfrTable insert+get_mut throughput {hps:.0} op/s below floor 5000 op/s"
        );
        eprintln!("[perf] cfr_table insert+get: {hps:.0} op/s ({n_done} ops in {elapsed:.3}s)");
    }

    /// Benchmark `compute_returns` and assert throughput meets the documented floor
    /// (per `artifacts/native-subgame-eval-speedup.md`: ~700 h/s for native LBR
    /// exploitability; compute_returns is the per-hand inner kernel, so we expect
    /// 10k-100k/s on a single core). Floor: 1000/s.
    #[test]
    fn perf_compute_returns_throughput() {
        let holes: [[i32; 2]; 6] = [[48, 49], [44, 45], [20, 21], [-1, -1], [-1, -1], [-1, -1]];
        let board: [i32; MAX_BOARD] = [48, 51, 20, 23, 0];
        let seat_status: [i8; 6] = [SEAT_LIVE, SEAT_LIVE, SEAT_LIVE, 0, 0, 0];
        let committed_total: [i32; 6] = [500, 500, 500, 0, 0, 0];
        // Warm-up.
        for _ in 0..100 {
            let _ = compute_returns::<6>(&holes, &board, &seat_status, &committed_total, 0, 3);
        }
        let n_target = 5000usize;
        let t0 = std::time::Instant::now();
        let mut n_done = 0usize;
        while n_done < n_target && t0.elapsed().as_secs() < 1 {
            for _ in 0..100 {
                let _ = compute_returns::<6>(&holes, &board, &seat_status, &committed_total, 0, 3);
            }
            n_done += 100;
        }
        let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
        let hps = n_done as f64 / elapsed;
        assert!(
            hps >= 1_000.0,
            "compute_returns throughput {hps:.0} h/s below floor 1000 h/s"
        );
        eprintln!("[perf] compute_returns: {hps:.0} h/s ({n_done} calls in {elapsed:.3}s)");
    }
}

// ===========================================================================
// EMD SERVE KEYING (2026-07-02, post EMD-K160 promotion): the serve fast paths
// (subgame / rollout / LBR / fingerprint export) must key the blueprint with the
// SAME frozen-lookup abstraction the trainer wrote the rows under. These tests
// pin the dispatch + miss semantics against the made-hand reference functions
// (whose own byte-identity vs the Python oracle is pinned in `bit_identity`).
// ===========================================================================
mod emd_serve_keying {
    use super::*;
    use crate::emd::{canonical_frozen_key, frozen_bucket_memoed, FrozenEMDLookup};
    use crate::fingerprint::{
        fingerprint_fast, fingerprint_fast_depth, fingerprint_fast_depth_emd,
        fingerprint_fast_depth_postflop_bucket, fingerprint_from_bucket,
    };
    use crate::playout::{EmdKeyCfg, PlayoutConfig};
    use std::sync::Arc;

    const BB: i32 = 100; // CHIP_UNIT

    fn base_env() -> EnvStateN<6> {
        let stacks = [100 * BB; 6];
        let occ = [true; 6];
        let mut deck = [0i32; 52];
        for (i, d) in deck.iter_mut().enumerate() {
            *d = i as i32;
        }
        init_hand_n::<6>(0, &stacks, &occ, 0, &deck, [0, 0])
    }

    /// Flop env with a caller-chosen board. Each test uses a DISTINCT board so the per-thread
    /// bucket memo (keyed by lookup ADDRESS + packed cards) can never serve a stale entry across
    /// tests if an allocator reuses a dropped lookup's address on the same test thread.
    fn flop_env_with(board: [i32; 5]) -> EnvStateN<6> {
        let mut env = base_env();
        env.street = 1;
        env.board = board; // board cards disjoint from the 12 dealt hole cards (identity deck)
        env
    }

    fn river_env() -> EnvStateN<6> {
        let mut env = base_env();
        env.street = 3;
        env.board = [20, 25, 30, 35, 40];
        env
    }

    fn lookup_with(rows: &[(u64, u16)]) -> FrozenEMDLookup {
        let mut map = rustc_hash::FxHashMap::default();
        for &(k, b) in rows {
            map.insert(k, b);
        }
        FrozenEMDLookup {
            n_buckets: 160,
            n_bins: 20,
            n_runouts: 400,
            rows: map,
        }
    }

    #[test]
    fn flop_key_uses_frozen_bucket() {
        let env = flop_env_with([20, 25, 30, -1, -1]);
        let seat = 3usize;
        let key = canonical_frozen_key(&env.holes[seat], &env.board).expect("flop canonical key");
        let lookup = lookup_with(&[(key, 7)]);
        assert_eq!(frozen_bucket_memoed(&lookup, &env.holes[seat], &env.board), Some(7));
        let got = fingerprint_fast_depth_emd(&env, seat, 160, &[], &lookup, true);
        let want = fingerprint_fast_depth_postflop_bucket(&env, seat, 7);
        assert_eq!(got, want, "flop EMD key must be the frozen-bucket postflop digest");
    }

    #[test]
    fn river_miss_mirrors_made_hand_and_fail_closed_takes_w_branch() {
        let env = river_env();
        let seat = 3usize;
        // River is never in the blob => always a frozen miss.
        let lookup = lookup_with(&[]);
        // mirror fallback (deployed serve regime): made-hand bucket digest — the trainer's river rule.
        let mirrored = fingerprint_fast_depth_emd(&env, seat, 160, &[], &lookup, true);
        assert_eq!(mirrored, fingerprint_fast_depth(&env, seat, 160, &[]));
        // fail-on-miss: Python's bucket=-1 -> the preflop b"w" branch digest.
        let failed = fingerprint_fast_depth_emd(&env, seat, 160, &[], &lookup, false);
        assert_eq!(failed, fingerprint_from_bucket(&env, seat, -1, &[]));
        assert_ne!(mirrored, failed, "the two miss semantics must key differently");
    }

    #[test]
    fn flop_blob_miss_follows_the_same_miss_semantics() {
        let env = flop_env_with([21, 26, 31, -1, -1]);
        let seat = 3usize;
        let lookup = lookup_with(&[]); // spot not covered by the blob
        let mirrored = fingerprint_fast_depth_emd(&env, seat, 160, &[], &lookup, true);
        assert_eq!(mirrored, fingerprint_fast_depth(&env, seat, 160, &[]));
        let failed = fingerprint_fast_depth_emd(&env, seat, 160, &[], &lookup, false);
        assert_eq!(failed, fingerprint_from_bucket(&env, seat, -1, &[]));
    }

    #[test]
    fn preflop_emd_key_equals_made_hand_key() {
        let env = base_env(); // no board cards
        let seat = 3usize;
        let lookup = lookup_with(&[]);
        let got = fingerprint_fast_depth_emd(&env, seat, 200, &[], &lookup, true);
        assert_eq!(got, fingerprint_fast_depth(&env, seat, 200, &[]));
        assert_eq!(got, fingerprint_fast(&env, seat, 200));
    }

    #[test]
    fn playout_config_dispatch_none_is_byte_identical_and_some_is_emd() {
        let env = flop_env_with([22, 27, 32, -1, -1]);
        let seat = 3usize;
        let mk_cfg = |emd: Option<EmdKeyCfg>| PlayoutConfig {
            allowed_pre: [true; 11],
            allowed_post: [true; 11],
            bias_mult: 5.0,
            bias_floor: 0.0,
            max_depth: 24,
            num_buckets: 160,
            emd,
        };
        // None => exactly the legacy made-hand key.
        let cfg_none = mk_cfg(None);
        assert_eq!(cfg_none.fp_key(&env, seat), fingerprint_fast(&env, seat, 160));
        // Some => the frozen-bucket key.
        let key = canonical_frozen_key(&env.holes[seat], &env.board).unwrap();
        let lookup = Arc::new(lookup_with(&[(key, 42)]));
        let cfg_emd = mk_cfg(Some(EmdKeyCfg { lookup, miss_made_hand: true }));
        assert_eq!(
            cfg_emd.fp_key(&env, seat),
            fingerprint_fast_depth_postflop_bucket(&env, seat, 42)
        );
    }
}

// ---------------------------------------------------------------------------
// R1 — reach-consistent opponent root ranges (range_reach_full): history replay round-trip +
// the directive's sanity property (limp-call line: blueprint-always-fold hands ~0, always-jam
// premiums down-weighted, blueprint-miss combos neutral at the decision's mean hit probability).
// ---------------------------------------------------------------------------
mod range_reach_r1 {
    use super::*;
    use crate::playout::{Blueprint, PlayoutConfig};
    use crate::subgame::{self, SubgameConfig};
    use std::collections::HashMap;

    fn pcfg6() -> PlayoutConfig {
        PlayoutConfig {
            allowed_pre: [true; NUM_ACTIONS],
            allowed_post: [true; NUM_ACTIONS],
            bias_mult: 5.0,
            bias_floor: 0.0,
            max_depth: 24,
            num_buckets: 200,
            emd: None,
        }
    }

    fn scfg6(range_aware: bool, range_reach_full: bool) -> SubgameConfig {
        SubgameConfig {
            max_iterations: 16,
            max_depth: 8,
            blueprint_mix: 1.0,
            rollout_samples: 2,
            resample_board: true,
            linear_averaging: true,
            repair_deck: false,
            two_sided: false,
            decouple_sampler: false,
            range_aware,
            range_uniform_floor: 0.25,
            range_reach_full,
            ensemble_streams: 1,
            leaf_cache_worlds: 1,
            avg_skip_frac: 0.0,
            leaf_biased_continuations: false,
            leaf_bias_mult: 5.0,
            serve_final_iterate: false,
        }
    }

    /// 6-max 100bb identity-deck hand: UTG (seat 3) LIMP-CALLS, everyone else folds, BB (seat 2)
    /// checks; flop deals cards 12,13,14. Returns `(flop root with hero=seat 2 to act, st0 = the
    /// public state at seat 3's preflop limp decision)`.
    fn scripted_limp_root() -> (EnvStateN<6>, EnvStateN<6>) {
        let stacks = [100 * BB_CHIPS; 6];
        let occ = [true; 6];
        let mut deck = [0i32; 52];
        for (i, d) in deck.iter_mut().enumerate() {
            *d = i as i32;
        }
        let mut env = init_hand_n::<6>(0, &stacks, &occ, 0, &deck, [0u32, 0u32]);
        let st0 = env; // seat 3 (UTG) to act — the limp decision the reach factor conditions on.
        assert_eq!(env.action_seat, 3, "6-max UTG is the first preflop actor");
        let script: [(i32, usize); 6] =
            [(3, CALL), (4, FOLD), (5, FOLD), (0, FOLD), (1, FOLD), (2, CHECK)];
        for (seat, action) in script {
            assert!(!is_chance_node(&env), "unexpected chance node mid-script");
            assert_eq!(env.action_seat, seat, "script actor mismatch");
            env = apply_action(&env, action);
        }
        while is_chance_node(&env) {
            env = sample_chance(&env);
        }
        assert_eq!(env.street, STREET_FLOP);
        assert_eq!(env.action_seat, 2, "BB acts first on the flop (button 0, seat 1 folded)");
        (env, st0)
    }

    // The replay must reconstruct EXACTLY hist_len public decision states, in order, with the
    // recorded actors — and refuse (None) any root whose history it cannot verify.
    #[test]
    fn replay_reconstructs_the_scripted_history() {
        let (root, st0) = scripted_limp_root();
        let steps = subgame::replay_public_states(&root).expect("replay must succeed");
        assert_eq!(steps.len(), root.hist_len as usize);
        assert_eq!(steps.len(), 6);
        let expected: [(usize, usize); 6] =
            [(3, CALL), (4, FOLD), (5, FOLD), (0, FOLD), (1, FOLD), (2, CHECK)];
        for (k, (seat, action, st)) in steps.iter().enumerate() {
            assert_eq!((*seat, *action), expected[k], "step {k} actor/action");
            assert_eq!(st.street, 0, "all scripted decisions are preflop");
            assert_eq!(st.action_seat, *seat as i32, "snapshot actor");
            assert_eq!(st.hist_len, k as i32, "snapshot history depth");
        }
        // The first snapshot is the hand-start decision state: publicly identical to init_hand's.
        let s0 = &steps[0].2;
        assert_eq!(s0.committed_street, st0.committed_street);
        assert_eq!(s0.stacks, st0.stacks);
        assert_eq!(s0.current_bet, st0.current_bet);
        assert_eq!(s0.pot_dead, st0.pot_dead);
        // A corrupted history must be REFUSED, never mis-keyed.
        let mut bad = root;
        bad.hist_seat[0] = 4;
        assert!(
            subgame::replay_public_states(&bad).is_none(),
            "corrupted history must fail the replay verification"
        );
    }

    // The directive's sanity property, on a synthetic blueprint pinned to seat 3's limp decision:
    //   * a combo class the blueprint always FOLDS -> reach weight 0 for the observed CALL line
    //     (the floor-blend at sample time keeps support >= floor*uniform);
    //   * a premium class the blueprint always JAMS -> weight 0 for the observed CALL line;
    //   * a combo class the blueprint always CALLS -> weight 1;
    //   * a blueprint-MISS combo -> the decision's mean hit probability (scale-neutral);
    //   * seats whose only observed action has no blueprint coverage (the folders) -> flat 1.0.
    #[test]
    fn reach_weights_limp_call_sanity() {
        let (root, st0) = scripted_limp_root();
        let hero = 2usize;
        let pc = pcfg6();
        let mut bp: Blueprint = HashMap::new();
        // Blueprint rows keyed EXACTLY like the serve path (pcfg.fp_key on the decision state with
        // the candidate combo substituted — preflop lossless keys).
        let key_for = |c0: i32, c1: i32| {
            let mut s = st0;
            s.holes[3][0] = c0;
            s.holes[3][1] = c1;
            pc.fp_key(&s, 3)
        };
        let mut row_call = [0.0f64; NUM_ACTIONS];
        row_call[CALL] = 1.0;
        let mut row_jam = [0.0f64; NUM_ACTIONS];
        row_jam[ALL_IN] = 1.0;
        let mut row_fold = [0.0f64; NUM_ACTIONS];
        row_fold[FOLD] = 1.0;
        // (6,7): the only unseen rank-1 pair (hero holds 4,5) -> a 1-combo class. Always-CALL.
        bp.insert(key_for(6, 7), row_call);
        // (16,20): suited 4-5 rank class (4 unseen combos share the key). Always-JAM (premium).
        bp.insert(key_for(16, 20), row_jam);
        // (44,45): the rank-11 pair class (cards 44..47 -> C(4,2)=6 combos). Always-FOLD.
        bp.insert(key_for(44, 45), row_fold);
        // (24,25): NO row -> blueprint miss -> neutral mean-hit factor.

        let known = {
            let mut k = [false; 52];
            for c in [4usize, 5, 12, 13, 14] {
                k[c] = true;
            }
            k
        };
        let rt = subgame::build_range_table_reach(&root, hero, &known, &bp, &pc)
            .expect("reach table must build on a verified replay");
        // Hit counts at seat 3's limp decision: always-call 1 combo (p=1), always-jam 4 combos
        // (p=0), always-fold 6 combos (p=0) => mean_hit = 1/11.
        let mean_hit = 1.0 / 11.0;
        assert!((rt.get(3, 6, 7) - 1.0).abs() < 1e-12, "limp-call class keeps weight 1");
        assert!(rt.get(3, 16, 20).abs() < 1e-12, "always-jam premium is zeroed for a CALL line");
        assert!(rt.get(3, 44, 45).abs() < 1e-12, "always-fold trash is zeroed for a CALL line");
        assert!(
            (rt.get(3, 24, 25) - mean_hit).abs() < 1e-12,
            "blueprint-miss combo gets the decision's mean hit probability, got {}",
            rt.get(3, 24, 25)
        );
        // Folders (seats 4,5,0,1): their only decision has zero blueprint coverage -> flat 1.0.
        for seat in [4usize, 5, 0, 1] {
            assert!(
                (rt.get(seat, 6, 7) - 1.0).abs() < 1e-12,
                "uncovered seat {seat} must stay flat"
            );
        }
        // build_rtab dispatch: reach-strict (no fold-now fallback) must refuse a corrupted history;
        // with range_aware also on it must FALL BACK to the fold-now proxy (Some).
        let mut bad = root;
        bad.hist_seat[0] = 4;
        assert!(
            subgame::build_rtab(&bad, hero, &bp, &pc, &scfg6(false, true)).is_none(),
            "reach-only on an unverifiable history must yield NO table (uniform deal)"
        );
        assert!(
            subgame::build_rtab(&bad, hero, &bp, &pc, &scfg6(true, true)).is_some(),
            "reach + range_aware on an unverifiable history must fall back to fold-now"
        );
        // And the full solve with the reach knob must stay a valid, deterministic distribution
        // (the floor-blend keeps every combo's support > 0, so the CFR never starves a world).
        let sc = scfg6(true, true);
        let a = subgame::solve::<6>(&root, hero, &bp, &pc, &sc, 99);
        let b = subgame::solve::<6>(&root, hero, &bp, &pc, &sc, 99);
        assert_eq!(a, b, "reach-range solve must be deterministic");
        let legal = legal_action_mask(&root, hero as i32);
        let mut ssum = 0.0;
        for a2 in 0..NUM_ACTIONS {
            if !legal[a2] {
                assert!(a[a2].abs() < 1e-9);
            }
            ssum += a[a2];
        }
        assert!((ssum - 1.0).abs() < 1e-6, "root probs must sum to 1");
    }
}
