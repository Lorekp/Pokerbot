"""P0 of the deep-ICM evaluator (workflow wexmo5p7v): the play_hand -> deal_hole_cards()+advance_to_terminal
refactor. Guards (a) that play_hand still plays valid, zero-sum hands (the refactor preserved behavior --
the rigorous byte-identity guard is the existing game<->jax equivalence suite), and (b) the NEW capability:
advance_to_terminal RESUMES a mid-hand state that has already been dealt + partially played (which raw
play_hand could never do, because it re-deals the hole at the top).
"""

from __future__ import annotations

import copy
import random

from pokerbot.engine.game import Action, ActionType, GameState, advance_to_terminal, play_hand


def _passive_to_midhand(state: GameState, n_actions: int) -> None:
    """Apply up to n passive (check/call) legal actions to push a fresh dealt state into the hand."""
    for _ in range(n_actions):
        if state.is_terminal() or not state.actors_remaining():
            return
        seat = state.action_seat
        if state.players[seat].folded or state.players[seat].all_in:
            state.advance_to_next_actor()
            continue
        legal = state.legal_actions()
        act = next((a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)),
                   legal[0] if legal else Action(ActionType.FOLD))
        state.apply_action(act)
        if not state.is_terminal():
            state.advance_to_next_actor()


def test_play_hand_still_plays_valid_zero_sum_hands():
    for seed in range(50):
        rng = random.Random(seed)
        state, deltas = play_hand({}, rng=rng)          # default policy = check/call/fold
        assert state.is_terminal()
        assert abs(sum(deltas.values())) < 1e-6         # chip-conserving / zero-sum


def test_advance_to_terminal_resumes_a_cloned_midhand_node():
    # This is the whole point of the refactor: clone a mid-hand state and run it to terminal.
    resumed = 0
    for seed in range(60):
        rng = random.Random(1000 + seed)
        state = GameState.new_hand(rng=rng)
        state.deal_hole_cards()
        _passive_to_midhand(state, n_actions=2)
        if state.is_terminal():
            continue
        clone = copy.deepcopy(state)
        fin, deltas = advance_to_terminal(clone, {}, rng=random.Random(7))
        assert fin.is_terminal()
        assert abs(sum(deltas.values())) < 1e-6         # resumed hand still conserves chips
        resumed += 1
    assert resumed >= 30                                # most seeds reach a genuine mid-hand resume


def test_advance_equals_play_hand_on_a_freshly_dealt_state():
    # play_hand == deal_hole_cards() + advance_to_terminal: same deck (CRN) -> identical terminal deltas.
    for seed in range(40):
        a, da = play_hand({}, rng=random.Random(seed))
        s = GameState.new_hand(rng=random.Random(seed))
        s.deal_hole_cards()
        b, db = advance_to_terminal(s, {}, rng=random.Random(seed))
        assert da == db                                 # byte-identical terminal payoffs on the same deck
