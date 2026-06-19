"""Pre-smoke audit Tier 6 — eval-engine reopen/min-raise (SE4). Eval-only (the live serve path is JAX),
but the Python game.py simulator must still track the last FULL-raise increment so its min-raise and
reopen logic follow NLHE rules instead of the old current_bet*2 / reset-on-any-raise approximation."""

from __future__ import annotations

from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.game import GameState, PlayerState, Position, Street


def _three_handed():
    st = GameState()
    st.players = [PlayerState(seat=i, position=p, stack_bb=500.0)
                  for i, p in enumerate([Position.BTN, Position.SB, Position.BB])]
    st.button_seat = 0
    st.street = Street.PREFLOP
    st.current_bet_bb = 1.0
    st.players[1].committed_street_bb = 0.5      # SB
    st.players[2].committed_street_bb = 1.0      # BB
    st.pot_bb = 1.5
    return st


def test_tracks_last_full_raise_increment_and_reopens():
    st = _three_handed()
    # BTN raises to 3 (increment 2 above the 1bb bet) -> a full raise: resets acted, records increment.
    st.apply_action(Action(ActionType.RAISE, 2.0), 0)
    assert abs(st.current_bet_bb - 3.0) < 1e-9
    assert abs(st.last_raise_increment - 2.0) < 1e-9
    assert st._acted_this_street == {0}
    # SB re-raises to 9 (increment 6) -> a full raise: resets acted to {1}, increment becomes 6.
    st.apply_action(Action(ActionType.RAISE, 6.0), 1)
    assert abs(st.current_bet_bb - 9.0) < 1e-9
    assert abs(st.last_raise_increment - 6.0) < 1e-9
    assert st._acted_this_street == {1}


def test_min_raise_floor_filters_sub_minimum_raises():
    # After BTN opens to 3 (increment 2) into a SMALL pot, the SB's legal raises must all be >= the min
    # legal raise-to of 5 (= 3 + the last full increment 2). No offered raise commits to less than that.
    st = _three_handed()
    st.pot_bb = 4.0
    st.apply_action(Action(ActionType.RAISE, 2.0), 0)        # BTN to 3
    raises = [a for a in st.legal_actions(1) if a.action_type == ActionType.RAISE]
    for a in raises:
        raise_to = 0.5 + a.amount_bb     # SB committed 0.5 this street
        assert raise_to >= 5.0 - 1e-9    # never below current_bet(3) + last increment(2)
