"""B1 regression: the live serve path recomputes the action-history abstraction index via
``action_to_index``. Engine raises carry an ABSOLUTE-BB ``amount_bb``, but ``action_to_index`` buckets a
POT FRACTION, so the pre-fix code collapsed every raise to the top bucket -> the search/blueprint infoset
key (which reads ``hist_action``) missed -> uniform-prior fallback at any node with a prior bet/raise.

The fix tags ``Action.bucket`` with the exact abstraction menu index where it is known (engine
``legal_actions`` = the frac position; live replay = the bucket of bet-TO / pot-at-action), and
``action_to_index`` returns it when set. Training/encoding Actions never set ``bucket`` (stay -1), so that
path is byte-identical (covered by tests/test_action_translation.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from pokerbot.engine.actions import (
    Action,
    ActionType,
    NUM_ACTIONS,
    action_to_index,
    bucket_index_for_bet,
)
from pokerbot.engine.game import GameState
from pokerbot.search.state_bridge import game_state_to_env


def test_legacy_pot_fraction_path_unchanged_when_bucket_unset():
    # Training/encoding path (bucket=-1): pot-fraction nearest-by-distance, byte-identical to before.
    assert action_to_index(Action(ActionType.RAISE, 1.0)) == 3 + 4   # 1.0 == BET_SIZE_BUCKETS[4]
    # An ABSOLUTE-BB raise with no resolved bucket collapses to the top bet bucket — this IS the bug the
    # serve path used to hit (every multi-bb raise looked like a 2.0x pot bet).
    assert action_to_index(Action(ActionType.RAISE, 7.5)) == NUM_ACTIONS - 2
    # With a resolved bucket the serve path keys the correct one instead.
    assert action_to_index(Action(ActionType.RAISE, 7.5, bucket=4)) == 4


def test_bucket_index_for_bet_distinct_and_ordered():
    # bet-TO / pot-before maps to a sensible, NON-collapsed bucket.
    small = bucket_index_for_bet(1.0, 3.0)   # ~0.33 pot
    big = bucket_index_for_bet(6.0, 3.0)     # 2.0 pot -> top bet bucket
    assert 3 <= small < big <= NUM_ACTIONS - 2
    assert big == NUM_ACTIONS - 2            # >= 2.0x pot clamps to the largest bet bucket


def test_engine_legal_raises_carry_distinct_buckets():
    st = GameState.new_hand(button_seat=3)
    utg = st.action_seat
    raises = [a for a in st.legal_actions(utg) if a.action_type in (ActionType.BET, ActionType.RAISE)]
    assert len(raises) >= 2
    buckets = [a.bucket for a in raises]
    assert all(b >= 3 for b in buckets)            # all tagged
    assert len(set(buckets)) == len(buckets)       # distinct (one per surviving frac)


def test_round_trip_hist_action_not_collapsed():
    # Two different-sized raises must reconstruct to TWO DIFFERENT abstraction indices through the live
    # serve bridge (game_state_to_env). The pre-fix code mapped both to the top bucket.
    st = GameState.new_hand(button_seat=3)
    utg = st.action_seat
    raises = [a for a in st.legal_actions(utg) if a.action_type in (ActionType.BET, ActionType.RAISE)]
    small, big = raises[0], raises[-1]
    assert small.bucket != big.bucket
    st._history = [(utg, small), ((utg + 1) % 6, big)]
    env = game_state_to_env(st, hero_seat=st.button_seat)
    ha = [int(x) for x in np.asarray(env.hist_action)[:2]]
    assert ha == [small.bucket, big.bucket]
    assert ha[0] != ha[1]


def test_short_table_env_padded_to_six_seats():
    # A short-handed live GameState (fewer than 6 seated players) must convert to a uniformly 6-wide
    # EnvState (absent seats = SEAT_EMPTY, 0 chips). A ragged (len(players),) array crashes the native
    # engine (_to_flat) and silently corrupts indexing on the Python path — this is the live 5/4/3-handed
    # + HU path.
    from pokerbot.engine.game import PlayerState, Position, Street
    from pokerbot.engine.np_engine import SEAT_EMPTY

    st = GameState()
    st.players = [PlayerState(seat=i, position=p, stack_bb=100.0)
                  for i, p in enumerate([Position.BTN, Position.SB, Position.BB])]
    st.button_seat = 0
    st.street = Street.PREFLOP
    st.players[1].committed_street_bb = 0.5
    st.players[2].committed_street_bb = 1.0
    st.current_bet_bb = 1.0
    env = game_state_to_env(st, hero_seat=0)
    for fld in ("seat_status", "stacks", "committed_street", "committed_total"):
        assert np.asarray(getattr(env, fld)).shape == (6,), fld
    assert int((np.asarray(env.seat_status) == SEAT_EMPTY).sum()) == 3   # seats 3,4,5 empty


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
