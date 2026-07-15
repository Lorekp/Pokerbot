"""Passive multiway-equity fallback for postflop coverage misses."""

from types import SimpleNamespace

import numpy as np

from pokerbot.engine.cards import Card
from pokerbot.search.postflop_miss_prior import postflop_passive_equity_policy


def _state(hole, board, *, to_call=10, pot=100, opponents=2):
    holes = np.full((opponents + 1, 2), -1, dtype=np.int32)
    holes[0] = [Card.from_str(c).index for c in hole]
    status = np.full(opponents + 1, 2, dtype=np.int8)
    committed = np.zeros(opponents + 1, dtype=np.int32)
    # The bettor's chips are already in the pot; hero must add ``to_call``.
    committed[1] = to_call
    return SimpleNamespace(
        board=np.asarray([Card.from_str(c).index for c in board], dtype=np.int32),
        holes=holes,
        committed_street=committed,
        current_bet=np.int32(to_call),
        pot_dead=np.int32(pot - to_call),
        seat_status=status,
        street=np.int32({3: 1, 4: 2, 5: 3}[len(board)]),
    )


def _legal_facing():
    legal = np.zeros(11, dtype=bool)
    legal[[0, 2, 7, 10]] = True
    return legal


def test_top_set_calls_cheap_multiway_bet_but_never_raises():
    state = _state(("As", "Ah"), ("Ac", "7d", "2c"),
                   to_call=10, pot=100, opponents=2)

    out, meta = postflop_passive_equity_policy(
        state, 0, _legal_facing(), n_iter=256)

    assert meta["fallback_source"] == "postflop_equity_prior"
    assert meta["postflop_guard_equity"] > meta["postflop_guard_required_equity"]
    assert out[2] == 1.0
    assert out[3:].sum() == 0.0


def test_air_folds_large_bet_and_never_raises():
    state = _state(("7s", "2d"), ("Ah", "Kc", "Qh"),
                   to_call=100, pot=200, opponents=1)

    out, meta = postflop_passive_equity_policy(
        state, 0, _legal_facing(), n_iter=256)

    assert meta["postflop_guard_equity"] < meta["postflop_guard_required_equity"]
    assert out[0] == 1.0
    assert out[3:].sum() == 0.0


def test_free_action_air_checks_and_never_invents_a_bluff():
    state = _state(("7s", "2d"), ("Ah", "Kc", "Qh"),
                   to_call=0, pot=100, opponents=3)
    legal = np.zeros(11, dtype=bool)
    legal[[1, 5, 10]] = True

    out, meta = postflop_passive_equity_policy(state, 0, legal)

    assert out[1] == 1.0
    assert meta["postflop_guard_equity"] < meta["postflop_guard_required_equity"]
    assert meta["postflop_guard_action"] == "check"
    assert out[3:].sum() == 0.0


def test_free_action_top_set_uses_only_smallest_non_allin_value_bet():
    state = _state(("As", "Ah"), ("Ac", "7d", "2c"),
                   to_call=0, pot=100, opponents=2)
    legal = np.zeros(11, dtype=bool)
    legal[[1, 5, 8, 10]] = True

    out, meta = postflop_passive_equity_policy(state, 0, legal, n_iter=256)

    assert meta["postflop_guard_equity"] >= meta["postflop_guard_required_equity"]
    assert meta["postflop_guard_action"] == "small_value_bet"
    assert out[5] == 1.0                 # smallest available sizing
    assert out[8] == 0.0 and out[10] == 0.0


def test_free_action_threshold_tightens_multiway():
    heads_up = _state(("As", "Ah"), ("Ac", "7d", "2c"),
                      to_call=0, pot=100, opponents=1)
    four_way = _state(("As", "Ah"), ("Ac", "7d", "2c"),
                      to_call=0, pot=100, opponents=3)
    legal = np.zeros(11, dtype=bool)
    legal[[1, 6, 10]] = True

    _, hu_meta = postflop_passive_equity_policy(heads_up, 0, legal, n_iter=64)
    _, mw_meta = postflop_passive_equity_policy(four_way, 0, legal, n_iter=64)

    assert mw_meta["postflop_guard_required_equity"] > (
        hu_meta["postflop_guard_required_equity"])
