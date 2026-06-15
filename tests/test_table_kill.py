"""Table-aggression kill-switch: disable the whole exploit when >= AGG_KILL_COUNT seated villains are
confirmed LOOSE-aggressive (lags). Must fire on a lag-dominated table (the -0.48 leak) but NOT on a tag
("pro") table (the +0.99 gain we keep) -- the classifier keys on loose aggression (raises AND folds
little), which separates a lag from a tag.
"""

from __future__ import annotations

from pokerbot.opponent.population_prior import context_prior
from pokerbot.opponent.root_mixer import (AGG_KILL_COUNT, _is_dangerous_aggressor,
                                          _table_too_aggressive)
from pokerbot.opponent.stats import OpponentStats

_ST = ("flop", "turn", "river")
_TEX = ("dry", "wet", "paired")


def _profile(facing_bins, unopened_bins, hands=200):
    prof = OpponentStats()
    prof.hands = hands
    for st in _ST:
        for tex in _TEX:
            for fac, bins in (("unopened", unopened_bins), ("facing_bet", facing_bins)):
                ck = f"{st}|{tex}|{fac}"
                for h in range(1, 50):
                    prof.decay_and_add(ck, bins, context_prior(ck), h)
                prof.add_ctx(ck, [b * 24 for b in bins])
    return prof


def _lag():      # raises facing bets, never folds (loose) -> dangerous
    return _profile([0, 0, 1, 1], [0, 0, 1, 1])


def _tag():      # raises for value BUT folds tight -> aggressive yet NOT loose -> not dangerous
    return _profile([3, 1, 1, 1], [1, 1, 1, 0])


def _station():  # passive caller -> not aggressive
    return _profile([0, 1, 0, 0], [0, 1, 0, 0])


def _nit():      # folds a lot -> not loose, not aggressive
    return _profile([1, 0, 0, 0], [0, 1, 0, 0])


def _unknown():
    return _profile([0, 0, 1, 1], [0, 0, 1, 1], hands=10)


# ---- classifier: lag vs tag is the crux ----

def test_lag_is_dangerous_tag_is_not():
    assert _is_dangerous_aggressor(_lag()) is True
    assert _is_dangerous_aggressor(_tag()) is False   # keeps the +0.99 vs pros
    assert _is_dangerous_aggressor(_station()) is False
    assert _is_dangerous_aggressor(_nit()) is False
    assert _is_dangerous_aggressor(_unknown()) is False
    assert _is_dangerous_aggressor(None) is False


# ---- the table kill-switch ----

def test_all_lag_table_triggers_kill():
    profs = {s: _lag() for s in range(1, 6)}          # 5 lags
    assert _table_too_aggressive(profs) is True


def test_all_tag_pro_table_does_not_kill():
    profs = {s: _tag() for s in range(1, 6)}          # 5 tags -> exploit stays ON (+0.99 kept)
    assert _table_too_aggressive(profs) is False


def test_two_lags_below_threshold_keeps_exploit():
    profs = {1: _lag(), 2: _lag(), 3: _tag(), 4: _tag(), 5: _nit()}  # 2 lags < 3 (irc_tough -> +0.71)
    assert _table_too_aggressive(profs) is False


def test_three_lags_triggers_kill():
    profs = {1: _lag(), 2: _lag(), 3: _lag(), 4: _station(), 5: _station()}
    assert _table_too_aggressive(profs) is True


def test_cold_start_and_none_do_not_kill():
    assert _table_too_aggressive({s: _unknown() for s in range(1, 6)}) is False
    assert _table_too_aggressive(None) is False
    assert _table_too_aggressive({}) is False


def test_threshold_constant_is_three():
    assert AGG_KILL_COUNT == 3
