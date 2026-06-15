"""Value-fish gate: the unopened thin-value exploit fires ONLY when EVERY live opponent is a confirmed
station, so it's do-no-harm in any table constellation (the fix for the -0.55 bb/100 leak vs lags).

Covers the classifier (station=fish; lag/nit/reg/unknown=not) and the all-live-villains gate across the
constellations the user asked about: all-fish, 2-fish+1-aggressor, 2-fish+1-unknown, a folded aggressor
(ignored), heads-up, cold-start, and the no-profiles default.
"""

from __future__ import annotations

from pokerbot.opponent.population_prior import context_prior
from pokerbot.opponent.root_mixer import _all_live_villains_fish, _is_value_fish
from pokerbot.opponent.stats import OpponentStats

_ST = ("flop", "turn", "river")
_TEX = ("dry", "wet", "paired")


def _profile(facing_bins, unopened_bins, hands=200):
    """Build an OpponentStats whose facing_bet / unopened reads match the given 4-bin shapes
    [fold, passive, agg_small, agg_large] across every postflop context."""
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


def _station():  # calls everything facing a bet, passive unopened -> pays thin value, never punishes
    return _profile([0, 1, 0, 0], [0, 1, 0, 0])


def _lag():      # raises facing bets, bets/raises unopened -> punishes thin value
    return _profile([0, 0, 1, 1], [0, 0, 1, 1])


def _nit():      # folds to bets -> won't pay thin value
    return _profile([1, 0, 0, 0], [0, 1, 0, 0])


def _reg():      # folds appropriately + raises for value -> tighter/aggressive vs the loose-passive pop
    return _profile([2, 2, 1, 1], [1, 1, 1, 0])


def _unknown():  # too few hands -> unidentified
    return _profile([0, 1, 0, 0], [0, 1, 0, 0], hands=10)


# ---- classifier ----

def test_classifier_station_is_fish():
    assert _is_value_fish(_station()) is True


def test_classifier_lag_nit_reg_unknown_are_not_fish():
    assert _is_value_fish(_lag()) is False
    assert _is_value_fish(_nit()) is False
    assert _is_value_fish(_reg()) is False
    assert _is_value_fish(_unknown()) is False
    assert _is_value_fish(None) is False


# ---- the all-live-villains gate, every constellation ----

class _P:
    def __init__(self, folded=False):
        self.folded = folded


class _S:
    def __init__(self, n=6, folded=()):
        self.players = [_P(i in folded) for i in range(n)]


def _profiles(d):
    return d


def test_gate_all_fish_fires():
    st = _S(6)
    profs = {s: _station() for s in range(1, 6)}
    assert _all_live_villains_fish(st, 0, profs) is True


def test_gate_two_fish_one_aggressor_defers():
    st = _S(4)  # hero 0 + seats 1,2,3
    profs = {1: _station(), 2: _station(), 3: _lag()}
    assert _all_live_villains_fish(st, 0, profs) is False


def test_gate_two_fish_one_unknown_defers():
    st = _S(4)
    profs = {1: _station(), 2: _station(), 3: _unknown()}
    assert _all_live_villains_fish(st, 0, profs) is False


def test_gate_folded_aggressor_is_ignored():
    # The lag (seat 2) has folded -> only the live station (seat 1) matters -> fire.
    st = _S(3, folded=(2,))
    profs = {1: _station(), 2: _lag()}
    assert _all_live_villains_fish(st, 0, profs) is True


def test_gate_heads_up_station_fires_lag_defers():
    st = _S(2)
    assert _all_live_villains_fish(st, 0, {1: _station()}) is True
    assert _all_live_villains_fish(st, 0, {1: _lag()}) is False


def test_gate_cold_start_and_no_profiles_defer():
    st = _S(3)
    assert _all_live_villains_fish(st, 0, {1: _unknown(), 2: _unknown()}) is False
    assert _all_live_villains_fish(st, 0, None) is False
    assert _all_live_villains_fish(st, 0, {}) is False


def test_gate_missing_seat_profile_defers():
    # Seat 2 live but has no entry in the profiles dict -> unidentified -> defer.
    st = _S(3)
    assert _all_live_villains_fish(st, 0, {1: _station()}) is False
