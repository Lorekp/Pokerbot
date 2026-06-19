"""Mechanics tests for the SNG simulator (pokerbot/tournament/sim.py).

Validates correctness invariants (chip conservation, valid payouts, runs to completion, no seat
bias) at small N -- fast. The ROI/profit numbers (which need large N) live in
scripts/tournament/sng3_roi_sim.py.
"""

from __future__ import annotations

import numpy as np

import pokerbot.tournament.sim as sim
from pokerbot.tournament.pushfold_icm import _hu_grid

EQ, _LAB, W = _hu_grid()
STR = EQ.mean(axis=1)


def _topf(frac):
    order = np.argsort(-STR)
    keep = np.cumsum(W[order]) <= frac * W.sum()
    r = np.zeros(169)
    r[order[keep]] = 1.0
    return r


FIELD = {"jam": _topf(0.6), "call": _topf(0.35)}
PAY = [65.0, 35.0, 0.0]


def test_conservation_and_valid_payouts():
    # The conservation invariant is asserted inside simulate_sng; here we also check the payouts.
    rng = np.random.default_rng(0)
    cache = {}
    for _ in range(40):
        out = sim.simulate_sng(["field", "field", "field"], PAY, ante=0.1, rng=rng,
                               field=FIELD, cache=cache)
        assert sorted(out) == sorted(PAY)            # exactly one seat gets each prize
        assert abs(sum(out) - sum(PAY)) < 1e-9


def test_no_seat_bias_field_vs_field():
    # Three identical field players: by symmetry each seat must average ~the fair share.
    rng = np.random.default_rng(1)
    cache = {}
    res = np.array([sim.simulate_sng(["field"] * 3, PAY, ante=0.1, rng=rng, field=FIELD, cache=cache)
                    for _ in range(400)])
    fair = sum(PAY) / 3
    for seat in range(3):
        assert abs(res[:, seat].mean() - fair) < 4.0   # within noise of fair (no structural bias)


def test_engine_runs_and_cashes():
    # The ICM engine produces valid finishes and cashes a sensible fraction of the time.
    rng = np.random.default_rng(2)
    cache = {}
    res = np.array([sim.simulate_sng(["icm", "field", "field"], PAY, ante=0.1, rng=rng,
                                     field=FIELD, cache=cache) for _ in range(60)])
    itm = (res[:, 0] > 0).mean()
    assert 0.4 < itm < 0.95                            # reaches the money a reasonable fraction
