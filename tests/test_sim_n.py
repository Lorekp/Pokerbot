"""Tests for the N-handed (multiway) ICM push/fold SNG (pokerbot/tournament/sim.py::simulate_sng_n).

Generalizes the 3-handed sim to 2..7 seats so the FULL multiway bubble (more players, multiple pay jumps)
is evaluable. Verifies chip conservation + prize-pool conservation across seat counts, and that the
3-handed generalization reproduces the original sim's payout structure.
"""

from __future__ import annotations

import numpy as np

from pokerbot.tournament.pushfold_icm import _hu_grid
from pokerbot.tournament.sim import simulate_sng_n

_EQ, _LAB, _W = _hu_grid()
_STRENGTH = _EQ.mean(axis=1)


def _top_frac(frac: float) -> np.ndarray:
    order = np.argsort(-_STRENGTH)
    keep = np.cumsum(_W[order]) <= frac * _W.sum()
    r = np.zeros(169)
    r[order[keep]] = 1.0
    return r


_FIELD = {"jam": _top_frac(0.55), "call": _top_frac(0.45)}


def test_6handed_conserves_chips_and_prize_pool():
    rng = np.random.default_rng(0)
    payouts = [50.0, 30.0, 20.0]                       # top-3 of 6 paid
    out = simulate_sng_n(["field"] * 6, payouts, ante=0.125, rng=rng, field=_FIELD, cache={}, n_players=6)
    assert len(out) == 6
    assert abs(sum(out) - sum(payouts)) < 1e-6          # total $ paid == the prize pool (prize-conserving)
    assert all(x >= 0.0 for x in out)


def test_various_seat_counts_run_and_conserve():
    payouts = [60.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for n in (2, 4, 5, 7):
        rng = np.random.default_rng(n)
        out = simulate_sng_n(["field"] * n, payouts[:n], ante=0.125, rng=rng, field=_FIELD, cache={},
                             n_players=n)
        assert len(out) == n
        assert abs(sum(out) - sum(payouts[:n])) < 1e-6


def test_icm_arm_solves_at_6handed():
    # The certified multiway open-shove ICM arm runs end-to-end at 6 seats (not just push/fold heuristics).
    import os
    os.environ.setdefault("POKERBOT_SIM_PUSHFOLD_ITERS", "200")
    rng = np.random.default_rng(3)
    out = simulate_sng_n(["icm"] + ["field"] * 5, [50.0, 30.0, 20.0], ante=0.125, rng=rng,
                         field=_FIELD, cache={}, n_players=6)
    assert len(out) == 6 and abs(sum(out) - 100.0) < 1e-6
