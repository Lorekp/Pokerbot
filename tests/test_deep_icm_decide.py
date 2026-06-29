"""Tests for the deep (>20bb) ICM call-vs-jam serve decision (live_adapter.deep_icm_decide).

Verifies the integration: equity-vs-range -> the deep-ICM gate -> a fold/defer Decision. The decision must
match the gate exactly, a clear monster defers (let the +EV call go to the search), and the override
actually FIRES for a marginal hand near the money (the measured deep-ICM value-add).
"""

from __future__ import annotations

import numpy as np

from pokerbot.tournament.deep_icm_gate import deep_icm_overrides_shove
from pokerbot.tournament.live_adapter import _EQ, _W, _equity_vs_range, deep_icm_decide

_STRENGTH = _EQ.mean(axis=1)
STACKS = [35.0, 45.0, 27.0, 20.0, 12.0, 7.0]            # hero 35bb DEEP, near the money
PAY = [50.0, 30.0, 20.0]


def _top_frac(frac: float) -> np.ndarray:
    order = np.argsort(-_STRENGTH)
    keep = np.cumsum(_W[order]) <= frac * _W.sum()
    r = np.zeros(169)
    r[order[keep]] = 1.0
    return r


def test_decision_matches_the_gate_for_every_hand():
    jam = _top_frac(0.5)
    for hand in ["AA", "KK", "TT", "99", "A9s", "KJo", "76s", "72o"]:
        d = deep_icm_decide(STACKS, PAY, 0, 1, hand, jam)
        eq = _equity_vs_range(hand, jam)
        override = deep_icm_overrides_shove(STACKS, PAY, 0, 1, eq, min(STACKS[0], STACKS[1]))
        assert (d.action == "fold") == override          # decide == gate
        assert d.action in ("fold", "defer") and d.source == "deep-icm-gate"


def test_clear_monster_defers_to_the_chip_ev_search():
    # AA is a clear +EV call -> ICM agrees -> defer (no override of the search)
    d = deep_icm_decide(STACKS, PAY, 0, 1, "AA", _top_frac(0.5))
    assert d.action == "defer"


def test_override_fires_for_a_marginal_hand_near_the_money():
    # Somewhere in the hand set there must be a hand the gate FOLDS (else the value-add never triggers here,
    # contradicting the grid). A medium hand vs a wide jam range lands in the ~0.50-0.60 flip band.
    jam = _top_frac(0.55)
    folds = [h for h in ["AA", "KK", "QQ", "JJ", "TT", "99", "88", "AQs", "AJs", "KQs", "ATo", "KJo", "QTo"]
             if deep_icm_decide(STACKS, PAY, 0, 1, h, jam).action == "fold"]
    assert folds, "deep-ICM gate never fired — the measured value-add is not reachable in this config"
