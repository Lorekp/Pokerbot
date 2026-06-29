"""Tests for the PKO SNG evaluator (pokerbot/tournament/sim_pko.py).

Verifies the bounty pool is conserved end-to-end, the total realized $ equals prize pool + bounty pool,
and the defining PKO lever: a positive villain bounty LOWERS the call threshold, so the bounty-aware arm
calls strictly wider than the prize-only baseline (chip-EV != $-EV).
"""

from __future__ import annotations

import numpy as np

from pokerbot.tournament.bounty import min_equity_to_call
from pokerbot.tournament.sim_pko import _EQ, _W, equity_vs_range, simulate_pko_sng

_STRENGTH = _EQ.mean(axis=1)


def _top_frac(frac: float) -> np.ndarray:
    """Indicator over the top ``frac`` of combos by strength (descending) -- a simple field range."""
    order = np.argsort(-_STRENGTH)
    keep = np.cumsum(_W[order]) <= frac * _W.sum()
    r = np.zeros(169)
    r[order[keep]] = 1.0
    return r


_FIELD = {"jam": _top_frac(0.5), "call": _top_frac(0.4)}
PRIZE = [65.0, 35.0, 0.0]
ENTRY_BOUNTY = 33.0


def test_pko_sng_conserves_prize_and_bounty():
    rng = np.random.default_rng(0)
    cache: dict = {}
    out = simulate_pko_sng(["pko", "field", "field"], PRIZE, ENTRY_BOUNTY, ante=0.125, rng=rng,
                           field=_FIELD, cache=cache)
    assert len(out) == 3 and all(x >= 0.0 for x in out)
    # total realized $ == prize pool + bounty pool (both conserved): sum(PRIZE) + 3*entry_bounty
    assert abs(sum(out) - (sum(PRIZE) + 3 * ENTRY_BOUNTY)) < 1e-6


def test_many_sngs_stay_conserved():
    rng = np.random.default_rng(7)
    cache: dict = {}
    target = sum(PRIZE) + 3 * ENTRY_BOUNTY
    for _ in range(50):
        out = simulate_pko_sng(["pko", "icm", "field"], PRIZE, ENTRY_BOUNTY, ante=0.125, rng=rng,
                               field=_FIELD, cache=cache)
        assert abs(sum(out) - target) < 1e-6


def test_bounty_lowers_call_threshold_widening_calls():
    stacks = [12.0, 12.0, 12.0]
    p_no_bounty = min_equity_to_call(stacks, PRIZE, 0, 1, villain_bounty=0.0)
    p_bounty = min_equity_to_call(stacks, PRIZE, 0, 1, villain_bounty=ENTRY_BOUNTY)
    assert p_bounty < p_no_bounty           # the bounty sweetens the call -> lower indifference equity
    assert 0.0 <= p_bounty < p_no_bounty <= 1.0


def test_equity_vs_range_is_sane():
    # AA vs a wide range should be very strong; 72o vs a wide range weak; empty range -> 0.5
    aa = equity_vs_range("AA", _top_frac(0.6))
    trash = equity_vs_range("72o", _top_frac(0.6))
    assert aa > 0.7 and trash < 0.5 and aa > trash
    assert equity_vs_range("AA", np.zeros(169)) == 0.5
