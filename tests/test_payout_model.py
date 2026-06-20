"""Tests for :mod:`pokerbot.tournament.payouts` -- the tournament payout-structure model.

Validates:
  * every captured NORMAL and PKO ladder is reproduced from the json (<1e-3 per rank), sums to 1.0,
    non-increasing;
  * the PKO clubbed top tiers (1st == 2nd) come out exactly tied;
  * ``places_paid`` is monotone non-decreasing and tracks the field-fraction for large fields;
  * uncaptured sizes interpolate to a valid (sum 1, non-increasing) structure;
  * satellite + normal currency vectors are pool-conserving;
  * classification mappings (incl. the misspelled ``MTT_SATTELITE``);
  * a payout_vector conserves the pool through ICM (sum(equity) == sum(vector)).
"""

import json
import os

import numpy as np
import pytest

from pokerbot.tournament import payouts as P
from pokerbot.tournament.payouts import TournamentFormat as TF
from pokerbot.tournament.icm import icm_equity


_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts",
    "tourney_payout_model.json",
)


def _model():
    with open(_MODEL_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


CAPTURED_NORMAL = [2, 8, 9, 11, 12, 13, 38, 42, 46, 56, 62, 68, 80, 96, 104]
CAPTURED_PKO = [6, 13, 20, 31, 41, 55, 62]


def _json_fracs(fmt_key, size):
    curve = _model()[fmt_key]["payout_curve"][str(size)]
    sbr = sorted(curve["share_by_rank"], key=lambda e: e["rank"])
    pct = [e["mean_pct"] for e in sbr]
    s = sum(pct)
    return [p / s for p in pct]  # json fractions, renormalized to sum 1 for a fair comparison


# --------------------------------------------------------------------------------------------------
# Captured curve reproduction
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("size", CAPTURED_NORMAL)
def test_captured_normal_reproduces(size):
    struct = P.payout_structure(TF.NORMAL, size)
    assert len(struct) == size
    assert abs(sum(struct) - 1.0) < 1e-9
    # non-increasing
    for i in range(len(struct) - 1):
        assert struct[i] >= struct[i + 1] - 1e-12
    # each rank within 1e-3 of the (renormalized) json fraction
    ref = _json_fracs("normal", size)
    for got, want in zip(struct, ref):
        assert abs(got - want) < 1e-3


@pytest.mark.parametrize("size", CAPTURED_PKO)
def test_captured_pko_reproduces(size):
    struct = P.payout_structure(TF.PKO, size)
    assert len(struct) == size
    assert abs(sum(struct) - 1.0) < 1e-9
    for i in range(len(struct) - 1):
        assert struct[i] >= struct[i + 1] - 1e-12
    ref = _json_fracs("pko", size)
    for got, want in zip(struct, ref):
        assert abs(got - want) < 1e-3


# --------------------------------------------------------------------------------------------------
# PKO tied top tiers
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("size", [13, 20, 31, 41, 55, 62])
def test_pko_top_tie_exact(size):
    """PKO 1st == 2nd: the json shows these as a clubbed '1-2' tier (or rank1~=rank2)."""
    struct = P.payout_structure(TF.PKO, size)
    assert abs(struct[0] - struct[1]) < 1e-6


# --------------------------------------------------------------------------------------------------
# places_paid
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fmt,frac", [(TF.NORMAL, 0.175), (TF.PKO, 0.133)])
def test_places_paid_monotone(fmt, frac):
    prev = 0
    for n in range(2, 3001):
        p = P.places_paid(fmt, n)
        assert p >= prev  # monotone non-decreasing
        prev = p
        if n >= 200:
            assert abs(p / n - frac) < 0.02  # tracks the field-fraction for large fields


def test_places_paid_floor_and_clamp():
    # min_paid floor
    assert P.places_paid(TF.NORMAL, 1) >= 1
    assert P.places_paid(TF.NORMAL, 5) == 2  # max(2, round(0.875)) = 2
    # never exceeds entrants
    assert P.places_paid(TF.NORMAL, 3) <= 3


def test_places_paid_satellite_seats():
    assert P.places_paid(TF.SATELLITE, 100, seats=10) == 10
    assert P.places_paid(TF.SATELLITE, 7, seats=10) == 7  # cannot pay more seats than entrants


# --------------------------------------------------------------------------------------------------
# Uncaptured interpolation
# --------------------------------------------------------------------------------------------------


def test_uncaptured_normal_50():
    struct = P.payout_structure(TF.NORMAL, 50)
    assert len(struct) == 50
    assert abs(sum(struct) - 1.0) < 1e-9
    for i in range(len(struct) - 1):
        assert struct[i] >= struct[i + 1] - 1e-12


def test_uncaptured_pko_25():
    struct = P.payout_structure(TF.PKO, 25)
    assert len(struct) == 25
    assert abs(sum(struct) - 1.0) < 1e-9
    for i in range(len(struct) - 1):
        assert struct[i] >= struct[i + 1] - 1e-12


# --------------------------------------------------------------------------------------------------
# Currency vectors
# --------------------------------------------------------------------------------------------------


def test_satellite_payout_vector():
    vec = P.payout_vector(TF.SATELLITE, seats=10, seat_value=530)
    assert vec == [530] * 10
    assert sum(vec) == 5300


def test_normal_payout_vector_conserves_pool():
    fmt = TF.NORMAL
    entrants, buy_in = 300, 150
    pool = P.prize_pool(fmt, entrants, buy_in)
    vec = P.payout_vector(fmt, entrants=entrants, buy_in=buy_in)
    assert len(vec) == P.places_paid(fmt, entrants)
    assert abs(sum(vec) - pool) < 1e-6


# --------------------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------------------


def test_classify_mappings():
    assert P.classify(tournament_type="MTT_FREEZEOUT") is TF.NORMAL
    assert P.classify(tournament_type="MTT_REBUY") is TF.NORMAL
    assert P.classify(tournament_type="MTT_PROGRESSIVE_KNOCK_OUT") is TF.PKO
    assert P.classify(tournament_type="MTT_SATTELITE") is TF.SATELLITE  # misspelled lobby value
    # flag-driven
    assert P.classify(bounty=True) is TF.PKO
    assert P.classify(is_satellite=True) is TF.SATELLITE
    # satellite wins over bounty
    assert P.classify(is_satellite=True, bounty=True) is TF.SATELLITE
    assert P.classify(tournament_type="MTT_SATTELITE", bounty=True) is TF.SATELLITE
    # default
    assert P.classify() is TF.NORMAL
    assert P.classify(tournament_type="SOMETHING_UNKNOWN") is TF.NORMAL


# --------------------------------------------------------------------------------------------------
# Pool conservation through ICM
# --------------------------------------------------------------------------------------------------


def test_pool_conserved_through_icm():
    # Malmuth-Harville ICM is O(n!/(n-k)!), so feed a final-table-sized (tractable) ladder where the
    # number of paid ranks == the number of stacks, making the WHOLE pool conserved. The pool scale
    # comes from a real entrants/buy_in computation.
    pool = P.prize_pool(TF.NORMAL, 300, 150)
    vec = P.payout_vector(TF.NORMAL, places=9, prize_pool=pool)  # 9 paid -> 9 stacks
    assert len(vec) == 9
    assert abs(sum(vec) - pool) < 1e-6
    rng = np.random.default_rng(7)
    stacks = (rng.random(len(vec)) + 0.1) * 1000.0  # arbitrary positive stacks, one per paid rank
    eq = icm_equity(stacks, vec)
    assert abs(float(np.sum(eq)) - sum(vec)) < 1e-6


def test_pool_conserved_through_icm_final_table():
    # a final-table snapshot: 6 paid, 6 stacks -> the whole vector is consumed.
    vec = P.payout_vector(TF.PKO, entrants=46, buy_in=50)  # places_paid(PKO,46)=6
    assert len(vec) == 6
    stacks = [1500.0, 1200.0, 800.0, 500.0, 300.0, 100.0]
    eq = icm_equity(stacks, vec)
    assert abs(float(np.sum(eq)) - sum(vec)) < 1e-6


# --------------------------------------------------------------------------------------------------
# buyin_split / pools
# --------------------------------------------------------------------------------------------------


def test_buyin_split_normal_and_pko():
    n = P.buyin_split(TF.NORMAL, 100.0, rake=0.08)
    assert abs(n["prize_contrib"] - 92.0) < 1e-9
    assert n["bounty_contrib"] == 0.0
    k = P.buyin_split(TF.PKO, 100.0, rake=0.08)
    assert abs(k["prize_contrib"] - 46.0) < 1e-9
    assert abs(k["bounty_contrib"] - 46.0) < 1e-9


def test_prize_pool_guarantee_floor():
    # tiny field floored by a guarantee
    pool = P.prize_pool(TF.NORMAL, 10, 1.0, guarantee=1000.0)
    assert pool == 1000.0
