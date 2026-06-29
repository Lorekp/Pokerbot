"""Certification of the chip->$ scoring + variance-reduction CORE (``pokerbot.tournament.eval_score``).

This is the single scoring core every tournament-eval consumer imports, so its invariants are load-bearing:

  * **C0 prize conservation** -- a NORMAL/SNG hand's per-seat $-deltas sum to ~0 (the prize pool is fixed,
    so ICM only REDISTRIBUTES it); and the underlying ICM equity itself conserves the pool.
  * **mean-preservation of the all-in leaf** -- ``settle_leaf`` on a FULLY-DEALT board equals the exact,
    deterministic side-pot settle (the equity average over 0 needed runouts is the settle itself), and is
    independent of ``settle_seed`` there (no RNG is consumed) -> mean-preserved and CRN-stable.
  * **ICM control-variate identity** -- ``score_with_cv`` reduces to 0 on a no-op (realized == ICM(pre)),
    and a synthetic per-spot series fed to ``cv_absolute.decompose_absolute`` reproduces the analytic CV mean
    (the baseline-subtraction algebra is the same control variate this core exposes).
  * **the chip-EV side-channel column** -- PKO scoring reduces EXACTLY to the ICM-only delta when there is no
    bounty, so the $-EV column degrades gracefully to the chip-EV/ICM column.

Net-free, CPU-only, tiny samples -- runs in any env with numpy (no jax/torch/engine search needed).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pytest

from pokerbot.tournament import bounty as _bounty
from pokerbot.tournament import payouts as _payouts
from pokerbot.tournament.eval_score import (
    TournamentFormat,
    score_chip_to_dollar,
    score_with_cv,
    settle_leaf,
    stat_block,
    verdict,
)
from pokerbot.tournament.icm import icm_equity
from pokerbot.tournament.rollout_leaf import equity_allin_settle

NORMAL = TournamentFormat.NORMAL
SATELLITE = TournamentFormat.SATELLITE
PKO = TournamentFormat.PKO


# --------------------------------------------------------------------------- lightweight state stub
# equity_allin_settle reads ONLY these attributes: state.players[i].{folded, hole, committed_total_bb},
# state.board, where each hole-card / board-card has an integer .index. A faithful, engine-free stub.
@dataclass
class _Card:
    index: int


@dataclass
class _Player:
    committed_total_bb: float
    folded: bool = False
    hole: tuple | None = None  # (Card, Card) for a contender, None for a folder


@dataclass
class _State:
    players: list
    board: list = field(default_factory=list)


def _cards(*idxs):
    return [_Card(i) for i in idxs]


def _full_board_allin_state():
    """Two contenders all-in on a COMPLETE 5-card board (need=0 -> deterministic exact settle), one folder.

    Seat 0: As Ks (idx via rank*4+suit), seat 1: Qd Qc, seat 2 folded. Board is 5 distinct cards none of
    which collide with the holes. Committed equally so it is one main pot.
    """
    # rank*4+suit with ranks 0..12 (2..A), suits 0..3. Pick disjoint indices.
    h0 = (_Card(51), _Card(47))   # rank12 s.. (top pair of high cards)
    h1 = (_Card(43), _Card(42))   # rank10 ..
    board = _cards(0, 5, 10, 15, 20)  # five low/distinct, no overlap with holes
    players = [
        _Player(committed_total_bb=10.0, folded=False, hole=h0),
        _Player(committed_total_bb=10.0, folded=False, hole=h1),
        _Player(committed_total_bb=4.0, folded=True, hole=None),
    ]
    return _State(players=players, board=board)


# =========================================================================== C0 prize conservation
def test_c0_normal_score_conserves_prize_pool():
    """A NORMAL hand only REDISTRIBUTES the fixed prize pool -> per-seat $-deltas sum to ~0."""
    pre = [50.0, 30.0, 20.0]
    chip = [+15.0, -15.0, 0.0]           # seat 0 wins 15 chips off seat 1 (sum 0)
    payouts = [100.0, 60.0]               # top-2 paid (3-handed SNG bubble)
    d = score_chip_to_dollar(pre, chip, payouts, NORMAL)
    assert abs(float(d.sum())) < 1e-9, "NORMAL $-delta must conserve the prize pool (sum 0)"
    # and it is exactly the ICM redistribution (post - pre equity).
    post = [65.0, 15.0, 20.0]
    expect = icm_equity(post, payouts) - icm_equity(pre, payouts)
    assert np.allclose(d, expect, atol=1e-12)


def test_icm_equity_conserves_pool_identity():
    """The underlying ICM identity the core relies on: sum(equity) == sum(payouts[:n])."""
    stacks = [40.0, 35.0, 25.0]
    payouts = [120.0, 70.0, 30.0]
    eq = icm_equity(stacks, payouts)
    assert abs(float(eq.sum()) - sum(payouts)) < 1e-9


def test_satellite_routes_through_icm_with_flat_seat_ladder():
    """SATELLITE uses the SAME ICM wrap over a flat [seat_value]*seats ladder (step-payout) -> conserves."""
    pre = [50.0, 30.0, 20.0]
    chip = [+20.0, -20.0, 0.0]
    seat_value = 11.0
    payouts = _payouts.payout_vector(SATELLITE, seats=2, seat_value=seat_value, entrants=3)
    assert payouts == [seat_value, seat_value]
    d = score_chip_to_dollar(pre, chip, payouts, SATELLITE)
    assert abs(float(d.sum())) < 1e-9, "SATELLITE step-payout must conserve seat value"


# =========================================================================== mean-preserving all-in leaf
def test_settle_leaf_equals_deterministic_exact_settle_on_full_board():
    """On a COMPLETE board the equity leaf needs 0 runouts -> it IS the deterministic side-pot settle.

    settle_leaf must reproduce that exact chip settle (mapped to $), and be independent of settle_seed
    (no RNG consumed) -> mean-preserved + CRN-stable at the exact case.
    """
    state = _full_board_allin_state()
    pre = [50.0, 30.0, 20.0]
    payouts = [100.0, 60.0]

    # Ground truth: the raw deterministic chip settle (no RNG path on a full board).
    chip_truth = equity_allin_settle(state, n_runouts=999, rng=random.Random(123))
    n = len(state.players)
    chip_vec = [chip_truth.get(i, 0.0) for i in range(n)]
    dollar_truth = score_chip_to_dollar(pre, chip_vec, payouts, NORMAL)

    # Two DIFFERENT settle_seeds must give the SAME result on a full board (no runout RNG).
    leaf_a = settle_leaf(state, payouts, pre, NORMAL, settle_seed=1, n_runouts=64)
    leaf_b = settle_leaf(state, payouts, pre, NORMAL, settle_seed=99999, n_runouts=8)
    for i in range(n):
        assert leaf_a[i] == pytest.approx(leaf_b[i], abs=1e-12), "full-board settle must ignore settle_seed"
        assert leaf_a[i] == pytest.approx(float(dollar_truth[i]), abs=1e-12), "must equal exact settle ($)"

    # And the chip leaf is zero-sum -> the NORMAL $-mapping conserves the pool.
    assert abs(sum(chip_vec)) < 1e-9
    assert abs(sum(leaf_a.values())) < 1e-9


def test_settle_leaf_crn_pairing_cancels_runout_noise_off_board():
    """Off a complete board the leaf samples runouts; a SHARED settle_seed makes two arms share the SAME
    runout draws (CRN) -> identical chip leaf for identical states. This is the variance-cancellation hook."""
    # Incomplete board (need>0 -> rng.sample path).
    state = _State(
        players=[
            _Player(10.0, False, (_Card(51), _Card(47))),
            _Player(10.0, False, (_Card(43), _Card(42))),
        ],
        board=_cards(0, 5, 10),  # flop only -> 2 cards to come
    )
    pre = [40.0, 60.0]
    payouts = [100.0, 60.0]
    a = settle_leaf(state, payouts, pre, NORMAL, settle_seed=7, n_runouts=32)
    b = settle_leaf(state, payouts, pre, NORMAL, settle_seed=7, n_runouts=32)
    assert a == b, "same settle_seed + same state -> byte-identical (CRN cancels runout noise)"


# =========================================================================== ICM control-variate identity
def test_score_with_cv_reduces_to_zero_on_noop():
    """A no-op hand: realized_$ already equals the ICM baseline -> the CV residual is exactly 0."""
    pre = [50.0, 30.0, 20.0]
    payouts = [100.0, 60.0]
    hero = 0
    baseline = float(icm_equity(pre, payouts)[hero])
    assert score_with_cv(baseline, pre, payouts, hero) == pytest.approx(0.0, abs=1e-12)


def test_score_with_cv_is_unbiased_baseline_subtraction():
    """score_with_cv = realized_$ - ICM(pre)[hero] exactly (the chip-EV baseline subtraction)."""
    pre = [44.0, 33.0, 23.0]
    payouts = [120.0, 60.0]
    hero = 1
    realized = 57.3
    expect = realized - float(icm_equity(pre, payouts)[hero])
    assert score_with_cv(realized, pre, payouts, hero) == pytest.approx(expect, abs=1e-12)


def test_cv_series_reproduces_analytic_decompose_mean():
    """A synthetic per-spot series fed to cv_absolute.decompose_absolute reproduces the analytic CV mean.

    Build a correlated (search S, blueprint B) pair + a disjoint cheap blueprint sample Bc. The CV core's
    baseline-subtraction algebra (E[S] = E[S - beta*B] + beta*E[Bc]) is EXACTLY what decompose_absolute
    computes; verify the reported mean matches the closed-form value and is unbiased (mean ~ E[S]).
    """
    from pokerbot.eval.cv_absolute import decompose_absolute

    rng = np.random.default_rng(0)
    n, m = 400, 4000
    # S = B + small independent noise -> high corr -> large variance reduction.
    B = rng.normal(0.0, 1.0, size=n)
    S = B + rng.normal(0.0, 0.1, size=n)
    Bc = rng.normal(0.0, 1.0, size=m)

    res = decompose_absolute(S, B, Bc, optimal_beta=True)

    # Closed form the core uses: mean = mean(S) - beta*(mean(B) - mean(Bc)), beta = cov(S,B)/var(B).
    beta = np.cov(S, B, ddof=1)[0, 1] / np.var(B, ddof=1)
    expect_mean = S.mean() - beta * (B.mean() - Bc.mean())
    assert res.mean == pytest.approx(float(expect_mean), rel=1e-9, abs=1e-12)
    assert res.beta == pytest.approx(float(beta), rel=1e-9)
    # The control variate sharpens, never biases: it cut the SE vs the naive estimate.
    assert res.var_reduction > 1.0
    assert res.se < res.naive_se


# =========================================================================== PKO chip-EV side-channel column
def test_pko_reduces_to_icm_only_when_no_bounty():
    """PKO scoring with villain_bounty==0 must equal the pure ICM-only delta on EVERY seat (the chip-EV
    side-channel: the $-EV column degrades gracefully to the ICM/chip-EV column when no bounty is at stake).
    """
    pre = [50.0, 30.0, 20.0]
    payouts = [100.0, 60.0]
    hero, villain = 0, 1
    risked = min(pre[hero], pre[villain])   # all-in for the smaller stack
    # Hero wins the all-in: +risked / -risked chip delta (sum 0).
    chip = [0.0, 0.0, 0.0]
    chip[hero] = +risked
    chip[villain] = -risked

    icm_only = score_chip_to_dollar(pre, chip, payouts, NORMAL)
    # hero_equity=1.0 == hero won the all-in -> the post-stacks pko prices match the chip-win ICM delta.
    pko = score_chip_to_dollar(pre, chip, payouts, PKO,
                               hero=hero, villain=villain, hero_equity=1.0, villain_bounty=0.0)
    assert np.allclose(pko, icm_only, atol=1e-9), "no-bounty PKO must reduce to the ICM-only column"


def test_pko_bounty_adds_positive_dollar_on_win():
    """A positive villain bounty must STRICTLY increase hero's $-equity delta vs the no-bounty case
    (the chip-EV != $-EV effect: covering an all-in collects bounty cash on a win)."""
    pre = [50.0, 30.0, 20.0]
    payouts = [100.0, 60.0]
    hero, villain = 0, 1
    risked = min(pre[hero], pre[villain])
    chip = [+risked if i == hero else (-risked if i == villain else 0.0) for i in range(3)]

    no_b = score_chip_to_dollar(pre, chip, payouts, PKO,
                                hero=hero, villain=villain, hero_equity=1.0, villain_bounty=0.0)
    with_b = score_chip_to_dollar(pre, chip, payouts, PKO,
                                  hero=hero, villain=villain, hero_equity=1.0, villain_bounty=8.0)
    assert with_b[hero] > no_b[hero] + 1e-9
    # the exact bounty cash on a certain win (p=1): cash + discounted growth.
    bcash = _bounty.bounty_cash_on_win(8.0)
    assert with_b[hero] - no_b[hero] == pytest.approx(bcash, abs=1e-9)


def test_pko_requires_allin_pair():
    with pytest.raises(ValueError):
        score_chip_to_dollar([50.0, 30.0], [+5.0, -5.0], [100.0], PKO)  # missing hero/villain/equity


# =========================================================================== stat_block + verdict
def test_stat_block_mirrors_battery_stat():
    vals = [1.0, 2.0, 3.0, 4.0]
    sb = stat_block(vals)
    a = np.asarray(vals, dtype=np.float64)
    assert sb["n"] == 4
    assert sb["mean"] == pytest.approx(2.5)
    assert sb["std"] == pytest.approx(float(a.std(ddof=1)))
    assert sb["se"] == pytest.approx(sb["std"] / np.sqrt(4))
    assert sb["bb_per_100"] == pytest.approx(250.0)


def test_stat_block_singleton_and_empty_have_zero_spread():
    assert stat_block([5.0]) == {"n": 1, "mean": 5.0, "std": 0.0, "se": 0.0, "bb_per_100": 500.0}
    assert stat_block([]) == {"n": 0, "mean": 0.0, "std": 0.0, "se": 0.0, "bb_per_100": 0.0}


def test_verdict_idiom():
    assert verdict(0.1, 0.5) == "POSITIVE"
    assert verdict(-0.5, -0.1) == "NEGATIVE"
    assert verdict(-0.1, 0.3) == "INCONCLUSIVE (CI spans 0)"
    assert verdict(0.0, 0.3) == "INCONCLUSIVE (CI spans 0)"   # boundary: ci_lo==0 does NOT exclude 0
    assert verdict(-0.3, 0.0) == "INCONCLUSIVE (CI spans 0)"
