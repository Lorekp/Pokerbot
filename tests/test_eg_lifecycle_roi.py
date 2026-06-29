"""Lifecycle-ROI NORMALIZATION pins (pokerbot/tournament/mtt_lifecycle.py).

This file owns the *normalization / field-composition* contract of the lifecycle ROI estimator — the
fix for the artifact where the headline ``roi`` looked constant-per-format and regime-insensitive and
could not be told apart from a sim/normalization artifact. The decisive invariant is the **self-play
control**: a fully SYMMETRIC field (the hero and every villain run the same ``decide_fn``) makes the
hero statistically interchangeable with every other seat, so its mean prize MUST equal the fair buy-in
share and ``roi -> 0`` (up to MC noise). Only once that control brackets 0 is a non-zero headline ROI a
REAL 'hero worse/better than the field' signal rather than a normalization artifact.

What is pinned here (net-free / CPU / small-N so the suite stays fast; the abstract resolver path is
used for speed and to keep the self-play symmetry exact — a fixed-threshold chip-EV player is perfectly
seat-symmetric, which is exactly what the control must verify):

  1. SELF-PLAY CONTROL ~ 0 — a symmetric ``chip_ev_nash`` field has ``roi`` whose 95% CI brackets 0 and
     ``mean_finish ~ (N+1)/2``, for an MTT (N=18) and an SNG (N=6). This is the normalization proof.
  2. ``self_play_roi`` FIELD — ``measure_lifecycle_roi(..., self_play_control=True)`` emits the control
     block, and the control CI brackets 0 even when the HERO is a DIFFERENT (weak/strong) policy: the
     control depends only on the field, not the measured hero.
  3. BUY-IN NORMALIZATION — ``buy_in_share == sum(payouts)/n_players`` exactly, and
     ``roi == (mean_payout - buy_in_share)/buy_in_share`` exactly (the formula, not just the sign).
     With an explicit rake-inclusive ``buy_in``, ``roi_net`` uses the real cost and is strictly below
     the gross pool-share ROI.
  4. ROI RESPONDS TO HERO STRENGTH — a deliberately WEAK hero (always-jam spew) has a clearly NEGATIVE
     ROI and a deliberately STRONG hero (tight) a clearly HIGHER ROI than the self-play baseline, vs the
     SAME chip-EV field; so once the control is centred, the sign/ordering of ROI is a real edge signal.
  5. FIELD DESCRIPTOR — the result carries a ``field`` block tagging homogeneous vs hetero composition.
"""

from __future__ import annotations

import random

import pytest

from pokerbot.tournament.mtt_lifecycle import (
    SpotDecision,
    anchor_jam,
    chip_ev_nash,
    measure_lifecycle_roi,
)

# MTT and SNG cells mirroring the suite's lifecycle_cell sizing (N=18 MTT, N=6 SNG).
_MTT_PAYOUTS = [35.4082, 27.0643, 21.2229, 16.3045]   # sums to 100.0 (a rake-free prize-pool fraction)
_MTT_N = 18
_SNG_PAYOUTS = [65.0, 35.0]
_SNG_N = 6


def _tight_hero(jam: float = 0.72, call: float = 0.82):
    """A deliberately STRONG hero: folds far more than the chip-EV field (deeper finishes)."""
    def fn(ctx):
        thr = jam if ctx.role == "jam" else call
        return SpotDecision(threshold=float(thr), source="tight")
    return fn


# ----------------------------------------------------------------- 1. self-play control ~ 0 (the proof)
@pytest.mark.parametrize("n_players,payouts,n_sims,seed", [
    (_MTT_N, _MTT_PAYOUTS, 500, 0),
    (_SNG_N, _SNG_PAYOUTS, 700, 1),
])
def test_self_play_field_roi_brackets_zero(n_players, payouts, n_sims, seed):
    """SYMMETRIC self-play field -> the hero is interchangeable with every seat -> roi must ~ 0.

    This is THE normalization control. If a symmetric field does not center the ROI at 0 the buy-in
    normalization / field composition is wrong (a sim artifact), regardless of the measured policy."""
    field = chip_ev_nash()
    res = measure_lifecycle_roi(
        n_players, payouts, field, villain_decide_fn=field,
        n_sims=n_sims, rng=random.Random(seed), realistic=True)
    lo, hi = res["roi_ci95"]
    # the symmetric field must center ROI at 0 (CI brackets 0)
    assert lo <= 0.0 <= hi, ("self-play ROI CI excludes 0 -> normalization bias", res["roi"], (lo, hi))
    # and the point estimate is small (within ~0.15 of 0 at this N)
    assert abs(res["roi"]) < 0.15, ("self-play ROI far from 0", res["roi"])
    # mean finish must be ~ (N+1)/2 (no seat/table/hero bias through the lifecycle)
    expected = (n_players + 1) / 2.0
    assert abs(res["mean_finish"] - expected) < 1.0, (res["mean_finish"], expected)


# ----------------------------------------------------- 2. self_play_control field is emitted & field-only
def test_self_play_control_block_emitted_and_brackets_zero():
    """``self_play_control=True`` emits a ``self_play_roi`` control block whose CI brackets 0, EVEN when
    the measured hero is a (weak) anchor — the control runs the FIELD as both seats, so it is independent
    of the hero policy."""
    field = chip_ev_nash()
    res = measure_lifecycle_roi(
        _MTT_N, _MTT_PAYOUTS, anchor_jam, villain_decide_fn=field,
        n_sims=400, rng=random.Random(3), realistic=True, self_play_control=True)
    assert "self_play_roi" in res
    ctrl = res["self_play_roi"]
    clo, chi = ctrl["roi_ci95"]
    assert clo <= 0.0 <= chi, ("control ROI CI excludes 0 -> normalization bias", ctrl["roi"], (clo, chi))
    # the control is the FIELD's self-play, so its mean_finish ~ (N+1)/2 regardless of the weak hero.
    assert abs(ctrl["mean_finish"] - ctrl["expected_mean_finish"]) < 1.0, ctrl
    # ... and a weak hero's HEADLINE roi (vs the field) is meaningfully WORSE than its own self-play
    # control (the control proves the gap is a real edge, not a normalization artifact).
    assert res["roi"] < ctrl["roi"] - 0.1, (res["roi"], ctrl["roi"])


# ------------------------------------------------------------------- 3. buy-in normalization is exact
def test_buy_in_share_and_roi_formula_are_exact():
    """``buy_in_share == sum(payouts)/n_players`` and ``roi == (mean_payout-share)/share`` exactly."""
    res = measure_lifecycle_roi(
        _MTT_N, _MTT_PAYOUTS, chip_ev_nash(), n_sims=120, rng=random.Random(5), realistic=False)
    share = sum(_MTT_PAYOUTS) / _MTT_N
    assert res["buy_in_share"] == pytest.approx(share, rel=0, abs=1e-12)
    expect_roi = (res["mean_payout"] - share) / share
    assert res["roi"] == pytest.approx(expect_roi, rel=0, abs=1e-12)
    rlo, rhi = res["roi_ci95"]
    assert rlo <= res["roi"] <= rhi


def test_roi_net_uses_real_cost_and_is_below_gross():
    """With an explicit rake-inclusive buy_in, ``roi_net`` uses the real cost (< the gross pool ROI)."""
    share = sum(_MTT_PAYOUTS) / _MTT_N
    buy_in = share * 1.10   # 10% rake -> real cost exceeds the rake-free pool share
    res = measure_lifecycle_roi(
        _MTT_N, _MTT_PAYOUTS, chip_ev_nash(), n_sims=120, rng=random.Random(8),
        realistic=False, buy_in=buy_in)
    assert res["buy_in"] == pytest.approx(buy_in)
    assert res["roi_net"] == pytest.approx((res["mean_payout"] - buy_in) / buy_in, abs=1e-12)
    assert res["roi_net"] < res["roi"]   # paying rake strictly lowers the net ROI


# ----------------------------------------------------- 4. ROI responds to hero strength (weak < self < strong)
def test_roi_orders_weak_below_self_below_strong():
    """vs the SAME chip-EV field: weak (always-jam) ROI < self-play ~0 < strong (tight) ROI.

    Once the self-play control is centred, the SIGN/ORDER of ROI is a real 'hero vs field' edge — this
    pins that a deliberately weak hero loses and a deliberately strong hero wins, monotonically."""
    field = chip_ev_nash()
    n_sims, seed = 500, 21

    weak = measure_lifecycle_roi(
        _MTT_N, _MTT_PAYOUTS, anchor_jam, villain_decide_fn=field,
        n_sims=n_sims, rng=random.Random(seed), realistic=True)
    selfp = measure_lifecycle_roi(
        _MTT_N, _MTT_PAYOUTS, chip_ev_nash(), villain_decide_fn=field,
        n_sims=n_sims, rng=random.Random(seed), realistic=True)
    strong = measure_lifecycle_roi(
        _MTT_N, _MTT_PAYOUTS, _tight_hero(), villain_decide_fn=field,
        n_sims=n_sims, rng=random.Random(seed), realistic=True)

    # weak hero is clearly negative (its CI excludes 0 below)
    assert weak["roi_ci95"][1] < 0.0, ("weak hero ROI not significantly negative", weak["roi_ci95"])
    # monotone ordering: weak < self-play(~0) < strong
    assert weak["roi"] < selfp["roi"] < strong["roi"], (weak["roi"], selfp["roi"], strong["roi"])
    # the self-play arm sits near 0, the strong hero clearly above it
    assert abs(selfp["roi"]) < 0.2, selfp["roi"]
    assert strong["roi"] > selfp["roi"] + 0.1, (strong["roi"], selfp["roi"])
    # the weak hero busts earlier, the strong hero finishes deeper (E4 corroboration)
    assert weak["mean_finish"] > selfp["mean_finish"] > strong["mean_finish"], (
        weak["mean_finish"], selfp["mean_finish"], strong["mean_finish"])


# -------------------------------------------------------------------------- 5. field-composition descriptor
def test_result_carries_field_descriptor():
    """The result tags the field composition (hetero vs homogeneous) so the ROI is interpretable."""
    field = chip_ev_nash()
    hetero = measure_lifecycle_roi(
        _MTT_N, _MTT_PAYOUTS, anchor_jam, villain_decide_fn=field,
        n_sims=20, rng=random.Random(0), realistic=False)
    assert "field" in hetero
    assert hetero["field"]["homogeneous"] is False   # distinct hero vs villain field

    one = chip_ev_nash()
    homo = measure_lifecycle_roi(
        _MTT_N, _MTT_PAYOUTS, one, villain_decide_fn=one,
        n_sims=20, rng=random.Random(0), realistic=False)
    assert homo["field"]["homogeneous"] is True      # same callable object on both sides


def test_lifecycle_cell_surfaces_self_play_control():
    """The suite adapter ``lifecycle_cell`` surfaces the self-play control + field descriptor so the
    grid JSON is self-documenting (a negative cell ROI is interpretable, not an opaque artifact).

    Tournament ROI is brutal-variance, so the self-play control is only a meaningful gate at adequate
    N (a smoke cell at n_sims<=12 is pure MC noise — which is itself the lesson: the control's CI width
    tells you whether N is large enough to trust the headline). We run a non-smoke cell at N=200 where
    the control's mean-finish is ~ (N+1)/2 and its ROI CI brackets 0 (normalization sound), and pin that
    the always-jam hero's ROI sits clearly BELOW the control."""
    from pokerbot.tournament.mtt_lifecycle import lifecycle_cell
    cell = lifecycle_cell("NORMAL", "deep", {
        "policy": "jam",            # net-free anchor hero so the cell is fast + deterministic
        "payouts": _MTT_PAYOUTS,
        "hands": 200, "smoke": False, "seed": 0,
    })
    assert "self_play_roi" in cell and "self_play_roi_ci95" in cell
    assert "field" in cell and isinstance(cell["field"], dict)
    slo, shi = cell["self_play_roi_ci95"]
    expected_finish = (cell["n_players"] + 1) / 2.0
    # at adequate N the control's mean-finish is ~ (N+1)/2 -> the symmetric field is unbiased.
    assert abs(cell["self_play_mean_finish"] - expected_finish) < 1.0, (
        cell["self_play_mean_finish"], expected_finish)
    # the control ROI CI brackets 0 (normalization is sound) ...
    assert slo <= 0.0 <= shi, ("cell self-play control excludes 0", cell["self_play_roi"], (slo, shi))
    # ... while the always-jam (weak) hero ROI sits clearly BELOW it -> a real edge, not an artifact.
    assert cell["roi"] < cell["self_play_roi"] - 0.2, (cell["roi"], cell["self_play_roi"])
