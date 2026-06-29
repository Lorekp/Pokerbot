r"""Certification of the cash-eval scoring + verdict CORE (``pokerbot.eval.cash_eval_score``).

These are the load-bearing invariants the ``cash_eval_suite`` driver's verdict rests on:

  * **net additivity** -- ``net_on = chip_on - net_cost_bb`` (and ``net_off`` likewise), so the net
    column is EXACTLY the chip column minus the rake cost, and the uplift is rake-invariant.
  * **the hump is non-monotone** -- a realistic net-bb/100 curve over ``rake.STAKE_ORDER`` rises into a
    mid-stake sweet-spot then falls as the BB-cap binds tighter, so the peak sits at an INTERIOR stake
    and ``is_hump`` is True (a monotone curve is not a hump).
  * **a chip-positive / net-negative stake reads NEGATIVE** -- the cell verdict is computed on the NET
    CI, so a stake that wins chips but loses to the rake in the BB-cap valley reports a NEGATIVE cell.
  * the VR-provenance overrides (``var_reduction <= 1`` / half-width-miss -> INCONCLUSIVE) and the
    three-gate GO/NO-GO are sane.

Net-free, CPU-only, numpy-only -- runs in any env (no jax/torch/engine).
"""

from __future__ import annotations

# --- the abstraction flags MUST be baked BEFORE pokerbot is imported (the encoder reads them at import).
# This module's serve-skew test (test_serve_fingerprint_mismatch_is_detected) cross-checks the SAME guard
# the cash_eval_suite serves the blueprint through, so set the flags up front exactly as the suite does.
import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")
_os.environ.setdefault("POKERBOT_INFO_SUITS", "1")
_os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
_os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
_os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
_os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")
_os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")

import numpy as np
import pytest

from pokerbot.economics.rake import STAKE_ORDER, net_cost_bb
from pokerbot.eval.cash_eval_score import (
    DEFAULT_EPS_RNR,
    CellResult,
    cell_verdict,
    ci95_from_se,
    hump_table,
    net_from_chip,
    stake_go_no_go,
    stat_block,
    verdict,
)


# ---------------------------------------------------------------------------------------------------
# (1) net additivity:  net_on = chip_on - net_cost_bb  (the spec's headline identity)
# ---------------------------------------------------------------------------------------------------
def test_net_from_chip_is_exact_subtraction():
    """net = chip - net_cost, byte-exact, for arbitrary inputs."""
    for chip, cost in [(120.0, 8.5), (-5.0, 8.5), (0.0, 0.0), (3.3, 3.3)]:
        assert net_from_chip(chip, cost) == chip - cost


def test_net_uplift_is_rake_invariant():
    """Because the rake cost is policy-agnostic at a fixed stake, the NET uplift equals the CHIP uplift:
    (chip_on - cost) - (chip_off - cost) == chip_on - chip_off. This is why we can sweep stakes off ONE
    deck series and the uplift column never moves with the stake."""
    chip_on, chip_off = 130.0, 25.0
    for cost in [0.0, 4.2, 8.5, 12.0]:
        net_on = net_from_chip(chip_on, cost)
        net_off = net_from_chip(chip_off, cost)
        assert abs((net_on - net_off) - (chip_on - chip_off)) < 1e-12


def test_net_cost_uses_real_rake_model_per_stake():
    """The per-hand net cost is the real ``rake.net_cost_bb`` (5%-flat capped, per-tier BB-cap, rakeback,
    splash). Sanity: a fixed postflop pot at every stake yields a NON-CONSTANT cost across stakes (the
    cap + rakeback band differ), and applying ``net_from_chip`` reproduces chip - that-cost exactly."""
    pot_bb, hero_committed, n_dealt = 40.0, 12.0, 6
    chip_on = 100.0
    costs = {}
    for stake in STAKE_ORDER:
        cost_bb_per_100 = 100.0 * net_cost_bb(pot_bb, hero_committed, n_dealt, False, stake)
        costs[stake] = cost_bb_per_100
        assert net_from_chip(chip_on, cost_bb_per_100) == chip_on - cost_bb_per_100
    # not all stakes cost the same (the cap + rakeback band vary) -> the sweep is meaningful.
    assert len({round(c, 6) for c in costs.values()}) > 1


# ---------------------------------------------------------------------------------------------------
# (2) the net curve is a NON-MONOTONE HUMP (peaks at an interior / mid stake)
# ---------------------------------------------------------------------------------------------------
def test_hump_table_detects_non_monotone_peak():
    """A synthetic rise-then-fall net curve over the first five STAKE_ORDER stakes humps: the peak sits
    at an INTERIOR stake (not an endpoint) and ``is_hump`` is True."""
    stakes = STAKE_ORDER[:5]
    # rise into the middle then fall (the BB-cap hump shape).
    vals = [2.0, 6.0, 8.5, 5.0, 1.0]
    net_by_stake = dict(zip(stakes, vals))
    ht = hump_table(net_by_stake)
    assert ht["is_hump"] is True
    assert ht["peak_stake"] == stakes[2]            # interior peak
    assert ht["peak_stake"] not in (stakes[0], stakes[-1])
    assert ht["peak_value"] == 8.5
    # curve is reported in STAKE_ORDER, not insertion order.
    assert [c["stake"] for c in ht["curve"]] == stakes


def test_hump_table_monotone_curve_is_not_a_hump():
    """A purely rising (or falling) curve is monotone -> NOT a hump (the peak is an endpoint)."""
    stakes = STAKE_ORDER[:4]
    rising = hump_table(dict(zip(stakes, [1.0, 2.0, 3.0, 4.0])))
    assert rising["is_hump"] is False
    assert rising["peak_stake"] == stakes[-1]       # endpoint
    falling = hump_table(dict(zip(stakes, [4.0, 3.0, 2.0, 1.0])))
    assert falling["is_hump"] is False
    assert falling["peak_stake"] == stakes[0]


def test_hump_table_models_real_rake_valley():
    """Built from the REAL rake model: a fixed chip win-rate net of the per-stake rake cost yields a curve
    whose valley (worst net) is a low/mid stake and whose best net is NOT that valley -- i.e. the BB-cap
    hump is visible from the actual rake.py numbers (not just a synthetic shape)."""
    pot_bb, hero_committed, n_dealt = 30.0, 10.0, 6
    chip_on = 40.0  # bb/100, a healthy chip edge
    net_by_stake = {
        s: net_from_chip(chip_on, 100.0 * net_cost_bb(pot_bb, hero_committed, n_dealt, False, s))
        for s in STAKE_ORDER
    }
    ht = hump_table(net_by_stake)
    # The worst net stake must NOT be the best net stake (the rake cost is non-uniform across the curve).
    assert ht["valley_stake"] != ht["peak_stake"]
    assert ht["valley_value"] < ht["peak_value"]


# ---------------------------------------------------------------------------------------------------
# (3) a CHIP-POSITIVE but NET-NEGATIVE stake -> NEGATIVE cell verdict
# ---------------------------------------------------------------------------------------------------
def test_chip_positive_net_negative_cell_is_negative():
    """A stake where the hero wins chips (+) but the rake eats more than the edge (net < 0, CI entirely
    below 0) must produce a NEGATIVE cell verdict -- the verdict is computed on the NET CI, so chip
    positivity does not rescue it."""
    chip_on = 5.0           # small chip edge, bb/100
    net_cost = 8.5          # the worst BB-cap valley cost, bb/100
    net_on = net_from_chip(chip_on, net_cost)   # = -3.5, net-losing
    assert net_on < 0.0
    se = 0.5                # tight CI entirely below 0
    lo, hi = ci95_from_se(net_on, se)
    assert hi < 0.0
    assert cell_verdict(lo, hi, var_reduction=5.0) == "NEGATIVE"


def test_chip_positive_net_positive_cell_is_positive():
    """The mirror: a chip edge that SURVIVES the rake (net > 0, tight CI above 0) reads POSITIVE."""
    net_on = net_from_chip(40.0, 8.5)           # = +31.5
    lo, hi = ci95_from_se(net_on, 1.0)
    assert lo > 0.0
    assert cell_verdict(lo, hi, var_reduction=5.0) == "POSITIVE"


# ---------------------------------------------------------------------------------------------------
# VR-provenance overrides + the base verdict reducer
# ---------------------------------------------------------------------------------------------------
def test_cell_verdict_var_reduction_le_zero_forces_inconclusive():
    """Even a CI that excludes 0 reads INCONCLUSIVE when the control variate ADDED variance
    (var_reduction <= 0): a negative VR means the CV is anti-correlated, never a clean signal."""
    v = cell_verdict(0.5, 2.0, var_reduction=-0.2)   # CI excludes 0 but VR went negative
    assert v.startswith("INCONCLUSIVE")
    assert "var_reduction" in v
    # VR=0 is the boundary (no variance reduction at all, but not anti-correlated) -> INCONCLUSIVE.
    v0 = cell_verdict(0.5, 2.0, var_reduction=0.0)
    assert v0.startswith("INCONCLUSIVE")


def test_cell_verdict_positive_var_reduction_below_one_passes_through():
    """Regression (F-EVAL-10): the gate was previously ``<= 1.0``, which wrongly demoted every positive
    CV whose VR happened to fall under 1 (e.g. 0.95). The fix is ``<= 0.0`` -- the CV only fails the
    gate when it actively added variance, not when it merely reduced by less than the target. A VR of
    0.95 must NOT force INCONCLUSIVE; the base reducer (CI excludes 0 -> POSITIVE) should win."""
    assert cell_verdict(0.5, 2.0, var_reduction=0.95) == "POSITIVE"
    assert cell_verdict(0.5, 2.0, var_reduction=1.5) == "POSITIVE"
    assert cell_verdict(0.5, 2.0, var_reduction=0.99) == "POSITIVE"


def test_cell_verdict_halfwidth_miss_forces_inconclusive():
    """A realized half-width that misses the target precision forces INCONCLUSIVE regardless of CI sign."""
    v = cell_verdict(0.5, 5.0, var_reduction=3.0, realized_halfwidth=2.25, target_halfwidth=1.0)
    assert v.startswith("INCONCLUSIVE")
    assert "half-width" in v


def test_cell_verdict_passes_through_base_reducer():
    """With good provenance, cell_verdict is exactly the tournament base verdict reducer."""
    assert cell_verdict(0.1, 0.5, var_reduction=4.0) == verdict(0.1, 0.5) == "POSITIVE"
    assert cell_verdict(-0.5, -0.1, var_reduction=4.0) == verdict(-0.5, -0.1) == "NEGATIVE"
    assert cell_verdict(-0.2, 0.3, var_reduction=4.0).startswith("INCONCLUSIVE")


# ---------------------------------------------------------------------------------------------------
# the three-gate per-stake GO / NO-GO
# ---------------------------------------------------------------------------------------------------
def test_go_requires_all_three_gates():
    """GO only when net-positive AND pro-safe AND the exploit prints (all three lower/upper CIs clear)."""
    go = stake_go_no_go(
        net_on_ci95=(2.0, 8.0),         # net lower CI > 0
        uplift_ci95=(1.0, 5.0),         # uplift lower CI > 0
        lbr_delta_ci95=(-3.0, 1.0),     # delta upper CI <= eps_rnr (default 5)
    )
    assert go["verdict"] == "GO"
    assert go["net_positive"] == "GO" and go["pro_safe"] == "GO" and go["exploit_prints"] == "GO"


def test_no_go_when_net_negative_even_if_exploit_prints():
    """A net-negative stake (chip-positive valley) is NO-GO even with a real, pro-safe uplift."""
    res = stake_go_no_go(
        net_on_ci95=(-6.0, -1.0),       # net lower & upper CI < 0
        uplift_ci95=(1.0, 5.0),
        lbr_delta_ci95=(-3.0, 1.0),
    )
    assert res["verdict"] == "NO-GO"
    assert res["net_positive"] == "NO-GO"


def test_no_go_when_counter_exploitable():
    """A Delta whose LOWER CI exceeds the RNR tolerance (provably more exploitable) is NO-GO (pro-unsafe),
    no matter how good the net win-rate is."""
    res = stake_go_no_go(
        net_on_ci95=(5.0, 12.0),
        uplift_ci95=(2.0, 6.0),
        lbr_delta_ci95=(10.0, 20.0),    # delta lower CI > eps_rnr=5 -> provably counter-exploitable
        eps_rnr=DEFAULT_EPS_RNR,
    )
    assert res["verdict"] == "NO-GO"
    assert res["pro_safe"] == "NO-GO"


def test_inconclusive_when_m3_absent():
    """No M3 (lbr_delta_ci95 None) -> pro_safe is INCONCLUSIVE -> the stake cannot be a clean GO."""
    res = stake_go_no_go(
        net_on_ci95=(2.0, 8.0),
        uplift_ci95=(1.0, 5.0),
        lbr_delta_ci95=None,
    )
    assert res["pro_safe"] == "INCONCLUSIVE"
    assert res["verdict"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------------------------------
# CellResult JSON row shape (the driver emits these)
# ---------------------------------------------------------------------------------------------------
def test_cell_result_as_json_round_trips_fields():
    c = CellResult(
        stake="0.50/1", depth_bb=100.0, position=3,
        net_off=20.0, net_on=31.5, net_uplift=11.5, uplift_ci95=(8.0, 15.0),
        var_reduction=4.2, corr=0.93, ci95=(28.0, 35.0), verdict="POSITIVE",
        lbr_delta=1.1, lbr_delta_ci95=(-2.0, 4.0), bet_shift=0.0119,
    )
    j = c.as_json()
    assert j["stake"] == "0.50/1" and j["depth_bb"] == 100.0 and j["position"] == 3
    assert j["net_uplift"] == 11.5 and j["verdict"] == "POSITIVE"
    assert j["uplift_ci95"] == [8.0, 15.0] and j["lbr_delta_ci95"] == [-2.0, 4.0]
    assert j["bet_shift"] == 0.0119


def test_stat_block_and_verdict_are_reexported():
    """cash_eval_score re-exports the tournament stat_block/verdict so cash consumers import ONE module."""
    sb = stat_block([1.0, 1.0, 1.0])
    assert sb["n"] == 3 and abs(sb["mean"] - 1.0) < 1e-12
    assert verdict(0.1, 0.2) == "POSITIVE"


# ---------------------------------------------------------------------------------------------------
# SERVE-SKEW FAIL-CLOSED (the #1 cash correctness risk — the -559 bb/100 "config artifact")
#
# HARD CONSTRAINT: the cash_eval_suite serves the 6-max blueprint, whose encoder bakes the abstraction
# flags AT IMPORT. If the served flags differ from the snapshot's stamped abstraction dict, EVERY affected
# node fingerprint-MISSES, postflop coverage collapses to ~0%, the serve falls back to uniform-legal, and
# the measured win-rate is garbage (the -559 bb/100 artifact, corrected to +79 only once all flags matched).
# Every cash estimator that serves the blueprint funnels through ``lbr.check_blueprint_serve_env(strict=True,
# expected=snap['abstraction'])`` (cash_serve_gate.assert_cash_serve, het_cv_absolute(snap_abstraction=...),
# cash_pro_safety, cash_hero). These tests pin that a deliberate single-field stamp MISMATCH is DETECTED
# (fail-closed), so the artifact can NEVER silently recur, and that a MATCHING stamp does NOT raise.
# ---------------------------------------------------------------------------------------------------
def _active_abstraction():
    """The abstraction config the encoder baked at import (the snapshot stamp it must match). Skips the
    test cleanly if the encoder cannot import (a net-free env without jax) — the guard itself is net-free,
    but ``active_abstraction_config`` lives in the engine encoder module."""
    try:
        from pokerbot.engine import jax_weva_encode as enc
        return enc.active_abstraction_config()
    except Exception as exc:  # noqa: BLE001 — no encoder in this env -> skip (CI runs it in the jax venv)
        pytest.skip(f"abstraction encoder unavailable ({exc!r}); serve-skew test runs in the jax venv")


def test_serve_fingerprint_match_does_not_raise():
    """The control: serving with a stamp EQUAL to the baked flags passes the strict guard (no false alarm).
    If this raised, the flags were not set before import (the test harness sets them at module top)."""
    from pokerbot.eval.lbr import check_blueprint_serve_env

    cfg = _active_abstraction()
    # A matching stamp must NOT raise (strict). This is the exact call the suite's serve gate makes.
    check_blueprint_serve_env(strict=True, expected=dict(cfg))


@pytest.mark.parametrize("field", ["preflop_lossless", "betting_abstraction", "postflop_num_buckets"])
def test_serve_fingerprint_mismatch_is_detected(field):
    """A deliberate single-field stamp MISMATCH must FAIL CLOSED (RuntimeError), for each of the
    fingerprint-bearing flags — including ``preflop_lossless`` (the one the LBR modules do NOT setdefault,
    the single most common cash serve skew). This is what makes the -559 bb/100 artifact unrepeatable: a
    served abstraction that does not match the snapshot stamp can never silently mis-measure."""
    from pokerbot.eval.lbr import check_blueprint_serve_env

    cfg = _active_abstraction()
    bad = dict(cfg)
    cur = cfg.get(field)
    # Flip/perturb just this one field so the stamp no longer matches the baked encoder flags.
    if isinstance(cur, bool) or cur is None:
        bad[field] = not bool(cur)
    elif isinstance(cur, (int, float)):
        bad[field] = cur + 1
    else:
        bad[field] = str(cur) + "_MISMATCH"

    with pytest.raises(RuntimeError):
        check_blueprint_serve_env(strict=True, expected=bad)
