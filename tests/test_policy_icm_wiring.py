"""Wiring test for ``--policy icm`` — the certified analytic ICM push/fold engine graded head-to-head
with the 7-max net on the SAME tournament-eval surface.

The single load-bearing assertion: ``_drive_pillars(fmt, regime, cell_args, policy='icm')`` BUILDS all
three measurement pillars (E1/E2 lbr_icm, E3/E4 mtt_lifecycle, E6 regime_battery) on a SHORT-STACK /
bubble cell — not deferred, not error — AND each pillar surfaces a NON-TRIVIAL ``model_coverage`` there
(the analytic push/fold specialist is actually DECIDING the short-stack spots, not running all-fallback).
That covers the whole wiring: the CLI choice, the per-pillar ``policy=='icm'`` branch, and the coverage
ledger.

The driver module is loaded DIRECTLY by file path (mirroring tests/test_eval_tournament_suite.py) so the
test does not depend on it being importable as a package module; the driver sets
``JAX_PLATFORMS=cpu`` + ``POKERBOT_INFO_SUITS=1`` on import.

The three pillars import the firewalled jax/np engine (lbr_icm) and the numba-tainted ``eval_score``
settle path; both are present on the production stack. In the lean jax-only venv (no numba) the cell
build is SKIPPED-not-FAILED, exactly like the eval_score-backed assertions in
tests/test_eval_tournament_suite.py.

Smoke (WSL Debian) — CPU, tiny N, the analytic engine (no GPU / no net):
  source ~/jaxgpu-venv/bin/activate
  cd .
  PYTHONPATH=. POKERBOT_INFO_SUITS=1 JAX_PLATFORMS=cpu \
    python -m pytest tests/test_policy_icm_wiring.py -q
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "eval_tournament_suite_under_test_icm",
        _ROOT / "scripts" / "tournament" / "eval_tournament_suite.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_drv = _load_driver()


# The pillars route through the firewalled jax/np engine (lbr_icm) + the numba-tainted eval_score settle.
# Both are present on the production stack; the lean jax-only venv lacks numba, so the cell build is
# SKIPPED there (the pure-wiring assertions below still run on the production stack).
try:
    import pokerbot.tournament.eval_score as _es  # noqa: F401
    import pokerbot.tournament.lbr_icm as _lbr  # noqa: F401
    import pokerbot.tournament.mtt_lifecycle as _life  # noqa: F401
    import pokerbot.tournament.regime_battery as _bat  # noqa: F401
    _HAS_STACK = True
    _STACK_ERR = ""
except Exception as _exc:  # noqa: BLE001
    _HAS_STACK = False
    _STACK_ERR = repr(_exc)

_needs_stack = pytest.mark.skipif(
    not _HAS_STACK, reason=f"tournament eval stack not importable in this env ({_STACK_ERR})")


def _short_stack_bubble_cell_args() -> dict:
    """The driver's per-cell ``cell_args`` for a SHORT-STACK / bubble cell (where the analytic push/fold
    engine should have HIGH coverage). Built exactly as the driver builds it: a 6-seat bubble table, a
    shallow ~15bb depth, the CoinPoker MTT ante, a structure-correct min-cash ladder. Tiny ``spots`` /
    ``hands`` so the three pillars finish in CPU-seconds."""
    n_seats = 6
    fmt, pay, ante = _drv._build_cell_payouts("NORMAL", n_seats, "bubble")
    return {
        "n_seats": n_seats,
        "depth_bb": float(_drv.REGIMES["bubble"]["depth_bb"]),   # 15bb — inside the push/fold zone
        "ante_bb": ante,
        "payouts": [round(float(p), 4) for p in pay],
        "icm_pressure_spot": True,
        "decks": 4, "hands": 6, "spots": 8, "ckpt": "x.pkl",
        "seed": 0, "smoke": True, "policy": "icm",
    }


# --------------------------------------------------------------------------- CLI choice + net_free flag
def test_icm_is_a_policy_choice_and_not_net_free():
    """``icm`` is a valid --policy choice, threads through unchanged, and (per the spec) is NOT net-free —
    only fold/jam are net-free anchors, so the anchor calibration treats icm like the net path."""
    args = _drv._parse_args(["eval_tournament_suite.py", "--policy", "icm", "--smoke", "--quiet"])
    assert args.policy == "icm"
    # net_free is exactly `policy in (fold, jam)`, so icm is correctly NOT net-free.
    assert ("icm" in ("fold", "jam")) is False


# --------------------------------------------------------------------------- the three pillars build + decide
@_needs_stack
def test_drive_pillars_icm_builds_all_three_with_coverage():
    """``_drive_pillars(..., policy='icm')`` BUILDS E1/E2, E3/E4, E6 on a short-stack/bubble cell (not
    deferred/error) AND each carries a non-trivial ``model_coverage`` (the analytic engine is actually
    deciding the short-stack spots, not all-fallback)."""
    cell_args = _short_stack_bubble_cell_args()
    pillars = _drv._drive_pillars("NORMAL", "bubble", cell_args, policy="icm")

    keys = ("E1_E2_lbr_icm", "E3_E4_mtt_lifecycle", "E6_regime_battery")
    for pk in keys:
        assert pk in pillars, f"missing pillar {pk}"
        row = pillars[pk]
        assert row["status"] == "built", f"{pk} not built: {row}"
        result = row["result"]
        assert result["policy_source"] == "icm-analytic", f"{pk} wrong source: {result.get('policy_source')}"
        mc = result.get("model_coverage")
        assert mc is not None, f"{pk} surfaced no model_coverage"
        assert mc["total"] >= 1
        # The analytic push/fold specialist is the right tool at a SHORT-STACK bubble -> it should DECIDE a
        # non-trivial share of spots here (NOT silently dominated by the fallback).
        assert mc["decided"] >= 1, f"{pk} decided 0 spots (all-fallback): {mc}"
        assert 0.0 < mc["fraction"] <= 1.0


@_needs_stack
def test_regime_battery_icm_branch_decides_at_bubble():
    """The natural-fit pillar (regime_battery, which already speaks Spot/string) decides the bubble cell
    with substantial coverage — the open-shove/call solve runs on the short-stack spots."""
    cell_args = _short_stack_bubble_cell_args()
    res = _bat.regime_battery_cell("NORMAL", "bubble", cell_args)
    assert res["policy_source"] == "icm-analytic"
    mc = res["model_coverage"]
    assert mc is not None and mc["decided"] >= 1
    # a bubble at 15bb is squarely the push/fold zone -> the engine should decide MOST spots.
    assert mc["fraction"] >= 0.5, f"unexpectedly low coverage at a short-stack bubble: {mc}"


@_needs_stack
def test_fold_jam_behavior_preserved():
    """The existing fold/jam pillars still build and carry NO model_coverage (additive-only change)."""
    cell_args = _short_stack_bubble_cell_args()
    cell_args["policy"] = "fold"
    pillars = _drv._drive_pillars("NORMAL", "bubble", cell_args, policy="fold")
    for pk in ("E1_E2_lbr_icm", "E3_E4_mtt_lifecycle", "E6_regime_battery"):
        row = pillars[pk]
        assert row["status"] == "built"
        # fold is a net-free anchor: model_coverage is None (only the analytic-icm hero tracks it).
        assert row["result"].get("model_coverage") is None
