"""Tests that the format x regime suite driver ACTUALLY GRADES the three E-pillars.

THE GAP THIS GUARDS (the real one)
----------------------------------
``scripts/tournament/eval_tournament_suite._drive_pillars`` looks up, per pillar, a stable entry point:
  * ``pokerbot.tournament.lbr_icm.lbr_icm_cell``
  * ``pokerbot.tournament.mtt_lifecycle.lifecycle_cell``
  * ``pokerbot.tournament.regime_battery.regime_battery_cell``
and marks the E-row ``deferred`` when it is absent (or ``error`` when the call raises). Before these
adapters existed, EVERY E-pillar reported ``deferred`` on a ``--policy net`` run — so no real net
measurement happened. This test asserts that ``_drive_pillars(... policy='net')`` returns status
``'built'`` (NOT ``deferred``/``error``) for all three pillars on a tiny cell, and that each result dict
carries its expected headline metric keys — i.e. the suite genuinely grades the served 7-max net.

Net-served: needs jax + numba + the 7-max checkpoint. Skips gracefully (not fails) when any is absent so
the suite stays green in a lean env; the wiring (entry-point NAME presence) is asserted unconditionally.

Smoke (WSL Debian):
  source ~/jaxgpu-venv/bin/activate
  cd .
  PYTHONPATH=. POKERBOT_INFO_SUITS=1 JAX_PLATFORMS=cpu \
    python -m pytest tests/test_tournament_suite_pillar_wiring.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Match the served encoder layout to training BEFORE importing pokerbot (the suite sets these too).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("POKERBOT_INFO_SUITS", "1")

import scripts.tournament.eval_tournament_suite as suite  # noqa: E402

_CKPT = "checkpoints/sd_cfr_7max_neural/seven_v1.pkl"

# The net-served path needs the checkpoint + jax + numba ($-scoring core). Probe once; skip the heavy
# net assertions (not fail) where unavailable — but the entry-point NAME wiring is asserted always.
try:
    import jax  # noqa: F401
    from pokerbot.tournament import eval_score as _es  # noqa: F401
    _HAS_NET_DEPS = True
except Exception as _e:  # noqa: BLE001
    _HAS_NET_DEPS = False
    _NET_ERR = repr(_e)

_HAS_CKPT = Path(_CKPT).exists()

_needs_net = pytest.mark.skipif(
    not (_HAS_NET_DEPS and _HAS_CKPT),
    reason=("net-served pillars need jax+numba+checkpoint; "
            + ("checkpoint missing" if not _HAS_CKPT else "jax/eval_score unavailable")),
)


def _tiny_cell_args(policy: str) -> dict:
    """A tiny per-cell context (matching the driver's cell_args construction) for a fast net-served grade."""
    return {
        "n_seats": 7, "depth_bb": 15.0, "ante_bb": 0.125,
        "payouts": [50.0, 30.0, 20.0], "icm_pressure_spot": True,
        "decks": 4, "hands": 6, "spots": 6, "ckpt": _CKPT,
        "seed": 0, "smoke": True, "policy": policy,
    }


# --------------------------------------------------------------------------- the wiring (name) guard
def test_pillar_entry_points_exist():
    """The three entry-point NAMES the driver looks up must be present (the exact gap that caused the
    universal ``deferred``). This runs in ANY env (no jax/numba needed)."""
    from pokerbot.tournament import lbr_icm, mtt_lifecycle, regime_battery
    assert callable(getattr(lbr_icm, "lbr_icm_cell", None)), "lbr_icm.lbr_icm_cell missing"
    assert callable(getattr(mtt_lifecycle, "lifecycle_cell", None)), "mtt_lifecycle.lifecycle_cell missing"
    assert callable(getattr(regime_battery, "regime_battery_cell", None)), \
        "regime_battery.regime_battery_cell missing"


# --------------------------------------------------------------------------- the grade-not-deferred guard
@_needs_net
def test_drive_pillars_all_built_not_deferred_on_net():
    """``_drive_pillars(... policy='net')`` returns status 'built' (not deferred/error) for all three
    pillars on a tiny net-served cell — i.e. the suite ACTUALLY grades the served 7-max net."""
    rows = suite._drive_pillars("NORMAL", "bubble", _tiny_cell_args("net"), policy="net")

    expected = {"E1_E2_lbr_icm", "E3_E4_mtt_lifecycle", "E6_regime_battery"}
    assert set(rows.keys()) == expected, f"unexpected pillar rows: {set(rows.keys())}"

    for name, row in rows.items():
        st = row.get("status")
        assert st == "built", f"pillar {name} not built: status={st!r} reason={row.get('reason')!r}"
        assert "result" in row and isinstance(row["result"], dict), f"{name} missing result dict"


# --------------------------------------------------------------------------- per-pillar metric keys
@_needs_net
def test_lbr_icm_row_carries_dollar_exploitability_metrics():
    rows = suite._drive_pillars("NORMAL", "bubble", _tiny_cell_args("net"), policy="net")
    r = rows["E1_E2_lbr_icm"]["result"]
    for k in ("dollar_exploit_per_game", "ci95", "chip_exploit_bb_per_100",
              "sevenmax_chip_anchor_bb_per_100", "less_exploitable_than_always_fold", "hands", "verdict"):
        assert k in r, f"lbr_icm result missing {k!r}"
    assert isinstance(r["ci95"], list) and len(r["ci95"]) == 2
    assert r["hands"] > 0
    assert r["policy_source"] == "net"


@_needs_net
def test_lifecycle_row_carries_roi_itm_finish_metrics():
    rows = suite._drive_pillars("NORMAL", "bubble", _tiny_cell_args("net"), policy="net")
    r = rows["E3_E4_mtt_lifecycle"]["result"]
    for k in ("roi", "roi_ci95", "itm_rate", "mean_finish", "paired_diff_mean",
              "paired_diff_ci95", "paired_diff_verdict", "n_sims"):
        assert k in r, f"lifecycle result missing {k!r}"
    assert 0.0 <= r["itm_rate"] <= 1.0
    assert r["mean_finish"] > 0.0
    assert r["policy_source"] == "net"


@_needs_net
def test_regime_battery_row_carries_delta_icm_metrics():
    rows = suite._drive_pillars("NORMAL", "bubble", _tiny_cell_args("net"), policy="net")
    r = rows["E6_regime_battery"]["result"]
    for k in ("delta_icm_mean", "delta_icm_ci95", "chipev_mean", "flip_rate",
              "uplift_zero_frac", "verdict", "n", "battery_cell"):
        assert k in r, f"regime_battery result missing {k!r}"
    assert isinstance(r["delta_icm_ci95"], list) and len(r["delta_icm_ci95"]) == 2
    assert r["n"] > 0
    assert r["policy_source"] == "net"


# --------------------------------------------------------------------------- net-free path still builds
@_needs_net
def test_drive_pillars_built_on_fold_baseline():
    """The net-free fold baseline also drives all three pillars to 'built' (the calibration arm)."""
    rows = suite._drive_pillars("NORMAL", "shallow", _tiny_cell_args("fold"), policy="fold")
    for name, row in rows.items():
        assert row.get("status") == "built", \
            f"{name} not built on fold: {row.get('status')!r} {row.get('reason')!r}"
