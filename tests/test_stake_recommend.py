"""Stake-targeting advisory (scripts/cash_rake_floor.py stake_recommendations).

Given a measured gross edge, the conservative (floor-rakeback) per-stake net + PLAY/avoid verdict.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from pokerbot.economics import rake as rk

_SPEC = importlib.util.spec_from_file_location(
    "cash_rake_floor", Path(__file__).resolve().parents[1] / "scripts" / "cash_rake_floor.py")
crf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(crf)


def _curve():
    return crf.analytic_rake_curve(26.0, 20.0, 0.45, 1.2, 1 << 14, 42)


def test_floor_mode_is_more_conservative_than_modeled():
    curve = _curve()
    # certain 15% rakeback leaves MORE fee than the modeled band -> higher floor -> harder to beat.
    for sk in rk.STAKE_ORDER:
        assert crf.total_floor(curve, sk, "floor") >= crf.total_floor(curve, sk, "modeled") - 1e-9


def test_recommendations_sorted_and_flagged():
    curve = _curve()
    rows = crf.stake_recommendations(curve, gross_edge_bb=6.0, mode="floor")
    assert [r["stake"] for r in rows]  # non-empty
    # sorted best-net first
    nets = [r["net"] for r in rows]
    assert nets == sorted(nets, reverse=True)
    # ok iff net positive
    for r in rows:
        assert r["ok"] == (r["net"] > 0.0)
        assert abs(r["net"] - (6.0 - r["floor"])) < 1e-9


def test_tough_field_clears_nowhere():
    curve = _curve()
    rows = crf.stake_recommendations(curve, gross_edge_bb=-2.0, mode="floor")
    assert all(not r["ok"] for r in rows)   # a losing gross edge beats no stake's floor


def test_high_stake_floor_below_low_valley():
    curve = _curve()
    rows = {r["stake"]: r for r in crf.stake_recommendations(curve, 5.0, mode="floor")}
    # the high band's floor (cap collapses ~25x, 15% rakeback persists) sits below the low-stake valley.
    assert rows["25/50"]["floor"] < rows["0.25/0.50"]["floor"]
    # and the low band (fat cap + splash) is the worst floor of all.
    assert rows["0.05/0.10"]["floor"] >= rows["0.50/1"]["floor"]
