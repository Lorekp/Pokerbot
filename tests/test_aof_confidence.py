"""Advisor confidence metadata tests."""

from __future__ import annotations

import json

from aof import evaluate_spot_detailed
from aof_confidence import ApproximationTier, build_confidence


def test_marginal_edge_emits_warning() -> None:
    conf = build_confidence(
        action="ALL IN",
        ev_primary=0.1,
        ev_alt=0.0,
        detail_prefix="static-profile:",
    )
    assert conf.marginal
    assert conf.warning and "thin edge" in conf.warning


def test_fallback_range_warning() -> None:
    conf = build_confidence(
        action="FOLD",
        ev_primary=0.0,
        ev_alt=-0.5,
        detail_prefix="approx fallback static 2-shover:",
        fallback_used=True,
    )
    assert conf.fallback_used
    assert conf.approximation_tier == ApproximationTier.APPROX_FALLBACK_STATIC.value
    assert conf.warning and "fallback" in conf.warning


def test_hu_open_json_has_tier() -> None:
    r = evaluate_spot_detailed("AA SB 8bb", nash=True, force_call=False, samples=4000, seed=1)
    payload = r.to_dict()
    assert "confidence" in payload
    assert payload["confidence"]["approximation_tier"] in (
        ApproximationTier.SOLVER_HU.value,
        ApproximationTier.STATIC_PROFILE_HU_OR_SINGLE.value,
    )
    raw = json.dumps(payload)
    assert "approximation_tier" in raw


def test_facing_two_has_tier() -> None:
    line = "99 BB vs UTG,BTN 8bb"
    r = evaluate_spot_detailed(line, nash=True, force_call=True, samples=6000, seed=2)
    assert r.confidence.approximation_tier in (
        ApproximationTier.APPROX_FACING_TWO_CONDITIONAL.value,
        ApproximationTier.APPROX_FALLBACK_STATIC.value,
    )
