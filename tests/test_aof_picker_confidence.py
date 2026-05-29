"""Picker / API confidence field tests."""

from __future__ import annotations

from aof import evaluate_spot_detailed
from aof_confidence import TIER_USER_LABELS


def test_open_response_tier_not_nash_claim() -> None:
    r = evaluate_spot_detailed("AJo BTN 8bb", nash=True, force_call=False, samples=8000, seed=42)
    label = TIER_USER_LABELS.get(r.confidence.approximation_tier, "")
    assert "not Nash" in label or "solver" in label.lower()
    assert label != "Nash equilibrium"


def test_facing_two_response_has_tier() -> None:
    r = evaluate_spot_detailed(
        "99 BB vs UTG,BTN 8bb", nash=True, force_call=True, samples=6000, seed=1
    )
    assert r.confidence.approximation_tier.startswith("approx_facing_two")


def test_confidence_dict_for_json_endpoint() -> None:
    r = evaluate_spot_detailed("AA SB 8bb", nash=True, force_call=False, samples=4000, seed=0)
    d = r.confidence.to_dict()
    assert "approximation_tier" in d
    assert "rake_configured" in d
    assert d["rake_configured"] is False
