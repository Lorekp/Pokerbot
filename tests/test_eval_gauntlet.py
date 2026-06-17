"""Empirical head-to-head bb/100 proof: exploit weak tables, never lose to GTO."""

from __future__ import annotations

from pathlib import Path

import pytest

_ART = Path("data/aof/cfr_base/cfr_base_3p.npz")
pytestmark = pytest.mark.skipif(not _ART.is_file(), reason="3-max base not built")


@pytest.mark.parametrize("stack_bb", [6.0, 8.0, 12.0])
def test_engine_exploits_weak_and_ties_gto(stack_bb) -> None:
    from scripts.eval_gauntlet import evaluate

    r = evaluate(stack_bb)
    # Never deviates vs a GTO/Pluribus-level table -> identical EV, loses nothing.
    assert r["exploit_vs_gto"] == pytest.approx(r["base_vs_gto"], abs=1e-6)
    # Exploitation captures large EV vs weak over-folders, well beyond non-exploiting base.
    assert r["exploit_vs_weak"] > r["base_vs_weak"] + 20.0   # >= +20 bb/100 uplift
    assert r["n_btn_deviations"] > 30                        # widens its jamming range
    # And it beats the weak table by far more than its equilibrium share vs GTO.
    assert r["exploit_vs_weak"] > r["base_vs_gto"] + 30.0


@pytest.mark.slow
def test_4max_engine_exploits_weak_table() -> None:
    """Headline 4-player claim: UTG extracts large bb/100 from a weak over-folding table."""
    from scripts.eval_gauntlet import evaluate_4max

    r = evaluate_4max(8.0, n_deals=6000)
    assert r["exploit_vs_weak"] > r["base_vs_weak"] + 20.0  # >= +20 bb/100 uplift in 4-max
    assert r["n_utg_deviations"] > 30
    assert r["exploit_vs_weak"] > 50.0                      # decisively beats the weak table
