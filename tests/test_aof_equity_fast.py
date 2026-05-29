"""Parity tests for optimized equity paths."""

from __future__ import annotations

import pytest

pytest.importorskip("pokerkit")
pytest.importorskip("phevaluator")

from aof import shove_ev_open
from aof_equity import equity_hu_exact, equity_vs_range, equity_vs_range_fast, warmup
from aof_parse import parse_spot_line


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    warmup()


def test_equity_vs_range_fast_near_legacy() -> None:
    hero = "AsKs"
    rng = "QQ"
    a = equity_vs_range(hero, rng, 8000, 42, 48)
    b = equity_vs_range_fast(hero, rng, 8000, 42, 48)
    assert abs(a - b) < 0.02


def test_equity_hu_exact_sane() -> None:
    eq = equity_hu_exact(("As", "Ad"), ("Kh", "Kd"), 6000, 7)
    assert 0.75 < eq < 0.92


def test_shove_ev_open_unchanged_vs_precompute() -> None:
    spot = parse_spot_line("8c5s UTG 8bb")
    ev, ef, _ = shove_ev_open(spot, samples=6000, seed=42)
    assert ev > ef
    spot2 = parse_spot_line("AKo BTN 8bb")
    ev2, _, _ = shove_ev_open(spot2, samples=6000, seed=42)
    assert ev2 > 0.5
