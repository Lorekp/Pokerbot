"""
Pinned preflop equity references for aof_equity.py.

Fast tests use combo-level HU exact or deterministic MC at fixed seed + max_combos=64.
Slow tests use higher sample counts vs chart references.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pokerkit")
pytest.importorskip("phevaluator")

from aof_equity import (
    equity_hu_exact,
    equity_vs_range,
    equity_vs_two_ranges_adaptive,
    warmup,
)
from aof_validation.equity_assert import assert_equity_close, assert_equity_result_close
from aof_equity_config import equity_vs_ranges_adaptive

# Seeded MC at max_combos=64 — assert repeatability, not external chart pins.
EQUITY_RANGE_CASES = [
    ("AsAd", "KK", 80_000, 42),
    ("AsKs", "QQ", 80_000, 43),
    ("AhKd", "22", 80_000, 44),
    ("8h8d", "77", 80_000, 45),
    ("7h2d", "AA", 80_000, 46),
]

# Combo-level HU exact (deterministic MC over all boards).
HU_EXACT_CASES = [
    (("As", "Ad"), ("Kh", "Kd"), 0.8196, 0.015, 80_000, 42),
    (("8h", "8d"), ("7h", "7d"), 0.824, 0.02, 80_000, 45),
]

# 3-way exact: one combo per villain.
REF_AA_VS_KK_QQ_3WAY_EXACT = 0.675125


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    warmup()


@pytest.mark.parametrize(
    "hero,range_str,samples,seed",
    EQUITY_RANGE_CASES,
    ids=[c[0] + "_vs_" + c[1] for c in EQUITY_RANGE_CASES],
)
def test_equity_vs_range_deterministic_mc(
    hero: str,
    range_str: str,
    samples: int,
    seed: int,
) -> None:
    got = equity_vs_range(hero, range_str, samples, seed, max_combos=64)
    got2 = equity_vs_range(hero, range_str, samples, seed, max_combos=64)
    assert_equity_close(
        got2, got, 1e-9, method="mc", samples=samples, seed=seed, note="repeat"
    )
    assert 0.0 <= got <= 1.0


@pytest.mark.parametrize(
    "hero_cards,villain_cards,ref,tol,samples,seed",
    HU_EXACT_CASES,
    ids=[f"{c[0]}_vs_{c[1]}" for c in HU_EXACT_CASES],
)
def test_equity_hu_exact_combo(
    hero_cards: tuple[str, str],
    villain_cards: tuple[str, str],
    ref: float,
    tol: float,
    samples: int,
    seed: int,
) -> None:
    eq = equity_hu_exact(hero_cards, villain_cards, samples, seed)
    assert_equity_close(
        eq, ref, tol, method="exact", samples=samples, seed=seed, note="hu_exact"
    )


def test_equity_vs_two_ranges_exact_combos() -> None:
    eq, meta = equity_vs_two_ranges_adaptive(
        "AsAh",
        "KhKd",
        "QhQd",
        8000,
        42,
        max_combos=8,
        return_meta=True,
    )
    assert meta["method"] == "exact"
    assert_equity_close(
        eq,
        REF_AA_VS_KK_QQ_3WAY_EXACT,
        1e-5,
        method=meta["method"],
        samples=meta["samples_used"],
        seed=42,
        valid_combo_products=meta["valid_products"],
    )
    result = equity_vs_ranges_adaptive("AsAh", ["KhKd", "QhQd"], 8000, seed=42)
    assert_equity_result_close(
        result, REF_AA_VS_KK_QQ_3WAY_EXACT, 1e-5, expect_method="exact"
    )


@pytest.mark.slow
def test_equity_vs_range_chart_reference_aa_vs_kk() -> None:
    """External chart ref; subsampled range MC — slow convergence."""
    got = equity_vs_range("AsAd", "KK", 200_000, 42, max_combos=96)
    assert_equity_close(
        got, 0.8196, 0.02, method="mc", samples=200_000, seed=42, note="chart ref"
    )
