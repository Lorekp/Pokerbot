"""Fast deterministic equity / spot smoke tests (Priority 9)."""

from __future__ import annotations

import pytest

pytest.importorskip("pokerkit")
pytest.importorskip("phevaluator")

from aof import call_ev_two_shovers, shove_ev_open
from aof_equity import equity_vs_four_ranges_adaptive, warmup
from aof_equity_config import equity_vs_ranges_adaptive
from aof_parse import parse_spot_line
from aof_validation.equity_assert import assert_equity_close, assert_equity_result_close


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    warmup()


def test_three_way_exact_via_adaptive() -> None:
    result = equity_vs_ranges_adaptive(
        "AsAh", ["KhKd", "QhQd"], 4000, seed=77
    )
    assert result.method == "exact"
    result2 = equity_vs_ranges_adaptive("AsAh", ["KhKd", "QhQd"], 4000, seed=77)
    assert_equity_result_close(
        result2, result.equity, 1e-9, expect_method="exact", note="3-way repeat"
    )


def test_four_way_exact_tiny_product() -> None:
    villains = ["KhKd", "QhQd", "JhJd", "ThTd"]
    eq, meta = equity_vs_four_ranges_adaptive(
        "AsAh", villains, 2000, 11, max_combos=8, return_meta=True
    )
    assert meta["method"] == "exact"
    assert meta["valid_products"] == 1
    eq2, _ = equity_vs_four_ranges_adaptive(
        "AsAh", villains, 2000, 11, max_combos=8, return_meta=True
    )
    assert_equity_close(
        eq2, eq, 1e-12, method="exact", samples=meta["samples_used"], seed=11
    )
    assert 0.0 < eq < 1.0


def test_facing_two_ev_repeatable() -> None:
    spot = parse_spot_line("Ks4d BB vs UTG,BTN 8bb")
    a = call_ev_two_shovers(spot, samples=6000, seed=123)
    b = call_ev_two_shovers(spot, samples=6000, seed=123)
    assert a.eq_display == pytest.approx(b.eq_display, abs=1e-9)
    assert a.ev_call == pytest.approx(b.ev_call, abs=1e-9)


def test_call_behind_open_multiway_seeded_smoke() -> None:
    spot_call = parse_spot_line("99 SB vs BTN behind 1 8bb")
    spot_open = parse_spot_line("AKo BTN 8bb")
    ev_c, _, _ = shove_ev_open(spot_call, samples=4000, seed=31)
    ev_o, ef_o, _ = shove_ev_open(spot_open, samples=4000, seed=31)
    assert ev_c == pytest.approx(
        shove_ev_open(spot_call, samples=4000, seed=31)[0], abs=1e-9
    )
    assert ev_o > ef_o
