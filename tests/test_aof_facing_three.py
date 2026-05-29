"""Facing 3 (three prior shovers) advisor tests."""

from __future__ import annotations

import re

import pytest

from aof import (
    FACING3_DEFAULT_SAMPLES,
    advise_call_vs_shove,
    call_ev_three_shovers,
    evaluate_spot_detailed,
)
from aof_equity import equity_vs_three_ranges_adaptive, warmup
from aof_multiway import multiway_call_terms
from aof_parse import CallSpot, parse_spot_line
from aof_ranges import lookup_facing_three_ranges


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    warmup()


def test_parse_facing_three_legal() -> None:
    spot = parse_spot_line("Ks4d BB vs UTG,BTN,SB 8bb")
    assert isinstance(spot, CallSpot)
    assert spot.shovers == ("UTG", "BTN", "SB")
    assert spot.third_shover == "SB"
    assert spot.players_behind == 0


def test_parse_facing_three_illegal_order() -> None:
    with pytest.raises(ValueError, match="preflop order"):
        parse_spot_line("Ks4d BB vs BTN,UTG,SB 8bb")


def test_parse_facing_three_duplicate_shover() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        parse_spot_line("Ks4d BB vs UTG,UTG,SB 8bb")


def test_parse_facing_three_table_size_rejected() -> None:
    with pytest.raises(ValueError, match="4-max"):
        parse_spot_line("Ks4d BB vs UTG,BTN,SB 8bb hu")


def test_parse_facing_three_illegal_hero() -> None:
    with pytest.raises(ValueError, match="preflop order"):
        parse_spot_line("Ks4d SB vs UTG,BTN,BB 8bb")


def test_multiway_pot_bb_vs_three_8bb() -> None:
    terms = multiway_call_terms(
        "BB",
        8.0,
        villain_positions=("UTG", "BTN", "SB"),
        table_size=4,
    )
    assert terms.gross_pot_after_call == pytest.approx(32.0, abs=0.01)
    assert terms.hero_total_invested == pytest.approx(8.0)
    assert terms.breakeven_equity == pytest.approx(0.25, rel=0.01)
    assert terms.ev_fold == pytest.approx(-1.0)


def test_facing_three_deterministic_mc() -> None:
    spot = parse_spot_line("Ks4d BB vs UTG,BTN,SB 8bb")
    a = call_ev_three_shovers(spot, samples=4000, seed=99)
    b = call_ev_three_shovers(spot, samples=4000, seed=99)
    assert a.eq_display == pytest.approx(b.eq_display, abs=1e-4)


def test_equity_three_ranges_tiny_exact() -> None:
    eq, meta = equity_vs_three_ranges_adaptive(
        "AsKh",
        ["AA", "KK", "QQ"],
        500,
        42,
        max_combos=32,
        return_meta=True,
    )
    assert 0.0 <= eq <= 1.0
    assert meta["method"] in ("exact", "mc")


def test_equity_blocked_combos_reduce_valid_products() -> None:
    _eq_open, meta_open = equity_vs_three_ranges_adaptive(
        "2c3d",
        ["AA", "KK", "QQ"],
        400,
        11,
        max_combos=48,
        return_meta=True,
    )
    _eq_blocked, meta_blocked = equity_vs_three_ranges_adaptive(
        "AsAh",
        ["AA", "KK", "QQ"],
        400,
        11,
        max_combos=48,
        return_meta=True,
    )
    assert meta_open["valid_products"] > meta_blocked["valid_products"]


def test_advisor_detail_4way_label() -> None:
    spot = parse_spot_line("Ks4d BB vs UTG,BTN,SB 8bb")
    action, detail = advise_call_vs_shove(
        spot, samples=FACING3_DEFAULT_SAMPLES, seed=7
    )
    assert action in ("ALL IN", "FOLD")
    assert "4way-" in detail
    assert "Nash" not in detail
    assert "need 25.0%" in detail or "need 25." in detail


def test_evaluate_spot_json_tier() -> None:
    res = evaluate_spot_detailed(
        "Ks4d BB vs UTG,BTN,SB 8bb",
        nash=True,
        force_call=False,
        samples=3000,
        seed=3,
    )
    assert res.confidence.approximation_tier.startswith("approx_facing_three")
    assert "4way-" in res.detail


def test_lookup_facing_three_ranges_note() -> None:
    ft = lookup_facing_three_ranges("UTG", "BTN", "SB", 8.0, table_size=4)
    assert "2nd=BTN|UTG" in ft.range_note
    assert "3rd=SB" in ft.range_note
    assert "third_jam_chart" in ft.range_note
    assert not ft.fallback_used


def test_advisor_facing_three_conditional_prefix() -> None:
    spot = parse_spot_line("Ks4d BB vs UTG,BTN,SB 8bb")
    _action, detail = advise_call_vs_shove(spot, samples=4000, seed=11)
    assert "4way-mc+conditional:" in detail or "4way-exact+conditional:" in detail
    assert "fallback_call_chart" not in detail


def test_single_facing_two_regression() -> None:
    spot = parse_spot_line("Ks4d BB vs UTG,BTN 8bb")
    assert spot.third_shover is None
    _a, detail = advise_call_vs_shove(spot, samples=4000, seed=1)
    assert "3way-" in detail or "4way-" not in detail.split("3way")[0]
