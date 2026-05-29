"""Facing 2 (two prior shovers) advisor accuracy tests."""

from __future__ import annotations

import re

import pytest

from aof import FACING2_DEFAULT_SAMPLES, advise_call_vs_shove, call_ev_two_shovers
from aof_equity import equity_vs_two_ranges, equity_vs_two_ranges_adaptive, warmup
from aof_multiway import multiway_call_terms
from aof_parse import CallSpot, parse_spot_line
from aof_ranges import (
    CALL_RANGE_STR,
    SECOND_JAM_RANGE_STR,
    has_call_chart,
    has_second_jam_chart,
    lookup_facing_two_ranges,
    lookup_shove,
)


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    warmup()


def test_multiway_breakeven_bb_8bb() -> None:
    terms = multiway_call_terms(
        "BB", 8.0, villain_positions=("UTG", "BTN"), table_size=4
    )
    # SB folds (0.5bb); hero + two villains full stack = 8+8+8+0.5 = 24.5
    assert terms.total_pot_after_call == pytest.approx(24.5, abs=0.01)
    assert terms.hero_total_invested == pytest.approx(8.0)
    assert terms.breakeven_equity == pytest.approx(8.0 / 24.5, rel=0.01)


def test_multiway_breakeven_sb_hero_not_hu_pot_odds() -> None:
    terms = multiway_call_terms(
        "SB", 8.0, villain_positions=("UTG", "BTN"), table_size=4
    )
    hu_need = (8.0 - 0.5) / (1.5 + 2 * 8.0)
    assert terms.breakeven_equity != pytest.approx(hu_need, rel=0.05)
    assert terms.breakeven_equity == pytest.approx(
        terms.hero_total_invested / terms.total_pot_after_call, rel=1e-6
    )


def test_card_blocking_in_adaptive_equity() -> None:
    # Identical ranges force many blocked pairs; result stays valid.
    rng = "AA"
    eq = equity_vs_two_ranges_adaptive("AsAh", rng, rng, 2000, 99)
    assert 0.0 <= eq <= 1.0


def test_conditional_second_range_utg_btn() -> None:
    ft = lookup_facing_two_ranges("UTG", "BTN", 8.0, table_size=4)
    assert ft.second_source == "conditional"
    assert has_second_jam_chart("UTG", "BTN", 8.0, table_size=4)
    static_btn, _ = lookup_shove("BTN", 8.0, table_size=4)
    second_jam = SECOND_JAM_RANGE_STR[8][("BTN", "UTG")]
    call_chart = CALL_RANGE_STR[8][("BTN", "UTG")]
    assert ft.second_range == second_jam
    assert ft.second_range != static_btn
    assert len(second_jam) >= len(call_chart)


def test_fallback_static_when_no_call_chart() -> None:
    from aof_ranges import lookup_second_shover_facing_jam

    # BTN jam then UTG reshove — no (UTG, BTN) continue chart
    assert not has_call_chart("BTN", "UTG", 8.0, table_size=4)
    _rng, src, _prov = lookup_second_shover_facing_jam(
        "BTN", "UTG", 8.0, table_size=4
    )
    assert src == "fallback_static"


def test_fallback_label_in_advice(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force static open-jam fallback when no call or second-jam charts.
    monkeypatch.setattr("aof_ranges.has_call_chart", lambda *a, **k: False)
    monkeypatch.setattr("aof_ranges._second_jam_map", lambda _stack: {})
    spot = parse_spot_line("Ks4d BB vs UTG,BTN 8bb")
    action, detail = advise_call_vs_shove(spot, samples=4000, seed=1)
    assert action in ("ALL IN", "FOLD")
    assert "fallback static" in detail


def test_single_facing_regression_unchanged() -> None:
    spot = parse_spot_line("AA BB vs BTN 8bb")
    assert spot.second_shover is None
    _a1, d1 = advise_call_vs_shove(spot, samples=8000, seed=42)
    _a2, d2 = advise_call_vs_shove(spot, samples=8000, seed=42)
    assert d1 == d2
    assert "need" in d1
    m = re.search(r"need ([\d.]+)%", d1)
    assert m is not None
    need = float(m.group(1))
    # HU pot odds ~ 7/17.5 ≈ 40%
    assert 35.0 < need < 45.0


def test_facing_two_deterministic() -> None:
    spot = parse_spot_line("Ks4d BB vs UTG,BTN 8bb")
    ft1 = call_ev_two_shovers(spot, samples=6000, seed=123)
    ft2 = call_ev_two_shovers(spot, samples=6000, seed=123)
    assert ft1.eq_display == pytest.approx(ft2.eq_display, abs=1e-4)
    assert ft1.ev_call == pytest.approx(ft2.ev_call, abs=1e-3)


def test_equity_adaptive_matches_two_ranges_on_toy() -> None:
    hero = "AsKh"
    ra = "QQ"
    rb = "JJ"
    seed = 77
    a = equity_vs_two_ranges_adaptive(hero, ra, rb, 5000, seed)
    b = equity_vs_two_ranges(hero, ra, rb, 5000, seed, max_combos=96)
    assert a == pytest.approx(b, abs=0.02)


def test_facing_two_detail_uses_3way_need() -> None:
    spot = parse_spot_line("AA BB vs UTG,BTN 8bb")
    _action, detail = advise_call_vs_shove(spot, samples=FACING2_DEFAULT_SAMPLES, seed=5)
    terms = multiway_call_terms("BB", 8.0, villain_positions=("UTG", "BTN"), table_size=4)
    m = re.search(r"need ([\d.]+)%", detail)
    assert m is not None
    need_pct = float(m.group(1))
    assert need_pct == pytest.approx(terms.breakeven_equity * 100, rel=0.02)
    assert "3way-mc+conditional:" in detail or "fallback" in detail
    assert "pot=" in detail


def test_call_spot_dataclass_second_shover() -> None:
    spot = CallSpot(
        hand_label="AA",
        hero_cards=("Ac", "Ad"),
        position="BB",
        shover="UTG",
        stack_bb=8.0,
        table_size=4,
        players_behind=0,
        behind_seats=(),
        second_shover="BTN",
    )
    ft = call_ev_two_shovers(spot, samples=3000, seed=0)
    assert ft.ranges.first_range
    assert ft.ranges.second_range
