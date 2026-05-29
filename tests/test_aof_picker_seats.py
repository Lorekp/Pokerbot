"""Seat normalization for AoF picker facing spots."""

from __future__ import annotations

import pytest

from aof_parse import _infer_players_behind
from aof_spot import build_spot_line


def test_infer_players_behind_3max_btn_vs_sb() -> None:
    n, behind = _infer_players_behind("SB", "BTN", 3)
    assert n == 1
    assert behind == ("BB",)


def test_infer_players_behind_rejects_shover_after_hero() -> None:
    with pytest.raises(ValueError, match="must act before"):
        _infer_players_behind("SB", "SB", 3)


def test_parse_facing_two_shorthand() -> None:
    from aof_parse import parse_spot_line

    spot = parse_spot_line("Ks4d BB vs UTG,BTN 8bb")
    assert spot.second_shover == "BTN"
    assert spot.shover == "UTG"
    assert spot.position == "BB"


def test_build_spot_line_3max_facing() -> None:
    line = build_spot_line(
        card_a="Ks",
        card_b="4d",
        hero_pos="SB",
        stack_bb=8.0,
        facing_shove=True,
        shover_pos="BTN",
        table_size=3,
    )
    assert "Ks4d SB vs BTN" in line
    assert "3" in line
    assert "8bb" in line


def test_build_spot_line_facing_two() -> None:
    line = build_spot_line(
        card_a="Ks",
        card_b="4d",
        hero_pos="BB",
        stack_bb=8.0,
        facing_shove=True,
        shover_pos="UTG",
        second_shover_pos="BTN",
        table_size=4,
    )
    assert "vs UTG,BTN" in line


def test_build_spot_line_facing_three() -> None:
    line = build_spot_line(
        card_a="Ks",
        card_b="4d",
        hero_pos="BB",
        stack_bb=8.0,
        facing_shove=True,
        shover_pos="UTG",
        second_shover_pos="BTN",
        third_shover_pos="SB",
        table_size=4,
    )
    assert line == "Ks4d BB vs UTG,BTN,SB 8bb"
