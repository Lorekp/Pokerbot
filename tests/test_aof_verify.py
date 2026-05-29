"""Invariant checks for AoF table sizes, HU mode, and spot round-trips."""

from __future__ import annotations

import pytest

pytest.importorskip("numba")
pytest.importorskip("phevaluator")
pytest.importorskip("pokerkit")

from aof import _villains_behind, evaluate_spot
from aof_parse import parse_spot_line
from aof_ranges import (
    ACTION_ORDER_4,
    active_seats,
    position_aliases,
    seat_cycle_clockwise,
)
from aof_spot import build_spot_line

pytestmark = pytest.mark.usefixtures("aof_warmup")


@pytest.fixture(scope="module")
def aof_warmup() -> None:
    from aof_equity import warmup

    warmup()


class TestSeatInvariants:
    @pytest.mark.parametrize(
        "n,expected",
        [
            (2, ("SB", "BB")),
            (3, ("BTN", "SB", "BB")),
            (4, ACTION_ORDER_4),
        ],
    )
    def test_active_seat_count_matches_table(self, n: int, expected: tuple[str, ...]) -> None:
        seats = active_seats(n)
        assert seats == expected
        assert len(seats) == n

    def test_hu_aliases_button_to_sb(self) -> None:
        aliases = position_aliases(2)
        assert aliases["BTN"] == "SB"
        assert aliases["BUTTON"] == "SB"

    def test_seat_cycles_are_permutations(self) -> None:
        for n in (2, 3, 4):
            cycle = seat_cycle_clockwise(n)
            seats = set(active_seats(n))
            assert set(cycle) == seats
            assert len(cycle) == len(seats)


class TestVillainsBehind:
    def test_hu_sb_open(self) -> None:
        assert _villains_behind("SB", 2) == ["BB"]

    def test_hu_bb_open(self) -> None:
        assert _villains_behind("BB", 2) == []

    def test_three_max_btn_open(self) -> None:
        assert _villains_behind("BTN", 3) == ["SB", "BB"]

    def test_four_max_utg_open(self) -> None:
        assert _villains_behind("UTG", 4) == ["BTN", "SB", "BB"]


class TestInferBehind:
    @pytest.mark.parametrize(
        "line,n_behind",
        [
            ("AA BB vs SB hu 8bb", 0),
            ("AA BB vs SB 2 8bb", 0),
            ("AJo SB vs BTN 3 8bb", 1),
            ("AA BB vs BTN", 0),
            ("AJo SB vs BTN behind 1", 1),
        ],
    )
    def test_players_behind(self, line: str, n_behind: int) -> None:
        from aof_parse import CallSpot

        spot = parse_spot_line(line)
        assert isinstance(spot, CallSpot)
        assert spot.players_behind == n_behind


class TestSpotRoundTrip:
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(
                card_a="As",
                card_b="Kh",
                hero_pos="BTN",
                stack_bb=8,
                facing_shove=False,
                table_size=4,
            ),
            dict(
                card_a="As",
                card_b="Kh",
                hero_pos="SB",
                stack_bb=8,
                facing_shove=False,
                table_size=2,
            ),
            dict(
                card_a="As",
                card_b="Kh",
                hero_pos="BB",
                stack_bb=8,
                facing_shove=True,
                shover_pos="SB",
                table_size=2,
            ),
            dict(
                card_a="As",
                card_b="Kh",
                hero_pos="SB",
                stack_bb=8,
                facing_shove=False,
                table_size=3,
            ),
        ],
    )
    def test_build_parse_roundtrip(self, kwargs: dict) -> None:
        line = build_spot_line(**kwargs)
        spot = parse_spot_line(line)
        assert spot.stack_bb == 8.0
        assert spot.position == kwargs["hero_pos"]
        assert spot.table_size == kwargs["table_size"]


class TestEvaluateSmoke:
    def test_hu_open_sb(self) -> None:
        action, _ = evaluate_spot(
            "AKo SB hu 8bb", nash=True, force_call=False, samples=2000, seed=1
        )
        assert action in ("ALL IN", "FOLD")

    def test_hu_facing_bb(self) -> None:
        action, detail = evaluate_spot(
            "AA BB vs SB hu 8bb",
            nash=True,
            force_call=True,
            samples=2000,
            seed=1,
        )
        assert action in ("ALL IN", "FOLD")
        assert detail.startswith(("static-profile:", "hu-solver:"))

    def test_three_max_open(self) -> None:
        action, _ = evaluate_spot(
            "77 BTN 3 8bb", nash=True, force_call=False, samples=2000, seed=1
        )
        assert action in ("ALL IN", "FOLD")
