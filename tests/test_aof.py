"""Smoke tests for AoF advisor (parser, call-vs-shove branches, anchors)."""

from __future__ import annotations

import pytest

pytest.importorskip("numba")
pytest.importorskip("phevaluator")
pytest.importorskip("pokerkit")

from aof import (
    NashTeacher,
    advise_call_vs_shove,
    nash_teacher_open,
    advise_nash,
    advice_display_lines,
    marginal_suffix,
    parse_marginal,
    shove_ev_nash,
)
from aof_equity import warmup
from aof import _villains_behind
from aof_ranges import ACTION_ORDER_4, active_seats
from aof_parse import CallSpot, OpenSpot, parse_spot, parse_spot_line

pytestmark = pytest.mark.usefixtures("aof_warmup")

MC = 8000
SEED = 42


@pytest.fixture(scope="module")
def aof_warmup() -> None:
    warmup()


class TestShortHanded:
    def test_active_seats(self) -> None:
        assert active_seats(2) == ("SB", "BB")
        assert active_seats(3) == ("BTN", "SB", "BB")
        assert active_seats(4) == ACTION_ORDER_4

    def test_three_max_open_btn(self) -> None:
        s = parse_spot_line("AJo BTN 3 8bb")
        assert isinstance(s, OpenSpot)
        assert s.table_size == 3
        assert s.position == "BTN"
        assert _villains_behind("BTN", 3) == ["SB", "BB"]

    def test_three_max_no_utg(self) -> None:
        with pytest.raises(ValueError, match="Unknown position"):
            parse_spot_line("AJo UTG 3 8bb")

    def test_two_max_facing(self) -> None:
        s = parse_spot_line("AA BB vs SB 2 8bb")
        assert isinstance(s, CallSpot)
        assert s.players_behind == 0
        assert s.shover == "SB"

    def test_hu_shorthand_token(self) -> None:
        s = parse_spot_line("AKo SB hu 8bb")
        assert isinstance(s, OpenSpot)
        assert s.table_size == 2
        assert s.position == "SB"

    def test_hu_btn_alias(self) -> None:
        s = parse_spot_line("AKo BTN hu 8bb")
        assert isinstance(s, OpenSpot)
        assert s.position == "SB"


class TestParserOpen:
    def test_four_max_has_four_seats(self) -> None:
        assert len(ACTION_ORDER_4) == 4
        assert "CO" not in ACTION_ORDER_4

    def test_co_alias_maps_to_btn_in_four_max(self) -> None:
        s = parse_spot_line("AJo CO 8bb")
        assert isinstance(s, OpenSpot)
        assert s.position == "BTN"
        assert s.table_size == 4

    def test_minimal(self) -> None:
        s = parse_spot_line("AJo BTN")
        assert isinstance(s, OpenSpot)
        assert s.position == "BTN"
        assert s.stack_bb == 8.0
        assert s.table_size == 4

    def test_stack_and_table(self) -> None:
        s = parse_spot_line("KQs CO 6 5bb")
        assert isinstance(s, OpenSpot)
        assert s.table_size == 6
        assert s.stack_bb == 5.0

    def test_invalid_hand(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_spot_line("xx BTN")

    def test_too_short(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            parse_spot_line("AJo")

    def test_bb_btn_without_vs_is_open(self) -> None:
        with pytest.raises(ValueError, match="Unexpected"):
            parse_spot_line("AJo BB BTN")


class TestParserCall:
    def test_vs_syntax(self) -> None:
        s = parse_spot_line("AA BB vs BTN")
        assert isinstance(s, CallSpot)
        assert s.position == "BB"
        assert s.shover == "BTN"
        assert s.players_behind == 0

    def test_call_syntax(self) -> None:
        s = parse_spot_line("99 SB call BTN 8bb")
        assert isinstance(s, CallSpot)
        assert s.shover == "BTN"
        assert s.stack_bb == 8.0

    def test_behind_explicit(self) -> None:
        s = parse_spot_line("AJo SB vs BTN behind 1")
        assert isinstance(s, CallSpot)
        assert s.players_behind == 1

    def test_force_call_flag(self) -> None:
        with pytest.raises(ValueError, match="shover"):
            parse_spot_line("AJo BB", force_call=True)

    def test_shover_after_hero_fails(self) -> None:
        with pytest.raises(ValueError, match="before hero"):
            parse_spot_line("AJo UTG vs BB")

    def test_missing_shover(self) -> None:
        with pytest.raises(ValueError, match="shover"):
            parse_spot_line("AA BB vs")


class TestCaseSelector:
    def test_clean_exact_label(self) -> None:
        spot = parse_spot_line("AA BB vs BTN")
        assert isinstance(spot, CallSpot)
        _, detail = advise_call_vs_shove(spot, samples=MC, seed=SEED)
        assert detail.startswith(
            ("static-profile:", "hu-solver:", "approx: behind conditional:")
        )

    def test_messy_approx_label(self) -> None:
        spot = parse_spot_line("AJo SB vs BTN behind 1")
        assert isinstance(spot, CallSpot)
        assert spot.players_behind == 1
        _, detail = advise_call_vs_shove(spot, samples=MC, seed=SEED)
        assert "approx:" in detail
        assert "1 behind" in detail


class TestAnchorsCall:
    def test_aa_beats_ajo_vs_btn(self) -> None:
        aa = parse_spot_line("AA BB vs BTN 8bb")
        ajo = parse_spot_line("AJo BB vs BTN 8bb")
        assert isinstance(aa, CallSpot) and isinstance(ajo, CallSpot)
        _, d_aa = advise_call_vs_shove(aa, samples=MC, seed=SEED)
        _, d_ajo = advise_call_vs_shove(ajo, samples=MC, seed=SEED)
        ev_aa = float(d_aa.split("callEV=")[1].split("bb")[0])
        ev_ajo = float(d_ajo.split("callEV=")[1].split("bb")[0])
        assert ev_aa > ev_ajo

    def test_72o_worse_than_fold_vs_utg(self) -> None:
        spot = parse_spot_line("72o BB vs UTG 8bb")
        assert isinstance(spot, CallSpot)
        action, detail = advise_call_vs_shove(spot, samples=MC, seed=SEED)
        ev_call = float(detail.split("callEV=")[1].split("bb")[0])
        ev_fold = float(detail.split("foldEV=")[1].split("bb")[0])
        assert ev_call <= ev_fold
        assert action == "FOLD"


class TestNashTeacherApi:
    def test_nash_teacher_open(self) -> None:
        spot = OpenSpot("AJo", ("Ah", "Jd"), "BTN", 8.0, 4)
        t = nash_teacher_open(spot, samples=500, seed=1)
        assert isinstance(t, NashTeacher)
        assert t.action in ("ALL IN", "FOLD")
        assert not t.facing


class TestMarginalFlag:
    def test_suffix_when_thin(self) -> None:
        s = marginal_suffix("ALL IN", 0.10, 0.05)
        assert "marginal win" in s
        assert parse_marginal("x" + s) is not None

    def test_no_suffix_when_clear(self) -> None:
        assert marginal_suffix("FOLD", 0.0, -1.0) == ""

    def test_display_lines(self) -> None:
        detail = (
            "static-profile: eq 50.0% need 40.0% callEV=+0.12bb foldEV=+0.05bb "
            f"(vs BTN){marginal_suffix('ALL IN', 0.12, 0.05)}"
        )
        lines = advice_display_lines("ALL IN", detail)
        assert len(lines) == 3
        assert lines[1].startswith("⚠ Marginal")


class TestOpenUnchanged:
    def test_advise_nash_returns_action(self) -> None:
        spot = parse_spot("AJo BTN")
        action, _ = advise_nash(spot, samples=MC, seed=SEED)
        assert action in ("ALL IN", "FOLD")

    def test_shove_ev_tuple(self) -> None:
        spot = parse_spot("77 SB 8")
        ev_s, ev_f, detail = shove_ev_nash(spot, samples=MC, seed=SEED)
        assert "shoveEV=" in detail
        assert isinstance(ev_s, float) and isinstance(ev_f, float)
