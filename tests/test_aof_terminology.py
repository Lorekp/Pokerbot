"""Priority 1: static-profile labels and backward-compatible aliases."""

from __future__ import annotations

import pytest

pytest.importorskip("numba")
pytest.importorskip("phevaluator")
pytest.importorskip("pokerkit")

from aof import (
    AdvisorTeacher,
    NashTeacher,
    advise_static_profile,
    shove_ev_nash,
    shove_ev_static_profile,
    static_profile_teacher_open,
)
from aof_equity import warmup
from aof_parse import parse_spot_line

pytestmark = pytest.mark.usefixtures("aof_warmup")


@pytest.fixture(scope="module")
def aof_warmup() -> None:
    warmup()


def test_nash_teacher_alias() -> None:
    assert NashTeacher is AdvisorTeacher


def test_open_detail_static_profile_prefix() -> None:
    spot = parse_spot_line("AJo BTN 8bb")
    _, detail = advise_static_profile(spot, samples=4000, seed=1)
    assert detail.startswith("static-profile:") or detail.startswith("hu-solver:")


def test_shove_ev_aliases_match() -> None:
    spot = parse_spot_line("77 UTG 8bb")
    a = shove_ev_static_profile(spot, samples=3000, seed=2)
    b = shove_ev_nash(spot, samples=3000, seed=2)
    assert a[0] == pytest.approx(b[0], abs=0.01)
    assert "multiway" in a[2]


def test_teacher_open() -> None:
    spot = parse_spot_line("AKo SB hu 8bb")
    t = static_profile_teacher_open(spot, samples=400, seed=1)
    assert isinstance(t, AdvisorTeacher)
    assert t.action in ("ALL IN", "FOLD")
