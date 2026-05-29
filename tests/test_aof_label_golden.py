"""Priority 7: advisor label golden cases (no misleading Nash labels)."""

from __future__ import annotations

import re

import pytest

pytest.importorskip("numba")
pytest.importorskip("phevaluator")
pytest.importorskip("pokerkit")

from aof import advise_call_vs_shove, advise_static_profile
from aof_equity import warmup
from aof_parse import parse_spot_line

pytestmark = pytest.mark.usefixtures("aof_warmup")


@pytest.fixture(scope="module")
def aof_warmup() -> None:
    warmup()


SAMPLES = 8000
SEED = 42


def test_hu_solver_open_prefix() -> None:
    spot = parse_spot_line("AA BB vs SB 2 8bb")
    action, detail = advise_call_vs_shove(spot, samples=SAMPLES, seed=SEED)
    assert action in ("ALL IN", "FOLD")
    assert detail.startswith("hu-solver:")


def test_static_profile_open_prefix() -> None:
    spot = parse_spot_line("AKo UTG 8bb")
    action, detail = advise_static_profile(spot, samples=SAMPLES, seed=SEED)
    assert action in ("ALL IN", "FOLD")
    assert detail.startswith("static-profile:")
    assert "Nash" not in detail


def test_facing_two_prefix() -> None:
    spot = parse_spot_line("Ks4d BB vs UTG,BTN 8bb")
    _, detail = advise_call_vs_shove(spot, samples=SAMPLES, seed=SEED)
    assert detail.startswith(
        ("3way-mc+conditional:", "approx fallback static 2-shover:")
    )
    assert "Nash" not in detail


def test_call_behind_prefix() -> None:
    spot = parse_spot_line("99 SB vs BTN behind 1 8bb")
    _, detail = advise_call_vs_shove(spot, samples=SAMPLES, seed=SEED)
    assert detail.startswith(
        ("approx: behind conditional:", "approx fallback: behind static:")
    )
    assert "Nash" not in detail


def test_no_nash_in_user_facing_open_details() -> None:
    for line in ("AKo UTG 8bb", "77 SB 8bb", "TT BTN 3 8bb"):
        spot = parse_spot_line(line)
        _, detail = advise_static_profile(spot, samples=SAMPLES, seed=SEED)
        assert "Nash" not in detail
        assert re.search(r"\bnash\b", detail, re.I) is None
