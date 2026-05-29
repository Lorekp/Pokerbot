"""Priority 3: call + players behind uses blind-aware pot terms."""

from __future__ import annotations

import pytest

pytest.importorskip("numba")
pytest.importorskip("phevaluator")
pytest.importorskip("pokerkit")

from aof import advise_call_vs_shove, call_ev_static_profile
from aof_equity import warmup
from aof_multiway import multiway_call_terms
from aof_parse import parse_spot_line

pytestmark = pytest.mark.usefixtures("aof_warmup")


@pytest.fixture(scope="module")
def aof_warmup() -> None:
    warmup()


def test_behind_pot_terms_bb_hero() -> None:
    """BB calls UTG with BTN behind — 3-way pot if BTN calls."""
    terms = multiway_call_terms(
        "BB",
        8.0,
        villain_positions=("UTG", "BTN"),
        table_size=4,
    )
    assert terms.total_pot_after_call == pytest.approx(24.5, abs=0.01)
    assert terms.hero_total_invested == 8.0


def test_call_behind_detail_prefix() -> None:
    spot = parse_spot_line("99 SB vs BTN behind 1 8bb")
    _, detail = advise_call_vs_shove(spot, samples=4000, seed=3)
    assert detail.startswith(
        ("approx: behind conditional:", "approx fallback: behind static:")
    )


def test_call_behind_deterministic() -> None:
    spot = parse_spot_line("AKs SB vs BTN behind 1 8bb")
    a, _, _, _, _ = call_ev_static_profile(spot, samples=5000, seed=11)
    b, _, _, _, _ = call_ev_static_profile(spot, samples=5000, seed=11)
    assert a == pytest.approx(b, abs=1e-9)
