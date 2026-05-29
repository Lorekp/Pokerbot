"""Priority 2: open-shove EV uses true multiway equity for 2+ callers."""

from __future__ import annotations

import pytest

pytest.importorskip("numba")
pytest.importorskip("phevaluator")
pytest.importorskip("pokerkit")

from aof import _hero_key, shove_ev_open
from aof_equity import equity_vs_range, warmup
from aof_parse import parse_spot_line

pytestmark = pytest.mark.usefixtures("aof_warmup")


@pytest.fixture(scope="module")
def aof_warmup() -> None:
    warmup()


def _legacy_max_1v1_ev(spot, samples: int, seed: int) -> float:
    """Old shortcut: max of per-villain 1v1 equities for 2 callers."""
    from aof import _blind_state, _call_subset_weights, _lookup_call, _villains_behind

    posted, fold_win_pot = _blind_state(spot.position)
    invest = spot.stack_bb
    villains = _villains_behind(spot.position, spot.table_size)
    call_p, call_rng = [], []
    for v in villains:
        pct, rng = _lookup_call(
            spot.position, v, spot.stack_bb, table_size=spot.table_size
        )
        call_p.append(min(max(pct, 0.0), 0.99))
        call_rng.append(rng)
    p_all_fold = 1.0
    for p in call_p:
        p_all_fold *= 1.0 - p
    ev = p_all_fold * fold_win_pot
    weights = _call_subset_weights(call_p, model="independent")
    n = len(villains)
    eq_by_v = [
        equity_vs_range(
            _hero_key(spot.hero_cards), call_rng[i], samples, seed + i * 9973, 64
        )
        for i in range(n)
    ]
    for mask, w in weights:
        if w <= 0 or mask == 0:
            continue
        callers = [i for i in range(n) if mask & (1 << i)]
        k = len(callers)
        if k == 1:
            i = callers[0]
            pot = 1.5 + k * spot.stack_bb + (spot.stack_bb - posted)
            ev += w * (eq_by_v[i] * pot - invest)
        elif k >= 2:
            i, j = callers[0], callers[1]
            eq = eq_by_v[i] if call_p[i] >= call_p[j] else eq_by_v[j]
            pot = 1.5 + k * spot.stack_bb + (spot.stack_bb - posted)
            ev += w * (eq * pot - invest)
    return ev


def test_two_caller_branch_not_max_1v1() -> None:
    """UTG open: both BTN and SB call paths must differ from max-1v1 shortcut."""
    spot = parse_spot_line("AKo UTG 8bb")
    samples, seed = 6000, 99
    ev_new, _, detail = shove_ev_open(spot, samples=samples, seed=seed)
    ev_old = _legacy_max_1v1_ev(spot, samples, seed)
    assert "multiway" in detail
    assert ev_new != pytest.approx(ev_old, abs=0.001)


def test_open_ev_deterministic() -> None:
    spot = parse_spot_line("QJs CO 8bb")
    a, _, _ = shove_ev_open(spot, samples=5000, seed=7)
    b, _, _ = shove_ev_open(spot, samples=5000, seed=7)
    assert a == pytest.approx(b, abs=1e-9)
