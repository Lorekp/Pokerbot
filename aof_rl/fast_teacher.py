"""Low-latency Nash teacher for RL training (uses precomputed open EV + fast equity)."""

from __future__ import annotations

import re

from aof import NashTeacher, _teacher_from_evs, nash_teacher_call, shove_ev_nash
from aof_equity import equity_vs_range_fast
from aof_parse import CallSpot, OpenSpot
from aof_ranges import lookup_shove


def nash_teacher_open_fast(
    spot: OpenSpot,
    *,
    samples: int,
    seed: int,
    margin_bb: float = 0.4,
    min_samples: int = 200,
) -> NashTeacher:
    """Open teacher: one ``shove_ev_open`` (equities precomputed per villain inside)."""
    n = max(min_samples, samples)
    ev_shove, ev_fold, detail = shove_ev_nash(spot, samples=n, seed=seed)
    p_fold = None
    m = re.search(r"P_fold=([\d.]+)", detail)
    if m:
        p_fold = float(m.group(1))
    return _teacher_from_evs(ev_shove, ev_fold, p_fold=p_fold, facing=False)


def nash_teacher_call_fast(
    spot: CallSpot,
    *,
    samples: int,
    seed: int,
    margin_bb: float = 0.4,
    min_samples: int = 200,
) -> NashTeacher:
    """Call-vs-shove teacher using fast range equity when HU vs shover."""
    from aof import _call_pot_terms, _ev_call_fold

    if spot.players_behind > 0:
        return nash_teacher_call(spot, samples=samples, seed=seed)

    shove_rng, _ = lookup_shove(
        spot.shover, spot.stack_bb, table_size=spot.table_size
    )
    posted, _cc, pot_hu, _req_eq = _call_pot_terms(spot.position, spot.stack_bb)
    hero_key = "".join(spot.hero_cards)
    n = max(min_samples, samples)
    eq = equity_vs_range_fast(hero_key, shove_rng, n, seed, max_combos=64)
    ev_call, ev_fold = _ev_call_fold(eq, pot_hu, spot.stack_bb, posted)
    return _teacher_from_evs(ev_call, ev_fold, p_fold=None, facing=True)
