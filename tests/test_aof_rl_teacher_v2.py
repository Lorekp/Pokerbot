"""Priority 6C: RL teacher v2 (Facing 2 + call-behind)."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("numba")

from aof_equity import warmup
from aof_parse import CallSpot
from aof_rl.env import ACTION_ALL_IN, reset_hand, step_hand
from aof_rl.spot_diagnostics import (
    classify_decision_spot,
    record_decision,
    teacher_path_for_spot,
)
from aof_rl.teacher import NashTeacherCache, build_spot_for_state


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    warmup()


def _two_shover_state():
    rng = random.Random(0)
    state = reset_hand(rng, table_size=4, stack_bb=8.0)
    utg, btn, sb, _bb = state.seats
    for seat in (utg, btn):
        state = step_hand(state, ACTION_ALL_IN, payoff_samples=200, payoff_seed=1)
    return state, sb


def test_v1_facing_two_still_missing_path() -> None:
    state, hero = _two_shover_state()
    kind = classify_decision_spot(state, hero)
    assert kind == "facing_two"
    assert teacher_path_for_spot(kind, teacher_version=1) == (
        "static_profile_facing_two_missing"
    )


def test_v2_builds_callspot_with_second_shover() -> None:
    state, hero = _two_shover_state()
    spot = build_spot_for_state(state, hero, teacher_version=2)
    assert isinstance(spot, CallSpot)
    assert spot.second_shover is not None
    assert spot.shover == state.seats[0]
    assert spot.second_shover == state.seats[1]


def test_v2_valid_teacher_action_two_shover() -> None:
    state, hero = _two_shover_state()
    cache = NashTeacherCache(teacher_version=2, use_fast=False)
    result = cache.get_with_path(state, hero, samples=2000, seed=42)
    assert result.teacher.action in ("ALL IN", "FOLD")
    assert result.path in (
        "static_profile_facing_two_conditional",
        "static_profile_facing_two_fallback",
    )


def test_v2_teacher_path_explicit() -> None:
    state, hero = _two_shover_state()
    from aof_rl.spot_diagnostics import SpotDiagnostics

    diag = SpotDiagnostics()
    cache = NashTeacherCache(teacher_version=2)
    result = cache.get_with_path(state, hero, samples=1500, seed=3)
    kind, path = record_decision(
        diag,
        state,
        hero,
        teacher_path=result.path,
        teacher_version=2,
    )
    assert kind == "facing_two"
    assert "facing_two" in path


def test_v2_call_behind_uses_improved_spot() -> None:
    rng = random.Random(5)
    state = reset_hand(rng, table_size=4, stack_bb=8.0)
    btn = state.seats[1]
    state = step_hand(state, ACTION_ALL_IN, payoff_samples=100, payoff_seed=1)
    hero = state.seats[2]
    assert state.players_behind(hero)
    spot = build_spot_for_state(state, hero, teacher_version=2)
    assert isinstance(spot, CallSpot)
    assert spot.players_behind >= 1
    cache = NashTeacherCache(teacher_version=2)
    result = cache.get_with_path(state, hero, samples=1500, seed=11)
    assert result.path == "static_profile_call_behind"


def test_v2_facing_three_build_and_missing_teacher() -> None:
    rng = random.Random(3)
    state = reset_hand(rng, table_size=4, stack_bb=8.0)
    utg, btn, sb, bb = state.seats
    for seat in (utg, btn, sb):
        state = step_hand(state, ACTION_ALL_IN, payoff_samples=200, payoff_seed=1)
    hero = bb
    kind = classify_decision_spot(state, hero)
    assert kind == "facing_three"
    spot = build_spot_for_state(state, hero, teacher_version=2)
    assert spot.third_shover == sb
    assert spot.shovers == (utg, btn, sb)
    cache = NashTeacherCache(teacher_version=2)
    result = cache.get_with_path(state, hero, samples=1500, seed=7)
    assert result.path == "static_profile_facing_three_missing"


def test_v2_deterministic_teacher() -> None:
    state, hero = _two_shover_state()
    cache = NashTeacherCache(teacher_version=2)
    a = cache.get_with_path(state, hero, samples=3000, seed=99)
    cache.clear()
    b = cache.get_with_path(state, hero, samples=3000, seed=99)
    assert a.teacher.action == b.teacher.action
    assert a.teacher.ev_primary == pytest.approx(b.teacher.ev_primary, abs=1e-9)
