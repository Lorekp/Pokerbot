"""Priority 6A: RL spot classification and teacher-path logging."""

from __future__ import annotations

import random

from aof_rl.env import ACTION_ALL_IN, reset_hand, step_hand
from aof_rl.spot_diagnostics import (
    SpotDiagnostics,
    classify_decision_spot,
    record_decision,
    teacher_path_for_spot,
)


def test_facing_two_classified() -> None:
    rng = random.Random(0)
    state = reset_hand(rng, table_size=4, stack_bb=8.0)
    order = state.seats
    utg, btn, sb, bb = order
    for seat in (utg, btn):
        state = step_hand(state, ACTION_ALL_IN, payoff_samples=200, payoff_seed=1)
    hero = sb
    assert state.action_seat == hero
    kind = classify_decision_spot(state, hero)
    assert kind == "facing_two"
    assert teacher_path_for_spot(kind) == "static_profile_facing_two_missing"


def test_facing_three_classified_and_teacher_missing() -> None:
    rng = random.Random(2)
    state = reset_hand(rng, table_size=4, stack_bb=8.0)
    utg, btn, sb, bb = state.seats
    for seat in (utg, btn, sb):
        state = step_hand(state, ACTION_ALL_IN, payoff_samples=200, payoff_seed=1)
    hero = bb
    assert state.action_seat == hero
    kind = classify_decision_spot(state, hero)
    assert kind == "facing_three"
    assert (
        teacher_path_for_spot(kind, teacher_version=2)
        == "static_profile_facing_three_missing"
    )


def test_record_decision_json() -> None:
    diag = SpotDiagnostics()
    rng = random.Random(1)
    state = reset_hand(rng, table_size=4)
    record_decision(diag, state, state.seats[0])
    data = diag.to_dict()
    assert "spot_counts" in data
    assert data["spot_counts"].get("open", 0) >= 1
