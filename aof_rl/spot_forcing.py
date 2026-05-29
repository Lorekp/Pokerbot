"""Deal hands until hero decision matches curriculum spot constraint."""

from __future__ import annotations

import random
from typing import Callable

from aof_ranges import active_seats
from aof_rl.curriculum import CurriculumStage, SpotConstraint
from aof_rl.env import (
    ACTION_ALL_IN,
    ACTION_FOLD,
    AoFHandState,
    reset_hand,
    step_hand,
)
from aof_rl.spot_diagnostics import all_ins_before_hero, classify_decision_spot


def _spot_matches(constraint: SpotConstraint, kind: str, n_prior: int) -> bool:
    if constraint == "any" or constraint == "full":
        return True
    if constraint == "open_or_facing":
        return kind in ("open", "call_clean", "call_behind")
    if constraint == "max_one_shover":
        return n_prior <= 1
    if constraint == "min_two_shovers":
        return n_prior >= 2
    return True


def reset_hand_for_curriculum(
    rng: random.Random,
    stage: CurriculumStage,
    *,
    stack_bb: float = 8.0,
    max_retries: int = 64,
    opponent_action: Callable[[AoFHandState, str], int] | None = None,
) -> tuple[AoFHandState, str]:
    """
    Return (state, hero_seat) where hero is next to act and spot matches stage.
    Opponent seats before hero are filled using opponent_action (default: fold).
    """
    table_size = rng.choice(stage.table_sizes)
    seats = active_seats(table_size)

    def _default_opp(st: AoFHandState, seat: str) -> int:
        if st.action_seat != seat:
            return ACTION_FOLD
        prior = all_ins_before_hero(st, seat)
        if stage.spot_constraint == "min_two_shovers" and len(prior) < 2:
            return ACTION_ALL_IN
        if stage.spot_constraint == "max_one_shover" and len(prior) == 0:
            if rng.random() < 0.35:
                return ACTION_ALL_IN
        return ACTION_FOLD

    opp_fn = opponent_action or _default_opp

    for attempt in range(max_retries):
        state = reset_hand(rng, table_size=table_size, stack_bb=stack_bb)
        hero_seat = rng.choice(seats)
        guard = 0
        while not state.terminal and state.action_seat != hero_seat:
            guard += 1
            if guard > 32:
                break
            seat = state.action_seat
            if seat is None:
                break
            if state.seats.index(seat) >= state.seats.index(hero_seat):
                break
            action = opp_fn(state, seat)
            state = step_hand(
                state,
                action,
                payoff_samples=200,
                payoff_seed=rng.randint(0, 10_000) + attempt,
            )
        if state.terminal or state.action_seat != hero_seat:
            continue
        kind = classify_decision_spot(state, hero_seat)
        n_prior = len(all_ins_before_hero(state, hero_seat))
        if _spot_matches(stage.spot_constraint, kind, n_prior):
            return state, hero_seat

    state = reset_hand(rng, table_size=table_size, stack_bb=stack_bb)
    return state, state.action_seat or seats[0]
