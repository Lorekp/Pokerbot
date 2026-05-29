"""JSON / hand-state bridges for inference."""

from __future__ import annotations

import json
from typing import Any

from aof_rl.env import ACTION_ALL_IN, ACTION_FOLD, AoFHandState, reset_hand_from_holes, step_hand
from aof_rl.game_state import (
    AoFEncodeError,
    AoFGameState,
    Action,
    action_order_labels,
    game_state_from_dict,
    game_state_to_dict,
    index_to_card,
    seats_by_id,
)

__all__ = [
    "AoFEncodeError",
    "game_state_from_dict",
    "game_state_to_dict",
    "game_state_from_json",
    "game_state_to_hand_state",
]


def game_state_from_json(raw: str | bytes | dict[str, Any]) -> AoFGameState:
    if isinstance(raw, (str, bytes)):
        data = json.loads(raw)
    else:
        data = raw
    if not isinstance(data, dict):
        raise AoFEncodeError("game state JSON must be an object")
    return game_state_from_dict(data)


def game_state_to_hand_state(game: AoFGameState) -> tuple[AoFHandState, str]:
    """Reconstruct env hand state for teacher lookup (hero holes only)."""
    action_order = action_order_labels(game)
    if len(action_order) != game.table_size:
        raise AoFEncodeError("action order length mismatch")
    hero_label = action_order[game.hero_seat_id]
    c1, c2 = game.hero_cards
    holes = {hero_label: (index_to_card(c1), index_to_card(c2))}

    state = reset_hand_from_holes(
        action_order,
        holes,
        table_size=game.table_size,
        stack_bb=game.effective_stack_bb,
    )

    by_id = seats_by_id(game)
    for label, sid in enumerate(action_order):
        seat = by_id.get(sid)
        if seat is None:
            continue
        state.invested[label] = float(seat.committed_bb)

    for seat_id, act in game.action_history:
        if seat_id < 0 or seat_id >= len(action_order):
            raise AoFEncodeError(f"invalid seat_id in history: {seat_id}")
        label = action_order[seat_id]
        if state.action_seat != label:
            raise AoFEncodeError(
                f"action history order mismatch: expected {state.action_seat!r}, "
                f"history acts for {label!r}"
            )
        action = ACTION_FOLD if act == "FOLD" else ACTION_ALL_IN
        state = step_hand(state, action, payoff_samples=50, payoff_seed=0)

    if state.terminal:
        raise AoFEncodeError("game state is terminal")
    if state.action_seat != hero_label:
        raise AoFEncodeError(
            f"hero {hero_label!r} is not the acting seat ({state.action_seat!r})"
        )
    hero_seat = by_id[game.hero_seat_id]
    if hero_seat.state != "pending":
        raise AoFEncodeError(f"hero seat state must be pending, got {hero_seat.state!r}")
    return state, hero_label


def action_name(action_idx: int) -> Action:
    if action_idx == ACTION_FOLD:
        return "FOLD"
    if action_idx == ACTION_ALL_IN:
        return "ALL_IN"
    raise ValueError(f"invalid action index {action_idx}")
