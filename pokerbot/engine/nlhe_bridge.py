"""Bridge JAX NLHE ``EnvState`` to Python ``GameState`` for WEVA encoding parity."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from pokerbot.engine.actions import (
    Action,
    ActionType,
    BET_SIZE_BUCKETS,
    index_to_bucket,
)
from pokerbot.engine.cards import Card, Rank, Suit
from pokerbot.engine.game import GameState, PlayerState, Position, Street
from pokerbot.engine.jax_engine import (
    ALL_IN,
    BET_BUCKET_START,
    CALL,
    CHECK,
    CHIP_UNIT,
    EnvState,
    FOLD,
    NUM_PLAYERS,
    SEAT_EMPTY,
    STREET_FLOP,
    STREET_PREFLOP,
    STREET_RIVER,
    STREET_TERMINAL,
    STREET_TURN,
)
from pokerbot.engine.side_pots_jax import SEAT_ALLIN, SEAT_FOLDED, SEAT_LIVE

_STREET_FROM_JAX = {
    STREET_PREFLOP: Street.PREFLOP,
    STREET_FLOP: Street.FLOP,
    STREET_TURN: Street.TURN,
    STREET_RIVER: Street.RIVER,
    STREET_TERMINAL: Street.SHOWDOWN,
}


def card_from_jax_index(idx: int) -> Card | None:
    if idx < 0:
        return None
    return Card(Rank(int(idx // 4 + 2)), Suit(int(idx % 4)))


def jax_action_to_python(action: int, amount_chips: int) -> Action:
    amount_bb = float(amount_chips) / float(CHIP_UNIT)
    if action == FOLD:
        return Action(ActionType.FOLD)
    if action == CHECK:
        return Action(ActionType.CHECK)
    if action == CALL:
        return Action(ActionType.CALL, amount_bb)
    if action == ALL_IN:
        return Action(ActionType.ALL_IN, amount_bb)
    if BET_BUCKET_START <= action < ALL_IN:
        frac = index_to_bucket(action)
        if frac is not None:
            atype = ActionType.BET
            return Action(atype, amount_bb)
    return Action(ActionType.CHECK)


def _position_for_seat(button: int, seat: int) -> Position:
    pos_order = [Position.BTN, Position.SB, Position.BB, Position.UTG, Position.HJ, Position.CO]
    offset = (seat - button) % NUM_PLAYERS
    return pos_order[offset]


def game_state_from_jax(
    state: EnvState | Any,
    *,
    spot_class: str = "",
    oop_range: str = "",
    ip_range: str = "",
) -> GameState:
    """Reconstruct a ``GameState`` snapshot matching JAX public/private fields."""
    holes = np.asarray(state.holes)
    board = np.asarray(state.board)
    seat_status = np.asarray(state.seat_status)
    stacks = np.asarray(state.stacks)
    committed_street = np.asarray(state.committed_street)
    committed_total = np.asarray(state.committed_total)
    button = int(np.asarray(state.button))
    street_j = int(np.asarray(state.street))
    current_bet = float(np.asarray(state.current_bet)) / float(CHIP_UNIT)
    pot_dead = float(np.asarray(state.pot_dead)) / float(CHIP_UNIT)
    pot_street = float(np.sum(committed_street)) / float(CHIP_UNIT)

    gs = GameState.new_hand(button_seat=button)
    gs._deck = None
    gs.street = _STREET_FROM_JAX.get(street_j, Street.SHOWDOWN)
    gs.current_bet_bb = current_bet
    gs.pot_bb = pot_dead + pot_street
    gs.action_seat = int(np.asarray(state.action_seat))
    gs.last_aggressor = int(np.asarray(state.last_aggressor))
    if gs.last_aggressor < 0:
        gs.last_aggressor = None

    for seat in range(NUM_PLAYERS):
        p = gs.players[seat]
        p.position = _position_for_seat(button, seat)
        if seat_status[seat] == SEAT_EMPTY:
            p.folded = True
            p.stack_bb = 0.0
        else:
            p.stack_bb = float(stacks[seat]) / float(CHIP_UNIT)
            p.committed_street_bb = float(committed_street[seat]) / float(CHIP_UNIT)
            p.committed_total_bb = float(committed_total[seat]) / float(CHIP_UNIT)
            p.folded = seat_status[seat] == SEAT_FOLDED
            p.all_in = seat_status[seat] == SEAT_ALLIN
        c0 = card_from_jax_index(int(holes[seat, 0]))
        c1 = card_from_jax_index(int(holes[seat, 1]))
        if c0 is not None and c1 is not None:
            p.hole = (c0, c1)

    gs.board = []
    for i in range(board.shape[0]):
        c = card_from_jax_index(int(board[i]))
        if c is not None:
            gs.board.append(c)

    hist_len = int(np.asarray(state.hist_len))
    gs._history = []
    gs._acted_this_street = set()
    for i in range(min(hist_len, int(state.hist_seat.shape[0]))):
        seat = int(state.hist_seat[i])
        act = int(state.hist_action[i])
        amt = int(state.hist_amount[i])
        gs._history.append((seat, jax_action_to_python(act, amt)))
        gs._acted_this_street.add(seat)

    _ = spot_class, oop_range, ip_range
    return gs


def numpy_env_state(state: EnvState) -> dict[str, np.ndarray]:
    """Materialize JAX state arrays for CPU traversal."""
    return {k: np.asarray(getattr(state, k)) for k in state._fields}


def env_state_from_numpy(d: dict[str, np.ndarray]) -> EnvState:
    return EnvState(**{k: jnp.asarray(v) for k, v in d.items()})


def live_seat_indices(occupancy: np.ndarray | tuple[bool, ...]) -> list[int]:
    occ = np.asarray(occupancy, dtype=bool)
    return [i for i in range(NUM_PLAYERS) if occ[i]]


def opponent_seats(seat: int, occupancy: np.ndarray | tuple[bool, ...]) -> list[int]:
    return [s for s in live_seat_indices(occupancy) if s != seat]
