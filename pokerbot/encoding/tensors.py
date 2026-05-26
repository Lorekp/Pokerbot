"""AlphaHoldem-style tensor encoding for 6-max NLH."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pokerbot.engine.actions import ActionType, NUM_ACTIONS
from pokerbot.engine.cards import Card
from pokerbot.engine.game import GameState, NUM_PLAYERS, Street
from pokerbot.encoding.weva_encoder import WEVAMapper


NUM_CARD_SLOTS = 9  # 2 hole + 5 board + 2 history placeholder
CARD_PLANES = 4
CARD_RANKS = 13
NUM_STREETS = 4
MAX_ACTIONS_PER_STREET = 12


@dataclass(frozen=True)
class TensorSpec:
    card_shape: tuple[int, ...] = (NUM_CARD_SLOTS, CARD_PLANES, CARD_RANKS)
    action_shape: tuple[int, ...] = (NUM_PLAYERS, NUM_STREETS, MAX_ACTIONS_PER_STREET)
    num_actions: int = NUM_ACTIONS


SPEC = TensorSpec()


def encode_card_plane(card: Card) -> np.ndarray:
    plane = np.zeros((CARD_PLANES, CARD_RANKS), dtype=np.float32)
    plane[int(card.suit), card.rank - 2] = 1.0
    return plane


def encode_cards(state: GameState, hero_seat: int) -> np.ndarray:
    """Card tensor: (slots, 4 suits, 13 ranks)."""
    tensor = np.zeros(SPEC.card_shape, dtype=np.float32)
    hero = state.players[hero_seat]
    slot = 0
    if hero.hole:
        for c in hero.hole:
            tensor[slot] = encode_card_plane(c)
            slot += 1
    for c in state.board:
        if slot < NUM_CARD_SLOTS:
            tensor[slot] = encode_card_plane(c)
            slot += 1
    return tensor


def _street_index(street: Street) -> int:
    return {
        Street.PREFLOP: 0,
        Street.FLOP: 1,
        Street.TURN: 2,
        Street.RIVER: 3,
    }.get(street, 3)


def encode_actions(state: GameState, hero_seat: int) -> np.ndarray:
    """
    Action tensor: (6 players, 4 streets, 12 action features).
    Features per action: [acted, fold, check_call, bet_raise, amount_norm, is_hero]
    packed into 12 dims via history indexing.
    """
    tensor = np.zeros(SPEC.action_shape, dtype=np.float32)
    street_actions: dict[int, list] = {0: [], 1: [], 2: [], 3: []}
    current_street = 0

    for seat, action in state.action_history():
        idx = min(len(street_actions[current_street]), MAX_ACTIONS_PER_STREET - 1)
        feat = np.zeros(MAX_ACTIONS_PER_STREET, dtype=np.float32)
        feat[0] = 1.0  # marker
        if action.action_type == ActionType.FOLD:
            feat[1] = 1.0
        elif action.action_type in (ActionType.CHECK, ActionType.CALL):
            feat[2] = 1.0
        else:
            feat[3] = 1.0
        feat[4] = min(action.amount_bb / 100.0, 1.0)
        feat[5] = 1.0 if seat == hero_seat else 0.0
        feat[6] = seat / NUM_PLAYERS
        tensor[seat, current_street, :] = np.maximum(tensor[seat, current_street, :], feat)

        if action.action_type in (ActionType.BET, ActionType.RAISE) and current_street < 3:
            pass  # stay on street

    # Encode per-player stack and pot context in row slices
    for seat, player in enumerate(state.players):
        tensor[seat, 0, 7] = player.stack_bb / 100.0
        tensor[seat, 0, 8] = 1.0 if player.folded else 0.0
        tensor[seat, 0, 9] = 1.0 if player.all_in else 0.0
        tensor[seat, 0, 10] = player.committed_street_bb / max(state.current_bet_bb, 1.0)
        tensor[seat, 0, 11] = float(int(player.position))

    tensor[:, :, 6] = state.pot_bb / 200.0
    return tensor


def encode_position(hero_seat: int, state: GameState) -> np.ndarray:
    """One-hot hero position (6,)."""
    pos = np.zeros(NUM_PLAYERS, dtype=np.float32)
    pos[int(state.players[hero_seat].position)] = 1.0
    return pos


def encode_state(
    state: GameState,
    hero_seat: int,
    mapper: WEVAMapper | None = None,
) -> dict[str, np.ndarray | int]:
    hero = state.players[hero_seat]
    if hero.hole is None:
        weva_id = 0
    else:
        m = mapper or WEVAMapper.default()
        weva_id = m.lookup(
            hero.hole,
            position=int(hero.position),
        )
    return {
        "weva_id": weva_id,
        "actions": encode_actions(state, hero_seat),
        "position": encode_position(hero_seat, state),
    }


def batch_encode(
    states: list[tuple[GameState, int]],
    mapper: WEVAMapper | None = None,
) -> dict[str, np.ndarray]:
    encoded = [encode_state(s, h, mapper=mapper) for s, h in states]
    return {
        "weva_ids": np.array([e["weva_id"] for e in encoded], dtype=np.int64),
        "actions": np.stack([e["actions"] for e in encoded]),
        "position": np.stack([e["position"] for e in encoded]),
    }
