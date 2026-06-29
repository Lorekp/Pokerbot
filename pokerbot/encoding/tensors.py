"""AlphaHoldem-style tensor encoding for 6-max NLH."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pokerbot.engine.actions import ActionType, NUM_ACTIONS
from pokerbot.engine.cards import Card
from pokerbot.engine.game import GameState, NUM_PLAYERS, Street
from pokerbot.encoding.spot_scope import encode_spot_scope, fuse_position_scope
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
    """Card tensor: (slots, 4 suits, 13 ranks). Absent slots stay zero vectors."""
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


def encode_cards_from_hole_board(
    hole: tuple[Card, Card] | list[Card],
    board: list[Card],
    hero_seat: int = 0,
) -> np.ndarray:
    """encode_cards() wrapper for label repair when only hole/board strings exist."""
    state = GameState.new_hand(button_seat=0)
    state.players[hero_seat].hole = tuple(hole[:2])
    state.board = list(board)
    return encode_cards(state, hero_seat)


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
    # CASH-ENC-A HIGH: stamp actions on the STREET on which they occurred (was street=0 always,
    # which corrupted any state that reached postflop — postflop actions overwrote the preflop
    # row via np.maximum and produced [fold AND raise AND call] feat masks on the same cell).
    street_idx_map = {
        Street.PREFLOP: 0,
        Street.FLOP: 1,
        Street.TURN: 2,
        Street.RIVER: 3,
    }
    current_street_idx = 0

    for seat, action, _street in state.action_history_with_street():
        current_street_idx = street_idx_map.get(_street, 0)
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
        # feat[6] is per-seat label — preserved on the per-street row; pot context broadcast
        # (formerly line 112) was overwriting this on ALL streets, losing seat info entirely.
        feat[6] = seat / NUM_PLAYERS
        tensor[seat, current_street_idx, :] = np.maximum(
            tensor[seat, current_street_idx, :], feat
        )

    # Per-player stack/pot context lives on EACH street row independently (was street 0 only,
    # which silently bled preflop stack info into postflop rows).
    for seat, player in enumerate(state.players):
        for s in range(NUM_STREETS):
            tensor[seat, s, 7] = player.stack_bb / 100.0
            tensor[seat, s, 8] = 1.0 if player.folded else 0.0
            tensor[seat, s, 9] = 1.0 if player.all_in else 0.0
            tensor[seat, s, 10] = player.committed_street_bb / max(state.current_bet_bb, 1.0)
            tensor[seat, s, 11] = float(int(player.position))

    # Pot broadcast now goes to the CURRENT street only (was all streets, which produced the
    # same pot value on every row and doubled the pot feature weight).
    tensor[:, current_street_idx, 6] = state.pot_bb / 200.0
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
    *,
    spot_class: str = "",
    oop_range: str = "",
    ip_range: str = "",
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
    position = encode_position(hero_seat, state)
    spot_scope = encode_spot_scope(spot_class, oop_range, ip_range)
    return {
        "weva_id": weva_id,
        "cards": encode_cards(state, hero_seat),
        "actions": encode_actions(state, hero_seat),
        "position": position,
        "spot_scope": spot_scope,
        "policy_position": fuse_position_scope(position, spot_scope),
    }


def batch_encode(
    states: list[tuple[GameState, int]],
    mapper: WEVAMapper | None = None,
) -> dict[str, np.ndarray]:
    encoded = [encode_state(s, h, mapper=mapper) for s, h in states]
    return {
        "weva_ids": np.array([e["weva_id"] for e in encoded], dtype=np.int64),
        "cards": np.stack([e["cards"] for e in encoded]),
        "actions": np.stack([e["actions"] for e in encoded]),
        "position": np.stack([e["policy_position"] for e in encoded]),
        "spot_scope": np.stack([e["spot_scope"] for e in encoded]),
    }
