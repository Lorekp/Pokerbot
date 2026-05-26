"""Backward-compatible shim for the MADT opponent transformer."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import torch

from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.opponent.opponent_transformer import (
    KVCache,
    OpponentTransformer as MADTOpponentTransformer,
    OpponentTransformerConfig,
)

MAX_HANDS = 20
MAX_TOKENS_PER_HAND = 32


@dataclass
class HandRecord:
    """Legacy hand summary token (deprecated — use HandTrajectoryBuilder)."""

    street: int = 0
    action_token: int = 0
    position: int = 0
    board_texture: int = 0
    won: int = 0
    showdown: int = 0


@dataclass
class HandHistoryBuffer:
    """Per-opponent rolling buffer of legacy hand records."""

    histories: dict[int, list[HandRecord]] = field(default_factory=dict)

    def add(self, opponent_id: int, record: HandRecord) -> None:
        if opponent_id not in self.histories:
            self.histories[opponent_id] = []
        self.histories[opponent_id].append(record)
        if len(self.histories[opponent_id]) > MAX_HANDS:
            self.histories[opponent_id] = self.histories[opponent_id][-MAX_HANDS:]


# Alias new model as OpponentTransformer for migrated imports.
OpponentTransformer = MADTOpponentTransformer

__all__ = [
    "OpponentTransformer",
    "OpponentTransformerConfig",
    "KVCache",
    "HandRecord",
    "HandHistoryBuffer",
    "MAX_HANDS",
    "MAX_TOKENS_PER_HAND",
    "NUM_ACTIONS",
]
