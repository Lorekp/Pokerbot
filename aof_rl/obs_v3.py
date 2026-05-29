"""Structured observation contract for AoF RL obs v3."""

from __future__ import annotations

from dataclasses import dataclass

import torch

HERO_HAND_DIM = 40
GLOBAL_FEAT_DIM = 24
SEAT_TOKEN_DIM = 32
HISTORY_TOKEN_DIM = 24
MAX_SEATS = 4
MAX_HISTORY = 3
NUM_ACTIONS = 2


@dataclass(frozen=True)
class ObsV3:
    hero_hand: torch.Tensor
    global_feats: torch.Tensor
    seat_tokens: torch.Tensor
    seat_mask: torch.Tensor
    history_tokens: torch.Tensor
    history_mask: torch.Tensor
    legal_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.hero_hand.dim() == 2:
            if self.hero_hand.shape[1] != HERO_HAND_DIM:
                raise ValueError(f"batched hero_hand {self.hero_hand.shape}")
            return
        if self.hero_hand.shape != (HERO_HAND_DIM,):
            raise ValueError(f"hero_hand shape {self.hero_hand.shape}")
        if self.global_feats.shape != (GLOBAL_FEAT_DIM,):
            raise ValueError(f"global_feats shape {self.global_feats.shape}")
        if self.seat_tokens.shape != (MAX_SEATS, SEAT_TOKEN_DIM):
            raise ValueError(f"seat_tokens shape {self.seat_tokens.shape}")
        if self.seat_mask.shape != (MAX_SEATS,):
            raise ValueError(f"seat_mask shape {self.seat_mask.shape}")
        if self.history_tokens.shape != (MAX_HISTORY, HISTORY_TOKEN_DIM):
            raise ValueError(f"history_tokens shape {self.history_tokens.shape}")
        if self.history_mask.shape != (MAX_HISTORY,):
            raise ValueError(f"history_mask shape {self.history_mask.shape}")
        if self.legal_mask.shape != (NUM_ACTIONS,):
            raise ValueError(f"legal_mask shape {self.legal_mask.shape}")

    @property
    def is_batched(self) -> bool:
        return self.hero_hand.dim() == 2


def obs_v3_dims() -> dict[str, int | tuple[int, ...]]:
    return {
        "hero_hand_dim": HERO_HAND_DIM,
        "global_feat_dim": GLOBAL_FEAT_DIM,
        "seat_token_dim": SEAT_TOKEN_DIM,
        "history_token_dim": HISTORY_TOKEN_DIM,
        "max_seats": MAX_SEATS,
        "max_history": MAX_HISTORY,
        "legal_actions": NUM_ACTIONS,
    }
