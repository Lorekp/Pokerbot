"""Batching utilities for ObsV3."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aof_rl.obs_v3 import ObsV3


@dataclass
class ObsV3Batch:
    hero_hand: torch.Tensor
    global_feats: torch.Tensor
    seat_tokens: torch.Tensor
    seat_mask: torch.Tensor
    history_tokens: torch.Tensor
    history_mask: torch.Tensor
    legal_mask: torch.Tensor

    def to_obs_v3(self) -> ObsV3:
        return ObsV3(
            hero_hand=self.hero_hand,
            global_feats=self.global_feats,
            seat_tokens=self.seat_tokens,
            seat_mask=self.seat_mask,
            history_tokens=self.history_tokens,
            history_mask=self.history_mask,
            legal_mask=self.legal_mask,
        )


def collate_obs_v3(obs_list: list[ObsV3]) -> ObsV3Batch:
    return ObsV3Batch(
        hero_hand=torch.stack([o.hero_hand for o in obs_list]),
        global_feats=torch.stack([o.global_feats for o in obs_list]),
        seat_tokens=torch.stack([o.seat_tokens for o in obs_list]),
        seat_mask=torch.stack([o.seat_mask for o in obs_list]),
        history_tokens=torch.stack([o.history_tokens for o in obs_list]),
        history_mask=torch.stack([o.history_mask for o in obs_list]),
        legal_mask=torch.stack([o.legal_mask for o in obs_list]),
    )
