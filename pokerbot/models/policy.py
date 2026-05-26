"""Pseudo-Siamese policy network with WEVA bucket embedding (6-max)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.encoding.tensors import SPEC
from pokerbot.encoding.weva_encoder import WEVA_DEFAULT_K


@dataclass
class PolicyOutput:
    logits: torch.Tensor
    value: torch.Tensor
    probs: torch.Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PseudoSiamesePolicy(nn.Module):
    """
    WEVA bucket embedding for cards; conv branch for action history.
    Input: weva_ids (B,) discrete bucket indices from WEVAMapper.
    """

    def __init__(
        self,
        hidden: int = 128,
        num_actions: int = NUM_ACTIONS,
        num_weva_buckets: int = WEVA_DEFAULT_K,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.num_weva_buckets = num_weva_buckets

        self.weva_embed = nn.Embedding(num_weva_buckets, hidden)

        self.action_encoder = nn.Sequential(
            ConvBlock(6, 32),
            ConvBlock(32, 64),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden),
            nn.ReLU(inplace=True),
        )

        self.pos_embed = nn.Linear(6, hidden // 4)

        fused = hidden * 2 + hidden // 4
        self.policy_head = nn.Sequential(
            nn.Linear(fused, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(fused, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        weva_ids: torch.Tensor,
        actions: torch.Tensor,
        position: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> PolicyOutput:
        card_feat = self.weva_embed(weva_ids)
        act_feat = self.action_encoder(actions)
        pos_feat = self.pos_embed(position)
        fused = torch.cat([card_feat, act_feat, pos_feat], dim=-1)

        logits = self.policy_head(fused)
        if legal_mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~legal_mask.bool(), mask_value)

        probs = F.softmax(logits, dim=-1)
        value = self.value_head(fused).squeeze(-1)
        return PolicyOutput(logits=logits, value=value, probs=probs)

    def act(
        self,
        weva_ids: torch.Tensor,
        actions: torch.Tensor,
        position: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, PolicyOutput]:
        out = self.forward(weva_ids, actions, position, legal_mask)
        if deterministic:
            action = out.logits.argmax(dim=-1)
        else:
            action = torch.multinomial(out.probs, num_samples=1).squeeze(-1)
        return action, out
