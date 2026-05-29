"""AoF policy network (MLP)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from aof_rl.encode import OBS_DIM_V1

NUM_ACTIONS = 2


@dataclass
class PolicyOutput:
    logits: torch.Tensor
    value: torch.Tensor
    probs: torch.Tensor


class AoFPolicyNet(nn.Module):
    def __init__(self, hidden: int = 128, obs_dim: int = OBS_DIM_V1) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(hidden, NUM_ACTIONS)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> PolicyOutput:
        h = self.trunk(obs)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        probs = F.softmax(logits, dim=-1)
        return PolicyOutput(logits=logits, value=value, probs=probs)

    def act(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[int, PolicyOutput]:
        out = self.forward(obs)
        if deterministic:
            idx = int(out.logits.argmax(dim=-1).item())
        else:
            idx = int(torch.distributions.Categorical(logits=out.logits).sample().item())
        return idx, out
