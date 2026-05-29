"""Structured obs v3 policy: Deep Sets / Set-Transformer-lite + multi-head outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from aof_rl.env import ACTION_ALL_IN, ACTION_FOLD
from aof_rl.obs_v3_batch import ObsV3Batch
from aof_rl.obs_v3 import (
    GLOBAL_FEAT_DIM,
    HERO_HAND_DIM,
    HISTORY_TOKEN_DIM,
    MAX_HISTORY,
    MAX_SEATS,
    NUM_ACTIONS,
    SEAT_TOKEN_DIM,
    ObsV3,
    obs_v3_dims,
)

EncoderType = Literal["deep_sets", "set_transformer_lite"]


@dataclass
class EncoderConfig:
    encoder_type: EncoderType = "set_transformer_lite"
    d_model: int = 96
    n_blocks: int = 2
    n_heads: int = 4
    n_inducing: int = 8
    trunk_hidden: int = 128


@dataclass
class PolicyOutputV3:
    logits: torch.Tensor
    q_bb: torch.Tensor
    value: torch.Tensor
    uncertainty: torch.Tensor
    probs: torch.Tensor
    log_probs: torch.Tensor


def _masked_mean(
    tokens: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """tokens [B,N,D], mask [B,N] -> [B,D]."""
    m = mask.unsqueeze(-1)
    summed = (tokens * m).sum(dim=1)
    denom = m.sum(dim=1).clamp(min=1.0)
    return summed / denom


class DeepSetsEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.seat_proj = nn.Linear(SEAT_TOKEN_DIM, d)
        self.hist_proj = nn.Linear(HISTORY_TOKEN_DIM, d)
        self.hero_proj = nn.Linear(HERO_HAND_DIM, d)
        self.global_proj = nn.Linear(GLOBAL_FEAT_DIM, d)

    def forward(self, obs: ObsV3) -> torch.Tensor:
        hero, glob, seats, seat_m, hist, hist_m, _ = _obs_batch_tensors(obs)
        seat_h = self.seat_proj(seats)
        hist_h = self.hist_proj(hist)
        pooled = _masked_mean(
            torch.cat([seat_h, hist_h], dim=1),
            torch.cat([seat_m, hist_m], dim=1),
        )
        return self.hero_proj(hero) + self.global_proj(glob) + pooled


def _obs_batch_tensors(obs: ObsV3) -> tuple:
    """Normalize ObsV3 to batched tensors [B,...]."""
    hero = obs.hero_hand
    single = hero.dim() == 1
    if single:
        return (
            hero.unsqueeze(0),
            obs.global_feats.unsqueeze(0),
            obs.seat_tokens.unsqueeze(0),
            obs.seat_mask.unsqueeze(0),
            obs.history_tokens.unsqueeze(0),
            obs.history_mask.unsqueeze(0),
            obs.legal_mask.unsqueeze(0),
        )
    return (
        hero,
        obs.global_feats,
        obs.seat_tokens,
        obs.seat_mask,
        obs.history_tokens,
        obs.history_mask,
        obs.legal_mask,
    )


class InducedAttentionBlock(nn.Module):
    """ISAB-lite: inducing points attend to set, then set attends to inducing."""

    def __init__(self, d_model: int, n_heads: int, n_inducing: int) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, n_inducing, d_model) * 0.02)
        self.attn1 = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        ind = self.inducing.expand(b, -1, -1)
        key_padding = mask < 0.5
        h, _ = self.attn1(ind, x, x, key_padding_mask=key_padding)
        ind = self.norm1(ind + h)
        h2, _ = self.attn2(x, ind, ind)
        x = self.norm2(x + h2)
        x = self.norm3(x + self.ff(x))
        return x


class SetTransformerLiteEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.seat_proj = nn.Linear(SEAT_TOKEN_DIM, d)
        self.hist_proj = nn.Linear(HISTORY_TOKEN_DIM, d)
        self.hero_proj = nn.Linear(HERO_HAND_DIM, d)
        self.global_proj = nn.Linear(GLOBAL_FEAT_DIM, d)
        self.blocks = nn.ModuleList(
            [
                InducedAttentionBlock(d, cfg.n_heads, cfg.n_inducing)
                for _ in range(cfg.n_blocks)
            ]
        )

    def forward(self, obs: ObsV3) -> torch.Tensor:
        (
            hero,
            glob,
            seats,
            seat_m,
            hist,
            hist_m,
            _legal,
        ) = _obs_batch_tensors(obs)
        seat_h = self.seat_proj(seats)
        hist_h = self.hist_proj(hist)
        tokens = torch.cat([seat_h, hist_h], dim=1)
        mask = torch.cat([seat_m, hist_m], dim=1)
        for block in self.blocks:
            tokens = block(tokens, mask)
        pooled = _masked_mean(tokens, mask)
        return self.hero_proj(hero) + self.global_proj(glob) + pooled


def build_encoder(cfg: EncoderConfig) -> nn.Module:
    if cfg.encoder_type == "deep_sets":
        return DeepSetsEncoder(cfg)
    return SetTransformerLiteEncoder(cfg)


class AoFPolicyNetV3(nn.Module):
    def __init__(self, encoder_cfg: EncoderConfig | None = None) -> None:
        super().__init__()
        self.encoder_cfg = encoder_cfg or EncoderConfig()
        self.encoder = build_encoder(self.encoder_cfg)
        h = self.encoder_cfg.trunk_hidden
        d = self.encoder_cfg.d_model
        self.trunk = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, h),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(h, NUM_ACTIONS)
        self.q_head = nn.Linear(h, NUM_ACTIONS)
        self.value_head = nn.Linear(h, 1)
        self.uncertainty_head = nn.Linear(h, 1)

    def forward(self, obs: ObsV3 | ObsV3Batch) -> PolicyOutputV3:
        if isinstance(obs, ObsV3Batch):
            obs = obs.to_obs_v3()
        h = self.trunk(self.encoder(obs))
        logits = self.policy_head(h)
        q_bb = self.q_head(h)
        value = self.value_head(h).squeeze(-1)
        uncertainty = F.softplus(self.uncertainty_head(h)).squeeze(-1)

        legal = obs.legal_mask
        if legal.dim() == 1:
            legal = legal.unsqueeze(0)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        mask_bool = legal > 0.5
        logits = logits.masked_fill(~mask_bool, -1e9)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return PolicyOutputV3(
            logits=logits,
            q_bb=q_bb,
            value=value,
            uncertainty=uncertainty,
            probs=probs,
            log_probs=log_probs,
        )

    def _gather_selected_log_prob(
        self, out: PolicyOutputV3, action: torch.Tensor
    ) -> torch.Tensor:
        lp = out.log_probs
        if lp.dim() == 1:
            return lp[action]
        return lp.gather(1, action.unsqueeze(1)).squeeze(1)

    def act_with_logprob(
        self,
        obs: ObsV3,
        *,
        deterministic: bool = False,
    ) -> tuple[int, float, float, PolicyOutputV3]:
        out = self.forward(obs)
        logits = out.logits
        logits_b = logits.unsqueeze(0) if logits.dim() == 1 else logits
        action = self._sample_actions(logits_b, deterministic=deterministic)
        if logits.dim() == 1:
            idx = int(action.item())
            lp = float(out.log_probs[idx].item())
            val = float(out.value.item())
        else:
            idx = int(action[0].item())
            lp = float(out.log_probs[0, idx].item())
            val = float(out.value[0].item())
        return idx, lp, val, out

    def act_batch_with_logprob(
        self,
        batch: ObsV3Batch,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, PolicyOutputV3]:
        out = self.forward(batch)
        idx = self._sample_actions(out.logits, deterministic=deterministic)
        lp = self._gather_selected_log_prob(out, idx)
        value = out.value
        return idx, lp, value, out

    def _sample_actions(
        self, logits: torch.Tensor, *, deterministic: bool
    ) -> torch.Tensor:
        if deterministic:
            return logits.argmax(dim=-1)
        return torch.distributions.Categorical(logits=logits).sample()

    def act_batch(
        self,
        batch: ObsV3Batch,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, PolicyOutputV3]:
        out = self.forward(batch)
        idx = self._sample_actions(out.logits, deterministic=deterministic)
        return idx, out

    def act_batch_single(
        self,
        obs: ObsV3,
        *,
        deterministic: bool = False,
    ) -> tuple[int, PolicyOutputV3]:
        idx, _, _, out = self.act_with_logprob(obs, deterministic=deterministic)
        return idx, out

    def act(
        self,
        obs: ObsV3,
        *,
        deterministic: bool = False,
    ) -> tuple[int, PolicyOutputV3]:
        idx, _, _, out = self.act_with_logprob(obs, deterministic=deterministic)
        return idx, out


def encoder_config_to_dict(cfg: EncoderConfig) -> dict:
    return {
        "encoder_type": cfg.encoder_type,
        "d_model": cfg.d_model,
        "n_blocks": cfg.n_blocks,
        "n_heads": cfg.n_heads,
        "n_inducing": cfg.n_inducing,
        "trunk_hidden": cfg.trunk_hidden,
        "obs_v3_dims": obs_v3_dims(),
    }


def encoder_config_from_dict(d: dict) -> EncoderConfig:
    return EncoderConfig(
        encoder_type=d.get("encoder_type", "set_transformer_lite"),
        d_model=int(d.get("d_model", 96)),
        n_blocks=int(d.get("n_blocks", 2)),
        n_heads=int(d.get("n_heads", 4)),
        n_inducing=int(d.get("n_inducing", 8)),
        trunk_hidden=int(d.get("trunk_hidden", 128)),
    )
