"""Flax policy net aligned with Phase 1 ``PseudoSiamesePolicy`` (SFT weight transfer)."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
from jax import Array

from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.encoding.spot_scope import POLICY_POSITION_DIM
from pokerbot.encoding.tensors import NUM_CARD_SLOTS, SPEC
from pokerbot.encoding.weva_encoder import WEVA_DEFAULT_K


class _ConvBlock(nn.Module):
    out_ch: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Conv(self.out_ch, (3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        return nn.relu(x)


class NLHEWEVAPolicyNet(nn.Module):
    """WEVA embed + card/action conv branches + policy head (matches SFT arch)."""

    hidden: int = 128
    num_actions: int = NUM_ACTIONS
    num_weva_buckets: int = WEVA_DEFAULT_K
    position_dim: int = POLICY_POSITION_DIM

    @nn.compact
    def __call__(
        self,
        weva_ids: Array,
        cards: Array,
        actions: Array,
        position: Array,
        legal_mask: Array,
    ) -> Array:
        # PyTorch SFT uses NCHW (B, 9, 4, 13) / (B, 6, 4, 12); Flax Conv expects NHWC.
        cards = jnp.transpose(cards, (0, 2, 3, 1))
        actions = jnp.transpose(actions, (0, 2, 3, 1))
        hole_feat = nn.Embed(self.num_weva_buckets, self.hidden)(weva_ids)
        card_feat = _ConvBlock(32)(cards)
        card_feat = _ConvBlock(64)(card_feat)
        card_feat = jnp.mean(card_feat, axis=(1, 2))
        card_feat = nn.Dense(self.hidden)(card_feat)
        card_feat = nn.relu(card_feat)

        act_feat = _ConvBlock(32)(actions)
        act_feat = _ConvBlock(64)(act_feat)
        act_feat = jnp.mean(act_feat, axis=(1, 2))
        act_feat = nn.Dense(self.hidden)(act_feat)
        act_feat = nn.relu(act_feat)

        pos_feat = nn.Dense(self.hidden // 4)(position)
        fused = jnp.concatenate([hole_feat, card_feat, act_feat, pos_feat], axis=-1)
        h = nn.Dense(self.hidden)(fused)
        h = nn.relu(h)
        logits = nn.Dense(self.num_actions)(h)
        return jnp.where(legal_mask, logits, -1e9)


def policy_inputs_from_bundle(
    weva_id: int,
    cards: Array,
    actions: Array,
    policy_position: Array,
    legal_mask: Array,
) -> dict[str, Array]:
    """Batch-1 NHWC tensors for ``NLHEWEVAPolicyNet``."""
    c = jnp.asarray(cards, dtype=jnp.float32).reshape(NUM_CARD_SLOTS, CARD_PLANES, CARD_RANKS)
    a = jnp.asarray(actions, dtype=jnp.float32).reshape(6, 4, 12)
    return {
        "weva_ids": jnp.asarray([weva_id], dtype=jnp.int32),
        "cards": c[None, ...],
        "actions": a[None, ...],
        "position": jnp.asarray(policy_position, dtype=jnp.float32)[None, :],
        "legal_mask": jnp.asarray(legal_mask, dtype=bool)[None, :],
    }


CARD_PLANES = 4
CARD_RANKS = 13
