"""Legal-mask correctness for AoFPolicyNetV3."""

from __future__ import annotations

import torch

from aof_rl.env import ACTION_ALL_IN, ACTION_FOLD
from aof_rl.obs_v3 import ObsV3
from aof_rl.policy_v3 import AoFPolicyNetV3, EncoderConfig


def test_legal_mask_forces_all_in() -> None:
    torch.manual_seed(0)
    model = AoFPolicyNetV3(
        EncoderConfig(encoder_type="deep_sets", d_model=32, trunk_hidden=32)
    )
    model.eval()
    obs = ObsV3(
        hero_hand=torch.randn(40),
        global_feats=torch.randn(24),
        seat_tokens=torch.randn(4, 32),
        seat_mask=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        history_tokens=torch.randn(3, 24),
        history_mask=torch.tensor([0.0, 0.0, 0.0]),
        legal_mask=torch.tensor([0.0, 1.0]),
    )
    for _ in range(20):
        idx, out = model.act(obs, deterministic=False)
        assert idx == ACTION_ALL_IN
    idx_det, _ = model.act(obs, deterministic=True)
    assert idx_det == ACTION_ALL_IN
    logits = out.logits
    if logits.dim() == 1:
        assert logits[ACTION_FOLD].item() < -1e8 or logits[ACTION_ALL_IN] >= logits[ACTION_FOLD]


def test_both_actions_legal() -> None:
    model = AoFPolicyNetV3(
        EncoderConfig(encoder_type="deep_sets", d_model=32, trunk_hidden=32)
    )
    obs = ObsV3(
        hero_hand=torch.randn(40),
        global_feats=torch.randn(24),
        seat_tokens=torch.randn(4, 32),
        seat_mask=torch.ones(4),
        history_tokens=torch.randn(3, 24),
        history_mask=torch.zeros(3),
        legal_mask=torch.tensor([1.0, 1.0]),
    )
    out = model.forward(obs)
    assert out.logits.shape[-1] == 2
    assert torch.isfinite(out.q_bb).all()
