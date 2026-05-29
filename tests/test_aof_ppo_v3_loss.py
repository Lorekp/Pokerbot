"""PPO v3 multi-head loss smoke tests."""

from __future__ import annotations

import torch

from aof_rl.obs_v3 import ObsV3
from aof_rl.obs_v3_batch import collate_obs_v3
from aof_rl.policy_v3 import AoFPolicyNetV3, EncoderConfig
from aof_rl.ppo_v3 import PPOv3Config, ppo_v3_loss


def _fake_obs() -> ObsV3:
    return ObsV3(
        hero_hand=torch.zeros(40),
        global_feats=torch.zeros(24),
        seat_tokens=torch.zeros(4, 32),
        seat_mask=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        history_tokens=torch.zeros(3, 24),
        history_mask=torch.zeros(3),
        legal_mask=torch.tensor([1.0, 1.0]),
    )


def test_ppo_v3_loss_backward() -> None:
    model = AoFPolicyNetV3(EncoderConfig(encoder_type="deep_sets", d_model=32, trunk_hidden=32))
    obs_list = [_fake_obs() for _ in range(4)]
    batch_obs = collate_obs_v3(obs_list)
    batch = {
        "action": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "teacher_action": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "teacher_q": torch.randn(4, 2),
        "teacher_edge": torch.rand(4),
        "old_value": torch.zeros(4),
    }
    old_lp = torch.zeros(4)
    adv = torch.randn(4)
    ret = torch.randn(4)
    loss, stats = ppo_v3_loss(
        model, batch_obs, batch, old_lp, adv, ret, PPOv3Config()
    )
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in stats.items():
        assert name in stats
    assert "q_loss" in stats
