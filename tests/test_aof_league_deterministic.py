"""Deterministic validation mode for v3 policy and PPO loss."""

from __future__ import annotations

import random

import pytest
import torch

pytest.importorskip("pokerkit")

from aof import NashTeacher
from aof_rl.obs_v3 import ObsV3
from aof_rl.obs_v3_batch import collate_obs_v3
from aof_rl.policy_v3 import AoFPolicyNetV3, EncoderConfig
from aof_rl.ppo_v3 import PPOv3Config, ppo_v3_loss
from aof_rl.teacher_targets import teacher_q_bb


def _fake_obs(seed: int) -> ObsV3:
    g = torch.Generator().manual_seed(seed)
    return ObsV3(
        hero_hand=torch.randn(40, generator=g),
        global_feats=torch.randn(24, generator=g),
        seat_tokens=torch.randn(4, 32, generator=g),
        seat_mask=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        history_tokens=torch.randn(3, 24, generator=g),
        history_mask=torch.zeros(3),
        legal_mask=torch.tensor([1.0, 1.0]),
    )


def test_policy_deterministic_act() -> None:
    torch.manual_seed(42)
    model = AoFPolicyNetV3(
        EncoderConfig(encoder_type="deep_sets", d_model=32, trunk_hidden=32)
    )
    model.eval()
    obs = _fake_obs(7)
    a1, o1 = model.act(obs, deterministic=True)
    a2, o2 = model.act(obs, deterministic=True)
    assert a1 == a2
    assert torch.allclose(o1.logits, o2.logits)
    assert torch.allclose(o1.q_bb, o2.q_bb)


def test_ppo_loss_deterministic() -> None:
    torch.manual_seed(0)
    model = AoFPolicyNetV3(EncoderConfig(encoder_type="deep_sets", d_model=32, trunk_hidden=32))
    obs_list = [_fake_obs(i) for i in range(3)]
    batch_obs = collate_obs_v3(obs_list)
    batch = {
        "action": torch.tensor([0, 1, 0]),
        "teacher_action": torch.tensor([0, 1, 0]),
        "teacher_q": torch.tensor([[0.0, 1.0], [0.5, 2.0], [0.1, 0.2]]),
        "teacher_edge": torch.tensor([0.5, 1.5, 0.1]),
        "old_value": torch.zeros(3),
    }
    old_lp = torch.zeros(3)
    adv = torch.tensor([0.1, -0.2, 0.3])
    ret = torch.tensor([0.1, 0.8, 0.3])
    cfg = PPOv3Config()
    loss1, s1 = ppo_v3_loss(model, batch_obs, batch, old_lp, adv, ret, cfg)
    loss2, s2 = ppo_v3_loss(model, batch_obs, batch, old_lp, adv, ret, cfg)
    assert loss1.item() == pytest.approx(loss2.item())
    assert s1["total_loss"] == pytest.approx(s2["total_loss"])


def test_teacher_q_mapping() -> None:
    t = NashTeacher(
        action="ALL IN",
        ev_primary=2.0,
        ev_alt=-0.5,
        edge_bb=2.5,
        p_fold=0.2,
        facing=False,
    )
    q = teacher_q_bb(t)
    assert q[1] == pytest.approx(2.0)
    assert q[0] == pytest.approx(-0.5)
