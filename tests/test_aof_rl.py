"""Tests for AoF RL environment, teacher features, and PPO step."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

pytest.importorskip("pokerkit")

from aof import NashTeacher, nash_teacher_open
from aof_equity import warmup
from aof_parse import OpenSpot
from aof_rl.encode import OBS_DIM, encode_obs
from aof_rl.env import ACTION_ALL_IN, ACTION_FOLD, reset_hand, step_hand
from aof_rl.policy import AoFPolicyNet
from aof_rl.ppo import PPOConfig, compute_gae, ppo_loss
from aof_rl.teacher import NashTeacherCache


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    warmup()


class TestEnv:
    def test_legal_actions_at_start(self) -> None:
        rng = random.Random(1)
        state = reset_hand(rng, table_size=2, stack_bb=8.0)
        assert state.action_seat is not None
        assert set(state.legal_actions()) == {ACTION_FOLD, ACTION_ALL_IN}

    def test_hand_terminates(self) -> None:
        rng = random.Random(2)
        state = reset_hand(rng, table_size=4, stack_bb=8.0)
        while not state.terminal:
            seat = state.action_seat
            assert seat is not None
            state = step_hand(state, ACTION_FOLD, payoff_samples=500, payoff_seed=1)
        assert sum(state.payoffs.values()) == pytest.approx(0.0, abs=1e-6)


class TestNashTeacher:
    def test_open_teacher_shape(self) -> None:
        spot = OpenSpot("AKo", ("Ah", "Kd"), "BTN", 8.0, 4)
        t = nash_teacher_open(spot, samples=500, seed=42)
        assert isinstance(t, NashTeacher)
        assert t.action in ("ALL IN", "FOLD")
        assert t.edge_bb >= 0.0
        assert t.p_fold is not None

    def test_encode_obs_dim(self) -> None:
        rng = random.Random(3)
        state = reset_hand(rng, table_size=4, stack_bb=8.0)
        seat = state.action_seat
        assert seat is not None
        cache = NashTeacherCache()
        teacher = cache.get(state, seat, samples=500, seed=7)
        obs = encode_obs(state, seat, teacher)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32


class TestPPO:
    def test_one_update_step(self) -> None:
        model = AoFPolicyNet(hidden=32)
        obs = torch.randn(8, OBS_DIM)
        batch = {
            "obs": obs,
            "action": torch.randint(0, 2, (8,)),
            "teacher_action": torch.randint(0, 2, (8,)),
            "old_value": torch.zeros(8),
        }
        old_lp = torch.zeros(8)
        adv, rets = compute_gae(
            [0.1] * 8,
            [0.0] * 8,
            [True] * 8,
        )
        advantages = torch.tensor(adv, dtype=torch.float32)
        returns = torch.tensor(rets, dtype=torch.float32)
        cfg = PPOConfig(teacher_coef=0.3, ppo_epochs=1)
        loss, stats = ppo_loss(model, batch, old_lp, advantages, returns, cfg)
        loss.backward()
        assert "policy_loss" in stats
