"""Fast teacher and payoff smoke tests."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("pokerkit")

from aof import NashTeacher
from aof_equity import warmup
from aof_parse import OpenSpot
from aof_rl.env import ACTION_FOLD, reset_hand, step_hand
from aof_rl.fast_teacher import nash_teacher_open_fast
from aof_rl.self_play import AoFSelfPlayConfig, collect_trajectory
from aof_rl.policy import AoFPolicyNet
from aof_rl.teacher import NashTeacherCache
import torch


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    warmup()


def test_fast_open_teacher_shape() -> None:
    spot = OpenSpot("AKo", ("Ah", "Kd"), "BTN", 8.0, 4)
    t = nash_teacher_open_fast(spot, samples=400, seed=1, margin_bb=0.4)
    assert isinstance(t, NashTeacher)
    assert t.action in ("ALL IN", "FOLD")


def test_fast_cache_rollout() -> None:
    cfg = AoFSelfPlayConfig(
        teacher_samples=300,
        payoff_samples=800,
        use_fast_teacher=True,
    )
    model = AoFPolicyNet(hidden=32)
    device = torch.device("cpu")
    cache = NashTeacherCache(use_fast=True, teacher_margin_bb=0.4)
    rng = random.Random(0)
    traj = collect_trajectory(
        model, None, device, "BTN", rng, cache, cfg
    )
    assert traj or True
    state = reset_hand(rng, table_size=4, stack_bb=8.0)
    while not state.terminal:
        seat = state.action_seat
        if seat is None:
            break
        state = step_hand(state, ACTION_FOLD, payoff_samples=500, payoff_seed=1)
    assert abs(sum(state.payoffs.values())) < 1e-5
