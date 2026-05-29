"""Smoke tests for parallel AoF rollout collection."""

from __future__ import annotations

import random

import pytest

from aof_equity import warmup
from aof_rl.policy import AoFPolicyNet
from aof_rl.self_play import AoFSelfPlayConfig, collect_trajectory


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    warmup()


def test_parallel_rollout_smoke() -> None:
    pytest.importorskip("torch")
    from aof_rl.parallel_rollout import collect_trajectories_parallel

    cfg = AoFSelfPlayConfig(
        hands_per_update=2,
        teacher_samples=200,
        payoff_samples=400,
        use_fast_teacher=True,
        rollout_workers=0,
    )
    model = AoFPolicyNet()
    rng = random.Random(7)
    traj = collect_trajectories_parallel(model, cfg, n_hands=2, rng=rng, workers=2)
    assert isinstance(traj, list)


def test_retain_teacher_cache_serial() -> None:
    pytest.importorskip("torch")
    cfg = AoFSelfPlayConfig(
        teacher_samples=150,
        payoff_samples=300,
        use_fast_teacher=True,
        retain_teacher_cache=True,
    )
    model = AoFPolicyNet()
    from aof_rl.teacher import NashTeacherCache

    cache = NashTeacherCache(use_fast=True, max_entries=16)
    rng = random.Random(1)
    t1 = collect_trajectory(
        model, None, __import__("torch").device("cpu"), "BTN", rng, cache, cfg
    )
    assert isinstance(t1, list)
    assert len(cache._cache) > 0
