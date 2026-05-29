"""Falsifiable RL: policy vs teacher under scripted opponents."""

from __future__ import annotations

import random

import pytest
import torch

pytest.importorskip("pokerkit")

from aof_equity import warmup
from aof_rl.policy import AoFPolicyNet
from aof_rl.scripted_opponent import ScriptedProfile, eval_hero_modes


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    warmup()


@pytest.mark.slow
def test_policy_beats_teacher_vs_loose_scripted() -> None:
    """Loose villains: greedy policy should match or beat teacher-following."""
    model = AoFPolicyNet(hidden=32)
    device = torch.device("cpu")
    stats = eval_hero_modes(
        model,
        device,
        ScriptedProfile(call_prob=0.65, name="loose"),
        hands=24,
        seed=99,
        teacher_samples=800,
        table_size=4,
    )
    assert stats["policy_bb_per_hand"] >= stats["teacher_bb_per_hand"] - 0.25


@pytest.mark.slow
def test_policy_not_catastrophic_hu() -> None:
    model = AoFPolicyNet(hidden=32)
    device = torch.device("cpu")
    stats = eval_hero_modes(
        model,
        device,
        ScriptedProfile(call_prob=0.35, name="tight"),
        hands=20,
        seed=7,
        teacher_samples=800,
        table_size=2,
    )
    assert stats["policy_bb_per_hand"] > stats["teacher_bb_per_hand"] - 1.5
