"""Priority 6B: RL observation v2."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from aof_rl.encode import (
    OBS_DIM_V1,
    OBS_DIM_V2,
    OBS_VERSION_V1,
    OBS_VERSION_V2,
    encode_obs,
    obs_dim,
    shover_seats_for_encode,
)
from aof_rl.env import ACTION_ALL_IN, reset_hand, step_hand
from aof_rl.policy import AoFPolicyNet
from aof_rl.teacher import NashTeacherCache


def test_v1_dimension_unchanged() -> None:
    assert obs_dim(1) == OBS_DIM_V1 == 117


def test_v2_dimension_stable() -> None:
    assert obs_dim(2) == OBS_DIM_V2 == 154


def test_v2_first_shover_single_shove() -> None:
    rng = random.Random(0)
    state = reset_hand(rng, table_size=4, stack_bb=8.0)
    utg = state.seats[0]
    state = step_hand(state, ACTION_ALL_IN, payoff_samples=100, payoff_seed=1)
    hero = state.action_seat
    assert hero is not None
    first, second = shover_seats_for_encode(state, hero)
    assert first == utg
    assert second is None
    cache = NashTeacherCache(teacher_version=2)
    result = cache.get_with_path(state, hero, samples=400, seed=7)
    obs = encode_obs(
        state,
        hero,
        result.teacher,
        version=OBS_VERSION_V2,
        teacher_path=result.path,
        provenance=result.provenance,
    )
    assert obs.shape == (OBS_DIM_V2,)
    # first shover one-hot block starts after cards+hero seat
    block = obs[104 + 4 : 104 + 4 + 5]
    assert block.sum() == pytest.approx(1.0)
    assert block[state.seats.index(utg)] == pytest.approx(1.0)


def test_v2_facing_two_encoding() -> None:
    rng = random.Random(1)
    state = reset_hand(rng, table_size=4, stack_bb=8.0)
    utg, btn, sb, _bb = state.seats
    state = step_hand(state, ACTION_ALL_IN, payoff_samples=100, payoff_seed=2)
    state = step_hand(state, ACTION_ALL_IN, payoff_samples=100, payoff_seed=3)
    hero = state.action_seat
    assert hero == sb
    first, second = shover_seats_for_encode(state, hero)
    assert first == utg
    assert second == btn
    cache = NashTeacherCache(teacher_version=2)
    result = cache.get_with_path(state, hero, samples=500, seed=9)
    obs = encode_obs(
        state,
        hero,
        result.teacher,
        version=OBS_VERSION_V2,
        teacher_path=result.path,
    )
    ctx_off = 104 + 4 + 5 + 5 + 2 + 4
    assert obs[ctx_off + 1] == pytest.approx(1.0)  # facing_two flag
    second_block = obs[104 + 4 + 5 : 104 + 4 + 5 + 5]
    assert second_block[state.seats.index(btn)] == pytest.approx(1.0)


def test_v1_checkpoint_net_forward() -> None:
    model = AoFPolicyNet(hidden=32, obs_dim=OBS_DIM_V1)
    x = torch.randn(4, OBS_DIM_V1)
    out = model(x)
    assert out.logits.shape == (4, 2)


def test_v2_policy_forward() -> None:
    model = AoFPolicyNet(hidden=32, obs_dim=OBS_DIM_V2)
    x = torch.randn(4, OBS_DIM_V2)
    out = model(x)
    assert out.logits.shape == (4, 2)
