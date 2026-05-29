"""Obs v3 encoder: parity, invariance, history sensitivity, malformed states."""

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
    OBS_VERSION_V3,
    encode_obs,
    obs_dim,
)
from aof_rl.encode_v3 import encode_obs_v3, encode_obs_v3_from_hand
from aof_rl.env import ACTION_ALL_IN, reset_hand, step_hand
from aof_rl.game_state import (
    AoFEncodeError,
    AoFGameState,
    SeatState,
    normalize_hand_state,
    permute_game_state_seats,
)
from aof_rl.obs_v3 import (
    GLOBAL_FEAT_DIM,
    HERO_HAND_DIM,
    HISTORY_TOKEN_DIM,
    MAX_HISTORY,
    MAX_SEATS,
    SEAT_TOKEN_DIM,
    ObsV3,
    obs_v3_dims,
)
from aof_rl.teacher import NashTeacherCache


def _obs_v3_tensors_equal(a: ObsV3, b: ObsV3) -> bool:
    return (
        torch.allclose(a.hero_hand, b.hero_hand)
        and torch.allclose(a.global_feats, b.global_feats)
        and torch.allclose(a.seat_tokens, b.seat_tokens)
        and torch.allclose(a.seat_mask, b.seat_mask)
        and torch.allclose(a.history_tokens, b.history_tokens)
        and torch.allclose(a.history_mask, b.history_mask)
        and torch.allclose(a.legal_mask, b.legal_mask)
    )


def test_obs_dim_v1_v2_unchanged() -> None:
    assert obs_dim(1) == OBS_DIM_V1 == 117
    assert obs_dim(2) == OBS_DIM_V2 == 154
    with pytest.raises(ValueError, match="structured"):
        obs_dim(3)


def test_obs_v3_dims_helper() -> None:
    d = obs_v3_dims()
    assert d["hero_hand_dim"] == HERO_HAND_DIM == 40
    assert d["seat_token_dim"] == SEAT_TOKEN_DIM == 32


def test_v1_v2_golden_parity() -> None:
    """Lock v1/v2 bytes for fixed rollout seeds (regression guard)."""
    golden_v1: list[tuple[int, ...]] = []
    golden_v2: list[tuple[int, ...]] = []
    for seed in (0, 1, 2):
        rng = random.Random(seed)
        state = reset_hand(rng, table_size=4, stack_bb=8.0)
        state = step_hand(state, ACTION_ALL_IN, payoff_samples=50, payoff_seed=seed)
        hero = state.action_seat
        assert hero is not None
        cache = NashTeacherCache(teacher_version=2)
        result = cache.get_with_path(state, hero, samples=200, seed=seed + 10)
        o1 = encode_obs(
            state, hero, result.teacher, version=OBS_VERSION_V1
        )
        o2 = encode_obs(
            state,
            hero,
            result.teacher,
            version=OBS_VERSION_V2,
            teacher_path=result.path,
        )
        assert o1.shape == (OBS_DIM_V1,)
        assert o2.shape == (OBS_DIM_V2,)
        golden_v1.append(tuple(float(x) for x in o1))
        golden_v2.append(tuple(float(x) for x in o2))

    for seed in (0, 1, 2):
        rng = random.Random(seed)
        state = reset_hand(rng, table_size=4, stack_bb=8.0)
        state = step_hand(state, ACTION_ALL_IN, payoff_samples=50, payoff_seed=seed)
        hero = state.action_seat
        assert hero is not None
        cache = NashTeacherCache(teacher_version=2)
        result = cache.get_with_path(state, hero, samples=200, seed=seed + 10)
        o1 = encode_obs(
            state, hero, result.teacher, version=OBS_VERSION_V1
        )
        o2 = encode_obs(
            state,
            hero,
            result.teacher,
            version=OBS_VERSION_V2,
            teacher_path=result.path,
        )
        assert tuple(float(x) for x in o1) == golden_v1[seed]
        assert tuple(float(x) for x in o2) == golden_v2[seed]


def test_v3_shapes_table_sizes() -> None:
    cache = NashTeacherCache(teacher_version=2)
    for table_size in (2, 3, 4):
        rng = random.Random(table_size + 40)
        state = reset_hand(rng, table_size=table_size, stack_bb=8.0)
        hero = state.action_seat
        assert hero is not None
        teacher = cache.get(state, hero, samples=100, seed=1)
        obs = encode_obs(state, hero, teacher, version=OBS_VERSION_V3)
        assert isinstance(obs, ObsV3)
        assert obs.hero_hand.dtype == torch.float32
        assert obs.seat_tokens.shape == (MAX_SEATS, SEAT_TOKEN_DIM)
        assert obs.history_tokens.shape == (MAX_HISTORY, HISTORY_TOKEN_DIM)
        assert obs.global_feats.shape == (GLOBAL_FEAT_DIM,)
        assert obs.legal_mask.sum() == pytest.approx(2.0)


def test_v3_prior_shover_counts() -> None:
    rng = random.Random(99)
    state = reset_hand(rng, table_size=4, stack_bb=8.0)
    for i in range(3):
        state = step_hand(state, ACTION_ALL_IN, payoff_samples=50, payoff_seed=i)
    hero = state.action_seat
    assert hero is not None
    game = normalize_hand_state(state, hero)
    obs = encode_obs_v3(game)
    assert int(obs.history_mask.sum()) == 3


def test_seat_order_invariance() -> None:
    rng = random.Random(7)
    state = reset_hand(rng, table_size=4, stack_bb=8.0)
    state = step_hand(state, ACTION_ALL_IN, payoff_samples=50, payoff_seed=1)
    state = step_hand(state, ACTION_ALL_IN, payoff_samples=50, payoff_seed=2)
    hero = state.action_seat
    assert hero is not None
    game = normalize_hand_state(state, hero)
    base = encode_obs_v3(game)
    for perm in ((3, 1, 0, 2), (2, 3, 1, 0), (1, 0, 3, 2)):
        shuffled = permute_game_state_seats(game, perm)
        other = encode_obs_v3(shuffled)
        assert _obs_v3_tensors_equal(base, other)


def test_shover_order_sensitivity() -> None:
    """Swapping prior ALL_IN order in history changes history tokens."""
    seats = (
        SeatState(
            seat_id=0,
            role="first_to_act",
            is_hero=False,
            state="acted_all_in",
            committed_bb=8.0,
            stack_remaining_bb=0.0,
            blind_posted_bb=0.0,
            shover_rank=1,
            eligible_pot_mask=1,
        ),
        SeatState(
            seat_id=1,
            role="button",
            is_hero=False,
            state="acted_all_in",
            committed_bb=8.0,
            stack_remaining_bb=0.0,
            blind_posted_bb=0.0,
            shover_rank=2,
            eligible_pot_mask=2,
        ),
        SeatState(
            seat_id=2,
            role="small_blind",
            is_hero=True,
            state="pending",
            committed_bb=0.5,
            stack_remaining_bb=7.5,
            blind_posted_bb=0.5,
            shover_rank=0,
            eligible_pot_mask=4,
        ),
        SeatState(
            seat_id=3,
            role="big_blind",
            is_hero=False,
            state="pending",
            committed_bb=1.0,
            stack_remaining_bb=7.0,
            blind_posted_bb=1.0,
            shover_rank=0,
            eligible_pot_mask=8,
        ),
    )
    base_game = AoFGameState(
        variant="NLH_AOF",
        table_size=4,
        hero_seat_id=2,
        hero_cards=(48, 49),
        action_history=((0, "ALL_IN"), (1, "ALL_IN")),
        seats=seats,
        small_blind_bb=0.5,
        big_blind_bb=1.0,
        effective_stack_bb=8.0,
        flat_rake_bb=0.0,
        pct_rake=0.0,
        rake_cap_bb=0.0,
        equal_stacks=True,
        side_pots_present=False,
    )
    swapped = AoFGameState(
        variant=base_game.variant,
        table_size=base_game.table_size,
        hero_seat_id=base_game.hero_seat_id,
        hero_cards=base_game.hero_cards,
        action_history=((1, "ALL_IN"), (0, "ALL_IN")),
        seats=base_game.seats,
        small_blind_bb=base_game.small_blind_bb,
        big_blind_bb=base_game.big_blind_bb,
        effective_stack_bb=base_game.effective_stack_bb,
        flat_rake_bb=base_game.flat_rake_bb,
        pct_rake=base_game.pct_rake,
        rake_cap_bb=base_game.rake_cap_bb,
        equal_stacks=base_game.equal_stacks,
        side_pots_present=base_game.side_pots_present,
    )
    o1 = encode_obs_v3(base_game)
    o2 = encode_obs_v3(swapped)
    assert not torch.allclose(o1.history_tokens, o2.history_tokens)


def test_no_board_features() -> None:
    """v3 tensors have fixed dims; no board block exists in ObsV3."""
    obs = obs_v3_dims()
    assert "board" not in obs
    rng = random.Random(0)
    state = reset_hand(rng, table_size=2, stack_bb=8.0)
    hero = state.action_seat
    assert hero is not None
    o = encode_obs_v3_from_hand(state, hero)
    assert o.hero_hand.numel() == HERO_HAND_DIM


@pytest.mark.parametrize(
    "factory,msg",
    [
        (lambda: _bad_table_size(), "table_size"),
        (lambda: _terminal(), "terminal"),
    ],
)
def test_malformed_rejection(factory, msg: str) -> None:
    state, hero = factory()
    with pytest.raises(AoFEncodeError, match=msg):
        normalize_hand_state(state, hero)


def test_more_than_three_prior_shovers(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = random.Random(0)
    state = reset_hand(rng, table_size=4, stack_bb=8.0)
    hero = state.action_seat
    assert hero is not None

    def _fake_prior(_state, _hero: str) -> tuple[str, ...]:
        return ("UTG", "BTN", "SB", "BB")

    monkeypatch.setattr(
        "aof_rl.game_state.all_ins_before_hero", _fake_prior
    )
    with pytest.raises(AoFEncodeError, match="prior shovers"):
        normalize_hand_state(state, hero)


def _bad_table_size() -> tuple:
    rng = random.Random(0)
    state = reset_hand(rng, table_size=4, stack_bb=8.0)
    state.table_size = 5
    return state, state.action_seat


def _terminal() -> tuple:
    rng = random.Random(0)
    state = reset_hand(rng, table_size=2, stack_bb=8.0)
    while not state.terminal:
        state = step_hand(state, ACTION_ALL_IN, payoff_samples=20, payoff_seed=1)
    hero = state.seats[0]
    return state, hero


def test_side_pot_flag_rejected() -> None:
    rng = random.Random(0)
    state = reset_hand(rng, table_size=4, stack_bb=8.0)
    hero = state.action_seat
    assert hero is not None
    with pytest.raises(AoFEncodeError, match="side pots"):
        normalize_hand_state(
            state, hero, equal_stacks=False, side_pots_present=True
        )


def test_encode_obs_v3_dispatch() -> None:
    rng = random.Random(11)
    state = reset_hand(rng, table_size=3, stack_bb=8.0)
    hero = state.action_seat
    assert hero is not None
    cache = NashTeacherCache()
    teacher = cache.get(state, hero, samples=100, seed=3)
    out = encode_obs(state, hero, teacher, version=OBS_VERSION_V3)
    assert isinstance(out, ObsV3)
    direct = encode_obs_v3_from_hand(state, hero)
    assert _obs_v3_tensors_equal(out, direct)
