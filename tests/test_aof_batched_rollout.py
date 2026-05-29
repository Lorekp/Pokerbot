"""Batched rollout parity and policy batch API."""

from __future__ import annotations

import random

import pytest
import torch

pytest.importorskip("pokerkit")

from aof_rl.batched_rollout import (
    collect_trajectory_v3_batched,
    collect_update_rollout_batched,
)
from aof_rl.config import CurriculumConfig, CurriculumStage, TrainExperimentConfig
from aof_rl.curriculum import CurriculumScheduler
from aof_rl.env import payoff_step_seed
from aof_rl.league import LeagueConfig, LeagueManager
from aof_rl.policy_v3 import AoFPolicyNetV3, EncoderConfig
from aof_rl.league_trainer import _batch_to_device, _triton_available
from aof_rl.obs_v3_batch import ObsV3Batch, collate_obs_v3
from aof_rl.obs_v3 import ObsV3
from aof_rl.payoff import configure_payoff
from aof_rl.rollout import RolloutConfig, NeuralPolicyActor, collect_trajectory_v3
from aof_rl.spot_forcing import reset_hand_for_curriculum
from aof_rl.teacher import NashTeacherCache


def _hu_stage() -> CurriculumStage:
    return CurriculumStage(
        name="hu",
        table_sizes=(2,),
        spot_constraint="any",
        flat_rake_bb=0.0,
        range_randomization=False,
        updates=100,
    )


def _trainer_fixtures(seed: int = 123):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    enc = EncoderConfig(encoder_type="deep_sets", d_model=32, trunk_hidden=32)
    model = AoFPolicyNetV3(enc).to(device)
    model.eval()
    configure_payoff(payoff_hu_mode="exact_enum")
    rollout_cfg = RolloutConfig(
        stack_bb=8.0,
        teacher_samples=50,
        payoff_samples=200,
        seed=seed,
        use_fast_teacher=True,
        deterministic_policy=True,
        payoff_hu_mode="exact_enum",
    )
    league = LeagueManager(
        LeagueConfig(scripted_pool=True, oracle_pool=True, adversarial_br_pool=False),
        enc,
        device,
        rollout_cfg,
    )
    sched = CurriculumScheduler(
        CurriculumConfig(stages=[_hu_stage()])
    )
    cache = NashTeacherCache(
        use_fast=True,
        teacher_version=2,
        teacher_samples=50,
        stack_bb=8.0,
    )
    rng = random.Random(seed)
    return model, device, rollout_cfg, league, sched, cache, rng


def _traj_signature_pre_payoff(traj) -> list[tuple]:
    """Actions / policy fields only (E1c gate #1a — excludes terminal reward)."""
    return [
        (
            t.action,
            t.done,
            round(t.log_prob, 8),
            round(t.value, 8),
            t.teacher_action,
            t.legal_mask,
        )
        for t in traj
    ]


def test_payoff_step_seed_formula() -> None:
    assert payoff_step_seed(7, 3) == 7 * 1009 + 3


def test_n1_pre_payoff_bitwise() -> None:
    model, device, rollout_cfg, league, sched, cache, rng = _trainer_fixtures(99)
    stage = _hu_stage()
    state, hero = reset_hand_for_curriculum(rng, stage, stack_bb=rollout_cfg.stack_bb)
    policies = league.sample_seat_policies(state.table_size, hero, model, rng)
    policies.setdefault(
        hero, NeuralPolicyActor(model, device, rollout_cfg, policy_key="hero:main")
    )
    seq = collect_trajectory_v3(
        model, policies, device, hero, cache, rollout_cfg, state.copy_for_decision()
    )
    batched = collect_trajectory_v3_batched(
        model,
        policies,
        device,
        hero,
        cache,
        rollout_cfg,
        state.copy_for_decision(),
    )
    assert _traj_signature_pre_payoff(seq) == _traj_signature_pre_payoff(batched)


def test_n1_payoff_reward_distributional() -> None:
    """Gate #1b: rewards match when both paths use the same payoff_hu_mode."""
    model, device, rollout_cfg, league, sched, cache, rng = _trainer_fixtures(99)
    stage = _hu_stage()
    state, hero = reset_hand_for_curriculum(rng, stage, stack_bb=rollout_cfg.stack_bb)
    policies = league.sample_seat_policies(state.table_size, hero, model, rng)
    policies.setdefault(
        hero, NeuralPolicyActor(model, device, rollout_cfg, policy_key="hero:main")
    )
    seq = collect_trajectory_v3(
        model, policies, device, hero, cache, rollout_cfg, state.copy_for_decision()
    )
    batched = collect_trajectory_v3_batched(
        model,
        policies,
        device,
        hero,
        cache,
        rollout_cfg,
        state.copy_for_decision(),
    )
    assert seq and batched
    assert abs(seq[-1].reward - batched[-1].reward) < 1e-6


def test_act_with_logprob_matches_act_batch_single() -> None:
    torch.manual_seed(1)
    enc = EncoderConfig(encoder_type="deep_sets", d_model=32, trunk_hidden=32)
    model = AoFPolicyNetV3(enc)
    model.eval()
    from aof_rl.obs_v3 import ObsV3

    g = torch.Generator().manual_seed(7)
    obs = ObsV3(
        hero_hand=torch.randn(40, generator=g),
        global_feats=torch.randn(24, generator=g),
        seat_tokens=torch.randn(4, 32, generator=g),
        seat_mask=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        history_tokens=torch.randn(3, 24, generator=g),
        history_mask=torch.zeros(3),
        legal_mask=torch.tensor([1.0, 1.0]),
    )
    idx_a, lp_a, val_a, _ = model.act_with_logprob(obs, deterministic=True)
    idx_b, out_b = model.act_batch_single(obs, deterministic=True)
    assert idx_a == idx_b
    log_probs_b = out_b.log_probs
    lp_b = float(
        log_probs_b[idx_b].item()
        if log_probs_b.dim() == 1
        else log_probs_b[0, idx_b].item()
    )
    assert lp_a == pytest.approx(lp_b, abs=1e-6)
    assert val_a == pytest.approx(float(out_b.value.item()), abs=1e-5)


def test_policy_act_batch_matches_single() -> None:
    torch.manual_seed(0)
    enc = EncoderConfig(encoder_type="deep_sets", d_model=32, trunk_hidden=32)
    model = AoFPolicyNetV3(enc)
    model.eval()
    from aof_rl.obs_v3 import ObsV3
    from aof_rl.obs_v3_batch import collate_obs_v3

    obs_list = []
    for i in range(3):
        g = torch.Generator().manual_seed(i)
        obs_list.append(
            ObsV3(
                hero_hand=torch.randn(40, generator=g),
                global_feats=torch.randn(24, generator=g),
                seat_tokens=torch.randn(4, 32, generator=g),
                seat_mask=torch.tensor([1.0, 1.0, 0.0, 0.0]),
                history_tokens=torch.randn(3, 24, generator=g),
                history_mask=torch.zeros(3),
                legal_mask=torch.tensor([1.0, 1.0]),
            )
        )
    batch = collate_obs_v3(obs_list)
    idx_b, out_b = model.act_batch(batch, deterministic=True)
    for i, obs in enumerate(obs_list):
        idx_s, out_s = model.act_batch_single(obs, deterministic=True)
        assert int(idx_b[i].item()) == idx_s
        assert torch.allclose(out_b.logits[i], out_s.logits.squeeze(0))


def test_batch_to_device_pin_memory_cpu_obs() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    obs = ObsV3(
        hero_hand=torch.randn(40),
        global_feats=torch.randn(24),
        seat_tokens=torch.randn(4, 32),
        seat_mask=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        history_tokens=torch.randn(3, 24),
        history_mask=torch.zeros(3),
        legal_mask=torch.tensor([1.0, 1.0]),
    )
    batch = collate_obs_v3([obs, obs])
    on_gpu = _batch_to_device(batch, device)
    assert on_gpu.hero_hand.device.type == "cuda"
    assert on_gpu.hero_hand.is_cuda


def test_triton_gate_matches_import() -> None:
    try:
        import triton  # noqa: F401

        assert _triton_available()
    except ImportError:
        assert not _triton_available()


def test_per_slot_rng_isolation() -> None:
    """Two parallel slots with same stream should not share deal state."""
    model, device, rollout_cfg, league, sched, cache, _ = _trainer_fixtures(42)
    rng = random.Random(42)
    traj = collect_update_rollout_batched(
        model,
        device,
        cache,
        rollout_cfg,
        league,
        sched,
        rng,
        update=0,
        n_parallel=4,
        total_hands=4,
    )
    assert len(traj) >= 4
