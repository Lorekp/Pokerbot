"""V3 rollout collection for PPO league training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

import torch

from aof_rl.curriculum import CurriculumStage
from aof_rl.encode_v3 import encode_obs_v3_from_hand
from aof_rl.env import ACTION_FOLD, AoFHandState, payoff_step_seed, step_hand
from aof_rl.obs_v3 import ObsV3
from aof_rl.policy_v3 import AoFPolicyNetV3, PolicyOutputV3
from aof_rl.spot_forcing import reset_hand_for_curriculum
from aof_rl.profiling.league_profiler import get_profiler
from aof_rl.teacher import NashTeacherCache
from aof_rl.teacher_targets import teacher_action_index, teacher_q_bb


def _prof():
    return get_profiler()


def _rollout_autocast(device: torch.device, enabled: bool):
    from contextlib import nullcontext

    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _obs_for_transition(obs: ObsV3, device: torch.device, store_on_cpu: bool) -> ObsV3:
    if store_on_cpu:
        return ObsV3(
            hero_hand=obs.hero_hand.detach().cpu(),
            global_feats=obs.global_feats.detach().cpu(),
            seat_tokens=obs.seat_tokens.detach().cpu(),
            seat_mask=obs.seat_mask.detach().cpu(),
            history_tokens=obs.history_tokens.detach().cpu(),
            history_mask=obs.history_mask.detach().cpu(),
            legal_mask=obs.legal_mask.detach().cpu(),
        )
    return obs


@dataclass
class RolloutConfig:
    stack_bb: float = 8.0
    teacher_samples: int = 300
    payoff_samples: int = 1200
    seed: int = 42
    teacher_version: int = 2
    use_fast_teacher: bool = True
    teacher_margin_bb: float = 0.4
    teacher_min_samples: int = 200
    deterministic_policy: bool = False
    flat_rake_bb: float = 0.0
    store_obs_on_cpu: bool = False
    mixed_precision: bool = False
    payoff_hu_mode: str = "table"
    hu_showdown_table_path: str = "data/aof/hu_showdown_equity.npz"


@dataclass
class StepTransition:
    obs: ObsV3
    action: int
    log_prob: float
    value: float
    q_bb: tuple[float, float]
    uncertainty: float
    teacher_action: int
    teacher_q: tuple[float, float]
    teacher_edge: float
    legal_mask: tuple[float, float]
    reward: float = 0.0
    done: bool = False


class PolicyActor(Protocol):
    def act(
        self, state: AoFHandState, seat: str, cache: NashTeacherCache
    ) -> int: ...


@dataclass
class NeuralPolicyActor:
    model: AoFPolicyNetV3
    device: torch.device
    cfg: RolloutConfig
    policy_key: str = "neural:unknown"

    def act(
        self, state: AoFHandState, seat: str, cache: NashTeacherCache
    ) -> int:
        obs = encode_obs_v3_from_hand(state, seat, device=self.device)
        with _rollout_autocast(self.device, self.cfg.mixed_precision):
            with torch.no_grad():
                idx, _ = self.model.act(obs, deterministic=self.cfg.deterministic_policy)
        return int(idx)


@dataclass
class OraclePolicyActor:
    cfg: RolloutConfig
    policy_key: str = "oracle:oracle"

    def act(
        self, state: AoFHandState, seat: str, cache: NashTeacherCache
    ) -> int:
        t = cache.get(
            state,
            seat,
            samples=self.cfg.teacher_samples,
            seed=self.cfg.seed + state.hand_id,
        )
        return teacher_action_index(t)


@dataclass
class ScriptedPolicyActor:
    call_prob: float = 0.55
    seed: int = 0
    policy_key: str = "scripted:scripted"

    def act(
        self, state: AoFHandState, seat: str, cache: NashTeacherCache
    ) -> int:
        rng = random.Random(self.seed + state.hand_id + hash(seat) % 1000)
        from aof_rl.env import ACTION_ALL_IN

        if state.facing_shove() or state.shover is not None:
            return ACTION_ALL_IN if rng.random() < self.call_prob else ACTION_FOLD
        return ACTION_ALL_IN if rng.random() < min(self.call_prob * 0.85, 0.99) else ACTION_FOLD


def policy_key_for_actor(actor: PolicyActor, *, hero: bool = False) -> str:
    if hero:
        return "hero:main"
    key = getattr(actor, "policy_key", None)
    if key:
        return str(key)
    return f"{type(actor).__name__}:unknown"


def _hero_teacher_lookup(
    state: AoFHandState,
    hero: str,
    cache: NashTeacherCache,
    cfg: RolloutConfig,
):
    prof = _prof()
    if prof is not None:
        with prof.measure("rollout.hero_teacher"):
            return cache.get_with_path(
                state, hero, samples=cfg.teacher_samples, seed=cfg.seed + state.hand_id
            )
    return cache.get_with_path(
        state, hero, samples=cfg.teacher_samples, seed=cfg.seed + state.hand_id
    )


def _hero_step_from_output(
    obs_cpu: ObsV3,
    idx: int,
    out: PolicyOutputV3,
    teacher,
    *,
    log_prob: float | None = None,
    value: float | None = None,
) -> StepTransition:
    if log_prob is not None:
        lp = log_prob
    else:
        lp_t = out.log_probs
        if lp_t.dim() == 1:
            lp = float(lp_t[idx].item())
        else:
            lp = float(lp_t[0, idx].item())
    if value is not None:
        val = value
    elif out.value.dim() == 0:
        val = float(out.value.item())
    else:
        val = float(out.value[0].item())
    q = out.q_bb
    if q.dim() > 1:
        q = q[0]
    unc = out.uncertainty
    if unc.dim() > 0:
        unc = unc[0]
    unc = float(unc.item())
    tq = teacher_q_bb(teacher)
    legal = tuple(float(x) for x in obs_cpu.legal_mask.detach().cpu().tolist())
    return StepTransition(
        obs=obs_cpu,
        action=idx,
        log_prob=lp,
        value=val,
        q_bb=(float(q[0].item()), float(q[1].item())),
        uncertainty=unc,
        teacher_action=teacher_action_index(teacher),
        teacher_q=(float(tq[0]), float(tq[1])),
        teacher_edge=float(teacher.edge_bb),
        legal_mask=legal,
    )


def _record_hero_step(
    state: AoFHandState,
    hero: str,
    model: AoFPolicyNetV3,
    device: torch.device,
    cache: NashTeacherCache,
    cfg: RolloutConfig,
) -> StepTransition | None:
    if state.terminal or state.action_seat != hero:
        return None
    t_result = _hero_teacher_lookup(state, hero, cache, cfg)
    teacher = t_result.teacher
    prof = _prof()
    if prof is not None:
        with prof.measure("rollout.hero_encode"):
            obs_d = encode_obs_v3_from_hand(state, hero, device=device)
    else:
        obs_d = encode_obs_v3_from_hand(state, hero, device=device)
    obs_store = _obs_for_transition(obs_d, device, cfg.store_obs_on_cpu)
    with _rollout_autocast(device, cfg.mixed_precision):
        with torch.no_grad():
            if prof is not None:
                with prof.measure("rollout.hero_policy_forward"):
                    idx, lp, val, out = model.act_with_logprob(
                        obs_d, deterministic=cfg.deterministic_policy
                    )
            else:
                idx, lp, val, out = model.act_with_logprob(
                    obs_d, deterministic=cfg.deterministic_policy
                )
    return _hero_step_from_output(
        obs_store, idx, out, teacher, log_prob=lp, value=val
    )


def collect_trajectory_v3(
    hero_model: AoFPolicyNetV3,
    seat_policies: dict[str, PolicyActor],
    device: torch.device,
    hero_seat: str,
    cache: NashTeacherCache,
    cfg: RolloutConfig,
    state: AoFHandState,
) -> list[StepTransition]:
    traj: list[StepTransition] = []
    guard = 0
    while not state.terminal:
        guard += 1
        if guard > 32:
            raise RuntimeError("Hand guard exceeded")
        seat = state.action_seat
        if seat is None:
            break
        actor = seat_policies.get(seat)
        if seat == hero_seat:
            step = _record_hero_step(state, hero_seat, hero_model, device, cache, cfg)
            if step is None:
                break
            action = step.action
            traj.append(step)
        elif actor is not None:
            prof = _prof()
            if prof is not None:
                with prof.measure("rollout.villain_act"):
                    action = actor.act(state, seat, cache)
            else:
                action = actor.act(state, seat, cache)
        else:
            action = ACTION_FOLD
        state = step_hand(
            state,
            action,
            payoff_samples=cfg.payoff_samples,
            payoff_seed=payoff_step_seed(state.hand_id, guard),
        )
    reward = state.payoffs.get(hero_seat, 0.0)
    if traj:
        traj[-1].reward = reward
        traj[-1].done = True
    return traj


def collect_hand_v3(
    hero_model: AoFPolicyNetV3,
    seat_policies: dict[str, PolicyActor],
    device: torch.device,
    cache: NashTeacherCache,
    cfg: RolloutConfig,
    rng: random.Random,
    stage: CurriculumStage,
) -> list[StepTransition]:
    prof = _prof()
    if prof is not None:
        with prof.measure("rollout.reset_curriculum"):
            state, hero_seat = reset_hand_for_curriculum(
                rng, stage, stack_bb=cfg.stack_bb
            )
    else:
        state, hero_seat = reset_hand_for_curriculum(
            rng, stage, stack_bb=cfg.stack_bb
        )
    prof = _prof()
    if prof is not None:
        prof.inc_counter(f"table_size.{state.table_size}")
    policies = dict(seat_policies)
    policies.setdefault(
        hero_seat,
        NeuralPolicyActor(hero_model, device, cfg),
    )
    return collect_trajectory_v3(
        hero_model, policies, device, hero_seat, cache, cfg, state
    )
