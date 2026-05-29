"""Per-hand timing buckets for AoF RL profiling."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import torch

from aof_rl.encode import encode_obs
from aof_rl.env import ACTION_FOLD, reset_hand, step_hand
from aof_rl.policy import AoFPolicyNet
from aof_rl.teacher import NashTeacherCache


@dataclass
class HandTiming:
    teacher_sec: float = 0.0
    policy_sec: float = 0.0
    payoff_sec: float = 0.0
    steps: int = 0


@dataclass
class InstrumentSummary:
    hands: int = 0
    teacher_sec: float = 0.0
    policy_sec: float = 0.0
    payoff_sec: float = 0.0
    total_sec: float = 0.0
    buckets: list[HandTiming] = field(default_factory=list)

    def to_dict(self) -> dict:
        n = max(self.hands, 1)
        return {
            "hands": self.hands,
            "total_sec": round(self.total_sec, 3),
            "teacher_sec": round(self.teacher_sec, 3),
            "policy_sec": round(self.policy_sec, 3),
            "payoff_sec": round(self.payoff_sec, 3),
            "teacher_pct": round(100.0 * self.teacher_sec / max(self.total_sec, 1e-9), 1),
            "policy_pct": round(100.0 * self.policy_sec / max(self.total_sec, 1e-9), 1),
            "payoff_pct": round(100.0 * self.payoff_sec / max(self.total_sec, 1e-9), 1),
            "sec_per_hand": round(self.total_sec / n, 3),
        }


def instrument_one_hand(
    model: AoFPolicyNet,
    device: torch.device,
    cache: NashTeacherCache,
    *,
    teacher_samples: int,
    payoff_samples: int,
    seed: int,
    table_size: int = 4,
    stack_bb: float = 8.0,
) -> HandTiming:
    """Run one hand with timing buckets (teacher / policy / payoff)."""
    rng = random.Random(seed)
    state = reset_hand(rng, table_size=table_size, stack_bb=stack_bb)
    timing = HandTiming()
    guard = 0

    while not state.terminal:
        guard += 1
        if guard > 32:
            break
        seat = state.action_seat
        if seat is None:
            break

        t0 = time.perf_counter()
        teacher = cache.get(state, seat, samples=teacher_samples, seed=seed + state.hand_id)
        timing.teacher_sec += time.perf_counter() - t0

        obs = encode_obs(state, seat, teacher)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model(obs_t)
            dist = torch.distributions.Categorical(logits=out.logits)
            action = int(dist.sample().item())
        timing.policy_sec += time.perf_counter() - t1

        t2 = time.perf_counter()
        state = step_hand(
            state,
            action,
            payoff_samples=payoff_samples,
            payoff_seed=seed + state.hand_id * 17 + guard,
        )
        timing.payoff_sec += time.perf_counter() - t2
        timing.steps += 1

    return timing


def run_instrumentation(
    model: AoFPolicyNet,
    device: torch.device,
    *,
    hands: int,
    teacher_samples: int,
    payoff_samples: int,
    seed: int,
    use_fast_teacher: bool,
) -> InstrumentSummary:
    from aof_equity import warmup

    warmup()
    cache = NashTeacherCache(use_fast=use_fast_teacher)
    summary = InstrumentSummary()
    t_all = time.perf_counter()
    for h in range(hands):
        ht = instrument_one_hand(
            model,
            device,
            cache,
            teacher_samples=teacher_samples,
            payoff_samples=payoff_samples,
            seed=seed + h,
            table_size=4,
        )
        summary.buckets.append(ht)
        summary.teacher_sec += ht.teacher_sec
        summary.policy_sec += ht.policy_sec
        summary.payoff_sec += ht.payoff_sec
        summary.hands += 1
    summary.total_sec = time.perf_counter() - t_all
    return summary
