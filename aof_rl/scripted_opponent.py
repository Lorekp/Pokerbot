"""Fixed-profile opponents for falsifiable RL evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import torch

from aof_rl.env import ACTION_ALL_IN, ACTION_FOLD, AoFHandState, reset_hand, step_hand
from aof_rl.encode import encode_obs
from aof_rl.policy import AoFPolicyNet
from aof_rl.teacher import NashTeacherCache


@dataclass
class ScriptedProfile:
    """Villain calls an open shove with fixed probability (else fold)."""

    call_prob: float = 0.55
    name: str = "loose_caller"


def play_hand_scripted(
    hero_policy: Callable[[AoFHandState, str], int],
    villain_profile: ScriptedProfile,
    rng: random.Random,
    *,
    table_size: int = 4,
    stack_bb: float = 8.0,
    payoff_samples: int = 4000,
    hero_seat: str | None = None,
) -> tuple[float, str]:
    """Return (hero_bb_profit, hero_seat)."""
    state = reset_hand(rng, table_size=table_size, stack_bb=stack_bb)
    if hero_seat is None:
        hero_seat = rng.choice(state.seats)

    def villain_fn(st: AoFHandState, seat: str) -> int:
        if seat == hero_seat:
            return hero_policy(st, seat)
        if st.facing_shove() or st.shover is not None:
            return ACTION_ALL_IN if rng.random() < villain_profile.call_prob else ACTION_FOLD
        return ACTION_ALL_IN if rng.random() < min(villain_profile.call_prob * 0.85, 0.99) else ACTION_FOLD

    guard = 0
    while not state.terminal:
        guard += 1
        if guard > 32:
            raise RuntimeError("hand guard")
        seat = state.action_seat
        if seat is None:
            break
        action = villain_fn(state, seat)
        state = step_hand(
            state, action, payoff_samples=payoff_samples, payoff_seed=guard
        )
    return state.payoffs.get(hero_seat, 0.0), hero_seat


def eval_hero_modes(
    model: AoFPolicyNet,
    device: torch.device,
    villain_profile: ScriptedProfile,
    *,
    hands: int,
    seed: int,
    teacher_samples: int = 2000,
    table_size: int = 4,
) -> dict[str, float]:
    """Compare hero policy vs always following NashTeacher.action."""
    rng = random.Random(seed)
    cache = NashTeacherCache()
    policy_bb = 0.0
    teacher_bb = 0.0

    def policy_fn(st: AoFHandState, seat: str) -> int:
        teacher = cache.get(st, seat, samples=teacher_samples, seed=seed)
        obs = encode_obs(st, seat, teacher)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            idx, _ = model.act(obs_t, deterministic=True)
        return int(idx)

    def teacher_fn(st: AoFHandState, seat: str) -> int:
        teacher = cache.get(st, seat, samples=teacher_samples, seed=seed)
        return ACTION_ALL_IN if teacher.action == "ALL IN" else ACTION_FOLD

    for h in range(hands):
        hero_seat = rng.choice(["SB", "BB", "BTN", "UTG"][:table_size])
        cache.clear()
        bb, _ = play_hand_scripted(
            policy_fn,
            villain_profile,
            rng,
            table_size=table_size,
            hero_seat=hero_seat,
            payoff_samples=4000,
        )
        policy_bb += bb
        cache.clear()
        bb2, _ = play_hand_scripted(
            teacher_fn,
            villain_profile,
            rng,
            table_size=table_size,
            hero_seat=hero_seat,
            payoff_samples=4000,
        )
        teacher_bb += bb2

    n = max(hands, 1)
    return {
        "policy_bb_per_hand": policy_bb / n,
        "teacher_bb_per_hand": teacher_bb / n,
        "hands": hands,
    }
