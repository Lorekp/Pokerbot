#!/usr/bin/env python3
"""Compare AoF RL policy checkpoints (v1/v2 obs and teacher)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from aof_equity import warmup
from aof_ranges import active_seats
from aof_rl.checkpoint import load_policy_from_checkpoint
from aof_rl.encode import encode_obs
from aof_rl.env import ACTION_ALL_IN, ACTION_FOLD, AoFHandState, reset_hand, step_hand
from aof_rl.spot_diagnostics import SpotDiagnostics, record_decision
from aof_rl.teacher import NashTeacherCache

_COMPARE_PROFILES: dict[str, dict[str, int]] = {
    "default": {"teacher_samples": 800, "payoff_samples": 1200},
    "fast": {"teacher_samples": 200, "payoff_samples": 400},
}


def _resolve_device(name: str, *, allow_cpu: bool) -> str:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        if not allow_cpu:
            cuda_build = getattr(torch.version, "cuda", None)
            raise SystemExit(
                "CUDA requested but not available. "
                f"(torch {torch.__version__}, cuda={cuda_build}). "
                "Pass --allow-cpu or install a CUDA PyTorch wheel."
            )
        return "cpu"
    if name.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    return name


@dataclass
class PolicyRunStats:
    name: str
    checkpoint: str
    obs_version: int
    teacher_version: int
    bb_total: float = 0.0
    hands: int = 0
    teacher_agree: int = 0
    teacher_decisions: int = 0
    spot_kind_counts: Counter = field(default_factory=Counter)
    teacher_path_counts: Counter = field(default_factory=Counter)
    facing_two_count: int = 0
    facing_two_calls: int = 0
    facing_two_edge_sum: float = 0.0
    call_behind_count: int = 0
    call_behind_calls: int = 0
    call_behind_edge_sum: float = 0.0


def _policy_action(
    model,
    device: torch.device,
    state: AoFHandState,
    seat: str,
    cache: NashTeacherCache,
    samples: int,
    seed: int,
    obs_version: int,
) -> tuple[int, object]:
    tr = cache.get_with_path(state, seat, samples=samples, seed=seed)
    obs = encode_obs(
        state,
        seat,
        tr.teacher,
        version=obs_version,
        teacher_path=tr.path,
        provenance=tr.provenance,
    )
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        idx, _ = model.act(obs_t, deterministic=True)
    return int(idx), tr


def _roll_policy(
    *,
    name: str,
    checkpoint: str,
    hands: int,
    seed: int,
    stack_bb: float,
    teacher_samples: int,
    payoff_samples: int,
    obs_version: int | None,
    teacher_version: int | None,
    device: str,
) -> PolicyRunStats:
    dev = torch.device(device)
    model, meta = load_policy_from_checkpoint(
        checkpoint,
        dev,
        obs_version=obs_version,
        teacher_version=teacher_version,
    )
    o_ver = int(obs_version if obs_version is not None else meta.get("obs_version", 1))
    t_ver = int(teacher_version if teacher_version is not None else meta.get("teacher_version", 1))
    cache = NashTeacherCache(teacher_version=t_ver, use_fast=True)
    diag = SpotDiagnostics()
    rng = random.Random(seed)
    stats = PolicyRunStats(
        name=name,
        checkpoint=checkpoint,
        obs_version=o_ver,
        teacher_version=t_ver,
    )

    for h in range(hands):
        table_size = rng.choice((2, 3, 4))
        hero = rng.choice(active_seats(table_size))
        state = reset_hand(rng, table_size=table_size, stack_bb=stack_bb)
        cache.clear()
        guard = 0
        while not state.terminal:
            guard += 1
            seat = state.action_seat
            if seat is None:
                break
            t_seed = seed + h * 1000 + guard
            if seat == hero:
                action, tr = _policy_action(
                    model, dev, state, seat, cache, teacher_samples, t_seed, o_ver
                )
                teach_act = ACTION_ALL_IN if tr.teacher.action == "ALL IN" else ACTION_FOLD
                stats.teacher_decisions += 1
                if action == teach_act:
                    stats.teacher_agree += 1
                kind, _path = record_decision(
                    diag,
                    state,
                    hero,
                    teacher_path=tr.path,
                    teacher_version=t_ver,
                )
                stats.spot_kind_counts[kind] += 1
                stats.teacher_path_counts[tr.path] += 1
                edge = abs(tr.teacher.ev_primary - tr.teacher.ev_alt)
                if kind == "facing_two":
                    stats.facing_two_count += 1
                    stats.facing_two_edge_sum += edge
                    if action == ACTION_ALL_IN:
                        stats.facing_two_calls += 1
                if kind == "call_behind":
                    stats.call_behind_count += 1
                    stats.call_behind_edge_sum += edge
                    if action == ACTION_ALL_IN:
                        stats.call_behind_calls += 1
            else:
                tr = cache.get_with_path(state, seat, samples=teacher_samples, seed=t_seed)
                action = ACTION_ALL_IN if tr.teacher.action == "ALL IN" else ACTION_FOLD
            state = step_hand(state, action, payoff_samples=payoff_samples, payoff_seed=t_seed)

        stats.bb_total += state.payoffs.get(hero, 0.0)
        stats.hands += 1

    return stats


def _stats_to_dict(s: PolicyRunStats) -> dict:
    n = max(s.hands, 1)
    ft = max(s.facing_two_count, 1)
    cb = max(s.call_behind_count, 1)
    td = max(s.teacher_decisions, 1)
    return {
        "name": s.name,
        "checkpoint": s.checkpoint,
        "obs_version": s.obs_version,
        "teacher_version": s.teacher_version,
        "bb_per_100": 100.0 * s.bb_total / n,
        "action_agreement_with_teacher": s.teacher_agree / td,
        "spot_kind_counts": dict(s.spot_kind_counts),
        "teacher_path_counts": dict(s.teacher_path_counts),
        "facing_two_decisions": {
            "count": s.facing_two_count,
            "call_rate": s.facing_two_calls / ft if s.facing_two_count else 0.0,
            "avg_teacher_edge_bb": s.facing_two_edge_sum / ft if s.facing_two_count else 0.0,
        },
        "call_behind_decisions": {
            "count": s.call_behind_count,
            "call_rate": s.call_behind_calls / cb if s.call_behind_count else 0.0,
            "avg_teacher_edge_bb": s.call_behind_edge_sum / cb if s.call_behind_count else 0.0,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compare AoF policy checkpoints")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--names", nargs="*", default=())
    p.add_argument("--hands", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "-o",
        "--out",
        type=str,
        default="artifacts/aof_policy_version_comparison.json",
    )
    p.add_argument("--obs-version", type=int, choices=(1, 2), default=None)
    p.add_argument("--teacher-version", type=int, choices=(1, 2), default=None)
    p.add_argument("--device", default="cuda", help="cuda (default), cpu, or auto")
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Fall back to CPU when CUDA is unavailable (default: require CUDA).",
    )
    p.add_argument(
        "--profile",
        choices=tuple(_COMPARE_PROFILES.keys()),
        default="default",
        help="Sample budget preset (fast=validation smoke)",
    )
    p.add_argument("--teacher-samples", type=int, default=None)
    p.add_argument("--payoff-samples", type=int, default=None)
    args = p.parse_args()
    prof = _COMPARE_PROFILES[args.profile]
    teacher_samples = (
        args.teacher_samples
        if args.teacher_samples is not None
        else prof["teacher_samples"]
    )
    payoff_samples = (
        args.payoff_samples
        if args.payoff_samples is not None
        else prof["payoff_samples"]
    )
    device = _resolve_device(args.device, allow_cpu=args.allow_cpu)
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    warmup()
    names = list(args.names) or [f"policy_{i}" for i in range(len(args.checkpoints))]
    if len(names) != len(args.checkpoints):
        raise SystemExit("--names count must match --checkpoints")

    policies: list[dict] = []
    for name, ckpt in zip(names, args.checkpoints):
        st = _roll_policy(
            name=name,
            checkpoint=ckpt,
            hands=args.hands,
            seed=args.seed,
            stack_bb=8.0,
            teacher_samples=teacher_samples,
            payoff_samples=payoff_samples,
            obs_version=args.obs_version,
            teacher_version=args.teacher_version,
            device=device,
        )
        policies.append(_stats_to_dict(st))

    head_to_head: list[dict] = []
    if len(policies) >= 2:
        a, b = policies[0], policies[1]
        head_to_head.append(
            {
                "policy_a": a["name"],
                "policy_b": b["name"],
                "hands": args.hands,
                "bb_per_100_a_minus_b": a["bb_per_100"] - b["bb_per_100"],
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "seed": args.seed,
        "num_hands": args.hands,
        "profile": args.profile,
        "teacher_samples": teacher_samples,
        "payoff_samples": payoff_samples,
        "policies": policies,
        "head_to_head": head_to_head,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
