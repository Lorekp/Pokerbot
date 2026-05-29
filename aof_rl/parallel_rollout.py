"""Process-pool rollout workers for AoF self-play."""

from __future__ import annotations

import os
import random
from typing import Any

import torch

from aof_ranges import active_seats
from aof_rl.self_play import AoFSelfPlayConfig, collect_trajectory
from aof_rl.policy import AoFPolicyNet
from aof_rl.teacher import NashTeacherCache

_WARMED = False


def _worker_init() -> None:
    global _WARMED
    os.environ["NUMBA_NUM_THREADS"] = "1"
    from aof_equity import warmup

    warmup()
    _WARMED = True


def _cfg_from_dict(d: dict[str, Any]) -> AoFSelfPlayConfig:
    from pathlib import Path

    d = dict(d)
    if "checkpoint_dir" in d:
        d["checkpoint_dir"] = Path(d["checkpoint_dir"])
    if "table_sizes" in d:
        d["table_sizes"] = tuple(d["table_sizes"])
    return AoFSelfPlayConfig(**d)


def _collect_hands_task(args: tuple) -> list[dict]:
    """Collect a chunk of hands in one worker (amortize model load + JIT)."""
    state_dict, hidden, cfg_dict, seed, hand_specs = args
    cfg = _cfg_from_dict(cfg_dict)
    device = torch.device("cpu")
    in_features = int(state_dict["trunk.0.weight"].shape[1])
    model = AoFPolicyNet(hidden=hidden, obs_dim=in_features).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    cache = NashTeacherCache(
        use_fast=cfg.use_fast_teacher,
        teacher_margin_bb=cfg.teacher_margin_bb,
        teacher_min_samples=cfg.teacher_min_samples,
        max_entries=cfg.teacher_cache_max_entries,
        teacher_version=cfg.teacher_version,
    )
    out: list[dict] = []
    for hand_index, table_size, hero_seat in hand_specs:
        rng = random.Random(seed + hand_index * 9973)
        out.extend(
            collect_trajectory(
                model,
                None,
                device,
                hero_seat,
                rng,
                cache,
                cfg,
                table_size=table_size,
            )
        )
    return out


def collect_trajectories_parallel(
    model: AoFPolicyNet,
    cfg: AoFSelfPlayConfig,
    n_hands: int,
    rng: random.Random,
    *,
    workers: int,
) -> list[dict]:
    """Collect n_hands trajectories across a process pool (chunked per worker)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    hidden = int(state_dict["trunk.0.weight"].shape[0])
    cfg_dict = {
        "hands_per_update": cfg.hands_per_update,
        "total_updates": cfg.total_updates,
        "lr": cfg.lr,
        "device": "cpu",
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "seed": cfg.seed,
        "table_sizes": list(cfg.table_sizes),
        "stack_bb": cfg.stack_bb,
        "teacher_samples": cfg.teacher_samples,
        "payoff_samples": cfg.payoff_samples,
        "league_size": 0,
        "league_snapshot_every": cfg.league_snapshot_every,
        "lambda_teacher": cfg.lambda_teacher,
        "ppo_epochs": cfg.ppo_epochs,
        "max_wall_seconds": cfg.max_wall_seconds,
        "use_fast_teacher": cfg.use_fast_teacher,
        "teacher_margin_bb": cfg.teacher_margin_bb,
        "teacher_min_samples": cfg.teacher_min_samples,
        "torch_compile": False,
        "retain_teacher_cache": cfg.retain_teacher_cache,
        "teacher_cache_max_entries": cfg.teacher_cache_max_entries,
        "rollout_workers": 0,
        "obs_version": cfg.obs_version,
        "teacher_version": cfg.teacher_version,
    }

    hand_specs: list[tuple[int, int, str]] = []
    for i in range(n_hands):
        table_size = rng.choice(cfg.table_sizes)
        hero_seat = rng.choice(active_seats(table_size))
        hand_specs.append((i, table_size, hero_seat))

    n_workers = max(1, min(workers, n_hands, 6))
    chunk = (n_hands + n_workers - 1) // n_workers
    tasks: list[tuple] = []
    for w in range(n_workers):
        start = w * chunk
        end = min(start + chunk, n_hands)
        if start >= end:
            continue
        tasks.append((state_dict, hidden, cfg_dict, cfg.seed, hand_specs[start:end]))

    all_traj: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as pool:
        futures = [pool.submit(_collect_hands_task, t) for t in tasks]
        for fut in as_completed(futures):
            all_traj.extend(fut.result())
    return all_traj
