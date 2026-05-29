#!/usr/bin/env python3
"""Benchmark batched villain rollout: mean batch occupancy, eager vs compile."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from aof_rl.batched_rollout import BatchedHandCollector, collect_update_rollout_batched
from aof_rl.config import CurriculumConfig, CurriculumStage, load_train_config
from aof_rl.curriculum import CurriculumScheduler
from aof_rl.league import LeagueManager
from aof_rl.policy_v3 import AoFPolicyNetV3
from aof_rl.rollout import RolloutConfig
from aof_rl.teacher import NashTeacherCache
import random


def _run_once(
    *,
    parallel: int,
    total_hands: int,
    compile_villain: bool,
    compile_hero: bool,
    device: torch.device,
) -> tuple[float, float]:
    enc_cfg = load_train_config("configs/aof_v3_ppo_league.yaml").model.to_encoder_config()
    model = AoFPolicyNetV3(enc_cfg).to(device)
    model.eval()
    rollout_cfg = RolloutConfig(deterministic_policy=True, payoff_samples=400, teacher_samples=100)
    league = LeagueManager(
        load_train_config("configs/aof_v3_ppo_league.yaml").league,
        enc_cfg,
        device,
        rollout_cfg,
    )
    sched = CurriculumScheduler(
        CurriculumConfig(
            stages=[
                CurriculumStage(
                    name="hu",
                    table_sizes=(2,),
                    spot_constraint="any",
                    updates=999,
                )
            ]
        )
    )
    cache = NashTeacherCache(use_fast=True, teacher_version=2, stack_bb=8.0)
    rng = random.Random(0)
    t0 = time.perf_counter()
    collector_holder: list[BatchedHandCollector] = []

    class _Capture:
        pass

    orig_init = BatchedHandCollector.__init__

    def _wrap_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        collector_holder.append(self)

    BatchedHandCollector.__init__ = _wrap_init  # type: ignore[method-assign]
    try:
        collect_update_rollout_batched(
            model,
            device,
            cache,
            rollout_cfg,
            league,
            sched,
            rng,
            update=0,
            n_parallel=parallel,
            total_hands=total_hands,
            pad_batch_size=parallel,
            torch_compile_villain=compile_villain,
            torch_compile_hero=compile_hero,
        )
    finally:
        BatchedHandCollector.__init__ = orig_init  # type: ignore[method-assign]

    elapsed = time.perf_counter() - t0
    mean_occ = collector_holder[0].occupancy.mean_batch_size() if collector_holder else 0.0
    return elapsed, mean_occ


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parallel", type=int, default=64)
    p.add_argument("--total-hands", type=int, default=128)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    cache_dir = _ROOT / ".torch_inductor_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))

    if not torch.cuda.is_available():
        print("CUDA not available; using CPU (timing not representative)")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    print(f"parallel={args.parallel} total_hands={args.total_hands} device={device}")
    e_eager, occ_eager = _run_once(
        parallel=args.parallel,
        total_hands=args.total_hands,
        compile_villain=False,
        compile_hero=False,
        device=device,
    )
    print(f"eager: {e_eager:.3f}s  mean_batch={occ_eager:.2f}")

    if args.compile and device.type == "cuda":
        try:
            import triton  # noqa: F401
        except ImportError:
            print("triton missing; skipping compile benchmark")
            return
        _run_once(
            parallel=args.parallel,
            total_hands=min(8, args.total_hands),
            compile_villain=True,
            compile_hero=True,
            device=device,
        )
        e_comp, occ_comp = _run_once(
            parallel=args.parallel,
            total_hands=args.total_hands,
            compile_villain=True,
            compile_hero=True,
            device=device,
        )
        print(f"compiled: {e_comp:.3f}s  mean_batch={occ_comp:.2f}")
        if occ_eager > 0:
            print(
                f"note: at batch_hands={args.total_hands}, "
                f"mean occupancy / parallel = {occ_eager / args.parallel:.2%}"
            )


if __name__ == "__main__":
    main()
