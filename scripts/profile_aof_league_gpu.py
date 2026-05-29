#!/usr/bin/env python3
"""Profile AoF PPO league training on GPU (~10 min wall-clock target)."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from aof_rl.config import TrainExperimentConfig, load_train_config
from aof_rl.league_trainer import train_ppo_league
from aof_rl.profiling.league_profiler import (
    LeagueProfilerReport,
    get_cprofile_stats,
    get_last_timings,
    set_cprofile_update,
    write_cprofile_text,
)


def _gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "none"


def _run_train(cfg: TrainExperimentConfig) -> LeagueProfilerReport:
    cfg.train.profile = True
    train_ppo_league(cfg)
    prof = get_last_timings()
    if prof is None:
        raise RuntimeError("profile timings missing after train_ppo_league")
    report = LeagueProfilerReport(
        timings=prof,
        updates=int(prof.counters.get("updates", cfg.train.total_updates)),
        hands=int(prof.counters.get("hands", 0)),
        transitions=int(prof.counters.get("transitions", 0)),
        cprofile_top=get_cprofile_stats(),
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="GPU profile AoF PPO league training")
    p.add_argument(
        "--config",
        default="configs/aof_v3_ppo_league_profile.yaml",
        help="Base YAML (profile flag forced on)",
    )
    p.add_argument("--target-seconds", type=float, default=600.0)
    p.add_argument("--probe-updates", type=int, default=2)
    p.add_argument("--out-dir", default="artifacts/profile")
    p.add_argument("--dry-run", action="store_true", help="Probe only, no full run")
    p.add_argument(
        "--total-updates",
        type=int,
        default=None,
        help="Fixed update count (skips probe calibration)",
    )
    p.add_argument(
        "--cprofile-update",
        type=int,
        default=None,
        help="Run cProfile on this update index (e.g. 0)",
    )
    args = p.parse_args()

    cache_dir = _ROOT / ".torch_inductor_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))

    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA not available. Use Python 3.13 + cu126 torch for GPU profiling.",
            file=sys.stderr,
        )
        sys.exit(1)

    base = load_train_config(_ROOT / args.config)
    base.train.device = "cuda"
    base.train.allow_cpu = False
    base.train.profile = True
    try:
        import triton  # noqa: F401

        base.train.torch_compile = True
    except ImportError:
        base.train.torch_compile = False
        print("triton not found; torch.compile disabled for profile run", flush=True)
    base.validation.paired_eval_hands = 0
    base.validation.br_exploit_hands = 0
    base.league.adversarial_br_pool = False

    print(f"GPU: {_gpu_name()}", flush=True)
    set_cprofile_update(args.cprofile_update)
    if args.cprofile_update is not None:
        print(f"cProfile enabled for update {args.cprofile_update}", flush=True)

    probe_wall = 0.0
    probe_per_update = 0.0
    if args.total_updates is None:
        probe_cfg = copy.deepcopy(base)
        probe_cfg.train.total_updates = args.probe_updates
        probe_cfg.checkpoint_path = str(
            Path(base.checkpoint_path).with_name("profile_probe.pt")
        )
        print(f"Probe: {args.probe_updates} updates...", flush=True)
        t0 = time.perf_counter()
        _run_train(probe_cfg)
        probe_wall = time.perf_counter() - t0
        probe_per_update = probe_wall / max(args.probe_updates, 1)
        print(
            f"Probe wall={probe_wall:.1f}s sec/update={probe_per_update:.2f}",
            flush=True,
        )
        if args.dry_run:
            return
        target_updates = max(5, int(args.target_seconds * 0.92 / probe_per_update))
        print(
            f"Calibrated total_updates={target_updates} for ~{args.target_seconds}s",
            flush=True,
        )
    else:
        if args.dry_run:
            print("--dry-run ignored when --total-updates is set", flush=True)
            return
        target_updates = args.total_updates
        print(f"Using fixed total_updates={target_updates}", flush=True)

    run_cfg = copy.deepcopy(base)
    run_cfg.train.total_updates = target_updates
    run_cfg.checkpoint_path = str(Path(base.checkpoint_path))

    print("Full profile run...", flush=True)
    t1 = time.perf_counter()
    report = _run_train(run_cfg)
    run_wall = time.perf_counter() - t1

    report.metadata = {
        "gpu": _gpu_name(),
        "config": str(args.config),
        "target_seconds": args.target_seconds,
        "probe_updates": args.probe_updates,
        "probe_wall_seconds": round(probe_wall, 3),
        "probe_sec_per_update": round(probe_per_update, 3),
        "calibrated_updates": target_updates,
        "fixed_updates": args.total_updates is not None,
        "run_wall_seconds": round(run_wall, 3),
        "batch_hands": base.train.batch_hands,
        "teacher_samples": base.train.teacher_samples,
        "payoff_samples": base.train.payoff_samples,
        "torch_compile": base.train.torch_compile,
        "cprofile_update": args.cprofile_update,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"league_gpu_{stamp}.json"
    md_path = out_dir / f"league_gpu_{stamp}.md"
    report.write_json(json_path)
    report.write_markdown(md_path)

    if args.cprofile_update is not None:
        cprofile_path = out_dir / f"cprofile_update{args.cprofile_update}_{stamp}.txt"
        write_cprofile_text(cprofile_path)
        print(f"Wrote {cprofile_path}", flush=True)

    latest = Path("logs/aof_rl_profile_latest.json")
    latest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(json_path, latest)

    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)
    print(f"Copied to {latest}", flush=True)

    top = report.timings.sorted_report(wall_total=report.timings.total_wall_seconds())[:8]
    print("\nTop bottlenecks:", flush=True)
    for row in top:
        print(
            f"  {row['rank']}. {row['name']}: {row['seconds']}s ({row['pct']}%)",
            flush=True,
        )

    if report.cprofile_top:
        print("\nTop cProfile functions:", flush=True)
        for rank, row in enumerate(report.cprofile_top[:5], start=1):
            print(
                f"  {rank}. {row['function']}: {row['cumtime']}s cumtime",
                flush=True,
            )


if __name__ == "__main__":
    main()
