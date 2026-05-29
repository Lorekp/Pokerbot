#!/usr/bin/env python3
"""Benchmark one AoF RL training update across optimization profiles.

CUDA training: use Python 3.13 with cu126 torch, e.g.
  py -3.13 scripts/benchmark_aof_rl.py --device cuda --instrument
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from aof_rl.policy import AoFPolicyNet
from aof_rl.self_play import AoFSelfPlayConfig, train_self_play

PROFILES = {
    "baseline": dict(
        teacher_samples=1500,
        payoff_samples=3000,
        use_fast_teacher=False,
        teacher_margin_bb=0.0,
    ),
    "fast_teacher": dict(
        teacher_samples=500,
        payoff_samples=2000,
        use_fast_teacher=True,
        teacher_margin_bb=0.4,
    ),
    "train": dict(
        teacher_samples=300,
        payoff_samples=1200,
        use_fast_teacher=True,
        teacher_margin_bb=0.4,
    ),
    "all": dict(
        teacher_samples=500,
        payoff_samples=2000,
        use_fast_teacher=True,
        teacher_margin_bb=0.4,
    ),
}


def bench_profile(name: str, hands: int, device: str, *, parallel_workers: int = 0) -> dict:
    kw = PROFILES[name]
    cfg = AoFSelfPlayConfig(
        hands_per_update=hands,
        total_updates=1,
        device=device,
        torch_compile=(device == "cuda"),
        retain_teacher_cache=True,
        rollout_workers=parallel_workers,
        **kw,
    )
    model = AoFPolicyNet(hidden=64).to(device)
    t0 = time.perf_counter()
    train_self_play(model, cfg)
    elapsed = time.perf_counter() - t0
    return {
        "profile": name,
        "hands": hands,
        "device": device,
        "parallel_workers": parallel_workers,
        "seconds": round(elapsed, 2),
        "sec_per_hand": round(elapsed / max(hands, 1), 3),
        **kw,
    }


def run_instrument(hands: int, device: str, profile: str) -> dict:
    from aof_rl.instrument import run_instrumentation

    kw = PROFILES.get(profile, PROFILES["train"])
    model = AoFPolicyNet(hidden=64).to(device)
    model.eval()
    summary = run_instrumentation(
        model,
        torch.device(device),
        hands=hands,
        teacher_samples=kw["teacher_samples"],
        payoff_samples=kw["payoff_samples"],
        seed=42,
        use_fast_teacher=kw["use_fast_teacher"],
    )
    return summary.to_dict()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--hands", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="logs/aof_rl_benchmark.json")
    p.add_argument("--instrument", action="store_true")
    p.add_argument("--profile", default="train", choices=tuple(PROFILES.keys()))
    p.add_argument("--parallel-workers", type=int, default=0)
    args = p.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )

    if args.instrument:
        row = run_instrument(args.hands, device, args.profile)
        row["mode"] = "instrument"
        row["profile"] = args.profile
        row["device"] = device
        rows = [row]
        print(json.dumps(row, indent=2), flush=True)
    else:
        rows = []
        for name in ("train", "fast_teacher", "baseline"):
            if name not in PROFILES:
                continue
            print(f"Running {name}...", flush=True)
            rows.append(
                bench_profile(
                    name, args.hands, device, parallel_workers=args.parallel_workers
                )
            )
            print(rows[-1], flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
