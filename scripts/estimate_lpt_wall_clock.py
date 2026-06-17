#!/usr/bin/env python3
"""Estimate fleet wall-clock: random shuffle vs slow-bucket LPT scheduling."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from pokerbot.solver.cloud_spot_solver import (
    board_timing_from_manifest,
    estimate_spot_seconds,
    order_pending_for_worker,
)


def _spot_times_from_manifest(manifest: dict, spot_keys: list[str]) -> dict[str, float]:
    per = {str(r["spot"]): float(r["elapsed_sec"]) for r in manifest.get("per_spot_results", [])}
    timing = board_timing_from_manifest(manifest)
    out: dict[str, float] = {}
    for k in spot_keys:
        if k in per:
            out[k] = per[k]
        else:
            out[k] = estimate_spot_seconds(k, timing) or 60.0
    return out


def _simulate_fleet(
    spot_times: dict[str, float],
    *,
    workers: int,
    order_fn,
) -> float:
    """Greedy multi-worker: each worker walks its list sequentially (no collisions)."""
    keys = list(spot_times.keys())
    loads = [0.0] * workers
    for wid in range(workers):
        ordered = order_fn(keys, f"worker-{wid}")
        for k in ordered:
            i = min(range(workers), key=lambda j: loads[j])
            loads[i] += spot_times[k]
    return max(loads)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("data/solver/manifest_prod600.json"))
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--scale-spots", type=int, default=3000)
    p.add_argument("--output", type=Path, default=Path("logs/lpt_scheduling_estimate.json"))
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    base_keys = [str(r["spot"]) for r in manifest.get("per_spot_results", [])]
    keys: list[str] = []
    while len(keys) < args.scale_spots:
        keys.extend(base_keys)
    keys = keys[: args.scale_spots]
    spot_times = _spot_times_from_manifest(manifest, keys)
    timing = board_timing_from_manifest(manifest)

    def random_order(klist: list[str], wid: str) -> list[str]:
        seed = int(hashlib.sha256(wid.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        out = list(klist)
        rng.shuffle(out)
        return out

    def lpt_order(klist: list[str], wid: str) -> list[str]:
        return order_pending_for_worker(klist, wid, board_timing=timing)

    random_wall = _simulate_fleet(spot_times, workers=args.workers, order_fn=random_order)
    lpt_wall = _simulate_fleet(spot_times, workers=args.workers, order_fn=lpt_order)
    report = {
        "workers": args.workers,
        "spots": args.scale_spots,
        "random_shuffle_wall_sec": round(random_wall, 1),
        "lpt_slow_bucket_wall_sec": round(lpt_wall, 1),
        "speedup_ratio": round(random_wall / lpt_wall, 3),
        "wall_hours_random": round(random_wall / 3600, 2),
        "wall_hours_lpt": round(lpt_wall / 3600, 2),
        "longest_single_spot_sec": max(spot_times.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
