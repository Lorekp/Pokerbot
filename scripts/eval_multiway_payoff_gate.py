#!/usr/bin/env python3
"""
Post-HU-table multiway payoff gate (two conditions).

1. Re-profile: standalone multiway wall % (from league_gpu profile JSON).
2. If >=10% wall: measure bucket-mean vs exact EV error vs reward_noise_floor_bb.

Does not ship E3 stratified MC — documents tabulate-vs-E3 fork only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aof_rl.reward_noise import reward_noise_floor_bb, reward_noise_floor_definition


def _wall_share(profile: dict, bucket: str, total: float) -> float:
    buckets = profile.get("buckets") or profile.get("timings") or {}
    sec = float(buckets.get(bucket, 0.0))
    return 100.0 * sec / total if total > 0 else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile",
        type=Path,
        default=Path("logs/aof_rl_profile_latest.json"),
    )
    p.add_argument(
        "--payoff-error",
        type=Path,
        default=Path("artifacts/audit/multiway_payoff_equity_error.json"),
    )
    p.add_argument("--wall-threshold-pct", type=float, default=10.0)
    p.add_argument("--batch-hands", type=int, default=128)
    p.add_argument("--noise-k", type=float, default=0.25)
    args = p.parse_args()

    if not args.profile.is_file():
        raise SystemExit(f"Profile not found: {args.profile}")

    prof = json.loads(args.profile.read_text(encoding="utf-8"))
    total = float(prof.get("total_wall_seconds") or prof.get("wall_seconds") or 1.0)
    mw_pct = _wall_share(prof, "rollout.step_payoff_multiway", total)
    hu_pct = _wall_share(prof, "rollout.step_payoff_hu", total)

    floor_bb = None
    bucket_below_floor = None
    if args.payoff_error.is_file():
        err = json.loads(args.payoff_error.read_text(encoding="utf-8"))
        floor_bb = err.get("reward_noise_floor_bb")
        bucket_below_floor = err.get("bucket_mean_below_floor")

    verdict = "close_no_e3"
    if mw_pct < args.wall_threshold_pct:
        verdict = "close_no_e3"
        reason = f"multiway payoff {mw_pct:.1f}% wall < {args.wall_threshold_pct}%"
    else:
        if bucket_below_floor is True:
            verdict = "tabulate_multiway"
            reason = "bucket error below reward noise floor"
        elif bucket_below_floor is False:
            verdict = "ship_e3_stratified_mc"
            reason = "bucket error above floor; prefer unbiased E3 MC"
        else:
            verdict = "measure_bucket_error_first"
            reason = (
                "multiway wall above threshold; run bucket-mean vs exact measurement"
            )

    out = {
        "multiway_wall_pct": mw_pct,
        "hu_payoff_wall_pct": hu_pct,
        "wall_threshold_pct": args.wall_threshold_pct,
        "reward_noise_floor": reward_noise_floor_definition(
            batch_hands=args.batch_hands, noise_k=args.noise_k
        ),
        "reward_noise_floor_bb": floor_bb,
        "verdict": verdict,
        "reason": reason,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
