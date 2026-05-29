#!/usr/bin/env python3
"""Meso-tier postflop benchmark: single golden solve request."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from bench.postflop._timing import DEFAULT_RUST_CLI, GOLDEN_REQUEST, resolve_cli, run_cli_once, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", choices=["golden"], default="golden")
    p.add_argument("--cli-path", type=Path, default=None)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    cli = resolve_cli(args.cli_path)
    if not cli.is_file():
        raise SystemExit(f"CLI not found: {cli}")

    request = json.loads(GOLDEN_REQUEST.read_text(encoding="utf-8"))
    reps: list[dict] = []
    for i in range(args.repeat):
        wall, stdout, stderr, code = run_cli_once(cli, request)
        reps.append({"repeat": i, "wall_sec": wall, "exit_code": code, "stderr_tail": stderr[:200]})
        print(f"rep{i}: wall={wall:.2f}s exit={code}", flush=True)

    walls = [r["wall_sec"] for r in reps]
    result = {
        "fixture": args.fixture,
        "cli": str(cli),
        "n_repeat": args.repeat,
        "s_per_spot_mean": statistics.mean(walls),
        "s_per_spot_std": statistics.stdev(walls) if len(walls) > 1 else 0.0,
        "spots_per_hour": 3600.0 / statistics.mean(walls) if walls else 0.0,
        "repeats": reps,
    }
    print(f"s/spot={result['s_per_spot_mean']:.2f} spots/hr={result['spots_per_hour']:.1f}")

    if args.json_out:
        write_json(args.json_out, result)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
