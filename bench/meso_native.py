#!/usr/bin/env python3
"""Meso-tier native HU showdown benchmarks (build_scores -> s/key ETA)."""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import numpy as np

from bench._keys import load_keys
from bench._timing import (
    EVAL7_PER_KEY,
    N_CANONICAL_KEYS,
    RUNOUTS_PER_KEY,
    projected_full_build_h,
    summarize_repeats_discard_first,
    write_json,
)
from tools.ompeval_cli.native import build_scores, native_backend_available

# Standard scaling points shared with bench_native wrapper (threads=1,2,4).
STANDARD_THREAD_COUNTS = (1, 2, 4)
DISCARD_FIRST_REPEAT = 1


def _run_keys(keys: np.ndarray, n_repeat: int, threads: int) -> dict:
    reps: list[dict] = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        build_scores(keys, threads=threads)
        wall = time.perf_counter() - t0
        n = len(keys)
        reps.append(
            {
                "wall_s": wall,
                "s_per_key": wall / n,
                "keys_per_s": n / wall,
                "boards_per_s": (n / wall) * RUNOUTS_PER_KEY,
                "effective_eval7_per_s": (n / wall) * EVAL7_PER_KEY,
            }
        )

    s_values = [r["s_per_key"] for r in reps]
    summary = summarize_repeats_discard_first(s_values, discard_first=DISCARD_FIRST_REPEAT)
    s_mean = summary["s_per_key"]
    return {
        "n_keys": len(keys),
        "n_repeat": n_repeat,
        "threads": threads,
        "discard_first_repeat": DISCARD_FIRST_REPEAT,
        **summary,
        "keys_per_s": statistics.mean(r["keys_per_s"] for r in reps[DISCARD_FIRST_REPEAT:]),
        "boards_per_s": statistics.mean(r["boards_per_s"] for r in reps[DISCARD_FIRST_REPEAT:]),
        "effective_eval7_per_s": statistics.mean(
            r["effective_eval7_per_s"] for r in reps[DISCARD_FIRST_REPEAT:]
        ),
        "projected_full_build_h": projected_full_build_h(s_mean),
        "projected_full_build_min": projected_full_build_h(s_mean) * 60.0,
        "repeats": reps,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", choices=["smoke", "stratified"], default="stratified")
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--scaling-sweep",
        action="store_true",
        help="Run STANDARD_THREAD_COUNTS and emit one JSON row per thread count",
    )
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    if not native_backend_available():
        raise SystemExit("native backend unavailable (build hu_table_native + sentinel gate)")

    keys = load_keys(args.fixture)
    if args.scaling_sweep:
        sweep = []
        for threads in STANDARD_THREAD_COUNTS:
            sweep.append(_run_keys(keys, args.repeat, threads=threads))
        result = {
            "fixture": args.fixture,
            "n_canonical_keys": N_CANONICAL_KEYS,
            "standard_thread_counts": list(STANDARD_THREAD_COUNTS),
            "scaling": sweep,
        }
    else:
        result = _run_keys(keys, args.repeat, threads=args.threads)
        result["fixture"] = args.fixture
        result["n_canonical_keys"] = N_CANONICAL_KEYS

    print(f"fixture={args.fixture} n_keys={len(keys)} repeat={args.repeat}")
    if args.scaling_sweep:
        for row in result["scaling"]:
            print(
                f"threads={row['threads']} s/key={row['s_per_key']:.4f} "
                f"± {row['s_per_key_std']:.4f} ({row['s_per_key_rel_std_pct']:.2f}% rel std)"
            )
            print(f"  projected_full_build_min={row['projected_full_build_min']:.1f}")
    else:
        print(
            f"threads={args.threads} s/key={result['s_per_key']:.4f} "
            f"± {result['s_per_key_std']:.4f} ({result['s_per_key_rel_std_pct']:.2f}% rel std)"
        )
        print(f"keys/s={result['keys_per_s']:.2f}")
        print(f"boards/s={result['boards_per_s']:,.0f}")
        print(f"effective eval7/s={result['effective_eval7_per_s']:,.0f}")
        print(f"projected_full_build_h={result['projected_full_build_h']:.2f}")
        print(f"projected_full_build_min={result['projected_full_build_min']:.1f}")

    if args.json_out is not None:
        write_json(args.json_out, result)


if __name__ == "__main__":
    main()
