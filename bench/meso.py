#!/usr/bin/env python3
"""Meso-tier HU showdown benchmarks (fixed key chunk -> s/key ETA)."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from aof_equity_grids import _equity_two_combos_exact
from bench._keys import load_keys, representative_from_key
from bench._timing import (
    EVAL7_PER_KEY,
    N_CANONICAL_KEYS,
    RUNOUTS_PER_KEY,
    projected_full_build_h,
    warmup_equity,
    write_json,
)


def _run_keys(keys, n_repeat: int) -> dict:
    reps = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        for k in keys:
            h0, h1, v0, v1 = representative_from_key(int(k))
            _equity_two_combos_exact(h0, h1, v0, v1)
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
    s_mean = statistics.mean(r["s_per_key"] for r in reps)
    s_std = statistics.stdev(r["s_per_key"] for r in reps) if len(reps) > 1 else 0.0
    rel_std_pct = 100.0 * s_std / s_mean if s_mean > 0 else 0.0
    return {
        "n_keys": len(keys),
        "n_repeat": n_repeat,
        "s_per_key": s_mean,
        "s_per_key_std": s_std,
        "s_per_key_rel_std_pct": rel_std_pct,
        "keys_per_s": statistics.mean(r["keys_per_s"] for r in reps),
        "boards_per_s": statistics.mean(r["boards_per_s"] for r in reps),
        "effective_eval7_per_s": statistics.mean(
            r["effective_eval7_per_s"] for r in reps
        ),
        "projected_full_build_h": projected_full_build_h(s_mean),
        "repeats": reps,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", choices=["smoke", "stratified"], default="stratified")
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    warmup_equity()
    keys = load_keys(args.fixture)
    result = _run_keys(keys, args.repeat)
    result["fixture"] = args.fixture
    result["n_canonical_keys"] = N_CANONICAL_KEYS

    print(f"fixture={args.fixture} n_keys={result['n_keys']} repeat={args.repeat}")
    print(
        f"s/key={result['s_per_key']:.4f} ± {result['s_per_key_std']:.4f} "
        f"({result['s_per_key_rel_std_pct']:.2f}% rel std)"
    )
    print(f"keys/s={result['keys_per_s']:.2f}")
    print(f"boards/s={result['boards_per_s']:,.0f}")
    print(f"effective eval7/s={result['effective_eval7_per_s']:,.0f}")
    print(f"projected full build={result['projected_full_build_h']:.2f} h")

    if args.json_out:
        write_json(args.json_out, result)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
