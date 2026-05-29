#!/usr/bin/env python3
"""Count canonical HU showdown keys (no equity). Gate 0b before full table build."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aof_equity_showdown import (
    canonical_showdown_key,
    enumerate_all_disjoint_pairs,
    key_spec_hash,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("artifacts/audit/hu_showdown_key_census.json"),
    )
    p.add_argument(
        "--benchmark-evals-per-sec",
        type=float,
        default=562_000.0,
        help="eval7/s for build-time projection (from benchmark_eval_backend)",
    )
    args = p.parse_args()

    t0 = time.perf_counter()
    pairs = enumerate_all_disjoint_pairs()
    n_ordered = len(pairs)
    keys: set[int] = set()
    for h0, h1, v0, v1 in pairs:
        keys.add(canonical_showdown_key(h0, h1, v0, v1))
    n_canon = len(keys)
    elapsed = time.perf_counter() - t0

    # Unordered matchup count (hero/villain swap): half of ordered when distinct hands
    n_unordered_approx = n_ordered // 2

    key_bytes = 8
    artifact_bytes = n_canon * (key_bytes + 4)
    runouts = 1_712_304
    eval7_per_runout = 2
    eval7_calls = n_canon * runouts * eval7_per_runout
    build_sec = eval7_calls / max(args.benchmark_evals_per_sec, 1.0)

    payload = {
        "key_spec_hash": key_spec_hash(),
        "n_raw_ordered": n_ordered,
        "n_raw_unordered_approx": n_unordered_approx,
        "n_canonical_keys": n_canon,
        "artifact_bytes_estimate": artifact_bytes,
        "eval7_calls_estimate": eval7_calls,
        "build_seconds_estimate": build_sec,
        "census_wall_seconds": elapsed,
        "benchmark_evals_per_sec": args.benchmark_evals_per_sec,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
