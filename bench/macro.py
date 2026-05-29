#!/usr/bin/env python3
"""Macro-tier HU showdown benchmarks (build-driver orchestration)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from aof_equity_grids import _equity_two_combos_exact
from aof_equity_showdown import canonical_showdown_key, enumerate_all_disjoint_pairs
from bench._keys import load_keys, representative_from_key
from bench._timing import N_CANONICAL_KEYS, projected_full_build_h, warmup_equity, write_json


def _bench_fixture(name: str) -> dict:
    keys = load_keys(name)
    t0 = time.perf_counter()
    for k in keys:
        h0, h1, v0, v1 = representative_from_key(int(k))
        _equity_two_combos_exact(h0, h1, v0, v1)
    wall = time.perf_counter() - t0
    n = len(keys)
    s_per_key = wall / n
    return {
        "mode": "fixture",
        "fixture": name,
        "n_keys": n,
        "wall_s": wall,
        "s_per_key": s_per_key,
        "projected_full_build_h": projected_full_build_h(s_per_key),
        "artifact_bytes_estimate": n * 12,
    }


def _bench_dedupe_loop(limit: int) -> dict:
    pairs = enumerate_all_disjoint_pairs()
    seen: set[int] = set()
    t0 = time.perf_counter()
    pairs_scanned = 0
    for h0, h1, v0, v1 in pairs:
        pairs_scanned += 1
        k = canonical_showdown_key(h0, h1, v0, v1)
        if k in seen:
            continue
        _equity_two_combos_exact(h0, h1, v0, v1)
        seen.add(k)
        if len(seen) >= limit:
            break
    wall = time.perf_counter() - t0
    n = len(seen)
    s_per_key = wall / n if n else 0.0
    return {
        "mode": "dedupe_loop",
        "n_keys": n,
        "pairs_scanned": pairs_scanned,
        "wall_s": wall,
        "s_per_key": s_per_key,
        "orchestration_overhead_pairs_per_key": pairs_scanned / n if n else 0,
        "projected_full_build_h": projected_full_build_h(s_per_key),
        "artifact_bytes_estimate": n * 12,
    }


def _dry_run_full() -> dict:
    t0 = time.perf_counter()
    keys: set[int] = set()
    for h0, h1, v0, v1 in enumerate_all_disjoint_pairs():
        keys.add(canonical_showdown_key(h0, h1, v0, v1))
    wall = time.perf_counter() - t0
    return {
        "mode": "dry_run",
        "n_keys": len(keys),
        "expected": N_CANONICAL_KEYS,
        "wall_s": wall,
        "match": len(keys) == N_CANONICAL_KEYS,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", choices=["smoke", "stratified", "full"], default=None)
    p.add_argument("--dedupe-loop", action="store_true")
    p.add_argument("--keys", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    warmup_equity()

    if args.dry_run:
        result = _dry_run_full()
    elif args.dedupe_loop:
        result = _bench_dedupe_loop(args.keys)
    elif args.fixture == "full" and args.dry_run:
        result = _dry_run_full()
    elif args.fixture:
        result = _bench_fixture(args.fixture)
    else:
        result = _bench_dedupe_loop(args.keys)

    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k}={v:.4f}" if abs(v) < 1e6 else f"{k}={v:,.0f}")
        else:
            print(f"{k}={v}")

    if args.json_out:
        write_json(args.json_out, result)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
