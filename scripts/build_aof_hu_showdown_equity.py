#!/usr/bin/env python3
"""Build combo-exact HU showdown equity NPZ (artifact v2 via native batch builder)."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from aof_equity import warmup
from aof_equity_grids import aa_vs_ako_equity_exact
from aof_equity_score2 import GATE_B_FLOAT32_ATOL, equity_float32_from_score2, score2_from_key
from aof_equity_showdown import (
    EVALUATOR_BACKEND,
    _git_commit,
    _key_search_index,
    build_showdown_table_v2,
    canonical_showdown_key,
    enumerate_all_disjoint_pairs,
    key_spec_hash,
    save_showdown_table,
)


def _gate_a() -> None:
    ref = aa_vs_ako_equity_exact()
    if not (0.8190 <= ref <= 0.8200):
        raise SystemExit(f"Gate A failed: AA vs AKo exact = {ref:.6f}")
    print(f"Gate A OK: AA vs AKo exact = {ref:.6f}")


def _gate_b_score2(n: int, seed: int, keys: np.ndarray, score2: np.ndarray) -> None:
    rng = random.Random(seed)
    pairs = enumerate_all_disjoint_pairs()
    mismatches = 0
    for _ in range(n):
        h0, h1, v0, v1 = rng.choice(pairs)
        k = canonical_showdown_key(h0, h1, v0, v1)
        idx = _key_search_index(keys, k)
        if idx >= len(keys) or int(keys[idx]) != k:
            raise SystemExit(f"Gate B score2: key {k} missing from table")
        expected = score2_from_key(k)
        if int(score2[idx]) != expected:
            mismatches += 1
    if mismatches:
        raise SystemExit(f"Gate B score2 failed: {mismatches}/{n} mismatches")
    print(f"Gate B score2 OK: {n} samples, exact integer parity")


def _gate_b_float32(
    n: int, seed: int, keys: np.ndarray, score2: np.ndarray, eq: np.ndarray
) -> None:
    rng = random.Random(seed)
    pairs = enumerate_all_disjoint_pairs()
    max_err = 0.0
    for _ in range(n):
        h0, h1, v0, v1 = rng.choice(pairs)
        k = canonical_showdown_key(h0, h1, v0, v1)
        idx = _key_search_index(keys, k)
        if idx >= len(keys) or int(keys[idx]) != k:
            raise SystemExit(f"Gate B float32: key {k} missing from table")
        lut = float(eq[idx])
        direct = float(equity_float32_from_score2(int(score2[idx])))
        err = abs(lut - direct)
        max_err = max(max_err, err)
    if max_err > GATE_B_FLOAT32_ATOL:
        raise SystemExit(f"Gate B float32 failed: max |lut - derived| = {max_err}")
    print(f"Gate B float32 OK: {n} samples, max err {max_err:.2e}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        type=Path,
        default=Path("data/aof/hu_showdown_equity.npz"),
    )
    p.add_argument(
        "--census",
        type=Path,
        default=Path("artifacts/audit/hu_showdown_key_census.json"),
    )
    p.add_argument(
        "--keys",
        type=Path,
        default=Path("bench/datasets/fixtures/full_keys.uint64.npy"),
        help="Frozen canonical keys (sorted uint64 npy)",
    )
    p.add_argument("--threads", type=int, default=0, help="OpenMP threads (0=default)")
    p.add_argument("--gate-b-samples", type=int, default=10_000)
    p.add_argument(
        "--gate-b-score2-samples",
        type=int,
        default=512,
        help="Python score2 oracle samples (each is a full enum; keep modest)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.census.is_file():
        raise SystemExit(f"Run census first: {args.census} missing")
    if not args.keys.is_file():
        raise SystemExit(f"Keys fixture missing: {args.keys}")

    warmup()
    _gate_a()

    census = json.loads(args.census.read_text(encoding="utf-8"))
    print(f"Census: {census['n_canonical_keys']} canonical keys", flush=True)

    t0 = time.perf_counter()
    print(f"Building with backend={EVALUATOR_BACKEND}, threads={args.threads}", flush=True)
    keys, score2, eq = build_showdown_table_v2(keys_path=args.keys, threads=args.threads)
    elapsed = time.perf_counter() - t0
    print(f"Built {len(keys)} keys in {elapsed:.1f}s", flush=True)

    if len(keys) != census["n_canonical_keys"]:
        print(
            f"Warning: built {len(keys)} keys != census {census['n_canonical_keys']}"
        )

    _gate_b_score2(args.gate_b_score2_samples, args.seed, keys, score2)
    _gate_b_float32(args.gate_b_samples, args.seed, keys, score2, eq)

    meta = {
        "build_seconds": elapsed,
        "build_git_commit": _git_commit(),
        "backend": EVALUATOR_BACKEND,
        "census_path": str(args.census),
        "keys_path": str(args.keys),
        "n_canonical_keys": int(len(keys)),
    }
    from scripts.hu_showdown_toolchain import collect_toolchain_meta

    meta.update(collect_toolchain_meta(portable_build=True))
    save_showdown_table(
        args.o,
        keys,
        score2=score2,
        schema_path=args.o.with_suffix(".schema.json"),
        meta=meta,
    )
    print(f"Saved -> {args.o} (schema {args.o.with_suffix('.schema.json')})")
    print(f"key_spec_hash={key_spec_hash()}")


if __name__ == "__main__":
    main()
