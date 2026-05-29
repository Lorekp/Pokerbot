#!/usr/bin/env python3
"""Benchmark 7-card eval throughput: phevaluator numba vs OMPEval CLI (if built).

Deprecated for build ETA: shuffle-heavy loop overstates cost structure.
Use ``python -m bench.micro`` (isolated eval7) and ``python -m bench.meso`` (s/key ETA).
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from aof_equity import warmup
from aof_equity_grids import _eval7


def _bench_numba(n: int, seed: int) -> float:
    rng = random.Random(seed)
    deck = list(range(52))
    t0 = time.perf_counter()
    for _ in range(n):
        rng.shuffle(deck)
        c = deck[:7]
        _eval7(c[0], c[1], c[2], c[3], c[4], c[5], c[6])
    return n / (time.perf_counter() - t0)


def _bench_ompeval_cli(n: int, seed: int) -> float | None:
    from tools.ompeval_cli.ompeval_backend import eval7_ompeval, find_cli_binary

    if find_cli_binary() is None:
        return None
    rng = random.Random(seed)
    deck = list(range(52))
    t0 = time.perf_counter()
    for _ in range(n):
        rng.shuffle(deck)
        c = deck[:7]
        eval7_ompeval(c)
    return n / (time.perf_counter() - t0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    warmup()
    nb = _bench_numba(args.n, args.seed)
    ob = _bench_ompeval_cli(args.n, args.seed)
    print(f"numba _eval7: {nb:,.0f} evals/sec")
    if ob is not None:
        print(f"OMPEval CLI:  {ob:,.0f} evals/sec")
    else:
        print("OMPEval CLI:  not built (see tools/ompeval_cli/README.md)")


if __name__ == "__main__":
    main()
