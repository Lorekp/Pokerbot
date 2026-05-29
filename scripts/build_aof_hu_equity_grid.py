#!/usr/bin/env python3
"""Build exact HU 169×169 preflop equity grid."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from aof_equity import warmup
from aof_equity_grids import (
    aa_vs_ako_equity,
    build_hu_equity_grid_169,
    save_hu_equity_grid,
)
from tools.ompeval_cli.ompeval_backend import backend_name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("data/aof/hu_equity_169.npz"),
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--oracle-min",
        type=float,
        default=0.8190,
        help="Min AA vs AKo equity (exact enumeration ≈ 0.81946)",
    )
    args = p.parse_args()
    warmup()
    print(f"169-class grid backend (combo cells): numba exact; ompeval_cli={backend_name()}")
    t0 = time.perf_counter()
    grid, labels = build_hu_equity_grid_169(workers=args.workers)
    elapsed = time.perf_counter() - t0
    aa_ako = aa_vs_ako_equity(grid, labels)
    print(f"Built {grid.shape} in {elapsed:.1f}s; AA vs AKo = {aa_ako:.6f}")
    if aa_ako < args.oracle_min:
        raise SystemExit(f"AA vs AKo {aa_ako:.6f} below oracle floor {args.oracle_min}")
    save_hu_equity_grid(args.out, grid, labels)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
