#!/usr/bin/env python3
"""Validate one bucket row: saved slice or live MC vs exact oracle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.cpu_pool import pin_compute_threads
from aof_cfr.trio_payoff_tensor import DEFAULT_MC_BOARDS
from aof_cfr.trio_row_validate import (
    MAX_WEIGHTED_EQUITY_ERROR,
    check_triple_live,
    row_slice_sanity,
    validate_row_samples,
)
from aof_equity import warmup

FIXED_SPOTS: list[tuple[int, int]] = [
    (0, 0),
    (0, 1),
    (12, 12),
    (168, 168),
    (0, 168),
]


def _load_row_slice(path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    row = int(data["bucket_row"])
    return (
        row,
        np.asarray(data["shares"], dtype=np.float64),
        np.asarray(data["pair_eq"], dtype=np.float64),
        np.asarray(data["combo_counts"], dtype=np.int32),
    )


def _print_checks(checks: list, *, title: str) -> None:
    print(title, flush=True)
    for c in sorted(checks, key=lambda x: x.max_error, reverse=True):
        print(
            f"  {c.label}: err={c.max_error:.5f} w={c.deal_weight:.2e} "
            f"combos={c.combo_count} "
            f"mc_sh=[{c.mc[0]:.3f},{c.mc[1]:.3f},{c.mc[2]:.3f}] "
            f"oracle=[{c.oracle[0]:.3f},{c.oracle[1]:.3f},{c.oracle[2]:.3f}]",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Validate one trio tensor bucket row")
    p.add_argument("--row", type=int, default=0, help="Bucket row index (BTN hand)")
    p.add_argument("--tensor", default="", help="Saved row slice .npz (optional)")
    p.add_argument("--samples", type=int, default=10, help="Random spot-check count")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mc-boards", type=int, default=DEFAULT_MC_BOARDS)
    p.add_argument(
        "--live",
        action="store_true",
        help="Compare live MC recipe to oracle (no saved slice required)",
    )
    args = p.parse_args()

    warmup()
    pin_compute_threads(1)

    row = args.row
    if args.tensor:
        path = Path(args.tensor)
        if not path.is_file():
            raise SystemExit(f"Missing {path}")
        row, shares, pair_eq, combo_counts = _load_row_slice(path)
        print(f"Loaded row slice row={row} btn={LABELS_169[row]} from {path}", flush=True)
        sanity = row_slice_sanity(shares, pair_eq, combo_counts)
        print(f"Slice sanity: {sanity}", flush=True)
        fixed = [(j, k) for j, k in FIXED_SPOTS if j < shares.shape[0] and k < shares.shape[1]]
        fixed_checks = [
            validate_row_samples(
                row,
                shares=shares,
                pair_eq=pair_eq,
                combo_counts=combo_counts,
                fixed_triples=[t],
            )[0][0]
            for t in fixed
        ]
        _print_checks(fixed_checks, title="Fixed spots vs oracle:")
        rand_checks, ok, werr, umax = validate_row_samples(
            row,
            samples=args.samples,
            seed=args.seed,
            shares=shares,
            pair_eq=pair_eq,
            combo_counts=combo_counts,
        )
        _print_checks(rand_checks, title=f"Random {args.samples} spots vs oracle:")
    elif args.live:
        print(
            f"Live MC vs oracle row={row} btn={LABELS_169[row]} "
            f"budget={args.mc_boards} (no saved slice)",
            flush=True,
        )
        fixed_checks = [check_triple_live(row, j, k, mc_boards=args.mc_boards) for j, k in FIXED_SPOTS]
        _print_checks(fixed_checks, title="Fixed spots:")
        rand_checks, ok, werr, umax = validate_row_samples(
            row, samples=args.samples, seed=args.seed, mc_boards=args.mc_boards
        )
        _print_checks(rand_checks, title=f"Random {args.samples} spots:")
    else:
        raise SystemExit("Provide --tensor PATH or --live")

    print(
        f"weighted_error={werr:.6f} unweighted_max={umax:.6f} "
        f"gate={MAX_WEIGHTED_EQUITY_ERROR} -> {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
