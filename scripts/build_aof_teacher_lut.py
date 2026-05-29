#!/usr/bin/env python3
# Offline equity: 169 HU grid via scripts/build_aof_hu_equity_grid.py (numba exact; optional OMPEval CLI).
"""Build teacher LUT from census ctx rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from aof_equity import warmup
from aof_rl.teacher_lut.build import build_from_ctx_rows_file, build_tiny_fixture_lut
from aof_rl.teacher_lut.context import enumerate_reachable_ctx_rows, save_ctx_rows_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ctx-rows",
        type=Path,
        default=Path("data/aof/teacher_lut_ctx_rows_8bb.jsonl"),
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("data/aof/teacher_lut_v2_8bb.npz"),
    )
    p.add_argument("--stack-bb", type=float, default=8.0)
    p.add_argument("--teacher-version", type=int, default=2)
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-fast", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--emit-census", action="store_true")
    p.add_argument("--tiny-fixture", type=Path, default=None)
    args = p.parse_args()
    warmup()

    if args.emit_census or not args.ctx_rows.is_file():
        rows = enumerate_reachable_ctx_rows(
            stack_bb=args.stack_bb, teacher_version=args.teacher_version
        )
        save_ctx_rows_jsonl(rows, args.ctx_rows)
        print(f"Census: {len(rows)} rows -> {args.ctx_rows}")

    if args.tiny_fixture is not None:
        build_tiny_fixture_lut(
            args.tiny_fixture,
            teacher_version=args.teacher_version,
            stack_bb=args.stack_bb,
            samples=min(args.samples, 100),
            seed=args.seed,
        )
        print(f"Tiny fixture -> {args.tiny_fixture}")
        return

    schema = build_from_ctx_rows_file(
        args.ctx_rows,
        args.out,
        teacher_version=args.teacher_version,
        stack_bb=args.stack_bb,
        samples=args.samples,
        seed=args.seed,
        use_fast=args.use_fast,
        max_workers=args.workers,
    )
    print(f"LUT {schema.n_ctx} ctx × 169 -> {args.out}")


if __name__ == "__main__":
    main()
