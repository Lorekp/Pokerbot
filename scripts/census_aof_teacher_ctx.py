#!/usr/bin/env python3
"""Emit census-validated teacher LUT context rows (JSONL)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from aof_rl.teacher_lut.context import enumerate_reachable_ctx_rows, save_ctx_rows_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stack-bb", type=float, default=8.0)
    p.add_argument("--teacher-version", type=int, default=2)
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("data/aof/teacher_lut_ctx_rows_8bb.jsonl"),
    )
    args = p.parse_args()
    rows = enumerate_reachable_ctx_rows(
        stack_bb=args.stack_bb, teacher_version=args.teacher_version
    )
    save_ctx_rows_jsonl(rows, args.out)
    hist = Counter(r.spot_kind for r in rows)
    print(f"Wrote {len(rows)} ctx rows -> {args.out}")
    print("spot_kind histogram:", dict(hist))


if __name__ == "__main__":
    main()
