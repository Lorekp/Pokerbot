#!/usr/bin/env python3
"""Build HU bucket equity matrix from hu_showdown_equity.npz (NPZ path)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_cfr.hu_payoff_matrix import (
    build_hu_bucket_equity_matrix,
    save_equity_matrix,
)
from aof_equity import warmup


def main() -> None:
    p = argparse.ArgumentParser(description="Build NPZ-averaged HU eq matrix")
    p.add_argument("--out", default="data/aof/cfr_base/hu_eq_matrix.npz")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true", help="Rebuild even if out exists")
    args = p.parse_args()

    out = Path(args.out)
    if out.is_file() and not args.force:
        print(f"{out} exists (use --force to rebuild)")
        return

    warmup()
    t0 = time.time()
    eq = build_hu_bucket_equity_matrix(workers=args.workers)
    save_equity_matrix(out, eq)
    print(f"Wrote {out} in {time.time() - t0:.1f}s shape={eq.shape}")


if __name__ == "__main__":
    main()
