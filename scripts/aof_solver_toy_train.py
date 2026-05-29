#!/usr/bin/env python3
"""Train experimental HU toy CFR+ skeleton and export JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_solver.cfr import RegretTable, run_one_cfr_iteration
from aof_solver.export import write_export, export_strategy_json
from aof_solver.game import HuGameConfig


def main() -> int:
    p = argparse.ArgumentParser(description="Experimental HU toy CFR+ train")
    p.add_argument("--players", type=int, default=2, choices=(2,))
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("artifacts/aof_solver_toy_hu.json"),
    )
    args = p.parse_args()
    if args.players != 2:
        raise SystemExit("Only HU toy (2 players) supported")

    cfg = HuGameConfig(stack_bb=8.0)
    table = RegretTable()
    buckets = ["strong", "weak"]
    bucket_eq = {"strong": 0.65, "weak": 0.35}
    for _ in range(args.iterations):
        run_one_cfr_iteration(table, buckets, bucket_eq, cfg)

    payload = export_strategy_json(
        table,
        stack_bb=cfg.stack_bb,
        iterations=args.iterations,
        seed=args.seed,
        exploitability_proxy=None,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_export(str(args.out), payload)
    print(f"Wrote {args.out} ({args.iterations} iterations, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
