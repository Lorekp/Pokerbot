#!/usr/bin/env python3
"""Validate experimental solver JSON export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_solver.import_experimental import (
    experimental_strategy_namespace,
    load_experimental_solver_json,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    args = p.parse_args()
    data = load_experimental_solver_json(args.input)
    ns = experimental_strategy_namespace(data)
    n_infosets = len(ns["strategy"] or {})
    print(
        f"OK provenance={ns['provenance']} game={ns['game']} "
        f"iterations={data.get('iterations')} infosets={n_infosets} "
        f"not_for_production={data.get('not_for_production')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
