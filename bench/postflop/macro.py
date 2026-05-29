#!/usr/bin/env python3
"""Macro-tier postflop benchmark: full label build via build_solver_labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.postflop._timing import write_json
from build_solver_labels import build_labels


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config/solver_labels_smoke.yaml"))
    p.add_argument("--max-spots", type=int, default=9)
    p.add_argument("--cli-path", type=Path, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    out_dir = Path("runs/bench_postflop_macro")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    manifest = build_labels(
        args.config,
        out_dir / "train.parquet",
        out_dir / "manifest.json",
        out_dir / "report.md",
        Path("data/weva"),
        args.cli_path,
        max_spots=args.max_spots,
        solver_backend="subprocess",
        num_workers=1,
    )
    wall = time.perf_counter() - t0
    included = int(manifest["enumeration"]["included"])
    attempted = int(manifest["enumeration"]["total_candidates"])
    result = {
        "config": str(args.config),
        "max_spots": args.max_spots,
        "cli": str(args.cli_path) if args.cli_path else "default",
        "wall_sec": wall,
        "spots_attempted": attempted,
        "labels_written": included,
        "spots_per_hour": included / wall * 3600.0 if wall > 0 else 0.0,
        "s_per_spot": wall / attempted if attempted else 0.0,
    }
    print(f"wall={wall:.1f}s spots={attempted} written={included} spots/hr={result['spots_per_hour']:.1f}")

    if args.json_out:
        write_json(args.json_out, result)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
