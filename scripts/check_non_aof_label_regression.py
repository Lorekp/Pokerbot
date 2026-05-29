#!/usr/bin/env python3
"""Regression check for non-AoF solver label generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from build_solver_labels import build_labels, _config_hash
from pokerbot.solver.label_diff import DiffMode, compare_label_dirs
from pokerbot.solver.postflop_solver_client import DEFAULT_CLI_PATH


def _sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Non-AoF label regression check")
    p.add_argument("--config", type=Path, default=Path("config/solver_labels_smoke.yaml"))
    p.add_argument("--max-spots", type=int, default=1)
    p.add_argument("--solver-backend", choices=["subprocess", "worker"], default="worker")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--baseline", type=Path, default=Path("runs/baselines/non_aof_labels_baseline.json"))
    p.add_argument("--weva-index", type=Path, default=Path("data/weva"))
    p.add_argument("--reference-dir", type=Path, default=None, help="Optional golden dir to diff against")
    p.add_argument("--perf-tolerance", type=float, default=0.5, help="Min speedup vs baseline (informational)")
    args = p.parse_args()

    if not DEFAULT_CLI_PATH.is_file():
        print("ERROR: postflop_solver_cli not built", file=sys.stderr)
        sys.exit(2)

    with tempfile.TemporaryDirectory(prefix="non_aof_regression_") as tmp:
        out = Path(tmp) / "run"
        out.mkdir()
        manifest = build_labels(
            args.config,
            out / "train.parquet",
            out / "manifest.json",
            out / "report.md",
            args.weva_index,
            DEFAULT_CLI_PATH,
            max_spots=args.max_spots,
            solver_backend=args.solver_backend,
            num_workers=args.num_workers,
        )

        diff_passed = True
        if args.reference_dir is not None:
            result = compare_label_dirs(args.reference_dir, out, DiffMode.SEMANTIC)
            diff_passed = result.passed
            if not diff_passed:
                print("DIFF FAILED:", file=sys.stderr)
                for issue in result.issues[:20]:
                    print(f"  {issue.path}: {issue.message}", file=sys.stderr)

        report: dict[str, Any] = {
            "config": str(args.config),
            "config_hash": _config_hash(args.config),
            "solver_backend": args.solver_backend,
            "num_workers": args.num_workers,
            "labels_written": manifest["enumeration"]["included"],
            "dataset_output_sha256": _sha256(out / "train.parquet"),
            "golden_diff_passed": diff_passed if args.reference_dir else None,
        }

        baseline = None
        if args.baseline.is_file():
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
            report["baseline_path"] = str(args.baseline)

        print(json.dumps(report, indent=2))

        if args.reference_dir is not None and not diff_passed:
            sys.exit(1)


if __name__ == "__main__":
    main()
