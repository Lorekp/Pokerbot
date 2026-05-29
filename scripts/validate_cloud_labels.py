#!/usr/bin/env python3
"""Validate cloud vs local label runs (semantic diff + iterations_used distribution)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pokerbot.solver.label_diff import DiffMode, compare_label_dirs


def _iterations_from_manifest(manifest_path: Path) -> dict[str, int]:
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: dict[str, int] = {}
    for row in manifest.get("per_spot_results", []):
        spot = str(row.get("spot", ""))
        iters = row.get("iterations_used")
        if spot and iters is not None:
            out[spot] = int(iters)
    return out


def _iterations_distribution(iters: dict[str, int]) -> dict[str, Any]:
    if not iters:
        return {"count": 0}
    values = list(iters.values())
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 2),
    }


def _compare_iterations(
    local: dict[str, int],
    cloud: dict[str, int],
    *,
    relative_tol: float = 0.25,
    absolute_tol: int = 500,
) -> list[dict[str, Any]]:
    divergences: list[dict[str, Any]] = []
    for spot in sorted(set(local) & set(cloud)):
        li, ci = local[spot], cloud[spot]
        delta = abs(li - ci)
        rel = delta / max(li, ci, 1)
        if delta > absolute_tol and rel > relative_tol:
            divergences.append(
                {
                    "spot": spot,
                    "local_iterations": li,
                    "cloud_iterations": ci,
                    "delta": delta,
                    "relative_delta": round(rel, 4),
                }
            )
    return divergences


def main() -> None:
    p = argparse.ArgumentParser(description="Validate cloud vs local label generation")
    p.add_argument("--local-dir", type=Path, required=True)
    p.add_argument("--cloud-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("logs/cloud_label_validation_50.json"))
    p.add_argument("--max-exploitability-pct", type=float, default=0.5)
    p.add_argument("--iter-relative-tol", type=float, default=0.25)
    p.add_argument("--iter-absolute-tol", type=int, default=500)
    args = p.parse_args()

    diff = compare_label_dirs(args.local_dir, args.cloud_dir, mode=DiffMode.SEMANTIC)
    local_iters = _iterations_from_manifest(args.local_dir / "manifest.json")
    cloud_iters = _iterations_from_manifest(args.cloud_dir / "manifest.json")
    iter_divergences = _compare_iterations(
        local_iters,
        cloud_iters,
        relative_tol=args.iter_relative_tol,
        absolute_tol=args.iter_absolute_tol,
    )

    local_manifest = json.loads((args.local_dir / "manifest.json").read_text(encoding="utf-8"))
    max_exploit = local_manifest.get("convergence", {}).get("max_exploitability_pct")

    report = {
        "semantic_diff_passed": diff.passed,
        "semantic_diff_issues": len(diff.issues),
        "max_exploitability_pct_local": max_exploit,
        "iterations_used": {
            "local": _iterations_distribution(local_iters),
            "cloud": _iterations_distribution(cloud_iters),
            "divergences": iter_divergences,
            "divergence_count": len(iter_divergences),
        },
        "ok": diff.passed and len(iter_divergences) == 0,
        "local_dir": str(args.local_dir),
        "cloud_dir": str(args.cloud_dir),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
