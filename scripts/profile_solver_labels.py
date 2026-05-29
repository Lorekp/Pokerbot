#!/usr/bin/env python3
"""Profile postflop-solver label generation (CPU / Rust CLI)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_rl.profiling.league_profiler import (
    get_cprofile_stats,
    set_cprofile_spot,
    write_cprofile_text,
)
from aof_rl.profiling.timer import TimingAccumulator
from build_solver_labels import build_labels
from pokerbot.solver.postflop_solver_client import DEFAULT_CLI_PATH


def _ensure_cli(cli_path: Path) -> Path:
    if cli_path.is_file():
        return cli_path
    crate_dir = _ROOT / "external" / "postflop_solver_cli"
    print(f"Building postflop_solver_cli at {crate_dir}...", flush=True)
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=crate_dir,
        check=True,
    )
    if not cli_path.is_file():
        raise FileNotFoundError(f"CLI not found after build: {cli_path}")
    return cli_path


@dataclass
class SolverLabelProfilerReport:
    timings: TimingAccumulator = field(default_factory=TimingAccumulator)
    spots_attempted: int = 0
    labels_written: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    cprofile_top: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        wall = self.timings.total_wall_seconds()
        solved = int(self.timings.counters.get("spots_solved", 0))
        return {
            "total_wall_seconds": round(wall, 3),
            "spots_attempted": self.spots_attempted,
            "labels_written": self.labels_written,
            "spots_solved": solved,
            "sec_per_spot": round(wall / max(self.spots_attempted, 1), 3),
            "metadata": self.metadata,
            "counters": dict(self.timings.counters),
            "buckets": self.timings.sorted_report(
                wall_total=self.timings.total_wall_seconds()
            ),
            "per_spot": list(self.timings.spot_snapshots),
            "cprofile_top": list(self.cprofile_top),
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def write_markdown(self, path: Path) -> None:
        d = self.to_dict()
        lines = [
            "# Postflop-solver label profile",
            "",
            f"- **Wall time**: {d['total_wall_seconds']} s",
            f"- **Spots attempted**: {d['spots_attempted']}",
            f"- **Labels written**: {d['labels_written']}",
            f"- **Sec/spot**: {d['sec_per_spot']}",
            "",
        ]
        meta = d.get("metadata") or {}
        if meta:
            lines.extend(["## Run metadata", ""])
            for k, v in meta.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        per_spot = d.get("per_spot") or []
        if per_spot:
            lines.extend(
                [
                    "## Per-spot breakdown",
                    "",
                    "| Spot | Stage | Wall (s) | Top buckets |",
                    "|------|-------|----------|-------------|",
                ]
            )
            for row in per_spot:
                top = ", ".join(
                    f"`{b['name']}` {b['seconds']}s"
                    for b in row.get("top_buckets", [])[:3]
                )
                key = row.get("spot_key") or row.get("spot")
                lines.append(
                    f"| {key} | {row.get('stage')} | "
                    f"{row.get('wall_seconds')} | {top or '-'} |"
                )
            lines.append("")

        cprofile_top = d.get("cprofile_top") or []
        if cprofile_top:
            lines.extend(
                [
                    "## cProfile top functions (by cumtime)",
                    "",
                    "| Rank | Function | Calls | Cumtime (s) | Tottime (s) |",
                    "|------|----------|-------|-------------|-------------|",
                ]
            )
            for rank, row in enumerate(cprofile_top, start=1):
                lines.append(
                    f"| {rank} | `{row['function']}` | {row['calls']} | "
                    f"{row['cumtime']} | {row['tottime']} |"
                )
            lines.append("")

        lines.extend(
            [
                "## Bottlenecks (sorted by wall time)",
                "",
                "| Rank | Bucket | Seconds | % | Calls | Sec/call |",
                "|------|--------|---------|---|-------|----------|",
            ]
        )
        for row in d["buckets"]:
            lines.append(
                f"| {row['rank']} | `{row['name']}` | {row['seconds']} | "
                f"{row['pct']} | {row['calls']} | {row['sec_per_call']} |"
            )
        lines.append("")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Profile postflop-solver label generation")
    p.add_argument("--config", type=Path, default=Path("config/solver_labels_smoke.yaml"))
    p.add_argument("--max-spots", type=int, default=10)
    p.add_argument("--cprofile-spot", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("data/solver/train_profile10.parquet"))
    p.add_argument("--manifest", type=Path, default=Path("data/solver/manifest_profile10.json"))
    p.add_argument("--report", type=Path, default=Path("logs/solver_label_run_profile10.md"))
    p.add_argument("--weva-index", type=Path, default=Path("data/weva"))
    p.add_argument("--cli-path", type=Path, default=None)
    p.add_argument("--out-dir", default="artifacts/profile")
    p.add_argument("--skip-build", action="store_true", help="Do not run cargo build")
    args = p.parse_args()

    cli_path = args.cli_path or DEFAULT_CLI_PATH
    if not args.skip_build:
        cli_path = _ensure_cli(cli_path)

    set_cprofile_spot(args.cprofile_spot)
    print(f"cProfile enabled for spot {args.cprofile_spot}", flush=True)

    prof = TimingAccumulator()
    t0 = time.perf_counter()
    manifest = build_labels(
        args.config,
        args.output,
        args.manifest,
        args.report,
        args.weva_index,
        cli_path,
        max_spots=args.max_spots,
        prof=prof,
        cprofile_spot=args.cprofile_spot,
    )
    run_wall = time.perf_counter() - t0

    report = SolverLabelProfilerReport(
        timings=prof,
        spots_attempted=int(manifest["enumeration"]["total_candidates"]),
        labels_written=int(manifest["enumeration"]["included"]),
        cprofile_top=get_cprofile_stats(),
        metadata={
            "config": str(args.config),
            "max_spots": args.max_spots,
            "cprofile_spot": args.cprofile_spot,
            "cli_path": str(cli_path),
            "run_wall_seconds": round(run_wall, 3),
            "output": str(args.output),
        },
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"solver_labels_{stamp}.json"
    md_path = out_dir / f"solver_labels_{stamp}.md"
    report.write_json(json_path)
    report.write_markdown(md_path)

    cprofile_path = out_dir / f"cprofile_spot{args.cprofile_spot}_{stamp}.txt"
    write_cprofile_text(cprofile_path)

    latest = Path("logs/solver_label_profile_latest.json")
    latest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(json_path, latest)

    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)
    print(f"Wrote {cprofile_path}", flush=True)
    print(f"Copied to {latest}", flush=True)

    top = prof.sorted_report(wall_total=prof.total_wall_seconds())[:5]
    print("\nTop bottlenecks:", flush=True)
    for row in top:
        print(
            f"  {row['rank']}. {row['name']}: {row['seconds']}s ({row['pct']}%)",
            flush=True,
        )
    if report.cprofile_top:
        print("\nTop cProfile functions:", flush=True)
        for rank, row in enumerate(report.cprofile_top[:5], start=1):
            print(
                f"  {rank}. {row['function']}: {row['cumtime']}s cumtime",
                flush=True,
            )

    if manifest["gto_acceptable"] and manifest["enumeration"]["included"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
