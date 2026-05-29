#!/usr/bin/env python3
"""Benchmark non-AoF postflop solver label generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_rl.profiling.timer import TimingAccumulator
from build_solver_labels import _config_hash, build_labels
from pokerbot.solver.label_diff import DiffMode, compare_label_dirs
from pokerbot.solver.postflop_solver_client import DEFAULT_CLI_PATH


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_ROOT,
        )
        return out.stdout.strip()[:12]
    except Exception:
        return "unknown"


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def _load_baseline(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark non-AoF solver label generation")
    p.add_argument("--config", type=Path, default=Path("config/solver_labels_smoke.yaml"))
    p.add_argument("--max-spots", type=int, default=None)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument(
        "--solver-backend",
        choices=["subprocess", "worker"],
        default="subprocess",
    )
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--max-in-flight", type=int, default=None)
    p.add_argument("--solver-threads-per-worker", type=int, default=None)
    p.add_argument("--disable-worker-cache", action="store_true")
    p.add_argument("--worker-cache-size", type=int, default=None)
    p.add_argument("--weva-index", type=Path, default=Path("data/weva"))
    p.add_argument("--cli-path", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("runs/bench_non_aof_baseline"))
    p.add_argument(
        "--compare-baseline",
        type=Path,
        default=None,
        help="Optional baseline JSON for throughput comparison",
    )
    p.add_argument(
        "--diff-reference",
        type=Path,
        default=None,
        help="Optional reference run directory for golden diff",
    )
    p.add_argument("--skip-build", action="store_true")
    args = p.parse_args()

    cli_path = args.cli_path or DEFAULT_CLI_PATH
    if not args.skip_build and not cli_path.is_file():
        crate = _ROOT / "external" / "postflop_solver_cli"
        print(f"Building postflop_solver_cli at {crate}...", flush=True)
        subprocess.run(["cargo", "build", "--release"], cwd=crate, check=True)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rep_wall_times: list[float] = []
    rep_spots_per_hour: list[float] = []
    all_spot_times: list[float] = []

    last_manifest: dict[str, Any] | None = None
    last_prof: TimingAccumulator | None = None
    last_parquet: Path | None = None

    for rep in range(args.repetitions):
        rep_dir = args.out / f"rep{rep}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        prof = TimingAccumulator()
        t0 = time.perf_counter()
        manifest = build_labels(
            args.config,
            rep_dir / "train.parquet",
            rep_dir / "manifest.json",
            rep_dir / "report.md",
            args.weva_index,
            cli_path,
            max_spots=args.max_spots,
            prof=prof,
            solver_backend=args.solver_backend,
            num_workers=args.num_workers,
            max_in_flight=args.max_in_flight,
            solver_threads_per_worker=args.solver_threads_per_worker,
            disable_worker_cache=args.disable_worker_cache,
            worker_cache_size=args.worker_cache_size,
        )
        wall = time.perf_counter() - t0
        rep_wall_times.append(wall)

        included = int(manifest["enumeration"]["included"])
        if wall > 0 and included > 0:
            rep_spots_per_hour.append(included / wall * 3600.0)

        for snap in prof.spot_snapshots:
            all_spot_times.append(float(snap.get("wall_seconds", 0.0)))

        last_manifest = manifest
        last_prof = prof
        last_parquet = rep_dir / "train.parquet"

    buckets = last_prof.sorted_report() if last_prof else []

    def _bucket_sec(name: str) -> float:
        for b in buckets:
            if b.get("name") == name:
                return float(b.get("seconds", 0.0))
        return 0.0

    pool_stats = None
    if last_manifest:
        pool_stats = last_manifest.get("solver_orchestration", {}).get("pool_stats")

    golden_diff_passed = None
    if args.diff_reference is not None and args.repetitions > 0:
        diff = compare_label_dirs(args.diff_reference, args.out / "rep0", DiffMode.SEMANTIC)
        golden_diff_passed = diff.passed

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "config": str(args.config),
        "config_hash": _config_hash(args.config),
        "max_spots": args.max_spots,
        "solver_backend": args.solver_backend,
        "num_workers": args.num_workers,
        "max_in_flight": args.max_in_flight,
        "solver_threads_per_worker": args.solver_threads_per_worker,
        "disable_worker_cache": args.disable_worker_cache,
        "repetitions": args.repetitions,
        "spots_attempted": last_manifest["enumeration"]["total_candidates"] if last_manifest else 0,
        "labels_written": last_manifest["enumeration"]["included"] if last_manifest else 0,
        "dataset_output_sha256": _sha256(last_parquet) if last_parquet else "",
        "golden_diff_passed": golden_diff_passed,
        "pool_stats": pool_stats,
        "wall_seconds": {
            "mean": statistics.mean(rep_wall_times) if rep_wall_times else None,
            "stdev": statistics.pstdev(rep_wall_times) if len(rep_wall_times) > 1 else 0.0,
            "per_repetition": rep_wall_times,
        },
        "spots_per_hour": {
            "mean": statistics.mean(rep_spots_per_hour) if rep_spots_per_hour else None,
            "per_repetition": rep_spots_per_hour,
        },
        "spot_latency_sec": {
            "p50": _percentile(all_spot_times, 50),
            "p95": _percentile(all_spot_times, 95),
            "p99": _percentile(all_spot_times, 99),
            "count": len(all_spot_times),
        },
        "timing_buckets": buckets,
        "solver_time_sec": _bucket_sec("spot.solve_subprocess"),
        "encode_time_sec": _bucket_sec("spot.encode_row"),
        "export_time_sec": _bucket_sec("export.parquet_write"),
        "failure_count": (
            last_manifest["enumeration"].get("flagged_failure_count", 0) if last_manifest else 0
        ),
    }

    baseline = _load_baseline(args.compare_baseline) if args.compare_baseline else None
    if baseline:
        base_sph = baseline.get("spots_per_hour", {}).get("mean")
        cur_sph = report["spots_per_hour"]["mean"]
        if base_sph and cur_sph:
            report["comparison"] = {
                "baseline_path": str(args.compare_baseline),
                "baseline_spots_per_hour": base_sph,
                "current_spots_per_hour": cur_sph,
                "speedup": cur_sph / base_sph,
            }

    json_path = args.out / f"benchmark_{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest = args.out / "benchmark_latest.json"
    latest.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Non-AoF label benchmark",
        "",
        f"- Timestamp: {report['timestamp']}",
        f"- Git: {report['git_commit']}",
        f"- Config: {report['config']} ({report['config_hash']})",
        f"- Backend: {report['solver_backend']} (workers={report['num_workers']})",
        f"- Spots: {report['spots_attempted']} attempted, {report['labels_written']} written",
        f"- Dataset SHA256: {report['dataset_output_sha256'][:16]}..." if report["dataset_output_sha256"] else "- Dataset: n/a",
        f"- Golden diff: {report['golden_diff_passed']}",
        f"- Wall (mean): {report['wall_seconds']['mean']:.2f}s over {args.repetitions} rep(s)",
        f"- Spots/hour (mean): {report['spots_per_hour']['mean']:.2f}"
        if report["spots_per_hour"]["mean"]
        else "- Spots/hour: n/a",
        f"- Spot latency p50/p95/p99: {report['spot_latency_sec']['p50']}/{report['spot_latency_sec']['p95']}/{report['spot_latency_sec']['p99']}s",
        f"- Solver / encode / export: {report['solver_time_sec']:.2f}s / {report['encode_time_sec']:.4f}s / {report['export_time_sec']:.4f}s",
        "",
        f"Full JSON: `{json_path}`",
    ]
    if "comparison" in report:
        md_lines.extend(
            [
                "",
                "## vs baseline",
                f"- Baseline spots/h: {report['comparison']['baseline_spots_per_hour']:.2f}",
                f"- Current spots/h: {report['comparison']['current_spots_per_hour']:.2f}",
                f"- Speedup: {report['comparison']['speedup']:.2f}x",
            ]
        )
    md_path = args.out / f"benchmark_{stamp}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
