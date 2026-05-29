#!/usr/bin/env python3
"""Canary: compare solver backends for non-AoF labels."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.postflop._timing import DEFAULT_CPP_CLI, DEFAULT_RUST_CLI
from pokerbot.solver.label_diff import DiffMode, compare_label_dirs


def _run_build(
    *,
    config: Path,
    out_dir: Path,
    backend: str,
    num_workers: int,
    max_spots: int,
    weva_index: Path,
    cli_path: Path | None = None,
) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "build_solver_labels.py"),
        "--config",
        str(config),
        "--max-spots",
        str(max_spots),
        "--output",
        str(out_dir / "train.parquet"),
        "--manifest",
        str(out_dir / "manifest.json"),
        "--report",
        str(out_dir / "report.md"),
        "--weva-index",
        str(weva_index),
        "--solver-backend",
        backend,
        "--num-workers",
        str(num_workers),
    ]
    if cli_path is not None:
        cmd.extend(["--cli-path", str(cli_path)])
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=_ROOT, check=True)
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description="Canary non-AoF label backends")
    p.add_argument("--config", type=Path, default=Path("tests/fixtures/non_aof_labels/golden_config.yaml"))
    p.add_argument("--max-spots", type=int, default=9)
    p.add_argument("--weva-index", type=Path, default=Path("data/weva"))
    p.add_argument("--out", type=Path, default=Path("runs/canary_non_aof"))
    p.add_argument("--mode", choices=["semantic", "strict", "float_contract"], default="semantic")
    p.add_argument(
        "--compare",
        choices=["orchestration", "solver_clis"],
        default="orchestration",
        help="orchestration: subprocess vs worker; solver_clis: Rust vs C++ CLI",
    )
    args = p.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    if args.compare == "solver_clis":
        if not DEFAULT_CPP_CLI.is_file():
            print(f"C++ CLI not built at {DEFAULT_CPP_CLI}; run scripts/build_texas_solver_cli.sh")
            sys.exit(0)
        if not (os.environ.get("TEXAS_SOLVER_CONSOLE") or os.environ.get("TEXAS_SOLVER_ROOT")):
            print("Set TEXAS_SOLVER_CONSOLE or TEXAS_SOLVER_ROOT for C++ engine")
            sys.exit(0)
        runs = [
            ("rust_subprocess", 1, args.out / "rust", DEFAULT_RUST_CLI),
            ("cpp_subprocess", 1, args.out / "cpp", DEFAULT_CPP_CLI),
        ]
    else:
        runs = [
            ("subprocess", 1, args.out / "subprocess", None),
            ("worker", 1, args.out / "worker_w1", None),
        ]

    timings: dict[str, float] = {}
    for label, workers, path, cli in runs:
        backend = "subprocess" if args.compare == "solver_clis" else label.split("_")[0]
        if label.startswith("worker"):
            backend = "worker"
        print(f"\n=== {label} workers={workers} ===", flush=True)
        timings[label] = _run_build(
            config=args.config,
            out_dir=path,
            backend=backend,
            num_workers=workers,
            max_spots=args.max_spots,
            weva_index=args.weva_index,
            cli_path=cli,
        )

    mode = DiffMode(args.mode)
    baseline = runs[0][2]
    results: list[dict] = []
    all_pass = True
    for _label, _workers, path, _cli in runs[1:]:
        diff = compare_label_dirs(baseline, path, mode)
        results.append(
            {
                "expected": str(baseline),
                "actual": str(path),
                "passed": diff.passed,
                "issues": len(diff.issues),
            }
        )
        if not diff.passed:
            all_pass = False

    summary = {
        "compare_mode": args.compare,
        "timings_sec": timings,
        "diff_results": results,
        "all_passed": all_pass,
    }
    (args.out / "canary_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== Canary summary ===")
    print(json.dumps(summary, indent=2))
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
