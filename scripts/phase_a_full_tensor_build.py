#!/usr/bin/env python3
"""Phase A: full 169-row tensor build at locked bpc + end-to-end validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_cfr.cpu_pool import (
    assert_no_oversubscription,
    describe_parallel_layout,
    kill_stale_tensor_builds,
    logical_cpus,
    omp_threads_per_row_worker,
)
from aof_cfr.run_logger import fmt_elapsed, init_run_log, log_step
from aof_cfr.trio_pair_eq_table import (
    DEFAULT_PAIR_EQ_TABLE,
    require_pair_eq_table,
    table_is_complete,
)
from aof_cfr.trio_payoff_tensor import DEFAULT_BOARDS_PER_COMBO, DEFAULT_TENSOR, DEFAULT_TENSOR_CHECKPOINT_DIR

GATE_JSON = Path("data/aof/cfr_base/phase_a_gate_derisk.json")
SUMMARY_JSON = Path("data/aof/cfr_base/phase_a_full_build_summary.json")
VALIDATE_JSON = Path("data/aof/cfr_base/phase_a_full_tensor_validate.json")


def _pair_eq_ready() -> tuple[bool, str]:
    if not DEFAULT_PAIR_EQ_TABLE.is_file():
        return False, f"missing {DEFAULT_PAIR_EQ_TABLE}"
    try:
        hu = require_pair_eq_table()
    except Exception as exc:
        return False, f"load failed: {exc}"
    if not table_is_complete(hu):
        return False, "169×169 table incomplete or invalid"
    return True, f"169×169 complete shape={hu.shape}"


def _run(cmd: list[str], *, step: str) -> None:
    log_step(f"{step}: {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(_ROOT))
    if rc.returncode != 0:
        raise SystemExit(f"{step} failed exit={rc.returncode}")


def _gate_boards() -> int:
    if GATE_JSON.is_file():
        data = json.loads(GATE_JSON.read_text(encoding="utf-8"))
        locked = data.get("locked_boards_per_combo")
        if locked is not None:
            return int(locked)
    return DEFAULT_BOARDS_PER_COMBO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row-workers", type=int, default=3)
    ap.add_argument(
        "--no-row-parallel",
        action="store_true",
        help="Serial rows: one process, one OpenMP pool (safer on 6-core boxes)",
    )
    ap.add_argument(
        "--omp-threads",
        type=int,
        default=6,
        help="OpenMP threads when --no-row-parallel (default: 6 physical cores)",
    )
    ap.add_argument("--pair-eq-workers", type=int, default=3)
    ap.add_argument("--boards", type=int, default=None)
    ap.add_argument("--samples-per-row", type=int, default=20)
    ap.add_argument("--skip-pair-eq", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_TENSOR_CHECKPOINT_DIR),
        help="Per-row checkpoint directory",
    )
    ap.add_argument("--no-checkpoint", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    log_path = init_run_log("phase_a_full_tensor_build")
    boards = args.boards if args.boards is not None else _gate_boards()
    if args.no_row_parallel:
        row_workers = 1
        omp_per = max(1, int(args.omp_threads))
        assert_no_oversubscription(1, omp_per, label="tensor build")
    else:
        row_workers = max(1, args.row_workers)
        omp_per = omp_threads_per_row_worker(row_workers)
        assert_no_oversubscription(row_workers, omp_per, label="tensor build")
    par = describe_parallel_layout(row_workers, omp_per)

    log_step(
        f"Phase A full build: bpc={boards} row_parallel={not args.no_row_parallel} "
        f"row_workers={row_workers} omp={omp_per} layout={par} "
        f"logical_cpus={logical_cpus()} log={log_path}"
    )

    ok_pe, pe_msg = _pair_eq_ready()
    log_step(f"Pair-eq table: {pe_msg}")

    if not ok_pe and not args.skip_pair_eq:
        log_step("Step 1/3: build 169×169 pair-eq table (single writer)")
        _run(
            [sys.executable, str(_ROOT / "scripts" / "build_trio_pair_eq_table.py"), "--kill-stale"],
            step="pair-eq table",
        )
    elif not ok_pe:
        raise SystemExit(f"Pair-eq table not ready: {pe_msg}")
    else:
        log_step("Step 1/3: pair-eq table complete — skip")

    if not args.skip_build:
        kill_stale_tensor_builds(include_pair_eq=False)
        log_step(f"Step 2/3: full tensor build {par}")
        t0 = time.monotonic()
        tensor_cmd = [
            sys.executable,
            str(_ROOT / "scripts" / "build_3way_eq_tensor_npz.py"),
            "--mc-boards",
            str(boards),
            "--out",
            str(DEFAULT_TENSOR),
            "--kill-stale",
            "--omp-threads",
            str(omp_per),
        ]
        if args.no_row_parallel:
            tensor_cmd.append("--no-row-parallel")
        else:
            tensor_cmd.extend(["--row-parallel", "--row-workers", str(row_workers)])
        if args.no_checkpoint:
            tensor_cmd.append("--no-checkpoint")
        else:
            tensor_cmd.extend(["--checkpoint-dir", str(args.checkpoint_dir)])
        if args.no_resume:
            tensor_cmd.append("--no-resume")
        _run(tensor_cmd, step="tensor build")
        build_elapsed = time.monotonic() - t0
        log_step(f"Tensor build wall time: {fmt_elapsed(t0)} ({build_elapsed:.1f}s)")
    else:
        build_elapsed = 0.0
        log_step("Step 2/3: tensor build skipped")

    if not args.skip_validate:
        log_step(f"Step 3/3: full-tensor validation samples_per_row={args.samples_per_row}")
        _run(
            [
                sys.executable,
                str(_ROOT / "scripts" / "phase_a_full_tensor_validate.py"),
                "--tensor",
                str(DEFAULT_TENSOR),
                "--samples-per-row",
                str(args.samples_per_row),
                "--omp-threads",
                str(omp_per),
            ],
            step="full tensor validate",
        )

    summary = {
        "pipeline": "phase_a_full_tensor_build",
        "boards_per_combo": boards,
        "row_parallel": not args.no_row_parallel,
        "row_workers": row_workers,
        "omp_threads": omp_per,
        "parallel_layout": par,
        "tensor": str(DEFAULT_TENSOR),
        "checkpoint_dir": None if args.no_checkpoint else str(args.checkpoint_dir),
        "resume": not args.no_resume,
        "build_elapsed_s": build_elapsed,
        "validation_json": str(VALIDATE_JSON),
        "log_file": str(log_path),
    }
    if VALIDATE_JSON.is_file():
        summary["validation"] = json.loads(VALIDATE_JSON.read_text(encoding="utf-8"))
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_step(f"Pipeline complete — wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
