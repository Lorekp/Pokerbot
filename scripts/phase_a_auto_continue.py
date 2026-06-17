#!/usr/bin/env python3
"""Wait for spot-check, then run full pair-eq table + tensor build + validation."""

from __future__ import annotations

import argparse
import json
import os
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
    native_row_pool_workers,
    omp_threads_per_row_worker,
)
from aof_cfr.run_logger import init_run_log, log_step
from aof_cfr.trio_payoff_tensor import DEFAULT_BOARDS_PER_COMBO, DEFAULT_TENSOR
from aof_cfr.trio_pair_eq_table import DEFAULT_PAIR_EQ_TABLE, require_pair_eq_table, table_is_complete

GATE_JSON = Path("data/aof/cfr_base/phase_a_gate_derisk.json")
PAIR_EQ_WORKERS = 3


def _wait_for_spot_check(*, wait_pid: int | None, wait_log: Path | None, poll_s: float = 15.0) -> None:
    """Wait for spot-check process exit and/or log footer."""
    marker = "phase_a_spot_check_run finished"
    while True:
        pid_done = wait_pid is None
        log_done = wait_log is None
        if wait_pid is not None:
            if os.name == "nt":
                proc = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {wait_pid}", "/NH"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                pid_done = str(wait_pid) not in (proc.stdout or "")
            else:
                try:
                    os.kill(wait_pid, 0)
                except OSError:
                    pid_done = True
        if wait_log is not None and wait_log.is_file():
            text = wait_log.read_text(encoding="utf-8", errors="replace")
            log_done = marker in text
        if pid_done and log_done:
            return
        time.sleep(poll_s)


def _gate_ready() -> tuple[bool, str, dict]:
    if not GATE_JSON.is_file():
        return False, f"missing {GATE_JSON}", {}
    data = json.loads(GATE_JSON.read_text(encoding="utf-8"))
    locked = data.get("locked_boards_per_combo")
    if locked is None:
        return False, "no locked boards_per_combo", data

    pooled = data.get("pooled_gate") or {}
    if pooled:
        if not pooled.get("pass_threshold", False):
            ucb = pooled.get("upper_bound_95", float("inf"))
            return False, f"pooled UCB95={ucb:.6f} > threshold", data
        return (
            True,
            f"locked bpc={locked} pooled_mu={pooled.get('weighted_mean', 0):.6f} "
            f"UCB95={pooled.get('upper_bound_95', 0):.6f}",
            data,
        )

    # Legacy JSON without pooled_gate: fall back to per-run pass
    matrix = data.get("gate_matrix") or []
    if not matrix:
        return False, "empty gate_matrix (no pooled_gate)", data
    threshold_fail = [r for r in matrix if not r.get("pass")]
    if threshold_fail:
        worst = max(threshold_fail, key=lambda r: r["weighted_error"])
        return (
            False,
            f"threshold fail row={worst['row']} werr={worst['weighted_error']:.6f}",
            data,
        )
    return True, f"locked bpc={locked} (legacy per-run pass)", data


def _built_pair_eq_rows() -> set[int]:
    import numpy as np

    if not DEFAULT_PAIR_EQ_TABLE.is_file():
        return set()
    data = np.load(DEFAULT_PAIR_EQ_TABLE)
    if "built_rows" not in data:
        return set()
    return {int(x) for x in data["built_rows"].tolist()}


def _run(cmd: list[str], *, step: str) -> None:
    log_step(f"{step}: {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(_ROOT))
    if rc.returncode != 0:
        raise SystemExit(f"{step} failed exit={rc.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--wait-pid",
        type=int,
        default=None,
        help="Spot-check parent PID to wait on (default: do not wait)",
    )
    ap.add_argument(
        "--wait-log",
        type=Path,
        default=None,
        help="Spot-check log to poll for completion",
    )
    ap.add_argument(
        "--skip-wait",
        action="store_true",
        help="Assume spot-check already finished",
    )
    ap.add_argument(
        "--pair-eq-workers",
        type=int,
        default=PAIR_EQ_WORKERS,
    )
    ap.add_argument(
        "--row-parallel",
        action="store_true",
        default=True,
        help="Build tensor rows in parallel (default: on)",
    )
    ap.add_argument(
        "--no-row-parallel",
        action="store_true",
        help="Serial rows, full OpenMP per row",
    )
    args = ap.parse_args()

    log_path = init_run_log("phase_a_auto_continue")
    row_par = bool(args.row_parallel and not args.no_row_parallel)
    row_workers = native_row_pool_workers() if row_par else 1
    omp_per = omp_threads_per_row_worker(row_workers) if row_par else None

    log_step(
        f"Auto-continue pipeline log={log_path} "
        f"boards_per_combo={DEFAULT_BOARDS_PER_COMBO} row_parallel={row_par}"
    )

    if not args.skip_wait:
        if args.wait_pid or args.wait_log:
            log_step(
                f"Waiting for spot-check pid={args.wait_pid} log={args.wait_log} ..."
            )
            _wait_for_spot_check(wait_pid=args.wait_pid, wait_log=args.wait_log)
            log_step("Spot-check finished")
        else:
            log_step("No --wait-pid/--wait-log; checking gate JSON only")

    ok, msg, gate = _gate_ready()
    log_step(f"Gate check: {'OK' if ok else 'FAIL'} — {msg}")
    if not ok:
        raise SystemExit(1)

    ok_pe, pe_msg = True, "ok"
    if DEFAULT_PAIR_EQ_TABLE.is_file():
        try:
            hu = require_pair_eq_table()
            ok_pe = table_is_complete(hu)
            pe_msg = f"169×169 complete={ok_pe}" if ok_pe else "169×169 incomplete"
        except Exception as exc:
            ok_pe = False
            pe_msg = str(exc)
    else:
        ok_pe = False
        pe_msg = f"missing {DEFAULT_PAIR_EQ_TABLE}"

    log_step(f"Pair-eq table: {pe_msg}")

    if not ok_pe:
        log_step("Step 1/3: build 169×169 pair-eq table (single writer)")
        _run(
            [
                sys.executable,
                str(_ROOT / "scripts" / "build_trio_pair_eq_table.py"),
                "--kill-stale",
            ],
            step="pair-eq table",
        )
    else:
        log_step("Step 1/3: pair-eq table complete — skip")

    kill_stale_tensor_builds(include_pair_eq=False)
    if row_par:
        assert_no_oversubscription(row_workers, omp_threads_per_row_worker(row_workers), label="tensor build")
        par = describe_parallel_layout(row_workers, omp_threads_per_row_worker(row_workers))
        log_step(f"Step 2/3: full tensor build row_parallel workers={row_workers} {par}")
        tensor_cmd = [
            sys.executable,
            str(_ROOT / "scripts" / "build_3way_eq_tensor_npz.py"),
            "--row-parallel",
            "--mc-boards",
            str(DEFAULT_BOARDS_PER_COMBO),
            "--out",
            str(DEFAULT_TENSOR),
        ]
    else:
        log_step("Step 2/3: full tensor build serial rows (full OpenMP)")
        tensor_cmd = [
            sys.executable,
            str(_ROOT / "scripts" / "build_3way_eq_tensor_npz.py"),
            "--no-row-parallel",
            "--mc-boards",
            str(DEFAULT_BOARDS_PER_COMBO),
            "--out",
            str(DEFAULT_TENSOR),
        ]
    _run(tensor_cmd, step="tensor build")

    log_step("Step 3/3: validate full tensor ...")
    _run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "validate_trio_tensor.py"),
            "--tensor",
            str(DEFAULT_TENSOR),
            "--samples",
            "100",
        ],
        step="validation",
    )

    summary = {
        "pipeline": "phase_a_auto_continue",
        "gate": msg,
        "boards_per_combo": gate.get("locked_boards_per_combo", DEFAULT_BOARDS_PER_COMBO),
        "tensor": str(DEFAULT_TENSOR),
        "row_parallel": row_par,
        "log_file": str(log_path),
    }
    out = Path("data/aof/cfr_base/phase_a_full_build_summary.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_step(f"Pipeline complete — wrote {out}")


if __name__ == "__main__":
    main()
