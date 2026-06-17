#!/usr/bin/env python3
"""Spot-check rows 34/83/167: build pair-eq slices, then gate at locked 192 bpc."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_cfr.buckets import LABELS_169
from aof_cfr.cpu_pool import kill_stale_tensor_builds
from aof_cfr.run_logger import init_run_log, log_step
from aof_cfr.trio_payoff_tensor import DEFAULT_BOARDS_PER_COMBO

# 34=72o (trash), 83=T9s (mid), 167=AKs (broadway)
SPOT_ROWS = (34, 83, 167)
PAIR_EQ_WORKERS = 3


def main() -> None:
    log_path = init_run_log("phase_a_spot_check_run")
    kill_stale_tensor_builds(include_pair_eq=False)
    labels = [f"{r}={LABELS_169[r]}" for r in SPOT_ROWS]
    log_step(
        f"Spot-check run start rows=[{', '.join(labels)}] "
        f"boards_per_combo={DEFAULT_BOARDS_PER_COMBO} "
        f"pair_eq_workers={PAIR_EQ_WORKERS} log={log_path}"
    )

    log_step("Step 1/2: build pair-eq table rows (parallel, thread-pinned) ...")
    pe = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "build_pair_eq_rows_parallel.py"),
            "--rows",
            *[str(r) for r in SPOT_ROWS],
            "--workers",
            str(PAIR_EQ_WORKERS),
        ],
        cwd=str(_ROOT),
    )
    if pe.returncode != 0:
        raise SystemExit(pe.returncode)

    log_step("Step 2/2: build tensor rows + multi-seed gate ...")
    gate = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "phase_a_gate_derisk.py"),
            "--rows",
            *[str(r) for r in SPOT_ROWS],
            "--boards",
            str(DEFAULT_BOARDS_PER_COMBO),
        ],
        cwd=str(_ROOT),
    )
    raise SystemExit(gate.returncode)


if __name__ == "__main__":
    main()
