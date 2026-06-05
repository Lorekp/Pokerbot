#!/usr/bin/env python3
"""Deprecated: builds full 169×169 table via build_trio_pair_eq_table.py (single writer)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_cfr.run_logger import init_run_log, log_step


def main() -> None:
    init_run_log("build_pair_eq_rows_parallel")
    log_step("Redirecting to scripts/build_trio_pair_eq_table.py (169×169 eager, single writer)")
    rc = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "build_trio_pair_eq_table.py"), "--kill-stale"],
        cwd=str(_ROOT),
    )
    raise SystemExit(rc.returncode)


if __name__ == "__main__":
    main()
