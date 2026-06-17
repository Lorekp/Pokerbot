#!/usr/bin/env python3
"""Run target-scale merge gate (+ optional cloud validate) with subprocess timeout and logging."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MERGE_DEFAULT_TIMEOUT_SEC = 10_800  # 3h
VALIDATE_DEFAULT_TIMEOUT_SEC = 14_400  # 4h


def _run(cmd: list[str], timeout_sec: float, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN: {' '.join(cmd)}", flush=True)
    print(f"  timeout_sec={timeout_sec} log={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"COMMAND: {' '.join(cmd)}\n")
        log_f.write(f"TIMEOUT_SEC: {timeout_sec}\n\n")
        log_f.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                timeout=timeout_sec if timeout_sec > 0 else None,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            return proc.returncode
        except subprocess.TimeoutExpired:
            log_f.write(f"\n\nKILLED: exceeded timeout_sec={timeout_sec}\n")
            log_f.flush()
            print(f"KILLED: exceeded timeout_sec={timeout_sec}", flush=True)
            return 124


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--skip-merge", action="store_true")
    p.add_argument("--merge-timeout-sec", type=float, default=MERGE_DEFAULT_TIMEOUT_SEC)
    p.add_argument("--validate-timeout-sec", type=float, default=VALIDATE_DEFAULT_TIMEOUT_SEC)
    p.add_argument("--max-wall-sec", type=float, default=MERGE_DEFAULT_TIMEOUT_SEC)
    args = p.parse_args()

    py = sys.executable
    report: dict = {"steps": []}

    if not args.skip_merge:
        merge_log = ROOT / "logs" / "sd_cfr_nlhe_merge_gate_target_scale.log"
        merge_cmd = [
            py,
            "scripts/run_sd_cfr_nlhe_merge_gate.py",
            "--target-scale",
            f"--max-wall-sec={args.max_wall_sec}",
        ]
        code = _run(merge_cmd, args.merge_timeout_sec, merge_log)
        report["steps"].append(
            {
                "name": "merge_gate_target_scale",
                "exit_code": code,
                "log": str(merge_log),
                "passed": code == 0,
            }
        )
        if code != 0:
            report["passed"] = False
            out = ROOT / "logs" / "sd_cfr_nlhe_target_gates.json"
            out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2), flush=True)
            raise SystemExit(code)

    if args.merge_only:
        report["passed"] = True
    else:
        val_log = ROOT / "logs" / "sd_cfr_hu_cloud_validate_target_scale.log"
        val_cmd = [py, "scripts/run_sd_cfr_hu_cloud_validate_one_iter.py"]
        code = _run(val_cmd, args.validate_timeout_sec, val_log)
        report["steps"].append(
            {
                "name": "cloud_validate_one_iter",
                "exit_code": code,
                "log": str(val_log),
                "passed": code == 0,
            }
        )
        report["passed"] = code == 0

    out = ROOT / "logs" / "sd_cfr_nlhe_target_gates.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
