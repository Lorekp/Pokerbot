#!/usr/bin/env python3
"""Staged cheap-pilot ladder with explicit kill criteria.

Runs increasing-cost validation tiers and stops on kill. Does NOT launch cloud CFR
unless --launch-cfr is passed (stage 4 only).

Stages:
  1. discriminant  - pytest on selection/gate/quality controls
  2. abstraction   - emd_abstraction_sweep.py (multi-seed, survivors)
  3. eval-harness  - profile_emd_native_eval summarize (or run if partials exist)
  4. cfr-pilot     - emd_pilot_matrix.ps1 dry-run (or launch with --launch-cfr)
  5. decision      - emd_full_run_decision.py recommendation

Writes artifacts/emd_pilot_ladder/<stamp>/ladder_report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


DISCRIMINANT_TESTS = [
    "tests/test_emd_selection_gate.py",
    "tests/test_auto_activate_gate.py",
    "tests/test_emd_eval_combine_magnitude.py",
    "tests/test_emd_quality.py",
    "tests/test_emd_frozen_miss.py",
]


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 3600) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-4000:] if proc.stderr else "",
    }


def stage_discriminant(*, skip: bool) -> dict[str, Any]:
    if skip:
        return {"skipped": True, "pass": True}
    cmd = [sys.executable, "-m", "pytest", "-q", *DISCRIMINANT_TESTS]
    result = _run(cmd, timeout=900)
    result["pass"] = result["returncode"] == 0
    if not result["pass"]:
        result["kill_reason"] = "known_bad_control_or_gate_test_failed"
    return result


def stage_abstraction(
    *,
    skip: bool,
    out_json: Path,
    k_grid: str,
    runouts_grid: str,
    train_samples: int,
) -> dict[str, Any]:
    if skip:
        return {"skipped": True, "pass": True}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "emd_abstraction_sweep.py"),
        "--k-grid", k_grid,
        "--runouts-grid", runouts_grid,
        "--train-samples", str(train_samples),
        "--out-json", str(out_json),
        "--out-md", str(out_json.with_suffix(".md")),
    ]
    result = _run(cmd, timeout=7200)
    result["pass"] = result["returncode"] == 0
    survivors: list[dict] = []
    if out_json.exists():
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        survivors = payload.get("survivors") or []
        result["n_survivors"] = len(survivors)
        result["survivors"] = survivors[:5]
    if not result["pass"]:
        result["kill_reason"] = "abstraction_sweep_failed"
    elif len(survivors) < 1:
        result["pass"] = False
        result["kill_reason"] = "no_abstraction_survivors"
    return result


def stage_eval_harness(
    *,
    skip: bool,
    emd_partial: Path,
    base_partial: Path,
    verdict_out: Path,
    tier: str,
) -> dict[str, Any]:
    if skip:
        return {"skipped": True, "pass": True}
    if not emd_partial.exists() or not base_partial.exists():
        return {
            "pass": False,
            "kill_reason": "eval_partials_missing",
            "hint": f"need {emd_partial} and {base_partial}",
        }
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "profile_emd_native_eval.py"),
        "summarize",
        "--emd-json", str(emd_partial),
        "--iter62000-json", str(base_partial),
        "--out-json", str(verdict_out),
        "--selection-tier", tier,
    ]
    result = _run(cmd, timeout=600)
    result["pass"] = result["returncode"] == 0
    if verdict_out.exists():
        verdict = json.loads(verdict_out.read_text(encoding="utf-8"))
        sel = verdict.get("selection") or {}
        result["selection_pass"] = sel.get("selection_pass")
        if sel.get("selection_pass") is False:
            result["pass"] = False
            result["kill_reason"] = "eval_harness_selection_failed"
            result["selection_reason_codes"] = sel.get("selection_reason_codes")
    elif not result["pass"]:
        result["kill_reason"] = "eval_harness_summarize_failed"
    return result


def stage_cfr_pilot(*, skip: bool, launch: bool, k_grid: str) -> dict[str, Any]:
    if skip:
        return {"skipped": True, "pass": True}
    ps1 = ROOT / "scripts" / "emd_pilot_matrix.ps1"
    args = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps1), "-KGrid", k_grid]
    if launch:
        if "," in k_grid.strip().strip(","):
            return {"pass": False, "kill_reason": "launch_cfr_requires_single_k"}
        args.append("-Launch")
    result = _run(args, timeout=300 if not launch else 86400)
    result["pass"] = result["returncode"] == 0
    result["dry_run"] = not launch
    if not result["pass"]:
        result["kill_reason"] = "cfr_pilot_launch_failed" if launch else "cfr_pilot_dry_run_failed"
    return result


def stage_decision(*, skip: bool, artifacts_dir: Path) -> dict[str, Any]:
    if skip:
        return {"skipped": True, "pass": True}
    out = artifacts_dir / "full_run_decision.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "emd_full_run_decision.py"),
        "--artifacts-dir", str(artifacts_dir),
        "--out", str(out),
    ]
    result = _run(cmd, timeout=120)
    result["pass"] = result["returncode"] == 0
    if out.exists():
        decision = json.loads(out.read_text(encoding="utf-8"))
        result["recommend_launch"] = decision.get("recommend_launch")
        result["decision"] = decision
        if not decision.get("recommend_launch"):
            result["pass"] = False
            result["kill_reason"] = decision.get("primary_blocker", "full_run_not_recommended")
    elif not result["pass"]:
        result["kill_reason"] = "decision_script_failed"
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stamp", default="", help="output subdir under artifacts/emd_pilot_ladder/")
    ap.add_argument("--from-stage", type=int, default=1, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--skip-discriminant", action="store_true")
    ap.add_argument("--skip-abstraction", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-cfr", action="store_true")
    ap.add_argument("--skip-decision", action="store_true")
    ap.add_argument("--k-grid", default="80,120,160,200")
    ap.add_argument("--runouts-grid", default="100,200,400")
    ap.add_argument("--train-samples", type=int, default=12000)
    ap.add_argument("--selection-tier", default="cheap")
    ap.add_argument("--emd-partial", default="artifacts/emd_k500_partial/native_eval_emd.json")
    ap.add_argument("--base-partial", default="artifacts/emd_k500_partial/native_eval_iter62000.json")
    ap.add_argument("--launch-cfr", action="store_true")
    ap.add_argument("--cfr-k-grid", default="120,160,200")
    args = ap.parse_args(argv)

    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    art_dir = ROOT / "artifacts" / "emd_pilot_ladder" / stamp
    art_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema": "emd_pilot_ladder_v1",
        "stamp": stamp,
        "stages": {},
        "killed": False,
        "kill_stage": None,
        "kill_reason": None,
    }

    stages = [
        (1, "discriminant", lambda: stage_discriminant(skip=args.skip_discriminant)),
        (2, "abstraction", lambda: stage_abstraction(
            skip=args.skip_abstraction,
            out_json=art_dir / "abstraction_sweep.json",
            k_grid=args.k_grid,
            runouts_grid=args.runouts_grid,
            train_samples=args.train_samples,
        )),
        (3, "eval_harness", lambda: stage_eval_harness(
            skip=args.skip_eval,
            emd_partial=ROOT / args.emd_partial,
            base_partial=ROOT / args.base_partial,
            verdict_out=art_dir / "eval_verdict.json",
            tier=args.selection_tier,
        )),
        (4, "cfr_pilot", lambda: stage_cfr_pilot(
            skip=args.skip_cfr,
            launch=args.launch_cfr,
            k_grid=args.cfr_k_grid,
        )),
        (5, "decision", lambda: stage_decision(skip=args.skip_decision, artifacts_dir=art_dir)),
    ]

    for num, name, fn in stages:
        if num < args.from_stage:
            continue
        stage_result = fn()
        report["stages"][name] = stage_result
        out_path = art_dir / "ladder_report.json"
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if not stage_result.get("pass", False) and not stage_result.get("skipped"):
            report["killed"] = True
            report["kill_stage"] = name
            report["kill_reason"] = stage_result.get("kill_reason", "stage_failed")
            out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"KILLED at stage {name}: {report['kill_reason']}", file=sys.stderr)
            return 2

    out_path = art_dir / "ladder_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
