#!/usr/bin/env python3
"""Run AoF snapshot scripts and build validation summary."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SEED = 42


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=_ROOT, check=True)


def _device_cli(device: str, allow_cpu: bool) -> list[str]:
    out = ["--device", device]
    if allow_cpu:
        out.append("--allow-cpu")
    return out


def _maybe_train_smoke(
    py: str,
    *,
    seed: int,
    device: str,
    allow_cpu: bool,
    train_smoke: bool,
) -> None:
    ck_v1 = _ROOT / "checkpoints/aof_smoke_v1.pt"
    ck_v2 = _ROOT / "checkpoints/aof_smoke_v2.pt"
    if ck_v1.is_file() and ck_v2.is_file():
        return
    if not train_smoke:
        print(
            "Skipping policy eval/compare: smoke checkpoints missing. "
            "Pass --train-smoke or train manually.",
            flush=True,
        )
        return
    common = [
        "--profile",
        "smoke",
        "--seed",
        str(seed),
        "--updates",
        "8",
        "--hands-per-update",
        "32",
        *_device_cli(device, allow_cpu),
    ]
    if not ck_v1.is_file():
        _run(
            [
                py,
                str(_ROOT / "scripts/train_aof_self_play.py"),
                *common,
                "--obs-version",
                "1",
                "--teacher-version",
                "1",
                "--out",
                str(ck_v1),
            ]
        )
    if not ck_v2.is_file():
        _run(
            [
                py,
                str(_ROOT / "scripts/train_aof_self_play.py"),
                *common,
                "--obs-version",
                "2",
                "--teacher-version",
                "2",
                "--out",
                str(ck_v2),
            ]
        )


def main() -> int:
    p = argparse.ArgumentParser(description="AoF validation artifact suite")
    p.add_argument("--seed", type=int, default=_SEED)
    p.add_argument("--skip-policy", action="store_true", help="Skip slow train/compare")
    p.add_argument(
        "--device",
        default="cuda",
        help="Device for RL eval/compare/train-smoke (default: cuda)",
    )
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback when CUDA unavailable",
    )
    p.add_argument(
        "--train-smoke",
        action="store_true",
        help="Train smoke v1/v2 checkpoints if missing before eval",
    )
    args = p.parse_args()
    py = sys.executable
    artifacts = _ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    dev_flags = _device_cli(args.device, args.allow_cpu)

    snapshots = [
        ("scripts/aof_multiway_equity_snapshot.py", "aof_multiway_equity_snapshot.json"),
        ("scripts/aof_open_multiway_snapshot.py", "aof_open_multiway_snapshot.json"),
        ("scripts/aof_call_behind_snapshot.py", "aof_call_behind_snapshot.json"),
        ("scripts/aof_range_sensitivity_snapshot.py", "aof_range_sensitivity_snapshot.json"),
    ]
    for script, name in snapshots:
        out = artifacts / name
        _run([py, str(_ROOT / script), "-o", str(out)])

    _run(
        [
            py,
            str(_ROOT / "scripts/aof_rl_teacher_mismatch_report.py"),
            "--teacher-version",
            "1",
            "--obs-version",
            "1",
            "--seed",
            str(args.seed),
            "-o",
            str(artifacts / "aof_rl_teacher_mismatch_v1.json"),
        ]
    )
    _run(
        [
            py,
            str(_ROOT / "scripts/aof_rl_teacher_mismatch_report.py"),
            "--teacher-version",
            "2",
            "--obs-version",
            "2",
            "--seed",
            str(args.seed),
            "-o",
            str(artifacts / "aof_rl_teacher_mismatch_v2.json"),
        ]
    )

    if not args.skip_policy:
        _maybe_train_smoke(
            py,
            seed=args.seed,
            device=args.device,
            allow_cpu=args.allow_cpu,
            train_smoke=args.train_smoke,
        )
        ck_v1 = _ROOT / "checkpoints/aof_smoke_v1.pt"
        ck_v2 = _ROOT / "checkpoints/aof_smoke_v2.pt"
        if ck_v1.is_file() and ck_v2.is_file():
            _run(
                [
                    py,
                    str(_ROOT / "scripts/eval_aof_policy.py"),
                    "--checkpoint",
                    str(ck_v1),
                    "--obs-version",
                    "1",
                    "--teacher-version",
                    "1",
                    "--seed",
                    str(args.seed),
                    "--hands",
                    "2000",
                    "--profile",
                    "fast",
                    *dev_flags,
                    "-o",
                    str(artifacts / "eval_v1.json"),
                ]
            )
            _run(
                [
                    py,
                    str(_ROOT / "scripts/eval_aof_policy.py"),
                    "--checkpoint",
                    str(ck_v2),
                    "--obs-version",
                    "2",
                    "--teacher-version",
                    "2",
                    "--seed",
                    str(args.seed),
                    "--hands",
                    "2000",
                    "--profile",
                    "fast",
                    *dev_flags,
                    "-o",
                    str(artifacts / "eval_v2.json"),
                ]
            )
            _run(
                [
                    py,
                    str(_ROOT / "scripts/aof_compare_policy_versions.py"),
                    "--checkpoints",
                    str(ck_v1),
                    str(ck_v2),
                    "--names",
                    "smoke_v1",
                    "smoke_v2",
                    "--hands",
                    "5000",
                    "--seed",
                    str(args.seed),
                    "--profile",
                    "fast",
                    *dev_flags,
                    "-o",
                    str(artifacts / "aof_policy_version_comparison.json"),
                ]
            )

    _run([py, str(_ROOT / "scripts/aof_summarize_validation_artifacts.py")])
    _run([py, str(_ROOT / "scripts/aof_static_range_audit.py")])
    print("Validation suite complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
