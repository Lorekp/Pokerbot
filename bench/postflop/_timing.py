"""Shared timing helpers for postflop solver benchmarks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RUST_CLI = (
    ROOT
    / "external"
    / "postflop_solver_cli"
    / "target"
    / "release"
    / ("postflop_solver_cli.exe" if sys.platform == "win32" else "postflop_solver_cli")
)

DEFAULT_CPP_CLI = (
    ROOT
    / "tools"
    / "texas_solver_cli"
    / "build"
    / ("texas_solver_cli.exe" if sys.platform == "win32" else "texas_solver_cli")
)

GOLDEN_REQUEST = ROOT / "tools" / "texas_solver_cli" / "fixtures" / "golden_request.json"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def resolve_cli(path: Path | None = None) -> Path:
    env = os.environ.get("POSTFLOP_SOLVER_CLI")
    if env:
        return Path(env)
    if path is not None:
        return path
    return DEFAULT_RUST_CLI


def run_cli_once(cli: Path, request: dict[str, Any], *, timeout_sec: float = 7200.0) -> tuple[float, str, str, int]:
    if not cli.is_file():
        raise FileNotFoundError(f"solver CLI not found: {cli}")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(cli)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    wall = time.perf_counter() - t0
    return wall, proc.stdout, proc.stderr, proc.returncode


def spawn_only_overhead(cli: Path, *, repeats: int = 5) -> dict[str, Any]:
    """Measure process spawn + minimal stdin read (invalid request exits fast)."""
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        subprocess.run(
            [str(cli)],
            input="{}",
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        times.append(time.perf_counter() - t0)
    mean = sum(times) / len(times)
    return {
        "cli": str(cli),
        "repeats": repeats,
        "spawn_mean_sec": mean,
        "spawn_samples_sec": times,
    }
