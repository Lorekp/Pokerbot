#!/usr/bin/env python3
"""Compare Rust postflop_solver_cli vs C++ texas_solver_cli on one request."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.postflop._timing import DEFAULT_CPP_CLI, DEFAULT_RUST_CLI, run_cli_once
from pokerbot.solver.postflop_solver_client import parse_solve_response
from pokerbot.solver.texas_solver_adapter import run_texas_solver


def _parse_stdout(stdout: str) -> dict:
    line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    return json.loads(line)


def _semantic_compare(rust: dict, cpp: dict) -> list[str]:
    issues: list[str] = []
    for key in ("converged",):
        if rust.get(key) != cpp.get(key):
            issues.append(f"{key}: {rust.get(key)!r} != {cpp.get(key)!r}")
    r_dist = rust.get("action_distribution") or {}
    c_dist = cpp.get("action_distribution") or {}
    keys = sorted(set(r_dist) | set(c_dist))
    for k in keys:
        rv = float(r_dist.get(k, 0.0))
        cv = float(c_dist.get(k, 0.0))
        if abs(rv - cv) > 1e-4 and abs(rv - cv) > 1e-3 * max(abs(rv), abs(cv), 1e-9):
            issues.append(f"action_distribution[{k}]: {rv} != {cv}")
    return issues


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--rust-cli", type=Path, default=None)
    p.add_argument("--cpp-cli", type=Path, default=None)
    p.add_argument("--skip-rust", action="store_true")
    p.add_argument(
        "--python-adapter",
        action="store_true",
        help="Use Python TexasSolver adapter (when C++ texas_solver_cli is not built)",
    )
    args = p.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    rust_cli = args.rust_cli or DEFAULT_RUST_CLI
    cpp_cli = args.cpp_cli or DEFAULT_CPP_CLI

    if not args.skip_rust:
        if not rust_cli.is_file():
            raise SystemExit(f"Rust CLI missing: {rust_cli}")
        rw, r_out, r_err, r_code = run_cli_once(rust_cli, request)
        print(f"rust: wall={rw:.2f}s exit={r_code}")
        if r_code != 0:
            print(r_err[:500])
            raise SystemExit(1)
        rust_resp = _parse_stdout(r_out)
    else:
        rust_resp = {}

    if args.python_adapter or not cpp_cli.is_file():
        if not os.environ.get("TEXAS_SOLVER_CONSOLE") and not os.environ.get("TEXAS_SOLVER_ROOT"):
            print("Set TEXAS_SOLVER_ROOT for Python adapter")
            raise SystemExit(0)
        import time

        t0 = time.perf_counter()
        cpp_resp = run_texas_solver(request)
        cw = time.perf_counter() - t0
        print(f"texas (python adapter): wall={cw:.2f}s")
        if cpp_resp.get("error"):
            print(f"error: {cpp_resp['error']}")
            raise SystemExit(1)
    else:
        if not os.environ.get("TEXAS_SOLVER_CONSOLE") and not os.environ.get("TEXAS_SOLVER_ROOT"):
            print("TEXAS_SOLVER_CONSOLE or TEXAS_SOLVER_ROOT not set — C++ adapter cannot run engine")
            raise SystemExit(0)

        cw, c_out, c_err, c_code = run_cli_once(cpp_cli, request, timeout_sec=7200.0)
        print(f"cpp:  wall={cw:.2f}s exit={c_code}")
        if c_code != 0:
            print(c_err[:500])
            print(c_out[:500])
            raise SystemExit(1)
        cpp_resp = _parse_stdout(c_out)

    if args.skip_rust:
        print(json.dumps(cpp_resp, indent=2)[:2000])
        return

    issues = _semantic_compare(rust_resp, cpp_resp)
    if issues:
        print("PARITY FAIL:")
        for i in issues:
            print(f"  - {i}")
        raise SystemExit(1)
    speedup = rw / cw if cw > 0 else 0.0
    print(f"PARITY PASS (semantic action distribution). speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
