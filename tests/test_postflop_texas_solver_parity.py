"""Parity tests: Rust postflop_solver_cli vs C++ texas_solver_cli."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bench.postflop._timing import DEFAULT_CPP_CLI, DEFAULT_RUST_CLI, GOLDEN_REQUEST, run_cli_once
from pokerbot.solver.label_diff import DiffMode, compare_label_dirs

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "non_aof_labels" / "solver_responses"


def _has_texas_engine() -> bool:
    return bool(os.environ.get("TEXAS_SOLVER_CONSOLE") or os.environ.get("TEXAS_SOLVER_ROOT"))


def _cpp_built() -> bool:
    return DEFAULT_CPP_CLI.is_file()


def _parse_stdout(stdout: str) -> dict:
    line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    return json.loads(line)


@pytest.mark.skipif(not DEFAULT_RUST_CLI.is_file(), reason="Rust CLI not built")
@pytest.mark.slow
def test_rust_cli_golden_request_converges() -> None:
    request = json.loads(GOLDEN_REQUEST.read_text(encoding="utf-8"))
    _, stdout, _, code = run_cli_once(DEFAULT_RUST_CLI, request, timeout_sec=7200.0)
    assert code == 0
    resp = _parse_stdout(stdout)
    assert resp.get("converged") is True
    assert resp.get("error") in (None, "null") or not resp.get("error")


@pytest.mark.skipif(not _cpp_built(), reason="texas_solver_cli not built")
def test_cpp_cli_reports_missing_engine_without_texas() -> None:
    if _has_texas_engine():
        pytest.skip("TexasSolver installed — use integration test instead")
    request = json.loads(GOLDEN_REQUEST.read_text(encoding="utf-8"))
    _, stdout, _, code = run_cli_once(DEFAULT_CPP_CLI, request, timeout_sec=30.0)
    resp = _parse_stdout(stdout)
    assert resp.get("error")
    assert resp.get("converged") is False


@pytest.mark.skipif(not _cpp_built() or not _has_texas_engine(), reason="needs texas_solver_cli + TEXAS_SOLVER_*")
@pytest.mark.slow
def test_cpp_vs_rust_golden_spot_semantic() -> None:
    request = json.loads(GOLDEN_REQUEST.read_text(encoding="utf-8"))
    _, r_out, _, r_code = run_cli_once(DEFAULT_RUST_CLI, request, timeout_sec=7200.0)
    _, c_out, _, c_code = run_cli_once(DEFAULT_CPP_CLI, request, timeout_sec=7200.0)
    assert r_code == 0 and c_code == 0
    rust = _parse_stdout(r_out)
    cpp = _parse_stdout(c_out)
    assert rust.get("converged") == cpp.get("converged")
    r_dist = rust.get("action_distribution") or {}
    c_dist = cpp.get("action_distribution") or {}
    for key in sorted(set(r_dist) | set(c_dist)):
        assert abs(float(r_dist.get(key, 0)) - float(c_dist.get(key, 0))) < 1e-3


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="fixtures missing")
def test_fixture_responses_have_action_distribution() -> None:
    for path in FIXTURES.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        resp = data.get("solver_response") or data
        dist = resp.get("action_distribution")
        assert isinstance(dist, dict), path.name
        assert abs(sum(float(v) for v in dist.values()) - 1.0) < 1e-6 or all(v == 0 for v in dist.values())
