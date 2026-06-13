"""Tests for persistent postflop_solver_cli worker backend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pokerbot.solver.postflop_solver_client import DEFAULT_CLI_PATH, run_solver
from pokerbot.solver.solver_worker import PersistentSolverWorker, SolverTimeoutError, WorkerError

CLI_TEST_REQUEST = (
    Path(__file__).resolve().parents[1]
    / "external"
    / "postflop_solver_cli"
    / "test_request.json"
)


@pytest.mark.skipif(not DEFAULT_CLI_PATH.is_file(), reason="postflop_solver_cli not built")
def test_worker_matches_subprocess_on_test_request() -> None:
    request = json.loads(CLI_TEST_REQUEST.read_text(encoding="utf-8"))
    sub = run_solver(request, cli_path=DEFAULT_CLI_PATH, timeout_sec=120.0)
    assert sub.error is None, sub.error

    with PersistentSolverWorker(cli_path=DEFAULT_CLI_PATH) as worker:
        wrk = worker.solve(request, request_id="test-0", timeout_sec=120.0)

    assert wrk.error is None
    assert wrk.converged == sub.converged
    assert wrk.iterations_used == sub.iterations_used
    assert wrk.action_distribution == sub.action_distribution
    assert wrk.exploitability_pct_of_pot == pytest.approx(sub.exploitability_pct_of_pot, rel=1e-9)


@pytest.mark.skipif(not DEFAULT_CLI_PATH.is_file(), reason="postflop_solver_cli not built")
def test_worker_multiple_requests_same_process() -> None:
    request = json.loads(CLI_TEST_REQUEST.read_text(encoding="utf-8"))
    with PersistentSolverWorker(cli_path=DEFAULT_CLI_PATH) as worker:
        r1 = worker.solve(request, request_id="0", timeout_sec=120.0)
        r2 = worker.solve(request, request_id="1", timeout_sec=120.0)
    assert r1.action_distribution == r2.action_distribution


@pytest.mark.skipif(not DEFAULT_CLI_PATH.is_file(), reason="postflop_solver_cli not built")
def test_worker_invalid_request_returns_structured_error() -> None:
    with PersistentSolverWorker(cli_path=DEFAULT_CLI_PATH) as worker:
        with pytest.raises(WorkerError) as exc:
            worker.solve({"bad": True}, request_id="bad-1", timeout_sec=30.0)
    assert exc.value.code in ("json_parse", "worker_failure", "missing_result")


def test_worker_solve_times_out_when_no_response(monkeypatch: pytest.MonkeyPatch) -> None:
    if not DEFAULT_CLI_PATH.is_file():
        pytest.skip("postflop_solver_cli not built")

    import pokerbot.solver.solver_worker as sw

    monkeypatch.setattr(sw, "_readline_with_timeout", lambda _stream, _timeout: None)

    with PersistentSolverWorker(cli_path=DEFAULT_CLI_PATH) as worker:
        with pytest.raises(SolverTimeoutError) as exc:
            worker.solve({"noop": True}, request_id="timeout-1", timeout_sec=0.5)
    assert exc.value.code == "timeout"
