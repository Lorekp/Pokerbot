"""Tests for RAYON_NUM_THREADS passthrough on subprocess solver path."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pokerbot.solver.postflop_solver_client import (
    DEFAULT_CLI_PATH,
    run_solver,
    solver_subprocess_env,
)


def test_solver_subprocess_env_sets_rayon_when_requested() -> None:
    env = solver_subprocess_env(1)
    assert env["RAYON_NUM_THREADS"] == "1"


def test_solver_subprocess_env_honors_existing_rayon_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAYON_NUM_THREADS", "4")
    env = solver_subprocess_env(1)
    assert env["RAYON_NUM_THREADS"] == "4"


@pytest.mark.skipif(not DEFAULT_CLI_PATH.is_file(), reason="postflop_solver_cli not built")
def test_run_solver_passes_rayon_threads() -> None:
    request = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "external"
            / "postflop_solver_cli"
            / "test_request.json"
        ).read_text(encoding="utf-8")
    )
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> MagicMock:
        captured["env"] = kwargs.get("env")
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = json.dumps(
            {
                "exploitability": 0.0,
                "target_exploitability": 0.0,
                "exploitability_pct_of_pot": 0.0,
                "converged": True,
                "iterations": 1,
                "iterations_used": 1,
                "available_actions": ["check"],
                "action_distribution": {"check": 1.0},
                "solver_version": "test",
            }
        )
        proc.stderr = ""
        return proc

    with patch("pokerbot.solver.postflop_solver_client.subprocess.run", side_effect=fake_run):
        run_solver(request, cli_path=DEFAULT_CLI_PATH, solver_threads_per_worker=1)

    env = captured.get("env")
    assert isinstance(env, dict)
    assert env.get("RAYON_NUM_THREADS") == "1"
