"""Experimental solver toy train/validate pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from aof_solver.export import export_strategy_json
from aof_solver.import_experimental import (
    ExperimentalSolverError,
    load_experimental_solver_json,
)


def test_import_rejects_production_flag(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "provenance": "experimental_cfr_skeleton",
                "not_for_production": False,
                "strategy_table": {},
            }
        ),
        encoding="utf-8",
    )
    try:
        load_experimental_solver_json(bad)
        raise AssertionError("expected error")
    except ExperimentalSolverError:
        pass


def test_toy_cli_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "toy.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/aof_solver_toy_train.py"),
            "--iterations",
            "5",
            "--seed",
            "42",
            "-o",
            str(out),
        ],
        cwd=root,
        check=True,
    )
    data = load_experimental_solver_json(out)
    assert data["not_for_production"] is True
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/aof_solver_toy_validate.py"),
            "--input",
            str(out),
        ],
        cwd=root,
        check=True,
    )


def test_export_has_not_for_production() -> None:
    from aof_solver.cfr import RegretTable

    payload = export_strategy_json(RegretTable(), iterations=1, seed=1)
    assert payload["not_for_production"] is True
