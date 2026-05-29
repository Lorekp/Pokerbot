"""Load experimental solver JSON into an isolated namespace (not production ranges)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExperimentalSolverError(ValueError):
    pass


def load_experimental_solver_json(path: str | Path) -> dict[str, Any]:
    """
    Parse toy CFR export. Raises if not marked experimental / not_for_production.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("provenance") != "experimental_cfr_skeleton":
        raise ExperimentalSolverError(
            f"provenance must be experimental_cfr_skeleton, got {data.get('provenance')!r}"
        )
    if not data.get("not_for_production", False):
        raise ExperimentalSolverError("not_for_production must be true")
    if "average_strategy" not in data and "strategy_table" not in data:
        raise ExperimentalSolverError("missing strategy table")
    return data


def experimental_strategy_namespace(data: dict[str, Any]) -> dict[str, Any]:
    """Return read-only view suitable for tests/tools — never merge into aof_ranges."""
    return {
        "provenance": data.get("provenance"),
        "game": data.get("game"),
        "iterations": data.get("iterations"),
        "strategy": data.get("strategy_table") or data.get("average_strategy"),
        "warning": data.get("warning"),
    }
