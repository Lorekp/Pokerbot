"""JSON export for experimental solver skeleton."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from aof_solver import EXPERIMENTAL_BANNER
from aof_solver.cfr import RegretTable


def export_strategy_json(
    table: RegretTable,
    *,
    stack_bb: float = 8.0,
    iterations: int = 1,
    seed: int | None = None,
    exploitability_proxy: float | None = None,
) -> dict[str, Any]:
    avg: dict[str, list[float]] = {}
    for key, sums in table.strategy_sum.items():
        total = sums.sum()
        if total > 0:
            avg[key] = (sums / total).tolist()
        else:
            avg[key] = [0.5, 0.5]
    return {
        "banner": EXPERIMENTAL_BANNER,
        "provenance": "experimental_cfr_skeleton",
        "not_for_production": True,
        "game": {"id": "hu_pushfold_toy", "players": 2, "stack_bb": stack_bb},
        "stack_bb": stack_bb,
        "iterations": iterations,
        "seed": seed,
        "exploitability_proxy": exploitability_proxy,
        "exploitability_note": (
            None
            if exploitability_proxy is not None
            else "HU toy game — no exact exploitability computed in skeleton"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy_table": avg,
        "average_strategy": avg,
        "warning": "NOT VALIDATED — do not load into production advisor",
    }


def write_export(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
