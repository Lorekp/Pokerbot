"""Experimental aof_solver package tests (Priority 14)."""

from __future__ import annotations

import importlib
import importlib.util

from aof_solver import EXPERIMENTAL_BANNER
from aof_solver.cfr import RegretTable, run_one_cfr_iteration
from aof_solver.eval import smoke_one_iteration
from aof_solver.export import export_strategy_json
from aof_solver.game import HuGameConfig, HuState
from aof_solver.infoset import infoset_key


def test_terminal_detection() -> None:
    s = HuState()
    assert not s.is_terminal()
    assert s.apply("F").is_terminal()
    assert s.apply("A").apply("F").is_terminal()
    assert s.apply("A").apply("A").is_terminal()


def test_infoset_key_stable() -> None:
    st = HuState(actions=("A",))
    k = infoset_key("BB", "AA", st)
    assert k == "BB|AA|A"


def test_one_cfr_iteration_updates_regrets() -> None:
    table = smoke_one_iteration()
    assert any(abs(r).sum() >= 0 for r in table.regret.values())


def test_export_provenance() -> None:
    table = RegretTable()
    run_one_cfr_iteration(table, ["AA"], {"AA": 0.8}, HuGameConfig())
    payload = export_strategy_json(table, iterations=1)
    assert payload["banner"] == EXPERIMENTAL_BANNER
    assert payload["provenance"] == "experimental_cfr_skeleton"
    assert payload["not_for_production"] is True


def test_aof_ranges_does_not_import_solver() -> None:
    spec = importlib.util.find_spec("aof_ranges")
    assert spec is not None
    importlib.import_module("aof_ranges")
    src = spec.origin
    assert src
    text = open(src, encoding="utf-8").read()
    assert "aof_solver" not in text
