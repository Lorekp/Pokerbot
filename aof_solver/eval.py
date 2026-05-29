"""Smoke evaluation for CFR skeleton."""

from __future__ import annotations

from aof_solver.cfr import RegretTable, run_one_cfr_iteration
from aof_solver.game import HuGameConfig, HuState


def smoke_one_iteration() -> RegretTable:
    cfg = HuGameConfig()
    table = RegretTable()
    buckets = ["AA", "72o"]
    eq = {"AA": 0.85, "72o": 0.35}
    run_one_cfr_iteration(table, buckets, eq, cfg)
    assert len(table.regret) > 0
    s = HuState()
    assert not s.is_terminal()
    s2 = s.apply("A")
    assert s2.current_player() == "BB"
    return table
