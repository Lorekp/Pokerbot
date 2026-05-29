"""Payoff wiring with combo-exact HU showdown table."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("phevaluator")

from aof_equity import warmup
from aof_equity_showdown import build_showdown_equity_table, save_showdown_table
from aof_rl.env import reset_hand
from aof_rl.payoff import compute_payoffs, configure_payoff
import random


def test_compute_payoffs_hu_table_mode(tmp_path: Path) -> None:
    warmup()
    pairs = [(0, 1, 2, 3), (4, 5, 6, 7)]
    keys, eq = build_showdown_equity_table(pairs)
    table = tmp_path / "hu.npz"
    save_showdown_table(table, keys, eq)
    configure_payoff(payoff_hu_mode="table", hu_showdown_table_path=table)

    ranks, suits = "23456789TJQKA", "cdhs"

    def cid(c: int) -> str:
        return f"{ranks[c // 4]}{suits[c % 4]}"

    rng = random.Random(0)
    state = reset_hand(rng, table_size=2, stack_bb=8.0)
    seats = list(state.seats)
    state.holes[seats[0]] = (cid(0), cid(1))
    state.holes[seats[1]] = (cid(2), cid(3))
    state.folded = set()
    pay = compute_payoffs(state, samples=100, seed=0)
    assert len(pay) == 2
    assert abs(sum(pay.values())) < 1e-6
