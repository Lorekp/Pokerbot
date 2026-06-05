"""Regression: the HU showdown canonical key must be invariant to within-pair card order.

The table is built from id-ascending pairs (enumerate_all_disjoint_pairs); lookups pass cards in
deal order. Before the fix, ~half of deal-order lookups produced keys absent from the table
(KeyError in aof_rl payoff). The key now normalizes within-pair order, so any order of the same
4 cards hits the table and returns the correct equity.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from aof_equity_showdown import DEFAULT_TABLE_PATH, canonical_showdown_key

_HAS_TABLE = Path(DEFAULT_TABLE_PATH).is_file()
_RANKS, _SUITS = "23456789TJQKA", "cdhs"
_DECK = [r + s for r in _RANKS for s in _SUITS]


def test_key_invariant_to_within_pair_order() -> None:
    from aof_equity_grids import _card_str_to_id as cid

    rng = random.Random(0)
    for _ in range(500):
        cs = rng.sample(_DECK, 4)
        h0, h1, v0, v1 = (cid(c) for c in cs)
        base = canonical_showdown_key(h0, h1, v0, v1)
        assert canonical_showdown_key(h1, h0, v0, v1) == base  # swap hero cards
        assert canonical_showdown_key(h0, h1, v1, v0) == base  # swap villain cards
        assert canonical_showdown_key(h1, h0, v1, v0) == base


@pytest.mark.skipif(not _HAS_TABLE, reason="HU showdown table not built")
def test_table_lookup_matches_exact_any_order() -> None:
    from aof_equity import warmup
    from aof_equity_showdown import clear_showdown_table_cache, equity_hu_showdown_exact

    warmup()
    clear_showdown_table_cache()
    rng = random.Random(1)
    for _ in range(60):  # exact_enum is the oracle; keep the sample modest
        cs = rng.sample(_DECK, 4)
        hero, vill = (cs[0], cs[1]), (cs[2], cs[3])  # arbitrary (deal) order
        t = equity_hu_showdown_exact(hero, vill, mode="table")
        e = equity_hu_showdown_exact(hero, vill, mode="exact_enum")
        assert abs(t - e) < 1e-6


@pytest.mark.skipif(not _HAS_TABLE, reason="HU showdown table not built")
def test_env_hands_no_keyerror() -> None:
    from aof_rl.env import ACTION_ALL_IN, ACTION_FOLD, reset_hand, step_hand

    rng = random.Random(2)
    for _ in range(150):
        for tbl in (2, 3, 4):
            st = reset_hand(rng, table_size=tbl, stack_bb=8.0)
            g = 0
            while not st.terminal and g < 32:
                g += 1
                st = step_hand(st, ACTION_ALL_IN if rng.random() < 0.6 else ACTION_FOLD)
            assert st.terminal
