"""Side-pot unsupported guard (Priority 13)."""

from __future__ import annotations

import pytest

from aof_game_config import AoFGameConfig
from aof_multiway import multiway_call_terms


def test_equal_stacks_unchanged() -> None:
    terms = multiway_call_terms("BB", 8.0, villain_positions=("UTG",), table_size=4)
    assert terms.total_pot_after_call == pytest.approx(16.5, abs=0.01)


def test_unequal_stacks_raises() -> None:
    with pytest.raises(NotImplementedError, match="side-pot"):
        multiway_call_terms(
            "BB",
            8.0,
            villain_positions=("UTG",),
            table_size=4,
            stacks_by_seat={"BB": 8.0, "UTG": 6.0, "BTN": 8.0, "SB": 8.0},
        )
