"""Side-pot expected structure tests (design-only; advisor still guarded)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aof_multiway import multiway_call_terms


@dataclass(frozen=True)
class PotLayer:
    amount: float
    eligible_seats: tuple[str, ...]


def test_side_pot_example_expected_structure() -> None:
    """
    UTG 5bb all-in, BTN 8bb all-in, SB folded (0.5 posted), BB calls 8bb.
    """
    main = PotLayer(amount=15.0, eligible_seats=("UTG", "BTN", "BB"))
    side = PotLayer(amount=6.0, eligible_seats=("BTN", "BB"))
    assert main.amount == 5.0 * 3
    assert side.amount == 3.0 * 2
    assert "UTG" in main.eligible_seats
    assert "UTG" not in side.eligible_seats


def test_unequal_stacks_still_guarded() -> None:
    with pytest.raises(NotImplementedError, match="side-pot"):
        multiway_call_terms(
            "BB",
            8.0,
            villain_positions=("UTG",),
            table_size=4,
            stacks_by_seat={"BB": 8.0, "UTG": 6.0, "BTN": 8.0, "SB": 8.0},
        )
