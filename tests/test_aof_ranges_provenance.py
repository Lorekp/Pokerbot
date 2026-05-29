"""Range table provenance (HU solver vs hand-tuned)."""

from __future__ import annotations

import pytest

from aof_ranges import lookup_call, lookup_shove, range_provenance


def test_hu_call_solver_when_json_present() -> None:
    prov = range_provenance("BB", "SB", 8.0, table_size=2, shove=False)
    if prov == "solver_hu_dcfr":
        pct, rng, p2 = lookup_call("SB", "BB", 8.0, table_size=2)
        assert p2 == "solver_hu_dcfr"
        assert rng
        assert 0.0 < pct < 1.0
    else:
        pytest.skip("data/aof/hu_pushfold_8bb.json not present")


def test_four_max_hand_tuned() -> None:
    assert (
        range_provenance("BB", "BTN", 8.0, table_size=4, shove=False)
        == "hand_tuned"
    )
    _, p = lookup_shove("BTN", 8.0, table_size=4)
    assert p == "hand_tuned"
