"""BR position probe: payoffs must differ across seats under fixed callers."""

from __future__ import annotations

from pathlib import Path

import pytest

from aof_cfr.trio_payoff_tensor import DEFAULT_TENSOR

pytest.importorskip("pokerkit")

from scripts.phase_a_br_position_probe import PROBE_HANDS_TIGHT, run_probe


@pytest.mark.skipif(not DEFAULT_TENSOR.is_file(), reason="trusted tensor not built")
def test_payoffs_not_position_blind_under_tight_callers() -> None:
    out = run_probe(tensor_path=DEFAULT_TENSOR, profile="tight")
    assert not out["position_blind"]
    assert out["max_spread_across_seats"] > 0.2
    for hand in PROBE_HANDS_TIGHT:
        vals = [out["br_table"][seat][hand] for seat in ("BTN", "SB", "BB")]
        assert max(vals) - min(vals) > 0.2


@pytest.mark.skipif(not DEFAULT_TENSOR.is_file(), reason="trusted tensor not built")
def test_72o_does_not_br_jam_into_100pct_call_field() -> None:
    from scripts.phase_a_br_position_probe import run_probe

    out = run_probe(tensor_path=DEFAULT_TENSOR, profile="field100")
    assert out["br_table"]["BTN"]["72o"] < 0.5
    assert not out["trash_jam_into_callers"]
