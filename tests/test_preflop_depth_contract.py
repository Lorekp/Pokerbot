from __future__ import annotations

import pytest

from pokerbot.eval.snapshot_dir import preflop_serve_band, validate_preflop_depth_contract


EDGES = [25.0, 50.0, 90.0, 140.0, 250.0, 400.0]


def _stamp(*, depth=True, status="certified", edges=EDGES, gates=None):
    stamp = {
        "abstraction": {
            "depth_bucketed": depth,
            "depth_edges": list(EDGES) if depth else [],
        }
    }
    if status is not None:
        stamp["preflop_depth_contract"] = {
            "status": status,
            "min_bb": 15,
            "max_bb": 500,
            "depth_edges": list(edges),
            "gates": gates or {
                "coverage": True,
                "strategy_shape": True,
                "paired_strength": True,
                "exploitability": True,
            },
        }
    return stamp


def test_certified_depth_contract_returns_serve_band():
    assert validate_preflop_depth_contract(_stamp()) == (15.0, 500.0)


def test_fixed_depth_legacy_artifact_has_no_widening_contract():
    assert validate_preflop_depth_contract(
        {"abstraction": {"depth_bucketed": False}}
    ) is None


def test_policy_band_comes_from_artifact_and_never_becomes_unbounded():
    class Snapshot:
        pass

    fixed = Snapshot()
    fixed.serve_config = {"abstraction": {"depth_bucketed": False}}
    assert preflop_serve_band(fixed, (80.0, 130.0)) == (80.0, 130.0)

    depth = Snapshot()
    depth.serve_config = _stamp()
    assert preflop_serve_band(depth, (80.0, 130.0)) == (15.0, 500.0)


@pytest.mark.parametrize(
    "stamp, match",
    [
        (_stamp(status=None), "no preflop_depth_contract"),
        (_stamp(status="candidate"), "not certified"),
        (_stamp(edges=[25, 50, 100]), "edges"),
        (_stamp(gates={"coverage": True, "strategy_shape": False}), "promotion gates"),
        (_stamp(depth=False), "non-depth-bucketed"),
    ],
)
def test_invalid_or_uncertified_contract_fails_closed(stamp, match):
    with pytest.raises(RuntimeError, match=match):
        validate_preflop_depth_contract(stamp)
