"""HU legal-BR marginals + route_hu: consistency with the base, and exploitation of leaks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aof_cfr.buckets import LABELS_169
from aof_cfr.infoset import encode_infoset_key
from aof_cfr.opponent_db import AoFOpponentDB

_EQ = Path("data/aof/cfr_base/hu_eq_matrix.npz")
_BASE = Path("data/aof/cfr_base/cfr_base_2p.npz")
_HAS = _EQ.is_file() and _BASE.is_file()

pytestmark = pytest.mark.skipif(not _HAS, reason="HU base/eq artifacts not built")


def _base_arrays(stack_bb: float):
    import aof_cfr.query as q
    from aof_cfr.exploit_router import _seat_array_hu
    from aof_cfr.profiler import base_policy_rows

    q.load_cfr_base()
    rows = base_policy_rows(2, stack_bb)
    return _seat_array_hu(rows, "SB", stack_bb), _seat_array_hu(rows, "BB", stack_bb)


def test_chip_model_constants() -> None:
    from aof_cfr.hu_br_legal import _legal_marginals_hu, _pot_called, load_hu_eq

    assert _pot_called(8.0) == pytest.approx(17.5)
    eq = load_hu_eq()
    assert eq.shape == (169, 169)
    assert eq[0, 0] == pytest.approx(0.5)  # mirror matchup is 50%
    sb, bb = _base_arrays(8.0)
    m = _legal_marginals_hu(sb, bb, 8.0, eq)
    assert m["sb"][1] == pytest.approx(-0.5)   # SB fold
    assert m["bb_a"][1] == pytest.approx(-1.0)  # BB fold facing a jam


def test_base_is_near_fixed_point() -> None:
    """BR to the shipped base must leave < the 0.30 bb indifference margin at every hand."""
    from aof_cfr.hu_br_legal import _legal_marginals_hu, load_hu_eq

    eq = load_hu_eq()
    for stack_bb in (4.0, 8.0, 20.0):
        sb, bb = _base_arrays(stack_bb)
        m = _legal_marginals_hu(sb, bb, stack_bb, eq)
        for key, p in (("sb", sb), ("bb_a", bb)):
            ep, ef, _w = m[key]
            on_policy = (1.0 - p) * ef + p * ep
            imp = np.maximum(ef, ep) - on_policy  # bb left on the table per hand
            assert float(imp.max()) < 0.30, f"{key}@{stack_bb}bb max imp {imp.max():.4f}"


def test_seat_values_dead_blind_relation() -> None:
    from aof_cfr.hu_br_legal import load_hu_eq, seat_values_hu

    eq = load_hu_eq()
    sb, bb = _base_arrays(8.0)
    sb_val, bb_val = seat_values_hu(sb, bb, 8.0, eq)
    # Sum equals dead money 1.5 * P(both jam) (the +1.5 convention), bounded by [0, 1.5].
    assert 0.0 <= (sb_val + bb_val) <= 1.5 + 1e-9


def test_route_hu_unknown_opponent_is_base() -> None:
    from aof_cfr.exploit_router import route_hu

    db = AoFOpponentDB(path=Path("data/aof_player_profiles/_test_route_hu_unknown.json"))
    db.counts.clear()
    d = route_hu(db, {"BB": None}, "SB", "AA", "-", 8.0)
    assert d.source == "base"
    assert d.confidence == 0.0
    assert d.p_push == pytest.approx(d.base_p)


def test_route_hu_exploits_overfolding_bb() -> None:
    """A BB that folds every hand facing a jam -> SB should jam wider (exploit a trash hand)."""
    from aof_cfr.exploit_router import route_hu

    db = AoFOpponentDB(path=Path("data/aof_player_profiles/_test_route_hu_overfold.json"))
    db.counts.clear()
    ident = "villain"
    for lab in LABELS_169:  # BB folds (pushed=False) at every bucket facing a jam
        key = encode_infoset_key("BB", lab, "A", 8.0)
        for _ in range(40):
            db.record_action(ident, key, pushed=False)

    d = route_hu(db, {"BB": ident}, "SB", "32o", "-", 8.0)
    assert d.source == "exploit"
    assert d.exploit_p == pytest.approx(1.0)   # jam any two vs a pure folder
    assert d.p_push > d.base_p
    assert d.ev_gain_bb >= 0.30
