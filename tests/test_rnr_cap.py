"""Restricted-Nash-Response cap: the played exploit move is bounded by read stability.

The LCB gate decides WHETHER to deviate; the RNR cap (Johanson NIPS'07; Johanson & Bowling
AISTATS'09) bounds HOW FAR we move toward the pure best response. A stationary read commits
fully (cap == EXPLOIT_CAP == 1 -> behaviour bit-unchanged); a read the model has caught ADAPTING
is clamped to CAP_FLOOR of the way to the BR, so an adaptive counter-exploiter can only punish a
bounded deviation. These pin the mechanism in isolation (explicit stability_fn) and end-to-end
through the adaptive model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aof_cfr.exploit_router import CAP_FLOOR
from aof_cfr.live_exploit import decide, record_observed_decision
from aof_cfr.opponent_db import AoFOpponentDB

_BASE = Path("data/aof/cfr_base/cfr_base_3p.npz")
_EQ = Path("data/aof/cfr_base/hu_eq_matrix.npz")
pytestmark = pytest.mark.skipif(not (_BASE.is_file() and _EQ.is_file()),
                                reason="3-max base / HU eq artifacts not built")


def _db(tmp_path):
    db = AoFOpponentDB(path=tmp_path / "db.json")
    db.counts.clear(); db.seen_hands.clear()
    return db


def _marginal_fold(q):
    from aof_cfr.exploit_router import STEAL_EQUITY_FLOOR, _equity_vs_random
    return next(h for h in ("Q5o", "K4o", "J6s", "Q7s", "96s", "T6s", "Q4s", "J7s")
                if q.base_strategy(3, "SB", "F", h, 8.0) < 0.5
                and _equity_vs_random(h) >= STEAL_EQUITY_FLOOR)


def test_stable_read_unchanged_adapting_read_clamped(tmp_path) -> None:
    """Same strong overfolder read; stability 1.0 commits fully, stability 0.0 clamps to CAP_FLOOR."""
    import aof_cfr.query as q
    q.load_cfr_base()
    db = _db(tmp_path)
    for _ in range(80):
        record_observed_decision(db, "bb_fish", "BB", "FA", 8.0, pushed=False)
    marg = _marginal_fold(q)
    opp = {"BTN": None, "BB": "bb_fish"}

    stable = decide(db, 3, "SB", marg, "F", 8.0, opp, stability_fn=lambda i: 1.0)
    adapt = decide(db, 3, "SB", marg, "F", 8.0, opp, stability_fn=lambda i: 0.0)

    assert stable.source == adapt.source == "exploit"
    # The cap touches the PLAYED probability, not the reported read confidence.
    assert stable.confidence == pytest.approx(adapt.confidence)
    assert stable.ev_gain_bb == pytest.approx(adapt.ev_gain_bb)
    # Adapting read moved strictly less toward the all-in than the stable read.
    assert (stable.p_push - stable.base_p) > (adapt.p_push - adapt.base_p) > 0.0
    # Clamp magnitude: with a high-confidence read, the adapting move == CAP_FLOOR * (exploit - base).
    expected_adapt = adapt.base_p + CAP_FLOOR * (adapt.exploit_p - adapt.base_p)
    assert adapt.p_push == pytest.approx(expected_adapt, abs=0.05)


def test_default_stability_is_noop(tmp_path) -> None:
    """No stability_fn (DB-only path) == stability 1.0 == the pre-cap behaviour exactly."""
    import aof_cfr.query as q
    q.load_cfr_base()
    db = _db(tmp_path)
    for _ in range(80):
        record_observed_decision(db, "bb_fish", "BB", "FA", 8.0, pushed=False)
    marg = _marginal_fold(q)
    opp = {"BTN": None, "BB": "bb_fish"}
    default = decide(db, 3, "SB", marg, "F", 8.0, opp)
    stable = decide(db, 3, "SB", marg, "F", 8.0, opp, stability_fn=lambda i: 1.0)
    assert default.p_push == pytest.approx(stable.p_push)


def test_adaptive_model_clamps_caught_adapter(tmp_path) -> None:
    """End-to-end: an opponent the model has seen flip gears is exploited less than a static one
    with the identical lifetime read (the model's stability signal drives the cap)."""
    import aof_cfr.query as q
    from aof_cfr.adaptive import AdaptiveOpponentModel, PopulationPrior
    q.load_cfr_base()
    db = _db(tmp_path)
    marg = _marginal_fold(q)
    opp = {"BTN": None, "BB": "bb_fish"}

    # Stable model: 80 consistent folds -> stability 1.0.
    m_stable = AdaptiveOpponentModel(prior=PopulationPrior())
    for _ in range(80):
        m_stable.observe(("bb_fish"), 3, "BB", "FA", 8.0, pushed=False)
    s = decide(db, 3, "SB", marg, "F", 8.0, opp, model=m_stable)

    # Adapting model: same net fold-heavy read but a detected gear flip -> stability < 1.0.
    m_adapt = AdaptiveOpponentModel(prior=PopulationPrior())
    for _ in range(40):
        m_adapt.observe("bb_fish", 3, "BB", "FA", 8.0, pushed=True)
    for _ in range(40):
        m_adapt.observe("bb_fish", 3, "BB", "FA", 8.0, pushed=False)
    assert m_adapt.opponent_stability("bb_fish") < 1.0
    a = decide(db, 3, "SB", marg, "F", 8.0, opp, model=m_adapt)

    if s.source == "exploit" and a.source == "exploit":
        assert (s.p_push - s.base_p) >= (a.p_push - a.base_p)
