"""Phase-3 router gating: DBR per-read confidence + beta-binomial LCB gate.

Pins the two EV unlocks on the canonical single-read spot (hero SB after a BTN fold, facing an
over-folding BB): an anonymous seat can no longer veto exploitation of a well-profiled opponent
(per-read confidence, Johanson & Bowling AISTATS'09), and small samples with large apparent
gains are blocked while solid reads fire (pessimistic LCB gate replacing the blunt point
margin). Measured separation that motivated the design (32o jam, BB folding vs jam):

    n=10 obs: point gain +0.51 bb (point gate FIRES) but 1σ-pessimistic gain −0.64 bb — the
              jam LOSES under a plausible misread; LCB blocks.
    n=80 obs: gain +1.26, pessimistic +0.77 — both gates fire.
    legacy min(): NEVER fires here — the anonymous (already folded!) BTN vetoes the table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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


def _fold_to_sb_jam(db, ident, n_obs):
    """BB folds to the SB jam after a BTN fold (cell BB|FA) n_obs times."""
    for _ in range(n_obs):
        record_observed_decision(db, ident, "BB", "FA", 8.0, pushed=False)


def _trash(q):
    return next(h for h in ("32o", "42o", "52o", "72o")
                if q.base_strategy(3, "SB", "F", h, 8.0) < 0.5)


# ---- per-read confidence (C1/C4) ------------------------------------------------------------
def test_anonymous_seat_does_not_veto(tmp_path) -> None:
    """BTN folded and is anonymous; BB is a well-read overfolder. Legacy min() still played
    base (the audited EV giveaway); per-read exploits the read that exists."""
    import aof_cfr.query as q
    q.load_cfr_base()
    db = _db(tmp_path)
    _fold_to_sb_jam(db, "bb_fish", 80)
    trash = _trash(q)
    opp = {"BTN": None, "BB": "bb_fish"}

    legacy = decide(db, 3, "SB", trash, "F", 8.0, opp, confidence_mode="min", gate_mode="point")
    assert legacy.source == "base"
    assert legacy.confidence == 0.0                    # the anonymous veto

    new = decide(db, 3, "SB", trash, "F", 8.0, opp)    # defaults: per_read/lcb
    assert new.source == "exploit"
    assert new.p_push > new.base_p
    assert new.confidence > 0.8                        # driven by the BB read
    assert new.mode == "per_read/lcb"


def test_unknown_table_is_exactly_base(tmp_path) -> None:
    import aof_cfr.query as q
    q.load_cfr_base()
    db = _db(tmp_path)
    d = decide(db, 3, "SB", "AA", "F", 8.0, {"BTN": None, "BB": None})
    assert d.source == "base"
    assert d.confidence == 0.0
    assert d.p_push == pytest.approx(d.base_p)


def test_per_read_weights_the_driving_read(tmp_path) -> None:
    """A thinly-profiled, base-consistent opponent must not drag the decision confidence
    toward its weak read when the gain is driven by the strong one."""
    import aof_cfr.query as q
    q.load_cfr_base()
    db = _db(tmp_path)
    _fold_to_sb_jam(db, "bb_fish", 80)
    for pushed in (True, False, True):                 # BTN: 3 obs, roughly base-like opens
        record_observed_decision(db, "btn_reg", "BTN", "-", 8.0, pushed=pushed)
    trash = _trash(q)
    d = decide(db, 3, "SB", trash, "F", 8.0, {"BTN": "btn_reg", "BB": "bb_fish"})
    weak_conf = 3 / (3 + 8.0)
    assert d.confidence > weak_conf + 0.3              # min() would have returned ~0.27
    assert d.source == "exploit"


# ---- LCB gate (C5) ---------------------------------------------------------------------------
def test_low_n_big_apparent_gain_blocked(tmp_path) -> None:
    """10 observed folds give a +0.5bb point gain — but the 1σ-pessimistic read says the jam
    LOSES. The point gate (at the new 0.10 margins) fires; the LCB gate blocks."""
    import aof_cfr.query as q
    q.load_cfr_base()
    db = _db(tmp_path)
    _fold_to_sb_jam(db, "bb_fish", 10)
    trash = _trash(q)
    opp = {"BTN": None, "BB": "bb_fish"}
    point = decide(db, 3, "SB", trash, "F", 8.0, opp, gate_mode="point",
                   min_confidence=0.10, margin_bb=0.10)
    lcb = decide(db, 3, "SB", trash, "F", 8.0, opp)
    assert point.source == "exploit"
    assert point.ev_gain_bb > 0.3
    assert lcb.source == "base"
    assert lcb.ev_gain_lcb_bb < 0.0                    # pessimistically the jam loses money


def test_high_n_read_fires_with_positive_lcb(tmp_path) -> None:
    import aof_cfr.query as q
    q.load_cfr_base()
    db = _db(tmp_path)
    _fold_to_sb_jam(db, "bb_fish", 80)
    trash = _trash(q)
    d = decide(db, 3, "SB", trash, "F", 8.0, {"BTN": None, "BB": "bb_fish"})
    assert d.source == "exploit"
    assert 0.10 <= d.ev_gain_lcb_bb <= d.ev_gain_bb    # pessimism never inflates the gain


def test_zero_n_everything_blocks(tmp_path) -> None:
    import aof_cfr.query as q
    q.load_cfr_base()
    db = _db(tmp_path)
    _fold_to_sb_jam(db, "bb_fish", 3)
    trash = _trash(q)
    d = decide(db, 3, "SB", trash, "F", 8.0, {"BTN": None, "BB": "bb_fish"})
    assert d.source == "base"                          # 3 obs cannot move a shrunk posterior


def test_hu_per_read_equals_single_opponent_confidence(tmp_path) -> None:
    import aof_cfr.query as q
    q.load_cfr_base()
    db = _db(tmp_path)
    for _ in range(60):
        record_observed_decision(db, "bb_fish", "BB", "A", 8.0, pushed=False)
    d_new = decide(db, 2, "SB", "T7o", "-", 8.0, {"BB": "bb_fish"})
    d_min = decide(db, 2, "SB", "T7o", "-", 8.0, {"BB": "bb_fish"},
                   confidence_mode="min", gate_mode="lcb")
    assert d_new.confidence == pytest.approx(d_min.confidence)   # HU: per_read == min


def test_4max_routes_with_new_gating(tmp_path) -> None:
    """End-to-end through route_4max (batch path): per-read + LCB fire on a strong read."""
    if not Path("data/aof/cfr_base/cfr_base_4p.npz").is_file():
        pytest.skip("4-max base not built")
    import aof_cfr.query as q
    q.load_cfr_base()
    db = _db(tmp_path)
    for _ in range(100):                               # BB folds to UTG jam after two folds
        record_observed_decision(db, "bb_fish", "BB", "AFF", 8.0, pushed=False)
        record_observed_decision(db, "sb_fish", "SB", "AF", 8.0, pushed=False)
    d = decide(db, 4, "UTG", "A5o", "-", 8.0,
               {"BTN": None, "SB": "sb_fish", "BB": "bb_fish"})
    assert d.mode == "per_read/lcb"
    assert d.confidence > 0.5                          # unknown BTN no longer vetoes
