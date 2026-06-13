"""Phase-5 opponent-model upgrades: showdown range inference + hierarchical depth pooling.

AoF shows down constantly; collector v2 records the reveals and (since Phase 5) ingestion
ALSO stores each revealed decision under its exact per-bucket infoset key. The estimator's
two-level posterior lets those exact observations override the range-width reconstruction —
calibrating the range SHAPE (e.g. a villain who jams weak aces the width-shift would have
excluded). Depth pooling shrinks each depth cell toward the player's all-depth (seat, hist)
pool, undoing the 7-bucket fragmentation. Regressions: no reveals + single depth ==
byte-identical estimator; pooling collapses at zero cross-depth data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aof_cfr.infoset import aggregate_infoset_key, encode_infoset_key
from aof_cfr.live_exploit import observations_from_record, record_observed_decision
from aof_cfr.opponent_db import AoFOpponentDB

_BASE = Path("data/aof/cfr_base/cfr_base_3p.npz")
pytestmark = pytest.mark.skipif(not _BASE.is_file(), reason="3-max base not built")


def _db(tmp_path):
    db = AoFOpponentDB(path=tmp_path / "db.json")
    db.counts.clear(); db.seen_hands.clear()
    return db


def _record_with_reveal(hand_id: str, *, bb: float = 10.0) -> dict:
    """BTN jams and shows A2o at showdown; SB folds; BB calls and shows KK."""
    return {
        "schema_version": 2,
        "hand_id": hand_id,
        "table": {"sb": bb / 2, "bb": bb, "dealer_seat_id": 0, "n_players": 3,
                  "game_mode": "aof"},
        "hero": {"seat": -1, "name": "hero", "user_id": None},
        "players": [{"seat": s, "name": f"v{s}", "user_id": None, "stack": 80.0,
                     "playing": True} for s in (0, 1, 2)],
        "actions": [{"street": "preflop", "seat": 0, "name": "v0", "type": "all_in", "amount": 80.0},
                    {"street": "preflop", "seat": 1, "name": "v1", "type": "fold", "amount": 0.0},
                    {"street": "preflop", "seat": 2, "name": "v2", "type": "all_in", "amount": 80.0}],
        "shown": [{"seat": 0, "user_id": None, "name": "v0", "cards": ["Ah", "2s"]},
                  {"seat": 2, "user_id": None, "name": "v2", "cards": ["Kh", "Kd"]}],
        "winners": [],
    }


# ---- ingestion -------------------------------------------------------------------------------
def test_reveals_ingested_under_exact_and_aggregate_keys(tmp_path) -> None:
    db = _db(tmp_path)
    obs = observations_from_record(_record_with_reveal("h1"))
    assert obs is not None
    keys = {k for _i, k, _p in obs}
    assert aggregate_infoset_key("BTN", "-", 8.0) in keys
    assert encode_infoset_key("BTN", "A2o", "-", 8.0) in keys      # the reveal
    assert encode_infoset_key("BB", "KK", "AF", 8.0) in keys
    assert not any("|*|" not in k and "SB" in k for k in keys)      # SB never showed


def test_n_obs_not_inflated_by_reveals(tmp_path) -> None:
    """A revealed decision is stored twice (aggregate + exact) but counts ONCE."""
    from aof_cfr.live_exploit import ingest_record
    db = _db(tmp_path)
    n = ingest_record(db, _record_with_reveal("h1"))
    assert n == 5                                                   # 3 aggregate + 2 exact rows
    assert db.n_obs("name:v0") == 1                                 # ...but ONE decision
    assert db.n_obs("name:v2") == 1


# ---- read side: exact-bucket override ---------------------------------------------------------
def test_exact_bucket_posterior_overrides_shift(tmp_path) -> None:
    """A villain whose reveals show TRASH jams gets those buckets raised far above anything
    the width-shift could imply (72o: base ~0, shift would keep it ~0), without touching
    unobserved buckets."""
    import aof_cfr.query as q
    q.load_cfr_base()
    from aof_cfr.profiler import estimate_policy_rows_shift

    db = _db(tmp_path)
    for i in range(40):                                  # aggregate: roughly base-rate opens
        record_observed_decision(db, "v", "BTN", "-", 8.0, pushed=(i % 5 < 2))
    for _ in range(12):                                  # showdowns: 12 revealed 72o jams
        db.record_action("v", encode_infoset_key("BTN", "72o", "-", 8.0), pushed=True)

    rows = estimate_policy_rows_shift(db, "v", 3, 8.0)
    base_p = q.base_strategy(3, "BTN", "-", "72o", 8.0)
    assert base_p < 0.5                                  # base does not open 72o here
    est_72o = float(rows[encode_infoset_key("BTN", "72o", "-", 8.0)][1])
    assert est_72o > 0.55                                # 12/12 revealed jams dominate
    assert est_72o > base_p + 0.3
    # an unobserved bucket is untouched relative to the pure shift
    db2 = _db(tmp_path / "ctrl")
    for i in range(40):
        record_observed_decision(db2, "v", "BTN", "-", 8.0, pushed=(i % 5 < 2))
    rows_ctrl = estimate_policy_rows_shift(db2, "v", 3, 8.0)
    k_aa = encode_infoset_key("BTN", "AA", "-", 8.0)
    assert float(rows[k_aa][1]) == pytest.approx(float(rows_ctrl[k_aa][1]))


def test_no_reveals_byte_identical(tmp_path) -> None:
    """Aggregate-only profile at a single depth: the upgraded estimator must reproduce the
    original shift exactly (pooling + override both collapse)."""
    import aof_cfr.query as q
    q.load_cfr_base()
    from aof_cfr.profiler import (
        _base_arrays_seathist, _query_base_push, _range_width_shift, posterior_push,
        _COMBO_W, _COMBO_W_SUM, estimate_policy_rows_shift)

    db = _db(tmp_path)
    for i in range(30):
        record_observed_decision(db, "v", "SB", "A", 8.0, pushed=(i % 4 == 0))
    rows = estimate_policy_rows_shift(db, "v", 3, 8.0)
    # original formula, recomputed by hand
    base_p = _base_arrays_seathist(3, 8.0, _query_base_push)[("SB", "A")]
    base_agg = float((_COMBO_W * base_p).sum() / _COMBO_W_SUM)
    expect = _range_width_shift(base_p, posterior_push(8, 30, base_agg, 8.0))
    got = np.array([float(rows[encode_infoset_key("SB", lab, "A", 8.0)][1])
                    for lab in __import__("aof_cfr.buckets", fromlist=["LABELS_169"]).LABELS_169])
    np.testing.assert_allclose(got, expect, atol=1e-12)


# ---- depth pooling -----------------------------------------------------------------------------
def test_depth_pooling_informs_neighbor_depths(tmp_path) -> None:
    """80 overfold observations at d8 must move the d12 read (shrunk, not ignored)."""
    import aof_cfr.query as q
    q.load_cfr_base()
    from aof_cfr.profiler import _COMBO_W, _COMBO_W_SUM, estimate_policy_rows_shift

    db = _db(tmp_path)
    for _ in range(80):
        record_observed_decision(db, "v", "BB", "FA", 8.0, pushed=False)   # observed at d8

    def agg_at(depth):
        rows = estimate_policy_rows_shift(db, "v", 3, depth)
        from aof_cfr.buckets import LABELS_169
        est = np.array([float(rows[encode_infoset_key("BB", lab, "FA", depth)][1])
                        for lab in LABELS_169])
        return float((est * _COMBO_W).sum() / _COMBO_W_SUM)

    from aof_cfr.profiler import _base_arrays_seathist, _query_base_push

    def base_agg(depth):
        b = _base_arrays_seathist(3, depth, _query_base_push)[("BB", "FA")]
        return float((b * _COMBO_W).sum() / _COMBO_W_SUM)

    a12, b12 = agg_at(12.0), base_agg(12.0)
    a8, b8 = agg_at(8.0), base_agg(8.0)
    assert a12 < b12 - 0.05             # the d8 signal leaks into d12 (shrunk toward pool)
    # the observed depth retains LESS of its base than the pooled depth retains of its own:
    # cell shrink s/(n+s)=8/88 < pool shrink s2/(n+s2)=10/90 (bases differ across depths, so
    # compare retained FRACTIONS, not absolute rates).
    assert a8 / b8 < a12 / b12


def test_adaptive_pooling_and_collapse() -> None:
    from aof_cfr.adaptive import AdaptiveOpponentModel, PopulationPrior
    m = AdaptiveOpponentModel(prior=PopulationPrior(flat_rate=0.30))
    for _ in range(40):
        m.observe("v", 3, "BB", "FA", 8.0, pushed=False)
    e8 = m.estimate("v", 3, "BB", "FA", 8.0)
    e12 = m.estimate("v", 3, "BB", "FA", 12.0)
    assert e8["rate"] < 0.10                       # direct read
    assert e12["rate"] < 0.30 - 0.05               # pooled read moves off the prior...
    assert e12["rate"] > e8["rate"]                # ...but less than the observed depth
    assert 0 < e12["confidence"] < e8["confidence"]
    # no data anywhere -> exactly the prior with zero confidence (collapse regression)
    e_none = m.estimate("ghost", 3, "BB", "FA", 8.0)
    assert e_none["rate"] == pytest.approx(0.30) and e_none["confidence"] == 0.0
