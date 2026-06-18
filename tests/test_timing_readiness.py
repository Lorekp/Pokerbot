"""Timing-readiness gate in the NLHE opponent report (scripts/nlhe_opponent_report.py).

Flags when we have enough TIMED hands to wire the dt-based timing-tell model. Headline gate =
>= 1000 timed hands; also tracks showdown-reconciled coverage (the data the tell is actually fit on).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "nlhe_opponent_report", Path(__file__).resolve().parents[1] / "scripts" / "nlhe_opponent_report.py")
rep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rep)


def _hand(hid, *, timed=True, shown=None, aggr_by=None):
    """Build a minimal NLHE hand record. ``aggr_by`` = list of (user_id) that make a dt-timed raise."""
    actions = [{"street": "flop", "seat": 1, "user_id": uid, "type": "raise", "amount": 50,
                "dt": (1.2 if timed else None)} for uid in (aggr_by or [])]
    if not actions:  # ensure at least one action; dt present iff timed
        actions = [{"street": "preflop", "seat": 9, "user_id": 999, "type": "call", "amount": 2,
                    "dt": (0.8 if timed else None)}]
    return {"hand_id": hid, "table": {"game_mode": "nlhe"}, "board": ["Ah", "Kd", "7s"],
            "actions": actions,
            "shown": [{"seat": 1, "user_id": u, "name": f"v{u}", "cards": ["Qc", "Qd"]} for u in (shown or [])]}


def test_below_target_not_ready():
    recs = [_hand(i) for i in range(50)]
    rd = rep.compute_timing_readiness(recs)  # default target 1000
    assert rd["timed_hands"] == 50 and rd["target_timed"] == 1000
    assert rd["ready"] is False


def test_meets_1000_timed_hand_gate():
    recs = [_hand(i) for i in range(1000)]
    rd = rep.compute_timing_readiness(recs)
    assert rd["timed_hands"] == 1000 and rd["ready"] is True


def test_untimed_hands_dont_count():
    recs = [_hand(i, timed=False) for i in range(1200)]
    rd = rep.compute_timing_readiness(recs)
    assert rd["timed_hands"] == 0 and rd["ready"] is False


def test_reconciled_samples_need_a_shown_villain():
    # villain 1 raises with dt AND shows down -> one reconciled sample; an unshown raiser does not count.
    recs = [_hand(i, shown=[1], aggr_by=[1, 2]) for i in range(5)]
    rd = rep.compute_timing_readiness(recs)
    assert rd["reconciled_samples"] == 5          # only villain 1 (shown) across 5 hands
    assert rd["showdown_hands"] == 5
    rd10 = rep.compute_timing_readiness(recs, min_per_villain=5)
    assert rd10["villains_reconciled_ge_min"] == 1  # villain 1 has 5 >= 5


def test_aof_hands_excluded():
    recs = [{"hand_id": i, "table": {"game_mode": "aof"}, "actions": [{"type": "raise", "dt": 1.0}]}
            for i in range(2000)]
    rd = rep.compute_timing_readiness(recs)
    assert rd["hands"] == 0 and rd["timed_hands"] == 0 and rd["ready"] is False


def test_custom_target():
    recs = [_hand(i) for i in range(20)]
    assert rep.compute_timing_readiness(recs, target_timed=10)["ready"] is True
    assert rep.compute_timing_readiness(recs, target_timed=50)["ready"] is False
