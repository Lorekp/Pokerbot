"""Hybrid data harness: synthetic round-trip — recover the FITTABLE leak signs, flag the rest."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_ART = Path("data/aof/cfr_base/cfr_base_3p.npz")
pytestmark = pytest.mark.skipif(not _ART.is_file(), reason="3-max base not built")


def test_fit_recovers_vs_jam_sign_and_flags_unidentifiable(tmp_path) -> None:
    import aof_cfr.query as q
    q.load_cfr_base()
    from aof_cfr.live_exploit import ingest_jsonl
    from aof_cfr.opponent_db import AoFOpponentDB
    from scripts.fit_tiers_from_hands import fit_leakprofile, measure_player, simulate_jsonl

    # BTN never faces a jam in 3-max (acts first), so the vs-jam leakers must sit at SB/BB:
    # seat0=BTN=professional, seat1=SB=intermediate (over-folder), seat2=BB=beginner (over-caller).
    path = tmp_path / "syn.jsonl"
    simulate_jsonl(path, ["professional", "intermediate", "beginner"], 3, 8.0, 4000, seed=1)
    db = AoFOpponentDB(path=tmp_path / "db.json")
    db.counts.clear(); db.seen_hands.clear()
    n = ingest_jsonl(db, path)
    assert n > 0

    over_folder = fit_leakprofile(measure_player(db, "name:p1_intermediate", 3))
    over_caller = fit_leakprofile(measure_player(db, "name:p2_beginner", 3))

    # Sign recovery: the over-folder shows overfold>0, the over-caller shows overcall>0.
    assert over_folder["fitted_frequency_leaks"]["overfold_vs_jam"] > over_folder["fitted_frequency_leaks"]["overcall_vs_jam"]
    assert over_caller["fitted_frequency_leaks"]["overcall_vs_jam"] > over_caller["fitted_frequency_leaks"]["overfold_vs_jam"]
    assert "over-fold" in over_folder["archetype_guess"]
    assert "calling-station" in over_caller["archetype_guess"]

    # Honesty: per-hand leaks are explicitly flagged as unfittable from aggregate (no hole cards).
    assert "ace_bias" in over_caller["unidentifiable_from_aggregate"]
    assert "per_hand_range_shape" in over_caller["unidentifiable_from_aggregate"]
