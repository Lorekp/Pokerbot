"""Persistent opponent DB + Bayesian profiler + per-player exploitability (the exploit loop).

Uses an injected synthetic GTO base so the unit tests don't depend on the shipped artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from aof_cfr.buckets import LABELS_169
from aof_cfr.infoset import encode_infoset_key
from aof_cfr.opponent_db import AoFOpponentDB, identity_key
from aof_cfr.profiler import (
    base_policy_rows,
    estimate_policy_rows,
    infoset_confidence,
    per_player_exploitability,
    posterior_push,
)

# Synthetic base: call/push strong hands ~always, weak ~rarely. Deterministic, artifact-free.
_STRONG = {"AA", "KK", "QQ", "JJ", "TT", "99", "88", "AKs", "AKo", "AQs", "AQo", "AJs", "KQs"}


def _synthetic_base(num_players, position, history, hand, stack_bb):
    return 0.95 if hand in _STRONG else 0.15


def test_identity_and_db_roundtrip(tmp_path) -> None:
    assert identity_key("1656024", "Bob") == "1656024"
    assert identity_key(None, "Bob") == "name:Bob"
    assert identity_key(None, None) is None

    path = tmp_path / "profiles.json"
    db = AoFOpponentDB(path=path)
    k = encode_infoset_key("BB", "AA", "A", 8.0)
    db.record_action("u1", k, pushed=False)
    db.record_action("u1", k, pushed=True)
    assert db.cell("u1", k) == (1, 2)
    db.save()

    db2 = AoFOpponentDB(path=path)
    assert db2.cell("u1", k) == (1, 2)
    assert db2.n_obs("u1") == 2


def test_record_hand_idempotent(tmp_path) -> None:
    db = AoFOpponentDB(path=tmp_path / "p.json")
    k = encode_infoset_key("SB", "72o", "-", 8.0)
    obs = [("u1", k, True), ("u2", k, False)]
    assert db.record_hand("hand-1", obs) == 2
    assert db.record_hand("hand-1", obs) == 0  # idempotent
    assert db.cell("u1", k) == (1, 1)
    assert db.cell("u2", k) == (0, 1)


def test_posterior_shifts_toward_observations() -> None:
    base_p = 0.2
    s = 8.0
    # No data -> base.
    assert posterior_push(0, 0, base_p, s) == pytest.approx(base_p)
    # Heavy evidence of pushing -> posterior moves up, confidence rises.
    p = posterior_push(push=90, total=100, base_p=base_p, prior_strength=s)
    assert p > 0.8
    assert infoset_confidence(100, s) > infoset_confidence(5, s)


@pytest.mark.slow
@pytest.mark.skipif(
    not __import__("pathlib").Path("data/aof/cfr_base/cfr_base_2p.npz").is_file(),
    reason="HU base not built",
)
def test_per_player_exploitability_detects_leak(tmp_path) -> None:
    """Against the REAL near-equilibrium base, a BB that folds strong hands vs a jam spills
    large EV, while an unknown opponent (collapses to the ~optimal base) spills ~0."""
    import aof_cfr.query as q

    q.load_cfr_base()  # default base_push_fn = q.base_strategy (near-Nash HU base)
    n, stack_bb, opp_seat = 2, 8.0, 1  # HU; seat 1 = BB

    leaky = AoFOpponentDB(path=tmp_path / "leaky.json")
    for hand in _STRONG:  # BB folding premiums vs a jam is a glaring leak
        key = encode_infoset_key("BB", hand, "A", stack_bb)
        for _ in range(60):
            leaky.record_action("leak", key, pushed=False)

    leak = per_player_exploitability(leaky, "leak", n, opp_seat, stack_bb, samples=4000, seed=1)
    clean = AoFOpponentDB(path=tmp_path / "clean.json")
    base = per_player_exploitability(clean, "unknown", n, opp_seat, stack_bb, samples=4000, seed=1)

    # Unknown -> near-optimal base -> tiny spill; leak (folding ~13 premium buckets vs a
    # jam) spills ~0.15 bb/hand (= ~15 bb/100), clearly above the near-zero base.
    assert base["opp_spilled_bb"] < 0.05
    assert leak["opp_spilled_bb"] > 0.10
    assert leak["opp_spilled_bb"] > base["opp_spilled_bb"] + 0.06
