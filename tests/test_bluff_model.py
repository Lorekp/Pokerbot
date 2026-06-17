"""Phase 2 (G2): showdown-conditioned bluff-ratio model — storage decay, hierarchical read,
mixer consumption direction, schema v5 persistence round-trip, and record-replay derivation."""

from __future__ import annotations

import os
import random

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

import numpy as np

from pokerbot.opponent.bayes_exploit import BLUFF_PRIOR, bluff_read
from pokerbot.opponent.stats import OpponentStats


def test_bluff_decay_and_add_math():
    s = OpponentStats()
    s.bluff_decay_and_add("flop|dry|unopened", True, 10)
    assert s.ctx_bluff["flop|dry|unopened"] == [1.0, 0.0]
    # Same hand index ⇒ Δ=0 ⇒ no decay between the two calls.
    s.bluff_decay_and_add("flop|dry|unopened", False, 10, weight=2.0)
    assert s.ctx_bluff["flop|dry|unopened"] == [1.0, 2.0]
    # 35 hands later: γ^35 ≈ 0.493 — recency halves stale evidence.
    s.bluff_decay_and_add("flop|dry|unopened", True, 45)
    b, v = s.ctx_bluff["flop|dry|unopened"]
    g = 0.98 ** 35
    assert abs(b - (g * 1.0 + 1.0)) < 1e-9
    assert abs(v - g * 2.0) < 1e-9


def test_bluff_read_hierarchy_and_none():
    s = OpponentStats()
    assert bluff_read(s, "flop|dry|unopened") is None  # no reveal data at all ⇒ None

    # Only OTHER-street data: the sparse exact context borrows through street→global→prior.
    s.ctx_bluff["river|dry|unopened"] = [8.0, 2.0]     # bluffy river
    p_sparse, c_sparse = bluff_read(s, "flop|dry|unopened")
    prior_mean = BLUFF_PRIOR[0] / sum(BLUFF_PRIOR)
    assert p_sparse > prior_mean                        # global evidence pulls above the prior
    assert 0.0 < c_sparse < 0.5                         # quarter-weighted ⇒ low confidence

    # Exact-context data dominates the coarser levels.
    s.ctx_bluff["flop|dry|unopened"] = [0.0, 12.0]      # value-heavy at THIS context
    p_exact, c_exact = bluff_read(s, "flop|dry|unopened")
    assert p_exact < prior_mean < p_sparse
    assert c_exact > c_sparse


def _facing_state_and_base(seed: int):
    """Hero (seat 0) faces a genuine preflop raise; returns (state, uniform-base probs)."""
    from pokerbot.engine.actions import Action, ActionType
    from pokerbot.engine.game import GameState

    st = GameState.new_hand(button_seat=0, rng=random.Random(seed))
    st.deal_hole_cards()
    for _ in range(6):
        if st.action_seat == 0:
            break
        legal = st.legal_actions()
        raise_a = next((a for a in legal if a.action_type == ActionType.RAISE), None)
        st.apply_action(raise_a if (st.action_seat == 3 and raise_a) else Action(ActionType.FOLD))
        st.advance_to_next_actor()
    assert st.action_seat == 0
    base = np.full(11, 1.0 / 11)
    return st, base


def _over_aggressive_profile():
    from pokerbot.opponent.population_prior import context_prior

    prof = OpponentStats()
    prof.hands = 200
    for stn in ("flop", "turn", "river"):
        for tex in ("dry", "wet", "paired"):
            for facing in ("unopened", "facing_bet"):
                ck = f"{stn}|{tex}|{facing}"
                for h in range(1, 50):
                    prof.decay_and_add(ck, [0, 0, 1, 1], context_prior(ck), h)
                prof.add_ctx(ck, [0, 0, 24, 24])
    # Preflop too (the rig's facing spot is preflop).
    for n in range(4):
        ck = f"preflop|-|{n}"
        for h in range(1, 50):
            prof.decay_and_add(ck, [0, 0, 1, 1], context_prior(ck), h)
        prof.add_ctx(ck, [0, 0, 24, 24])
    return prof


def _set_all_bluff(prof: OpponentStats, n_bluff: float, n_value: float) -> None:
    prof.ctx_bluff = {}
    for stn in ("preflop", "flop", "turn", "river"):
        for tex in ("-",) if stn == "preflop" else ("dry", "wet", "paired"):
            for tail in ("0", "1", "2", "3") if stn == "preflop" else ("unopened", "facing_bet"):
                prof.ctx_bluff[f"{stn}|{tex}|{tail}"] = [float(n_bluff), float(n_value)]


def test_mixer_read_gated_bluff_catch():
    """READ-GATED facing branch (post-2026-06-13 cloud gate): vs an over-aggressor, the call-boost
    fires ONLY when showdown ground truth CONFIRMS bluff-heavy. Frequency-only and value-heavy
    reveals both DEFER to the search policy (no call-boost) — the fix that kills the lag -1.1."""
    from pokerbot.opponent.root_mixer import apply_root_exploit_mixer

    bluff_fired = 0
    for seed in range(20):
        st, base = _facing_state_and_base(seed)
        prof_nb = _over_aggressive_profile()                      # no reveal data (frequency-only)
        prof_vh = _over_aggressive_profile()
        _set_all_bluff(prof_vh, 0.5, 20.0)                        # value-heavy ground truth
        prof_bh = _over_aggressive_profile()
        _set_all_bluff(prof_bh, 20.0, 0.5)                        # bluff-heavy ground truth

        out_nb = apply_root_exploit_mixer(st, 0, base.copy(), {1: prof_nb})
        out_vh = apply_root_exploit_mixer(st, 0, base.copy(), {1: prof_vh})
        out_bh = apply_root_exploit_mixer(st, 0, base.copy(), {1: prof_bh})
        # Frequency-only and value-heavy must NOT bluff-catch more than each other (both defer).
        assert abs(out_vh[2] - out_nb[2]) < 1e-9                  # value-heavy == frequency (deferred)
        assert out_bh[2] >= out_nb[2] - 1e-9                      # confirmed bluffer => call >= defer
        if out_bh[2] > out_nb[2] + 1e-9:
            bluff_fired += 1
    assert bluff_fired >= 1, "confirmed bluff-heavy read never triggered the gated call-boost"


def test_schema_v5_round_trip_and_v4_compat(tmp_path):
    from pokerbot.opponent.player_profiles import PlayerProfileStore

    p = tmp_path / "profiles.json"
    store = PlayerProfileStore(p)
    s = store.get("42")
    s.hands = 3
    s.bluff_decay_and_add("flop|dry|unopened", True, 3)
    s.bluff_decay_and_add("flop|dry|unopened", False, 3)
    store.save()

    store2 = PlayerProfileStore(p)
    s2 = store2.get("42")
    assert s2.ctx_bluff["flop|dry|unopened"] == [1.0, 1.0]
    assert s2.ctx_bluff_last["flop|dry|unopened"] == 3
    assert bluff_read(s2, "flop|dry|unopened") is not None

    # A v4-style file (no ctx_bluff) loads with the field empty ⇒ bluff_read None.
    p4 = tmp_path / "v4.json"
    p4.write_text('{"schema_version": 4, "profiles": {"7": {"hands": 5}}, "seen_hands": []}',
                  encoding="utf-8")
    store4 = PlayerProfileStore(p4)
    assert store4.get("7").ctx_bluff == {}
    assert bluff_read(store4.get("7"), "flop|dry|unopened") is None


def test_derive_bluffs_from_shown_record():
    """A revealed bluffy flop bet classifies as bluff; a revealed strong bet as value; an
    UNREVEALED player contributes nothing."""
    from pokerbot.opponent.player_profiles import derive_hand_stats

    record = {
        "hand_id": "h1",
        "board": ["Kh", "7d", "2c"],
        "actions": [
            {"street": "preflop", "type": "call", "user_id": 1, "name": "bluffy"},
            {"street": "preflop", "type": "call", "user_id": 2, "name": "strong"},
            {"street": "preflop", "type": "check", "user_id": 3, "name": "hidden"},
            {"street": "flop", "type": "bet", "amount": 2.0, "user_id": 1, "name": "bluffy"},
            {"street": "flop", "type": "raise", "amount": 6.0, "user_id": 2, "name": "strong"},
            {"street": "flop", "type": "raise", "amount": 18.0, "user_id": 3, "name": "hidden"},
        ],
        "winners": [{"user_id": 2}],
        "shown": [
            {"user_id": 1, "name": "bluffy", "cards": ["5h", "4s"]},   # air on Kh7d2c
            {"user_id": 2, "name": "strong", "cards": ["Kd", "Ks"]},   # top set
        ],
    }
    inc = derive_hand_stats(record)
    b1 = inc["1"]["_ctx_bluff"]
    assert sum(v[0] for v in b1.values()) == 1.0 and sum(v[1] for v in b1.values()) == 0.0
    b2 = inc["2"]["_ctx_bluff"]
    assert sum(v[1] for v in b2.values()) == 1.0 and sum(v[0] for v in b2.values()) == 0.0
    assert not inc["3"].get("_ctx_bluff")  # never shown ⇒ no ground truth

    # update_from_hand folds the increments into the persistent profile.
    from pokerbot.opponent.player_profiles import PlayerProfileStore
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        store = PlayerProfileStore(Path(td) / "p.json")
        store.update_from_hand(record)
        assert sum(v[0] for v in store.get("1").ctx_bluff.values()) == 1.0
        assert sum(v[1] for v in store.get("2").ctx_bluff.values()) == 1.0


def test_battery_bluff_flag_off_is_inert():
    """bluff_model=False (the Phase-0 arms): no ctx_bluff is ever populated and the result keys
    report zero — the running fair battery is byte-unaffected by the Phase-2 code."""
    from pokerbot.eval.exploitation_battery import run_archetype

    r = run_archetype("hs_lag", hands=15, seed=0, progress_every=0)
    assert r["bluff_obs"] == 0.0
    assert r["_bot_stats"].ctx_bluff == {}


def test_battery_bluff_flag_on_learns_at_showdown():
    from pokerbot.eval.exploitation_battery import run_archetype

    r = run_archetype("hs_lag", hands=120, seed=0, progress_every=0, bluff_model=True)
    assert r["bluff_obs"] > 0.0, "no showdown reveals ingested in 120 hands vs hs_lag"
    assert 0.0 <= r["bluff_ratio_raw"] <= 1.0
