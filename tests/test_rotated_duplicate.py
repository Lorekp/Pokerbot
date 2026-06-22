"""Seat-rotated duplicate absolute-EV estimator (run_field_absolute_rotated).

The search path emits no AIVAT chance control variate, so AIVAT gives NEGATIVE variance reduction
vs a non-blueprint field; seat-rotated duplicate dealing is the value-net-free estimator that cancels
hole-card luck instead. These lock its contract: homogeneous-field guard, output schema, the per-deck
seat-mean math, and CRN determinism. Uses the lightweight ExploitHero (no checkpoint).
"""

from __future__ import annotations

import random

from pokerbot.eval.exploitation_battery import (
    ExploitHero,
    field_spec,
    run_field_absolute_rotated,
)


def _hero_factory(*, exploit: bool):
    return ExploitHero(random.Random(0), exploit=exploit)


def test_homogeneous_field_guard():
    # A heterogeneous field breaks the duplicate cancellation -> must be rejected, not silently wrong.
    hetero = [("calling_station", None), ("nit", None), ("nit", None), ("nit", None), ("nit", None)]
    try:
        run_field_absolute_rotated(hetero, decks=2, hero_factory=_hero_factory)
    except ValueError as e:
        assert "homogeneous" in str(e).lower()
    else:
        raise AssertionError("expected a ValueError on a heterogeneous field")


def test_schema_and_seat_rotation():
    field = field_spec("homog:calling_station")
    r = run_field_absolute_rotated(field, decks=4, seed=0, hero_factory=_hero_factory,
                                   allin_ev=True, progress_every=0)
    assert r["method"] == "seat_rotated_duplicate"
    assert r["decks"] == 4
    for key in ("off_base", "on_exploit"):
        assert set(r[key]) >= {"decks", "mean_bb", "std", "se"}
        assert r[key]["decks"] == 4
    # uplift is the paired (on - off) seat-mean delta.
    assert abs(r["uplift_bb_per_hand"] - (r["on_exploit"]["mean_bb"] - r["off_base"]["mean_bb"])) < 1e-9


def test_rotation_reseats_the_hero_each_rotation():
    """REGRESSION: a search hero reads state.players[hero_seat].hole, so each rotation MUST update the
    hero's seat before on_new_hand(). The original bug left hero_seat=0 for all 6 rotations -> the hero
    played seat-0's cards from every physical seat (5/6 rotations garbage -> absurd -hundreds bb/100)."""
    from pokerbot.engine.actions import ActionType

    seen_resets, seen_acts = [], []

    class _Probe:
        def __init__(self):
            self.hero_seat = 0
            self.profile = None

        def set_profile(self, p):
            self.profile = p

        def reseed(self, s):
            pass

        def on_new_hand(self):
            seen_resets.append(self.hero_seat)

        def __call__(self, state):
            seen_acts.append((self.hero_seat, state.action_seat))
            legal = state.legal_actions()
            for at in (ActionType.CHECK, ActionType.CALL, ActionType.FOLD):
                a = next((x for x in legal if x.action_type == at), None)
                if a is not None:
                    return a
            return legal[0]

    field = field_spec("homog:calling_station")
    run_field_absolute_rotated(field, decks=1, seed=0, hero_factory=lambda *, exploit: _Probe(),
                               allin_ev=True, progress_every=0)
    # all six physical seats are occupied by the hero across the rotations (both arms)
    assert set(seen_resets) == set(range(6))
    # whenever the hero acts, it believes it sits in the seat that is actually acting
    assert seen_acts and all(hs == act for hs, act in seen_acts)


def test_crn_determinism():
    field = field_spec("homog:calling_station")
    a = run_field_absolute_rotated(field, decks=4, seed=7, hero_factory=_hero_factory, progress_every=0)
    b = run_field_absolute_rotated(field, decks=4, seed=7, hero_factory=_hero_factory, progress_every=0)
    assert a["off_base"]["mean_bb"] == b["off_base"]["mean_bb"]
    assert a["on_exploit"]["mean_bb"] == b["on_exploit"]["mean_bb"]


def test_net_of_rake_column_is_additive():
    """GAP 3 — Workstream-D ground-rule 6: the rotated-ladder battery must carry a rake-adjusted
    net_of_rake bb/100 column, ADDITIVELY (the chip-EV numbers must be byte-unchanged). Mirrors the
    rake math the dedicated sizing_channel harness already uses (pokerbot.economics.rake.net_cost_bb:
    5%-flat capped rake with the per-tier BB-cap hump, contributed attribution, rakeback, splash fee)."""
    field = field_spec("homog:calling_station")
    base = run_field_absolute_rotated(field, decks=8, seed=3, hero_factory=_hero_factory,
                                      allin_ev=False, progress_every=0)
    raked = run_field_absolute_rotated(field, decks=8, seed=3, hero_factory=_hero_factory,
                                       allin_ev=False, progress_every=0, net_of_rake_stake="0.10/0.25")
    # 1) the chip-EV columns are byte-identical with rake on vs off (same play_hand call).
    assert raked["off_base"]["mean_bb"] == base["off_base"]["mean_bb"]
    assert raked["on_exploit"]["mean_bb"] == base["on_exploit"]["mean_bb"]
    # 2) the net_of_rake block is present and well-formed.
    n = raked["net_of_rake"]
    assert n["stake"] == "0.10/0.25" and n["rakeback_mode"] == "modeled"
    for key in ("off_base", "on_exploit"):
        assert "bb_per_100" in n[key]
    # 3) rake is a non-negative cost: net bb/100 <= chip bb/100, and the reported per-100 cost matches.
    assert n["off_base"]["bb_per_100"] <= raked["off_base"]["bb_per_100"] + 1e-9
    assert n["on_exploit"]["bb_per_100"] <= raked["on_exploit"]["bb_per_100"] + 1e-9
    assert n["rake_cost_bb_per_100_off"] >= -1e-9
    expect = (raked["off_base"]["bb_per_100"] - n["off_base"]["bb_per_100"])
    assert abs(n["rake_cost_bb_per_100_off"] - expect) < 1e-6
    # 4) the BB-cap hump differs across stakes (micro cap looser -> different net than the low valley).
    raked_hi = run_field_absolute_rotated(field, decks=8, seed=3, hero_factory=_hero_factory,
                                          allin_ev=False, progress_every=0, net_of_rake_stake="25/50")
    assert raked_hi["off_base"]["mean_bb"] == base["off_base"]["mean_bb"]  # still additive
