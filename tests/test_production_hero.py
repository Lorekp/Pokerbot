"""R2: the production policy plugged into the exploitation battery via ProductionHero.

Uses a STUB policy (legal-uniform) so no checkpoints are needed — the point is the ADAPTER + battery
wiring (protocol, seeded sampling for AIVAT, CRN reseed, online profile learning), not EV magnitude.
"""

from __future__ import annotations

import random
import types

import numpy as np
import torch

from pokerbot.agent import HybridAgent
from pokerbot.engine.game import GameState
from pokerbot.eval.exploitation_battery import run_archetype
from pokerbot.eval.production_hero import ProductionHero


class _StubPolicy:
    """Returns a legal-uniform policy; satisfies HybridPolicy.get_probs' model interface."""

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, weva_ids, actions, position, legal_mask, cards=None):
        lm = legal_mask.float()
        probs = lm / lm.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return types.SimpleNamespace(probs=probs)


def _make_hero(*, exploit: bool):
    agent = HybridAgent(_StubPolicy(), device="cpu", use_transformer=False)
    return ProductionHero(agent, random.Random(0), exploit=exploit, hero_seat=0)


def test_production_hero_satisfies_protocol():
    from pokerbot.opponent.stats import OpponentStats

    hero = _make_hero(exploit=True)
    # protocol surface the battery relies on
    for attr in ("set_profile", "pop_decisions", "pop_chance_corrections", "on_new_hand", "reseed"):
        assert hasattr(hero, attr)
    hero.set_profile(OpponentStats())
    hero.reseed(123)
    hero.on_new_hand()

    st = GameState.new_hand(button_seat=3, rng=random.Random(0))
    action = hero(st)
    assert action is not None
    decs = hero.pop_decisions()
    assert len(decs) >= 1
    probs, bv, idx = decs[0]
    assert np.asarray(probs).shape == (11,)
    assert np.asarray(bv).shape == (11,)
    assert 0 <= int(idx) < 11
    assert abs(float(np.asarray(probs).sum()) - 1.0) < 1e-6
    assert 0.0 <= hero.bet_freq <= 1.0


def test_production_hero_battery_runs_and_learns_online():
    r = run_archetype("calling_station", hands=30, seed=1,
                      hero_factory=lambda *, exploit: _make_hero(exploit=exploit))
    for k in ("uplift_bb_per_hand", "uplift_se", "on_aivat", "off_aivat", "profile_contexts"):
        assert k in r
    assert r["hands"] == 30
    assert r["profile_contexts"] > 0           # the villain was learned online into the profile
    # AIVAT ran on both arms (decision corrections were collected → finite-sample variance numbers)
    assert r["on_aivat"]["hands"] == 30
    assert r["off_aivat"]["hands"] == 30


def test_seeded_sampling_is_reproducible_under_crn():
    # Two heroes with the same seed + same deck must take the same first action (CRN determinism).
    h1, h2 = _make_hero(exploit=False), _make_hero(exploit=False)
    for h in (h1, h2):
        h.reseed(777)
        h.on_new_hand()
    st1 = GameState.new_hand(button_seat=3, rng=random.Random(42))
    st2 = GameState.new_hand(button_seat=3, rng=random.Random(42))
    a1, a2 = h1(st1), h2(st2)
    d1, d2 = h1.pop_decisions()[0], h2.pop_decisions()[0]
    assert d1[2] == d2[2]                       # identical sampled action index
