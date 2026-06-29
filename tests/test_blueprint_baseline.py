"""Unit-test the blueprint-direct CV baseline hero shim (scripts/eval_cv_realistic._BlueprintHero).

This is the diagnosed fix for the under-delivering CV (the 1-iteration baseline was near-uniform, corr 0.58;
the blueprint-direct baseline shares the search hero's blueprint -> high corr). The shim's hero-INTERFACE and
play logic are tested here with an INJECTED fake adapter (no 66MB blueprint load); the real-snapshot
correlation gain is a separate CPU-gated measurement (run scripts/eval_cv_realistic.py --baseline blueprint).
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np

from pokerbot.engine.actions import Action
from pokerbot.engine.game import GameState

ROOT = Path(__file__).resolve().parent.parent


def _load_runner():
    spec = importlib.util.spec_from_file_location("ecvr", ROOT / "scripts" / "eval_cv_realistic.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _FakeAdapter:
    """Returns a fixed legal-ish 11-way distribution; records that blueprint() was called."""

    def __init__(self):
        self.calls = 0

    def blueprint(self, env, seat):
        self.calls += 1
        p = np.zeros(11, dtype=float)
        p[0] = 0.5   # fold
        p[2] = 0.5   # call
        return p


def _sample_state():
    st = GameState.new_hand(rng=random.Random(0))
    st.deal_hole_cards()
    return st


def test_hero_interface_present():
    m = _load_runner()
    h = m._BlueprintHero(_FakeAdapter(), seed=0)
    # the run_field_absolute_rotated hero interface
    assert callable(h)
    assert hasattr(h, "hero_seat") and h.hero_seat == 0
    assert hasattr(h, "rng")
    h.set_profile({"x": 1})        # no-op but must exist + not raise
    assert h.profile == {"x": 1}
    h.reseed(123)                  # must reset rng deterministically
    h.on_new_hand()                # no-op


def test_plays_a_valid_action_from_the_policy():
    m = _load_runner()
    fake = _FakeAdapter()
    h = m._BlueprintHero(fake, seed=0)
    st = _sample_state()
    a = h(st)
    assert isinstance(a, Action)
    assert fake.calls == 1          # it consulted the blueprint adapter


def test_reseed_makes_sampling_deterministic():
    m = _load_runner()
    st = _sample_state()
    h1 = m._BlueprintHero(_FakeAdapter(), seed=0); h1.reseed(7)
    h2 = m._BlueprintHero(_FakeAdapter(), seed=0); h2.reseed(7)
    # same seed + same fixed policy -> same sampled action type/amount across a few draws
    seq1 = [str(h1(_sample_state())) for _ in range(5)]
    seq2 = [str(h2(_sample_state())) for _ in range(5)]
    assert seq1 == seq2


def test_fallback_on_adapter_failure_still_returns_an_action():
    m = _load_runner()

    class _Boom:
        def blueprint(self, env, seat):
            raise RuntimeError("bridge miss")

    h = m._BlueprintHero(_Boom(), seed=0)
    a = h(_sample_state())
    assert isinstance(a, Action)    # safe legal fallback (fold), never crashes the eval
