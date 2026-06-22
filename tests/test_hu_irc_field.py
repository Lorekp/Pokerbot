"""Acceptance for the seat-rotated HU-vs-IRC harness (pokerbot/eval/hu_irc_field.py).

Pure-Python eval (np_engine), NO Rust crate. Validates the harness MECHANICS + calibration ordering,
not the net's convergence (that is the training-gated number, see docs/hu_engine_status.md).
"""

from __future__ import annotations

import os

os.environ.setdefault("POKERBOT_INFO_SUITS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from pokerbot.eval.hu_irc_field import (
    HUFieldConfig,
    hu_vs_irc_rotated,
    list_archetypes,
)
from pokerbot.eval.lbr import always_fold_policy, uniform_policy


def test_archetypes_registered():
    assert set(list_archetypes()) == {"irc_station", "irc_lag", "irc_nit", "irc_tag"}


def test_fold_hero_loses_blinds_bounded():
    """Fold-hero must lose to a real villain but only modestly (check-fold line, not a full BB/hand)."""
    cfg = HUFieldConfig(decks=120, seed=0, villain="irc_station", stack_bb=20, villain_rollouts=24)
    res = hu_vs_irc_rotated(always_fold_policy, cfg)
    assert res.hero_hands == 2 * cfg.decks
    assert -100.0 < res.bb_per_100 < 0.0  # loses, but bounded (not -inf, not winning)


def test_seed_reproducible():
    cfg = HUFieldConfig(decks=60, seed=7, villain="irc_nit", stack_bb=20, villain_rollouts=16)
    a = hu_vs_irc_rotated(always_fold_policy, cfg)
    b = hu_vs_irc_rotated(always_fold_policy, cfg)
    assert a.bb_per_100 == b.bb_per_100  # deterministic under a fixed seed


def test_uniform_worse_than_fold_vs_station():
    """Uniform (random all-ins) must do clearly WORSE than disciplined fold vs the loose station --
    a basic calibration ordering the harness must respect."""
    cfg = HUFieldConfig(decks=120, seed=1, villain="irc_station", stack_bb=20, villain_rollouts=24)
    fold = hu_vs_irc_rotated(always_fold_policy, cfg).bb_per_100
    unif = hu_vs_irc_rotated(uniform_policy, cfg).bb_per_100
    assert unif < fold


def test_result_fields_finite():
    cfg = HUFieldConfig(decks=40, seed=2, villain="irc_tag", stack_bb=20, villain_rollouts=16)
    res = hu_vs_irc_rotated(always_fold_policy, cfg)
    for v in (res.bb_per_100, res.bb_per_100_ci95, res.bb_per_hand, res.stderr_bb_per_hand):
        assert np.isfinite(v)
    assert res.bb_per_100_ci95 >= 0.0
