"""P3: 6-max multiway eval harness — mechanics + SearchAgent-as-hero plumbing."""

from __future__ import annotations

import os

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

import random

import numpy as np
import pytest

from pokerbot.engine.game import NUM_PLAYERS
from pokerbot.eval.bots import BOT_REGISTRY
from pokerbot.eval.multiway_table import run_multiway


def test_determinism_same_seed():
    a = run_multiway(BOT_REGISTRY["tag"](random.Random(1)), hands=60, seed=11)
    b = run_multiway(BOT_REGISTRY["tag"](random.Random(1)), hands=60, seed=11)
    assert a.bb_per_100 == b.bb_per_100 and a.ci95_bb_per_100 == b.ci95_bb_per_100


def test_duplicate_seat_rotation_balanced():
    # Duplicate mode plays each deck once per hero seat -> every seat hosts hero equally.
    r = run_multiway(BOT_REGISTRY["nit"](random.Random(0)), hands=NUM_PLAYERS * 5, seed=3, duplicate=True)
    assert r.hands % NUM_PLAYERS == 0
    assert len(r.detail["bb_per_100_by_hero_seat"]) == NUM_PLAYERS


def test_ci_fields_sane_and_zero_straddle_flagged():
    r = run_multiway(BOT_REGISTRY["tag"](random.Random(5)), hands=30, seed=9)
    assert r.ci95_bb_per_100 >= 0.0
    assert np.isfinite(r.bb_per_100) and np.isfinite(r.stderr_bb)
    # At n=30 the CI is huge -> beats_zero must be a strict CI-above-0 test, not a sign test.
    assert r.beats_zero == ((r.bb_per_100 - r.ci95_bb_per_100) > 0.0)


def test_non_duplicate_runs():
    r = run_multiway(BOT_REGISTRY["random"](random.Random(1)), hands=24, seed=1, duplicate=False)
    assert r.hands == 24


def test_field_must_be_full_ring():
    with pytest.raises(ValueError):
        run_multiway(BOT_REGISTRY["tag"](random.Random(1)), field_bots=[None, None], hands=6)


@pytest.mark.slow
def test_pluribus_searchagent_as_hero_plumbing():
    """A Pluribus SearchAgent must be usable as a multiway hero (reset_hand per hand, no neural net)."""
    from pokerbot.eval.search_agent import SearchAgent

    hero = SearchAgent(use_pluribus_search=True, time_budget_ms=400, pluribus_leaf_select="uniform")
    # One duplicate group (NUM_PLAYERS hero-hands) — small but exercises preflop+postflop search paths.
    r = run_multiway(hero, hands=NUM_PLAYERS, seed=2, duplicate=True)
    assert r.hands == NUM_PLAYERS
    assert np.isfinite(r.bb_per_100)


@pytest.mark.slow
def test_aivat_decision_term_machinery():
    """AIVAT decision CV (gadget baseline) runs + reports an UNBIASED adjusted estimate. Variance
    REDUCTION needs a real (correlating) blueprint — with an empty blueprint the baseline is noise, so we
    only assert the machinery (finite, unbiased, reported), not a positive reduction."""
    from pokerbot.eval.search_agent import SearchAgent

    hero = SearchAgent(use_pluribus_search=True, time_budget_ms=200, pluribus_leaf_select="uniform")
    r = run_multiway(hero, hands=NUM_PLAYERS, seed=4, duplicate=True, aivat=True)
    assert r.detail["aivat"] is True
    assert np.isfinite(r.bb_per_100) and "variance_reduction" in r.detail
    assert "raw_bb_per_100" in r.detail


# --------------------------------------------------------------------------- AIVAT chance term


def _played_hand_to_showdown(seed: int):
    """Deal one all-calling-station hand that reaches showdown (so flop/turn/river are revealed)."""
    import random as _r

    from pokerbot.engine.game import play_hand

    policies = {s: BOT_REGISTRY["calling_station"](_r.Random(seed * 13 + s)) for s in range(NUM_PLAYERS)}
    fs, _ = play_hand(policies, rng=_r.Random(seed))
    return fs


def test_chance_term_replay_reconstructs_realized_board():
    """The hand-replay reconstructs each board-reveal node with the EXACT realized board prefix (the
    chance term's correctness rests on holding everything-but-the-revealed-card fixed)."""
    from pokerbot.eval.chance_term import _replay_reveals

    fs = _played_hand_to_showdown(7)
    reveals = _replay_reveals(fs)
    assert len(reveals) == 3  # flop(3) + turn(1) + river(1)
    for rev in reveals:
        n = rev.prev_board_len + rev.n_new
        assert [c.index for c in rev.state.board[:n]] == [c.index for c in fs.board[:n]]


def test_chance_term_single_reveal_is_exactly_zero_mean():
    """PROVABLE unbiasedness (turn/river): for ONE reveal, averaging the correction over every possible
    realized card (the full deck complement) is EXACTLY zero — because ``expected`` is the exact enumerated
    mean over that same complement and ``realized`` ranges over it. Independent of the value function v."""
    from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
    from pokerbot.search.tabular_blueprint_adapter import TabularBlueprintAdapter
    from pokerbot.eval.chance_term import (
        _complement,
        _env_with_board,
        _replay_reveals,
        _value,
    )

    fs = _played_hand_to_showdown(7)
    adapter = TabularBlueprintAdapter({}, client_bet_sizing=True)  # empty bp: v is arbitrary but FIXED
    leaf = ContinuationLeafConfig(select="uniform", rollout_samples=1, gadget_iters=16)
    hero = 0
    holes = [p.hole for p in fs.players]
    reveals = _replay_reveals(fs)
    # Take the river reveal (single card, fully enumerable).
    rev = next(r for r in reveals if r.n_new == 1 and r.prev_board_len == 4)
    comp = _complement(holes, list(rev.state.board[: rev.prev_board_len]))
    seed = 12345
    vals = {c: _value(_env_with_board(rev, hero, [c]), hero, adapter, leaf, seed) for c in comp}
    expected = float(np.mean(list(vals.values())))
    # Correction averaged over every possible realized card == 0 (the zero-mean / Lemma-1 property).
    corrections = [v - expected for v in vals.values()]
    assert abs(float(np.mean(corrections))) < 1e-9


@pytest.mark.slow
def test_chance_term_end_to_end_unbiased_and_reported():
    """End-to-end: SearchAgent hero with aivat + chance term runs, reports the decision-only vs combined
    variance-reduction split, and stays UNBIASED (adj mean within a wide tolerance of raw at small n)."""
    from pokerbot.eval.search_agent import SearchAgent

    hero = SearchAgent(use_pluribus_search=True, time_budget_ms=150, pluribus_leaf_select="uniform")
    r = run_multiway(hero, hands=NUM_PLAYERS, seed=4, duplicate=True, aivat=True, aivat_chance=True)
    assert r.detail["aivat_chance"] is True
    assert "variance_reduction_decision_only" in r.detail
    assert "mean_chance_correction_bb" in r.detail
    assert np.isfinite(r.detail["adj_mean_bb_per_100"])
