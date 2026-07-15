"""Dependency-light tests for deep-stack preflop action semantics."""

import numpy as np
import pytest

from pokerbot.search.preflop_projection import project_deep_preflop_blueprint


def test_moves_allin_to_largest_legal_raise_without_changing_other_mass():
    p = np.zeros(11)
    p[0], p[2], p[5], p[10] = 0.10, 0.30, 0.20, 0.40
    legal = np.zeros(11, dtype=bool)
    legal[[0, 2, 5, 7, 10]] = True

    out, target, moved = project_deep_preflop_blueprint(p, legal)

    assert target == 7
    assert moved == pytest.approx(0.40)
    assert out[10] == 0.0
    assert out[0] == pytest.approx(0.10)
    assert out[2] == pytest.approx(0.30)
    assert out[5] == pytest.approx(0.20)
    assert out[7] == pytest.approx(0.40)
    assert out.sum() == pytest.approx(1.0)


def test_facing_terminal_bet_maps_duplicate_allin_to_call():
    p = np.zeros(11)
    p[0], p[2], p[10] = 0.2, 0.3, 0.5
    legal = np.zeros(11, dtype=bool)
    legal[[0, 2, 10]] = True

    out, target, moved = project_deep_preflop_blueprint(p, legal)

    assert target == 2
    assert moved == pytest.approx(0.5)
    assert out[2] == pytest.approx(0.8)
    assert out[10] == 0.0


def test_zero_allin_mass_is_identity():
    p = np.zeros(11)
    p[0], p[2], p[6] = 0.2, 0.3, 0.5
    legal = p > 0

    out, target, moved = project_deep_preflop_blueprint(p, legal)

    np.testing.assert_array_equal(out, p)
    assert target is None
    assert moved == 0.0


def test_observed_tier1_ako_shape_cannot_become_a_deep_shove():
    """Regression for the live 500bb BB-vs-3bb row: AI .38/call .37/fold .14."""
    p = np.zeros(11)
    p[0], p[2], p[7], p[10] = 0.14, 0.37, 0.11, 0.38
    legal = np.zeros(11, dtype=bool)
    legal[[0, 2, 7, 10]] = True

    out, target, moved = project_deep_preflop_blueprint(p, legal)

    assert target == 7
    assert moved == pytest.approx(0.38)
    assert out[10] == 0.0
    assert out[7] == pytest.approx(0.49)
    assert int(np.argmax(out)) == 7


def test_lookup_miss_fails_closed_instead_of_manufacturing_a_raise():
    legal = np.zeros(11, dtype=bool)
    legal[[0, 2, 5, 7, 10]] = True
    uniform_fallback = legal.astype(float) / legal.sum()

    out, target, moved = project_deep_preflop_blueprint(
        uniform_fallback, legal, source_hit=False)

    assert target is None
    assert moved == pytest.approx(0.2)
    assert out[0] == 1.0
    assert out[7] == 0.0
    assert out[10] == 0.0


def test_lookup_miss_checks_when_checking_is_free():
    legal = np.zeros(11, dtype=bool)
    legal[[1, 4, 10]] = True
    uniform_fallback = legal.astype(float) / legal.sum()

    out, target, _ = project_deep_preflop_blueprint(
        uniform_fallback, legal, source_hit=False)

    assert target is None
    assert out[1] == 1.0


def test_reference_depth_moves_oversized_raise_buckets_into_source_menu():
    """A live-deep bucket that was terminal/illegal at 100bb is not a valid
    nonterminal action even though the 500bb legal mask exposes it."""
    p = np.zeros(11)
    p[0], p[2], p[5], p[7], p[9], p[10] = .05, .05, .10, .20, .25, .35
    live = np.zeros(11, dtype=bool)
    live[[0, 2, 5, 7, 9, 10]] = True
    source = np.zeros(11, dtype=bool)
    source[[0, 2, 5, 10]] = True  # only bucket 5 remains nonterminal at 100bb

    out, target, moved = project_deep_preflop_blueprint(
        p, live, source_legal=source)

    assert target == 5
    assert moved == pytest.approx(.80)
    assert out[5] == pytest.approx(.90)
    assert out[7] == out[9] == out[10] == 0.0
    assert out.sum() == pytest.approx(1.0)


def test_reference_depth_with_no_safe_raise_keeps_learned_fold_call_mix():
    """Regression for the live 93o 4-bet-to-661/fold line: once the public pot
    is already source-stack-sized, no raise is nonterminal in the trained game.
    Terminal bluff mass is removed rather than transplanted into a 330bb raise."""
    p = np.zeros(11)
    p[0], p[2], p[8], p[10] = .20, .30, .25, .25
    live = np.zeros(11, dtype=bool)
    live[[0, 2, 8, 10]] = True
    source = np.zeros(11, dtype=bool)
    source[[0, 2, 10]] = True

    out, target, moved = project_deep_preflop_blueprint(
        p, live, source_legal=source)

    assert target is None
    assert moved == pytest.approx(.50)
    assert out[0] == pytest.approx(.40)
    assert out[2] == pytest.approx(.60)
    assert out[8] == out[10] == 0.0


def test_reference_depth_all_terminal_mass_fails_closed_without_calling():
    p = np.zeros(11)
    p[9] = 1.0
    live = np.zeros(11, dtype=bool)
    live[[0, 2, 9, 10]] = True
    source = np.zeros(11, dtype=bool)
    source[[0, 2, 10]] = True

    out, target, _ = project_deep_preflop_blueprint(
        p, live, source_legal=source)

    assert target is None
    assert out[0] == 1.0
    assert out[2] == out[9] == out[10] == 0.0
