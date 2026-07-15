"""Regression tests for passive, strength-aware blueprint coverage fallback."""

import numpy as np

from pokerbot.engine.cards import Card
from pokerbot.search.preflop_miss_prior import coverage_hole_policy


def _idx(*cards: str):
    return [Card.from_str(c).index for c in cards]


def _facing():
    legal = np.zeros(11, dtype=bool)
    legal[[0, 2, 7, 10]] = True
    return legal


def test_cheap_aqs_miss_calls_but_never_raises():
    out, score, threshold, *_ = coverage_hole_policy(
        _idx("Ah", "Qh"), _facing(), to_call_bb=1.5, effective_stack_bb=184.0)
    assert score == 11.0 and threshold == 9.0
    assert out[2] == 1.0
    assert out[3:].sum() == 0.0


def test_cheap_93o_miss_folds():
    out, score, threshold, *_ = coverage_hole_policy(
        _idx("9s", "3d"), _facing(), to_call_bb=1.5, effective_stack_bb=184.0)
    assert score < threshold
    assert out[0] == 1.0


def test_large_call_tightens_aqs_to_fold_but_keeps_kings():
    aq, _, threshold, *_ = coverage_hole_policy(
        _idx("Ah", "Qh"), _facing(), to_call_bb=40.0, effective_stack_bb=100.0)
    kk, score, _, *_ = coverage_hole_policy(
        _idx("Ks", "Kh"), _facing(), to_call_bb=40.0, effective_stack_bb=100.0)
    assert threshold == 16.0
    assert aq[0] == 1.0
    assert score == 16.0 and kk[2] == 1.0


def test_free_action_checks_even_with_trash():
    legal = np.zeros(11, dtype=bool)
    legal[[1, 5, 10]] = True
    out, _, _, *_ = coverage_hole_policy(
        _idx("7s", "2d"), legal, to_call_bb=0.0, effective_stack_bb=100.0)
    assert out[1] == 1.0


def test_cheap_late_position_defence_calls_kjo_and_kts_on_coverage_miss():
    kjo, score, threshold, *_ = coverage_hole_policy(
        _idx("Kh", "Jc"), _facing(), to_call_bb=2.5,
        effective_stack_bb=193.0, hero_position="BTN", opener_position="HJ")
    kts, score2, threshold2, *_ = coverage_hole_policy(
        _idx("Kh", "Th"), _facing(), to_call_bb=2.5,
        effective_stack_bb=193.0, hero_position="BB", opener_position="CO")
    assert score == 7.0 and threshold == 7.0 and kjo[2] == 1.0
    assert score2 == 8.0 and threshold2 == 7.0 and kts[2] == 1.0
    assert kjo[3:].sum() == 0.0 and kts[3:].sum() == 0.0


def test_position_relaxation_never_applies_to_utg_or_large_prices():
    vs_utg, _, utg_threshold, *_ = coverage_hole_policy(
        _idx("Kh", "Jc"), _facing(), to_call_bb=2.5,
        effective_stack_bb=193.0, hero_position="BTN", opener_position="UTG")
    expensive, _, expensive_threshold, *_ = coverage_hole_policy(
        _idx("Kh", "Jc"), _facing(), to_call_bb=15.0,
        effective_stack_bb=193.0, hero_position="BTN", opener_position="HJ")
    assert utg_threshold == 9.0 and vs_utg[0] == 1.0
    assert expensive_threshold == 12.0 and expensive[0] == 1.0


def test_short_stack_aks_never_folds_cheap_open_on_chen_pair_misordering():
    """Live regression: AKs BTN vs a 3.25bb CO open at ~13bb relevant
    depth had Chen 12 < pair-biased threshold 14 and folded."""
    out, score, threshold, equity, required, override = coverage_hole_policy(
        _idx("Ac", "Kc"), _facing(), to_call_bb=3.25,
        effective_stack_bb=13.0, pot_bb=4.75,
        hero_position="BTN", opener_position="CO")

    assert score == 12.0 and threshold == 14.0
    assert equity is not None and required is not None and equity > required
    assert override is True
    assert out[2] == 1.0 and out[3:].sum() == 0.0


def test_equity_override_does_not_rescue_nonpremium_dominated_hand():
    out, score, threshold, equity, required, override = coverage_hole_policy(
        _idx("Kh", "Jc"), _facing(), to_call_bb=3.25,
        effective_stack_bb=13.0, pot_bb=4.75,
        hero_position="BTN", opener_position="UTG")

    assert score < 10.0 and score < threshold
    assert equity is None and required is None and override is False
    assert out[0] == 1.0
