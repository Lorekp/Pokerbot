"""Validate the open-shove solver's product-pairwise multiway-equity approximation against EXACT
3-way enumeration. The open-shove EV uses eq_multi(hero) ~= prod_o eq(hero vs villain_o); this checks
that approximation is tight on representative all-in triples (so the multiway branches -- a minority
of first-in shove EV -- are handled accurately, raising confidence in the ranges)."""

from __future__ import annotations

import pytest

from pokerbot.tournament.allin_ev import hu_outcome_probs, three_way_equity


def _pairwise_eq(hero, vil):
    pw, pc, pl = hu_outcome_probs(hero, vil)
    return pw + 0.5 * pc                         # hero's heads-up equity vs vil


# Made-hand triples (pairs/made): the independence approximation is essentially exact.
MADE_TRIPLES = [
    ("AsAh", "KsKd", "QcQh"),
    ("7c7d", "AhKs", "Ts9s"),
    ("AsKs", "AhQh", "2c2d"),
]


def test_product_pairwise_tight_for_made_hands():
    for hero, v1, v2 in MADE_TRIPLES:
        exact = three_way_equity(hero, v1, v2)
        approx = _pairwise_eq(hero, v1) * _pairwise_eq(hero, v2)
        assert abs(exact - approx) < 0.05            # tight where it matters most


def test_overcard_hands_are_conservatively_underestimated():
    # AK vs two pairs: AK's overcard outs beat BOTH villains (correlated), which the independence
    # product misses -> it UNDERESTIMATES multiway equity. Documented: the error is bounded and in
    # the SAFE direction (the solver shoves/calls these slightly too tight in multiway, never too
    # loose). Multiway branches are a minority of first-in shove EV, so range impact is small.
    exact = three_way_equity("AhKs", "QcQd", "JsJh")
    approx = _pairwise_eq("AhKs", "QcQd") * _pairwise_eq("AhKs", "JsJh")
    assert exact > approx                            # conservative (under-estimates -> safe)
    assert (exact - approx) < 0.20                   # bounded


def test_three_way_equity_basic_sanity():
    # A dominant hand has higher 3-way equity than a dominated one in the same spot.
    strong = three_way_equity("AsAh", "KsKd", "QcQh")
    weak = three_way_equity("7c7d", "AsAh", "KsKd")
    assert strong > weak
    assert 0.0 < weak < strong < 1.0
