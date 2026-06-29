"""Tests for the serve-time payout resolver (pokerbot/tournament/serve_payouts.py).

Verifies the thin orchestration layer over the certified payout model: format-correct ladders, PKO
bounty economics, satellite seat shape, the soft-fail signal the live gate keys on, per-table caching,
and that a resolved ladder feeds the ICM serve path (tournament_decide) unchanged.
"""

from __future__ import annotations

import pytest

from pokerbot.tournament import (
    PayoutBook,
    ResolvedPayouts,
    remaining_payout_vector,
    resolve_from_prize_list,
    resolve_payouts,
    tournament_decide,
    try_resolve,
)
from pokerbot.tournament.payouts import (
    TournamentFormat,
    places_paid,
    prize_pool,
)


# ----------------------------------------------------------------------------------------------------
# NORMAL
# ----------------------------------------------------------------------------------------------------

def test_normal_ladder_is_pool_conserving_and_monotone():
    rp = resolve_payouts(tournament_type="MTT_FREEZEOUT", entrants=100, buy_in=10.0)
    assert rp.fmt is TournamentFormat.NORMAL and not rp.is_pko and not rp.is_satellite
    # length == places_paid, conserves the prize pool, strictly best-first (non-increasing).
    assert rp.places == places_paid(TournamentFormat.NORMAL, 100) == len(rp.payouts)
    assert sum(rp.payouts) == pytest.approx(prize_pool(TournamentFormat.NORMAL, 100, 10.0), rel=1e-9)
    assert all(rp.payouts[i] >= rp.payouts[i + 1] - 1e-12 for i in range(len(rp.payouts) - 1))
    assert rp.payouts[0] > rp.payouts[-1]                 # a real ladder, not flat
    assert rp.bounty_pool == 0.0 and rp.per_entry_bounty == 0.0


def test_normal_accepts_explicit_prize_pool_and_places():
    rp = resolve_payouts(tournament_type="MTT_FREEZEOUT", entrants=50, prize_pool=1000.0, places=5)
    assert rp.places == 5 and sum(rp.payouts) == pytest.approx(1000.0, rel=1e-9)


# ----------------------------------------------------------------------------------------------------
# PKO
# ----------------------------------------------------------------------------------------------------

def test_pko_captures_bounty_economics_alongside_prize_ladder():
    rp = resolve_payouts(bounty=True, entrants=100, buy_in=10.0)
    assert rp.fmt is TournamentFormat.PKO and rp.is_pko
    # The prize ladder is the PRIZE HALF; bounty pool == prize pool (50/50 of the non-rake buy-in).
    assert sum(rp.payouts) == pytest.approx(prize_pool(TournamentFormat.PKO, 100, 10.0), rel=1e-9)
    assert rp.bounty_pool == pytest.approx(sum(rp.payouts), rel=1e-9)
    assert rp.per_entry_bounty == pytest.approx(rp.bounty_pool / 100, rel=1e-9)
    # PKO pays fewer places than NORMAL for the same field (0.133 vs 0.175).
    assert rp.places == places_paid(TournamentFormat.PKO, 100)
    assert rp.places < places_paid(TournamentFormat.NORMAL, 100)


def test_tournament_type_pko_string_classifies():
    rp = resolve_payouts(tournament_type="MTT_PROGRESSIVE_KNOCK_OUT", entrants=100, buy_in=10.0)
    assert rp.is_pko and rp.bounty_pool > 0.0


# ----------------------------------------------------------------------------------------------------
# SATELLITE
# ----------------------------------------------------------------------------------------------------

def test_satellite_is_flat_seat_ladder_with_buyin_proxy_value():
    rp = resolve_payouts(is_satellite=True, seats=10, buy_in=20.0, entrants=100)
    assert rp.is_satellite and rp.fmt is TournamentFormat.SATELLITE
    assert rp.places == 10 and rp.payouts == (20.0,) * 10      # seat_value defaults to buy_in
    assert rp.seat_value == 20.0


def test_satellite_seats_capped_by_entrants():
    rp = resolve_payouts(is_satellite=True, seats=10, seat_value=33.0, entrants=4)
    assert rp.places == 4 and rp.payouts == (33.0,) * 4


def test_satellite_requires_seats():
    with pytest.raises(ValueError):
        resolve_payouts(is_satellite=True, buy_in=20.0, entrants=100)


# ----------------------------------------------------------------------------------------------------
# captured prize_list (the exact, highest-fidelity live path)
# ----------------------------------------------------------------------------------------------------

def test_resolve_from_prize_list_sorts_best_first_and_trims_zero_tail():
    # TourneyPrizeListBean rows arrive out of order, with a zero/unpaid tail.
    pl = [{"rank": 3, "amount": 20.0}, {"rank": 1, "amount": 50.0},
          {"rank": 2, "amount": 30.0}, {"rank": 4, "amount": 0.0}]
    rp = resolve_from_prize_list(pl, entrants=100)
    assert rp.payouts == (50.0, 30.0, 20.0) and rp.places == 3 and rp.entrants == 100


def test_resolve_from_prize_list_accepts_bare_amounts_and_empty():
    assert resolve_from_prize_list([100.0, 60.0, 40.0]).payouts == (100.0, 60.0, 40.0)
    assert resolve_from_prize_list([]) is None
    assert resolve_from_prize_list([{"rank": 1, "amount": None}]) is None


# ----------------------------------------------------------------------------------------------------
# soft-fail (the live gate's "payouts unknown -> stay safe" signal)
# ----------------------------------------------------------------------------------------------------

def test_try_resolve_returns_none_on_insufficient_inputs():
    assert try_resolve(tournament_type="MTT_FREEZEOUT") is None          # no entrants/pool/buy_in
    assert try_resolve(bounty=True, entrants=100) is None                # PKO needs buy_in or pool
    # but a sufficient call still resolves
    assert try_resolve(tournament_type="MTT_FREEZEOUT", entrants=100, buy_in=10.0) is not None


# ----------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------

def test_top_and_remaining_pad_and_truncate():
    rp = resolve_payouts(tournament_type="MTT_FREEZEOUT", entrants=100, buy_in=10.0)
    n = len(rp.payouts)
    assert rp.top(3) == rp.payouts[:3]                                   # truncate
    assert rp.top(n + 2) == rp.payouts + (0.0, 0.0)                      # pad with 0
    assert remaining_payout_vector(rp, 2) == rp.payouts[:2]
    assert remaining_payout_vector([10.0, 5.0], 4) == (10.0, 5.0, 0.0, 0.0)
    assert rp.top(0) == ()


# ----------------------------------------------------------------------------------------------------
# PayoutBook caching
# ----------------------------------------------------------------------------------------------------

def test_payout_book_caches_until_inputs_change():
    book = PayoutBook()
    a = book.resolve("t1", tournament_type="MTT_FREEZEOUT", entrants=100, buy_in=10.0)
    b = book.resolve("t1", tournament_type="MTT_FREEZEOUT", entrants=100, buy_in=10.0)
    assert a is b is book.get("t1")                                      # cache hit -> same object
    # entrants grew during late reg -> re-resolve (new object, more places)
    c = book.resolve("t1", tournament_type="MTT_FREEZEOUT", entrants=200, buy_in=10.0)
    assert c is not a and c.entrants == 200 and c.places >= a.places


def test_payout_book_does_not_cache_failures_and_clears():
    book = PayoutBook()
    assert book.resolve("t2", tournament_type="MTT_FREEZEOUT") is None   # insufficient -> None
    assert book.get("t2") is None                                        # not cached
    book.resolve("t2", tournament_type="MTT_FREEZEOUT", entrants=100, buy_in=10.0)
    assert book.get("t2") is not None
    book.clear("t2")
    assert book.get("t2") is None


# ----------------------------------------------------------------------------------------------------
# integration: a resolved ladder feeds the ICM serve path unchanged
# ----------------------------------------------------------------------------------------------------

def test_resolved_ladder_drives_tournament_decide():
    rp = resolve_payouts(tournament_type="MTT_FREEZEOUT", entrants=180, buy_in=20.0)
    stacks = [10.0] * 6                                                  # shallow push/fold band
    blinds = {4: 0.5, 5: 1.0}
    # folded to hero (seat 3), seats 4/5 behind: AA must jam, 72o must fold -- with the live ladder.
    d_jam = tournament_decide(stacks, list(rp.payouts), hero_seat=3, hero_hand="AhAs",
                              blinds=blinds, ante=0.0, action_order=[0, 1, 2, 3, 4, 5],
                              history=[(0, "fold"), (1, "fold"), (2, "fold")])
    d_fold = tournament_decide(stacks, list(rp.payouts), hero_seat=3, hero_hand="7c2d",
                               blinds=blinds, ante=0.0, action_order=[0, 1, 2, 3, 4, 5],
                               history=[(0, "fold"), (1, "fold"), (2, "fold")])
    assert d_jam.action == "jam" and d_fold.action == "fold"
