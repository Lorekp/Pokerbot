"""Bubble Protection -- CoinPoker's min-cash safety mechanic, modeled as a softened ICM cliff.

CoinPoker MTTs (Info panel "Bubble Protection: Available") refund the ENTIRE buy-in to a player who
busts in the ``B(N)`` seats *just before* the money -- for eligible entrants (registered pre-start,
cash/satellite buy-in; NOT promo tickets; excludes satellites/freerolls). ``B`` by field bracket:

    1-9 -> 0,  10-99 -> 1,  100-499 -> 2,  500-1499 -> 3,  1500-2999 -> 4,  3000-5000 -> 5  (cap 5).

The strategic point (per ``tournament_example.txt`` and the ULTRACODE spec): model those ``B``
pre-ITM seats as **net-EV ~= 0 (buy-in returned), NOT a hard-0 bust**, and apply this floor
**BEFORE** computing ``bubble_factor``. Concretely we append ``B`` protected ranks paying ``buy_in``
to the ICM payout vector. This FLATTENS the cliff between "bust just outside the money" (now a
refund, not zero) and "min-cash", so the local risk premium drops and **protected players' near-money
shove/fold ranges are WIDER, not tighter**. Ineligible (promo-ticket) entrants get the standard
hard-bubble payout vector unchanged.

This module only *constructs the protected payout vector*; the ICM math lives in
:mod:`pokerbot.tournament.icm`. Apply it as ``bubble_factor(stacks, protected_payouts(...), h, v)``.
"""

from __future__ import annotations

from typing import Sequence

from pokerbot.tournament.icm import bubble_factor

# Field-size -> number of bubble-protected pre-ITM seats. Brackets per CoinPoker (tournament_example.txt).
_BUBBLE_BRACKETS = (
    (9, 0),
    (99, 1),
    (499, 2),
    (1499, 3),
    (2999, 4),
    (5000, 5),
)
_BUBBLE_CAP = 5


def bubble_seats(field: int) -> int:
    """``B(N)``: how many pre-ITM seats are buy-in-refund protected for a field of ``field`` entrants."""
    if field <= 0:
        return 0
    for hi, b in _BUBBLE_BRACKETS:
        if field <= hi:
            return b
    return _BUBBLE_CAP  # very large fields cap at 5 (documented bracket ceiling)


def protected_payouts(base_payouts: Sequence[float], *, field: int, buy_in: float,
                      eligible: bool = True) -> list[float]:
    """The ICM payout vector with bubble protection folded in.

    ``base_payouts`` is the in-the-money ladder (best-first, length = ``places_paid``). For an
    eligible entrant we append ``B(field)`` protected ranks each paying ``buy_in`` (the refund) --
    these are the seats immediately below the money; finishes worse than that still pay 0 (ICM pads
    them). The result feeds ``icm_equity`` / ``bubble_factor`` directly. Ineligible entrants get
    ``base_payouts`` unchanged (hard bubble).

    Net-EV interpretation: a refund of exactly ``buy_in`` is net-EV ~= 0 for the busting player; in
    the gross ICM payout space the model simply pays ``buy_in`` at those ranks, which is what softens
    the cliff. Pass ``buy_in`` in the SAME currency as ``base_payouts``.
    """
    base = [float(p) for p in base_payouts]
    if not eligible:
        return base
    b = bubble_seats(field)
    if b <= 0:
        return base
    return base + [float(buy_in)] * b


def protected_bubble_factor(stacks: Sequence[float], base_payouts: Sequence[float], hero: int,
                            villain: int, *, field: int, buy_in: float,
                            eligible: bool = True) -> float:
    """``bubble_factor`` evaluated on the bubble-protected payout vector (the floor applied BEFORE the
    risk premium). For an eligible near-money player this is <= the hard-bubble factor -> wider ranges.
    """
    pay = protected_payouts(base_payouts, field=field, buy_in=buy_in, eligible=eligible)
    return bubble_factor(stacks, pay, hero, villain)


def hard_bubble_factor(stacks: Sequence[float], base_payouts: Sequence[float], hero: int,
                       villain: int) -> float:
    """The unprotected (hard-0 bust) bubble factor -- the baseline protection is measured against."""
    return bubble_factor(stacks, base_payouts, hero, villain)
