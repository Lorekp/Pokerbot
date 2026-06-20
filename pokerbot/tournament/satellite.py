"""Satellite payoffs -- the step-function ICM regime (seats/tickets, not a cash ladder).

A satellite pays ``seats`` identical prizes (each worth the parent buy-in) to the top ``seats``
finishers; everyone else gets nothing. That makes the payoff a **step function** (flat above the
bubble), which drives **extreme ICM**: marginal chips above a locked seat are worth ~nothing, so a
covered player should *fold everything that risks the seat* once a seat is mathematically locked.

Two pieces:
  * :func:`satellite_payouts` -- the descending prize vector ``[seat_value]*seats`` that feeds
    :func:`pokerbot.tournament.icm.icm_equity` / ``bubble_factor`` directly (the flat top is what
    produces the extreme risk premium near the bubble).
  * :func:`seat_locked` -- the rigorous, play-independent chip-fraction lock: a player holding at
    least ``1/(seats+1)`` of the live chips is **uncatchable** (you cannot carve ``seats`` rival
    piles each strictly larger than hero out of the remaining chips), hence guaranteed a seat. Once
    locked, :func:`should_fold_for_seat` says fold any chip-risking action.

The lock is the *adversarial-worst-case* statement (true regardless of how everyone else plays), so
it is a genuine certificate, not a heuristic. Blind/ante erosion only matters between hands; the
chip-fraction test is evaluated on the current live stacks (a momentary refinement for exact
hand-timing is noted inline). Dependency-light: numpy + the ICM primitive only.
"""

from __future__ import annotations

from typing import Sequence

from pokerbot.tournament.icm import bubble_factor


def satellite_payouts(seats: int, seat_value: float = 1.0) -> list[float]:
    """The satellite prize vector: ``seats`` identical prizes of ``seat_value`` (best-first), which
    is what ``icm_equity`` consumes (it pads the unpaid finishes with 0). ``seat_value`` defaults to
    1.0 for the unit-agnostic (pool-fraction) form; pass the parent buy-in for currency."""
    if seats < 0:
        raise ValueError(f"seats must be >= 0, got {seats}")
    return [float(seat_value)] * seats


def live_count(stacks: Sequence[float]) -> int:
    return sum(1 for s in stacks if s > 0.0)


def seat_locked(stacks: Sequence[float], seats: int, hero: int) -> bool:
    """Is ``hero`` *mathematically guaranteed* a seat (uncatchable), regardless of future play?

    Rigorous chip-fraction lock: to push hero below ``seats``-th place, an adversary must create
    ``seats`` rival stacks each strictly larger than hero's. The chips available to rivals total
    ``total - hero``, so this is possible only if ``total - hero > seats * hero`` i.e.
    ``hero < total/(seats+1)``. Therefore ``hero >= total/(seats+1)`` => **locked**. Trivially locked
    when at most ``seats`` players remain (everyone left is paid). A busted hero is never locked.

    (Refinement: subtracting hero's guaranteed future blind/ante exposure before the bubble bursts
    tightens this for an exact real-time call; the pure-chips form is the play-independent theorem.)
    """
    if stacks[hero] <= 0.0:
        return False
    m = live_count(stacks)
    if m <= seats:
        return True
    total = float(sum(s for s in stacks if s > 0.0))
    if total <= 0.0:
        return False
    return float(stacks[hero]) >= total / (seats + 1)


def should_fold_for_seat(stacks: Sequence[float], seats: int, hero: int,
                         *, action_risks_chips: bool = True) -> bool:
    """Once hero's seat is locked, fold anything that risks chips (the locked seat is max equity;
    no ladder upside exists, only downside). Returns True iff ``seat_locked`` and the action puts
    chips at risk. A zero-risk action (check behind, fold) is never forced."""
    if not action_risks_chips:
        return False
    return seat_locked(stacks, seats, hero)


def seat_bubble_factor(stacks: Sequence[float], seats: int, hero: int, villain: int,
                       *, seat_value: float = 1.0) -> float:
    """The pairwise ICM risk premium under the satellite step payoff. Near a seat it is very large
    (often ``inf``: hero gains ~no equity by winning more chips but risks the seat by losing), which
    is exactly the extreme-ICM 'fold to the seat' pressure. Provided so callers can apply the same
    bubble-factor tilt the cash game uses, with the step payoff doing the work."""
    return bubble_factor(stacks, satellite_payouts(seats, seat_value), hero, villain)
