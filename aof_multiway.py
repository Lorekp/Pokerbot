"""Multiway preflop all-in pot terms for AoF advisor.

Chip splits at showdown use the same ranking logic as ``aof_equity._mc_*way_*`` and
``aof_rl.payoff``; this module builds blind-aware pot / breakeven math only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from aof_game_config import AoFGameConfig, default_game_config, net_pot_bb
from aof_ranges import active_seats


def _posted_blind(position: str, stack_bb: float) -> float:
    if position == "SB":
        return min(0.5, stack_bb)
    if position == "BB":
        return min(1.0, stack_bb)
    return 0.0


def _check_equal_stacks(
    stack_bb: float,
    stacks_by_seat: Mapping[str, float] | None,
    game: AoFGameConfig,
) -> None:
    if stacks_by_seat is None:
        return
    values = list(stacks_by_seat.values())
    if not values:
        return
    if all(abs(v - stack_bb) < 1e-9 for v in values):
        return
    if game.equal_stacks_only and not game.allow_side_pots:
        raise NotImplementedError(
            "Unequal stacks require side-pot modeling; set allow_side_pots=True "
            "when implemented. Equal stacks only for now."
        )


def _seat_commitment(
    seat: str,
    *,
    hero: str,
    villains: tuple[str, ...],
    stack_bb: float,
    stacks_by_seat: Mapping[str, float] | None = None,
) -> float:
    eff = stack_bb
    if stacks_by_seat is not None and seat in stacks_by_seat:
        eff = stacks_by_seat[seat]
    if seat == hero or seat in villains:
        return eff
    return _posted_blind(seat, eff)


@dataclass(frozen=True)
class MultiwayCallTerms:
    total_pot_after_call: float
    gross_pot_after_call: float
    hero_call_amount: float
    hero_already_posted: float
    hero_total_invested: float
    ev_fold: float
    breakeven_equity: float
    rake_bb: float = 0.0


def multiway_call_terms(
    hero_pos: str,
    stack_bb: float,
    *,
    villain_positions: tuple[str, ...],
    table_size: int,
    game: AoFGameConfig | None = None,
    stacks_by_seat: Mapping[str, float] | None = None,
) -> MultiwayCallTerms:
    """
    Full-stack AoF: hero and listed villains are all-in for ``stack_bb``;
    other seats contribute blinds only.
    """
    g = game or default_game_config()
    _check_equal_stacks(stack_bb, stacks_by_seat, g)

    seats = active_seats(table_size)
    if hero_pos not in seats:
        raise ValueError(f"Hero position {hero_pos!r} not in {table_size}-max order.")
    for v in villain_positions:
        if v not in seats:
            raise ValueError(f"Villain position {v!r} not in {table_size}-max order.")

    gross_pot = sum(
        _seat_commitment(
            s,
            hero=hero_pos,
            villains=villain_positions,
            stack_bb=stack_bb,
            stacks_by_seat=stacks_by_seat,
        )
        for s in seats
    )
    hero_already = _posted_blind(
        hero_pos, stacks_by_seat.get(hero_pos, stack_bb) if stacks_by_seat else stack_bb
    )
    hero_total = stacks_by_seat.get(hero_pos, stack_bb) if stacks_by_seat else stack_bb
    hero_call = hero_total - hero_already
    ev_fold = -hero_already
    rake_bb = gross_pot - net_pot_bb(gross_pot, g) if g.rake_model else 0.0
    total_pot = net_pot_bb(gross_pot, g)
    breakeven = hero_total / total_pot if total_pot > 0 else 1.0
    return MultiwayCallTerms(
        total_pot_after_call=total_pot,
        gross_pot_after_call=gross_pot,
        hero_call_amount=hero_call,
        hero_already_posted=hero_already,
        hero_total_invested=hero_total,
        ev_fold=ev_fold,
        breakeven_equity=breakeven,
        rake_bb=rake_bb,
    )


@dataclass(frozen=True)
class AllInPotTerms:
    hero: str
    participants: tuple[str, ...]
    folded_seats: tuple[str, ...]
    stack_bb: float
    committed_by_seat: dict[str, float]
    total_pot: float
    gross_pot: float
    hero_total_invested: float
    hero_already_posted: float
    hero_incremental_investment: float
    fold_ev: float
    breakeven_equity: float
    rake_bb: float = 0.0


def all_in_pot_terms(
    hero_pos: str,
    stack_bb: float,
    *,
    all_in_villains: tuple[str, ...],
    table_size: int,
    game: AoFGameConfig | None = None,
    stacks_by_seat: Mapping[str, float] | None = None,
) -> AllInPotTerms:
    """Pot after hero and listed villains are all-in; other seats contribute blinds only."""
    g = game or default_game_config()
    _check_equal_stacks(stack_bb, stacks_by_seat, g)

    seats = active_seats(table_size)
    if hero_pos not in seats:
        raise ValueError(f"Hero position {hero_pos!r} not in {table_size}-max order.")
    participants = (hero_pos,) + tuple(all_in_villains)
    for v in all_in_villains:
        if v not in seats:
            raise ValueError(f"Villain position {v!r} not in {table_size}-max order.")

    committed = {
        s: _seat_commitment(
            s,
            hero=hero_pos,
            villains=all_in_villains,
            stack_bb=stack_bb,
            stacks_by_seat=stacks_by_seat,
        )
        for s in seats
    }
    gross_pot = sum(committed.values())
    hero_stack = committed[hero_pos]
    hero_already = _posted_blind(hero_pos, hero_stack)
    hero_total = hero_stack
    hero_incr = hero_total - hero_already
    fold_ev = -hero_already
    rake_bb = gross_pot - net_pot_bb(gross_pot, g) if g.rake_model else 0.0
    total_pot = net_pot_bb(gross_pot, g)
    breakeven = hero_total / total_pot if total_pot > 0 else 1.0
    folded = tuple(s for s in seats if s not in participants)
    return AllInPotTerms(
        hero=hero_pos,
        participants=participants,
        folded_seats=folded,
        stack_bb=stack_bb,
        committed_by_seat=committed,
        total_pot=total_pot,
        gross_pot=gross_pot,
        hero_total_invested=hero_total,
        hero_already_posted=hero_already,
        hero_incremental_investment=hero_incr,
        fold_ev=fold_ev,
        breakeven_equity=breakeven,
        rake_bb=rake_bb,
    )


def multiway_call_terms_from_villains(
    hero_pos: str,
    stack_bb: float,
    *,
    villain_positions: tuple[str, ...],
    table_size: int,
    game: AoFGameConfig | None = None,
    stacks_by_seat: Mapping[str, float] | None = None,
) -> MultiwayCallTerms:
    """Alias for call paths that list only villains (hero joins by calling)."""
    return multiway_call_terms(
        hero_pos,
        stack_bb,
        villain_positions=villain_positions,
        table_size=table_size,
        game=game,
        stacks_by_seat=stacks_by_seat,
    )
