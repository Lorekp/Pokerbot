"""Multi-table tournament (MTT) FIELD model — the load-bearing piece the single-table tournament layer
lacks: a hero's $-equity in an MTT depends on the stack distribution across **all** tables, not just its own.

The existing :func:`pokerbot.tournament.icm.icm_equity` computes exact Malmuth-Harville ICM, but its
recursion is exponential in the number of players — fine at a final table (<=~9), intractable for a 60/200/
1000-runner field. This module adds:

  * :class:`MttTable` / :class:`MttField` — the field state (every table's stacks + the prize ladder).
  * :func:`field_icm_equity` — a **Monte-Carlo Malmuth-Harville** field ICM that scales to ANY field size
    (samples finish orders, ``P(finish next) proportional to stack``), returning the hero's $-equity over the
    whole remaining field. It is the SAME model as the exact primitive (cross-validated against it at small N
    in the tests) but O(sims * paid * N) instead of exponential.
  * :func:`field_bubble_factor` — the field-wide risk premium (busting cost / doubling gain) the serve-time
    ICM overlay needs to bend a chip-EV decision near a global pay jump, computed off the FULL field.
  * table balancing (:func:`needs_balancing` / :func:`balance_moves`) — the standard MTT seat-flow rules
    (break the shortest table to fill empty seats; otherwise move one player big->small when tables differ
    by >=2), so the field stays legal as players bust.

Design: only the hero's equity (and the pairwise bubble factor) are needed at serve time, so the MC samples
only the top ``len(payouts)`` finishers and stops once the hero is placed — the rest finish out of the money
(0). RNG is injected for determinism/testing (no global ``random`` reliance)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from pokerbot.tournament.icm import icm_equity


@dataclass
class MttTable:
    """One table in the field. ``stacks_bb`` is per-seat; a seat with ``<= 0`` chips is empty/busted."""

    table_id: int
    stacks_bb: list[float] = field(default_factory=list)
    button: int = 0

    @property
    def live_stacks(self) -> list[float]:
        return [float(s) for s in self.stacks_bb if s > 0.0]

    @property
    def num_players(self) -> int:
        return sum(1 for s in self.stacks_bb if s > 0.0)

    @property
    def empty_seats(self) -> int:
        return sum(1 for s in self.stacks_bb if s <= 0.0)


@dataclass
class MttField:
    """The whole tournament: every table + the best-first prize ladder + the original entrant count."""

    tables: list[MttTable]
    payouts: list[float]
    entrants: int = 0

    @property
    def all_live_stacks(self) -> list[float]:
        return [s for t in self.tables for s in t.live_stacks]

    @property
    def players_left(self) -> int:
        return sum(t.num_players for t in self.tables)

    @property
    def places_paid(self) -> int:
        return sum(1 for p in self.payouts if p > 0.0)

    @property
    def in_the_money(self) -> bool:
        return self.players_left <= self.places_paid

    @property
    def on_the_bubble(self) -> bool:
        """One bust from the money (the highest-pressure ICM spot)."""
        return self.players_left == self.places_paid + 1

    def hero_field_equity(self, hero_table: int, hero_seat: int, *, n_sims: int = 4000,
                          rng: random.Random | None = None) -> float:
        """The hero's $-equity over the WHOLE field (not just its table)."""
        hero_idx = _flat_hero_index(self.tables, hero_table, hero_seat)
        if hero_idx is None:
            return 0.0
        return field_icm_equity(self.all_live_stacks, self.payouts, hero_idx, n_sims=n_sims, rng=rng)


def _flat_hero_index(tables: list[MttTable], hero_table: int, hero_seat: int) -> int | None:
    """Index of (hero_table, hero_seat) within the flattened list of LIVE stacks, or None if not live."""
    idx = 0
    for t in tables:
        for seat, s in enumerate(t.stacks_bb):
            if s > 0.0:
                if t.table_id == hero_table and seat == hero_seat:
                    return idx
                idx += 1
    return None


def field_icm_equity(stacks, payouts, hero_idx: int, *, n_sims: int = 4000,
                     rng: random.Random | None = None) -> float:
    """Monte-Carlo Malmuth-Harville $-equity of ``stacks[hero_idx]`` over the whole field.

    Samples ``n_sims`` finish orders: the next finisher among the remaining players is drawn with
    probability proportional to chip stack (the Malmuth-Harville assumption). Only the top
    ``places_paid`` finishers are sampled per trial (the rest are out of the money -> $0), and a trial
    stops as soon as the hero is placed. Returns the average prize the hero collects. Converges to
    :func:`pokerbot.tournament.icm.icm_equity` (exact) as ``n_sims -> inf`` (locked in the tests)."""
    s = np.asarray([max(0.0, float(x)) for x in stacks], dtype=float)
    n = s.size
    if n == 0 or hero_idx < 0 or hero_idx >= n or s[hero_idx] <= 0.0:
        return 0.0
    pay = list(payouts) + [0.0] * max(0, n - len(payouts))
    paid = min(n, sum(1 for p in payouts if p > 0.0))
    if paid == 0:
        return 0.0
    rng = rng or random.Random(0)
    total = 0.0
    for _ in range(n_sims):
        remaining = s.copy()
        live_sum = float(remaining.sum())
        for place in range(paid):                      # place 0 == 1st prize
            r = rng.random() * live_sum
            # draw the finisher at this place proportional to remaining stack
            c = 0.0
            winner = -1
            for i in range(n):
                if remaining[i] > 0.0:
                    c += remaining[i]
                    if r <= c:
                        winner = i
                        break
            if winner < 0:                             # numerical tail -> last live player
                winner = int(np.argmax(remaining))
            if winner == hero_idx:
                total += pay[place]
                break
            live_sum -= remaining[winner]
            remaining[winner] = 0.0
    return total / n_sims


def field_bubble_factor(stacks, payouts, hero_idx: int, villain_idx: int, *, n_sims: int = 6000,
                        rng: random.Random | None = None) -> float:
    """Field-wide pairwise ICM risk premium for the hero stacking off vs the villain for the smaller stack.

    ``BF = (equity risked by busting) / (equity gained by winning)`` computed over the FULL field via MC
    Malmuth-Harville. ``1.0`` == chips are $-linear here (deep / far from the money -> chip-EV); ``> 1`` ==
    busting costs more than doubling gains (tighten near a global pay jump). Mirrors
    :func:`pokerbot.tournament.icm.bubble_factor` but on the multi-table field rather than one table."""
    s = [max(0.0, float(x)) for x in stacks]
    if hero_idx == villain_idx or s[hero_idx] <= 0.0 or s[villain_idx] <= 0.0:
        return 1.0
    risked = min(s[hero_idx], s[villain_idx])
    base = field_icm_equity(s, payouts, hero_idx, n_sims=n_sims, rng=_clone(rng, 1))

    lose = list(s)
    lose[hero_idx] -= risked
    lose[villain_idx] += risked
    eq_lose = field_icm_equity(lose, payouts, hero_idx, n_sims=n_sims, rng=_clone(rng, 2))

    win = list(s)
    win[hero_idx] += risked
    win[villain_idx] -= risked
    eq_win = field_icm_equity(win, payouts, hero_idx, n_sims=n_sims, rng=_clone(rng, 3))

    downside = base - eq_lose
    upside = eq_win - base
    if upside <= 1e-12:
        return 1.0
    return max(0.0, downside / upside)


def _clone(rng: random.Random | None, salt: int) -> random.Random:
    """A deterministic child RNG so each MC call is reproducible but independent across the BF legs."""
    seed = (rng.randrange(1 << 30) if rng is not None else 0) * 8 + salt
    return random.Random(seed)


# --- table balancing (standard MTT seat-flow) ------------------------------------------------------------

@dataclass(frozen=True)
class SeatMove:
    """Move the player at ``(from_table, from_seat)`` to ``(to_table, to_seat)`` (an empty seat)."""

    from_table: int
    from_seat: int
    to_table: int
    to_seat: int
    stack_bb: float


def needs_balancing(field_state: MttField, *, seats_per_table: int = 9) -> bool:
    """True if the field violates the standard balance rule: either a table can be BROKEN (the field fits in
    one fewer table) or two tables differ by >= 2 players."""
    sizes = [t.num_players for t in field_state.tables if t.num_players > 0]
    if len(sizes) <= 1:
        return False
    if field_state.players_left <= (len(sizes) - 1) * seats_per_table:
        return True
    return (max(sizes) - min(sizes)) >= 2


def balance_moves(field_state: MttField, *, seats_per_table: int = 9) -> list[SeatMove]:
    """Compute the seat moves that re-balance the field (standard rules):

    * **Break** the shortest table when the field fits in one fewer table — move all its players into the
      empty seats of the remaining tables (most-empty table first).
    * Otherwise, while two live tables differ by >= 2 players, move one player from the largest to an empty
      seat at the smallest.

    Returns the list of moves (does NOT mutate ``field_state`` — apply with :func:`apply_moves`)."""
    live = [t for t in field_state.tables if t.num_players > 0]
    if len(live) <= 1:
        return []
    moves: list[SeatMove] = []

    if field_state.players_left <= (len(live) - 1) * seats_per_table:
        # break the shortest table
        shortest = min(live, key=lambda t: t.num_players)
        targets = sorted([t for t in live if t.table_id != shortest.table_id],
                         key=lambda t: -t.empty_seats)
        movers = [(seat, s) for seat, s in enumerate(shortest.stacks_bb) if s > 0.0]
        # round-robin the busted-table players into empty seats
        empties = [(t, seat) for t in targets for seat, s in enumerate(t.stacks_bb) if s <= 0.0]
        for (from_seat, stack), (to_t, to_seat) in zip(movers, empties):
            moves.append(SeatMove(shortest.table_id, from_seat, to_t.table_id, to_seat, stack))
        return moves

    # incremental balance: biggest -> smallest until within 1
    sizes = {t.table_id: t.num_players for t in live}
    work = {t.table_id: list(t.stacks_bb) for t in live}
    by_id = {t.table_id: t for t in live}
    guard = 0
    while max(sizes.values()) - min(sizes.values()) >= 2 and guard < len(live) * seats_per_table:
        guard += 1
        big_id = max(sizes, key=lambda k: sizes[k])
        small_id = min(sizes, key=lambda k: sizes[k])
        from_seat = next((i for i, s in enumerate(work[big_id]) if s > 0.0), None)
        to_seat = next((i for i, s in enumerate(work[small_id]) if s <= 0.0), None)
        if from_seat is None or to_seat is None:
            break
        stack = work[big_id][from_seat]
        moves.append(SeatMove(big_id, from_seat, small_id, to_seat, stack))
        work[big_id][from_seat] = 0.0
        work[small_id][to_seat] = stack
        sizes[big_id] -= 1
        sizes[small_id] += 1
    return moves


def apply_moves(field_state: MttField, moves: list[SeatMove]) -> MttField:
    """Return a new :class:`MttField` with ``moves`` applied (the source seats emptied, targets filled)."""
    tables = {t.table_id: MttTable(t.table_id, list(t.stacks_bb), t.button) for t in field_state.tables}
    for m in moves:
        tables[m.from_table].stacks_bb[m.from_seat] = 0.0
        tables[m.to_table].stacks_bb[m.to_seat] = m.stack_bb
    return MttField(list(tables.values()), list(field_state.payouts), field_state.entrants)
