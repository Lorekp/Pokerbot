"""Multi-table tournament SIMULATOR — plays a whole MTT out end-to-end so the multi-table engine
(:mod:`pokerbot.tournament.mtt_field`) can be exercised and a strategy's $ROI measured, the way the
single-table :mod:`pokerbot.tournament.sim` does for a final-table SNG.

A turbo MTT is push/fold-dominated, so the hand model is an **all-in confrontation**: at each step a random
live table is picked, its shortest stack (the player under the most push/fold pressure) faces a jam decision
keyed on a drawn hand strength; one caller may look it up; the all-in is resolved by equity (higher strength
wins, ties chop) with chip-conservation. Busted players are recorded in finish order. The field is
**re-balanced** between confrontations via the standard rules (:func:`mtt_field.balance_moves`). Play runs to a
single survivor; the hero's finish maps to the prize ladder.

A ``Strategy`` is just two thresholds (jam, call) as a function of the player's **field bubble factor** — so a
chip-EV player ignores it (constant thresholds) while an ICM-aware player tightens as the global pay jump
nears. This makes the field-ICM lever (:func:`mtt_field.field_bubble_factor`) measurable: does ICM-aware
survival beat chip-EV $ROI against a given field? RNG is injected (deterministic, no global ``random``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import random

from pokerbot.tournament.mtt_field import (
    MttField,
    MttTable,
    apply_moves,
    balance_moves,
    field_bubble_factor,
    needs_balancing,
)

# A strategy maps (bubble_factor) -> (jam_threshold, call_threshold) in [0,1] hand-strength quantile space.
# Higher threshold = tighter. Chip-EV: constant. ICM-aware: rises with bubble_factor (tighten near the money).
Strategy = Callable[[float], "tuple[float, float]"]


def chip_ev_strategy(jam: float = 0.45, call: float = 0.60) -> Strategy:
    """Bubble-factor-blind: fixed jam/call thresholds (the ICM-naive field)."""
    return lambda _bf: (jam, call)


def icm_aware_strategy(jam: float = 0.45, call: float = 0.60, sens: float = 0.12) -> Strategy:
    """Tightens both thresholds as the field bubble factor exceeds 1 (survival worth more near a pay jump),
    capped at near-1.0. ``sens`` is the tightening per unit of (bubble_factor - 1)."""
    def strat(bf: float) -> tuple[float, float]:
        tighten = max(0.0, bf - 1.0) * sens
        return (min(0.97, jam + tighten), min(0.99, call + tighten))
    return strat


@dataclass
class MttResult:
    hero_finish: int          # 1 == won, 2 == runner-up, ...
    hero_payout: float
    players: int


_RANKS = "AKQJT98765432"
_SUITS = "shdc"


class _RealEquity:
    """Live-fidelity confrontation resolver: real hole cards mapped to the 169 HU buckets, decisions by REAL
    preflop strength quantile, and all-ins resolved by the ACTUAL matchup equity (data/aof/cfr_base/
    hu_eq_matrix.npz, eq[i,j] = P(bucket i beats bucket j)) — not a uniform-strength draw. Loaded once."""

    _INST = None

    def __init__(self):
        import numpy as np
        from pokerbot.tournament.pushfold_icm import _hu_grid, hand_to_label

        eq, labels, cw = _hu_grid()
        self.eq = np.asarray(eq, dtype=float)
        self.cw = np.asarray(cw, dtype=float)
        self._idx = {lab: i for i, lab in enumerate(labels)}
        self._label = hand_to_label
        # each bucket's strength = equity vs a combo-weighted random hand
        self.bucket_strength = (self.eq @ self.cw) / self.cw.sum()
        order = np.argsort(self.bucket_strength)
        self._sorted_strength = self.bucket_strength[order]
        self._cum_w = np.cumsum(self.cw[order]) / self.cw.sum()

    @classmethod
    def get(cls):
        if cls._INST is None:
            cls._INST = cls()
        return cls._INST

    def strength_cutoff(self, quantile: float) -> float:
        """The strength such that ~``quantile`` of hands (combo-weighted) are weaker — the jam/call cutoff."""
        import numpy as np
        i = int(np.searchsorted(self._cum_w, max(0.0, min(1.0, quantile))))
        return float(self._sorted_strength[min(i, len(self._sorted_strength) - 1)])

    def deal_two(self, rng) -> tuple[int, int]:
        """Deal 4 distinct cards -> (jammer bucket, caller bucket)."""
        deck = [r + s for r in _RANKS for s in _SUITS]
        cards = rng.sample(deck, 4)
        bi = self._idx[self._label(cards[0] + cards[1])]
        bj = self._idx[self._label(cards[2] + cards[3])]
        return bi, bj

    def win_prob(self, bi: int, bj: int) -> float:
        """P(bucket bi beats bucket bj) heads-up (ties counted as half by the grid)."""
        return float(self.eq[bi, bj])


def _build_field(n_players: int, seats_per_table: int, start_bb: float, payouts, rng) -> tuple[MttField, int]:
    """Seat ``n_players`` across tables of ``seats_per_table`` (last table short); return (field, hero_flat_idx).
    The hero is seated at table 0 seat 0."""
    tables = []
    p = n_players
    tid = 0
    while p > 0:
        k = min(seats_per_table, p)
        tables.append(MttTable(tid, [start_bb] * k + [0.0] * (seats_per_table - k)))
        p -= k
        tid += 1
    return MttField(tables, list(payouts), entrants=n_players), 0


def _live_players(field: MttField):
    """Yield (table_id, seat, stack) for every live player."""
    for t in field.tables:
        for seat, s in enumerate(t.stacks_bb):
            if s > 0.0:
                yield t.table_id, seat, s


def simulate_mtt(n_players: int, payouts, hero_strategy: Strategy, villain_strategy: Strategy, *,
                 seats_per_table: int = 9, start_bb: float = 20.0, rng: random.Random | None = None,
                 bf_sims: int = 400, realistic: bool = False) -> MttResult:
    """Play ONE MTT to completion; return the hero's finish + payout. Hero is player id 0 (table 0 seat 0).

    Players are tracked by IDENTITY (``pid[(table_id, seat)] = player_id``) so the hero is followed correctly
    through table breaks/balances — a balance move carries the id with the stack.

    ``realistic``: when True, each confrontation deals REAL hole cards, decides by REAL preflop strength
    quantile, and resolves the all-in by the ACTUAL HU matchup equity (vs the default uniform-strength
    abstraction). Threshold semantics are preserved (jam_thr is the fold-below quantile) so a strategy means
    the same thing in both modes; only the hand realism changes."""
    rng = rng or random.Random(0)
    resolver = _RealEquity.get() if realistic else None
    field, _ = _build_field(n_players, seats_per_table, start_bb, payouts, rng)
    HERO = 0
    pid: dict[tuple[int, int], int] = {}
    nxt = 0
    for t in field.tables:                        # id 0 lands at table0 seat0 (filled first) == hero
        for seat, s in enumerate(t.stacks_bb):
            if s > 0.0:
                pid[(t.table_id, seat)] = nxt
                nxt += 1
    finish_rank = n_players
    hero_finish = None

    def hero_loc():
        for k, v in pid.items():
            if v == HERO:
                return k
        return None

    from pokerbot.tournament.mtt_field import _flat_hero_index

    guard = 0
    while field.players_left > 1 and guard < 200000:
        guard += 1
        live_tables = [t for t in field.tables if t.num_players >= 2]
        if not live_tables:
            break
        table = live_tables[rng.randrange(len(live_tables))]
        seats = [(seat, s) for seat, s in enumerate(table.stacks_bb) if s > 0.0]
        # shortest stack is the push/fold actor, but tie-break RANDOMLY (else seat 0 — the hero — is always
        # picked on equal stacks, biasing the hero into extra all-ins; the symmetry test catches this).
        min_stack = min(s for _, s in seats)
        shortest = [seat for seat, s in seats if s <= min_stack + 1e-9]
        jam_seat = shortest[rng.randrange(len(shortest))]
        jam_stack = table.stacks_bb[jam_seat]
        others = [(seat, s) for seat, s in seats if seat != jam_seat]
        call_seat, call_stack = others[rng.randrange(len(others))]

        is_hero_jam = pid.get((table.table_id, jam_seat)) == HERO
        is_hero_call = pid.get((table.table_id, call_seat)) == HERO

        def _bf(seat, other_seat):
            idx = _flat_hero_index(field.tables, table.table_id, seat)
            vidx = _flat_hero_index(field.tables, table.table_id, other_seat)
            if idx is None or vidx is None:
                return 1.0
            return field_bubble_factor(field.all_live_stacks, field.payouts, idx, vidx,
                                       n_sims=bf_sims, rng=rng)

        jam_bf = _bf(jam_seat, call_seat) if is_hero_jam else 1.0
        call_bf = _bf(call_seat, jam_seat) if is_hero_call else 1.0
        jam_thr, _ = (hero_strategy if is_hero_jam else villain_strategy)(jam_bf)
        _, call_thr = (hero_strategy if is_hero_call else villain_strategy)(call_bf)

        if resolver is None:
            jam_strength = rng.random()
            if jam_strength < jam_thr:
                continue                          # actor folds
            call_strength = rng.random()
            if call_strength < call_thr:
                continue                          # caller folds
            jam_wins = jam_strength > call_strength
        else:
            # REAL cards: decide by real preflop strength quantile, resolve by the ACTUAL matchup equity
            bi, bj = resolver.deal_two(rng)
            if resolver.bucket_strength[bi] < resolver.strength_cutoff(jam_thr):
                continue                          # jammer folds (hand below its jam range)
            if resolver.bucket_strength[bj] < resolver.strength_cutoff(call_thr):
                continue                          # caller folds (hand below its call range)
            jam_wins = rng.random() < resolver.win_prob(bi, bj)

        risked = min(jam_stack, call_stack)
        winner_seat, loser_seat = (jam_seat, call_seat) if jam_wins else (call_seat, jam_seat)
        table.stacks_bb[winner_seat] += risked
        table.stacks_bb[loser_seat] -= risked
        if table.stacks_bb[loser_seat] <= 1e-9:
            table.stacks_bb[loser_seat] = 0.0
            loser_id = pid.pop((table.table_id, loser_seat), None)
            if loser_id == HERO:
                hero_finish = finish_rank
            finish_rank -= 1
            if needs_balancing(field, seats_per_table=seats_per_table):
                moves = balance_moves(field, seats_per_table=seats_per_table)
                field = apply_moves(field, moves)
                for m in moves:                   # carry identities with the moved stacks
                    mover = pid.pop((m.from_table, m.from_seat), None)
                    if mover is not None:
                        pid[(m.to_table, m.to_seat)] = mover

    if hero_finish is None:                       # hero survived to the win (rank 1)
        hero_finish = 1
    pay = list(payouts) + [0.0] * max(0, n_players - len(payouts))
    return MttResult(hero_finish, pay[hero_finish - 1], n_players)


def measure_roi(n_players: int, payouts, hero_strategy: Strategy, villain_strategy: Strategy, *,
                n_sims: int = 200, seats_per_table: int = 9, start_bb: float = 20.0,
                rng: random.Random | None = None, bf_sims: int = 200, realistic: bool = False) -> dict:
    """Hero $ROI over ``n_sims`` MTTs (mean payout vs the fair buy-in share). buy_in_share = prize_pool /
    entrants; ROI = (mean_payout - buy_in_share) / buy_in_share. ``realistic`` -> real-card/real-equity hands."""
    rng = rng or random.Random(0)
    pays, finishes = [], []
    for _ in range(n_sims):
        r = simulate_mtt(n_players, payouts, hero_strategy, villain_strategy,
                         seats_per_table=seats_per_table, start_bb=start_bb, rng=rng, bf_sims=bf_sims,
                         realistic=realistic)
        pays.append(r.hero_payout)
        finishes.append(r.hero_finish)
    mean_pay = sum(pays) / len(pays)
    buy_in_share = sum(payouts) / n_players
    return {
        "mean_payout": mean_pay,
        "buy_in_share": buy_in_share,
        "roi": (mean_pay - buy_in_share) / buy_in_share if buy_in_share > 0 else 0.0,
        "itm_rate": sum(1 for p in pays if p > 0) / len(pays),
        "mean_finish": sum(finishes) / len(finishes),
        "n_sims": n_sims,
    }
