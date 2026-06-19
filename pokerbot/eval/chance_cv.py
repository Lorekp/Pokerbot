"""Chance (board-reveal) control variate for the seat-rotated absolute-EV estimator.

This supplies AIVAT's CHANCE term, which the pluribus search path never emitted (only the per-decision
control variate is — see ``pokerbot/eval/aivat.py`` and the search's ``pop_decisions_raw``). Board-runout
luck is the dominant residual variance once seat rotation cancels the hole-card allocation and the all-in
EV settle integrates out the runout on all-in pots; this corrects the rest as a PURE POST-PROCESSING
control variate on the realized hand trajectory, touching no search-core code.

At each board reveal the hero is still LIVE-AND-BETTING for, subtract

    c_C = pot_pre * (equity(realized board) - mean_c equity(board with candidate card c))

over the EXACT candidate set the engine dealt from (the realized card re-pooled with the remaining deck,
mirroring ``_allin_ev_settle``'s re-pooling). ``E[c_C] = 0`` over the uniform deal for ANY ``pot_pre`` that
is fixed BEFORE the reveal, so the adjusted score is UNBIASED for any baseline; the variance reduction
comes from (a) scaling by the PRE-reveal pot — never the realized final pot, which would smuggle realized
future information into the counterfactual and bias the mean — and (b) full-information showdown equity.

Two correctness gates:
  * ``actors_remaining()`` — only correct a reveal that is followed by real betting. A reveal during an
    all-in runout is integrated by ``_allin_ev_settle`` instead (it never reaches the engine's normal
    apply-action deal path under that settle, but we gate defensively): correcting it would REMOVE luck
    that is not in the realized score and INFLATE variance.
  * hero live with hole cards — if the hero folded, its chips are board-independent (u is constant) so the
    correction is identically 0; skip.

Turn and river reveals are enumerated exactly (cheap: <= ~37*37 showdown evals). The flop and preflop
reveals integrate a far larger runout (a flop candidate needs the turn+river integrated, ~10k*666 evals)
and are left uncorrected here — their marginal board luck is the smaller share once all-in pots are
EV-settled, and they can be added later by unbiased MC sampling of the candidate/runout if needed.
"""

from __future__ import annotations

from itertools import combinations

try:  # same fast 7-card evaluator + index convention as _allin_ev_settle (LOWER rank wins)
    from phevaluator.evaluator import evaluate_cards as _phe
except ImportError:  # noqa: BLE001
    _phe = None


def _hero_share(hero_idx: int, hole_ids, board_ids) -> float:
    """Hero's pot share at showdown on a COMPLETE (5-card) board: 1/len(winners) if hero wins/ties, else 0."""
    ranks = [_phe(h0, h1, *board_ids) for (h0, h1) in hole_ids]
    best = min(ranks)
    winners = [i for i, r in enumerate(ranks) if r == best]
    return (1.0 / len(winners)) if hero_idx in winners else 0.0


def _hero_equity(hero_idx: int, hole_ids, seen_ids, runout_pool, need: int) -> float:
    """Hero's expected showdown share given a partial board ``seen_ids``, integrating ``need`` more cards
    uniformly from ``runout_pool`` (exact enumeration)."""
    if need <= 0:
        return _hero_share(hero_idx, hole_ids, seen_ids)
    total = 0.0
    n = 0
    for combo in combinations(runout_pool, need):
        total += _hero_share(hero_idx, hole_ids, seen_ids + list(combo))
        n += 1
    return total / n if n else 0.0


class ChanceCV:
    """Accumulates the per-reveal chance control variates for ONE played hand from the hero's seat.

    Wire ``on_reveal`` as ``play_hand(..., chance_sink=cv.on_reveal)``; after the hand, ``total()`` is the
    sum of corrections to subtract from the hero's raw chip delta. A fresh instance per (deck, rotation)."""

    def __init__(self, hero_seat: int) -> None:
        self.hero_seat = int(hero_seat)
        self.corrections: list[float] = []

    def on_reveal(self, state, prev_board_len: int) -> None:
        if _phe is None:
            return
        # Only correct a reveal followed by real betting; an all-in runout is integrated by _allin_ev_settle.
        if not state.actors_remaining():
            return
        hero = state.players[self.hero_seat]
        if hero.folded or not hero.hole:
            return  # hero's chips are board-independent from here -> u constant -> correction 0
        new_count = len(state.board) - int(prev_board_len)
        if new_count != 1:
            return  # exact handling for single-card reveals (turn, river); flop/preflop deferred
        contenders = [p for p in state.players if not p.folded and p.hole]
        if len(contenders) < 2 or hero not in contenders:
            return
        hero_idx = contenders.index(hero)
        hole_ids = [(p.hole[0].index, p.hole[1].index) for p in contenders]
        board_before = [c.index for c in state.board[:prev_board_len]]
        realized_card = state.board[prev_board_len].index
        deck = state._deck
        # Candidate set the engine dealt this card from = realized card re-pooled with the undealt deck.
        pool = [realized_card] + [c.index for c in deck._cards[deck._idx:]]
        need_after = 5 - (len(board_before) + 1)  # turn -> 1 (integrate river); river -> 0
        pot_pre = float(state.pot_bb)

        def _eq(card: int) -> float:
            seen = board_before + [card]
            ro_pool = [x for x in pool if x != card]
            return _hero_equity(hero_idx, hole_ids, seen, ro_pool, need_after)

        vals = [_eq(c) for c in pool]
        mean_eq = sum(vals) / len(vals)
        realized_eq = vals[0]  # pool[0] is the realized card
        self.corrections.append(pot_pre * (realized_eq - mean_eq))

    def total(self) -> float:
        return float(sum(self.corrections))
