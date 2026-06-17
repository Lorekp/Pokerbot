"""AIVAT chance term (board-reveal / nature control variate) for the multiway battery.

This is the second AIVAT control variate (Burch, Schmid, Moravcik, Schmid, Bowling, "AIVAT", AAAI 2018).
The decision term (``aivat.decision_correction``) removes variance from the HERO's own action
randomization; this module removes the LARGER half — variance from the public board reveals (flop/turn/
river), which is "nature" / chance in the game tree.

Formula (the chance specialization of the paper's value-correction term ``k_H``, with ``P_a = {chance}``
playing the fixed, common-knowledge deal distribution ``sigma_c``):

    k_C(z) = E_{c ~ sigma_c}[ v(public . c) ]  -  v(public . c_O)

where ``c_O`` is the card(s) actually dealt at this public reveal, ``v`` is a FIXED state-value function
(here the continuation gadget's hero-EV), and the expectation is over the deck complement — all 52 cards
minus every known hole card and every prior board card (a board card can never duplicate ANY hole card).
``chance_correction(expected, realized) = realized - expected = -k_C``; the AIVAT-adjusted per-hand score
subtracts ``+(realized - expected)`` per reveal (see ``AIVATAccumulator.add_hand``).

WHY IT IS UNBIASED (paper Lemma 1 / Theorem 1): the realized board card ``c_O`` is itself drawn from the
SAME distribution ``sigma_c`` that the expectation marginalizes, and ``v`` is fixed (does not depend on the
realized draw). Hence ``E_{c_O ~ sigma_c}[ v(public . c_O) - E_{c ~ sigma_c}[v(public . c)] ] = 0`` — the
correction is mean-zero conditional on the public state, for ANY ``v``. Subtracting it therefore leaves the
estimator's expectation unchanged (``E[adjusted] = E[raw]``) while cancelling card-luck variance whenever
``v`` correlates with the realized outcome. A board reveal is a PUBLIC event (common knowledge), so the
single-node partition is exactly the paper's chance partition and the term composes additively with the
decision term.

Integration: we REPLAY the recorded ``action_history`` of a finished hand on a fresh ``GameState`` (no
engine change, no neural net) to reconstruct the exact public state at every board-reveal node — including
reveals on streets where the hero never acted (e.g. the hero is all-in) — then evaluate ``v`` with the
hero's own continuation-gadget config so ``realized`` and ``expected`` use an identical, deterministic value
function (a clean control variate).

TURN / RIVER (single card): the complement has <=47 cards and is enumerated exactly -> the expectation is
EXACT and the term is provably unbiased.

FLOP (3 cards): C(47,3) ~ 16k is enumerated only when cheap; otherwise we SAMPLE ``flop_samples`` flops
uniformly without replacement from the complement. Sampling keeps the estimator unbiased (the realized
flop is drawn from the same uniform-over-complement distribution), at the cost of a slightly noisier
``expected``. Set ``include_flop=False`` to apply the term ONLY to the enumerable turn+river reveals (a
correct partial reduction); this is the conservative default for the harness.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from pokerbot.engine.actions import Action
from pokerbot.engine.game import GameState, NUM_PLAYERS, Street
from pokerbot.engine.np_engine import to_numpy_state
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig, continuation_leaf_value
from pokerbot.search.state_bridge import game_state_to_env

_STREET_BOARD_LEN = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}


@dataclass
class _Reveal:
    """One public board-reveal node captured during replay (pre-action, board just dealt)."""

    state: GameState           # snapshot at the reveal node (board grown, no new-street betting yet)
    prev_board_len: int        # board length BEFORE this reveal
    n_new: int                 # cards revealed (3 flop, 1 turn/river)


def _replay_reveals(final_state: GameState) -> list[_Reveal]:
    """Replay the finished hand and snapshot the public state at each board-reveal node.

    Reconstructs a fresh ``GameState`` with the SAME button, blinds and (crucially) the same dealt cards
    as ``final_state`` — hole cards and board are copied so the deck is fixed — then applies the recorded
    actions one at a time, snapshotting whenever the board grows. The snapshot is taken AFTER the engine's
    street-advance deal but BEFORE the next action, i.e. the post-reveal pre-action public node.
    """
    import copy

    history = final_state.action_history()
    # Rebuild a clean hand with identical seating/holes/board (deterministic replay; rng unused).
    gs = GameState.new_hand(button_seat=final_state.button_seat, rng=random.Random(0))
    for i in range(NUM_PLAYERS):
        gs.players[i].hole = final_state.players[i].hole
    # The engine deals the board from its own deck on street advance; force it to use the SAME cards by
    # pre-seeding the deck's remaining cards to the realized board (in order), so reveals match exactly.
    realized_board = list(final_state.board)
    _force_board_deck(gs, realized_board)

    reveals: list[_Reveal] = []
    board_len = len(gs.board)
    for seat, action in history:
        # Defensive: only apply actions for the seat the engine expects to act.
        if gs.is_terminal():
            break
        if gs.action_seat != seat:
            # Skip to the recorded actor (folded/all-in seats are auto-skipped by the engine driver).
            for _ in range(NUM_PLAYERS):
                if gs.action_seat == seat or gs.is_terminal():
                    break
                gs.advance_to_next_actor()
        if gs.is_terminal() or gs.action_seat != seat:
            break
        gs.apply_action(Action(action.action_type, action.amount_bb), seat)
        if len(gs.board) > board_len:
            n_new = len(gs.board) - board_len
            reveals.append(_Reveal(state=copy.deepcopy(gs), prev_board_len=board_len, n_new=n_new))
            board_len = len(gs.board)
        if not gs.is_terminal():
            gs.advance_to_next_actor()
    return reveals


def _force_board_deck(gs: GameState, board) -> None:
    """Make ``gs``'s deck deal exactly ``board`` (in order) on the upcoming street advances.

    The hole cards are already set (not drawn from the deck during replay), so the next cards the engine
    deals for flop/turn/river must be the realized board cards. We splice them to the FRONT of the deck's
    undealt tail. Robust to the engine having already advanced its internal deck index for the blinds
    (blinds commit chips, they do not draw cards)."""
    deck = gs._deck
    if deck is None:
        return
    from pokerbot.engine.cards import Card, Rank, Suit

    def _to_card(c):
        if isinstance(c, Card):
            return c
        idx = int(c)
        return Card(Rank((idx // 4) + 2), Suit(idx % 4))

    want = [_to_card(c) for c in board]
    # Rebuild the undealt tail: realized board first, then every other card (so showdown never errors).
    want_idx = {c.index for c in want}
    rest = [c for c in deck._cards if c.index not in want_idx]
    deck._cards = list(deck._cards[: deck._idx]) + want + rest
    # Keep _idx where it is; the next ``deal`` returns ``want`` in order.


def _complement(holes, board_prefix) -> list[int]:
    """Deck complement: all 52 minus every known hole card and every prior board card (0..51 indices)."""
    known = set()
    for h in holes:
        if not h:
            continue
        for c in h:
            known.add(_card_idx(c))
    for c in board_prefix:
        known.add(_card_idx(c))
    return [c for c in range(52) if c not in known]


def _card_idx(c) -> int:
    return c.index if hasattr(c, "index") else int(c)


def _env_with_board(reveal: _Reveal, hero_seat: int, new_cards: list[int]):
    """Build the post-reveal pre-action np_engine env with the just-revealed slot(s) set to ``new_cards``.

    Everything except the revealed board card(s) is held fixed (holes, prior board, stacks, pot, street,
    action_seat) — those do not depend on the revealed card because no new-street betting has occurred yet.
    """
    env = to_numpy_state(game_state_to_env(reveal.state, hero_seat))
    board = np.asarray(env.board).copy()
    for k, c in enumerate(new_cards):
        board[reveal.prev_board_len + k] = int(c)
    return env._replace(board=board)


def _value(env, hero_seat: int, adapter, leaf_cfg: ContinuationLeafConfig, base_seed: int) -> float:
    """The FIXED state-value function v(public state) = continuation-gadget hero-EV at ``env``.

    Same gadget config for ``realized`` and ``expected`` (deterministic given ``base_seed``) so the chance
    correction is a clean control variate. ``base_seed`` is shared across the realized card and ALL
    candidate cards of one reveal, so the only thing varying inside the expectation is the board card."""
    cfg = ContinuationLeafConfig(
        select=leaf_cfg.select,
        rollout_samples=leaf_cfg.rollout_samples,
        gadget_iters=leaf_cfg.gadget_iters,
        max_playout_depth=leaf_cfg.max_playout_depth,
        base_seed=base_seed,
        resample_board=leaf_cfg.resample_board,
    )
    return continuation_leaf_value(env, hero_seat, adapter, cfg)


def chance_corrections_for_hand(
    final_state: GameState,
    hero_seat: int,
    adapter,
    leaf_cfg: ContinuationLeafConfig,
    *,
    include_flop: bool = False,
    flop_samples: int = 24,
    base_seed: int = 0,
) -> list[float]:
    """Per-reveal AIVAT chance corrections ``c_C = realized - expected`` for one finished hand.

    Returns one float per applicable board reveal along the hand. The AIVAT-adjusted hand score subtracts
    ``sum`` of these (see ``AIVATAccumulator``). Turn/river are enumerated exactly (unbiased); the flop is
    sampled and applied only when ``include_flop=True``.
    """
    if hero_seat < 0 or final_state.players[hero_seat].hole is None:
        return []
    reveals = _replay_reveals(final_state)
    if not reveals:
        return []
    holes = [p.hole for p in final_state.players]
    out: list[float] = []

    for ri, rev in enumerate(reveals):
        n_new = rev.n_new
        if n_new == 3 and not include_flop:
            continue
        board_prefix = list(rev.state.board[: rev.prev_board_len])
        complement = _complement(holes, board_prefix)
        # Cards actually revealed at this node.
        realized_cards = [_card_idx(c) for c in rev.state.board[rev.prev_board_len : rev.prev_board_len + n_new]]
        # Shared per-reveal value seed: realized + every candidate use the SAME runout seed so the matrix
        # game / rollout integrates an identical board tail, isolating the just-revealed card.
        seed = base_seed + 100003 * (ri + 1)

        realized_env = _env_with_board(rev, hero_seat, realized_cards)
        realized = _value(realized_env, hero_seat, adapter, leaf_cfg, seed)

        if n_new == 1:
            # EXACT enumeration over the <=47-card complement -> unbiased expectation.
            vals = [
                _value(_env_with_board(rev, hero_seat, [c]), hero_seat, adapter, leaf_cfg, seed)
                for c in complement
            ]
            expected = float(np.mean(vals)) if vals else realized
        else:
            # Flop: sample flop_samples 3-card combos uniformly without replacement (unbiased estimator
            # of E[v] since the realized flop is itself a uniform draw from the same complement).
            rng = random.Random(seed ^ 0xF10F)
            n_enum = len(complement) * (len(complement) - 1) * (len(complement) - 2) // 6
            samples: list[list[int]] = []
            if n_enum <= flop_samples:
                # Cheap enough -> exact enumeration.
                from itertools import combinations

                samples = [list(t) for t in combinations(complement, 3)]
            else:
                seen = set()
                while len(samples) < flop_samples:
                    trio = tuple(sorted(rng.sample(complement, 3)))
                    if trio not in seen:
                        seen.add(trio)
                        samples.append(list(trio))
            vals = [
                _value(_env_with_board(rev, hero_seat, trio), hero_seat, adapter, leaf_cfg, seed)
                for trio in samples
            ]
            expected = float(np.mean(vals)) if vals else realized

        out.append(float(realized) - float(expected))  # c_C = realized - expected
    return out
