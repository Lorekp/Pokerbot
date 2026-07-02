"""Shared seam utilities for external-bot adapters.

Every adapter imports from here:
  - GenericView      : info-hygienic read-only view of (env, seat)
  - card helpers     : id <-> our 'As' style AND PyPokerEngine 'SA' (suit-first) style
  - finalize_action  : fold-when-free coercion, raise-TO bucketing, degradation chain
  - stable_seed      : CRN-deterministic per-decision seed
  - safe_policy      : try/except wrapper that never raises, logs failures, counts errors
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from pokerbot.engine.actions import BET_SIZE_BUCKETS, bucket_index_for_bet, pseudo_harmonic_bucket
from pokerbot.engine.np_engine import (
    ALL_IN,
    BET_BUCKET_START,
    CALL,
    CHECK,
    FOLD,
    NUM_ACTIONS,
    legal_action_mask_np,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level failure counter (incremented by safe_policy on any exception)
# ---------------------------------------------------------------------------
_failure_count: int = 0

# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------

# Rank characters index: id // 4 + 2 => 2..14
_RANK_CHARS = "23456789TJQKA"  # index 0 = rank-2, index 12 = rank-A (Ace=14)
_SUIT_CHARS_OURS = "cdhs"       # our style: id % 4 => c d h s
_SUIT_CHARS_PPE = "CDHS"        # PyPokerEngine suit-first: C D H S


def card_id_to_str(card_id: int) -> str:
    """Convert card id (0..51) to our format: rank+suit, e.g. 'As', 'Td', '2c'."""
    rank_idx = int(card_id) // 4   # 0..12
    suit_idx = int(card_id) % 4    # 0..3
    return _RANK_CHARS[rank_idx] + _SUIT_CHARS_OURS[suit_idx]


def card_id_to_ppe(card_id: int) -> str:
    """Convert card id (0..51) to PyPokerEngine suit-first format: suit+rank, e.g. 'SA', 'DT', 'C2'."""
    rank_idx = int(card_id) // 4
    suit_idx = int(card_id) % 4
    return _SUIT_CHARS_PPE[suit_idx] + _RANK_CHARS[rank_idx]


def card_str_to_id(s: str) -> int:
    """Convert our 'As' / 'Td' string to card id."""
    rank_char, suit_char = s[0], s[1]
    rank_idx = _RANK_CHARS.index(rank_char)
    suit_idx = _SUIT_CHARS_OURS.index(suit_char)
    return rank_idx * 4 + suit_idx


def card_ppe_to_id(s: str) -> int:
    """Convert PyPokerEngine 'SA' / 'DT' string to card id."""
    suit_char, rank_char = s[0], s[1]
    rank_idx = _RANK_CHARS.index(rank_char)
    suit_idx = _SUIT_CHARS_PPE.index(suit_char)
    return rank_idx * 4 + suit_idx


# ---------------------------------------------------------------------------
# GenericView — info-hygienic read of (env, seat)
# ---------------------------------------------------------------------------

@dataclass
class GenericView:
    """A read-only, info-hygienic snapshot of one decision node for an external bot.

    Only hero's hole cards are exposed; all other holes are withheld.
    All chip values are in chips (CHIP_UNIT=100/bb; 100 chips = 1 BB).
    """
    # Hero
    seat: int
    hero_hole_ids: tuple[int, int]            # (c0, c1) card ids
    hero_hole_strs: tuple[str, str]           # ('As', 'Kd') — our format
    hero_hole_ppe: tuple[str, str]            # ('SA', 'DK') — PPE suit-first format
    hero_stack: int                            # chips remaining
    hero_committed_street: int
    hero_committed_total: int

    # Board
    board_ids: list[int]                       # only dealt cards (board[i] >= 0)
    board_strs: list[str]                      # our format
    board_ppe: list[str]                       # PPE format

    # Pot / bet
    pot_before: int                            # pot BEFORE this action (pot_dead + committed_street.sum())
    to_call: int                               # chips to call (0 = check is free)
    current_bet: int                           # current street bet total
    min_raise: int

    # Street / table
    street: int                                # 0=preflop 1=flop 2=turn 3=river
    button: int
    n_live: int                                # number of LIVE or ALLIN seats
    n_active_live: int                         # number of LIVE (not yet all-in) seats

    # Per-seat arrays (6 elements each, index = seat number)
    stacks: np.ndarray                         # int32[6]
    committed_street: np.ndarray               # int32[6]
    committed_total: np.ndarray                # int32[6]
    seat_live: np.ndarray                      # bool[6] True if LIVE or ALLIN
    seat_folded: np.ndarray                    # bool[6]
    seat_allin: np.ndarray                     # bool[6]


def build_view(env, seat: int) -> GenericView:
    """Build a GenericView from (env, seat). Only reads env.holes[seat] for holes."""
    from pokerbot.engine.np_engine import (
        SEAT_ALLIN, SEAT_FOLDED, SEAT_LIVE,
        _pot_for_sizing,
    )
    seat = int(seat)
    ss = np.asarray(env.seat_status)
    stacks = np.asarray(env.stacks, dtype=np.int32)
    cs = np.asarray(env.committed_street, dtype=np.int32)
    ct = np.asarray(env.committed_total, dtype=np.int32)
    cur_bet = int(env.current_bet)
    committed_here = int(cs[seat])
    to_call = max(0, cur_bet - committed_here)
    pot = int(_pot_for_sizing(env))

    hole_ids = (int(env.holes[seat][0]), int(env.holes[seat][1]))
    board_ids = [int(c) for c in np.asarray(env.board) if int(c) >= 0]

    seat_live = (ss == SEAT_LIVE) | (ss == SEAT_ALLIN)
    seat_folded = ss == SEAT_FOLDED
    seat_allin = ss == SEAT_ALLIN

    return GenericView(
        seat=seat,
        hero_hole_ids=hole_ids,
        hero_hole_strs=(card_id_to_str(hole_ids[0]), card_id_to_str(hole_ids[1])),
        hero_hole_ppe=(card_id_to_ppe(hole_ids[0]), card_id_to_ppe(hole_ids[1])),
        hero_stack=int(stacks[seat]),
        hero_committed_street=committed_here,
        hero_committed_total=int(ct[seat]),
        board_ids=board_ids,
        board_strs=[card_id_to_str(c) for c in board_ids],
        board_ppe=[card_id_to_ppe(c) for c in board_ids],
        pot_before=pot,
        to_call=to_call,
        current_bet=cur_bet,
        min_raise=int(env.min_raise),
        street=int(env.street),
        button=int(env.button),
        n_live=int(np.sum(seat_live)),
        n_active_live=int(np.sum(ss == SEAT_LIVE)),
        stacks=stacks,
        committed_street=cs,
        committed_total=ct,
        seat_live=seat_live,
        seat_folded=seat_folded,
        seat_allin=seat_allin,
    )


# ---------------------------------------------------------------------------
# finalize_action — convert external action to our 11-vector
# ---------------------------------------------------------------------------

def finalize_action(env, seat: int, kind: str, raise_to: Optional[int] = None) -> np.ndarray:
    """Convert an external bot's declared action to a one-hot float64 11-vector.

    Parameters
    ----------
    env   : numpy EnvState
    seat  : acting seat
    kind  : 'fold' | 'check' | 'call' | 'raise' | 'all_in'
    raise_to : chip amount the STREET bet should reach TOTAL (i.e. the raise-TO).
               Required when kind='raise'; ignored otherwise.

    Returns
    -------
    float64 ndarray of shape (11,), sums to 1, zeros on illegal indices.

    Rules (in order):
      1. fold-when-free coercion: FOLD with to_call==0 -> CHECK
      2. kind dispatch -> candidate action index
      3. 'raise' + raise_to -> pseudo-harmonic bucket; shove -> ALL_IN
      4. legality AND: if candidate is illegal, degrade:
           bucket -> try adjacent legal bucket -> CALL -> CHECK -> ALL_IN -> argmax(mask)
      5. all-zero guard: if somehow still all-zero, fall back to argmax(mask)
    """
    mask = legal_action_mask_np(env, seat)
    seat = int(seat)
    cs = int(env.committed_street[seat])
    cur_bet = int(env.current_bet)
    to_call = max(0, cur_bet - cs)
    stack = int(env.stacks[seat])

    # 1. fold-when-free coercion
    if kind == 'fold' and to_call == 0:
        kind = 'check'

    # 2. candidate action index
    if kind in ('fold', 'f'):
        candidate = FOLD
    elif kind in ('check', 'x'):
        candidate = CHECK
    elif kind in ('call', 'c'):
        candidate = CALL
    elif kind in ('all_in', 'allin', 'all in', 'shove'):
        candidate = ALL_IN
    elif kind in ('raise', 'bet', 'r', 'b'):
        if raise_to is None:
            # No amount: treat as call
            candidate = CALL
        else:
            raise_to = int(raise_to)
            # Shove check: if raise_to >= stack + committed_street => ALL_IN
            if raise_to >= stack + cs:
                candidate = ALL_IN
            else:
                from pokerbot.engine.np_engine import _pot_for_sizing
                pot = int(_pot_for_sizing(env))
                # bucket_index_for_bet returns index in [3..NUM_ACTIONS-2]
                candidate = bucket_index_for_bet(raise_to, pot)
    else:
        # Unknown kind — default to call/check
        candidate = CALL if to_call > 0 else CHECK

    # 3. legality check + degradation
    if mask[candidate]:
        idx = candidate
    else:
        idx = _degrade(candidate, mask, to_call, stack)

    # 4. all-zero guard
    if idx < 0 or idx >= NUM_ACTIONS:
        idx = int(np.argmax(mask))

    out = np.zeros(NUM_ACTIONS, dtype=np.float64)
    out[idx] = 1.0
    return out


def _degrade(candidate: int, mask: np.ndarray, to_call: int, stack: int) -> int:
    """Degradation chain: adjacent legal bucket -> CALL/CHECK -> ALL_IN -> argmax(mask)."""
    # If candidate is a bet bucket, try adjacent buckets first
    if BET_BUCKET_START <= candidate < ALL_IN:
        # Try neighbors outward (smaller first, then larger)
        for delta in range(1, ALL_IN - BET_BUCKET_START + 1):
            for sign in (-1, +1):
                alt = candidate + sign * delta
                if BET_BUCKET_START <= alt < ALL_IN and mask[alt]:
                    return alt
        # Fall through to call/check
        if to_call > 0 and mask[CALL]:
            return CALL
        if mask[CHECK]:
            return CHECK
        if mask[ALL_IN]:
            return ALL_IN
    elif candidate == FOLD:
        if mask[CHECK]:
            return CHECK
        if mask[CALL]:
            return CALL
    elif candidate in (CHECK, CALL):
        # Wrong one of the two: try the other
        other = CALL if candidate == CHECK else CHECK
        if mask[other]:
            return other
        if mask[ALL_IN]:
            return ALL_IN
    elif candidate == ALL_IN:
        # ALL_IN illegal: find highest legal bucket
        for b in range(ALL_IN - 1, BET_BUCKET_START - 1, -1):
            if mask[b]:
                return b
        if mask[CALL]:
            return CALL
        if mask[CHECK]:
            return CHECK
    # Final fallback
    live = np.flatnonzero(mask)
    if len(live):
        return int(live[0])
    return 0


# ---------------------------------------------------------------------------
# stable_seed — CRN-deterministic per-decision seed
# ---------------------------------------------------------------------------

def stable_seed(env, seat: int) -> int:
    """Derive a stable integer seed from (env, seat) for CRN-deterministic sampling.

    Hashes: (hist_len, street, seat, hero_hole_bytes, board_bytes).
    Deterministic given identical env state; changing any game-tree feature changes the seed.
    """
    seat = int(seat)
    h = int(env.hist_len)
    st = int(env.street)
    hole_bytes = np.asarray(env.holes[seat], dtype=np.int32).tobytes()
    board_bytes = np.asarray(env.board, dtype=np.int32).tobytes()
    raw = hash((h, st, seat, hole_bytes, board_bytes))
    # Mix to uint64 range to avoid negative seeds
    return raw & 0xFFFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# safe_policy — never-raise wrapper
# ---------------------------------------------------------------------------

def safe_policy(fn: Callable) -> Callable:
    """Wrap a policy function so it never raises.

    On any exception: increment module failure counter, log a one-line warning,
    and return a legal-uniform distribution over the current legal mask.
    """
    def _wrapped(env, seat: int) -> np.ndarray:
        global _failure_count
        try:
            return fn(env, seat)
        except Exception as exc:  # noqa: BLE001
            _failure_count += 1
            log.warning(
                "safe_policy[%s]: exception on seat %d (total failures=%d): %s",
                getattr(fn, "__name__", repr(fn)),
                seat,
                _failure_count,
                exc,
            )
            mask = legal_action_mask_np(env, seat).astype(np.float64)
            s = mask.sum()
            if s > 0:
                return mask / s
            out = np.zeros(NUM_ACTIONS, dtype=np.float64)
            out[CHECK] = 1.0
            return out

    _wrapped.__name__ = getattr(fn, "__name__", "safe_policy_wrapper")
    return _wrapped
