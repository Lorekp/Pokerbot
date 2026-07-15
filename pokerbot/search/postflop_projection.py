"""Source-depth projection for fixed-stack postflop blueprint fallbacks.

The promoted tier-1 artifact was trained at 100bb.  Its bounded postflop key
contains SPR, so a strategically identical 200-500bb state often addresses a
different (usually absent) row.  This module presents only the *lookup* with a
100bb shadow stack, then translates source-terminal aggression back into the
live deep action menu.  Public cards, pot, commitments and perfect-recall
action history are never changed.
"""

from __future__ import annotations

import copy

import numpy as np

from pokerbot.engine.jax_engine import CHIP_UNIT, SEAT_EMPTY, SEAT_FOLDED
from pokerbot.engine.np_engine import ALL_IN, BET_BUCKET_START, CALL, CHECK, FOLD


REFERENCE_DEPTH_BB = 100.0
DEEP_PROJECTION_MIN_BB = 130.0


def project_postflop_state_to_reference(
    state,
    *,
    reference_bb: float = REFERENCE_DEPTH_BB,
    deep_min_bb: float = DEEP_PROJECTION_MIN_BB,
) -> tuple[object | None, float]:
    """Return ``(shadow_state, effective_starting_bb)`` for a deep postflop node.

    Every still-eligible seat's starting total (stack + committed_total) is
    capped at the artifact's source depth.  Commitments are preserved exactly,
    so projection abstains once any eligible seat has already invested the
    whole source stack.  Shallow/mixed-depth pots abstain as well: their direct
    key already has the relevant effective stack and must not be inflated.
    """
    if int(np.asarray(state.street)) <= 0:
        return None, 0.0
    stacks = np.asarray(state.stacks)
    committed = np.asarray(state.committed_total)
    status = np.asarray(state.seat_status)
    active = np.asarray([
        int(st) not in (int(SEAT_EMPTY), int(SEAT_FOLDED))
        for st in status
    ], dtype=bool)
    totals = stacks.astype(np.float64) + committed.astype(np.float64)
    eligible = totals[active & (totals > 0)]
    if eligible.size == 0:
        return None, 0.0
    effective_bb = float(eligible.min()) / float(CHIP_UNIT)
    if effective_bb <= float(deep_min_bb):
        return None, effective_bb
    target = int(round(float(reference_bb) * float(CHIP_UNIT)))
    if target <= 0 or np.any(committed[active] >= target):
        return None, effective_bb

    projected_stacks = stacks.copy()
    projected_stacks[active] = target - committed[active]
    if hasattr(state, "_replace"):
        shadow = state._replace(stacks=projected_stacks)
    else:
        shadow = copy.copy(state)
        shadow.stacks = projected_stacks
    return shadow, effective_bb


def project_postflop_source_policy(
    probs: np.ndarray,
    source_state,
    hero: int,
    source_legal: np.ndarray,
    live_legal: np.ndarray,
) -> tuple[np.ndarray, int | None, float]:
    """Map a trusted 100bb row into the live deep action semantics.

    Explicit ALL_IN and any source bucket that consumes the entire shadow stack
    are terminal at training depth.  Their mass moves to the largest aggressive
    bucket that remains nonterminal at source depth and legal live.  Thus the
    fallback retains the learned value/aggression signal without turning a
    100bb stack-off into a 200-500bb stack-off.
    """
    from pokerbot.engine.np_engine import _action_amount

    p = np.asarray(probs, dtype=np.float64).copy()
    src = np.asarray(source_legal, dtype=bool)
    live = np.asarray(live_legal, dtype=bool)
    stack = int(np.asarray(source_state.stacks)[int(hero)])
    terminal = {ALL_IN}
    for action in range(BET_BUCKET_START, ALL_IN):
        if action < src.size and src[action]:
            if int(_action_amount(source_state, int(hero), action)) >= stack:
                terminal.add(action)
    moved = float(sum(p[a] for a in terminal if a < p.size))

    candidates = [
        action for action in range(BET_BUCKET_START, ALL_IN)
        if action < src.size and action < live.size and src[action] and live[action]
        and int(_action_amount(source_state, int(hero), action)) < stack
    ]
    target = (max(candidates,
                  key=lambda a: int(_action_amount(source_state, int(hero), a)))
              if candidates else None)
    for action in terminal:
        if action < p.size:
            p[action] = 0.0
    if target is not None:
        p[target] += moved
    p *= live
    total = float(p.sum())
    if np.isfinite(total) and total > 1e-12:
        return p / total, target, moved

    # Availability-only fail-closed path.  A projected row must never turn into
    # arbitrary live aggression if all of its source actions disappear.
    safe = np.zeros_like(p)
    action = next((a for a in (CHECK, FOLD, CALL)
                   if a < live.size and live[a]), None)
    if action is not None:
        safe[action] = 1.0
    return safe, target, moved
