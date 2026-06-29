"""Lightweight action abstraction helpers — split out of sd_cfr_nlhe so lean
boxes (no flax/optax) can still build blueprint policies.

The original ``build_allowed_action_mask`` lived in
``pokerbot.training.sd_cfr_nlhe``, but that module's top-level
``import optax`` / ``from flax.training import train_state`` made it
unimportable on the eval-only AWS box. This module re-implements the
function against the same action constants imported from
``pokerbot.engine.jax_engine`` so the lookups match byte-for-byte.
"""
from __future__ import annotations

import numpy as np

from pokerbot.engine.actions import BET_SIZE_BUCKETS
from pokerbot.engine.jax_engine import (
    ALL_IN,
    BET_BUCKET_START,
    CALL,
    CHECK,
    FOLD,
    NUM_ACTIONS,
)


def build_allowed_action_mask(bet_bucket_keep: tuple[int, ...] | None) -> np.ndarray | None:
    """Boolean (NUM_ACTIONS,) of actions the traversal may branch on under action abstraction.

    fold/check/call/all-in are always allowed; only the configured subset of the 7 bet-size buckets
    is. ANDed into the per-node legal mask, this restricts branching to the abstracted bet sizes
    (engine untouched, deterministic => pool/single split stays bit-identical). None => no abstraction.
    """
    if bet_bucket_keep is None:
        return None
    allowed = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in (FOLD, CHECK, CALL, ALL_IN):
        allowed[a] = True
    for i in bet_bucket_keep:
        if not 0 <= int(i) < len(BET_SIZE_BUCKETS):
            raise ValueError(
                f"bet_bucket_keep index {i} out of range [0,{len(BET_SIZE_BUCKETS)})"
            )
        allowed[BET_BUCKET_START + int(i)] = True
    return allowed


__all__ = ["build_allowed_action_mask"]