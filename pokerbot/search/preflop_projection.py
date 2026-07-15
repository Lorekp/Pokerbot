"""Pure action-semantic projection for fixed-depth preflop blueprints."""

from __future__ import annotations

import numpy as np


def project_deep_preflop_blueprint(
    probs: np.ndarray,
    legal: np.ndarray,
    *,
    bet_bucket_start: int = 3,
    num_actions: int = 11,
    source_hit: bool = True,
    source_legal: np.ndarray | None = None,
) -> tuple[np.ndarray, int | None, float]:
    """Translate fixed-depth terminal actions into the live action menu.

    ``source_legal`` is the action mask in a shadow copy of the same public state
    capped at the blueprint's reference stack.  A live raise bucket that is legal
    only because the real stack is deeper is terminal-like at training depth, just
    like the explicit ALL_IN label.  Its mass therefore cannot be served unchanged
    at 250-500bb.  When omitted, only explicit ALL_IN is translated (legacy API).
    """
    p = np.asarray(probs, dtype=np.float64).copy()
    mask = np.asarray(legal, dtype=bool)
    all_in = num_actions - 1
    src = (np.asarray(source_legal, dtype=bool)
           if source_legal is not None else mask)
    terminal_like = {all_in}
    if source_legal is not None:
        terminal_like.update(
            i for i in range(bet_bucket_start, all_in)
            if i < mask.size and mask[i] and (i >= src.size or not src[i])
        )
    moved = float(sum(p[i] for i in terminal_like if i < p.size))

    # A blueprint miss is represented by uniform-legal probabilities for backwards
    # compatibility. That fallback contains no range/aggression signal to project:
    # moving its ALL_IN share onto a raise would manufacture aggression, especially
    # for hands absent from the table. Fail closed instead (free check, else fold).
    if not source_hit:
        out = np.zeros_like(p)
        safe = next((i for i in (1, 0, 2) if i < mask.size and mask[i]), None)
        if safe is None:
            safe = next((int(i) for i in np.flatnonzero(mask)), None)
        if safe is not None:
            out[safe] = 1.0
        return out, None, moved
    if moved <= 0.0:
        return p, None, 0.0

    raises = [i for i in range(bet_bucket_start, all_in)
              if i < mask.size and mask[i] and i < src.size and src[i]]
    target = max(raises) if raises else None
    if target is None and source_legal is None:
        # Backwards-compatible pure ALL_IN translation: when callers have no
        # reference-depth mask, retain the old call/check/fold target behavior.
        target = next((i for i in (2, 1, 0) if i < mask.size and mask[i]), None)
    for i in terminal_like:
        if i < p.size:
            p[i] = 0.0
    if target is not None:
        p[target] += moved
    p *= mask
    total = float(p.sum())
    if total <= 1e-12:
        # If every learned action was terminal at the reference depth and no
        # nonterminal raise exists live, fail closed without inventing a call.
        safe = next((i for i in (1, 0, 2) if i < mask.size and mask[i]), None)
        if safe is not None:
            p[safe] = 1.0
        return p, target, moved
    return p / total, target, moved
