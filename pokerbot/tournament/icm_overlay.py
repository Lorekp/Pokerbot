"""ICM overlay -- makes a chip-EV blueprint/search ICM-correct near the money (Phase 6).

Two forms, matching what information is available at decision time:

  * RIGOROUS (search-wired): :func:`icm_action_value` -- given an action's chip-outcome distribution
    ``[(prob, chip_delta_vector), ...]`` (which depth-limited search produces at its leaves), the
    action's ICM value is the ICM-weighted average over outcomes. This is the *correct* overlay:
    replace each chip-EV leaf with ``icm_equity(pre + delta)`` and the search returns $-EV. It
    reduces to chip-EV exactly when the bubble factor is 1 (deep / far from the money), so it is
    safe to leave on everywhere.

  * HEURISTIC (precomputed-strategy): :func:`icm_tilt` -- when only a chip-EV action-value vector is
    available (no per-action outcome distribution), tilt it by the local risk premium (bubble
    factor): penalize stack-threatening actions in proportion to how much ICM pressure the spot
    carries. Converges to identity as the bubble factor -> 1.

The bubble factor itself comes from :mod:`pokerbot.tournament.icm`. This module is the deep-play
counterpart to the short-stack push/fold solver: the solver bakes ICM into the equilibrium directly;
the overlay bends an existing chip-EV policy toward ICM where a full re-solve is not available.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pokerbot.tournament.icm import bubble_factor, icm_equity


def icm_action_value(pre_stacks: Sequence[float], outcomes: Sequence[tuple[float, Sequence[float]]],
                     payouts: Sequence[float], hero: int) -> float:
    """Rigorous ICM $-EV of an action whose chip result is the distribution ``outcomes`` =
    ``[(prob, chip_delta_per_seat), ...]`` (probabilities should sum to ~1). Each outcome's chips are
    applied (busts clamp at 0) and valued under ICM, then averaged. Equals chip-EV when chips are
    $-linear (deep). This is the value a depth-limited search should return at an ICM leaf."""
    pre = [float(s) for s in pre_stacks]
    total_p = 0.0
    val = 0.0
    for prob, delta in outcomes:
        if prob == 0:
            continue
        post = [max(0.0, pre[i] + float(delta[i])) for i in range(len(pre))]
        val += prob * float(icm_equity(post, payouts)[hero])
        total_p += prob
    return val / total_p if total_p > 0 else float(icm_equity(pre, payouts)[hero])


def icm_tilt(chip_values: Sequence[float], action_risk: Sequence[float],
             stacks: Sequence[float], payouts: Sequence[float], hero: int, villain: int) -> np.ndarray:
    """Heuristic ICM tilt of a chip-EV action-value vector when no outcome distribution is available.

    ``chip_values[a]`` is action ``a``'s chip-EV; ``action_risk[a]`` is the chips it puts at risk
    (e.g. the amount that busts hero if it loses). We subtract a risk premium ``(BF - 1) * risk``,
    where ``BF`` is the pairwise bubble factor vs the most threatening opponent ``villain``. ``BF==1``
    (deep / far from money) -> identity. Monotone: more ICM pressure penalizes risk more.
    """
    bf = bubble_factor(stacks, payouts, hero, villain)
    if bf == float("inf"):
        bf = 50.0                                   # hero cannot gain -> extreme aversion (clamp)
    premium = max(0.0, bf - 1.0)
    cv = np.asarray(chip_values, dtype=float)
    risk = np.asarray(action_risk, dtype=float)
    return cv - premium * risk
