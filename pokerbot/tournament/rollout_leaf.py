"""Mean-preserving multiway all-in-EQUITY leaf for the deep-ICM evaluator rollouts (workflow wexmo5p7v P1).

A deep-ICM rollout (``pokerbot.tournament.action_outcomes``) advances a cloned node to terminal and needs a
LOW-VARIANCE per-seat chip-delta at the leaf. The engine's ``allin_settle`` hook fires the moment betting is
over (``not actors_remaining()`` in ``game.advance_to_terminal``); this module is that hook. Instead of
dealing ONE runout (high variance), it averages the side-pot-correct settle (``sim._settle_allin``, reusing
``aof_equity_grids._eval7``) over many sampled board completions -> each contender's all-in EQUITY share, so
the leaf is mean-preserving (same expectation as a real runout, far lower variance). Returns per-seat NET
deltas (won minus committed; folders lose their committed) -- the same shape ``resolve_showdown`` returns,
so it drops straight into the ``allin_settle`` slot. Zero-sum by construction (sum of deltas == 0).
"""

from __future__ import annotations

import random

from pokerbot.tournament.sim import _settle_allin


def equity_allin_settle(state, n_board_seen=None, *, n_runouts: int = 120,
                        rng: random.Random | None = None) -> dict[int, float]:
    """``allin_settle`` hook: equity-weighted, side-pot-correct settle of the live all-in pot.

    Live contenders (not folded, hole-carded) split each pot layer by their EQUITY over the remaining board
    (mean of ``_settle_allin`` across ``n_runouts`` completions; exact if the board is already complete).
    Returns ``{seat: net_chip_delta}`` (won - committed for contenders, -committed for folders), summing to
    0. ``n_board_seen`` is accepted (hook signature) but unused -- the equity is over ``state.board`` itself.
    """
    rng = rng or random.Random(0)
    players = state.players
    n = len(players)
    committed = {i: float(getattr(players[i], "committed_total_bb", 0.0)) for i in range(n)}
    contenders = [i for i in range(n) if (not players[i].folded) and players[i].hole is not None]
    if not contenders:                                     # degenerate: nobody to show down -> no movement
        return {i: 0.0 for i in range(n)}
    holes = {i: (int(players[i].hole[0].index), int(players[i].hole[1].index)) for i in contenders}
    board = [int(c.index) for c in state.board]
    dead = sum(committed[i] for i in range(n) if i not in contenders)   # folders' chips -> main pot
    c_committed = {i: committed[i] for i in contenders}

    need = 5 - len(board)
    if need <= 0:
        won = _settle_allin(contenders, c_committed, holes, board[:5], dead)
    else:
        used = set(board) | {c for i in contenders for c in holes[i]}
        deck = [c for c in range(52) if c not in used]
        acc = {i: 0.0 for i in contenders}
        for _ in range(n_runouts):
            runout = rng.sample(deck, need)
            w = _settle_allin(contenders, c_committed, holes, board + runout, dead)
            for i, v in w.items():
                acc[i] += v
        won = {i: acc[i] / n_runouts for i in contenders}

    deltas = {}
    for i in range(n):
        deltas[i] = (won.get(i, 0.0) - committed[i]) if i in contenders else -committed[i]
    return deltas
