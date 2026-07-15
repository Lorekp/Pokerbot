"""Equity-certified integrity backstop for under-searched postflop misses."""

from __future__ import annotations

import hashlib
import random

import numpy as np


def _showdown_equity_share(
    hole: tuple[int, int], board: tuple[int, ...], opponents: int, *, n_iter: int = 256
) -> float:
    """Deterministic multiway showdown equity versus uniform legal hands."""
    import battery_native

    known = set(hole) | set(board)
    deck = [c for c in range(52) if c not in known]
    runout_n = 5 - len(board)
    draw_n = runout_n + 2 * opponents
    if opponents < 1 or draw_n > len(deck):
        raise ValueError("invalid postflop equity state")
    seed_material = ",".join(map(str, (*hole, *board, opponents))).encode()
    seed = int.from_bytes(hashlib.blake2b(seed_material, digest_size=8).digest(), "big")
    rng = random.Random(seed)
    share = 0.0
    for _ in range(max(32, int(n_iter))):
        draw = rng.sample(deck, draw_n)
        full_board = list(board) + draw[:runout_n]
        hero_score = int(battery_native.evaluate_seven([*hole, *full_board]))
        best = hero_score
        tied = 1
        offset = runout_n
        for _opp in range(opponents):
            opp_hole = draw[offset:offset + 2]
            offset += 2
            score = int(battery_native.evaluate_seven([*opp_hole, *full_board]))
            if score > best:
                best, tied = score, 1
            elif score == best:
                tied += 1
        if hero_score == best:
            share += 1.0 / tied
    return share / max(32, int(n_iter))


def postflop_passive_equity_policy(
    state, hero: int, legal: np.ndarray, *, n_iter: int = 256,
) -> tuple[np.ndarray | None, dict]:
    """Return a tightly bounded policy with observable multiway-equity proof.

    The opponent hands and remaining board are integrated jointly, so the
    result is a true multiway pot share rather than HU equity raised to a power.
    A betting-range margin keeps calls conservative. When checked to, only an
    exceptionally strong equity proof may use the SMALLEST non-all-in bet; this
    recovers obvious value from sets/trips without granting a coverage miss any
    bluff, large-bet or stack-off authority.
    """
    mask = np.asarray(legal, dtype=bool)
    out = np.zeros(mask.shape, dtype=np.float64)
    board = tuple(int(c) for c in np.asarray(state.board) if int(c) >= 0)
    holes = tuple(int(c) for c in np.asarray(state.holes)[int(hero)] if int(c) >= 0)
    if len(holes) != 2 or not (3 <= len(board) <= 5):
        return None, {}
    committed = np.asarray(state.committed_street, dtype=np.float64)
    to_call = max(0.0, float(state.current_bet) - float(committed[int(hero)]))
    status = np.asarray(state.seat_status)
    opponents = int(sum(
        1 for s, st in enumerate(status)
        if s != int(hero) and int(st) in (2, 3)))
    if opponents < 1:
        return None, {}
    if to_call <= 1e-12 and mask.size > 1 and mask[1]:
        equity = _showdown_equity_share(holes, board, opponents, n_iter=n_iter)
        # Uniform hands are weaker than a range that continues versus a bet.
        # Demand overwhelming raw equity, increasingly strict multiway. The
        # action cap is independently structural: smallest abstract bet only,
        # never explicit ALL_IN.
        value_threshold = min(0.92, 0.76 + 0.05 * max(0, opponents - 1))
        bet = next((i for i in range(3, max(3, mask.size - 1)) if mask[i]), None)
        if bet is not None and equity >= value_threshold:
            out[bet] = 1.0
            action = "small_value_bet"
        else:
            out[1] = 1.0
            action = "check"
        return out, {
            "fallback_source": "postflop_equity_prior",
            "postflop_guard_equity": equity,
            "postflop_guard_required_equity": value_threshold,
            "postflop_guard_opponents": opponents,
            "postflop_guard_action": action,
        }
    if mask.size <= 2 or not mask[2]:
        return None, {}
    pot_now = max(0.0, float(state.pot_dead) + float(committed.sum()))
    pot_odds = to_call / max(pot_now + to_call, 1e-12)
    equity = _showdown_equity_share(holes, board, opponents, n_iter=n_iter)
    # Uniform villain hands understate a bettor's strength. Require eight
    # percentage points of excess equity before a coverage backstop may call.
    required = min(0.95, pot_odds + 0.08)
    if equity >= required:
        out[2] = 1.0
    elif mask.size and mask[0]:
        out[0] = 1.0
    else:
        return None, {}
    return out, {
        "fallback_source": "postflop_equity_prior",
        "postflop_guard_equity": equity,
        "postflop_guard_required_equity": required,
        "postflop_guard_opponents": opponents,
        "postflop_guard_action": "call" if out[2] else "fold",
    }
