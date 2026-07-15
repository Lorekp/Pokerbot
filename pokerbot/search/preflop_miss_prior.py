"""Conservative prior for structurally uncovered preflop blueprint nodes."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from pokerbot.engine.preflop_strength import card_index_to_code, chen_score


_OPEN_RANGE_FRACTION = {
    "UTG": 0.12,
    "HJ": 0.18,
    "MP": 0.18,
    "CO": 0.28,
    "BTN": 0.42,
    "SB": 0.50,
}


@lru_cache(maxsize=512)
def _equity_vs_conservative_open(label: str, opener_position: str | None) -> float | None:
    """Removal-free 169-grid equity versus a conservative positional open.

    This reuses the certified push/fold equity artifact already shipped with the
    bot.  It is intentionally only a coverage-miss backstop; failure to load the
    grid returns ``None`` and preserves the legacy fail-closed decision.
    """
    try:
        from pokerbot.tournament.pushfold_icm import _hu_grid

        eq, labels, weights = _hu_grid()
        hero = labels.index(label)
        # Rank all 169 classes by equity against the combo-weighted uniform
        # distribution, then construct the opener's tight top-X% range.
        strength = np.asarray(eq, dtype=np.float64) @ np.asarray(weights, dtype=np.float64)
        strength /= max(float(np.asarray(weights).sum()), 1e-12)
        order = np.argsort(-strength)
        fraction = _OPEN_RANGE_FRACTION.get(
            str(opener_position or "").upper(), 0.08)
        target = fraction * float(np.asarray(weights).sum())
        range_weights = np.zeros_like(np.asarray(weights, dtype=np.float64))
        accumulated = 0.0
        for idx in order:
            remaining = target - accumulated
            if remaining <= 1e-12:
                break
            take = min(float(weights[idx]), remaining)
            range_weights[idx] = take
            accumulated += take
        total = float(range_weights.sum())
        if total <= 1e-12:
            return None
        return float(np.dot(np.asarray(eq)[hero], range_weights) / total)
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return None


def coverage_hole_policy(
    card_indices,
    legal: np.ndarray,
    *,
    to_call_bb: float,
    effective_stack_bb: float,
    pot_bb: float | None = None,
    hero_position: str | None = None,
    opener_position: str | None = None,
) -> tuple[np.ndarray, float | None, float, float | None, float | None, bool]:
    """Return a passive, price-aware policy for a proven blueprint miss.

    The policy never bets, raises, or jams.  Its continuation threshold uses
    both absolute price and stack risk.  For an ordinary <=3bb open, late
    position and the big blind use a wider passive defence only against a
    non-UTG opener; early position and unknown contexts retain the conservative
    baseline. Progressively larger fractions of the effective stack require
    AQ+/JJ+, then QQ+, then KK+. This is an explicit abstraction backstop, not
    a reinterpretation of uniform fallback probabilities.
    """
    mask = np.asarray(legal, dtype=bool)
    out = np.zeros(mask.shape, dtype=np.float64)
    cards = [card_index_to_code(int(c)) for c in card_indices if int(c) >= 0]
    score = chen_score(cards)
    risk = max(0.0, float(to_call_bb)) / max(float(effective_stack_bb), 1e-9)
    if risk <= 0.05:
        threshold = 9.0
        # A fixed score>=9 cutoff folded KJo on the BTN and KTs in the BB to a
        # 2.5bb open whenever the deep blueprint key missed.  That was not a
        # poker judgment; it was missing position/price context.  Widen only the
        # well-defined cheap late/blind defence lane.  SB flats stay tighter,
        # and an UTG/unknown opener never receives this relaxation.
        hp = str(hero_position or "").upper()
        op = str(opener_position or "").upper()
        if float(to_call_bb) <= 3.0 and op in {"HJ", "CO", "BTN", "SB"}:
            if hp in {"BTN", "CO", "BB"}:
                threshold = 7.0
            elif hp == "SB":
                threshold = 8.0
    elif risk <= 0.15:
        threshold = 12.0
    elif risk <= 0.30:
        threshold = 14.0
    else:
        threshold = 16.0

    # Chen thresholds are deliberately conservative, but Chen's nonlinear pair
    # bonus makes a numeric cutoff mis-order AK behind QQ at shallow stacks.  A
    # live AKs fold to a 3.25bb CO open exposed that mismatch.  For premium
    # candidates only, use the certified 169-grid to prove that a passive call
    # clears actual pot odds plus a depth-dependent realization margin.  This is
    # a one-way fold->call correction: it can never manufacture aggression.
    equity = None
    required_equity = None
    equity_override = False
    if (pot_bb is not None and float(pot_bb) > 0.0 and score is not None
            and score >= 10.0 and to_call_bb > 0.0):
        try:
            from pokerbot.tournament.pushfold_icm import hand_to_label

            label = hand_to_label(cards[0] + cards[1])
            equity = _equity_vs_conservative_open(label, opener_position)
        except (IndexError, ValueError):
            equity = None
        if equity is not None:
            pot_odds = float(to_call_bb) / (float(pot_bb) + float(to_call_bb))
            eff = max(float(effective_stack_bb), 0.0)
            margin = 0.04 if eff <= 30.0 else (0.07 if eff <= 80.0 else 0.12)
            required_equity = min(0.95, pot_odds + margin)
            equity_override = bool(equity >= required_equity)

    # np_engine layout: 0 fold, 1 check, 2 call. Unknown rows never create
    # aggression; if checking is free it dominates every other fallback.
    if 1 < mask.size and mask[1] and to_call_bb <= 1e-9:
        out[1] = 1.0
    elif (2 < mask.size and mask[2] and score is not None
          and (score >= threshold or equity_override)):
        out[2] = 1.0
    elif mask.size and mask[0]:
        out[0] = 1.0
    else:
        first = next((int(i) for i in np.flatnonzero(mask)), None)
        if first is not None:
            out[first] = 1.0
    return out, score, threshold, equity, required_equity, equity_override
