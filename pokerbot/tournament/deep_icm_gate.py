"""Deep (>20bb) ICM shove/fold gate — the measured-justified deep-postflop value-add.

The shallow push/fold prior (``decide_pushfold``) DEFERS above ~20bb effective, handing the deep spot to
the chip-EV search. But the targeted grid (``scripts/tournament/deep_icm_grid.py``) MEASURED that ICM
reorders the shove/fold decision even at 35-50bb near the money: it folds a +chip-EV but -$EV all-in across
an equity band ~0.50-0.60, and that band PERSISTS deep. This gate serves that band — given a deep all-in
spot (hero's showdown equity vs the caller + the chips at risk), it returns the ICM-correct shove/fold by
grading the all-in outcomes under ``decide_deep`` (the same valuation the grid validated), and flags when
ICM DIFFERS from chip-EV (the value-add). Firewall-safe: pure ICM math over <=6 seats, no engine change.

The live bridge calls this at a deep tournament all-in node (eff stack >20bb, payouts known) and OVERRIDES
the chip-EV search line only when ``flipped`` and the ICM choice is FOLD — i.e. it folds the +chip-EV but
-$EV shoves the search would otherwise make. ICM-neutral spots (deep / far from money) return shove==chip-EV
(flipped False) so the gate is a no-op there.
"""

from __future__ import annotations

from typing import Sequence

from pokerbot.tournament.decide import decide_deep


def _shove_outcomes(n: int, hero: int, villain: int, hero_equity: float, risked: float):
    """HU all-in (hero vs the covering ``villain``): win ``hero_equity`` -> hero +risked / villain -risked;
    lose -> the reverse; all other seats unchanged. The chip-delta vectors decide_deep / icm_action_value
    consume."""
    win = [0.0] * n;  win[hero] = +risked;  win[villain] = -risked
    lose = [0.0] * n; lose[hero] = -risked; lose[villain] = +risked
    e = float(max(0.0, min(1.0, hero_equity)))
    return {"shove": [(e, win), (1.0 - e, lose)], "fold": [(1.0, [0.0] * n)]}


def deep_icm_shove_fold(stacks: Sequence[float], payouts: Sequence[float], hero: int, villain: int,
                        hero_equity: float, risked: float) -> tuple[str, bool]:
    """ICM-correct shove/fold at a deep near-money all-in, and whether it DIFFERS from chip-EV.

    Returns ``(action, flipped)`` where ``action in {"shove","fold"}`` is the ICM choice and ``flipped`` is
    ``True`` iff ICM disagrees with the chip-EV choice over the SAME outcomes (the deep-ICM value-add). The
    live serve overrides the search ONLY when ``flipped and action == "fold"`` (fold a +chip-EV/-$EV shove).
    ``risked`` = chips the hero puts at risk (min of the two stacks); ``hero_equity`` = P(hero wins the
    showdown vs the caller's range)."""
    n = len(stacks)
    outs = _shove_outcomes(n, hero, villain, hero_equity, risked)
    a_icm = decide_deep(stacks, payouts, hero, outs, deal_struck=False).action
    a_chip = decide_deep(stacks, payouts, hero, outs, deal_struck=True).action
    return a_icm, (a_icm != a_chip)


def deep_icm_overrides_shove(stacks: Sequence[float], payouts: Sequence[float], hero: int, villain: int,
                             hero_equity: float, risked: float) -> bool:
    """True iff ICM says FOLD a shove that chip-EV would take here — the one case the live serve overrides
    the chip-EV search. False for ICM-neutral / clear spots (the gate is a no-op)."""
    action, flipped = deep_icm_shove_fold(stacks, payouts, hero, villain, hero_equity, risked)
    return flipped and action == "fold"
