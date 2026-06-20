"""Unified short-stack tournament decision router (the certified engine, made usable).

Given a live spot, returns a push/fold action from the certified ICM engine:
  * first-in (folded to hero)  -> open-shove jam/fold   (:mod:`open_shove_icm`)
  * facing a single jam        -> ICM call/fold         (caller range from the jammer's open-shove solve)

It ABSTAINS (returns ``action="defer"``) outside the certified push/fold zone -- effective stack
above ``pushfold_max_bb`` (deep play is the blueprint + ICM overlay, a separate track) or spots the
push/fold abstraction does not cover (limped/raised pots, multiway already-opened action). Abstaining
is deliberate: the engine only speaks where it is certified, so it never bluffs a guess as GTO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pokerbot.tournament.icm_overlay import chip_ev_action_value, icm_action_value
from pokerbot.tournament.open_shove_icm import solve_open_shove_icm
from pokerbot.tournament.pushfold_icm import hand_to_label

PUSHFOLD_MAX_BB = 20.0          # above this effective depth, hand off to the deep (chip-EV) provider


@dataclass
class Decision:
    action: str                 # "jam" | "fold" | "call" | "defer"
    freq: float                 # equilibrium frequency of the chosen action (1.0 = pure)
    source: str                 # provenance
    nash_gap: float = 0.0       # measured exploitability of the solve (0 for defer)
    note: str = ""


def _eff_bb(stacks: Sequence[float], hero: int, contenders: Sequence[int]) -> float:
    """Effective stack (bb) = min of hero and the relevant opponents."""
    rel = [stacks[hero]] + [stacks[o] for o in contenders]
    return float(min(rel))


def decide_deep(stacks: Sequence[float], payouts: Sequence[float], hero: int,
                action_outcomes: dict[str, Sequence[tuple[float, Sequence[float]]]],
                *, deal_struck: bool = False) -> Decision:
    """Deep-stack decision: the ICM-correct choice over a chip-EV strategy's actions.

    ``action_outcomes`` maps each candidate action -> its chip-outcome distribution
    ``[(prob, chip_delta_per_seat), ...]`` (what the cash blueprint+search produces at its leaves).
    Each action is valued by :func:`icm_action_value` (ICM-weighted over outcomes) and the max is
    taken. Far from the money this reduces to the chip-EV choice automatically (chips are $-linear),
    so the strong cash engine is used unchanged; near the money it tightens correctly. This removes
    the blind >20bb defer: deep play = the cash engine's lines, ICM-graded.

    ``deal_struck`` toggles ICM OFF: after a final-table deal the money is already split, so play is
    payout-agnostic CHIP-EV -- the action is valued by :func:`chip_ev_action_value` and the bot plays
    exactly the cash chip-EV line, handing off cleanly to the value-extraction layer.
    """
    if not action_outcomes:
        return Decision("defer", 0.0, "deep", note="no chip-EV actions provided")
    if deal_struck:
        vals = {a: chip_ev_action_value(stacks, outs, hero) for a, outs in action_outcomes.items()}
        best = max(vals, key=vals.get)
        return Decision(best, 1.0, "deal-chip-ev", note=f"chip_ev={vals[best]:.4f}")
    vals = {a: icm_action_value(stacks, outs, payouts, hero) for a, outs in action_outcomes.items()}
    best = max(vals, key=vals.get)
    return Decision(best, 1.0, "deep-icm-graded", note=f"icm_value={vals[best]:.4f}")


def decide_pushfold(stacks: Sequence[float], payouts: Sequence[float], hero: int, hero_hand: str, *,
                    blinds: dict[int, float], behind: Sequence[int] | None = None,
                    jammer: int | None = None, ante: float = 0.0,
                    pushfold_max_bb: float = PUSHFOLD_MAX_BB, iters: int = 3000,
                    deep_provider=None) -> Decision:
    """Decide a short-stack spot.

    Provide EITHER ``behind`` (hero is first-in; the seats yet to act) for an open-shove decision,
    OR ``jammer`` (a seat that has open-jammed; ``behind`` then = seats still to act after hero,
    default none) for a call/fold decision. ``hero_hand`` is a label ('AKs') or 4-char string
    ('AhKd'). Stacks/blinds in bb.
    """
    label = hero_hand if len(hero_hand.strip()) <= 3 else hand_to_label(hero_hand)

    if jammer is None:
        # ---- first-in open-shove ----
        if behind is None:
            raise ValueError("open-shove decision needs `behind` (seats yet to act)")
        eff = _eff_bb(stacks, hero, behind)
        if eff > pushfold_max_bb:
            if deep_provider is None:
                return Decision("defer", 0.0, "deep-blueprint",
                                note=f"eff {eff:.0f}bb > {pushfold_max_bb:.0f}bb (no deep provider)")
            return decide_deep(stacks, payouts, hero, deep_provider())
        sol = solve_open_shove_icm(stacks, payouts, hero, list(behind), blinds=blinds,
                                   ante=ante, iters=iters)
        freq = sol.jam_freq(label)
        return Decision("jam" if freq >= 0.5 else "fold", freq if freq >= 0.5 else 1 - freq,
                        "open-shove-icm", sol.nash_gap)

    # ---- facing an open-jam: call or fold ----
    others = list(behind or [])
    eff = _eff_bb(stacks, hero, [jammer])
    if eff > pushfold_max_bb:
        if deep_provider is None:
            return Decision("defer", 0.0, "deep-blueprint",
                            note=f"eff {eff:.0f}bb > {pushfold_max_bb:.0f}bb (no deep provider)")
        return decide_deep(stacks, payouts, hero, deep_provider())
    # Solve the jammer's open-shove vs the live behind set (hero among them) and read hero's GTO call.
    jam_behind = sorted(set(others) | {hero})
    sol = solve_open_shove_icm(stacks, payouts, jammer, jam_behind, blinds=blinds,
                               ante=ante, iters=iters)
    call_freq = float(sol.call[hero][list(sol.labels).index(label)])
    return Decision("call" if call_freq >= 0.5 else "fold",
                    call_freq if call_freq >= 0.5 else 1 - call_freq, "call-vs-jam-icm", sol.nash_gap)
