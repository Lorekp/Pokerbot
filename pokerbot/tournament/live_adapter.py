"""Live decision adapter for the tournament push/fold engine (Phase 8 core).

Maps a live preflop spot -> the certified :func:`pokerbot.tournament.decide.decide_pushfold`. The hard
logic is **spot detection** from the action history (is hero first-in, or facing a single jam?); the
remaining driver wiring (reading stacks/blinds/ante/payouts off the live state and routing tournament
mode in ``advisors.make_view``) is mechanical and sits on top of this.

Kept free of the live state-machine types so it is unit-testable with plain inputs. The driver passes:
  * ``stacks_bb``    -- per-seat stacks in bb (full remaining field for ICM; final-table = the table),
  * ``payouts``      -- remaining prize structure (best first), from the tournament lobby/config,
  * ``blinds``       -- {sb_seat: 0.5, bb_seat: 1.0} (in bb), ``ante`` -- per-player ante in bb,
  * ``action_order`` -- seats in preflop action order, and ``history`` -- [(seat, action), ...] so far
    where action in {"fold","jam","call"} (the push/fold abstraction).

Returns a :class:`pokerbot.tournament.decide.Decision`. Spots the push/fold abstraction does not cover
(a limp/min-raise, an already-multiway or re-jammed pot) return ``action="defer"`` -- the engine only
speaks where it is certified.
"""

from __future__ import annotations

from typing import Sequence

from pokerbot.tournament.decide import Decision, decide_pushfold


def detect_spot(history: Sequence[tuple[int, str]], hero_seat: int,
                action_order: Sequence[int]) -> dict:
    """Classify hero's preflop push/fold spot from the action history.

    Returns one of:
      * ``{"kind": "open_shove", "behind": [...]}``   -- folded to hero (no jam yet),
      * ``{"kind": "call", "jammer": j, "behind": [...]}`` -- exactly one jam, hero to act,
      * ``{"kind": "defer", "why": ...}``             -- anything the push/fold engine should skip.
    ``behind`` = seats that act after hero and have not folded.
    """
    acted = {s for s, _ in history}
    folded = {s for s, a in history if a == "fold"}
    jammers = [s for s, a in history if a == "jam"]
    callers = [s for s, a in history if a == "call"]

    if callers:
        return {"kind": "defer", "why": "pot already called/multiway"}
    if len(jammers) > 1:
        return {"kind": "defer", "why": "multiple jams (re-jam / multiway all-in)"}

    pos = list(action_order)
    if hero_seat not in pos:
        return {"kind": "defer", "why": "hero not in action order"}
    hi = pos.index(hero_seat)
    behind = [s for s in pos[hi + 1:] if s not in folded]

    if not jammers:
        # everyone who acted before hero must have folded for this to be a clean first-in
        if any(s not in folded for s, _ in history):
            return {"kind": "defer", "why": "limped/raised action before hero (not push-fold)"}
        return {"kind": "open_shove", "behind": behind}

    j = jammers[0]
    # facing exactly one jam: everyone else who acted (besides the jammer) folded
    if any(a not in ("fold", "jam") for _, a in history):
        return {"kind": "defer", "why": "non push/fold action present"}
    return {"kind": "call", "jammer": j, "behind": behind}


def tournament_decide(stacks_bb: Sequence[float], payouts: Sequence[float], hero_seat: int,
                      hero_hand: str, *, blinds: dict[int, float], ante: float,
                      action_order: Sequence[int], history: Sequence[tuple[int, str]],
                      pushfold_max_bb: float = 20.0, iters: int = 3000) -> Decision:
    """Full live decision: detect the spot, then solve it with the certified ICM push/fold engine."""
    spot = detect_spot(history, hero_seat, action_order)
    if spot["kind"] == "defer":
        return Decision("defer", 0.0, "live-adapter", note=spot["why"])
    if spot["kind"] == "open_shove":
        return decide_pushfold(stacks_bb, payouts, hero_seat, hero_hand, blinds=blinds,
                               behind=spot["behind"], ante=ante,
                               pushfold_max_bb=pushfold_max_bb, iters=iters)
    return decide_pushfold(stacks_bb, payouts, hero_seat, hero_hand, blinds=blinds,
                           jammer=spot["jammer"], behind=spot["behind"], ante=ante,
                           pushfold_max_bb=pushfold_max_bb, iters=iters)
