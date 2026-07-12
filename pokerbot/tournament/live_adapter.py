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

import random
from typing import Any, Optional, Sequence

import numpy as np

from pokerbot.tournament.decide import Decision, decide_pushfold
from pokerbot.tournament.pushfold_icm import _hu_grid, hand_to_label

# Above this effective depth the certified shallow push/fold prior DEFERS and the deep-ICM gate owns
# the (single-jam) call-vs-jam spot. Mirrors decide.PUSHFOLD_MAX_BB.
PUSHFOLD_MAX_BB = 20.0

# ---------------------------------------------------------------------------------------------------
# 169-grid equity plumbing shared by the deep-ICM / PKO / field-ICM live overrides.
# ``_EQ[i, j]`` = P(bucket i beats bucket j heads-up), ``_LABELS`` the 169 labels, ``_W`` the combo
# multiplicities (pair 6 / suited 4 / offsuit 12) used to weight a range vector.
# ---------------------------------------------------------------------------------------------------
_EQ, _LABELS, _W = _hu_grid()


def _default_deep_jam() -> np.ndarray:
    """A generic wide deep-jam range: the top ~50% of combo mass by mean HU equity. Used when the live
    caller has no sharper read on the jammer's range (conservative: wide keeps hero's required-equity
    reads honest against typical deep open-jams)."""
    strength = _EQ.mean(axis=1)
    order = np.argsort(-strength)
    keep = np.cumsum(_W[order]) <= 0.5 * _W.sum()
    rng = np.zeros(len(_LABELS))
    rng[order[keep]] = 1.0
    return rng


#: Default jammer range for the deep-ICM / PKO overrides when no live read exists.
_DEFAULT_DEEP_JAM = _default_deep_jam()


def _card_code(card: Any) -> str:
    """Canonical 2-char code ('Ah') for either live card shape: a normalized-state card (``.code``)
    or an engine ``Card`` (``str(card)``)."""
    code = getattr(card, "code", None)
    return str(code) if code else str(card)


def _hand_label(hole) -> Optional[str]:
    """Hero's 169-grid label ('T9s') from two hole cards, or ``None`` on any odd card rep."""
    try:
        return hand_to_label(_card_code(hole[0]) + _card_code(hole[1]))
    except Exception:  # noqa: BLE001 -- unreadable cards -> no certified decision -> fail-safe None
        return None


def _equity_vs_range(label: str, jam_range: Sequence[float]) -> float:
    """Hero's HU showdown equity for ``label`` vs a combo-weighted 169 range vector.

    ``jam_range`` is a per-bucket weight vector (0/1 for a cut range, or fractional frequencies); the
    equity is the combo-weighted average of the pairwise HU grid over the range. Card removal is not
    modeled (the 169 grid is removal-free), matching the certified deep-ICM grid inputs."""
    i = _LABELS.index(label)
    wr = np.asarray(jam_range, dtype=np.float64) * _W
    tot = float(wr.sum())
    if tot <= 0.0:
        return 0.5
    return float((_EQ[i] * wr).sum() / tot)


def _start_stacks(players) -> list[float]:
    """Hand-start stacks per seat: chips behind + chips already committed this hand (bb)."""
    return [float(getattr(p, "stack_bb", 0.0)) + float(getattr(p, "committed_total_bb", 0.0))
            for p in players]


def _single_jammer(players, hero: int) -> Optional[int]:
    """The seat index of the single all-in villain hero faces, or ``None`` (no jam / multiway all-in).
    Conservative by construction: anything other than EXACTLY ONE all-in villain returns None."""
    jammers = [i for i, p in enumerate(players) if i != hero and bool(getattr(p, "all_in", False))]
    return jammers[0] if len(jammers) == 1 else None


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


# ====================================================================================================
# Live override layer -- thin, conservative adapters over the certified ICM math that the live serve
# consults AFTER (or around) the chip-EV search decision. Every function returns None (defer, no
# override) outside its certified band and on ANY read error; only a proven +$EV correction fires.
# ====================================================================================================

def deep_icm_decide(stacks: Sequence[float], payouts: Sequence[float], hero: int, villain: int,
                    hero_hand: str, jam_range: Sequence[float]) -> Decision:
    """The deep (>20bb) ICM call-vs-jam serve decision: equity-vs-range -> the certified deep-ICM gate.

    Returns ``Decision("fold", ...)`` iff the gate says ICM folds a call chip-EV would take (the
    measured deep-ICM value-add), else ``Decision("defer", ...)`` (the chip-EV search owns the spot).
    ``source`` is always ``"deep-icm-gate"``."""
    from pokerbot.tournament.deep_icm_gate import deep_icm_overrides_shove

    label = hero_hand if len(hero_hand.strip()) <= 3 else hand_to_label(hero_hand)
    eq = _equity_vs_range(label, jam_range)
    risked = min(float(stacks[hero]), float(stacks[villain]))
    if deep_icm_overrides_shove(list(stacks), list(payouts), hero, villain, eq, risked):
        return Decision("fold", 1.0, "deep-icm-gate", note=f"eq={eq:.3f} icm folds a +chipEV call")
    return Decision("defer", 0.0, "deep-icm-gate", note=f"eq={eq:.3f} icm==chipEV (no override)")


def deep_icm_live_override(state: Any, hero: int, payouts: Optional[Sequence[float]], *,
                           jam_range: Optional[Sequence[float]] = None) -> Optional[str]:
    """Live hook: fold a DEEP (+chip-EV / -$EV) call of a single jam, else ``None`` (no override).

    Conservative spot detection on the engine-shaped ``state``: fires ONLY when payouts are known,
    hero faces EXACTLY ONE all-in villain, the effective hand-start stack is ABOVE the certified
    shallow push/fold band (>20bb -- below it the shallow prior owns the spot), and hero's hole cards
    read. Any ambiguity or read error returns ``None`` (the chip-EV search line stands)."""
    try:
        if not payouts:
            return None
        players = getattr(state, "players", None) or []
        if hero < 0 or hero >= len(players):
            return None
        villain = _single_jammer(players, hero)
        if villain is None:
            return None
        stacks = _start_stacks(players)
        if min(stacks[hero], stacks[villain]) <= PUSHFOLD_MAX_BB:
            return None            # shallow: the certified push/fold prior owns it
        label = _hand_label(getattr(players[hero], "hole", None))
        if label is None:
            return None
        rng = _DEFAULT_DEEP_JAM if jam_range is None else jam_range
        d = deep_icm_decide(stacks, list(payouts), hero, villain, label, rng)
        return "fold" if d.action == "fold" else None
    except Exception:  # noqa: BLE001 -- any read/solve error -> no override (fail-safe)
        return None


def pko_call_override(state: Any, hero: int, payouts: Optional[Sequence[float]],
                      villain_bounty: float, *,
                      jam_range: Optional[Sequence[float]] = None) -> Optional[str]:
    """Live hook: PKO bounty-aware CALL of a single jam, else ``None`` (no override).

    The certified chip-EV != $-EV serve: the jammer's bounty LOWERS hero's indifference equity, so a
    marginal hand the prize-only ICM folds becomes +$EV. Returns ``"call"`` ONLY in the flip band
    ``[p*_with_bounty, p*_prize_only)`` and ``None`` everywhere else (no bounty, no single jam,
    unknown payouts, a hand already in/out of the money on prize alone, or any read error).

    Stack convention: hero counts CHIPS BEHIND only (the amount a call risks; committed chips are
    dead either way), the jammer counts the full jam (behind + committed)."""
    try:
        if not payouts or not villain_bounty or float(villain_bounty) <= 0.0:
            return None
        players = getattr(state, "players", None) or []
        if hero < 0 or hero >= len(players):
            return None
        villain = _single_jammer(players, hero)
        if villain is None:
            return None
        stacks = _start_stacks(players)
        stacks[hero] = float(getattr(players[hero], "stack_bb", 0.0))  # hero risks chips behind only
        label = _hand_label(getattr(players[hero], "hole", None))
        if label is None:
            return None
        from pokerbot.tournament.bounty import min_equity_to_call

        p_prize = min_equity_to_call(stacks, list(payouts), hero, villain, villain_bounty=0.0)
        p_bounty = min_equity_to_call(stacks, list(payouts), hero, villain,
                                      villain_bounty=float(villain_bounty))
        rng = _DEFAULT_DEEP_JAM if jam_range is None else jam_range
        eq = _equity_vs_range(label, rng)
        # Only the bounty FLIP band overrides: prize-only would fold (eq < p_prize) but the bounty
        # makes the call +$EV (eq >= p_bounty). In/out of the money on prize alone -> None.
        return "call" if (p_bounty <= eq < p_prize) else None
    except Exception:  # noqa: BLE001 -- fail-safe None
        return None


def field_icm_live_override(state: Any, hero: int, payouts: Optional[Sequence[float]],
                            field_stacks: Optional[Sequence[float]],
                            hero_field_idx: Optional[int], jammer_field_idx: Optional[int], *,
                            jam_range: Optional[Sequence[float]] = None,
                            n_sims: int = 2000,
                            rng: Optional[random.Random] = None) -> Optional[str]:
    """Live hook: multi-table FIELD-ICM fold of a call-vs-jam, else ``None``.

    The multi-table analog of the deep-ICM fold: the WHOLE-FIELD bubble factor can make a call the
    single-table view takes -$EV (the global pay-jump position is tighter than hero's table alone).
    Folds iff ``equity < min(0.99, BF_field / 2)``. Conservative no-ops: no field feed, no hero/jammer
    field index, no single jam on the table, unknown payouts, or any read error. The MC bubble factor
    is seeded deterministically (``random.Random(0)``) for a reproducible serve."""
    try:
        if not payouts or field_stacks is None or hero_field_idx is None or jammer_field_idx is None:
            return None
        players = getattr(state, "players", None) or []
        if hero < 0 or hero >= len(players):
            return None
        if _single_jammer(players, hero) is None:
            return None
        label = _hand_label(getattr(players[hero], "hole", None))
        if label is None:
            return None
        from pokerbot.tournament.mtt_field import field_bubble_factor

        bf = field_bubble_factor(list(field_stacks), list(payouts),
                                 int(hero_field_idx), int(jammer_field_idx),
                                 n_sims=int(n_sims), rng=rng if rng is not None else random.Random(0))
        rng_vec = _DEFAULT_DEEP_JAM if jam_range is None else jam_range
        eq = _equity_vs_range(label, rng_vec)
        return "fold" if eq < min(0.99, bf / 2.0) else None
    except Exception:  # noqa: BLE001 -- fail-safe None
        return None


def satellite_fold_override(resolved_payouts: Any, stacks: Sequence[float], hero: int, *,
                            field_is_complete: bool, action_risks_chips: bool = True) -> bool:
    """Live hook: satellite seat-lock fold gate. ``True`` == fold any chip-risking action.

    Fires ONLY when (a) ``resolved_payouts`` is a SATELLITE ladder, (b) the caller certifies the
    passed ``stacks`` are the WHOLE remaining field (``field_is_complete`` -- a table-local lock does
    not certify a field lock), and (c) hero is mathematically locked and the action risks chips.
    Everything else returns ``False`` (play normally) -- folding a not-actually-locked seat burns EV,
    so the gate is conservative by construction."""
    try:
        if resolved_payouts is None or not bool(getattr(resolved_payouts, "is_satellite", False)):
            return False
        if not field_is_complete:
            return False
        seats = int(getattr(resolved_payouts, "places", 0) or 0)
        if seats <= 0:
            return False
        from pokerbot.tournament.satellite import should_fold_for_seat

        return bool(should_fold_for_seat(list(stacks), seats, hero,
                                         action_risks_chips=action_risks_chips))
    except Exception:  # noqa: BLE001 -- fail-safe: never force a fold on a read error
        return False
