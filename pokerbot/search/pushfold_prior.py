"""Certified push/fold ICM range -> search ROOT action prior (the tournament shallow-stack prior).

At N>6 there is no trained blueprint, so the search's root prior is the uniform-over-legal fallback
(``_root_blueprint`` -> ``make_blueprint_policy`` misses the fingerprint and returns uniform). Pure
search from a flat prior plays weak short-stack tournament poker — the regime where ICM bites hardest
and most decision-EV lives.

This module replaces that uniform fallback, for the PREFLOP push/fold band only, with the CERTIFIED
ante-aware ICM open-shove / call-vs-jam Nash from :mod:`pokerbot.tournament.open_shove_icm` (the same
solver :mod:`pokerbot.tournament.decide` is built on, reused verbatim — no forked solver). It:

  * detects, from the root ``EnvState`` alone, whether hero is first-in (folded to hero) or facing
    exactly one all-in jam (the two spots the certified push/fold abstraction covers),
  * maps hero's hole cards to the 169 bucket and reads hero's equilibrium jam / call frequency,
  * returns a length-``NUM_ACTIONS`` action prior: first-in -> {ALL_IN: f, FOLD: 1-f}; facing a jam ->
    {CALL: f, FOLD: 1-f} (or {CHECK} when hero is the BB and can check a completed jam — never here,
    a jam always sets to_call>0).

Generalizes 2..7 (``solve_open_shove_icm`` enumerates the live ``behind`` set, any count). Ante-aware
(antes -> dead money -> wider ranges, exactly as the certified engine models). ICM-aware when payouts
are supplied; with payouts ``None`` it defaults to winner-take-all ``[1, 0, ...]``, which makes
``icm_equity`` chip-proportional, i.e. the classic CHIP-EV push/fold Nash — still a real, position- and
ante-aware range, vastly stronger than uniform. Returns ``None`` (caller keeps its existing fallback)
whenever the spot is NOT a clean preflop push/fold (limped/raised pot, multiway already-opened, two
jams, deep stack, hero card unknown) so the prior never bluffs a guess where it is not certified.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pokerbot.engine.jax_engine import (
    ALL_IN,
    CALL,
    CHECK,
    CHIP_UNIT,
    FOLD,
    NUM_ACTIONS,
)
from pokerbot.engine.np_engine import SEAT_EMPTY, SEAT_FOLDED, SEAT_LIVE, SEAT_ALLIN

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"

# Default shallow-stack band: trust the push/fold prior only when the effective stack is in the
# jam/fold zone (matches pokerbot.tournament.decide.PUSHFOLD_MAX_BB = 20bb). Above this the prior
# returns None and the deeper-than-jam path (search / the future open-range blueprint) handles it.
DEFAULT_PUSHFOLD_BAND_BB: tuple[float, float] = (0.0, 20.0)


def _card_label(c0: int, c1: int) -> str | None:
    """169 bucket label ('AKs'/'AKo'/'AA') for two engine card ids, or None if either is undealt."""
    c0, c1 = int(c0), int(c1)
    if c0 < 0 or c1 < 0:
        return None
    s0 = _RANKS[c0 // 4] + _SUITS[c0 % 4]
    s1 = _RANKS[c1 // 4] + _SUITS[c1 % 4]
    from pokerbot.tournament.pushfold_icm import hand_to_label

    return hand_to_label(s0 + s1)


def detect_pushfold_spot(env, hero: int) -> dict | None:
    """Classify hero's PREFLOP push/fold spot from the root ``EnvState`` (numpy), or None.

    Returns one of (all seats are env indices, behind in action order = seats yet to act after hero):
      * ``{"kind": "open_shove", "behind": [...], "blinds": {...}}``  -- folded to hero, first-in,
      * ``{"kind": "call", "jammer": j, "behind": [...], "blinds": {...}}`` -- facing exactly one jam.
    None when this is not a clean push/fold spot (postflop, limped/raised, multiway opened, >1 jam,
    no live seat behind a fold-around, etc.). Pure read of ``env``; never mutates it.
    """
    if int(np.asarray(env.street)) != 0:
        return None  # preflop only
    status = np.asarray(env.seat_status)
    n = status.shape[0]
    cs = np.asarray(env.committed_street, dtype=np.int64)
    stacks = np.asarray(env.stacks, dtype=np.int64)
    bb_chips = int(CHIP_UNIT)
    sb_chips = bb_chips // 2

    # Live (yet-to-be-decided) seats and the all-in seats (potential jammers).
    folded = status == SEAT_FOLDED
    allin = status == SEAT_ALLIN
    empty = status == SEAT_EMPTY

    # Identify blinds from the posted street commitments (SB seat posted ~0.5bb, BB ~1bb) for the
    # solver's dead-money / fold-credit bookkeeping. A seat already all-in for more than the BB is a
    # jam, not a blind post; only seats committed exactly the blind amount are treated as blinds.
    blinds: dict[int, float] = {}
    for s in range(n):
        if empty[s]:
            continue
        c = int(cs[s])
        if c == sb_chips:
            blinds[s] = 0.5
        elif c == bb_chips:
            blinds[s] = 1.0

    # Jammers = all-in seats whose total street commitment exceeds the BB (a real shove, not a blind
    # that happens to be all-in for <= 1bb). A short-stack BB all-in for exactly 1bb is a blind, not a jam.
    jammers = [s for s in range(n) if allin[s] and int(cs[s]) > bb_chips]
    # Any seat that VOLUNTARILY put in chips above the BB without being all-in = an open raise/limp-raise
    # -> outside the push/fold abstraction (someone min-raised; not jam/fold). Detect a non-blind, non-jam
    # raiser: a live or folded seat committed strictly between BB and its jam is a non-pushfold open.
    for s in range(n):
        if empty[s] or s in jammers:
            continue
        c = int(cs[s])
        if c > bb_chips and blinds.get(s) is None:
            return None  # an open raise/limp-raise that is not a full jam -> not push/fold

    if len(jammers) > 1:
        return None  # multiway all-in / re-jam -> outside the single-jam abstraction

    # behind = live seats that act AFTER hero this orbit. With no explicit action-order on the env we
    # use seat order from hero+1 .. wrapping, restricted to seats still live (not folded/all-in/empty).
    # For a clean first-in spot every seat before hero must already be out (folded) or a blind yet to act
    # behind; we approximate "behind" as the live seats other than hero, which is what the open-shove
    # solver needs (it values the jam vs every live caller).
    live_others = [s for s in range(n) if status[s] == SEAT_LIVE and s != hero]

    if not jammers:
        # First-in open shove: hero must itself be live and able to act.
        if status[hero] != SEAT_LIVE:
            return None
        if not live_others:
            return None  # nobody to jam into
        return {"kind": "open_shove", "behind": live_others, "blinds": blinds}

    j = jammers[0]
    if j == hero:
        return None  # hero already jammed
    if status[hero] != SEAT_LIVE:
        return None
    # Facing exactly one jam: the behind set (seats still to act after hero) are the other live seats.
    behind = [s for s in live_others if s != j]
    return {"kind": "call", "jammer": j, "behind": behind, "blinds": blinds}


def _effective_bb(env, hero: int, contenders: Sequence[int]) -> float:
    """Effective stack (bb) = min over hero + contenders of (stack + committed_total)/CHIP_UNIT."""
    totals = (np.asarray(env.stacks, dtype=np.float64)
              + np.asarray(env.committed_total, dtype=np.float64))
    rel = [int(hero)] + [int(c) for c in contenders]
    chips = min(float(totals[s]) for s in rel) if rel else float(totals.max())
    return chips / float(CHIP_UNIT)


def pushfold_root_prior(env, hero: int, legal: np.ndarray, *,
                        payouts: Sequence[float] | None = None,
                        ante_bb: float | None = None,
                        band_bb: tuple[float, float] | None = DEFAULT_PUSHFOLD_BAND_BB,
                        iters: int = 1500) -> np.ndarray | None:
    """Certified ICM push/fold ROOT action prior over ``NUM_ACTIONS`` for a shallow preflop spot, or None.

    ``env`` is the numpy root state; ``hero`` the acting seat; ``legal`` the (abstracted) legal mask the
    solver will serve under. ``payouts`` = remaining prize structure (best first); None => winner-take-all
    (chip-EV push/fold Nash, still position/ante-aware). ``ante_bb`` overrides the per-player ante (else
    derived from ``env.pot_dead`` split over live seats). ``band_bb`` gates on effective stack; None =>
    no band gate. Returns None whenever the spot is not a certified push/fold one (caller keeps its
    existing fallback prior, byte-identical).
    """
    spot = detect_pushfold_spot(env, hero)
    if spot is None:
        return None

    # Effective stack band gate: only serve the push/fold prior in the jam/fold zone.
    contenders = spot["behind"] if spot["kind"] == "open_shove" else [spot["jammer"], *spot["behind"]]
    if band_bb is not None:
        eff = _effective_bb(env, hero, contenders)
        if not (band_bb[0] <= eff <= band_bb[1]):
            return None

    label = _card_label(*np.asarray(env.holes)[hero])
    if label is None:
        return None  # hero hole unknown (search base samples it) -> no certified read

    stacks_bb = ((np.asarray(env.stacks, dtype=np.float64)
                  + np.asarray(env.committed_total, dtype=np.float64)) / float(CHIP_UNIT)).tolist()

    # Per-player ante in bb: explicit override, else split the dead pot over the dealt-in seats.
    if ante_bb is not None:
        ante = float(ante_bb)
    else:
        status = np.asarray(env.seat_status)
        dealt = int(((status == SEAT_LIVE) | (status == SEAT_ALLIN) | (status == SEAT_FOLDED)).sum())
        pot_dead_bb = float(int(np.asarray(env.pot_dead))) / float(CHIP_UNIT)
        ante = (pot_dead_bb / dealt) if dealt > 0 else 0.0

    pay = list(payouts) if payouts is not None else [1.0, 0.0]
    blinds = spot["blinds"]

    from pokerbot.tournament.open_shove_icm import solve_open_shove_icm

    if spot["kind"] == "open_shove":
        sol = solve_open_shove_icm(stacks_bb, pay, hero, list(spot["behind"]),
                                   blinds=blinds, ante=ante, iters=iters)
        f = sol.jam_freq(label)
        return _two_action_prior(legal, aggressive=ALL_IN, freq=f)

    # Facing one jam: solve the jammer's open-shove vs the live behind set (hero among them) and read
    # hero's GTO call frequency from that solve (same construction as decide.decide_pushfold).
    jammer = spot["jammer"]
    jam_behind = sorted(set(spot["behind"]) | {hero})
    sol = solve_open_shove_icm(stacks_bb, pay, jammer, jam_behind,
                               blinds=blinds, ante=ante, iters=iters)
    f = float(sol.call[hero][list(sol.labels).index(label)])
    return _two_action_prior(legal, aggressive=CALL, freq=f)


def _two_action_prior(legal: np.ndarray, *, aggressive: int, freq: float) -> np.ndarray | None:
    """Build a NUM_ACTIONS prior placing ``freq`` on the aggressive action and ``1-freq`` on the passive
    action (FOLD if legal, else CHECK). Mass on an illegal action is redistributed to the legal one;
    if neither the aggressive action nor a passive action is legal, return None (let the caller fall back)."""
    legal = np.asarray(legal, dtype=bool)
    f = float(np.clip(freq, 0.0, 1.0))
    prior = np.zeros(NUM_ACTIONS, dtype=np.float64)
    passive = FOLD if legal[FOLD] else (CHECK if legal[CHECK] else -1)
    agg_ok = bool(legal[aggressive])
    if not agg_ok and passive < 0:
        return None
    if not agg_ok:
        prior[passive] = 1.0          # can't jam/call -> all mass on the passive action
    elif passive < 0:
        prior[aggressive] = 1.0       # can't fold/check (rare) -> all mass on aggressive
    else:
        prior[aggressive] = f
        prior[passive] = 1.0 - f
    s = prior.sum()
    if s <= 1e-12:
        return None
    return prior / s
