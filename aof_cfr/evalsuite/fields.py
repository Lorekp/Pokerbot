"""Opponent fields for the eval suite — scripted archetypes + per-hand assignment.

The archetypes mirror ``scripts/gauntlet_tournament.py`` (kept there for backwards-compat of its
shipped JSON); the assignment here fixes that script's field-composition bug: ``bots[i % len]``
keyed by SLOT meant HU "mixed" fielded only ``bots[0]`` (overfolder) and 3-max never fielded the
third archetype. We rotate by hand — ``bots[(slot + hand) % len]`` — so every archetype appears at
every seat, and identities follow the BOT (not the slot) so profiles never mix archetypes.
"""

from __future__ import annotations

from dataclasses import dataclass

PREMIUM = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}


@dataclass
class Bot:
    name: str
    push_fn: object  # f(q, n, pos, hist, hand, stack_bb) -> p_push
    leaky: bool


def _gto(q, n, pos, hist, hand, stack_bb):
    return q.base_strategy(n, pos, hist, hand, stack_bb)


def _overfolder(q, n, pos, hist, hand, stack_bb):
    if "A" in hist:                                  # facing a jam: call premiums only
        return 1.0 if hand in PREMIUM else 0.0
    return q.base_strategy(n, pos, hist, hand, stack_bb)


def _calling_station(q, n, pos, hist, hand, stack_bb):
    if "A" in hist:                                  # facing a jam: call far too wide
        return 0.85
    return q.base_strategy(n, pos, hist, hand, stack_bb)


def _depth_blind(q, n, pos, hist, hand, stack_bb):
    return q.base_strategy(n, pos, hist, hand, 8.0)  # ignores true depth (leaks off 8bb)


GTO = Bot("gto", _gto, leaky=False)
OVERFOLDER = Bot("overfolder", _overfolder, leaky=True)
CALLING_STATION = Bot("calling_station", _calling_station, leaky=True)
DEPTH_BLIND = Bot("depth_blind", _depth_blind, leaky=True)

FIELDS: dict[str, list[Bot]] = {
    "all_leaky": [OVERFOLDER, CALLING_STATION, DEPTH_BLIND],
    "mixed": [OVERFOLDER, GTO, CALLING_STATION],
    "all_gto": [GTO, GTO, GTO],
    "overfolder_only": [OVERFOLDER],
    "station_only": [CALLING_STATION],
}

ARCHETYPES: dict[str, Bot] = {b.name: b for b in (GTO, OVERFOLDER, CALLING_STATION, DEPTH_BLIND)}


class GearSwitcher:
    """Adversary: plays ``before`` until ``switch_hand``, then ``after`` — exercises the
    dual-EWMA change detection end-to-end (recovery half-life is the metric)."""

    def __init__(self, before: Bot, after: Bot, switch_hand: int):
        self.name = f"gear:{before.name}->{after.name}@{switch_hand}"
        self.leaky = True
        self._before, self._after, self._switch = before, after, int(switch_hand)
        self.hand_index = 0  # advanced by the arena each hand

    def push_fn(self, q, n, pos, hist, hand, stack_bb):
        bot = self._before if self.hand_index < self._switch else self._after
        return bot.push_fn(q, n, pos, hist, hand, stack_bb)


def seat_assignment(bots: list[Bot], non_agent_seats: list[str], hand_index: int):
    """Per-hand bot at each non-agent seat + a stable per-BOT identity.

    Rotation by hand index guarantees every bot appears at every seat over a run; the identity
    key is the bot's index in the field (NOT the seat slot), so the profile DB accumulates one
    coherent profile per bot."""
    out = {}
    k = len(bots)
    for slot, seat in enumerate(non_agent_seats):
        bi = (slot + hand_index) % k
        out[seat] = (bots[bi], f"bot:{bi}:{bots[bi].name}")
    return out
