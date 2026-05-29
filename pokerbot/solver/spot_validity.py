"""Hard scope guard: only genuinely 2-player-valid spots get solver labels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pokerbot.engine.game import GameState, NUM_PLAYERS, Street


class SpotClass(str, Enum):
    HU_POSTFLOP_2P = "hu_postflop_2p"
    PSEUDO_HU_BUNCHING = "pseudo_hu_bunching"
    MULTIWAY_6MAX = "multiway_6max"
    DEFERRED_PREFLOP = "deferred_preflop_v1"


@dataclass(frozen=True, slots=True)
class SpotNode:
    """Decision node candidate for solver labeling."""

    street: Street
    live_seats: frozenset[int]
    folded_seats: frozenset[int]
    hero_seat: int
    spot_class: SpotClass
    bunching: bool = False


@dataclass(frozen=True, slots=True)
class ValidityResult:
    valid: bool
    reason: str
    spot_class: SpotClass | None = None


_POSTFLOP_STREETS = frozenset({Street.FLOP, Street.TURN, Street.RIVER})


def _live_seats_from_state(state: GameState) -> frozenset[int]:
    return frozenset(p.seat for p in state.players if not p.folded)


def _folded_seats_from_state(state: GameState) -> frozenset[int]:
    return frozenset(p.seat for p in state.players if p.folded)


def spot_node_from_state(
    state: GameState,
    hero_seat: int,
    *,
    bunching: bool = False,
) -> SpotNode:
    live = _live_seats_from_state(state)
    folded = _folded_seats_from_state(state)
    if bunching and len(live) == 2:
        spot_class = SpotClass.PSEUDO_HU_BUNCHING
    elif len(live) == 2 and state.street in _POSTFLOP_STREETS:
        spot_class = SpotClass.HU_POSTFLOP_2P
    elif state.street == Street.PREFLOP:
        spot_class = SpotClass.DEFERRED_PREFLOP
    else:
        spot_class = SpotClass.MULTIWAY_6MAX
    return SpotNode(
        street=state.street,
        live_seats=live,
        folded_seats=folded,
        hero_seat=hero_seat,
        spot_class=spot_class,
        bunching=bunching,
    )


def is_two_player_valid(node: SpotNode, *, allow_preflop: bool = False) -> ValidityResult:
    """Return whether this node may receive postflop-solver GTO labels.

    Rules (all must pass):
    1. Exactly two live (non-folded) players.
    2. All other seats are folded — no players behind in a full ring.
    3. Postflop only in v1 (unless allow_preflop=True for future HU preflop DCFR).
    4. Hero must be one of the two live players.
    """
    if len(node.live_seats) != 2:
        return ValidityResult(
            False,
            f"multiway_live_count_{len(node.live_seats)}",
            SpotClass.MULTIWAY_6MAX,
        )

    if node.hero_seat not in node.live_seats:
        return ValidityResult(False, "hero_not_live", node.spot_class)

    non_live = set(range(NUM_PLAYERS)) - node.live_seats
    not_folded = non_live - node.folded_seats
    if not_folded:
        return ValidityResult(
            False,
            "inactive_seats_not_folded",
            SpotClass.MULTIWAY_6MAX,
        )

    if node.street == Street.PREFLOP:
        if not allow_preflop:
            return ValidityResult(
                False,
                "deferred_preflop_v1",
                SpotClass.DEFERRED_PREFLOP,
            )
        return ValidityResult(True, "hu_preflop_collapsed", SpotClass.DEFERRED_PREFLOP)

    if node.street not in _POSTFLOP_STREETS:
        return ValidityResult(False, "not_postflop", node.spot_class)

    if node.street == Street.SHOWDOWN:
        return ValidityResult(False, "showdown_terminal", node.spot_class)

    return ValidityResult(True, "hu_postflop_valid", node.spot_class)


def is_two_player_valid_state(
    state: GameState,
    hero_seat: int,
    *,
    bunching: bool = False,
    allow_preflop: bool = False,
) -> ValidityResult:
    """Convenience wrapper building SpotNode from GameState."""
    node = spot_node_from_state(state, hero_seat, bunching=bunching)
    return is_two_player_valid(node, allow_preflop=allow_preflop)
