"""Normalized AoF game state for obs v3 (semantic roles, action history)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from aof_ranges import active_seats
from aof_rl.encode import RANKS, SUITS, _card_indices
from aof_rl.env import AoFHandState
from aof_rl.spot_diagnostics import all_ins_before_hero

SeatRole = Literal["first_to_act", "button", "small_blind", "big_blind"]
ActState = Literal["pending", "folded", "all_in", "acted_fold", "acted_all_in"]
Action = Literal["FOLD", "ALL_IN"]

ROLE_ORDER: tuple[SeatRole, ...] = (
    "first_to_act",
    "button",
    "small_blind",
    "big_blind",
)
ROLE_INDEX: dict[SeatRole, int] = {r: i for i, r in enumerate(ROLE_ORDER)}

VALID_TABLE_SIZES = frozenset({2, 3, 4})
KNOWN_SEAT_LABELS = frozenset({"UTG", "BTN", "SB", "BB", "MP", "CO"})


class AoFEncodeError(ValueError):
    """Raised when a state cannot be encoded (malformed or unsupported)."""


@dataclass(frozen=True)
class SeatState:
    seat_id: int
    role: SeatRole
    is_hero: bool
    state: ActState
    committed_bb: float
    stack_remaining_bb: float
    blind_posted_bb: float
    shover_rank: int
    eligible_pot_mask: int


@dataclass(frozen=True)
class AoFGameState:
    variant: Literal["NLH_AOF"]
    table_size: int
    hero_seat_id: int
    hero_cards: tuple[int, int]
    action_history: tuple[tuple[int, Action], ...]
    seats: tuple[SeatState, ...]
    small_blind_bb: float
    big_blind_bb: float
    effective_stack_bb: float
    flat_rake_bb: float
    pct_rake: float
    rake_cap_bb: float
    equal_stacks: bool
    side_pots_present: bool


def card_to_index(card: str) -> int:
    ri, si = _card_indices(card)
    return ri * len(SUITS) + si


def index_to_card(idx: int) -> str:
    if idx < 0 or idx > 51:
        raise AoFEncodeError(f"card index out of range: {idx}")
    return f"{RANKS[idx // 4]}{SUITS[idx % 4]}"


def hand_label_index(cards: tuple[int, int]) -> int:
    """169-class index: pairs 0-12, suited 13-90, offsuit 91-168."""
    c1, c2 = cards
    r1, s1 = c1 // 4, c1 % 4
    r2, s2 = c2 // 4, c2 % 4
    lo, hi = (r1, r2) if r1 <= r2 else (r2, r1)
    suited = s1 == s2
    if lo == hi:
        return lo
    below = sum(12 - l for l in range(lo))
    offset = hi - lo - 1
    if suited:
        return 13 + below + offset
    return 91 + below + offset


def seat_label_for_role(role: SeatRole, action_order: tuple[str, ...]) -> str | None:
    """Map canonical role to position label for this table."""
    if not action_order:
        return None
    if role == "first_to_act":
        return action_order[0]
    if role == "small_blind" and "SB" in action_order:
        return "SB"
    if role == "big_blind" and "BB" in action_order:
        return "BB"
    if role == "button":
        if "BTN" in action_order:
            return "BTN"
        if len(action_order) == 2 and "SB" in action_order:
            return "SB"
    return None


def primary_role_for_label(label: str, action_order: tuple[str, ...]) -> SeatRole:
    """Primary semantic role used when storing SeatState on a seat record."""
    if label == action_order[0]:
        return "first_to_act"
    if label == "SB":
        if len(action_order) == 2:
            return "button"
        return "small_blind"
    if label == "BB":
        return "big_blind"
    if label == "BTN":
        return "button"
    if label == "UTG":
        return "first_to_act"
    return "first_to_act"


def _act_state_for(seat: str, state: AoFHandState, hero: str) -> ActState:
    if state.terminal:
        if seat in state.folded:
            return "acted_fold" if seat in state.acted else "folded"
        if seat in state.all_in:
            return "acted_all_in" if seat in state.acted else "all_in"
        return "pending"
    if seat == state.action_seat:
        return "pending"
    if seat in state.folded:
        return "acted_fold" if seat in state.acted else "folded"
    if seat in state.all_in:
        return "acted_all_in" if seat in state.acted else "all_in"
    if seat in state.acted:
        return "acted_fold"
    return "pending"


def _blind_posted_bb(label: str, stack_bb: float) -> float:
    if label == "SB":
        return min(0.5, stack_bb)
    if label == "BB":
        return min(1.0, stack_bb)
    return 0.0


def _build_action_history(
    state: AoFHandState, hero: str, seat_id_by_label: dict[str, int]
) -> tuple[tuple[int, Action], ...]:
    if hero not in state.seats:
        return ()
    idx = state.seats.index(hero)
    hist: list[tuple[int, Action]] = []
    for seat in state.seats[:idx]:
        sid = seat_id_by_label[seat]
        if seat in state.all_in:
            hist.append((sid, "ALL_IN"))
        elif seat in state.folded:
            hist.append((sid, "FOLD"))
    return tuple(hist)


def _eligible_pot_mask_for_seat(
    label: str, action_order: tuple[str, ...], *, folded: bool
) -> int:
    """4-bit mask over ROLE_ORDER slots this seat occupies and is eligible for main pot."""
    if folded:
        return 0
    mask = 0
    for role in ROLE_ORDER:
        occupant = seat_label_for_role(role, action_order)
        if occupant == label:
            mask |= 1 << ROLE_INDEX[role]
    return mask


def _validate_hand_state(state: AoFHandState, hero: str) -> None:
    if state.table_size not in VALID_TABLE_SIZES:
        raise AoFEncodeError(f"table_size must be 2, 3, or 4, got {state.table_size}")
    if state.terminal:
        raise AoFEncodeError("cannot encode terminal hand state")
    if state.action_seat is None:
        raise AoFEncodeError("no acting seat")
    if hero not in state.seats:
        raise AoFEncodeError(f"hero {hero!r} not in seats")
    if len(state.seats) != state.table_size:
        raise AoFEncodeError("len(seats) != table_size")
    if len(set(state.seats)) != len(state.seats):
        raise AoFEncodeError("duplicate seat labels")
    for label in state.seats:
        if label not in KNOWN_SEAT_LABELS:
            raise AoFEncodeError(f"unknown seat label {label!r}")
    expected = active_seats(state.table_size)
    if set(state.seats) != set(expected):
        raise AoFEncodeError(f"seats {state.seats!r} do not match table {expected!r}")
    if state.stack_bb <= 0:
        raise AoFEncodeError("stack_bb must be positive")
    prior = all_ins_before_hero(state, hero)
    if len(prior) > 3:
        raise AoFEncodeError("more than 3 prior shovers before hero")
    if hero not in state.holes:
        raise AoFEncodeError("hero has no hole cards")
    c1, c2 = state.holes[hero]
    try:
        i1, i2 = card_to_index(c1), card_to_index(c2)
    except (KeyError, IndexError) as exc:
        raise AoFEncodeError(f"invalid hole card: {exc}") from exc
    if i1 == i2:
        raise AoFEncodeError("hero hole cards must be distinct")
    if i1 < 0 or i1 > 51 or i2 < 0 or i2 > 51:
        raise AoFEncodeError("card index out of range")


def normalize_hand_state(
    state: AoFHandState,
    hero: str,
    *,
    flat_rake_bb: float = 0.0,
    pct_rake: float = 0.0,
    rake_cap_bb: float = 0.0,
    equal_stacks: bool = True,
    side_pots_present: bool = False,
) -> AoFGameState:
    """Build AoFGameState from env hand state (action-order seats)."""
    _validate_hand_state(state, hero)
    if side_pots_present and not equal_stacks:
        raise AoFEncodeError("side pots with unequal stacks are not supported")

    action_order = state.seats
    seat_id_by_label = {label: i for i, label in enumerate(action_order)}
    prior_shovers = all_ins_before_hero(state, hero)
    shover_rank_map = {s: rank for rank, s in enumerate(prior_shovers, start=1)}

    stack = state.stack_bb
    seat_states: list[SeatState] = []
    for i, label in enumerate(action_order):
        inv = state.invested.get(label, 0.0)
        committed = inv
        remaining = max(0.0, stack - inv)
        folded = label in state.folded
        seat_states.append(
            SeatState(
                seat_id=i,
                role=primary_role_for_label(label, action_order),
                is_hero=(label == hero),
                state=_act_state_for(label, state, hero),
                committed_bb=committed,
                stack_remaining_bb=remaining,
                blind_posted_bb=_blind_posted_bb(label, stack),
                shover_rank=shover_rank_map.get(label, 0),
                eligible_pot_mask=_eligible_pot_mask_for_seat(
                    label, action_order, folded=folded
                ),
            )
        )

    c1, c2 = state.holes[hero]
    hero_cards = (card_to_index(c1), card_to_index(c2))

    return AoFGameState(
        variant="NLH_AOF",
        table_size=state.table_size,
        hero_seat_id=seat_id_by_label[hero],
        hero_cards=hero_cards,
        action_history=_build_action_history(state, hero, seat_id_by_label),
        seats=tuple(seat_states),
        small_blind_bb=0.5,
        big_blind_bb=1.0,
        effective_stack_bb=stack,
        flat_rake_bb=flat_rake_bb,
        pct_rake=pct_rake,
        rake_cap_bb=rake_cap_bb,
        equal_stacks=equal_stacks,
        side_pots_present=side_pots_present,
    )


def permute_game_state_seats(
    game: AoFGameState, permutation: tuple[int, ...]
) -> AoFGameState:
    """Reorder seat record tuple only (test helper); seat_id and history unchanged."""
    n = len(game.seats)
    if len(permutation) != n or sorted(permutation) != list(range(n)):
        raise AoFEncodeError("permutation must be a reordering of seat indices")
    new_seats = tuple(game.seats[i] for i in permutation)
    return AoFGameState(
        variant=game.variant,
        table_size=game.table_size,
        hero_seat_id=game.hero_seat_id,
        hero_cards=game.hero_cards,
        action_history=game.action_history,
        seats=new_seats,
        small_blind_bb=game.small_blind_bb,
        big_blind_bb=game.big_blind_bb,
        effective_stack_bb=game.effective_stack_bb,
        flat_rake_bb=game.flat_rake_bb,
        pct_rake=game.pct_rake,
        rake_cap_bb=game.rake_cap_bb,
        equal_stacks=game.equal_stacks,
        side_pots_present=game.side_pots_present,
    )


def seats_by_id(game: AoFGameState) -> dict[int, SeatState]:
    return {s.seat_id: s for s in game.seats}


def action_order_labels(game: AoFGameState) -> tuple[str, ...]:
    """Reconstruct position labels from normalized state (for encoder role lookup)."""
    from aof_ranges import active_seats

    return active_seats(game.table_size)


def game_state_to_dict(game: AoFGameState) -> dict:
    """Serialize AoFGameState for JSON APIs."""
    return {
        "variant": game.variant,
        "table_size": game.table_size,
        "hero_seat_id": game.hero_seat_id,
        "hero_cards": list(game.hero_cards),
        "action_history": [
            {"seat_id": sid, "action": act} for sid, act in game.action_history
        ],
        "seats": [
            {
                "seat_id": s.seat_id,
                "role": s.role,
                "is_hero": s.is_hero,
                "state": s.state,
                "committed_bb": s.committed_bb,
                "stack_remaining_bb": s.stack_remaining_bb,
                "blind_posted_bb": s.blind_posted_bb,
                "shover_rank": s.shover_rank,
                "eligible_pot_mask": s.eligible_pot_mask,
            }
            for s in game.seats
        ],
        "small_blind_bb": game.small_blind_bb,
        "big_blind_bb": game.big_blind_bb,
        "effective_stack_bb": game.effective_stack_bb,
        "flat_rake_bb": game.flat_rake_bb,
        "pct_rake": game.pct_rake,
        "rake_cap_bb": game.rake_cap_bb,
        "equal_stacks": game.equal_stacks,
        "side_pots_present": game.side_pots_present,
    }


def _parse_hero_cards(raw: object) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise AoFEncodeError("hero_cards must be a length-2 list")
    out: list[int] = []
    for item in raw:
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, str):
            out.append(card_to_index(item))
        else:
            raise AoFEncodeError(f"invalid hero card entry: {item!r}")
    if out[0] == out[1]:
        raise AoFEncodeError("hero hole cards must be distinct")
    if any(c < 0 or c > 51 for c in out):
        raise AoFEncodeError("card index out of range")
    return (out[0], out[1])


ACT_STATE_LABELS = (
    "pending",
    "folded",
    "all_in",
    "acted_fold",
    "acted_all_in",
)


def _parse_action_history(raw: object) -> tuple[tuple[int, Action], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise AoFEncodeError("action_history must be a list")
    hist: list[tuple[int, Action]] = []
    for entry in raw:
        if isinstance(entry, dict):
            sid = int(entry["seat_id"])
            act = str(entry["action"]).upper().replace(" ", "_")
            if act == "ALLIN":
                act = "ALL_IN"
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            sid = int(entry[0])
            act = str(entry[1]).upper().replace(" ", "_")
            if act == "ALLIN":
                act = "ALL_IN"
        else:
            raise AoFEncodeError(f"invalid action_history entry: {entry!r}")
        if act not in ("FOLD", "ALL_IN"):
            raise AoFEncodeError(f"invalid action {act!r}")
        hist.append((sid, act))  # type: ignore[arg-type]
    return tuple(hist)


def _parse_seats(raw: object, table_size: int) -> tuple[SeatState, ...]:
    if not isinstance(raw, list):
        raise AoFEncodeError("seats must be a list")
    if len(raw) != table_size:
        raise AoFEncodeError(f"expected {table_size} seats, got {len(raw)}")
    seats: list[SeatState] = []
    for item in raw:
        if not isinstance(item, dict):
            raise AoFEncodeError("each seat must be an object")
        role = str(item.get("role", "first_to_act"))
        if role not in ROLE_ORDER:
            raise AoFEncodeError(f"invalid seat role {role!r}")
        state = str(item.get("state", "pending"))
        if state not in ACT_STATE_LABELS:
            raise AoFEncodeError(f"invalid seat state {state!r}")
        seats.append(
            SeatState(
                seat_id=int(item["seat_id"]),
                role=role,  # type: ignore[arg-type]
                is_hero=bool(item.get("is_hero", False)),
                state=state,  # type: ignore[arg-type]
                committed_bb=float(item.get("committed_bb", 0.0)),
                stack_remaining_bb=float(item.get("stack_remaining_bb", 0.0)),
                blind_posted_bb=float(item.get("blind_posted_bb", 0.0)),
                shover_rank=int(item.get("shover_rank", 0)),
                eligible_pot_mask=int(item.get("eligible_pot_mask", 0)),
            )
        )
    ids = [s.seat_id for s in seats]
    if sorted(ids) != list(range(table_size)):
        raise AoFEncodeError("seat_id values must be 0..table_size-1")
    heroes = [s for s in seats if s.is_hero]
    if len(heroes) != 1:
        raise AoFEncodeError("exactly one seat must have is_hero=true")
    return tuple(seats)


def game_state_from_dict(data: dict) -> AoFGameState:
    """Parse normalized AoF game state JSON into AoFGameState."""
    if not isinstance(data, dict):
        raise AoFEncodeError("game state must be an object")
    variant = data.get("variant", "NLH_AOF")
    if variant != "NLH_AOF":
        raise AoFEncodeError(f"unsupported variant {variant!r}")
    table_size = int(data["table_size"])
    if table_size not in VALID_TABLE_SIZES:
        raise AoFEncodeError(f"table_size must be 2, 3, or 4, got {table_size}")
    hero_seat_id = int(data["hero_seat_id"])
    if hero_seat_id < 0 or hero_seat_id >= table_size:
        raise AoFEncodeError("hero_seat_id out of range")
    hero_cards = _parse_hero_cards(data["hero_cards"])
    action_history = _parse_action_history(data.get("action_history", []))
    seats = _parse_seats(data["seats"], table_size)
    if seats[hero_seat_id].seat_id != hero_seat_id:
        raise AoFEncodeError("hero_seat_id does not match is_hero seat")
    if not seats[hero_seat_id].is_hero:
        raise AoFEncodeError("hero_seat_id must reference the hero seat")
    equal_stacks = bool(data.get("equal_stacks", True))
    side_pots_present = bool(data.get("side_pots_present", False))
    if side_pots_present and not equal_stacks:
        raise AoFEncodeError("side pots with unequal stacks are not supported")
    return AoFGameState(
        variant="NLH_AOF",
        table_size=table_size,
        hero_seat_id=hero_seat_id,
        hero_cards=hero_cards,
        action_history=action_history,
        seats=seats,
        small_blind_bb=float(data.get("small_blind_bb", 0.5)),
        big_blind_bb=float(data.get("big_blind_bb", 1.0)),
        effective_stack_bb=float(data["effective_stack_bb"]),
        flat_rake_bb=float(data.get("flat_rake_bb", 0.0)),
        pct_rake=float(data.get("pct_rake", 0.0)),
        rake_cap_bb=float(data.get("rake_cap_bb", 0.0)),
        equal_stacks=equal_stacks,
        side_pots_present=side_pots_present,
    )
