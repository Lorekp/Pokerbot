"""Config-driven enumeration of 2-player-valid postflop solver spots."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.cards import Card
from pokerbot.engine.game import (
    BB_BB,
    NUM_PLAYERS,
    SB_BB,
    GameState,
    Position,
    Street,
)
from pokerbot.solver.spot_validity import (
    SpotClass,
    ValidityResult,
    is_two_player_valid_state,
    spot_node_from_state,
)

POSITION_TO_SEAT_OFFSET = {
    Position.UTG: 3,
    Position.HJ: 4,
    Position.CO: 5,
    Position.BTN: 0,
    Position.SB: 1,
    Position.BB: 2,
}


def seat_for_position(button_seat: int, position: Position) -> int:
    """Seat index for position when button is at button_seat."""
    order = [Position.BTN, Position.SB, Position.BB, Position.UTG, Position.HJ, Position.CO]
    btn_idx = order.index(Position.BTN)
    pos_idx = order.index(position)
    return (button_seat + (pos_idx - btn_idx)) % NUM_PLAYERS


@dataclass
class SolverSpotConfig:
    seed: int = 42
    stack_depths_bb: list[float] = field(default_factory=lambda: [100.0])
    boards: dict[str, Any] = field(default_factory=dict)
    bet_tree: dict[str, Any] = field(default_factory=dict)
    solver: dict[str, Any] = field(default_factory=dict)
    decision_nodes: dict[str, Any] = field(default_factory=dict)
    ranges: dict[str, str] = field(default_factory=dict)
    hero_hands: list[list[str]] = field(default_factory=list)
    spots: list[dict[str, Any]] = field(default_factory=list)
    enumeration_mode: str = "interleaved"
    disable_bunching: bool = False


@dataclass
class SolverSpot:
    """A validated decision point ready for postflop-solver labeling."""

    spot_id: str
    state: GameState
    hero_seat: int
    hero_hole: tuple[Card, Card]
    oop_seat: int
    ip_seat: int
    oop_range: str
    ip_range: str
    board_cards: list[Card]
    turn_card: Card | None
    river_card: Card | None
    starting_pot_bb: float
    effective_stack_bb: float
    stack_depth_bb: float
    action_path: list[str]
    bunching: bool
    spot_class: SpotClass
    preflop_line: str
    postflop_line: str
    validity: ValidityResult


def spot_key(spot: SolverSpot) -> str:
    return f"{spot.spot_id}|{''.join(c.to_str() for c in spot.board_cards)}|hero={spot.hero_seat}"


def load_solver_config(path: Path | str) -> SolverSpotConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SolverSpotConfig(
        seed=int(raw.get("seed", 42)),
        stack_depths_bb=[float(x) for x in raw.get("stack_depths_bb", [100])],
        boards=raw.get("boards", {}),
        bet_tree=raw.get("bet_tree", {}),
        solver=raw.get("solver", {}),
        decision_nodes=raw.get("decision_nodes", {}),
        ranges=raw.get("ranges", {}),
        hero_hands=raw.get("hero_hands", [["Ah", "Kd"]]),
        spots=raw.get("spots", []),
        enumeration_mode=str(raw.get("enumeration_mode", "interleaved")),
        disable_bunching=bool(raw.get("disable_bunching", False)),
    )


def sample_boards(config: SolverSpotConfig, rng: random.Random) -> list[list[Card]]:
    boards_cfg = config.boards
    strategy = boards_cfg.get("strategy", "fixed_list")
    if strategy == "fixed_list":
        boards = boards_cfg.get("fixed_boards", ["Td9d6h"])
        return [[Card.from_str(b[i : i + 2]) for i in range(0, len(b), 2)] for b in boards]
    if strategy == "stratified_sample":
        count = int(boards_cfg.get("count", 50))
        sample_seed = int(boards_cfg.get("seed", config.seed))
        exclude = {str(b) for b in boards_cfg.get("exclude_boards", [])}
        sample_rng = random.Random(sample_seed)
        return _sample_unique_flop_boards(count, sample_rng, exclude)
    return [[Card.from_str("Td"), Card.from_str("9d"), Card.from_str("6h")]]


def _sample_unique_flop_boards(
    count: int,
    rng: random.Random,
    exclude: set[str],
) -> list[list[Card]]:
    """Sample ``count`` unique 3-card flop boards (strings like Td9d6h)."""
    from pokerbot.engine.cards import Deck

    boards: list[list[Card]] = []
    seen = set(exclude)
    attempts = 0
    max_attempts = max(count * 200, count + len(exclude))
    while len(boards) < count and attempts < max_attempts:
        attempts += 1
        deck = Deck(rng)
        flop = deck.deal(3)
        key = "".join(c.to_str() for c in flop)
        if key in seen:
            continue
        seen.add(key)
        boards.append(flop)
    if len(boards) < count:
        raise ValueError(
            f"stratified_sample: only generated {len(boards)}/{count} unique flop boards "
            f"after {attempts} attempts (exclude={len(exclude)})"
        )
    return boards


def _assign_hole(state: GameState, seat: int, hole: tuple[Card, Card]) -> None:
    state.players[seat].hole = hole


def _fold_seat(state: GameState, seat: int) -> None:
    state.players[seat].folded = True
    state.players[seat].hole = None


def build_sb_vs_bb_limp_pot(stack_bb: float) -> tuple[GameState, int, int, float]:
    """SB limp, BB check — flop HU blind battle."""
    button_seat = 3
    sb = seat_for_position(button_seat, Position.SB)
    bb = seat_for_position(button_seat, Position.BB)
    rng = random.Random(0)
    state = GameState.new_hand(button_seat=button_seat, rng=rng)
    for seat in range(NUM_PLAYERS):
        if seat not in (sb, bb):
            _fold_seat(state, seat)
    state.players[sb].stack_bb = stack_bb - SB_BB
    state.players[bb].stack_bb = stack_bb - BB_BB
    state.pot_bb = SB_BB + BB_BB
    state.current_bet_bb = BB_BB
    state.street = Street.PREFLOP
    state.action_seat = sb
    state.apply_action(Action(ActionType.CALL), seat=sb)
    state.advance_to_next_actor()
    state.apply_action(Action(ActionType.CHECK), seat=bb)
    pot = state.pot_bb
    oop, ip = sb, bb
    return state, oop, ip, pot


def build_btn_open_bb_call(stack_bb: float) -> tuple[GameState, int, int, float]:
    """BTN open 2.5bb, folds to BB, BB call."""
    button_seat = 3
    btn = seat_for_position(button_seat, Position.BTN)
    sb = seat_for_position(button_seat, Position.SB)
    bb = seat_for_position(button_seat, Position.BB)
    rng = random.Random(0)
    state = GameState.new_hand(button_seat=button_seat, rng=rng)
    open_total = 2.5
    for seat in range(NUM_PLAYERS):
        if seat not in (btn, sb, bb):
            _fold_seat(state, seat)
    _fold_seat(state, sb)
    state.players[btn].stack_bb = stack_bb
    state.players[bb].stack_bb = stack_bb
    state.pot_bb = SB_BB + BB_BB
    state.current_bet_bb = BB_BB
    state.street = Street.PREFLOP
    state.action_seat = btn
    raise_amt = open_total - state.players[btn].committed_street_bb
    state.apply_action(Action(ActionType.RAISE, round(raise_amt, 2)), seat=btn)
    state.advance_to_next_actor()
    state.apply_action(Action(ActionType.CALL), seat=bb)
    pot = state.pot_bb
    oop, ip = bb, btn
    return state, oop, ip, pot


def apply_flop_board(state: GameState, board: list[Card]) -> None:
    state.board = list(board)
    state.street = Street.FLOP
    for p in state.players:
        p.committed_street_bb = 0.0
    state.current_bet_bb = 0.0
    state._acted_this_street.clear()
    state.action_seat = state._first_postflop_actor()


def effective_stack_bb(state: GameState, seats: tuple[int, int]) -> float:
    return min(state.players[s].stack_bb for s in seats)


def pick_hero_hand(
    hero_hands: list[list[str]],
    board: list[Card],
    rng: random.Random,
) -> tuple[Card, Card]:
    board_set = {c.index for c in board}
    candidates = []
    for pair in hero_hands:
        c0 = Card.from_str(pair[0])
        c1 = Card.from_str(pair[1])
        if c0.index not in board_set and c1.index not in board_set:
            candidates.append((c0, c1))
    if not candidates:
        c0, c1 = Card.from_str("Ah"), Card.from_str("Kd")
        return c0, c1
    return rng.choice(candidates)


def _active_spot_configs(config: SolverSpotConfig) -> list[dict[str, Any]]:
    spots = list(config.spots)
    if config.disable_bunching:
        spots = [s for s in spots if not bool(s.get("bunching", False))]
    return spots


def _yield_spot_unit(
    spot_cfg: dict[str, Any],
    *,
    stack_bb: float,
    board: list[Card],
    default_oop: str,
    default_ip: str,
    bunching_oop: str,
    bunching_ip: str,
    hero_hands: list[list[str]],
    rng: random.Random,
) -> Iterator[SolverSpot]:
    preflop_key = str(spot_cfg.get("preflop_line", ""))
    if preflop_key == "fold_to_sb_bb_limp":
        state, oop, ip, pot = build_sb_vs_bb_limp_pot(stack_bb)
        preflop_line = "UTG/fold/HJ/fold/CO/fold/BTN/fold/SB/call/BB/check"
    elif preflop_key == "btn_open_bb_call":
        state, oop, ip, pot = build_btn_open_bb_call(stack_bb)
        preflop_line = "UTG/fold/HJ/fold/CO/fold/BTN/raise_2.5/SB/fold/BB/call"
    else:
        return

    spot_id = str(spot_cfg["id"])
    bunching = bool(spot_cfg.get("bunching", False))
    apply_flop_board(state, board)
    eff = effective_stack_bb(state, (oop, ip))
    oop_range = bunching_oop if bunching else default_oop
    ip_range = bunching_ip if bunching else default_ip

    for hero_seat in (oop, ip):
        hero_hole = pick_hero_hand(hero_hands, board, rng)
        _assign_hole(state, hero_seat, hero_hole)
        other = ip if hero_seat == oop else oop
        _assign_hole(state, other, (Card.from_str("2c"), Card.from_str("3d")))

        validity = is_two_player_valid_state(state, hero_seat, bunching=bunching)
        node = spot_node_from_state(state, hero_seat, bunching=bunching)
        postflop_line = "OOP_CHECK" if hero_seat == oop else ""

        yield SolverSpot(
            spot_id=spot_id,
            state=state.clone(),
            hero_seat=hero_seat,
            hero_hole=hero_hole,
            oop_seat=oop,
            ip_seat=ip,
            oop_range=oop_range,
            ip_range=ip_range,
            board_cards=list(board),
            turn_card=None,
            river_card=None,
            starting_pot_bb=pot,
            effective_stack_bb=eff,
            stack_depth_bb=stack_bb,
            action_path=[],
            bunching=bunching,
            spot_class=node.spot_class,
            preflop_line=preflop_line,
            postflop_line=postflop_line,
            validity=validity,
        )


def enumerate_spots(config: SolverSpotConfig) -> Iterator[SolverSpot]:
    """Yield candidate spots; caller filters with validity predicate."""
    rng = random.Random(config.seed)
    boards = sample_boards(config, rng)
    default_oop = config.ranges.get("default_oop", "66+")
    default_ip = config.ranges.get("default_ip", "QQ-22")
    bunching_oop = config.ranges.get("bunching_oop", default_oop)
    bunching_ip = config.ranges.get("bunching_ip", default_ip)
    spot_cfgs = _active_spot_configs(config)

    if config.enumeration_mode == "legacy_sequential":
        for spot_cfg in spot_cfgs:
            for stack_bb in config.stack_depths_bb:
                for board in boards:
                    yield from _yield_spot_unit(
                        spot_cfg,
                        stack_bb=stack_bb,
                        board=board,
                        default_oop=default_oop,
                        default_ip=default_ip,
                        bunching_oop=bunching_oop,
                        bunching_ip=bunching_ip,
                        hero_hands=config.hero_hands,
                        rng=rng,
                    )
        return

    # interleaved: stack → board → spot_type → hero
    for stack_bb in config.stack_depths_bb:
        for board in boards:
            for spot_cfg in spot_cfgs:
                yield from _yield_spot_unit(
                    spot_cfg,
                    stack_bb=stack_bb,
                    board=board,
                    default_oop=default_oop,
                    default_ip=default_ip,
                    bunching_oop=bunching_oop,
                    bunching_ip=bunching_ip,
                    hero_hands=config.hero_hands,
                    rng=rng,
                )


def enumerate_valid_spots(config: SolverSpotConfig) -> tuple[list[SolverSpot], dict[str, int]]:
    """Enumerate and partition by validity."""
    included: list[SolverSpot] = []
    excluded_by_reason: dict[str, int] = {}

    for spot in enumerate_spots(config):
        if spot.validity.valid:
            included.append(spot)
        else:
            excluded_by_reason[spot.validity.reason] = (
                excluded_by_reason.get(spot.validity.reason, 0) + 1
            )

    return included, excluded_by_reason


def enumerate_spot_keys(config: SolverSpotConfig) -> list[str]:
    """Return spot_key strings in enumeration order (validity not filtered)."""
    return [spot_key(s) for s in enumerate_spots(config)]
