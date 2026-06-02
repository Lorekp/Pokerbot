"""Replay NLHE hand scenarios through pokerkit and JAX; compare terminal payoffs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np

from pokerbot.engine.cards import Card as BotCard, Rank, Suit
from pokerbot.engine.jax_engine import (
    ALL_IN,
    BB_CHIPS,
    CALL,
    CHECK,
    CHIP_UNIT,
    FOLD,
    NUM_ACTIONS,
    NUM_PLAYERS,
    SB_CHIPS,
    SEAT_EMPTY,
    STREET_TERMINAL,
    apply_action,
    init_hand,
    is_chance_node,
    legal_action_mask,
    sample_chance,
)
from pokerbot.engine.side_pots_jax import SEAT_ALLIN, SEAT_FOLDED, SEAT_LIVE

try:
    from pokerkit import Automation, Mode, NoLimitTexasHoldem
    from pokerkit.utilities import Card as PKCard
except ImportError:  # pragma: no cover
    NoLimitTexasHoldem = None  # type: ignore[misc, assignment]
    PKCard = None  # type: ignore[misc, assignment]

ActionKind = Literal["fold", "call", "raise_to", "check"]


@dataclass(frozen=True)
class ReplayScenario:
    """Fixed cards + seat-indexed betting for cross-engine replay."""

    name: str
    stacks: tuple[int, ...]
    holes: tuple[tuple[int, int], ...]  # card indices per seat (JAX convention)
    board: tuple[int, ...]
    actions: tuple[tuple[ActionKind, int | None], ...]
    occupancy: tuple[bool, ...] = (True, True, True, True, True, True)
    button: int = 0


def idx_to_pk_str(card_idx: int) -> str:
    rank = card_idx // 4 + 2
    suit = card_idx % 4
    return BotCard(Rank(rank), Suit(suit)).to_str()


def _pk_card(card_idx: int):
    assert PKCard is not None
    return next(PKCard.parse(idx_to_pk_str(card_idx)))


def _pokerkit_automations() -> tuple:
    assert Automation is not None
    return (
        Automation.ANTE_POSTING,
        Automation.BET_COLLECTION,
        Automation.BLIND_OR_STRADDLE_POSTING,
        Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
        Automation.CHIPS_PUSHING,
        Automation.CHIPS_PULLING,
    )


def _in_hand_pokerkit(state, seat: int) -> bool:
    """True if seat has not folded (pokerkit ``statuses`` entry)."""
    return bool(state.statuses[seat])


def _settle_pokerkit_payoffs(state, scenario: ReplayScenario) -> np.ndarray:
    """Award pots using pokerkit hand strength (independent of JAX side_pots_jax)."""
    from pokerkit.hands import StandardHighHand

    n = len(scenario.stacks)
    starting = list(scenario.stacks)
    put_in = [starting[i] - int(state.stacks[i]) for i in range(n)]
    winnings = [0] * n
    board_str = "".join(idx_to_pk_str(c) for c in scenario.board)
    hand_by_seat = {}
    for i, (c0, c1) in enumerate(scenario.holes):
        hole_str = idx_to_pk_str(c0) + idx_to_pk_str(c1)
        hand_by_seat[i] = StandardHighHand.from_game(hole_str, board_str)

    for pot in state.pots:
        amount = int(pot.unraked_amount + pot.raked_amount)
        if amount <= 0:
            continue
        eligible = tuple(i for i in pot.player_indices if _in_hand_pokerkit(state, i))
        if not eligible:
            continue
        best = max(hand_by_seat[i] for i in eligible)
        winners = [i for i in eligible if hand_by_seat[i] == best]
        share = amount // len(winners)
        rem = amount - share * len(winners)
        for w in winners:
            winnings[w] += share
        if rem:
            winnings[winners[0]] += rem

    return np.asarray([winnings[i] - put_in[i] for i in range(n)], dtype=np.int64)


def _advance_pokerkit_to_terminal(state, board: tuple[int, ...]) -> None:
    bi = 0
    for _ in range(200):
        if state.can_collect_bets():
            state.collect_bets()
        elif state.can_select_runout_count():
            state.select_runout_count(1)
        elif state.can_burn_card():
            state.burn_card()
        elif state.can_deal_board():
            n = int(state.board_dealing_count or 0)
            if n > 0 and bi + n <= len(board):
                state.deal_board(tuple(_pk_card(c) for c in board[bi : bi + n]))
                bi += n
            else:
                state.deal_board()
        elif state.can_show_or_muck_hole_cards():
            state.show_or_muck_hole_cards()
        elif state.can_push_chips():
            state.push_chips()
        elif state.can_pull_chips():
            state.pull_chips()
        elif state.status:
            break
        else:
            break


def replay_pokerkit(scenario: ReplayScenario) -> np.ndarray:
    """Return per-seat chip deltas (int) from pokerkit payoffs."""
    if NoLimitTexasHoldem is None:
        raise ImportError("pokerkit is required for replay reference tests")

    n = len(scenario.stacks)
    state = NoLimitTexasHoldem.create_state(
        _pokerkit_automations(),
        False,
        0,
        (SB_CHIPS, BB_CHIPS),
        BB_CHIPS,
        scenario.stacks,
        n,
        mode=Mode.CASH_GAME,
    )
    for seat, (c0, c1) in enumerate(scenario.holes):
        state.deal_hole((_pk_card(c0), _pk_card(c1)), seat)

    for kind, amount in scenario.actions:
        if state.status and state.actor_index is None:
            break
        if kind == "fold":
            state.fold()
        elif kind in ("call", "check"):
            state.check_or_call()
        elif kind == "raise_to":
            assert amount is not None
            state.complete_bet_or_raise_to(int(amount))
        else:
            raise ValueError(f"unknown action {kind!r}")

    _advance_pokerkit_to_terminal(state, scenario.board)
    payoffs = _settle_pokerkit_payoffs(state, scenario)
    if payoffs.shape[0] < NUM_PLAYERS:
        out = np.zeros(NUM_PLAYERS, dtype=np.int64)
        out[: len(payoffs)] = payoffs
        return out
    return payoffs[:NUM_PLAYERS]


def _py_first_live_after(seat_status: np.ndarray, from_seat: int) -> int:
    for offset in range(1, NUM_PLAYERS + 1):
        seat = (from_seat + offset) % NUM_PLAYERS
        if int(seat_status[seat]) == int(SEAT_LIVE):
            return seat
    return -1


def _py_can_act(state, seat: int) -> bool:
    return int(state.seat_status[seat]) == int(SEAT_LIVE)


def _py_street_complete(state) -> bool:
    can_act = [_py_can_act(state, s) for s in range(NUM_PLAYERS)]
    n_can_act = sum(can_act)
    matched = all(
        int(state.committed_street[s]) >= int(state.current_bet)
        for s in range(NUM_PLAYERS)
        if can_act[s]
    )
    all_acted = all(bool(state.acted[s]) for s in range(NUM_PLAYERS) if can_act[s])
    back_to_aggressor = int(state.action_seat) == int(state.last_aggressor)
    no_aggressor = int(state.last_aggressor) < 0
    return n_can_act == 0 or (matched and all_acted and (back_to_aggressor or no_aggressor))


def _py_close_street(state):
    pot_dead = int(state.pot_dead) + int(np.sum(state.committed_street))
    return state._replace(
        pot_dead=jnp.int32(pot_dead),
        committed_street=jnp.zeros(NUM_PLAYERS, dtype=jnp.int32),
        current_bet=jnp.int32(0),
        pending_deal=jnp.bool_(True),
        action_seat=jnp.int32(-1),
    )


def _py_live_count(seat_status: np.ndarray) -> int:
    return int(np.sum((seat_status == int(SEAT_LIVE)) | (seat_status == int(SEAT_ALLIN))))


def _apply_replay_action(state, kind: ActionKind, amount: int | None):
    """Apply pokerkit-style action with exact raise_to totals (replay-only)."""
    from pokerbot.engine.jax_engine import MAX_HISTORY, STREET_TERMINAL
    from pokerbot.engine.side_pots_jax import compute_returns

    seat = int(state.action_seat)
    if kind == "fold":
        seat_status = state.seat_status.at[seat].set(SEAT_FOLDED)
        state = state._replace(seat_status=seat_status)
        chip_amount = 0
        is_call = False
        is_aggressive = False
        action_code = FOLD
    elif kind in ("call", "check"):
        mask = np.asarray(legal_action_mask(state), dtype=bool)
        if kind == "check":
            assert mask[CHECK], "illegal check in replay scenario"
            action_code = CHECK
            chip_amount = 0
        else:
            committed = int(state.committed_street[seat])
            to_call = max(0, int(state.current_bet) - committed)
            stack = int(state.stacks[seat])
            if to_call == 0 and mask[CHECK]:
                action_code = CHECK
                chip_amount = 0
            else:
                assert mask[CALL], "illegal call in replay scenario"
                action_code = CALL
                chip_amount = min(stack, to_call)
        is_call = action_code == CALL
        is_aggressive = False
    else:
        assert kind == "raise_to" and amount is not None
        action_code = ALL_IN
        committed = int(state.committed_street[seat])
        stack = int(state.stacks[seat])
        target = int(amount)
        new_street_total = min(target, committed + stack)
        chip_amount = new_street_total - committed
        is_call = False
        is_aggressive = True

    if chip_amount > 0:
        chip_amount = min(chip_amount, int(state.stacks[seat]))
        stacks = state.stacks.at[seat].add(-chip_amount)
        committed_street = state.committed_street.at[seat].add(chip_amount)
        committed_total = state.committed_total.at[seat].add(chip_amount)
        all_in = int(stacks[seat]) <= 0
        seat_status = state.seat_status
        if all_in:
            seat_status = seat_status.at[seat].set(SEAT_ALLIN)
        state = state._replace(
            stacks=stacks,
            committed_street=committed_street,
            committed_total=committed_total,
            seat_status=seat_status,
        )

    prev_bet = int(state.current_bet)
    new_street_bet = int(state.committed_street[seat])
    raised = is_aggressive and new_street_bet > prev_bet
    full_raise = raised and new_street_bet >= prev_bet + int(state.min_raise)
    acted = np.asarray(state.acted, dtype=bool).copy()
    acted[seat] = True
    if full_raise:
        acted[:] = False
        acted[seat] = True
    current_bet = new_street_bet if raised else int(state.current_bet)
    raise_size = (new_street_bet - prev_bet) if raised else int(state.min_raise)
    min_raise = max(BB_CHIPS, raise_size) if raised else int(state.min_raise)
    last_aggressor = seat if raised else int(state.last_aggressor)
    last_raise_full = full_raise if raised else bool(state.last_raise_full)

    idx = int(state.hist_len)
    state = state._replace(
        acted=jnp.asarray(acted, dtype=bool),
        current_bet=jnp.int32(current_bet),
        min_raise=jnp.int32(min_raise),
        last_aggressor=jnp.int32(last_aggressor),
        last_raise_full=jnp.bool_(last_raise_full),
        hist_seat=state.hist_seat.at[idx].set(seat),
        hist_action=state.hist_action.at[idx].set(action_code),
        hist_amount=state.hist_amount.at[idx].set(chip_amount),
        hist_len=jnp.minimum(idx + 1, MAX_HISTORY),
    )

    seat_status_np = np.asarray(state.seat_status)
    n_live = _py_live_count(seat_status_np)
    terminal = n_live <= 1
    if terminal:
        state = state._replace(street=jnp.int32(STREET_TERMINAL), pending_deal=jnp.bool_(False))
    elif _py_street_complete(state):
        state = _py_close_street(state)

    seat_status_np = np.asarray(state.seat_status)
    n_live = _py_live_count(seat_status_np)
    terminal = n_live <= 1 or int(state.street) >= STREET_TERMINAL
    action_seat = -1 if terminal or bool(state.pending_deal) else _py_first_live_after(
        seat_status_np, int(state.action_seat)
    )
    rets = jnp.zeros(NUM_PLAYERS, dtype=jnp.int32)
    if terminal:
        rets = compute_returns(
            holes=state.holes,
            board=state.board,
            seat_status=state.seat_status,
            committed_total=state.committed_total,
            button=state.button,
            n_showdown=jnp.int32(n_live),
        )
    return state._replace(action_seat=jnp.int32(action_seat), done=jnp.bool_(terminal), rewards=rets)


def _build_deck_for_scenario(scenario: ReplayScenario) -> np.ndarray:
    used = set(scenario.board)
    for h in scenario.holes:
        used.update(h)
    deck = list(scenario.board)
    for h in scenario.holes:
        deck.extend(h)
    for i in range(52):
        if i not in used:
            deck.append(i)
    return np.asarray(deck[:52], dtype=np.int32)


def replay_jax(scenario: ReplayScenario) -> np.ndarray:
    """Return per-seat chip deltas (int32) from JAX env terminal returns."""
    import jax

    occ = jnp.asarray(scenario.occupancy, dtype=bool)
    stacks = jnp.asarray(
        [s if occ[i] else 0 for i, s in enumerate(scenario.stacks)] + [0] * (NUM_PLAYERS - len(scenario.stacks)),
        dtype=jnp.int32,
    )[:NUM_PLAYERS]
    key = jax.random.PRNGKey(0)
    state = init_hand(key, button=jnp.int32(scenario.button), stacks=stacks, occupancy=occ)
    deck = _build_deck_for_scenario(scenario)
    holes = np.full((NUM_PLAYERS, 2), -1, dtype=np.int32)
    for i, (c0, c1) in enumerate(scenario.holes):
        if i < NUM_PLAYERS and scenario.occupancy[i]:
            holes[i, 0] = c0
            holes[i, 1] = c1
    board = np.full(5, -1, dtype=np.int32)
    for i, c in enumerate(scenario.board):
        board[i] = c
    state = state._replace(deck=jnp.asarray(deck), holes=jnp.asarray(holes), board=jnp.asarray(board))

    for kind, amount in scenario.actions:
        if bool(state.done):
            break
        while bool(is_chance_node(state)):
            state = sample_chance(state)
        if bool(state.done):
            break
        state = _apply_replay_action(state, kind, amount)

    while bool(is_chance_node(state)) and not bool(state.done):
        state = sample_chance(state)

    board_arr = jnp.asarray(board, dtype=jnp.int32)
    state = state._replace(
        board=board_arr,
        street=jnp.int32(STREET_TERMINAL),
        pending_deal=jnp.bool_(False),
    )

    if not bool(state.done):
        from pokerbot.engine.side_pots_jax import compute_returns

        n_live = int(jnp.sum((state.seat_status == SEAT_LIVE) | (state.seat_status == SEAT_ALLIN)))
        rets = compute_returns(
            holes=state.holes,
            board=state.board,
            seat_status=state.seat_status,
            committed_total=state.committed_total,
            button=state.button,
            n_showdown=jnp.int32(max(n_live, 1)),
        )
        state = state._replace(done=jnp.bool_(True), rewards=rets, pending_deal=jnp.bool_(False))

    return np.asarray(state.rewards, dtype=np.int64)


def assert_replay_match(scenario: ReplayScenario) -> tuple[np.ndarray, np.ndarray]:
    pk = replay_pokerkit(scenario)
    jax = replay_jax(scenario)
    if pk.sum() != 0:
        raise AssertionError(f"{scenario.name}: pokerkit payoffs not zero-sum: {pk}")
    if jax.sum() != 0:
        raise AssertionError(f"{scenario.name}: JAX returns not zero-sum: {jax}")
    if not np.array_equal(pk[: len(scenario.stacks)], jax[: len(scenario.stacks)]):
        raise AssertionError(
            f"{scenario.name}: payout mismatch\npokerkit={pk[: len(scenario.stacks)]}\njax={jax[: len(scenario.stacks)]}"
        )
    return pk, jax


# Canonical §7.1 cross-engine scenarios
CASE_A = ReplayScenario(
    name="case_a_three_way_distinct_stacks",
    stacks=(10000, 6000, 3000),
    holes=((48, 49), (40, 36), (32, 28)),
    board=(0, 13, 17, 21, 25),
    actions=(
        ("raise_to", 3000),
        ("call", None),
        ("raise_to", 6000),
        ("call", None),
    ),
    occupancy=(True, True, True, False, False, False),
    button=2,
)

CASE_G = ReplayScenario(
    name="case_g_folded_contributor_multi_layer",
    stacks=(10000, 10000, 6000, 4000),
    holes=((48, 44), (48, 49), (40, 36), (32, 28)),
    board=(0, 13, 17, 21, 25),
    actions=(
        ("raise_to", 4000),
        ("call", None),
        ("call", None),
        ("raise_to", 10000),
        ("call", None),
        ("fold", None),
    ),
    occupancy=(True, True, True, True, False, False),
    button=3,
)

CASE_D = ReplayScenario(
    name="case_d_folded_contributor_showdown",
    stacks=(10000, 10000, 500, 10000),
    holes=((48, 44), (40, 36), (0, 4), (32, 28)),
    board=(0, 13, 17, 21, 25),
    actions=(
        ("call", None),
        ("call", None),
        ("call", None),
        ("raise_to", 6000),
        ("fold", None),
        ("call", None),
        ("fold", None),
    ),
    occupancy=(True, True, True, True, False, False),
    button=3,
)
