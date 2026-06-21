"""§7.1 side-pot known-answer suite (Cases A–G)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from pokerbot.engine.jax_engine import NUM_PLAYERS, SEAT_EMPTY
from pokerbot.engine.side_pots_jax import (
    SEAT_ALLIN,
    SEAT_FOLDED,
    SEAT_LIVE,
    award_layer,
    compute_returns as side_pot_returns,
)


def _manual_state(
    *,
    committed: list[int],
    status: list[int],
    holes: np.ndarray | None = None,
    board: np.ndarray | None = None,
    button: int = 0,
) -> dict:
    holes = holes if holes is not None else np.full((NUM_PLAYERS, 2), -1, np.int32)
    board = board if board is not None else np.full(5, -1, np.int32)
    return dict(
        holes=jnp.asarray(holes),
        board=jnp.asarray(board),
        seat_status=jnp.asarray(status, dtype=jnp.int8),
        committed_total=jnp.asarray(committed, dtype=jnp.int32),
        button=jnp.int32(button),
    )


def _scores_from_ranks(ranks: list[int]) -> None:
    """Placeholder — cases use fixed hole cards where needed."""


def test_case_a_three_way_distinct_stacks_uncalled_excess() -> None:
    """Case A: 3-way all-in at distinct stack depths; excess returned to top contributor."""
    committed = [10000, 6000, 3000, 0, 0, 0]
    status = [SEAT_ALLIN, SEAT_ALLIN, SEAT_ALLIN, SEAT_EMPTY, SEAT_EMPTY, SEAT_EMPTY]
    holes = np.array([[48, 44], [40, 36], [32, 28], [-1, -1], [-1, -1], [-1, -1]], dtype=np.int32)
    board = np.array([8, 12, 16, 20, 24], dtype=np.int32)
    kw = _manual_state(committed=committed, status=status, holes=holes, board=board)
    r = np.asarray(side_pot_returns(n_showdown=jnp.int32(3), **kw), dtype=np.int64)
    assert r.sum() == 0
    # phevaluator on board 8,12,16,20,24: seat2=6075010 > seat0==seat1=6075008.
    # Layers: main 9000 (all 3 eligible) -> seat2 scoops; side1 6000 (seats 0&1, tie)
    # -> 3000 each; side2 4000 uncalled -> returned to seat0.
    # seat0 = 3000 + 4000 - 10000 = -3000; seat1 = 3000 - 6000 = -3000; seat2 = 9000 - 3000 = +6000.
    assert r[0] == -3000
    assert r[1] == -3000
    assert r[2] == 6000


def test_case_c_odd_chip() -> None:
    """Case C: odd chip goes to first eligible seat clockwise from button."""
    eligible = jnp.array([True, True, False, False, False, False])
    scores = jnp.array([100, 100, -1, -1, -1, -1], dtype=jnp.int32)
    win = np.asarray(award_layer(jnp.int32(101), eligible, scores, jnp.int32(5)), dtype=np.int64)
    assert int(win.sum()) == 101
    # button=5 -> first eligible seat clockwise is 0.
    assert win[0] == 51
    assert win[1] == 50


def test_case_e_all_in_for_less_no_reopen() -> None:
    """Case E: sub-min raise (all-in-for-less) must not reopen action."""
    import jax
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import ALL_IN, CALL, apply_action, init_hand, legal_action_mask

    # HU: seat 0 short all-in for less than min-raise after seat 1 bet.
    occ = jnp.array([True, True, False, False, False, False])
    key = jax.random.PRNGKey(7)
    state = init_hand(key, occupancy=occ)
    # Force a short-stack spot: seat 0 has 150 chips, facing 100 bet with 200 min-raise.
    state = state._replace(
        stacks=jnp.array([150, 10000, 0, 0, 0, 0], dtype=jnp.int32),
        committed_street=jnp.array([0, 100, 0, 0, 0, 0], dtype=jnp.int32),
        committed_total=state.committed_total.at[1].set(100),
        current_bet=jnp.int32(100),
        min_raise=jnp.int32(100),
        action_seat=jnp.int32(0),
        last_aggressor=jnp.int32(1),
        acted=jnp.array([False, True, False, False, False, False]),
        last_raise_full=jnp.bool_(True),
    )
    mask_before = legal_action_mask(state)
    state = apply_action(state, ALL_IN)  # all-in 150 total (50 raise over 100)
    # Seat 1 already acted facing full raise — should not re-raise.
    state = state._replace(action_seat=jnp.int32(1))
    mask_after = legal_action_mask(state)
    assert bool(mask_after[ALL_IN]) is False
    assert bool(mask_after[1]) or bool(mask_after[2])  # check/call only


def test_case_f_everyone_folds_to_one() -> None:
    """Case F: uncontested win."""
    committed = [100, 50, 0, 0, 0, 0]
    status = [SEAT_LIVE, SEAT_FOLDED, SEAT_FOLDED, SEAT_FOLDED, SEAT_FOLDED, SEAT_FOLDED]
    kw = _manual_state(committed=committed, status=status)
    r = np.asarray(side_pot_returns(n_showdown=jnp.int32(1), **kw), dtype=np.int64)
    assert r.sum() == 0
    assert r[0] == 50  # wins 150, put in 100


def test_case_b_split_main_pot() -> None:
    """Case B: 3-way all-in equal stacks, single pot split among ties."""
    amt = 5000
    committed = [amt, amt, amt, 0, 0, 0]
    status = [SEAT_ALLIN, SEAT_ALLIN, SEAT_ALLIN, SEAT_EMPTY, SEAT_EMPTY, SEAT_EMPTY]
    holes = np.array([[48, 44], [48, 44], [0, 4], [-1, -1], [-1, -1], [-1, -1]], dtype=np.int32)
    board = np.array([8, 12, 16, 20, 24], dtype=np.int32)
    kw = _manual_state(committed=committed, status=status, holes=holes, board=board)
    r = np.asarray(side_pot_returns(n_showdown=jnp.int32(3), **kw), dtype=np.int64)
    assert r.sum() == 0
    # phevaluator on board 8,12,16,20,24: all three play the board, scoring 6075008 each
    # (the [0,4] hole does not raise/lower the seat-2 hand here). Equal stacks -> one
    # 15000 pot split 3 ways = 5000 each -> every seat nets 0.
    assert r[0] == 0
    assert r[1] == 0
    assert r[2] == 0
    assert r[0] == r[1] == r[2]


def test_case_d_folded_contributor() -> None:
    """Case D: folded seat's chips stay in pot but cannot win."""
    committed = [6000, 6000, 100, 6000, 0, 0]
    status = [SEAT_ALLIN, SEAT_ALLIN, SEAT_FOLDED, SEAT_ALLIN, SEAT_EMPTY, SEAT_EMPTY]
    holes = np.array([[48, 44], [40, 36], [0, 4], [32, 28], [-1, -1], [-1, -1]], dtype=np.int32)
    board = np.array([8, 12, 16, 20, 24], dtype=np.int32)
    kw = _manual_state(committed=committed, status=status, holes=holes, board=board)
    r = np.asarray(side_pot_returns(n_showdown=jnp.int32(3), **kw), dtype=np.int64)
    assert r.sum() == 0
    assert r[2] == -100  # folded loses blind only


def test_case_g_combined_folded_and_multi_layer() -> None:
    """Case G: folded contributor in main + side1; three others all-in distinct."""
    # A folded after posting 4000 into main+side layers; B/C/D all-in 10000/6000/4000
    committed = [4000, 10000, 6000, 4000, 0, 0]
    status = [SEAT_FOLDED, SEAT_ALLIN, SEAT_ALLIN, SEAT_ALLIN, SEAT_EMPTY, SEAT_EMPTY]
    holes = np.array([[48, 44], [40, 36], [32, 28], [24, 20], [-1, -1], [-1, -1]], dtype=np.int32)
    board = np.array([8, 12, 16, 20, 24], dtype=np.int32)
    kw = _manual_state(committed=committed, status=status, holes=holes, board=board)
    r = np.asarray(side_pot_returns(n_showdown=jnp.int32(3), **kw), dtype=np.int64)
    assert r.sum() == 0
    # phevaluator on board 8,12,16,20,24: seat2=6075010 beats seat1==seat3=6075008
    # (seat0 folded, ineligible). Layers: main 16000 (seats 1,2,3 eligible) -> seat2 scoops;
    # side1 4000 (seats 1,2) -> seat2 scoops; side2 4000 uncalled -> returned to seat1.
    # seat0 = -4000 (folded contribution); seat1 = 4000 - 10000 = -6000;
    # seat2 = 16000 + 4000 - 6000 = +14000; seat3 = -4000.
    assert r[0] == -4000
    assert r[1] == -6000
    assert r[2] == 14000
    assert r[3] == -4000
