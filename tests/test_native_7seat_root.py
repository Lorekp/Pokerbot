"""Variable-seat (2..7) tournament root: constructable + plays to a valid terminal.

This is the critical-path unblocker for the up-to-7 tournament engine (Workstream C) and the 7-max LBR
anchor (D). It pins three things:

1. The JAX engine (the byte-identity SOURCE OF TRUTH the native core mirrors) constructs a REAL root at
   every seat count 2..7 via ``init_hand(..., n_players=N)`` / explicit ``occupancy``/``stacks`` width,
   posts blinds + per-player ante + button rotation correctly, and plays a full hand to a terminal with
   chip conservation (per-seat net deltas sum to 0). The 7-seat root is the headline (the 7-max anchor).

2. ``ante=0`` is byte-identical to the no-ante engine (regression gate), and antes are dead money
   (``pot_dead`` + ``committed_total``, NOT ``committed_street``).

3. The N=6 DEFAULT path is unchanged: ``init_hand()`` with no width argument is byte-identical to passing
   ``n_players=6`` / full-6 occupancy. (The native 6-seat byte-identity gate lives in
   ``tests/test_native_init_hand.py`` + ``tests/test_np_engine_equiv.py``; the native GENERIC-over-N
   ``init_hand_n`` / ``post_blinds_n`` path — the single source for any seat count — is exercised by the
   pure-Rust unit tests ``cargo test`` in ``tools/battery_native/src/tests.rs``: ``n7_root_*``,
   ``n7_ante_*``, ``n7_occupancy_*``, ``n2_heads_up_root_*``.)

The native flat marshaling / PyO3 binding is fixed-width 6 by design (NEVER flip the default), so the
PyO3 ``init_hand`` stays 6-seat; the >6 native path is the const-generic ``EnvStateN<N>`` exercised in
Rust. Here we validate the Python engine the native core is byte-identical to.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from pokerbot.engine import jax_engine as je  # noqa: E402
from pokerbot.engine.np_engine import (  # noqa: E402
    to_numpy_state,
    apply_action_np,
    legal_action_mask_np,
    sample_chance_np,
    is_chance_node_np,
    compute_returns_np,
)

BB = je.BB_CHIPS
SB = je.SB_CHIPS
START = je.STARTING_STACK_CHIPS


def _play_passive_to_terminal(env):
    """Drive a numpy EnvState to terminal taking the most passive legal action (check>call>fold),
    resolving chance nodes. Returns the terminal numpy EnvState."""
    for _ in range(400):
        if bool(env.done) or int(np.asarray(env.street)) >= je.STREET_TERMINAL:
            break
        if is_chance_node_np(env):
            env = sample_chance_np(env)
            continue
        seat = int(np.asarray(env.action_seat))
        if seat < 0:
            break
        mask = np.asarray(legal_action_mask_np(env, seat))
        if mask[je.CHECK] if hasattr(je, "CHECK") else mask[1]:
            action = 1  # CHECK
        elif mask[2]:
            action = 2  # CALL
        elif mask[0]:
            action = 0  # FOLD
        else:
            action = int(np.flatnonzero(mask)[0])
        env = apply_action_np(env, action)
    return env


def _net_deltas(env):
    """Per-seat net chip delta for a terminal state (rewards if set, else compute_returns)."""
    if bool(env.done):
        return np.asarray(env.rewards, dtype=np.int64)
    ss = np.asarray(env.seat_status)
    n_showdown = int(((ss == je.SEAT_LIVE) | (ss == je.SEAT_ALLIN)).sum())
    r = compute_returns_np(
        holes=np.asarray(env.holes),
        board=np.asarray(env.board),
        seat_status=ss,
        committed_total=np.asarray(env.committed_total),
        button=int(np.asarray(env.button)),
        n_showdown=n_showdown,
    )
    return np.asarray(r, dtype=np.int64)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
def test_variable_seat_root_constructs_and_conserves(n):
    """A full-ring N-seat root (2..7) constructs via n_players=N, has the right width, posts blinds,
    and plays a full passive hand to a terminal with chips conserved."""
    button = 0
    stacks = jnp.full(n, START, dtype=jnp.int32)
    occupancy = jnp.ones(n, dtype=bool)
    jstate = je.init_hand(
        jax.random.PRNGKey(1000 + n),
        button=jnp.int32(button),
        stacks=stacks,
        occupancy=occupancy,
        ante=jnp.int32(0),
        n_players=n,
    )
    env = to_numpy_state(jstate)

    # Width is exactly N.
    assert np.asarray(env.seat_status).shape[0] == n
    assert np.asarray(env.stacks).shape[0] == n
    assert np.asarray(env.holes).shape[0] == n

    # Blinds: SB = 1st live after button, BB = 2nd live after button.
    cs = np.asarray(env.committed_street)
    sb_seat = 1 % n  # button=0, all live -> SB = seat 1
    bb_seat = 2 % n
    assert cs[sb_seat] == SB, f"SB seat {sb_seat} posts {cs[sb_seat]} != {SB}"
    assert cs[bb_seat] == BB, f"BB seat {bb_seat} posts {cs[bb_seat]} != {BB}"
    assert int(np.asarray(env.current_bet)) == BB

    # First actor: UTG (3rd live after button) when n_live > 2, else button (HU/3-handed-to-2 edge).
    n_live = int((np.asarray(env.seat_status) == je.SEAT_LIVE).sum())
    assert n_live == n
    expected_actor = (1 % n) if n_live <= 2 else (3 % n)
    assert int(np.asarray(env.action_seat)) == expected_actor

    # ante=0 byte-identity invariant: committed_total == committed_street, pot_dead == 0.
    assert int(np.asarray(env.pot_dead)) == 0
    assert np.array_equal(np.asarray(env.committed_total), cs)

    # Every live seat dealt 2 holes; deck_ptr = 2*N.
    holes = np.asarray(env.holes)
    assert np.all(holes >= 0), "all seats dealt"
    assert int(np.asarray(env.deck_ptr)) == 2 * n

    # Plays a full hand to terminal with chip conservation.
    term = _play_passive_to_terminal(env)
    assert int(np.asarray(_net_deltas(term)).sum()) == 0, "chips must conserve"


def test_7seat_root_is_headline_full_hand():
    """The 7-max anchor: a 7-seat root reaches a full 5-card-board showdown with all 7 live, chips
    conserved. This is the seat count Workstream C / the 7-max LBR anchor depend on."""
    n = 7
    jstate = je.init_hand(
        jax.random.PRNGKey(77),
        button=jnp.int32(0),
        stacks=jnp.full(n, START, dtype=jnp.int32),
        occupancy=jnp.ones(n, dtype=bool),
        n_players=n,
    )
    env = to_numpy_state(jstate)
    assert np.asarray(env.seat_status).shape[0] == 7
    assert int(np.asarray(env.action_seat)) == 3  # UTG 7-handed

    term = _play_passive_to_terminal(env)
    # Passive check-down: river showdown street, full board, all 7 live.
    assert int(np.asarray(term.street)) == je.STREET_TERMINAL
    assert int((np.asarray(term.board) >= 0).sum()) == 5
    ss = np.asarray(term.seat_status)
    assert int(((ss == je.SEAT_LIVE) | (ss == je.SEAT_ALLIN)).sum()) == 7
    assert int(np.asarray(_net_deltas(term)).sum()) == 0


def test_7seat_ante_is_dead_money_and_conserves():
    """7-handed with a per-player ante: ante is dead money -> pot_dead + committed_total, NOT
    committed_street; chips still conserve through a full hand."""
    n = 7
    ante = 13
    jstate = je.init_hand(
        jax.random.PRNGKey(7),
        button=jnp.int32(0),
        stacks=jnp.full(n, START, dtype=jnp.int32),
        occupancy=jnp.ones(n, dtype=bool),
        ante=jnp.int32(ante),
        n_players=n,
    )
    env = to_numpy_state(jstate)
    cs = np.asarray(env.committed_street)
    ct = np.asarray(env.committed_total)
    # committed_street = blinds only (no ante).
    assert cs[1] == SB and cs[2] == BB
    assert cs[3] == 0
    # committed_total = committed_street + ante per live seat; pot_dead = sum of antes.
    assert np.array_equal(ct, cs + ante)
    assert int(np.asarray(env.pot_dead)) == ante * n
    term = _play_passive_to_terminal(env)
    assert int(np.asarray(_net_deltas(term)).sum()) == 0, "ante pot conserves chips"


def test_7seat_occupancy_empty_seats_skipped():
    """7-seat table with empties: SEAT_EMPTY seats get no holes and are skipped by the blind/UTG walk."""
    n = 7
    occ = np.array([True, False, True, False, True, False, True])  # 4 live: 0,2,4,6
    jstate = je.init_hand(
        jax.random.PRNGKey(42),
        button=jnp.int32(0),
        stacks=jnp.full(n, START, dtype=jnp.int32),
        occupancy=jnp.asarray(occ),
        n_players=n,
    )
    env = to_numpy_state(jstate)
    ss = np.asarray(env.seat_status)
    holes = np.asarray(env.holes)
    for s in (1, 3, 5):
        assert ss[s] == je.SEAT_EMPTY
        assert np.all(holes[s] < 0), f"empty seat {s} undealt"
    assert int((ss == je.SEAT_LIVE).sum()) == 4
    cs = np.asarray(env.committed_street)
    # button=0; live = {0,2,4,6}. SB = 1st live after = 2, BB = 2nd live after = 4, UTG = 3rd = 6.
    assert cs[2] == SB
    assert cs[4] == BB
    assert int(np.asarray(env.action_seat)) == 6
    assert int(np.asarray(env.deck_ptr)) == 2 * 4
    term = _play_passive_to_terminal(env)
    assert int(np.asarray(_net_deltas(term)).sum()) == 0


def test_2seat_heads_up_button_posts_bb():
    """HU root: jax posts SB = nth_live_after(button,1) = non-button, BB = nth_live_after(button,2) =
    button (wrap). Preflop the SB (= btn_first under the n_live<=2 branch) acts first."""
    n = 2
    jstate = je.init_hand(
        jax.random.PRNGKey(2),
        button=jnp.int32(0),
        stacks=jnp.full(n, START, dtype=jnp.int32),
        occupancy=jnp.ones(n, dtype=bool),
        n_players=n,
    )
    env = to_numpy_state(jstate)
    cs = np.asarray(env.committed_street)
    assert cs[1] == SB, "HU SB = non-button seat"
    assert cs[0] == BB, "HU BB = button seat"
    assert int(np.asarray(env.action_seat)) == 1, "HU preflop SB acts first"
    term = _play_passive_to_terminal(env)
    assert int(np.asarray(_net_deltas(term)).sum()) == 0


def test_n6_default_is_byte_identical_to_explicit():
    """The N=6 DEFAULT must be unchanged: init_hand() with no width == n_players=6 == full-6 occupancy,
    field for field. Guards against any accidental default flip from the generic refactor."""
    key = jax.random.PRNGKey(31337)
    button = jnp.int32(2)
    default = to_numpy_state(je.init_hand(key, button=button))
    explicit_n = to_numpy_state(je.init_hand(key, button=button, n_players=6))
    explicit_occ = to_numpy_state(
        je.init_hand(key, button=button, occupancy=jnp.ones(6, dtype=bool))
    )
    for f in default._fields:
        a = np.asarray(getattr(default, f))
        assert np.array_equal(a, np.asarray(getattr(explicit_n, f))), f"n_players=6 differs at {f}"
        assert np.array_equal(a, np.asarray(getattr(explicit_occ, f))), f"occupancy-6 differs at {f}"
