"""SETTLE_SHOWDOWN_LEAF gate (showdown-leaf-zero bug fix; memory: search-leaf-showdown-zero-bug).

Pins, on the pure-Python engine (mirrors tools/battery_native/src/tests.rs::settle_showdown_leaf):
  * the bug class exists exactly as diagnosed: a checked-down hand ends at street=STREET_TERMINAL
    with done=False and STALE rewards=[0]*N (the engine settles only inside apply_action_np);
  * gate OFF (default) is byte-identical: the terminal read returns the stale 0;
  * gate ON settles with the trainer's terminal_return semantics (n_show = LIVE|ALLIN ->
    compute_returns_np), zero-sum, and matches the native settle EXACTLY when the extension is
    available;
  * engine-settled terminals and depth-capped mid-hand states are untouched under both settings;
  * the consumer entry points (_playout / continuation_leaf_value already-terminal input) follow
    the gate.
The gate is process-global; every test restores it OFF.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from pokerbot.engine.jax_engine import CHIP_UNIT, init_hand  # noqa: E402
from pokerbot.engine import np_engine as npe  # noqa: E402
from pokerbot.engine.np_engine import (  # noqa: E402
    CALL,
    CHECK,
    SEAT_ALLIN,
    SEAT_LIVE,
    STREET_TERMINAL,
    apply_action_np,
    compute_returns_np,
    is_chance_node_np,
    is_unsettled_showdown_np,
    leaf_terminal_reward_np,
    legal_action_mask_np,
    sample_chance_np,
    set_settle_showdown_leaf,
    settle_showdown_leaf_enabled,
    settle_showdown_value_np,
    to_numpy_state,
)


@pytest.fixture(autouse=True)
def _gate_off_after():
    yield
    set_settle_showdown_leaf(False)


def _drive_checkdown(seed: int):
    occ = jnp.array([True] * 6)
    env = to_numpy_state(init_hand(jax.random.PRNGKey(seed), occupancy=occ))
    for _ in range(200):
        if bool(env.done):
            return env
        if is_chance_node_np(env):
            env = sample_chance_np(env)
            continue
        if int(env.action_seat) < 0:
            return env
        m = legal_action_mask_np(env, int(env.action_seat))
        a = CHECK if m[CHECK] else (CALL if m[CALL] else int(np.flatnonzero(m)[0]))
        env = apply_action_np(env, a)
    return env


def test_default_gate_is_off():
    assert settle_showdown_leaf_enabled() is False


def test_checked_down_showdown_is_unsettled_with_stale_zero_rewards():
    env = _drive_checkdown(7)
    assert not bool(env.done), "checked-down hand must NOT be engine-settled"
    assert int(env.street) >= int(STREET_TERMINAL)
    assert is_unsettled_showdown_np(env)
    assert all(int(r) == 0 for r in np.asarray(env.rewards))
    # gate OFF: the terminal read preserves the shipped stale-0 behavior
    for s in range(6):
        assert leaf_terminal_reward_np(env, s) == 0.0


def test_gate_on_settles_with_trainer_semantics_zero_sum():
    env = _drive_checkdown(7)
    st = np.asarray(env.seat_status)
    n_show = int(((st == SEAT_LIVE) | (st == SEAT_ALLIN)).sum())
    assert n_show >= 2
    rets = compute_returns_np(
        holes=np.asarray(env.holes), board=np.asarray(env.board), seat_status=st,
        committed_total=np.asarray(env.committed_total), button=int(env.button),
        n_showdown=n_show)
    assert int(np.sum(rets)) == 0
    assert any(int(r) != 0 for r in rets), "blinds are in the pot; all-zeros is the bug"
    set_settle_showdown_leaf(True)
    vals = [leaf_terminal_reward_np(env, s) for s in range(6)]
    assert vals == [float(rets[s]) for s in range(6)]
    assert vals == [settle_showdown_value_np(env, s) for s in range(6)]
    assert abs(sum(vals)) < 1e-9


def test_engine_settled_and_midhand_untouched_by_gate():
    # fold-around terminal: engine-settled; both gate settings read env.rewards verbatim
    occ = jnp.array([True] * 6)
    env = to_numpy_state(init_hand(jax.random.PRNGKey(3), occupancy=occ))
    for _ in range(6):
        if bool(env.done):
            break
        m = legal_action_mask_np(env, int(env.action_seat))
        env = apply_action_np(env, 0 if m[0] else CHECK)
    assert bool(env.done)
    want = [float(np.asarray(env.rewards)[s]) for s in range(6)]
    for on in (False, True):
        set_settle_showdown_leaf(on)
        assert [leaf_terminal_reward_np(env, s) for s in range(6)] == want
    # mid-hand root: stale-0 under both settings (never invent a value mid-betting-round)
    mid = to_numpy_state(init_hand(jax.random.PRNGKey(4), occupancy=occ))
    assert not is_unsettled_showdown_np(mid)
    for on in (False, True):
        set_settle_showdown_leaf(on)
        assert all(leaf_terminal_reward_np(mid, s) == 0.0 for s in range(6))


def test_playout_entry_points_follow_the_gate():
    """_playout / continuation_leaf_value on an ALREADY-unsettled terminal env: loop exits
    immediately, so the returned value IS the gate-aware terminal read (deterministic)."""
    from pokerbot.search.pluribus_rollout import _playout

    env = _drive_checkdown(11)
    assert is_unsettled_showdown_np(env)
    rng = np.random.default_rng(0)
    v_off = _playout(env, 0, adapter=None, choice_fn=lambda s: 0, rng=rng, max_depth=64)
    assert v_off == 0.0
    set_settle_showdown_leaf(True)
    v_on = _playout(env, 0, adapter=None, choice_fn=lambda s: 0, rng=rng, max_depth=64)
    assert v_on == settle_showdown_value_np(env, 0) / CHIP_UNIT
    assert v_on != 0.0 or settle_showdown_value_np(env, 0) == 0.0


def test_native_settle_parity_exact():
    from pokerbot.search import native_solver as ns

    if not ns.available():
        pytest.skip("battery_native unavailable")
    ext = ns._EXT
    if "settle_showdown_leaf" not in set(ext.native_caps()):
        pytest.skip("stale pyd without the settle_showdown_leaf cap")
    env = _drive_checkdown(21)
    assert is_unsettled_showdown_np(env)
    st = np.asarray(env.seat_status)
    n_show = int(((st == SEAT_LIVE) | (st == SEAT_ALLIN)).sum())
    nat = [float(x) for x in ext.compute_returns(
        [int(x) for x in np.asarray(env.holes).reshape(-1)],
        [int(x) for x in np.asarray(env.board)],
        [int(x) for x in st],
        [int(x) for x in np.asarray(env.committed_total)],
        int(env.button), n_show)]
    py = [settle_showdown_value_np(env, s) for s in range(6)]
    assert nat == py
    # unified setter keeps both gates in lockstep
    assert ns.set_settle_showdown_leaf(True) is True
    assert ext.settle_showdown_leaf_enabled() is True
    assert npe.settle_showdown_leaf_enabled() is True
    ns.set_settle_showdown_leaf(False)
    assert ext.settle_showdown_leaf_enabled() is False
    assert npe.settle_showdown_leaf_enabled() is False
