"""Regression test for F-TRAIN-1 (six_max_bug_hunt_report.md).

``encode_history_state`` historically hard-divided ``state.hist_seat`` by the
module constant ``NUM_PLAYERS`` (==6), so a 7-max table whose top seat index
is 6 normalized to ``6 / 6 == 1.0`` (saturated). The fix adds an explicit
``n_players`` parameter that defaults to 6, so the deployed 6-max path is
BYTE-IDENTICAL to the pre-fix encoder while 7-max (and any future N-max)
callers can pass the real seat count.

FIREWALL: the 6-max normalization MUST NOT change. This test pins both:
  * 6-max seat 5 / 6 == 0.8333... (unchanged)
  * 7-max seat 6 / 7 == 0.8571... (newly correct)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokerbot.engine.jax_engine import (
    HISTORY_FEATURE_DIM,
    HISTORY_SLOTS,
    MAX_HISTORY,
    NUM_PLAYERS,
    SEAT_LIVE,
    STARTING_STACK_CHIPS,
    encode_history_state,
    init_hand,
)


def _put_hist(state, idx: int, seat: int, action: int = 0, amount: int = 0):
    """Stamp a single (seat, action, amount) into the state history at idx."""
    return state._replace(
        hist_seat=state.hist_seat.at[idx].set(jnp.int32(seat)),
        hist_action=state.hist_action.at[idx].set(jnp.int32(action)),
        hist_amount=state.hist_amount.at[idx].set(jnp.int32(amount)),
        hist_len=jnp.int32(max(int(state.hist_len), idx + 1)),
    )


def _seat_slice(feat: np.ndarray, slot: int) -> float:
    """Pull the first ``slot`` entry of the seat slice out of the history feature.

    The encoder emits ``jnp.concatenate([seats[:HISTORY_SLOTS], actions[:HISTORY_SLOTS],
    amounts[:HISTORY_SLOTS]])`` flat into the first ``3 * HISTORY_SLOTS`` of the pad.
    """
    return float(feat[slot])


def test_encode_history_6max_default_byte_identical() -> None:
    """Default call (n_players defaults to NUM_PLAYERS=6) is unchanged.

    Seat 5 normalized by 6 must equal 5/6 exactly -- this is the byte-identical
    guarantee that protects the deployed 6-max search base.
    """
    key = jax.random.PRNGKey(0)
    state = init_hand(key, n_players=NUM_PLAYERS)
    state = _put_hist(state, 0, seat=5, action=2, amount=100)

    feat = np.asarray(encode_history_state(state), dtype=np.float32)
    assert feat.shape == (HISTORY_FEATURE_DIM,)

    expected = 5.0 / 6.0
    got = _seat_slice(feat, 0)
    assert got == pytest.approx(expected, abs=1e-6), (
        f"6-max default must be byte-identical: seat 5/6 expected {expected}, got {got}"
    )


def test_encode_history_6max_explicit_n_players_byte_identical() -> None:
    """Explicitly passing n_players=6 reproduces the default (still byte-identical)."""
    key = jax.random.PRNGKey(0)
    state = init_hand(key, n_players=NUM_PLAYERS)
    state = _put_hist(state, 0, seat=5, action=2, amount=100)

    feat_default = np.asarray(encode_history_state(state), dtype=np.float32)
    feat_explicit = np.asarray(encode_history_state(state, NUM_PLAYERS), dtype=np.float32)
    assert np.array_equal(feat_default, feat_explicit), (
        "encode_history_state(state) and encode_history_state(state, 6) must match exactly"
    )


def test_encode_history_7max_no_longer_saturates() -> None:
    """7-max seat 6 must normalize to 6/7 ~= 0.857, NOT saturate to 1.0.

    This is the F-TRAIN-1 fix: pre-fix this returned 6/6 == 1.0, destroying
    VR baseline injectivity for any 7-max trainer/serve path.
    """
    key = jax.random.PRNGKey(0)
    state = init_hand(
        key,
        n_players=7,
        stacks=jnp.full(7, STARTING_STACK_CHIPS, dtype=jnp.int32),
    )
    assert state.seat_status.shape == (7,)
    state = _put_hist(state, 0, seat=6, action=2, amount=100)

    feat = np.asarray(encode_history_state(state, n_players=7), dtype=np.float32)
    expected = 6.0 / 7.0
    got = _seat_slice(feat, 0)
    assert got == pytest.approx(expected, abs=1e-6), (
        f"7-max seat 6 must normalize to 6/7 ~= {expected:.4f}, got {got:.4f}"
    )
    assert got < 1.0, (
        f"F-TRAIN-1 regression: 7-max seat 6 must NOT saturate to 1.0 (got {got})"
    )


def test_encode_history_7max_default_still_saturates_until_param_passed() -> None:
    """Explicit firewall check: WITHOUT the new param, the call still uses NUM_PLAYERS=6.

    This documents the upgrade contract: 7-max callers MUST pass ``n_players=7``
    explicitly; the default is locked to 6 so the deployed 6-max path is
    byte-identical (NUM_PLAYERS=6 firewall).
    """
    key = jax.random.PRNGKey(0)
    state = init_hand(
        key,
        n_players=7,
        stacks=jnp.full(7, STARTING_STACK_CHIPS, dtype=jnp.int32),
    )
    state = _put_hist(state, 0, seat=6, action=2, amount=100)

    feat = np.asarray(encode_history_state(state), dtype=np.float32)
    got = _seat_slice(feat, 0)
    assert got == pytest.approx(1.0, abs=1e-6), (
        f"default path (no n_players) MUST still divide by NUM_PLAYERS=6 -- "
        f"seat 6/6 == 1.0 confirms the 6-max firewall is intact; got {got}"
    )


def test_encode_history_7max_seat_zero_unchanged() -> None:
    """Seat 0 (any width) must normalize to 0.0 regardless of n_players."""
    key = jax.random.PRNGKey(0)
    state = init_hand(
        key,
        n_players=7,
        stacks=jnp.full(7, STARTING_STACK_CHIPS, dtype=jnp.int32),
    )
    state = _put_hist(state, 0, seat=0, action=0, amount=0)

    feat = np.asarray(encode_history_state(state, n_players=7), dtype=np.float32)
    got = _seat_slice(feat, 0)
    assert got == pytest.approx(0.0, abs=1e-6)


def test_encode_history_7max_full_seat_sweep() -> None:
    """For 7-max, seats 0..6 must each map to i/7 (i.e. injectivity preserved)."""
    key = jax.random.PRNGKey(0)
    state = init_hand(
        key,
        n_players=7,
        stacks=jnp.full(7, STARTING_STACK_CHIPS, dtype=jnp.int32),
    )

    normalized = []
    for s in range(7):
        s_state = _put_hist(state, 0, seat=s, action=0, amount=0)
        feat = np.asarray(encode_history_state(s_state, n_players=7), dtype=np.float32)
        normalized.append(_seat_slice(feat, 0))

    expected = [i / 7.0 for i in range(7)]
    for got, exp, s in zip(normalized, expected, range(7)):
        assert got == pytest.approx(exp, abs=1e-6), (
            f"7-max seat {s}: expected {exp:.4f}, got {got:.4f}"
        )
    # All distinct -> injectivity preserved (pre-fix would have seat 6 collide with seat 0 modulo 6)
    assert len(set(round(v, 6) for v in normalized)) == 7, (
        f"7-max seat normalization must be injective across seats 0..6; got {normalized}"
    )
