"""Tests for the HU neural endgame integration (pokerbot/api/hu_neural_endgame.py + PokerBot.advise).

The load-bearing test is `test_remap_reproduces_native_hu_encoding`: it proves the live serve bridge
reconstructs the EXACT training seat layout ({3,5}, button rotating across them), so the absolute
seat/button one-hots in `encode_info_state` match what the net saw during training — i.e. NO serve-time
encoder skew (the class of bug that silently turns a trained net into garbage).
"""

from __future__ import annotations

import importlib.util
import pathlib
from dataclasses import dataclass

import numpy as np
import pytest

# Load pokerbot/api/hu_neural_endgame.py DIRECTLY (by file path), bypassing pokerbot.api.__init__.
# That package init eagerly imports poker_bot -> torch/pydantic, which the lean jax test venv lacks;
# the HU module itself is torch-free (numpy + lazy jax/engine), so loading it standalone is faithful
# and lets this suite run in any env that has jax (the fidelity tests) — no torch required.
_MOD_PATH = pathlib.Path(__file__).resolve().parents[1] / "pokerbot" / "api" / "hu_neural_endgame.py"
_spec = importlib.util.spec_from_file_location("hu_neural_endgame_under_test", _MOD_PATH)
_hne = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hne)

_HU_SEAT_PERM = _hne._HU_SEAT_PERM
HUNeuralEndgamePolicy = _hne.HUNeuralEndgamePolicy
_remap_env_to_hu_seats = _hne._remap_env_to_hu_seats
hu_endgame_enabled = _hne.hu_endgame_enabled
is_genuine_heads_up = _hne.is_genuine_heads_up


# --------------------------------------------------------------------------- lightweight view stubs
@dataclass
class _Seat:
    seat: int
    status: str = "active"


@dataclass
class _View:
    seats: list


# --------------------------------------------------------------------------- detection
def test_is_genuine_heads_up_two_seated():
    v = _View(seats=[_Seat(0, "active"), _Seat(1, "active")])
    assert is_genuine_heads_up(v) is True


def test_not_genuine_hu_when_three_seated_even_if_one_folded():
    # A 6-max hand that FOLDED to two players still has 3 dealt-in seats -> NOT genuine HU
    # (belongs to the 6-max blueprint, which holds the correct 6-max ranges).
    v = _View(seats=[_Seat(0, "active"), _Seat(1, "active"), _Seat(2, "folded")])
    assert is_genuine_heads_up(v) is False


def test_genuine_hu_ignores_empty_and_sitting_out():
    v = _View(seats=[
        _Seat(0, "active"), _Seat(1, "active"),
        _Seat(2, "empty"), _Seat(3, "sitting_out"), _Seat(4, "waiting"),
    ])
    assert is_genuine_heads_up(v) is True


def test_flag_off_by_default():
    # Default (no env) is the SAFE dormant state.
    import os
    prev = os.environ.pop("POKERBOT_HU_NEURAL_ENDGAME", None)
    try:
        assert hu_endgame_enabled() is False
    finally:
        if prev is not None:
            os.environ["POKERBOT_HU_NEURAL_ENDGAME"] = prev


# --------------------------------------------------------------------------- permutation invariants
def test_seat_perm_is_a_bijection_with_training_layout():
    perm = _HU_SEAT_PERM
    assert sorted(perm) == [0, 1, 2, 3, 4, 5], "must be a permutation of 0..5 (no seat datum dropped)"
    # live D-order {0,1} -> training live seats {3,5}
    assert {perm[0], perm[1]} == {3, 5}
    assert perm[0] == 3, "D-order 0 (button/SB) maps to seat 3 (the training button seat)"
    # the four empty GameState seats {2,3,4,5} -> the training dead seats {0,1,2,4}
    assert {perm[2], perm[3], perm[4], perm[5]} == {0, 1, 2, 4}


def test_failsafe_missing_checkpoint_returns_none(tmp_path):
    pol = HUNeuralEndgamePolicy(checkpoint=str(tmp_path / "does_not_exist.pkl"))
    # Loading must fail SAFE: disabled, no raise. policy() returns None regardless of input.
    assert pol.policy(game_state=None, hero_seat_d=0) is None
    assert pol._disabled is True


# --------------------------------------------------------------------------- the anti-skew fidelity test
def test_remap_reproduces_native_hu_encoding():
    """A native HU node (the training reset, button=3, live seats {3,5}) is ground truth. Down-map it to
    the {0,1} layout the live state-bridge emits, then run the PRODUCTION forward remap. The result must
    be byte-identical to the native node on every field AND produce a byte-identical encode_info_state.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import (
        NUM_PLAYERS,
        encode_info_state,
        init_hand,
    )

    occ = jnp.array([i in (3, 5) for i in range(NUM_PLAYERS)], dtype=bool)
    native = init_hand(jax.random.PRNGKey(0), button=jnp.int32(3), occupancy=occ)
    hero = int(native.action_seat)  # preflop HU: the button/SB (seat 3) acts first
    assert hero in (3, 5)

    # Inverse of _HU_SEAT_PERM: build the {0,1}-layout env the bridge would produce from this spot.
    perm = np.asarray(_HU_SEAT_PERM, dtype=np.int64)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.shape[0])

    native_np = jax.tree.map(lambda x: np.asarray(x), native)

    def scatter(arr, mapping):
        a = np.asarray(arr)
        out = np.empty_like(a)
        out[mapping] = a
        return out

    def relabel(x, mapping):
        xi = int(x)
        return int(mapping[xi]) if 0 <= xi < mapping.shape[0] else xi

    hist_seat_down = np.asarray(native_np.hist_seat).copy()
    vmask = (hist_seat_down >= 0) & (hist_seat_down < perm.shape[0])
    hist_seat_down[vmask] = inv[hist_seat_down[vmask]]

    down = native_np._replace(
        holes=scatter(native_np.holes, inv),
        seat_status=scatter(native_np.seat_status, inv),
        stacks=scatter(native_np.stacks, inv),
        committed_street=scatter(native_np.committed_street, inv),
        committed_total=scatter(native_np.committed_total, inv),
        acted=scatter(native_np.acted, inv),
        rewards=scatter(native_np.rewards, inv),
        button=np.asarray(relabel(native_np.button, inv), dtype=np.asarray(native_np.button).dtype),
        action_seat=np.asarray(relabel(native_np.action_seat, inv),
                               dtype=np.asarray(native_np.action_seat).dtype),
        last_aggressor=np.asarray(relabel(native_np.last_aggressor, inv),
                                  dtype=np.asarray(native_np.last_aggressor).dtype),
        hist_seat=hist_seat_down,
    )
    hero_down = int(inv[hero])
    assert {int(down.action_seat)} <= {0, 1}, "down-mapped HU live seats are at D-order {0,1}"

    # Production forward remap: {0,1} -> {3,5}, must round-trip back to the native node.
    back, hero_back = _remap_env_to_hu_seats(down, hero_down)
    assert hero_back == hero

    for f in native_np._fields:
        a = np.asarray(getattr(native_np, f))
        b = np.asarray(getattr(back, f))
        assert np.array_equal(a, b), f"field {f!r} differs after round-trip: {a} != {b}"

    enc_native = np.asarray(encode_info_state(native, jnp.int32(hero)))
    back_j = type(back)(**{f: jnp.asarray(getattr(back, f)) for f in back._fields})
    enc_back = np.asarray(encode_info_state(back_j, jnp.int32(hero_back)))
    assert np.array_equal(enc_native, enc_back), "encode_info_state skew after bridge remap"


def test_remap_lights_seat35_one_hots_from_d01():
    """Sanity at the encoding level: a {0,1}-live env remaps so the seat & button one-hots fall on the
    training seats {3,5} (NOT {0,1}). This is the exact bit that a naive (no-remap) serve would get wrong.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import NUM_PLAYERS, init_hand

    occ01 = jnp.array([i in (0, 1) for i in range(NUM_PLAYERS)], dtype=bool)
    env01 = init_hand(jax.random.PRNGKey(1), button=jnp.int32(0), occupancy=occ01)
    env01_np = jax.tree.map(lambda x: np.asarray(x), env01)
    hero_d = int(env01_np.action_seat)
    assert hero_d in (0, 1)

    remapped, hero_hu = _remap_env_to_hu_seats(env01_np, hero_d)
    assert int(remapped.button) == 3
    assert hero_hu in (3, 5)
    # live seats (non-empty status) are exactly {3,5}
    from pokerbot.engine.jax_engine import SEAT_EMPTY as JE_EMPTY
    live = [i for i in range(NUM_PLAYERS) if int(np.asarray(remapped.seat_status)[i]) != int(JE_EMPTY)]
    assert set(live) == {3, 5}
