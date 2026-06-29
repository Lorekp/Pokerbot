"""Tests for the 7-max neural serve bridge (pokerbot/tournament/neural_serve.py + seat_remap.py).

The load-bearing test is ``test_seven_remap_reproduces_native_encoding``: it proves the serve bridge
reconstructs the EXACT 7-seat training layout (all 7 seats live in natural absolute order, button
rotating), so the ABSOLUTE seat/button one-hots in ``encode_info_state`` match what the net saw during
training — i.e. NO serve-time encoder skew (the class of bug that silently turns a trained net into
garbage). It mirrors ``tests/test_hu_neural_endgame.py::test_remap_reproduces_native_hu_encoding`` for 7
seats; the key difference is that a genuinely-7-live root needs the IDENTITY remap (no permutation),
unlike HU's {3,5} pin.

The modules are loaded DIRECTLY by file path (bypassing ``pokerbot.tournament.__init__``, which eagerly
imports the whole MTT layer -> torch/pydantic that the lean jax test venv lacks). Both modules under
test are torch-free (numpy + lazy jax/engine imports), so loading them standalone is faithful and lets
this suite run in any env that has jax.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(mod_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, _ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_seat_remap = _load("seat_remap_under_test", "pokerbot/tournament/seat_remap.py")
_neural_serve = _load("neural_serve_under_test", "pokerbot/tournament/neural_serve.py")

place_live_seats = _seat_remap.place_live_seats
_placement_perm = _seat_remap._placement_perm
SEVEN = _seat_remap.SEVEN
SevenMaxNeuralPolicy = _neural_serve.SevenMaxNeuralPolicy


# --------------------------------------------------------------------------- placement-perm invariants
def test_placement_perm_full_table_is_identity():
    perm = _placement_perm(list(range(7)), 7)
    assert perm.tolist() == list(range(7)), "a full 7-live table must map identity (no permutation)"


def test_placement_perm_is_a_bijection():
    # 3 live seats placed onto trained absolute seats {3,5,1}; the remaining empties fill the rest.
    perm = _placement_perm([3, 5, 1], 7)
    assert sorted(perm.tolist()) == list(range(7)), "must be a bijection of 0..6 (no seat datum dropped)"
    # the leading 3 input positions land on exactly the requested trained seats, in order.
    assert perm[0] == 3 and perm[1] == 5 and perm[2] == 1


def test_placement_perm_rejects_bad_input():
    with pytest.raises(ValueError):
        _placement_perm([3, 3], 7)          # duplicate trained seat
    with pytest.raises(ValueError):
        _placement_perm([7], 7)             # out of range
    with pytest.raises(ValueError):
        _placement_perm([0, 1, 2, 3, 4, 5, 6, 6], 7)  # longer than table width (and dup)


# --------------------------------------------------------------------------- fail-safe (no checkpoint)
def test_failsafe_missing_checkpoint_returns_none(tmp_path):
    pol = SevenMaxNeuralPolicy(checkpoint=str(tmp_path / "does_not_exist.pkl"))
    # Loading must fail SAFE: disabled, no raise. policy() returns None regardless of input.
    assert pol.policy(env_np=None, seat=0) is None
    assert pol._disabled is True


# --------------------------------------------------------------------------- the anti-skew fidelity test
def test_seven_remap_reproduces_native_encoding():
    """A native 7-seat node (the training reset: all 7 live, natural order, a rotated button) is ground
    truth. Pass it through the PRODUCTION placement (identity, list(range(7))). The result must be
    byte-identical on EVERY EnvState field AND produce a byte-identical encode_info_state — proving the
    served absolute seat/button one-hots match the trained layout."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import (
        CHIP_UNIT,
        NUM_PLAYERS,  # noqa: F401 — module default is 6; the 7-seat width is set explicitly below
        encode_info_state,
        init_hand,
    )

    N7 = SEVEN
    occ = jnp.array([True] * N7, dtype=bool)
    button = 4  # a rotated button (not seat 0) to catch any button-onehot skew
    ante = jnp.int32(round(0.125 * CHIP_UNIT))  # the CoinPoker MTT per-player ante
    native = init_hand(jax.random.PRNGKey(7), button=jnp.int32(button), occupancy=occ, ante=ante)
    assert native.seat_status.shape[0] == N7
    hero = int(native.action_seat)
    assert 0 <= hero < N7

    native_np = jax.tree.map(lambda x: np.asarray(x), native)

    # Production placement: full table -> identity remap, must round-trip back to the native node.
    back, mapping = place_live_seats(native_np, list(range(N7)))
    assert mapping == {i: i for i in range(N7)}, "full table placement must be the identity map"
    hero_back = mapping[hero]
    assert hero_back == hero

    for f in native_np._fields:
        a = np.asarray(getattr(native_np, f))
        b = np.asarray(getattr(back, f))
        assert np.array_equal(a, b), f"field {f!r} differs after round-trip: {a} != {b}"

    enc_native = np.asarray(encode_info_state(native, jnp.int32(hero)))
    back_j = type(back)(**{f: jnp.asarray(getattr(back, f)) for f in back._fields})
    enc_back = np.asarray(encode_info_state(back_j, jnp.int32(hero_back)))
    assert np.array_equal(enc_native, enc_back), "encode_info_state skew after bridge placement"


def test_subtable_lands_one_hots_on_trained_absolute_seats():
    """The <7-live sub-table case: an env whose k live players sit at the leading D-order positions
    {0..k-1} is placed onto chosen trained absolute seats. The live (non-empty) seats AND the button
    one-hot must fall on the trained indices we requested — the exact bit a naive (no-remap) serve of a
    short table would get wrong."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import SEAT_EMPTY as JE_EMPTY
    from pokerbot.engine.jax_engine import init_hand

    N7 = SEVEN
    # Build a genuine 3-handed table at the LEADING seats {0,1,2} (the layout a live sub-table bridge
    # emits), then place those onto trained absolute seats {3,5,6}.
    occ3 = jnp.array([i in (0, 1, 2) for i in range(N7)], dtype=bool)
    env3 = init_hand(jax.random.PRNGKey(11), button=jnp.int32(0), occupancy=occ3)
    env3_np = jax.tree.map(lambda x: np.asarray(x), env3)
    hero_old = int(env3_np.action_seat)
    assert hero_old in (0, 1, 2)

    target = [3, 5, 6]  # trained absolute seats to place the 3 live players on
    remapped, mapping = place_live_seats(env3_np, target)

    # Live (non-empty) seats are exactly the requested trained set.
    live = [i for i in range(N7) if int(np.asarray(remapped.seat_status)[i]) != int(JE_EMPTY)]
    assert set(live) == set(target)
    # Button relabelled through the same map (old button was D-order 0 -> target[0] == 3).
    assert int(remapped.button) == mapping[int(env3_np.button)]
    assert int(remapped.button) in target
    # Hero seat translates through the mapping and lands on a trained live seat.
    assert mapping[hero_old] in target
    # No live seat datum lost: the placed stacks equal the originals gathered through the inverse map.
    for old in range(N7):
        new = mapping[old]
        assert int(np.asarray(remapped.stacks)[new]) == int(np.asarray(env3_np.stacks)[old])


def test_subtable_placement_preserves_encoder_fidelity():
    """A natively-built 3-handed node at trained seats {3,5,6} is ground truth. Down-map it to the
    leading {0,1,2} layout the live bridge emits, then run the PRODUCTION placement back onto {3,5,6}.
    The encode_info_state must be byte-identical to the native node — i.e. placement is encoder-faithful
    for the sub-table case too, not just the full table."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import encode_info_state, init_hand

    N7 = SEVEN
    target = [3, 5, 6]
    occ = jnp.array([i in target for i in range(N7)], dtype=bool)
    native = init_hand(jax.random.PRNGKey(23), button=jnp.int32(3), occupancy=occ)
    hero = int(native.action_seat)
    assert hero in target

    native_np = jax.tree.map(lambda x: np.asarray(x), native)

    # Down-map native {3,5,6} -> leading {0,1,2} via the inverse of the placement perm.
    perm = _placement_perm(target, N7)          # perm[old_leading] = trained_abs
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.shape[0])         # inv[trained_abs] = old_leading

    def scatter(arr, mp):
        a = np.asarray(arr)
        out = np.empty_like(a)
        out[mp] = a
        return out

    def relabel(x, mp):
        xi = int(x)
        return int(mp[xi]) if 0 <= xi < mp.shape[0] else xi

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
    live_down = [i for i in range(N7) if int(np.asarray(down.seat_status)[i]) != 0]
    assert set(live_down) == {0, 1, 2}, "down-mapped live seats are at leading D-order {0,1,2}"

    # Production placement: leading {0,1,2} -> {3,5,6}, must round-trip back to the native node.
    back, mapping = place_live_seats(down, target)
    assert mapping[hero_down] == hero

    for f in native_np._fields:
        a = np.asarray(getattr(native_np, f))
        b = np.asarray(getattr(back, f))
        assert np.array_equal(a, b), f"field {f!r} differs after sub-table round-trip"

    enc_native = np.asarray(encode_info_state(native, jnp.int32(hero)))
    back_j = type(back)(**{f: jnp.asarray(getattr(back, f)) for f in back._fields})
    enc_back = np.asarray(encode_info_state(back_j, jnp.int32(mapping[hero_down])))
    assert np.array_equal(enc_native, enc_back), "encode_info_state skew after sub-table placement"


# --------------------------------------------------------------------------- live serve smoke (net)
def test_policy_serves_valid_distribution_when_checkpoint_present():
    """If seven_v1.pkl is present (and POKERBOT_INFO_SUITS matches), policy() returns a valid probability
    distribution over the legal mask for a natively-built 7-seat root. Skipped when the checkpoint is
    absent or the info-suits flag mismatches (the load fails SAFE -> None, which we don't assert here)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    ckpt = _ROOT / "checkpoints" / "sd_cfr_7max_neural" / "seven_v1.pkl"
    if not ckpt.exists():
        pytest.skip("seven_v1.pkl checkpoint not present")

    import pokerbot.engine.jax_engine as je
    from pokerbot.engine.actions import NUM_ACTIONS
    from pokerbot.engine.jax_engine import CHIP_UNIT, init_hand, legal_action_mask

    pol = SevenMaxNeuralPolicy(checkpoint=str(ckpt))
    if not pol._ensure_loaded():
        # Almost always an info-suits skew vs this process; the checkpoint trains with suits ON.
        pytest.skip(f"7-max net failed to load (likely POKERBOT_INFO_SUITS != {je._INFO_SUITS}); "
                    f"fail-safe path exercised by the missing-checkpoint test")

    occ = jnp.array([True] * SEVEN, dtype=bool)
    native = init_hand(jax.random.PRNGKey(3), button=jnp.int32(2), occupancy=occ,
                       ante=jnp.int32(round(0.125 * CHIP_UNIT)))
    native_np = jax.tree.map(lambda x: np.asarray(x), native)
    seat = int(native.action_seat)

    probs = pol.policy(native_np, seat)
    assert probs is not None, "valid root must serve a distribution"
    assert probs.shape[0] == NUM_ACTIONS
    assert np.isfinite(probs).all()
    assert abs(float(probs.sum()) - 1.0) < 1e-6, "served probs must be normalized"
    mask = np.asarray(legal_action_mask(native, jnp.int32(seat)), dtype=bool)
    assert (probs[~mask] == 0.0).all(), "no mass on illegal actions"
