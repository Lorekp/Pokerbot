"""Bit-identity gate for the NATIVE Rust ``init_hand`` (with ante) in
``tools/battery_native/src/core.rs`` (exposed via ``lib.rs`` PyO3 ``init_hand``).

The native root construction is byte-identical to ``jax_engine.init_hand(rng_key, button, stacks,
occupancy, ante)`` WHEN GIVEN jax's own permuted deck + rng_key (the Threefry deck permutation is NOT
replicated natively; the caller supplies the already-permuted deck — see core.rs module note). So the
cross-check here: build the jax state, extract its ``deck`` + ``rng_key``, feed them to the native
``init_hand``, reconstruct the EnvState from the flat buffer, and assert EVERY field is byte-identical.

This covers the B2/B3 ``_post_blinds`` fixes: sub-BB short blinds (clamp, no negative stack), an
exhausted live seat -> SEAT_ALLIN, per-player ante as dead money in pot_dead + committed_total, and
``current_bet`` = the largest street commitment (not an unconditional full BB).
"""
from __future__ import annotations

import random

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from pokerbot.engine import jax_engine as je  # noqa: E402
from pokerbot.engine.np_engine import to_numpy_state  # noqa: E402
from pokerbot.search import native_solver as ns  # noqa: E402

pytestmark = pytest.mark.skipif(not ns.available(), reason="battery_native extension not built")

NP = je.NUM_PLAYERS


def _ext():
    """The loaded extension module (must expose the new ``init_hand`` pyfunction)."""
    ext = ns._try_load()
    assert ext is not None, "native extension failed to load"
    assert hasattr(ext, "init_hand"), "native extension is missing init_hand (rebuild needed)"
    return ext


def _native_init(ext, button, stacks, occupancy, ante, deck, rng_key):
    """Call the native init_hand and reconstruct a numpy EnvState via native_solver._from_flat."""
    out = ext.init_hand(
        int(button),
        [int(x) for x in stacks],
        [int(bool(x)) for x in occupancy],
        int(ante),
        [int(x) for x in deck],
        [int(x) & 0xFFFFFFFF for x in rng_key],
    )
    return ns._from_flat(out)


def _eq_state(a, b):
    """Return the name of the first field that differs (byte-wise via np.array_equal), or None."""
    assert a._fields == b._fields
    for f in a._fields:
        if not np.array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f))):
            return f
    return None


# Occupancy layouts: full-6, heads-up at seats 3 & 5, and a 4-handed table.
_OCC_FULL = [True] * NP
_OCC_HU = [i in (3, 5) for i in range(NP)]
_OCC_4MAX = [i in (0, 1, 3, 4) for i in range(NP)]
_OCCUPANCIES = {"full6": _OCC_FULL, "hu35": _OCC_HU, "4max": _OCC_4MAX}

_ANTES = [0, 13, 50]


def _sample_stacks(rng: random.Random) -> list[int]:
    """Varied stacks incl. sub-BB shorts that trigger the B2/B3 clamp + the all-in-from-blind path."""
    out = []
    for _ in range(NP):
        r = rng.random()
        if r < 0.18:
            # Sub-BB short (0..BB_CHIPS-1): a SB/BB poster here clamps + may go all-in.
            out.append(rng.randint(0, je.BB_CHIPS - 1))
        elif r < 0.30:
            out.append(0)  # exhausted / empty-ish stack
        elif r < 0.45:
            # 1..a few BB: BB-poster may be all-in after blind + ante.
            out.append(rng.randint(je.BB_CHIPS, 3 * je.BB_CHIPS))
        else:
            out.append(rng.randint(5 * je.BB_CHIPS, je.STARTING_STACK_CHIPS))
    return out


def test_native_init_hand_byte_identical():
    """~500 seeds x {full6, hu, 4max} x ante {0,13,50} x varied stacks: native == jax field-for-field."""
    ext = _ext()
    rng = random.Random(20260620)
    checked = 0
    saw_allin_from_blind = False
    saw_clamped_blind = False
    for seed in range(500):
        occ_name = rng.choice(list(_OCCUPANCIES))
        occ = _OCCUPANCIES[occ_name]
        ante = rng.choice(_ANTES)
        button = rng.randrange(NP)
        stacks = _sample_stacks(rng)

        occ_arr = jnp.asarray(occ)
        stacks_arr = jnp.asarray(stacks, dtype=jnp.int32)
        jax_state = je.init_hand(
            jax.random.PRNGKey(seed),
            button=jnp.int32(button),
            stacks=stacks_arr,
            occupancy=occ_arr,
            ante=jnp.int32(ante),
        )
        jx = to_numpy_state(jax_state)

        deck = np.asarray(jx.deck, dtype=np.int64)
        rng_key = np.asarray(jx.rng_key, dtype=np.uint32)
        nat = _native_init(ext, button, stacks, occ, ante, deck, rng_key)

        diff = _eq_state(nat, jx)
        assert diff is None, (
            f"field {diff!r} differs (seed={seed}, occ={occ_name}, ante={ante}, button={button}, "
            f"stacks={stacks})\n  native={getattr(nat, diff)!r}\n  jax   ={getattr(jx, diff)!r}"
        )
        checked += 1

        # Track that the B2/B3 edge cases were actually exercised (live seat all-in from a short blind,
        # and a blind clamped below its nominal size).
        ss = np.asarray(jx.seat_status)
        cs = np.asarray(jx.committed_street)
        if np.any(ss == je.SEAT_ALLIN):
            saw_allin_from_blind = True
        if np.any((cs > 0) & (cs < je.SB_CHIPS)) or (
            np.any(ss != je.SEAT_EMPTY) and np.any((cs > je.SB_CHIPS) & (cs < je.BB_CHIPS))
        ):
            saw_clamped_blind = True

    assert checked == 500
    assert saw_allin_from_blind, "no all-in-from-blind spot exercised — sampler did not hit B2/B3"
    assert saw_clamped_blind, "no clamped-blind spot exercised — sampler did not hit B2/B3"


def test_native_ante_zero_equals_no_ante_baseline():
    """ante=0 native root == the no-ante jax baseline (ante is non-perturbing at 0)."""
    ext = _ext()
    rng = random.Random(7)
    for seed in range(60):
        occ = _OCCUPANCIES[rng.choice(list(_OCCUPANCIES))]
        button = rng.randrange(NP)
        stacks = _sample_stacks(rng)

        # jax baseline WITHOUT passing ante at all (default 0).
        base = to_numpy_state(je.init_hand(
            jax.random.PRNGKey(seed),
            button=jnp.int32(button),
            stacks=jnp.asarray(stacks, dtype=jnp.int32),
            occupancy=jnp.asarray(occ),
        ))
        deck = np.asarray(base.deck, dtype=np.int64)
        rng_key = np.asarray(base.rng_key, dtype=np.uint32)

        nat0 = _native_init(ext, button, stacks, occ, 0, deck, rng_key)
        diff = _eq_state(nat0, base)
        assert diff is None, (
            f"ante=0 native field {diff!r} differs from no-ante baseline "
            f"(seed={seed}, button={button}, stacks={stacks})"
        )
        # And: native ante=0 must have empty dead-money bucket + committed_total == committed_street.
        assert int(np.asarray(nat0.pot_dead)) == 0
        assert np.array_equal(np.asarray(nat0.committed_total), np.asarray(nat0.committed_street))


def test_native_init_full6_no_ante_smoke():
    """A plain full-ring deep-stack root (the common case) is byte-identical — sanity anchor."""
    ext = _ext()
    button = 2
    stacks = [je.STARTING_STACK_CHIPS] * NP
    occ = _OCC_FULL
    base = to_numpy_state(je.init_hand(
        jax.random.PRNGKey(123),
        button=jnp.int32(button),
        stacks=jnp.asarray(stacks, dtype=jnp.int32),
        occupancy=jnp.asarray(occ),
    ))
    nat = _native_init(
        ext, button, stacks, occ, 0,
        np.asarray(base.deck, dtype=np.int64), np.asarray(base.rng_key, dtype=np.uint32),
    )
    assert _eq_state(nat, base) is None
    # action_seat = UTG = 3rd live after button (n_live > 2 path).
    assert int(np.asarray(nat.action_seat)) == int(np.asarray(base.action_seat))
