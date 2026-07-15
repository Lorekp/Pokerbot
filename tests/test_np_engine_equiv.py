"""Lockstep equivalence gate: pure-NumPy NLHE engine vs the JAX reference.

This is CORRECTNESS-CRITICAL.  ``pokerbot.engine.np_engine`` must produce BIT-IDENTICAL
results to ``pokerbot.engine.jax_engine`` for every transition, or a CFR blueprint built on
the fast CPU traversal is silently corrupted.

For thousands of random hands across two occupancy layouts and varied starting stacks we
drive BOTH engines in lockstep and assert every EnvState field, the legal-action mask, the
terminal rewards (and zero-sum), and the FAST encode fingerprint all match exactly.
"""

from __future__ import annotations

import os

# Match the training betting-abstraction config BEFORE importing the encoder.
os.environ["POKERBOT_BETTING_ABSTRACTION"] = "1"
os.environ["POKERBOT_POSTFLOP_BUCKETS"] = "1"
os.environ["POKERBOT_POSTFLOP_MODE"] = "fast"
os.environ["POKERBOT_POSTFLOP_NUM_BUCKETS"] = "200"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokerbot.engine import jax_engine as je
from pokerbot.engine import np_engine as ne
from pokerbot.engine.jax_weva_fast import encode_nlhe_bundle_fast

_FIELDS = je.EnvState._fields

# Occupancy layouts under test.
_OCC_HU = (False, False, False, True, False, True)  # heads-up at seats 3 / 5
_OCC_FULL = (True, True, True, True, True, True)

_RNG = np.random.default_rng(20240608)


def _assert_states_equal(s_jax: je.EnvState, s_np: je.EnvState, ctx: str) -> None:
    for field in _FIELDS:
        a = np.asarray(getattr(s_jax, field))
        b = np.asarray(getattr(s_np, field))
        assert a.shape == b.shape, f"{ctx}: field {field} shape {a.shape} != {b.shape}"
        assert np.array_equal(a, b), (
            f"{ctx}: field {field} mismatch\n  jax={a!r}\n  np ={b!r}"
        )


def _assert_mask_equal(s_jax: je.EnvState, s_np: je.EnvState, ctx: str) -> np.ndarray:
    m_jax = np.asarray(je.legal_action_mask(s_jax))
    m_np = np.asarray(ne.legal_action_mask_np(s_np))
    assert np.array_equal(m_jax, m_np), f"{ctx}: legal mask mismatch\n  jax={m_jax}\n  np ={m_np}"
    return m_jax


def _live_seat(s) -> int:
    ss = np.asarray(s.seat_status)
    for i in range(je.NUM_PLAYERS):
        if ss[i] in (int(ne.SEAT_LIVE), int(ne.SEAT_ALLIN)):
            return i
    return 0


def _run_one_hand(seed: int, occ: tuple[bool, ...], stacks=None) -> dict:
    key = jax.random.PRNGKey(seed)
    occ_arr = jnp.asarray(occ, dtype=bool)
    if stacks is None:
        s_jax = je.init_hand(key, occupancy=occ_arr)
    else:
        s_jax = je.init_hand(key, stacks=jnp.asarray(stacks, dtype=jnp.int32), occupancy=occ_arr)
    s_np = ne.to_numpy_state(s_jax)

    _assert_states_equal(s_jax, s_np, f"seed={seed} init")

    rng = np.random.default_rng(seed * 7919 + 13)
    steps = 0
    saw_showdown = False
    while not bool(np.asarray(s_jax.done)):
        steps += 1
        assert steps < 400, f"seed={seed}: runaway loop"
        # is_chance_node parity
        cn_jax = bool(np.asarray(je.is_chance_node(s_jax)))
        cn_np = ne.is_chance_node_np(s_np)
        assert cn_jax == cn_np, f"seed={seed}: is_chance_node {cn_jax}!={cn_np}"

        if cn_jax:
            s_jax = je.sample_chance(s_jax)
            s_np = ne.sample_chance_np(s_np)
            _assert_states_equal(s_jax, s_np, f"seed={seed} after sample_chance step{steps}")
            continue

        mask = _assert_mask_equal(s_jax, s_np, f"seed={seed} step{steps}")
        legal = np.nonzero(mask)[0]
        assert legal.size > 0, f"seed={seed}: no legal action"
        action = int(rng.choice(legal))

        s_jax = je.apply_action(s_jax, jnp.int32(action))
        s_np = ne.apply_action_np(s_np, action)
        _assert_states_equal(s_jax, s_np, f"seed={seed} after action={action} step{steps}")

    # Terminal checks.
    r_jax = np.asarray(s_jax.rewards)
    r_np = np.asarray(s_np.rewards)
    assert np.array_equal(r_jax, r_np), f"seed={seed}: rewards mismatch {r_jax} vs {r_np}"
    assert int(np.sum(r_np)) == 0, f"seed={seed}: rewards not zero-sum: {r_np} (sum={int(np.sum(r_np))})"

    # Fingerprint parity for a live seat.
    seat = _live_seat(s_np)
    fp_jax = encode_nlhe_bundle_fast(s_jax, seat).fingerprint()
    fp_np = encode_nlhe_bundle_fast(s_np, seat).fingerprint()
    assert fp_jax == fp_np, f"seed={seed}: fingerprint mismatch at seat {seat}"

    n_show = int(np.sum((np.asarray(s_np.seat_status) == int(ne.SEAT_LIVE)) | (np.asarray(s_np.seat_status) == int(ne.SEAT_ALLIN))))
    saw_showdown = n_show >= 2
    return {"steps": steps, "showdown": saw_showdown, "max_street": int(np.asarray(s_np.street))}


def test_short_allin_does_not_alias_overstack_raise_buckets():
    """A full re-raise that cannot fit must expose only the dedicated ALL_IN.

    Regression for the live 2026-07-15 deep KK river node: hero faced a raise
    from 241 to 792 with 751 behind.  Paying the 551 call plus any legal full
    raise exceeded the stack, but the old mask checked only the post-call
    component.  Four bucket actions were therefore admitted and execution
    clamped all four to the same short all-in, giving CFR five shove aliases.
    """
    occ = jnp.asarray([True, False, True, False, True, False], dtype=bool)
    s_jax = je.init_hand(jax.random.PRNGKey(91), occupancy=occ)
    status = jnp.asarray(
        [ne.SEAT_FOLDED, ne.SEAT_EMPTY, ne.SEAT_LIVE,
         ne.SEAT_EMPTY, ne.SEAT_LIVE, ne.SEAT_EMPTY],
        dtype=jnp.int8,
    )
    stacks = jnp.asarray([0, 0, 16100, 0, 37550, 0], dtype=jnp.int32)
    committed = jnp.asarray([0, 0, 39600, 0, 12050, 0], dtype=jnp.int32)
    s_jax = s_jax._replace(
        street=jnp.asarray(je.STREET_RIVER, dtype=jnp.int32),
        action_seat=jnp.asarray(4, dtype=jnp.int32),
        seat_status=status,
        stacks=stacks,
        committed_street=committed,
        committed_total=committed,
        pot_dead=jnp.asarray(6550, dtype=jnp.int32),
        current_bet=jnp.asarray(39600, dtype=jnp.int32),
        min_raise=jnp.asarray(27550, dtype=jnp.int32),
        acted=jnp.asarray([False, False, True, False, False, False]),
        last_raise_full=jnp.asarray(True),
        last_aggressor=jnp.asarray(2, dtype=jnp.int32),
        done=jnp.asarray(False),
    )
    s_np = ne.to_numpy_state(s_jax)
    mask = _assert_mask_equal(s_jax, s_np, "short-allin bucket alias")

    assert mask[ne.FOLD] and mask[ne.CALL] and mask[ne.ALL_IN]
    assert not mask[ne.BET_BUCKET_START:ne.ALL_IN].any(), mask.astype(int).tolist()

    # The one legal short all-in still returns action to the prior aggressor,
    # who owes the unmatched 200-chip increment.
    after = ne.apply_action_np(s_np, ne.ALL_IN)
    assert not ne.is_terminal_np(after)
    assert ne.current_player_np(after) == 2
    assert int(after.current_bet) == 49600
    assert int(after.committed_street[2]) == 39600


# Build the hand matrix: >=2000 hands per layout, with stack variation that forces
# unequal all-in side pots.
def _stack_variants(seed: int):
    """Yield None (default 100bb) plus a couple of skewed-stack layouts for side pots."""
    yield None
    base = 100 * je.CHIP_UNIT
    # Random small/uneven stacks so all-ins create unequal pots and side pots.
    s1 = (_RNG.integers(5, 60, size=6) * je.CHIP_UNIT).astype(np.int32).tolist()
    s2 = [3 * je.CHIP_UNIT, 7 * je.CHIP_UNIT, 12 * je.CHIP_UNIT,
          25 * je.CHIP_UNIT, 40 * je.CHIP_UNIT, base]
    yield s1
    yield s2


_HANDS_PER_LAYOUT = 700  # seeds; x3 stack variants x2 layouts => >= 4200 hands total.


@pytest.mark.parametrize("occ", [_OCC_HU, _OCC_FULL], ids=["HU_3_5", "all_six"])
def test_np_engine_lockstep_equivalence(occ):
    total = 0
    showdowns = 0
    rivers = 0
    for seed in range(_HANDS_PER_LAYOUT):
        for stacks in _stack_variants(seed):
            info = _run_one_hand(seed, occ, stacks)
            total += 1
            showdowns += int(info["showdown"])
            rivers += int(info["max_street"] >= je.STREET_RIVER)
    assert total >= 2000 // 2, f"layout {occ}: only {total} hands"
    # Sanity: the matrix actually exercises showdowns / deep streets.
    assert showdowns > 0, f"layout {occ}: no showdowns exercised"
    print(f"\n[equiv] layout occ={occ}: {total} hands OK "
          f"({showdowns} showdowns, {rivers} reached >=river)")


if __name__ == "__main__":
    grand = 0
    for occ, name in [(_OCC_HU, "HU_3_5"), (_OCC_FULL, "all_six")]:
        cnt = 0
        showdowns = 0
        for seed in range(_HANDS_PER_LAYOUT):
            for stacks in _stack_variants(seed):
                info = _run_one_hand(seed, occ, stacks)
                cnt += 1
                showdowns += int(info["showdown"])
        grand += cnt
        print(f"layout {name}: {cnt} hands OK ({showdowns} showdowns)")
    print(f"TOTAL: {grand} hands passed lockstep equivalence with zero mismatches.")
