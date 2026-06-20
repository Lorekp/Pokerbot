"""Seat-count COEXISTENCE gate: the JAX/np NLHE engine must play tables of ANY width (2..7) in lockstep,
without a global seat-count flip, while keeping the 6-max default BYTE-IDENTICAL.

This is the acceptance gate for parameterizing ``jax_engine``/``np_engine``/``side_pots_jax`` for seat-
count coexistence. For each width ``n`` in {2..7} (full occupancy AND a partial layout that leaves
empties), we init an n-wide hand via ``init_hand`` (occupancy + explicit ``n_players``) and drive BOTH
engines in lockstep over many random hands, asserting:

  * every ``EnvState`` field is exactly equal (``np.array_equal``) at init and after every transition,
  * the legal-action mask is exactly equal,
  * the terminal rewards are exactly equal AND zero-sum.

n=7 is a width the pre-parameter engine could NOT represent (arrays were hard-sized to 6). We
specifically require a full 7-handed hand to play to a valid zero-sum terminal.

We also assert ``init_hand()`` with NO ``n_players`` is still exactly 6-wide -- the deployed cash
default is unchanged. (Byte-identity of the 6-wide numbers vs the pre-change engine is enforced
separately by ``tests/test_np_engine_equiv.py``.)
"""

from __future__ import annotations

import os

# Match the training betting-abstraction config BEFORE importing any encoder (mirrors the equiv gate).
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokerbot.engine import jax_engine as je
from pokerbot.engine import np_engine as ne

_FIELDS = je.EnvState._fields
_RNG = np.random.default_rng(20260620)


def _assert_states_equal(s_jax: je.EnvState, s_np: je.EnvState, ctx: str) -> None:
    for field in _FIELDS:
        a = np.asarray(getattr(s_jax, field))
        b = np.asarray(getattr(s_np, field))
        assert a.shape == b.shape, f"{ctx}: field {field} shape {a.shape} != {b.shape}"
        assert np.array_equal(a, b), f"{ctx}: field {field} mismatch\n  jax={a!r}\n  np ={b!r}"


def _assert_mask_equal(s_jax: je.EnvState, s_np: je.EnvState, ctx: str) -> np.ndarray:
    m_jax = np.asarray(je.legal_action_mask(s_jax))
    m_np = np.asarray(ne.legal_action_mask_np(s_np))
    assert np.array_equal(m_jax, m_np), f"{ctx}: legal mask mismatch\n  jax={m_jax}\n  np ={m_np}"
    return m_jax


def _occupancy_variants(n: int):
    """Full table, plus a partial layout that leaves at least 2 live seats (so a hand is playable)."""
    full = tuple(True for _ in range(n))
    yield full
    if n >= 3:
        # Drop one seat (the partial layout). For n==2 a partial would be <2 live -> no hand.
        partial = tuple(i != 1 for i in range(n))  # seat 1 empty
        yield partial
    if n >= 5:
        # A second, sparser layout: every other seat empty but >=2 live.
        sparse = tuple(i % 2 == 0 for i in range(n))
        yield sparse


def _stack_variants(n: int, seed: int):
    """Default 100bb plus skewed stacks that force unequal all-in side pots across n seats."""
    yield None
    base = 100 * je.CHIP_UNIT
    s1 = (_RNG.integers(5, 60, size=n) * je.CHIP_UNIT).astype(np.int32).tolist()
    yield s1


def _run_one_hand(n: int, occ: tuple[bool, ...], seed: int, stacks=None) -> dict:
    key = jax.random.PRNGKey(seed)
    occ_arr = jnp.asarray(occ, dtype=bool)
    if stacks is None:
        s_jax = je.init_hand(key, occupancy=occ_arr, n_players=n)
    else:
        s_jax = je.init_hand(
            key, stacks=jnp.asarray(stacks, dtype=jnp.int32), occupancy=occ_arr, n_players=n
        )
    # Width sanity: every dealt array is n-wide.
    assert s_jax.seat_status.shape[0] == n
    assert s_jax.stacks.shape[0] == n
    assert s_jax.committed_total.shape[0] == n
    assert s_jax.holes.shape[0] == n
    assert s_jax.rewards.shape[0] == n

    s_np = ne.to_numpy_state(s_jax)
    _assert_states_equal(s_jax, s_np, f"n={n} occ={occ} seed={seed} init")

    rng = np.random.default_rng(seed * 7919 + 13 + n)
    steps = 0
    while not bool(np.asarray(s_jax.done)):
        steps += 1
        assert steps < 500, f"n={n} seed={seed}: runaway loop"
        cn_jax = bool(np.asarray(je.is_chance_node(s_jax)))
        cn_np = ne.is_chance_node_np(s_np)
        assert cn_jax == cn_np, f"n={n} seed={seed}: is_chance_node {cn_jax}!={cn_np}"

        if cn_jax:
            s_jax = je.sample_chance(s_jax)
            s_np = ne.sample_chance_np(s_np)
            _assert_states_equal(s_jax, s_np, f"n={n} seed={seed} after sample_chance step{steps}")
            continue

        mask = _assert_mask_equal(s_jax, s_np, f"n={n} seed={seed} step{steps}")
        legal = np.nonzero(mask)[0]
        assert legal.size > 0, f"n={n} seed={seed}: no legal action"
        action = int(rng.choice(legal))

        s_jax = je.apply_action(s_jax, jnp.int32(action))
        s_np = ne.apply_action_np(s_np, action)
        _assert_states_equal(s_jax, s_np, f"n={n} seed={seed} after action={action} step{steps}")

    r_jax = np.asarray(s_jax.rewards)
    r_np = np.asarray(s_np.rewards)
    assert np.array_equal(r_jax, r_np), f"n={n} seed={seed}: rewards mismatch {r_jax} vs {r_np}"
    assert r_np.shape[0] == n, f"n={n}: rewards width {r_np.shape[0]} != {n}"
    assert int(np.sum(r_np)) == 0, f"n={n} seed={seed}: rewards not zero-sum: {r_np}"

    n_show = int(
        np.sum(
            (np.asarray(s_np.seat_status) == int(ne.SEAT_LIVE))
            | (np.asarray(s_np.seat_status) == int(ne.SEAT_ALLIN))
        )
    )
    return {"steps": steps, "showdown": n_show >= 2, "max_street": int(np.asarray(s_np.street))}


# seeds per (occupancy, stack) cell. n=2 is the sparsest: 1 occupancy x 2 stack variants = 2 hands/seed,
# so 160 seeds -> 320 hands; wider tables have more layouts and far exceed 300.
_HANDS_PER_LAYOUT = 160


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
def test_seat_count_lockstep(n: int):
    total = 0
    showdowns = 0
    for occ in _occupancy_variants(n):
        for seed in range(_HANDS_PER_LAYOUT):
            for stacks in _stack_variants(n, seed):
                info = _run_one_hand(n, occ, seed, stacks)
                total += 1
                showdowns += int(info["showdown"])
    assert total >= 300, f"n={n}: only {total} hands (need >=300)"
    assert showdowns > 0, f"n={n}: no showdowns exercised"
    print(f"\n[coexist] n={n}: {total} hands OK ({showdowns} showdowns)")


def test_default_init_is_exactly_six_wide():
    """init_hand() with no n_players must remain exactly 6-wide (deployed cash default unchanged)."""
    s = je.init_hand(jax.random.PRNGKey(0))
    assert s.seat_status.shape[0] == je.NUM_PLAYERS == 6
    for field in ("stacks", "committed_street", "committed_total", "acted", "rewards"):
        assert np.asarray(getattr(s, field)).shape[0] == 6, f"{field} not 6-wide by default"
    assert np.asarray(s.holes).shape == (6, 2)
    # reset_state default is also 6-wide.
    assert je.reset_state(jax.random.PRNGKey(1)).seat_status.shape[0] == 6


def test_full_seven_handed_plays_to_zero_sum_terminal():
    """A width the pre-parameter engine could not represent: a full 7-handed hand must reach a valid
    zero-sum terminal in both engines, in lockstep."""
    occ7 = tuple(True for _ in range(7))
    saw_terminal = 0
    for seed in range(50):
        info = _run_one_hand(7, occ7, seed)
        saw_terminal += 1
        assert info["steps"] > 0
    assert saw_terminal == 50


def test_seven_handed_occupancy_implies_width_without_n_players():
    """A 7-wide occupancy alone (no explicit n_players) must size the table to 7."""
    occ7 = jnp.asarray([True] * 7, dtype=bool)
    s = je.init_hand(jax.random.PRNGKey(3), occupancy=occ7)
    assert s.seat_status.shape[0] == 7
    # Inconsistent explicit n_players must assert.
    with pytest.raises(AssertionError):
        je.init_hand(jax.random.PRNGKey(3), occupancy=occ7, n_players=6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
