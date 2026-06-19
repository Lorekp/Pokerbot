"""Correctness tests for the VR-DeepPDCFR+ fidelity fixes (audit vs rpSebastian/DeepPDCFR).

These lock in the four math/target fixes the reference comparison surfaced. The pre-existing suite only
checked shapes/finiteness, so all of these bugs passed CI; each test here fails on the old behavior.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from pokerbot.engine.jax_engine import NUM_ACTIONS
from pokerbot.training.neural_cfr import (
    history_value_loss,
    pdcfr_predicted_regrets,
    regret_matching,
)


def _legal(*idx):
    m = np.zeros(NUM_ACTIONS, dtype=bool)
    for i in idx:
        m[i] = True
    return jnp.asarray(m)


def test_regret_matching_argmax_fallback_not_uniform():
    """F3: all-negative regrets -> mass 1 on the HIGHEST-regret legal action (argmax), not uniform."""
    r = np.full(NUM_ACTIONS, -5.0, dtype=np.float32)
    r[2] = -1.0                                              # action 2 = least-negative legal
    p = np.array(regret_matching(jnp.asarray(r), _legal(0, 2, 4)))
    assert np.isclose(p[2], 1.0), p                          # argmax legal gets all the mass
    assert np.isclose(p.sum(), 1.0)
    assert p[1] == 0.0 and p[3] == 0.0                       # illegal stay zero
    assert not np.allclose(p[[0, 2, 4]], 1.0 / 3)           # explicitly NOT the old uniform fallback


def test_regret_matching_proportional_normal_case():
    """Positive regrets still produce the usual proportional strategy."""
    r = np.zeros(NUM_ACTIONS, dtype=np.float32)
    r[0], r[2] = 2.0, 1.0
    p = np.array(regret_matching(jnp.asarray(r), _legal(0, 1, 2)))
    assert np.allclose(p[[0, 1, 2]], [2 / 3, 0.0, 1 / 3])


def test_pdcfr_predicted_regrets_outer_clamp_nonnegative():
    """F1: the predicted-regret vector is clamped to >= 0 (outer max), matching the reference."""
    cum = jnp.asarray(np.full(NUM_ACTIONS, -10.0, dtype=np.float32))   # strongly negative cumulative
    inst = jnp.asarray(np.full(NUM_ACTIONS, -3.0, dtype=np.float32))   # negative instantaneous
    out = np.array(pdcfr_predicted_regrets(cum, inst, jnp.int32(5), 2.3))
    assert np.all(out >= 0.0), out                          # never negative after the outer clamp


def test_history_value_done_gate():
    """F2: target = utility + (1-done)*bootstrap. Terminal rows (done=1) drop the bootstrap; non-terminal
    rows keep it. Verified against a hand computation with a constant Q-net (q=1)."""
    B = 4
    util = np.array([0.5, -0.3, 1.0, -1.0], dtype=np.float32)
    done = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    action = np.array([0, 1, 0, 1], dtype=np.int32)
    batch = {
        "history": jnp.zeros((B, 8)), "next_history": jnp.zeros((B, 8)),
        "next_info_state": jnp.zeros((B, 8)), "next_legal_mask": jnp.ones((B, NUM_ACTIONS), bool),
        "action": jnp.asarray(action), "utility": jnp.asarray(util), "done": jnp.asarray(done),
    }
    apply_fn = lambda _p, _h: jnp.ones((B, NUM_ACTIONS))                          # noqa: E731  q==1
    avg_apply_fn = lambda _p, _i, _m: jnp.ones((B, NUM_ACTIONS)) / NUM_ACTIONS    # noqa: E731  uniform
    # bootstrap = sum(uniform * 1) = 1; taken = q[action] = 1.
    # err = utility + (1-done)*1 - 1 = (utility-1) if done else utility.
    expected = np.mean(np.where(done > 0, util - 1.0, util) ** 2)
    loss = float(history_value_loss(None, apply_fn, batch, avg_apply_fn, None))
    assert np.isclose(loss, expected, atol=1e-5), (loss, expected)
    # sanity: if we (wrongly) added the bootstrap at terminals too, loss would equal mean(util^2)+...
    assert not np.isclose(loss, float(np.mean(util ** 2)), atol=1e-5)
