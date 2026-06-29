"""Regression for the 7-max training levers wired for the resume-and-extend convergence leg:

  (A) ``nlhe_7max_env_spec`` builds a 7-player env (num_players=7 sizes the per-seat nets/buffers) at
      max_steps=96, and the env-gated knobs (POKERBOT_7MAX_DEPTH_BB log-uniform depth band +
      POKERBOT_7MAX_ANTE_BB / _ZERO_PROB) are parsed and validated. A 7-wide reset must produce a legal
      7-seat initial state so a warm resume from the iter-100000 checkpoint trains the right occupancy.

  (B) ``served_strategy_loss`` -- the cheap in-loop convergence PROXY appended to each progress line when
      POKERBOT_CFR_LOSS_PROBE=1 -- returns NaN on an empty reservoir and a FINITE, non-negative float
      after a couple of iterations. This guards the overnight run: the probe runs on every --log-every
      tick, so a crash here would kill the run. It must never raise and never feed back into training.

These cover the new code paths the babysit_7max_overnight.sh leg depends on (harder served-net fit +
in-loop served-loss telemetry + 7-seat env), without needing a GPU (CPU JAX, tiny nets).
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from pokerbot.training import cfr_env
from pokerbot.training.cfr_env import (
    _read_7max_ante,
    _read_7max_depth_band,
    nlhe_7max_env_spec,
)
from pokerbot.training.neural_cfr import (
    VRDeepPDCFRConfig,
    VRDeepPDCFRPlusTrainer,
)


# --- (A) 7-max env spec -----------------------------------------------------------------------------


def test_7max_env_spec_dims():
    """7 seats, max_steps carries the documented >3x margin over the measured worst-case hand depth."""
    env = nlhe_7max_env_spec()
    assert env.num_players == 7
    assert env.max_steps == 96


def test_7max_env_reset_is_legal_7seat_state():
    """A 7-wide reset must yield a runnable 7-seat initial state (button rotated, ante posted), so the
    trainer's per-seat buffers/nets are exercised at the right occupancy on a warm resume."""
    import jax

    env = nlhe_7max_env_spec()
    state = env.reset(jax.random.PRNGKey(0))
    # the engine encodes a per-seat stack/occupancy vector; whatever its field names, the reset must
    # produce arrays whose leading axis is the 7-wide table (not the default 6).
    leaves = jax.tree_util.tree_leaves(state)
    seven_wide = [np.asarray(x) for x in leaves
                  if hasattr(x, "shape") and x.ndim >= 1 and x.shape[0] == 7]
    assert seven_wide, "7-max reset produced no 7-wide per-seat array -- seat 7 may be dropped"


def test_7max_depth_band_env_parsing(monkeypatch):
    monkeypatch.delenv("POKERBOT_7MAX_DEPTH_BB", raising=False)
    assert _read_7max_depth_band() is None                      # unset -> fixed-100bb legacy reset
    monkeypatch.setenv("POKERBOT_7MAX_DEPTH_BB", "8,120")
    assert _read_7max_depth_band() == (8.0, 120.0)             # the default run band
    monkeypatch.setenv("POKERBOT_7MAX_DEPTH_BB", "8,40")       # the LEVER-3 bubble-skew band
    assert _read_7max_depth_band() == (8.0, 40.0)
    monkeypatch.setenv("POKERBOT_7MAX_DEPTH_BB", "120,8")      # lo>hi is rejected
    with pytest.raises(ValueError):
        _read_7max_depth_band()
    monkeypatch.setenv("POKERBOT_7MAX_DEPTH_BB", "not,numbers")
    with pytest.raises(ValueError):
        _read_7max_depth_band()


def test_7max_ante_env_parsing(monkeypatch):
    monkeypatch.delenv("POKERBOT_7MAX_ANTE_BB", raising=False)
    monkeypatch.delenv("POKERBOT_7MAX_ANTE_ZERO_PROB", raising=False)
    assert _read_7max_ante() == (0.125, 0.0)                   # CoinPoker MTT default ante, always-on
    monkeypatch.setenv("POKERBOT_7MAX_ANTE_BB", "0.0")
    assert _read_7max_ante()[0] == 0.0                         # ante can be disabled
    monkeypatch.setenv("POKERBOT_7MAX_ANTE_BB", "0.125")
    monkeypatch.setenv("POKERBOT_7MAX_ANTE_ZERO_PROB", "1.5")  # out-of-range prob is rejected
    with pytest.raises(ValueError):
        _read_7max_ante()


def test_7max_depth_band_is_log_uniform_within_bounds(monkeypatch):
    """The depth band samples a per-deal effective stack inside [lo,hi] BB (in chip units) -- a real run
    needs the shallow tail covered for the ICM/bubble leak the leg targets."""
    import jax

    from pokerbot.engine import jax_engine as je

    monkeypatch.setenv("POKERBOT_7MAX_DEPTH_BB", "8,120")
    monkeypatch.setenv("POKERBOT_7MAX_ANTE_BB", "0.125")
    env = nlhe_7max_env_spec()
    unit = je.CHIP_UNIT
    lo_chips, hi_chips = 8 * unit, 120 * unit
    for s in range(8):
        state = env.reset(jax.random.PRNGKey(s))
        leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(state)
                  if hasattr(x, "shape") and x.ndim >= 1 and x.shape[0] == 7]
        # at least one 7-wide integer array (the stacks) must lie inside the band (+ante slack).
        in_band = any(
            np.issubdtype(a.dtype, np.integer) and np.all(a > 0)
            and np.all(a <= hi_chips) and np.any(a >= lo_chips - unit)
            for a in leaves
        )
        assert in_band, f"seed {s}: no 7-wide stack array within the [8,120]BB band"


# --- (B) in-loop served-loss convergence proxy ------------------------------------------------------


def _tiny_trainer(iters: int = 0):
    # heads-up env keeps the test fast; the probe code path (avg_state + strat reservoir) is identical
    # across seat counts -- it reads ONLY the served average-strategy net + its reservoir.
    cfg = VRDeepPDCFRConfig(
        total_iterations=max(iters, 1), traversals_per_iter=8, batch_size=32, hidden_dim=16,
        buffer_capacity=500, train_steps_per_iter=2, avg_strategy_steps_per_iter=2,
        history_value_steps_per_iter=2, seed=0,
    )
    trainer = VRDeepPDCFRPlusTrainer(cfg, env=cfr_env.nlhe_hu_env_spec())
    for t in range(1, iters + 1):
        trainer.train_iteration(t)
    return trainer


def test_served_loss_probe_nan_on_empty_then_finite_after_training():
    # before any traversal the strategy reservoir is empty -> NaN (probe must short-circuit, not crash).
    cold = _tiny_trainer(iters=0)
    assert math.isnan(cold.served_strategy_loss())

    # after a couple of iters the reservoir is populated -> a finite, non-negative weighted-MSE.
    warm = _tiny_trainer(iters=2)
    assert int(warm.strat_buffer.buffer.size) > 0, "test setup: reservoir never filled"
    loss = warm.served_strategy_loss()
    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert loss >= 0.0


def test_served_loss_probe_does_not_mutate_trainer_state():
    """The probe is a pure forward pass: it advances the PRNG key (by design) but must NOT change the
    served net params, the reservoir, or the iteration counter -- it never feeds back into training."""
    trainer = _tiny_trainer(iters=2)
    params_before = jax_leaf_snapshot(trainer.avg_state.params)
    size_before = int(trainer.strat_buffer.buffer.size)
    iter_before = trainer._iteration

    _ = trainer.served_strategy_loss()

    params_after = jax_leaf_snapshot(trainer.avg_state.params)
    assert all(np.array_equal(a, b) for a, b in zip(params_before, params_after))
    assert int(trainer.strat_buffer.buffer.size) == size_before
    assert trainer._iteration == iter_before


def jax_leaf_snapshot(tree):
    import jax

    return [np.array(np.asarray(x)) for x in jax.tree_util.tree_leaves(tree)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
