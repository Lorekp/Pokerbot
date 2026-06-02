"""Verify SD-CFR NLHE advantage targets are bb-scaled after CHIP_UNIT normalization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pokerbot.engine.jax_engine import CHIP_UNIT
from pokerbot.training.sd_cfr import SDCFRNLHEConfig, SDCFRNLHETrainer


def test_nlhe_traverse_divides_terminal_reward_by_chip_unit() -> None:
    """Terminal returns in chips must become bb-scale utility in traverse."""
    trainer = SDCFRNLHETrainer(SDCFRNLHEConfig(seed=0))
    key = jax.random.PRNGKey(0)
    chip_rewards = jnp.array([8000, -8000, 0, 0, 0, 0], dtype=jnp.int32)
    state = trainer._init_state(key)._replace(done=jnp.bool_(True), rewards=chip_rewards)
    util = trainer.traverse(state, 0, key, depth=0)
    assert abs(float(util) - 80.0) < 1e-4
    assert abs(float(util) - float(chip_rewards[0]) / CHIP_UNIT) < 1e-4


def _shallow_traverse_wrapper(original):
    """Cap NLHE recursion depth so one iteration finishes quickly in CI."""

    def wrapped(state, traverser, key, depth=0):
        if depth > 8:
            if bool(state.done):
                return state.rewards[traverser].astype(jnp.float32) / float(CHIP_UNIT)
            return jnp.float32(0.0)
        return original(state, traverser, key, depth)

    return wrapped


def test_nlhe_advantage_targets_bb_scale_after_one_iteration() -> None:
    """One shallow iteration: stored targets stay in bb scale (not raw chips)."""
    cfg = SDCFRNLHEConfig(iterations=1, traversals_per_iter=1, seed=1)
    trainer = SDCFRNLHETrainer(cfg)
    trainer.traverse = _shallow_traverse_wrapper(trainer.traverse)  # type: ignore[method-assign]
    trainer.train_iteration()
    if not trainer.adv_buffer:
        pytest.skip("no samples in shallow smoke traversal")
    targets = jnp.stack([b[1] for b in trainer.adv_buffer])
    max_abs = float(jnp.max(jnp.abs(targets)))
    assert max_abs < 200.0, f"targets look like raw chips (max |target|={max_abs})"
