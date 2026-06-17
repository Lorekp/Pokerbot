"""Verify SD-CFR NLHE advantage targets are bb-scaled after CHIP_UNIT normalization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pokerbot.engine.jax_engine import CHIP_UNIT, init_hand
from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer


def test_nlhe_traverse_divides_terminal_reward_by_chip_unit() -> None:
    trainer = SDCFRNLHETrainer(SDCFRNLHEConfig(seed=0, max_traverse_depth=8))
    key = jax.random.PRNGKey(0)
    chip_rewards = jnp.array([8000, -8000, 0, 0, 0, 0], dtype=jnp.int32)
    occ = jnp.array([True, True, False, False, False, False])
    state = init_hand(key, occupancy=occ)._replace(done=jnp.bool_(True), rewards=chip_rewards)
    util = trainer.traverse(state, 0, key, depth=0)
    assert abs(float(util) - 80.0) < 1e-4


def test_nlhe_advantage_targets_bb_scale_after_one_iteration() -> None:
    cfg = SDCFRNLHEConfig(
        iterations=1,
        traversals_per_iter=1,
        seed=1,
        max_traverse_depth=12,
    )
    trainer = SDCFRNLHETrainer(cfg)
    trainer.train_iteration()
    samples = getattr(trainer, "_iter_samples", [])
    if not samples:
        pytest.skip("no samples in shallow smoke traversal")
    max_abs = max(abs(float(x)) for _, inst, _ in samples for x in inst)
    assert max_abs < 600.0, f"targets exceed NLHE bound scale (max |target|={max_abs})"
