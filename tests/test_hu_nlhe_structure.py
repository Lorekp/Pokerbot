"""HU NLHE structure tests (Phase 2b)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine.jax_engine import CHIP_UNIT, init_hand, step_state
from pokerbot.engine.nlhe_bridge import game_state_from_jax
from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer


def test_hu_sb_first_preflop_from_bridge() -> None:
    """button=0, seats 0+1 live: SB (seat 1) acts first preflop in JAX HU."""
    occ = jnp.array([True, True, False, False, False, False])
    state = init_hand(jax.random.PRNGKey(0), button=jnp.int32(0), occupancy=occ)
    gs = game_state_from_jax(state)
    assert int(state.action_seat) == 1
    assert gs.street.name == "PREFLOP"


def test_profile_ev_zero_sum_hu() -> None:
    cfg = SDCFRNLHEConfig(
        iterations=1,
        traversals_per_iter=1,
        max_traverse_depth=12,
    )
    trainer = SDCFRNLHETrainer(cfg)
    trainer.assert_profile_ev_in_range(samples=2)
