"""A1 stack-agnostic preflop: the 100bb-trained preflop blueprint is served only when the table is within
its depth band; off-band (deep/short) the solver falls through to the depth-general search instead of an
off-depth range. Standard 100bb tables stay on the blueprint path bit-for-bit."""

from __future__ import annotations

import jax
import numpy as np

from pokerbot.engine.jax_engine import CHIP_UNIT, NUM_PLAYERS, init_hand
from pokerbot.engine.np_engine import legal_action_mask_np, to_numpy_state
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver
from pokerbot.search.state_bridge import SubgameRoot

_OCC = np.array([True] * 6, dtype=bool)


class _StubPBS:
    def __init__(self, env, hero):
        self._h = np.asarray(env.holes)
        self.hero = hero

    def sample_opponent_holes(self, rng):
        return {s: (int(self._h[s, 0]), int(self._h[s, 1])) for s in range(NUM_PLAYERS) if s != self.hero}


def _preflop_env(stack_bb: float):
    import jax.numpy as jnp
    stacks = jnp.full(NUM_PLAYERS, int(stack_bb * CHIP_UNIT), dtype=jnp.int32)
    return to_numpy_state(init_hand(jax.random.PRNGKey(0), occupancy=jnp.asarray(_OCC), stacks=stacks))


def _solve(stack_bb: float, band):
    env = _preflop_env(stack_bb)
    hero = int(env.action_seat)
    root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=None,
                       legal_mask=legal_action_mask_np(env, hero).astype(bool))
    cfg = PluribusReBeLConfig(max_iterations=4, time_budget_ms=20000, preflop_blueprint_band_bb=band,
                              leaf=ContinuationLeafConfig(select="uniform", rollout_samples=1))
    res = PluribusReBeLSolver({}, cfg, rng=np.random.default_rng(0)).solve(root, _StubPBS(env, hero))
    return res.metadata.get("served")


def test_effective_stack_bb_reads_table_depth():
    assert abs(PluribusReBeLSolver._effective_stack_bb(_preflop_env(100.0)) - 100.0) < 1e-6
    assert abs(PluribusReBeLSolver._effective_stack_bb(_preflop_env(200.0)) - 200.0) < 1e-6
    assert abs(PluribusReBeLSolver._effective_stack_bb(_preflop_env(40.0)) - 40.0) < 1e-6


def test_in_band_serves_blueprint_offband_searches():
    band = (80.0, 130.0)
    assert _solve(100.0, band) == "blueprint_preflop"   # standard depth -> blueprint (unchanged)
    assert _solve(200.0, band) != "blueprint_preflop"   # deep -> falls through to search (A1)
    assert _solve(40.0, band) != "blueprint_preflop"    # short -> search


def test_band_none_is_legacy_always_blueprint():
    assert _solve(200.0, None) == "blueprint_preflop"   # disabled band -> legacy fixed-100bb behaviour
