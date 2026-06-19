"""Leduc exploitability gate for the VR-DeepPDCFR+ neural trainer.

The pre-cloud convergence instrument: run the REAL neural traversal + VR estimator on Leduc Hold'em
(via the pluggable EnvSpec), then measure the trained average strategy's EXACT infoset-BR
exploitability against the tabular-CFR floor. This is the only gate that exercises the neural
traversal end-to-end, so it adjudicates the F4 (TD-bootstrap-vs-recursive) estimator question:
  * neural exploitability approaches the tabular floor  => the estimator converges => faithful, scale it;
  * it plateaus far above the floor                     => the estimator is broken => recursive rewrite.

``jax_from_py`` returns the genuine jax LeducState, so the eval encoding is identical to training's
(no train/serve skew), and the suit-isomorphic Leduc encoding is injective over ``py_info_key``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# Leduc max |return|: invested is bounded by ante + 2 raises/street over 2 streets ~ 9 chips. Scale
# leaf returns to ~[-1,1] like the reference /max_utility.
LEDUC_REWARD_SCALE: float = 14.0


def build_leduc_policy_fn(avg_module, avg_params, env):
    """Wrap the trained AverageStrategyNet as policy_fn(jax_leduc_state, seat) -> probs[3]."""

    @jax.jit
    def _probs(info, mask):
        return avg_module.apply(avg_params, info, mask)

    def policy_fn(jax_state, seat: int) -> np.ndarray:
        mask = env.legal_mask(jax_state)
        info = env.encode_info(jax_state, jnp.int32(seat))
        return np.asarray(_probs(info, mask), dtype=np.float64)

    return policy_fn


def run_leduc_gate(
    *,
    iters: int = 400,
    traversals: int = 256,
    batch_size: int = 256,
    hidden: int = 64,
    train_steps: int = 1,
    samples: int = 90,
    seed: int = 0,
    buffer_capacity: int = 100_000,
    reference_iters: int = 2000,
    log_every: int | None = None,
    verbose: bool = True,
) -> dict:
    """Train on Leduc and return exploitability (mbb) vs the tabular floor + uniform ceiling."""
    from pokerbot.training.cfr_env import leduc_env_spec
    from pokerbot.training.leduc_tabular_cfr import (
        equilibrium_reference_exploitability_mbb,
        infoset_exploitability_mbb_from_policy,
    )
    from pokerbot.training.neural_cfr import (
        AverageStrategyNet,
        VRDeepPDCFRConfig,
        VRDeepPDCFRPlusTrainer,
    )

    env = leduc_env_spec()
    cfg = VRDeepPDCFRConfig(
        total_iterations=iters, traversals_per_iter=traversals, batch_size=batch_size,
        hidden_dim=hidden, buffer_capacity=buffer_capacity, reward_scale=LEDUC_REWARD_SCALE,
        seed=seed, train_steps_per_iter=train_steps,
    )
    from pokerbot.training.train_logging import ProgressLogger

    trainer = VRDeepPDCFRPlusTrainer(cfg, env=env)
    # Drive the train loop directly so we can emit ETA progress (train() loops train_iteration the
    # same way from start_iter+1..total; start_iter=0 here, so this is behaviorally identical + logged).
    every = log_every if log_every is not None else (max(1, iters // 20) if verbose else 0)
    plog = ProgressLogger(iters, every=every, label="leduc-gate",
                          extra=lambda: f"strat_buf={int(trainer.strat_buffer.buffer.size)} "
                                        f"hist_buf={int(trainer.hist_buffer.buffer.size)}").begin()
    for t in range(1, iters + 1):
        trainer.train_iteration(t)
        plog.step(t)

    avg_module = AverageStrategyNet(hidden, env.num_actions)
    policy_fn = build_leduc_policy_fn(avg_module, trainer.avg_state.params, env)

    # The measurement phase below is exact (full infoset-BR + tabular CFR) and SINGLE-THREADED, so it
    # pegs ~1 core with no per-iter output. Signpost each step so the run isn't a black box here.
    if verbose:
        print(f"[leduc-gate] training done. computing exact infoset-BR exploitability over {samples} "
              f"deals (single-threaded)...", flush=True)
    neural_expl, u0, u1, br0, br1 = infoset_exploitability_mbb_from_policy(policy_fn, samples=samples)
    if verbose:
        print(f"[leduc-gate] neural exploitability = {neural_expl:.1f} mbb/g. computing tabular-CFR "
              f"floor ({reference_iters} iters, single-threaded)...", flush=True)
    floor = equilibrium_reference_exploitability_mbb(iterations=reference_iters)

    result = {
        "neural_exploitability_mbb": float(neural_expl),
        "tabular_floor_mbb": float(floor),
        "ratio_to_floor": float(neural_expl / floor) if floor > 0 else float("inf"),
        "u0": float(u0), "u1": float(u1), "br0": float(br0), "br1": float(br1),
        "iters": iters, "traversals": traversals, "train_steps": train_steps, "hidden": hidden,
    }
    if verbose:
        print(f"neural avg-strategy exploitability : {neural_expl:8.2f} mbb/g")
        print(f"tabular-CFR floor (ruler)          : {floor:8.2f} mbb/g")
        print(f"ratio neural/floor                 : {result['ratio_to_floor']:8.2f}x")
    return result
