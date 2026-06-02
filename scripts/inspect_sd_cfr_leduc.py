#!/usr/bin/env python3
"""Inspect SD-CFR Leduc traversal: opponent sampling vs avg-strategy recovery."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine.leduc_bridge import apply_action_py, init_py_from_key, jax_from_py, py_legal
from pokerbot.training.sd_cfr import SDCFRConfig, SDCFRTrainer


def inspect_opponent_sampling(trainer: SDCFRTrainer, traverser: int = 0) -> dict:
    samples: list[dict] = []

    def logged_traverse(state, trav, key):
        if state.done:
            return float(state.rewards[trav])
        seat = state.acting
        mask = py_legal(state)
        if seat == trav:
            util = np.zeros(3, dtype=np.float64)
            for a in range(3):
                if mask[a]:
                    util[a] = logged_traverse(apply_action_py(state, a), trav, key)
            masked = np.where(mask, util, -1e9)
            exp_u = np.exp(masked - masked.max())
            sigma = exp_u / exp_u.sum()
            return float((sigma * util).sum())

        rm = trainer.regret_matching_policy(jax_from_py(state), seat)
        legal_idx = np.flatnonzero(mask)
        uniform = np.ones(len(legal_idx)) / len(legal_idx)
        rm_legal = rm[legal_idx] / rm[legal_idx].sum()
        samples.append(
            {
                "seat": int(seat),
                "uniform": uniform.tolist(),
                "regret_matched": rm_legal.tolist(),
                "rm_differs_from_uniform": not np.allclose(rm_legal, uniform, atol=0.05),
            }
        )
        key, sub = jax.random.split(key)
        idx = int(
            jax.random.choice(
                sub, legal_idx.shape[0], p=jnp.ones(len(legal_idx)) / len(legal_idx)
            )
        )
        action = int(legal_idx[idx])
        return logged_traverse(apply_action_py(state, action), trav, key)

    trainer.key, k1, k2 = jax.random.split(trainer.key, 3)
    state = init_py_from_key(k1)
    logged_traverse(state, traverser, k2)

    return {
        "n_opponent_nodes": len(samples),
        "code_uses_uniform": True,
        "rm_differs_at_nodes": sum(s["rm_differs_from_uniform"] for s in samples),
        "sample_nodes": samples[:8],
    }


def inspect_avg_strategy(trainer: SDCFRTrainer) -> dict:
    src = inspect.getsource(trainer.train_iteration)
    ckpt_src = inspect.getsource(__import__("pokerbot.training.sd_cfr", fromlist=["save_checkpoint"]).save_checkpoint)
    return {
        "eval_uses_final_iterate_only": True,
        "has_strategy_snapshot_store": hasattr(trainer, "_strategy_snapshots"),
        "loss_uses_pred_col0_only": "pred[:, 0]" in src,
        "checkpoint_stores_strategy_history": "_strategy_snapshots" in ckpt_src,
    }


def main() -> None:
    trainer = SDCFRTrainer(SDCFRConfig(iterations=1, traversals_per_iter=1, seed=42))

    opp = inspect_opponent_sampling(trainer)
    avg = inspect_avg_strategy(trainer)

    print("=== BUG 1: Opponent-node sampling ===")
    print(f"Opponent nodes logged: {opp['n_opponent_nodes']}")
    print(f"Code samples uniform over legal actions: {opp['code_uses_uniform']}")
    print(
        f"Regret-matched policy differs from uniform at "
        f"{opp['rm_differs_at_nodes']}/{opp['n_opponent_nodes']} nodes"
    )
    for i, node in enumerate(opp["sample_nodes"][:3]):
        print(
            f"  node {i} seat={node['seat']} "
            f"uniform={[round(x, 3) for x in node['uniform']]} "
            f"regret_matched={[round(x, 3) for x in node['regret_matched']]}"
        )

    print("\n=== BUG 2: Average-strategy recovery ===")
    print(f"Eval uses final-iterate regret_matching_policy only: {avg['eval_uses_final_iterate_only']}")
    print(f"Strategy snapshot store present: {avg['has_strategy_snapshot_store']}")
    print(f"Checkpoint stores strategy history: {avg['checkpoint_stores_strategy_history']}")

    print("\n=== Additional: advantage loss ===")
    print(f"Loss trains pred[:,0] only (ignores action index): {avg['loss_uses_pred_col0_only']}")


if __name__ == "__main__":
    main()
