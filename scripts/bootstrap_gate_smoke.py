#!/usr/bin/env python3
"""Bootstrap WEVA-compatible smoke artifacts for gate harness CI (not gate baselines)."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from pokerbot.encoding.weva_encoder import WEVA_DEFAULT_K
from pokerbot.models.policy import PseudoSiamesePolicy
from pokerbot.opponent.opponent_transformer import OpponentTransformer, OpponentTransformerConfig


def _write_smoke_hero(path: Path, k: int) -> None:
    model = PseudoSiamesePolicy(num_weva_buckets=k)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "num_weva_buckets": k, "smoke": True}, path)


def _write_smoke_madt(path: Path) -> None:
    model = OpponentTransformer(OpponentTransformerConfig())
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "smoke": True}, path)


def _write_smoke_hvn(path: Path) -> None:
    """Minimal HVN pickle with average_strategy_params for validate_phase4."""
    import jax
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import HISTORY_FEATURE_DIM, NUM_ACTIONS
    from pokerbot.training.neural_cfr import AverageStrategyNet, HistoryValueNet

    hidden = 64
    key = jax.random.PRNGKey(0)
    hvn = HistoryValueNet(hidden)
    avg = AverageStrategyNet(hidden)
    dummy = jnp.zeros((1, HISTORY_FEATURE_DIM), dtype=jnp.float32)
    hvn_params = hvn.init(key, dummy)
    mask = jnp.ones((1, NUM_ACTIONS), dtype=jnp.bool_)
    avg_params = avg.init(key, dummy, mask)
    payload = {
        "history_value_params": hvn_params,
        "average_strategy_params": avg_params,
        "hidden_dim": hidden,
        "total_iterations": 1,
        "smoke": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def main() -> None:
    p = argparse.ArgumentParser(description="Bootstrap smoke gate harness artifacts")
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--weva-k", type=int, default=WEVA_DEFAULT_K)
    args = p.parse_args()

    out = Path(args.output_dir)
    hero = out / "smoke_hero_weva.pt"
    madt = out / "smoke_madt.pt"
    hvn = out / "smoke_hvn.pkl"

    _write_smoke_hero(hero, args.weva_k)
    _write_smoke_madt(madt)
    _write_smoke_hvn(hvn)

    print(f"Wrote smoke artifacts under {out}/")
    print(f"  {hero.name}  (WEVA k={args.weva_k})")
    print(f"  {madt.name}")
    print(f"  {hvn.name}")
    print("These are CI smoke artifacts — do not promote to gate baselines.")


if __name__ == "__main__":
    main()
