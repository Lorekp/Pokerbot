#!/usr/bin/env python3
"""Train VR-DeepPDCFR+ on the JAX-vectorized 6-max NLHE environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from pokerbot.encoding.weva_encoder import assert_production_weva_index
from pokerbot.training.neural_cfr import VRDeepPDCFRConfig, VRDeepPDCFRPlusTrainer, save_trainer_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="VR-DeepPDCFR+ training")
    assert_production_weva_index(Path("data/weva"), context="Phase 2 neural CFR / HistoryValueNet training")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--traversals", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--alpha", type=float, default=2.3)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/history_value_net.pkl",
        help="Output path for HistoryValueNet + blueprint params",
    )
    args = parser.parse_args()

    cfg = VRDeepPDCFRConfig(
        total_iterations=args.iterations,
        traversals_per_iter=args.traversals,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        buffer_capacity=args.buffer_capacity,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
    )
    trainer = VRDeepPDCFRPlusTrainer(cfg)
    trainer.train()
    out = save_trainer_checkpoint(trainer, args.checkpoint)
    print(f"Completed {args.iterations} VR-DeepPDCFR+ iterations.")
    print(f"Saved checkpoint: {out}")


if __name__ == "__main__":
    main()
