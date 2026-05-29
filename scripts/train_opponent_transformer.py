#!/usr/bin/env python3
"""Train MADT opponent transformer on self-play trajectories."""

import argparse
from pathlib import Path

import torch

from pokerbot.models.policy import PseudoSiamesePolicy
from pokerbot.opponent.scapt_manager import SCAPTConfig, SCAPTManager
from pokerbot.training.mixed_self_play import rollout_opponent_trajectories
from pokerbot.training.opponent_transformer_train import (
    OpponentTransformerTrainConfig,
    train_opponent_transformer,
)
from pokerbot.training.phase3 import resolve_hero_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="Train MADT opponent transformer")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--parquet-glob", default="data/trajectories/*.parquet")
    p.add_argument("--rollout-hands", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--exploitative-limit", type=int, default=0)
    p.add_argument("--no-scapt", action="store_true")
    p.add_argument("--rollout-only", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else resolve_hero_checkpoint()
    hero_model = PseudoSiamesePolicy().to(device)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        hero_model.load_state_dict(ckpt["model"])

    scapt = None
    if not args.no_scapt:
        scapt = SCAPTManager()
        scapt.init_population(hero_model)

    if args.rollout_only:
        rollout_opponent_trajectories(
            hero_model,
            device,
            args.rollout_hands,
            Path("data/trajectories"),
            scapt=scapt,
        )
        return

    train_opponent_transformer(
        OpponentTransformerTrainConfig(
            parquet_glob=args.parquet_glob,
            rollout_hands=args.rollout_hands,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            exploitative_limit=args.exploitative_limit,
            use_scapt=not args.no_scapt,
            hero_checkpoint=str(ckpt_path),
        ),
        hero_model=hero_model,
        scapt=scapt,
    )


if __name__ == "__main__":
    main()
