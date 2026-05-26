#!/usr/bin/env python3
"""Phase 3: mixed-opponent self-play + opponent transformer pretrain."""

import argparse
from pathlib import Path

import torch

from pokerbot.agent import HybridAgent
from pokerbot.models.policy import PseudoSiamesePolicy
from pokerbot.training.mixed_self_play import train_self_play_mixed
from pokerbot.training.phase3 import PretrainConfig, pretrain_opponent_transformer
from pokerbot.training.self_play import SelfPlayConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3 hybrid training")
    p.add_argument("--checkpoint", default="checkpoints/policy_distilled.pt")
    p.add_argument("--updates", type=int, default=30)
    p.add_argument("--hands", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--league-opponents",
        default="cnn,human_like,nit,lag,donk,alpha_hu",
        help="Comma-separated opponent types",
    )
    p.add_argument("--pretrain-transformer", action="store_true")
    p.add_argument("--exploitative-limit", type=int, default=5000)
    args = p.parse_args()

    if args.pretrain_transformer:
        pretrain_opponent_transformer(
            PretrainConfig(limit=args.exploitative_limit, device=args.device)
        )

    model = PseudoSiamesePolicy()
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    opponents = [x.strip() for x in args.league_opponents.split(",") if x.strip()]
    train_self_play_mixed(
        model,
        SelfPlayConfig(
            hands_per_update=args.hands,
            total_updates=args.updates,
            device=args.device,
        ),
        league_opponents=opponents,
    )


if __name__ == "__main__":
    main()
