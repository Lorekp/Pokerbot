#!/usr/bin/env python3
"""Phase 2: K-Best league self-play with Trinal-Clip PPO."""

import argparse
from pathlib import Path

import torch

from pokerbot.encoding.weva_encoder import assert_production_weva_index
from pokerbot.models.policy import PseudoSiamesePolicy
from pokerbot.training.mixed_self_play import train_self_play_mixed
from pokerbot.training.self_play import SelfPlayConfig, train_self_play


def main() -> None:
    p = argparse.ArgumentParser(description="Self-play RL training")
    p.add_argument("--checkpoint", type=str, default=None, help="SFT checkpoint to warm-start")
    p.add_argument("--updates", type=int, default=50)
    p.add_argument("--hands", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument(
        "--league-opponents",
        default="",
        help="Comma-separated: cnn,nit,lag,donk,alpha_hu,human_like",
    )
    args = p.parse_args()

    assert_production_weva_index(Path("data/weva"), context="Phase 2 self-play (WEVA-encoded policy)")

    device = args.device
    model = None
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = PseudoSiamesePolicy().to(device)
        model.load_state_dict(ckpt["model"])

    cfg = SelfPlayConfig(
        hands_per_update=args.hands,
        total_updates=args.updates,
        device=device,
        checkpoint_dir=Path(args.checkpoint_dir),
    )

    if args.league_opponents.strip():
        opponents = [x.strip() for x in args.league_opponents.split(",") if x.strip()]
        train_self_play_mixed(model, cfg, league_opponents=opponents)
    else:
        train_self_play(model, cfg)


if __name__ == "__main__":
    main()