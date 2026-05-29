#!/usr/bin/env python3
"""Phase 3: MADT opponent transformer pretrain + SCAPT league rollouts (no hero PPO)."""

from __future__ import annotations

import argparse
from pathlib import Path

from pokerbot.training.phase3 import PretrainConfig, pretrain_opponent_transformer, resolve_hero_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3 MADT + SCAPT opponent training")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Hero policy for rollouts (default: selfplay → sft → sft_bc)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--rollout-hands", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--exploitative-limit", type=int, default=5000)
    p.add_argument("--pretrain-transformer", action="store_true", help="Deprecated alias; pretrain is default")
    p.add_argument("--skip-transformer-pretrain", action="store_true")
    p.add_argument("--no-scapt", action="store_true", help="Static bot league instead of SCAPT population")
    p.add_argument("--scapt-population", type=int, default=16)
    p.add_argument("--scapt-mutation-sigma", type=float, default=None, help="Override SCAPT σ (default 0.08)")
    p.add_argument(
        "--scapt-diversity-floor",
        type=float,
        default=None,
        help=f"Min mean pairwise action KL (default {0.02})",
    )
    p.add_argument(
        "--require-scapt-diversity",
        action="store_true",
        help="Hard-fail rollouts when SCAPT league diversity is below floor",
    )
    p.add_argument("--diversity-probe-states", type=int, default=16)
    args = p.parse_args()

    hero_ckpt = args.checkpoint or str(resolve_hero_checkpoint())
    print(f"Phase 3 hero checkpoint for opponent rollouts: {hero_ckpt}")

    if args.skip_transformer_pretrain:
        out = Path("checkpoints/opponent_transformer_madt.pt")
        if not out.exists():
            raise SystemExit(f"--skip-transformer-pretrain but missing {out}")
        print(f"Skipping transformer pretrain (using existing {out})")
        return

    pretrain_opponent_transformer(
        PretrainConfig(
            limit=args.exploitative_limit,
            device=args.device,
            rollout_hands=args.rollout_hands,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hero_checkpoint=hero_ckpt,
            use_scapt=not args.no_scapt,
            scapt_population_size=args.scapt_population,
            scapt_mutation_sigma=args.scapt_mutation_sigma,
            scapt_diversity_floor=args.scapt_diversity_floor,
            require_scapt_diversity=args.require_scapt_diversity,
            diversity_probe_states=args.diversity_probe_states,
        )
    )


if __name__ == "__main__":
    main()
