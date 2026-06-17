#!/usr/bin/env python3
"""HU NLHE SD-CFR training (WEVA-bucket belief-weighted)."""

from __future__ import annotations

import argparse
from pathlib import Path

from pokerbot.training.sd_cfr_nlhe import (
    SDCFRNLHEConfig,
    SDCFRNLHETrainer,
    load_nlhe_checkpoint,
    save_nlhe_checkpoint,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Train HU NLHE SD-CFR")
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--traversals", type=int, default=8)
    p.add_argument("--deal-sweeps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--checkpoint-dir", default="checkpoints/sd_cfr_hu")
    p.add_argument("--resume", default=None, help="Resume from checkpoint pkl")
    p.add_argument("--sft-checkpoint", default="checkpoints/policy_sft_prod6000.pt")
    p.add_argument("--no-sft-init", action="store_true", help="Train from scratch (not recommended)")
    p.add_argument("--output", default="checkpoints/sd_cfr_hu_latest.pkl")
    args = p.parse_args()

    cfg = SDCFRNLHEConfig(
        iterations=args.iterations,
        traversals_per_iter=args.traversals,
        deal_sweeps=args.deal_sweeps,
        seed=args.seed,
        lr=args.lr,
        sft_checkpoint=None if args.no_sft_init else args.sft_checkpoint,
    )
    trainer = SDCFRNLHETrainer(cfg)
    start = 0
    if args.resume:
        start = load_nlhe_checkpoint(trainer, Path(args.resume))
        print(f"Resumed from {args.resume} at iter {start}")

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_every > 0 else None
    trainer.train(
        log_every=1,
        checkpoint_dir=ckpt_dir,
        checkpoint_every=args.checkpoint_every,
        start_iter=start,
    )
    trainer.assert_profile_ev_in_range()
    save_nlhe_checkpoint(trainer, Path(args.output))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
