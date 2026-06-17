#!/usr/bin/env python3
"""HU NLHE SD-CFR smoke with optional Phase 1 SFT init."""

from __future__ import annotations

import argparse
from pathlib import Path

from pokerbot.training.sd_cfr_nlhe import (
    SDCFRNLHEConfig,
    SDCFRNLHETrainer,
    save_nlhe_checkpoint,
)


def main() -> None:
    p = argparse.ArgumentParser(description="HU NLHE SD-CFR smoke")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--traversals", type=int, default=4)
    p.add_argument("--sft-checkpoint", default="checkpoints/policy_sft_prod6000.pt")
    p.add_argument("--output", default="checkpoints/sd_cfr_hu_smoke.pkl")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = SDCFRNLHEConfig(
        iterations=args.iterations,
        traversals_per_iter=args.traversals,
        seed=args.seed,
        max_traverse_depth=24,
        train_batch_size=256,
        sft_checkpoint=args.sft_checkpoint,
    )
    trainer = SDCFRNLHETrainer(cfg)
    losses = trainer.train()
    trainer.assert_profile_ev_in_range(samples=4)
    save_nlhe_checkpoint(trainer, Path(args.output))
    print(f"iterations={args.iterations} final_loss={losses[-1]:.6f}" if losses else "done")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
