#!/usr/bin/env python3
"""Cache LLM labels and/or distill CNN from YiPz PokerBench teacher."""

import argparse
from pathlib import Path

from pokerbot.training.distill import DistillConfig, train_distill
from pokerbot.training.distill_cache import CacheConfig, build_label_cache


def main() -> None:
    p = argparse.ArgumentParser(description="LLM distillation pipeline")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--preflop-limit", type=int, default=0)
    p.add_argument("--postflop-limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--disagreement-only", action="store_true")
    p.add_argument("--checkpoint-in", default="checkpoints/policy_sft.pt")
    p.add_argument("--cache-path", default="data/cache/llm_labels.parquet")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    pre_limit = None if args.preflop_limit <= 0 else args.preflop_limit
    post_limit = None if args.postflop_limit <= 0 else args.postflop_limit

    cache_path = Path(args.cache_path)

    if args.cache_only or not cache_path.exists():
        cache_path = build_label_cache(
            CacheConfig(
                preflop_limit=pre_limit,
                postflop_limit=post_limit,
                batch_size=args.batch_size,
                cache_name=cache_path.name,
                output_dir=cache_path.parent,
            )
        )

    if args.train or (not args.cache_only and cache_path.exists()):
        train_distill(
            DistillConfig(
                cache_path=cache_path,
                checkpoint_in=Path(args.checkpoint_in) if args.checkpoint_in else None,
                alpha=args.alpha,
                disagreement_only=args.disagreement_only,
                device=args.device,
            )
        )


if __name__ == "__main__":
    main()
