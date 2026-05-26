#!/usr/bin/env python3
"""Phase 1: WEVA SFT on ToolPoker solver-annotated Parquet or BC fallback."""

import argparse
from pathlib import Path

from pokerbot.encoding.weva_encoder import WEVA_DEFAULT_K
from pokerbot.training.sft import SFTConfig, train_sft


def main() -> None:
    p = argparse.ArgumentParser(description="Train policy via solver or BC Parquet SFT")
    p.add_argument(
        "--mode",
        choices=("solver", "bc"),
        default="solver",
        help="solver=ToolPoker JSONL Parquet; bc=PokerBench CSV (not GTO-verified)",
    )
    p.add_argument("--parquet-glob", default=None, help="Override Parquet glob")
    p.add_argument("--weva-index", default="data/weva")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--solver-beta", type=float, default=0.3)
    p.add_argument("--num-weva-buckets", type=int, default=WEVA_DEFAULT_K)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()

    if args.parquet_glob is None:
        parquet_glob = "data/solver/*.parquet" if args.mode == "solver" else "data/bc/*.parquet"
    else:
        parquet_glob = args.parquet_glob

    cfg = SFTConfig(
        parquet_glob=parquet_glob,
        mode=args.mode,
        weva_index_path=Path(args.weva_index),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        solver_beta=args.solver_beta,
        num_weva_buckets=args.num_weva_buckets,
        device=args.device,
        checkpoint_dir=Path(args.checkpoint_dir),
        amp=args.amp,
        grad_accum=args.grad_accum,
        num_workers=args.num_workers,
        max_steps=args.max_steps,
    )
    train_sft(cfg)


if __name__ == "__main__":
    main()
