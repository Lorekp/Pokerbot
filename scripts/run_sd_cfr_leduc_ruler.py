#!/usr/bin/env python3
"""Fresh SD-CFR Leduc short ruler: BR + L1-vs-tabular at checkpoints 10/20/40."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pokerbot.training.leduc_tabular_cfr import tabular_cfr
from pokerbot.training.sd_cfr import SDCFRConfig, SDCFRTrainer, save_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="SD-CFR Leduc pre-committed ruler run")
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--traversals", type=int, default=32)
    p.add_argument(
        "--enumerate-deals",
        action="store_true",
        help="All 90 Leduc deals × 2 seats per sweep (see --deal-sweeps)",
    )
    p.add_argument(
        "--deal-sweeps",
        type=int,
        default=1,
        help="Repeat full deal enumeration per iteration (1080 visits when 6)",
    )
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--checkpoint-dir", default="checkpoints/sd_cfr_leduc_ruler_v4")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--tabular-iters", type=int, default=800)
    args = p.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    if ckpt_dir.exists():
        for f in ckpt_dir.glob("*"):
            f.unlink()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tabular CFR reference ({args.tabular_iters} iters)...", flush=True)
    tabular_avg, _ = tabular_cfr(iterations=args.tabular_iters, seed=0)

    cfg = SDCFRConfig(
        iterations=args.iterations,
        traversals_per_iter=args.traversals,
        seed=args.seed,
        enumerate_all_deals=args.enumerate_deals,
        deal_sweeps=args.deal_sweeps,
    )
    trainer = SDCFRTrainer(cfg)
    trainer._tabular_avg_ref = tabular_avg

    visits = (
        f"{90 * 2 * max(1, args.deal_sweeps)} enumerated"
        if args.enumerate_deals
        else f"{args.traversals} random×2"
    )
    print(
        f"SD-CFR ruler: {args.iterations} iters, {visits} traverser visits/iter, "
        f"ckpt every {args.checkpoint_every}, dir={ckpt_dir}",
        flush=True,
    )
    trainer.train(
        log_every=1,
        checkpoint_dir=ckpt_dir,
        checkpoint_every=args.checkpoint_every,
        start_iter=0,
    )
    trainer._completed_iterations = args.iterations
    out = ckpt_dir / "final.pkl"
    save_checkpoint(trainer, out)
    shutil.copy2(ckpt_dir / "sd_cfr_leduc_latest.pkl", out)
    print("Ruler run complete.", flush=True)


if __name__ == "__main__":
    main()
