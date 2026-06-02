#!/usr/bin/env python3
"""Train SD-CFR on Leduc poker (Phase 2b validation ruler)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pokerbot.training.leduc_tabular_cfr import (
    LEDUC_SD_CFR_EXPLOITABILITY_MBB,
    infoset_exploitability_mbb_from_policy,
)
from pokerbot.training.sd_cfr import (
    SDCFRConfig,
    SDCFRTrainer,
    load_checkpoint,
    save_checkpoint,
)


def main() -> None:
    p = argparse.ArgumentParser(description="SD-CFR on Leduc Hold'em")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--traversals", type=int, default=32)
    p.add_argument("--output", default="checkpoints/sd_cfr_leduc.pkl")
    p.add_argument("--checkpoint-dir", default="checkpoints/sd_cfr_leduc_run")
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--resume", default="", help="Path to checkpoint to resume from")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = SDCFRConfig(
        iterations=args.iterations,
        traversals_per_iter=args.traversals,
        seed=args.seed,
    )
    trainer = SDCFRTrainer(cfg)
    start_iter = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_iter = load_checkpoint(trainer, resume_path)
        print(f"Resumed from {resume_path} at iter {start_iter}", flush=True)

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    losses = trainer.train(
        log_every=1,
        checkpoint_dir=ckpt_dir,
        checkpoint_every=args.checkpoint_every,
        start_iter=start_iter,
    )
    trainer._completed_iterations = args.iterations
    trainer.clear_policy_cache()
    sd_expl, u0, u1, br0, br1 = infoset_exploitability_mbb_from_policy(
        trainer.average_strategy_policy, samples=90
    )
    out_path = Path(args.output)
    save_checkpoint(trainer, out_path)
    if ckpt_dir:
        save_checkpoint(trainer, ckpt_dir / "sd_cfr_leduc_latest.pkl")
        if out_path.resolve() != (ckpt_dir / "sd_cfr_leduc_latest.pkl").resolve():
            shutil.copy2(ckpt_dir / "sd_cfr_leduc_latest.pkl", out_path)

    print(f"Final loss: {losses[-1]:.6f}" if losses else "no loss")
    print(
        f"Infoset BR: {sd_expl:.2f} mbb (gate {LEDUC_SD_CFR_EXPLOITABILITY_MBB:.0f} mbb), "
        f"u0={u0:.4f} u1={u1:.4f} br0={br0:.4f} br1={br1:.4f}"
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
