#!/usr/bin/env python3
"""Evaluate infoset-BR from an SD-CFR Leduc checkpoint (average strategy)."""

from __future__ import annotations

import argparse
from pathlib import Path

from pokerbot.training.leduc_tabular_cfr import (
    LEDUC_SD_CFR_EXPLOITABILITY_MBB,
    infoset_exploitability_mbb_from_policy,
)
from pokerbot.training.sd_cfr import SDCFRConfig, SDCFRTrainer, load_checkpoint


def eval_checkpoint(path: Path) -> tuple[float, float, float, float, float, int]:
    trainer = SDCFRTrainer(SDCFRConfig())
    completed = load_checkpoint(trainer, path)
    trainer.clear_policy_cache()
    expl, u0, u1, br0, br1 = infoset_exploitability_mbb_from_policy(
        trainer.average_strategy_policy, samples=90
    )
    return expl, u0, u1, br0, br1, completed


def main() -> None:
    p = argparse.ArgumentParser(description="Eval SD-CFR Leduc checkpoint infoset BR")
    p.add_argument("checkpoints", nargs="+", help="Checkpoint .pkl paths")
    args = p.parse_args()

    for ckpt in args.checkpoints:
        path = Path(ckpt)
        expl, u0, u1, br0, br1, completed = eval_checkpoint(path)
        print(
            f"{path.name}: iter={completed} infoset_BR={expl:.2f} mbb "
            f"(gate {LEDUC_SD_CFR_EXPLOITABILITY_MBB:.0f}) "
            f"u0={u0:.4f} u1={u1:.4f} br0={br0:.4f} br1={br1:.4f} "
            f"snapshots={completed}"
        )


if __name__ == "__main__":
    main()
