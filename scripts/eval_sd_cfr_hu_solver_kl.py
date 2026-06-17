#!/usr/bin/env python3

"""Gate A: diagnostic held-out KL — SD-CFR policy vs solver labels (JAX spot replay)."""



from __future__ import annotations



import argparse

import json

from pathlib import Path



from pokerbot.training.nlhe_gate_a import run_gate_a_report

from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer, load_nlhe_checkpoint





def main() -> None:

    p = argparse.ArgumentParser(description="Gate A diagnostic solver KL (SD-CFR policy)")

    p.add_argument("--checkpoint", default=None, help="SD-CFR pkl; omit for SFT-init iter-0 trainer")

    p.add_argument("--sft-checkpoint", default="checkpoints/policy_sft_prod6000.pt")

    p.add_argument("--heldout-vanilla", default="data/solver/heldout_prod600_vanilla.parquet")

    p.add_argument("--heldout-bunching", default="data/solver/heldout_prod600_bunching.parquet")

    p.add_argument("--output", default="logs/sd_cfr_hu_gate_a_kl.json")

    p.add_argument("--device", default="cpu")

    args = p.parse_args()



    if args.checkpoint:

        trainer = SDCFRNLHETrainer(SDCFRNLHEConfig(iterations=0, traversals_per_iter=0, sft_checkpoint=None))

        load_nlhe_checkpoint(trainer, Path(args.checkpoint))

    else:

        trainer = SDCFRNLHETrainer(

            SDCFRNLHEConfig(iterations=0, traversals_per_iter=0, sft_checkpoint=args.sft_checkpoint)

        )



    report = run_gate_a_report(

        trainer=trainer,

        checkpoint=args.checkpoint,

        sft_path=args.sft_checkpoint,

        heldout_vanilla=args.heldout_vanilla,

        heldout_bunching=args.heldout_bunching,

        device=args.device,

    )

    out = Path(args.output)

    out.parent.mkdir(parents=True, exist_ok=True)

    slim = {

        k: v

        for k, v in report.items()

        if k not in ("vanilla", "bunching")

    }

    slim["vanilla_mean_kl"] = report["vanilla"]["mean_kl"]

    slim["bunching_mean_kl"] = report["bunching"]["mean_kl"]

    slim["vanilla_replay_failures"] = report["vanilla"]["replay_failures"]

    slim["bunching_replay_failures"] = report["bunching"]["replay_failures"]

    out.write_text(json.dumps(slim, indent=2), encoding="utf-8")

    print(json.dumps(slim, indent=2))





if __name__ == "__main__":

    main()

