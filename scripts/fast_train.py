#!/usr/bin/env python3

"""2–4 hour fast path to a functioning hybrid poker model."""



from __future__ import annotations



import argparse

import subprocess

import sys

import time

from glob import glob

from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]





def run(cmd: list[str], desc: str) -> None:

    print(f"\n{'=' * 60}\n{desc}\n{'=' * 60}", flush=True)

    t0 = time.time()

    subprocess.run(cmd, cwd=ROOT, check=True)

    print(f"Done in {(time.time() - t0) / 60:.1f} min", flush=True)





def _has_solver_parquet() -> bool:

    return bool(glob(str(ROOT / "data" / "solver" / "*.parquet")))





def _has_bc_parquet() -> bool:

    return bool(glob(str(ROOT / "data" / "bc" / "*.parquet")))





def main() -> None:

    p = argparse.ArgumentParser(description="Fast 2–4h training pipeline")

    p.add_argument("--device", default="cuda")

    p.add_argument("--skip-sft", action="store_true", help="Resume: SFT checkpoint already exists")

    p.add_argument(

        "--legacy-llm-distill",

        action="store_true",

        help="Opt-in: run deprecated LLM cache + distill steps (Steps 2–3)",

    )

    p.add_argument("--skip-selfplay", action="store_true")

    p.add_argument("--hours", type=float, default=3.0, help="Target budget ( tunes data limits )")

    p.add_argument(

        "--quick-weva",

        action="store_true",

        help="Dev only: build WEVA index with K=1000 hole_only and fewer rollouts",

    )

    args = p.parse_args()



    py = sys.executable

    device = args.device

    h = args.hours



    sp_updates = max(15, int(40 * h / 3))

    sft_epochs = 3 if h >= 2.5 else 2

    sp_hands = 48

    llm_n = max(2_000, int(8_000 * h / 3))



    total_t0 = time.time()

    print(f"Fast train started (target ~{h:.1f}h, device={device})")



    if args.skip_sft:

        bc_ckpt = ROOT / "checkpoints" / "policy_sft_bc.pt"

        solver_ckpt = ROOT / "checkpoints" / "policy_sft.pt"

        if bc_ckpt.exists():

            print(f"Skipping SFT (using existing {bc_ckpt.name} — BC, not GTO-verified)")

            ckpt_in = str(bc_ckpt)

        else:

            print(f"Skipping SFT (using existing {solver_ckpt.name})")

            ckpt_in = str(solver_ckpt)

    else:

        weva_dir = ROOT / "data" / "weva"

        weva_cmd = [

            py,

            "scripts/build_weva_index.py",

            "--output",

            "data/weva",

            "--device",

            device,

        ]

        if args.quick_weva:

            weva_cmd.extend(["--k", "1000", "--feature-mode", "hole_only", "--rollouts", "32", "--max-roots", "128"])

        run(weva_cmd, "Step 0: Build or refresh WEVA index (skips if build_fingerprint matches)")



        sft_mode = "solver"

        parquet_glob = "data/solver/*.parquet"

        if not _has_solver_parquet():

            print(

                "\n*** No solver Parquet at data/solver/*.parquet ***\n"

                "Falling back to BC mode (PokerBench CSV — NOT GTO-verified).\n"

                "For real Phase 1 SFT, export ToolPoker JSONL per docs/phase1-solver-data.md\n"

            )

            sft_mode = "bc"

            parquet_glob = "data/bc/*.parquet"

            if not _has_bc_parquet():

                run(

                    [

                        py,

                        "scripts/bootstrap_bc_parquet.py",

                        "--output",

                        "data/bc/train.parquet",

                        "--weva-index",

                        "data/weva",

                        "--preflop-limit",

                        "2000",

                        "--postflop-limit",

                        "2000",

                    ],

                    "Step 0b: Bootstrap BC Parquet (smoke-test fallback)",

                )



        run(

            [

                py,

                "scripts/train_sft.py",

                "--mode",

                sft_mode,

                "--parquet-glob",

                parquet_glob,

                "--weva-index",

                "data/weva",

                "--epochs",

                str(sft_epochs),

                "--batch-size",

                "128",

                "--device",

                device,

                "--amp",

                "--solver-beta",

                "0.3",

            ],

            f"Step 1/3: WEVA SFT ({sft_mode} mode, {sft_epochs} epochs)",

        )

        ckpt_in = str(ROOT / "checkpoints" / ("policy_sft_bc.pt" if sft_mode == "bc" else "policy_sft.pt"))



    final_ckpt = ckpt_in



    if args.legacy_llm_distill:

        cache_path = ROOT / "data" / "cache" / "llm_labels.parquet"

        if not cache_path.exists():

            run(

                [

                    py,

                    "scripts/distill_from_llm.py",

                    "--cache-only",

                    "--preflop-limit",

                    str(llm_n),

                    "--postflop-limit",

                    str(llm_n),

                    "--batch-size",

                    "8",

                    "--device",

                    device,

                ],

                f"Legacy Step 2: LLM label cache ({llm_n:,} pre + {llm_n:,} post)",

            )

        else:

            print(f"Using existing LLM cache at {cache_path}")

        run(

            [

                py,

                "scripts/distill_from_llm.py",

                "--train",

                "--checkpoint-in",

                ckpt_in,

                "--device",

                device,

            ],

            "Legacy Step 3: Distill CNN from LLM cache",

        )

        final_ckpt = "checkpoints/policy_distilled.pt"



    if not args.skip_selfplay:

        run(

            [

                py,

                "scripts/train_self_play.py",

                "--checkpoint",

                final_ckpt,

                "--updates",

                str(sp_updates),

                "--hands",

                str(sp_hands),

                "--league-opponents",

                "cnn,nit,lag,donk,human_like",

                "--device",

                device,

            ],

            f"Step 2/3: Mixed self-play ({sp_updates} updates)",

        )

        final_ckpt = "checkpoints/policy_selfplay_mixed.pt"



        run(

            [

                py,

                "scripts/train_phase3.py",

                "--checkpoint",

                final_ckpt,

                "--pretrain-transformer",

                "--exploitative-limit",

                "2000",

                "--updates",

                "10",

                "--hands",

                "32",

                "--device",

                device,

            ],

            "Step 3/3: Phase 3 pretrain + short mixed opponents",

        )

        final_ckpt = "checkpoints/policy_selfplay_mixed.pt"



    run(

        [

            py,

            "scripts/evaluate.py",

            "--checkpoint",

            final_ckpt,

            "--device",

            device,

            "--hands",

            "200",

            "--output",

            "eval_results_fast.json",

        ],

        "Evaluation",

    )



    elapsed = (time.time() - total_t0) / 3600

    print(f"\nFast train complete in {elapsed:.2f}h")

    print(f"Final checkpoint: {final_ckpt}")

    print("Load with HybridAgent or GTOCopyBot from pokerbot.eval.bots")





if __name__ == "__main__":

    main()

