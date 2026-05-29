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
    p.add_argument("--skip-neural-cfr", action="store_true", help="Skip Step 1b neural CFR")
    p.add_argument("--neural-cfr-iterations", type=int, default=None)
    p.add_argument(
        "--quick-neural-cfr",
        action="store_true",
        help="Dev smoke: 10 CFR iterations, 8 traversals",
    )
    p.add_argument(
        "--allow-dev-weva-cfr",
        action="store_true",
        help="Allow neural CFR when --quick-weva dev index was used",
    )
    p.add_argument(
        "--compare-phase2",
        action="store_true",
        help="Run PPO vs CFR exploitability harness after training",
    )
    p.add_argument("--skip-phase3", action="store_true", help="Skip Step 3 MADT + SCAPT")
    p.add_argument("--quick-phase3", action="store_true", help="Dev smoke: 32 rollouts, 1 epoch, SCAPT pop 4")
    p.add_argument("--no-scapt", action="store_true", help="Step 3: static bot league instead of SCAPT")
    p.add_argument("--phase3-rollout-hands", type=int, default=None)
    p.add_argument("--phase3-epochs", type=int, default=None)
    p.add_argument(
        "--compare-phase3",
        action="store_true",
        help="Run HybridAgent vs frozen GTOCopyBot harness after Step 3",
    )
    p.add_argument("--skip-phase4", action="store_true", help="Skip Step 4 TurboReBeL validation")
    p.add_argument("--quick-phase4", action="store_true", help="Dev smoke: 50 hands, 500ms budget, mock leaf ok")
    p.add_argument(
        "--compare-phase4",
        action="store_true",
        help="Run SearchAgent vs frozen HybridAgent harness after Step 3",
    )
    p.add_argument(
        "--legacy-ppo-selfplay",
        action="store_true",
        help="Force Step 2 PPO even when gate manifest records Phase 2 CFR hero cutover",
    )
    args = p.parse_args()



    py = sys.executable

    device = args.device

    h = args.hours



    sp_updates = max(15, int(40 * h / 3))

    sft_epochs = 3 if h >= 2.5 else 2

    sp_hands = 48

    llm_n = max(2_000, int(8_000 * h / 3))
    cfr_iterations = args.neural_cfr_iterations or max(50, int(200 * h / 3))
    cfr_traversals = 8 if args.quick_neural_cfr else 64
    if args.quick_neural_cfr:
        cfr_iterations = min(cfr_iterations, 10)

    phase3_rollouts = args.phase3_rollout_hands or (32 if args.quick_phase3 else max(128, int(512 * h / 3)))
    phase3_epochs = args.phase3_epochs or (1 if args.quick_phase3 else 3)
    phase3_exploit_limit = 500 if args.quick_phase3 else 2000



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

    if not args.skip_neural_cfr:
        if args.quick_weva and not args.allow_dev_weva_cfr:
            print("Skipping neural CFR (--quick-weva dev index; pass --allow-dev-weva-cfr to override)")
        else:
            cfr_cmd = [
                py,
                "scripts/train_neural_cfr.py",
                "--iterations",
                str(cfr_iterations),
                "--traversals",
                str(cfr_traversals),
                "--checkpoint",
                "checkpoints/history_value_net.pkl",
            ]
            if args.quick_weva or args.allow_dev_weva_cfr:
                cfr_cmd.append("--allow-dev-weva")
            run(cfr_cmd, f"Step 1b: Neural CFR ({cfr_iterations} iterations)")
            run(
                [
                    py,
                    "scripts/distill_cfr_blueprint.py",
                    "--cfr-checkpoint",
                    "checkpoints/history_value_net.pkl",
                    "--hands",
                    "100" if args.quick_neural_cfr else "200",
                    "--epochs",
                    "2" if args.quick_neural_cfr else "3",
                    "--device",
                    device,
                ],
                "Step 1b-ii: Distill CFR blueprint to WEVA policy",
            )
    elif args.skip_neural_cfr:
        print("Skipping neural CFR (--skip-neural-cfr)")

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
        from pokerbot.eval.gate_reference import should_use_cfr_hero, resolve_cfr_hero_path

        use_cfr_hero = should_use_cfr_hero() and not args.legacy_ppo_selfplay
        if use_cfr_hero:
            cfr_path = resolve_cfr_hero_path()
            final_ckpt = str(cfr_path)
            print(f"Step 2: CFR blueprint hero (Phase 2 gate passed) — {cfr_path.name}")
        else:
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

    if not args.skip_phase3:
        phase3_cmd = [
            py,
            "scripts/train_phase3.py",
            "--checkpoint",
            final_ckpt,
            "--rollout-hands",
            str(phase3_rollouts),
            "--epochs",
            str(phase3_epochs),
            "--exploitative-limit",
            str(phase3_exploit_limit),
            "--device",
            device,
        ]
        if args.no_scapt:
            phase3_cmd.append("--no-scapt")
        if args.quick_phase3:
            phase3_cmd.extend(["--scapt-population", "4"])
        run(
            phase3_cmd,
            f"Step 3: MADT + SCAPT pretrain ({phase3_rollouts} rollouts, {phase3_epochs} epochs)",
        )
    else:
        print("Skipping Phase 3 (--skip-phase3)")

    if args.compare_phase3:
        run(
            [
                py,
                "scripts/eval_phase3_compare.py",
                "--hands",
                "200",
                "--device",
                device,
                "--output",
                "eval_phase3_compare.json",
            ],
            "Phase 3 comparison harness (HybridAgent vs frozen GTOCopyBot)",
        )

    if not args.skip_phase4:
        phase4_hands = "50" if args.quick_phase4 else "200"
        phase4_budget = "500" if args.quick_phase4 else "2000"
        phase4_cmd = [
            py,
            "scripts/validate_phase4.py",
            "--gate-manifest",
            "checkpoints/gate_reference.json",
        ]
        if args.quick_phase4 or not args.compare_phase4:
            phase4_cmd.append("--allow-mock-leaf")
            phase4_cmd.append("--no-require-hybrid-baseline")
        run(phase4_cmd, "Phase 4 artifact validation (warnings only unless --compare-phase4)")

        if args.compare_phase4 or args.quick_phase4:
            compare_cmd = [
                py,
                "scripts/eval_phase4_compare.py",
                "--hands",
                phase4_hands,
                "--time-budget-ms",
                phase4_budget,
                "--device",
                device,
                "--output",
                "eval_phase4_compare.json",
            ]
            if args.quick_phase4:
                compare_cmd.append("--use-mock-leaf")
                compare_cmd.append("--no-require-gate-reference")
            run(compare_cmd, "Phase 4 comparison harness (SearchAgent vs frozen HybridAgent)")
    else:
        print("Skipping Phase 4 (--skip-phase4)")

    if args.compare_phase2:
        run(
            [
                py,
                "scripts/eval_phase2_compare.py",
                "--ppo-checkpoint",
                final_ckpt,
                "--hands",
                "200",
                "--device",
                device,
                "--output",
                "eval_phase2_compare.json",
            ],
            "Phase 2 comparison harness (PPO vs CFR blueprint)",
        )

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

