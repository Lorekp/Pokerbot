#!/usr/bin/env python3
"""Evaluate AoF RL policies with paired CRN, exploitability, and reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from aof_rl.eval.harness import AofEvalHarness, EvalConfig


def _resolve_device(name: str, *, allow_cpu: bool) -> str:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        if not allow_cpu:
            raise SystemExit(
                "CUDA requested but not available. Pass --allow-cpu or use --device cpu."
            )
        return "cpu"
    if name.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    return name


def main() -> None:
    p = argparse.ArgumentParser(description="AoF RL evaluation harness")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--baseline-checkpoint", type=str, default=None)
    p.add_argument(
        "--mode",
        choices=("checkpoint", "oracle", "scripted"),
        default="checkpoint",
    )
    p.add_argument("--eval-id", type=str, default="aof_eval")
    p.add_argument("--hands", type=int, default=2000)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--table-sizes", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--stack-bb", type=float, default=8.0)
    p.add_argument("--profile", choices=("fast", "default"), default="fast")
    p.add_argument("--teacher-samples", type=int, default=None)
    p.add_argument("--payoff-samples", type=int, default=None)
    p.add_argument("--teacher-version", type=int, default=2)
    p.add_argument("--obs-version", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--no-deterministic", action="store_false", dest="deterministic")
    p.add_argument("--paired", action="store_true")
    p.add_argument("--no-hu-exploitability", action="store_true")
    p.add_argument(
        "--hu-br-mode",
        choices=("bucketed_169", "exact_combo"),
        default="bucketed_169",
    )
    p.add_argument("--hu-equity-samples", type=int, default=400)
    p.add_argument("--hu-max-pairs", type=int, default=None)
    p.add_argument("--no-multiway", action="store_true")
    p.add_argument("--out-json", type=str, default=None)
    p.add_argument("--out-md", type=str, default=None)
    args = p.parse_args()

    if args.mode == "checkpoint" and not args.checkpoint:
        args.checkpoint = "checkpoints/aof_rl/policy_league_v3_smoke.pt"

    device = _resolve_device(args.device, allow_cpu=args.allow_cpu)
    cfg = EvalConfig(
        checkpoint=args.checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
        eval_id=args.eval_id,
        hands=args.hands,
        seeds=args.seeds,
        table_sizes=tuple(args.table_sizes),
        stack_bb=args.stack_bb,
        profile=args.profile,
        teacher_samples=args.teacher_samples,
        payoff_samples=args.payoff_samples,
        teacher_version=args.teacher_version,
        obs_version=args.obs_version,
        device=device,
        deterministic=args.deterministic,
        paired=args.paired,
        compute_hu_exploitability=not args.no_hu_exploitability,
        hu_br_mode=args.hu_br_mode,
        hu_equity_samples=args.hu_equity_samples,
        hu_max_pairs=args.hu_max_pairs,
        compute_multiway=not args.no_multiway,
        policy_mode=args.mode,
        out_json=args.out_json,
        out_md=args.out_md,
    )
    report = AofEvalHarness(cfg).run()
    a = report.aggregate
    print(
        f"{report.eval_id}: bb/hand={a.bb_per_hand:.4f} "
        f"CI=[{a.bb_per_hand_ci95[0]:.4f},{a.bb_per_hand_ci95[1]:.4f}] "
        f"hands={a.hands}"
    )
    if report.aggregate.disagreement_rate_vs_teacher is not None:
        print(f"  teacher disagreement: {report.aggregate.disagreement_rate_vs_teacher:.2%}")
    if report.hu_exploitability:
        hu = report.hu_exploitability
        print(f"  HU exploitability ({hu.mode}): {hu.exploitability_bb_per_hand:.4f} bb/hand")
    if report.paired_comparison:
        pc = report.paired_comparison
        print(
            f"  paired Δ bb/hand ({pc.policy_a} - {pc.policy_b}): "
            f"{pc.mean_delta_bb_per_hand:.4f}"
        )
    if args.out_json:
        print(f"Wrote {args.out_json}")
    if args.out_md:
        print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
