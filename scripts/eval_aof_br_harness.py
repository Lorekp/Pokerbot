#!/usr/bin/env python3
"""Best-response and exploitability harness for AoF RL policies (obs v3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from aof_rl.exploit.harness import BrExploitConfig, BrExploitHarness


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
    p = argparse.ArgumentParser(description="AoF BR / exploitability harness")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--mode",
        choices=("checkpoint", "oracle", "scripted"),
        default="checkpoint",
    )
    p.add_argument("--eval-id", type=str, default="aof_br_exploit")
    p.add_argument("--hands", type=int, default=500)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--table-sizes", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--stack-bb", type=float, default=8.0)
    p.add_argument("--profile", choices=("fast", "default"), default="fast")
    p.add_argument("--teacher-samples", type=int, default=None)
    p.add_argument("--payoff-samples", type=int, default=None)
    p.add_argument("--teacher-version", type=int, default=2)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--no-hu", action="store_true")
    p.add_argument("--hu-mode", choices=("bucketed_169", "exact_combo"), default="bucketed_169")
    p.add_argument("--hu-equity-samples", type=int, default=400)
    p.add_argument("--hu-max-pairs", type=int, default=None)
    p.add_argument(
        "--multiway-br",
        choices=("oracle", "learned", "both"),
        default="oracle",
        help="Multiway BR modes (oracle and/or learned PPO BR)",
    )
    p.add_argument("--learned-br-steps", type=int, default=64)
    p.add_argument("--learned-br-seat", type=str, default=None)
    p.add_argument("--restricted-spec", type=str, action="append", default=[])
    p.add_argument("--archive-dir", type=str, default=None)
    p.add_argument("--archive-path", type=str, action="append", default=[])
    p.add_argument("--out-json", type=str, default=None)
    p.add_argument("--out-md", type=str, default=None)
    p.add_argument("--update-checkpoint", action="store_true")
    args = p.parse_args()

    device = _resolve_device(args.device, allow_cpu=args.allow_cpu)
    mw: tuple[str, ...]
    if args.multiway_br == "both":
        mw = ("oracle", "learned")
    else:
        mw = (args.multiway_br,)

    report = BrExploitHarness(
        BrExploitConfig(
            checkpoint=args.checkpoint,
            eval_id=args.eval_id,
            hands=args.hands,
            seeds=args.seeds,
            table_sizes=tuple(args.table_sizes),
            stack_bb=args.stack_bb,
            profile=args.profile,
            teacher_samples=args.teacher_samples,
            payoff_samples=args.payoff_samples,
            teacher_version=args.teacher_version,
            device=device,
            compute_hu=not args.no_hu,
            hu_mode=args.hu_mode,
            hu_equity_samples=args.hu_equity_samples,
            hu_max_pairs=args.hu_max_pairs,
            multiway_br=mw,  # type: ignore[arg-type]
            learned_br_steps=args.learned_br_steps,
            learned_br_seat=args.learned_br_seat,
            restricted_spec_paths=args.restricted_spec,
            archive_dir=args.archive_dir,
            archive_paths=args.archive_path,
            policy_mode=args.mode if not args.checkpoint else "checkpoint",
            out_json=args.out_json,
            out_md=args.out_md,
            update_checkpoint=args.update_checkpoint,
        )
    ).run()

    if report.hu_exploitability:
        hu = report.hu_exploitability
        print(
            f"aof_br_exploit: HU exploitability ({hu.mode})="
            f"{hu.exploitability_bb_per_hand:.4f} bb/hand"
        )
    for ts, rows in report.multiway_deviation_oracle.items():
        for r in rows:
            print(
                f"  oracle {ts}p {r.seat}: deviation_gain={r.deviation_gain_bb_per_hand:.4f}"
            )
    if args.out_json:
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
