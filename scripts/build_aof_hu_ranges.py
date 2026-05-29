#!/usr/bin/env python3
"""Build HU push/fold tables via tabular CFR+ and write data/aof/hu_pushfold_8bb.json."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_equity import warmup
from pokerbot.solver.hu_preflop_dcfr import (
    HuSolveResult,
    _build_169_buckets,
    _freqs_to_range_string,
    expand_to_169,
    save_hu_solution,
    solve_hu_pushfold,
)


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description="Build AoF HU solver ranges (8bb)")
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--equity-samples", type=int, default=600)
    p.add_argument(
        "--bucket-stride",
        type=int,
        default=5,
        help="Subsample 169 buckets (5 -> ~34 buckets) for tractable eq matrix",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        default="data/aof/hu_pushfold_8bb.json",
        help="Output JSON path",
    )
    args = p.parse_args()

    warmup()
    result = solve_hu_pushfold(
        stack_bb=8.0,
        iterations=args.iterations,
        seed=args.seed,
        equity_samples=args.equity_samples,
        bucket_stride=args.bucket_stride,
    )
    result.manifest["git_hash"] = _git_hash()
  # Expand coarse frequencies to full 169 for export
    full_labels, full_combos = _build_169_buckets()
    _, sb_full = expand_to_169(result.bucket_labels, result.sb_push_freq)
    _, bb_full = expand_to_169(result.bucket_labels, result.bb_call_freq)
    call_rng, call_pct = _freqs_to_range_string(
        full_labels, bb_full, full_combos, min_freq=0.35
    )
    shove_rng, _ = _freqs_to_range_string(
        full_labels, sb_full, full_combos, min_freq=0.45
    )
    result = HuSolveResult(
        stack_bb=result.stack_bb,
        iterations=result.iterations,
        exploitability_bb=result.exploitability_bb,
        sb_push_freq=sb_full,
        bb_call_freq=bb_full,
        bucket_labels=full_labels,
        call_range_str=call_rng,
        call_pct=call_pct,
        shove_range_str=shove_rng,
        manifest=result.manifest,
    )
    out = _ROOT / args.out
    save_hu_solution(out, result)
    print(f"Wrote {out}")
    print(f"exploitability_bb={result.exploitability_bb:.4f}")
    print(f"call_pct={result.call_pct:.3f} range={result.call_range_str}")
    print(f"shove_range={result.shove_range_str}")


if __name__ == "__main__":
    main()
