#!/usr/bin/env python3
"""Definitive prod6000 held-out KL trend: full 3000-label SFT + per-scope read."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from pokerbot.data.sft_collision_check import require_no_sft_input_collisions
from pokerbot.solver.export import merge_shards
from pokerbot.solver.label_completion import spot_key_hash

ROOT = Path(__file__).resolve().parents[1]

TREND_PRIOR = {
    "prod60_combined_heldout_kl": 1.004,
    "prod600_combined_heldout_kl": 0.309,
    "prod600_vanilla_heldout_kl": 0.176,
    "prod600_bunching_heldout_kl": 0.443,
}


def exclude_prod600_heldout_rows(df: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    heldout_keys = {
        r["row_key"]
        for r in manifest.get("heldout_vanilla", []) + manifest.get("heldout_bunching", [])
    }

    def row_key(row: pd.Series) -> str:
        raw = json.loads(row["raw_solver_json"])
        board = raw["solver_request"]["board"]
        hero = raw.get("hero_hole") or []
        return f"{raw.get('spot_id')}|{''.join(board)}|hero={''.join(hero)}"

    keys = df.apply(row_key, axis=1)
    overlap = int(keys.isin(heldout_keys).sum())
    if overlap:
        df = df.loc[~keys.isin(heldout_keys)].reset_index(drop=True)
    return df, overlap


def spot_type_counts(df: pd.DataFrame) -> dict[str, int]:
    def spot_id(row: pd.Series) -> str:
        raw = json.loads(row["raw_solver_json"])
        return str(raw.get("spot_id", ""))

    ids = df.apply(spot_id, axis=1)
    return dict(Counter(ids))


def main() -> None:
    p = argparse.ArgumentParser(description="Definitive prod6000 held-out KL trend")
    p.add_argument("--shard-dir", default="data/solver/train_prod6000_shards")
    p.add_argument("--merged-out", default="data/solver/train_prod6000.parquet")
    p.add_argument("--heldout-manifest", default="data/solver/heldout_prod600_manifest.json")
    p.add_argument("--train-out", default="data/solver/train_prod6000_sft.parquet")
    p.add_argument("--checkpoint-out", default="checkpoints/policy_sft_prod6000.pt")
    p.add_argument("--kl-out", default="logs/prod6000_definitive_holdout_kl.json")
    p.add_argument("--pre-sft-json", default="logs/prod600_pre_sft_diagnostic.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    shard_dir = ROOT / args.shard_dir
    merged_out = ROOT / args.merged_out
    n_shards = len(list(shard_dir.glob("*.parquet")))
    n_merged = merge_shards(shard_dir, merged_out, spot_key_hash_fn=spot_key_hash)
    print(json.dumps({"shard_files": n_shards, "merged_rows": n_merged, "output": str(merged_out)}))

    if args.merge_only:
        return

    df = pd.read_parquet(merged_out)
    df, heldout_overlap = exclude_prod600_heldout_rows(df, ROOT / args.heldout_manifest)
    spot_counts = spot_type_counts(df)
    train_out = ROOT / args.train_out
    train_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(train_out, index=False)

    collision = require_no_sft_input_collisions(str(train_out))
    readiness = {
        "train_rows": len(df),
        "heldout_overlap_excluded": heldout_overlap,
        "spot_type_counts": spot_counts,
        "expected_sb_btn_bunching": {"sb_vs_bb_flop": 1000, "btn_vs_bb_flop": 1000, "bunching_postflop": 1000},
        "collision_precondition": collision,
        "coverage_complete": (
            spot_counts.get("sb_vs_bb_flop", 0) == 1000
            and spot_counts.get("btn_vs_bb_flop", 0) == 1000
            and spot_counts.get("bunching_postflop", 0) == 1000
            and collision["n_unique_inputs"] == collision["n_rows"]
        ),
    }
    print(json.dumps(readiness, indent=2))
    if not readiness["coverage_complete"]:
        print("WARNING: train set not at expected 3000/3000 unique coverage", file=sys.stderr)

    if args.skip_train:
        return

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train_sft.py"),
            "--mode",
            "solver",
            "--parquet-glob",
            str(train_out),
            "--epochs",
            "200",
            "--batch-size",
            "8",
            "--lr",
            "1e-3",
            "--device",
            args.device,
            "--checkpoint-output",
            str(ROOT / args.checkpoint_out),
        ]
        + (["--amp"] if args.device != "cpu" else []),
        cwd=ROOT,
        check=True,
    )

    kl_out = ROOT / args.kl_out
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prod600_post_sft_holdout_kl.py"),
            "--checkpoint",
            str(ROOT / args.checkpoint_out),
            "--heldout-vanilla",
            str(ROOT / "data/solver/heldout_prod600_vanilla.parquet"),
            "--heldout-bunching",
            str(ROOT / "data/solver/heldout_prod600_bunching.parquet"),
            "--pre-sft-json",
            str(ROOT / args.pre_sft_json),
            "--output",
            str(kl_out),
            "--device",
            args.device,
        ],
        cwd=ROOT,
        check=True,
    )

    kl = json.loads(kl_out.read_text(encoding="utf-8"))
    vanilla = kl["heldout_vanilla_kl"]
    bunching = kl["heldout_bunching_kl"]
    combined = kl["heldout_combined_kl"]

    trend = {
        "prior_trend_points": TREND_PRIOR,
        "prod6000_definitive": {
            "combined_mean_kl": combined["mean_kl"],
            "combined_median_kl": combined["median_kl"],
            "combined_count_kl_lt_0_10": combined["count_kl_lt_0_10"],
            "vanilla_mean_kl": vanilla["mean_kl"],
            "vanilla_median_kl": vanilla["median_kl"],
            "vanilla_count_kl_lt_0_10": vanilla["count_kl_lt_0_10"],
            "bunching_mean_kl": bunching["mean_kl"],
            "bunching_median_kl": bunching["median_kl"],
            "bunching_count_kl_lt_0_10": bunching["count_kl_lt_0_10"],
        },
        "trend_deltas_vs_prod600": {
            "combined": round(combined["mean_kl"] - TREND_PRIOR["prod600_combined_heldout_kl"], 4),
            "vanilla": round(vanilla["mean_kl"] - TREND_PRIOR["prod600_vanilla_heldout_kl"], 4),
            "bunching": round(bunching["mean_kl"] - TREND_PRIOR["prod600_bunching_heldout_kl"], 4),
        },
        "readiness": readiness,
        "train_parquet": str(train_out),
        "checkpoint": str(ROOT / args.checkpoint_out),
        "kl_report": str(kl_out),
    }

    trend_out = ROOT / "logs/prod6000_definitive_kl_trend.json"
    trend_out.write_text(json.dumps(trend, indent=2), encoding="utf-8")
    print(json.dumps(trend, indent=2))
    print(f"Wrote {trend_out}")


if __name__ == "__main__":
    main()
