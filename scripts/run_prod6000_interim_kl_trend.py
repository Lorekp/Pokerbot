#!/usr/bin/env python3
"""Merge prod6000 shards + run interim SFT + held-out KL trend read."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from pokerbot.solver.export import merge_shards
from pokerbot.solver.label_completion import spot_key_hash

ROOT = Path(__file__).resolve().parents[1]


def merge_prod6000_shards(shard_dir: Path, output: Path) -> int:
    n = merge_shards(shard_dir, output, spot_key_hash_fn=spot_key_hash)
    return n


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
    overlap = keys.isin(heldout_keys).sum()
    if overlap:
        df = df.loc[~keys.isin(heldout_keys)].reset_index(drop=True)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard-dir", default="data/solver/train_prod6000_shards")
    p.add_argument("--merged-out", default="data/solver/train_prod6000_interim.parquet")
    p.add_argument("--heldout-manifest", default="data/solver/heldout_prod600_manifest.json")
    p.add_argument("--train-out", default="data/solver/train_prod6000_interim_train.parquet")
    p.add_argument("--checkpoint-out", default="checkpoints/policy_sft_prod6000_interim.pt")
    p.add_argument("--kl-out", default="logs/prod6000_interim_holdout_kl.json")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    shard_dir = ROOT / args.shard_dir
    merged_out = ROOT / args.merged_out
    n = merge_prod6000_shards(shard_dir, merged_out)
    print(json.dumps({"merged_rows": n, "output": str(merged_out)}))

    if args.merge_only:
        return

    df = pd.read_parquet(merged_out)
    df = exclude_prod600_heldout_rows(df, ROOT / args.heldout_manifest)
    train_out = ROOT / args.train_out
    train_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(train_out, index=False)
    vanilla_n = int((df["spot_class"] == "hu_postflop_2p").sum())
    bunch_n = int((df["spot_class"] == "pseudo_hu_bunching").sum())
    print(json.dumps({"train_rows": len(df), "vanilla": vanilla_n, "bunching": bunch_n}))

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
            "--amp",
            "--checkpoint-output",
            str(ROOT / args.checkpoint_out),
        ],
        cwd=ROOT,
        check=True,
    )

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
            str(ROOT / "logs/prod600_post_sft_holdout_kl.json"),
            "--output",
            str(ROOT / args.kl_out),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
