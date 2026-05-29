#!/usr/bin/env python3
"""Split prod600 into train_prod580 + 10 vanilla + 10 bunching held-out rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_SEED = 600580
HELDOUT_PER_CLASS = 10


def _board_str(board: list[str]) -> str:
    return "".join(board)


def _extract_meta(row: pd.Series) -> dict:
    raw = json.loads(row["raw_solver_json"])
    board = list(raw["solver_request"]["board"])
    hero = list(raw.get("hero_hole") or [])
    return {
        "spot_id": raw.get("spot_id"),
        "board": board,
        "board_key": _board_str(board),
        "hero_hole": hero,
        "row_key": f"{raw.get('spot_id')}|{_board_str(board)}|hero={''.join(hero)}",
    }


def _select_heldout_indices(
    df: pd.DataFrame,
    spot_class: str,
    n_boards: int,
    seed: int,
) -> list[int]:
    """Pick one row per distinct board, maximizing board diversity."""
    subset = df[df["spot_class"] == spot_class].copy()
    subset["_idx"] = subset.index
    metas = [_extract_meta(subset.iloc[i]) for i in range(len(subset))]
    subset["_board_key"] = [m["board_key"] for m in metas]
    subset["_row_key"] = [m["row_key"] for m in metas]

    # One representative row per board (stable: smallest row_key).
    reps = (
        subset.sort_values("_row_key")
        .groupby("_board_key", sort=True)
        .first()
        .reset_index()
    )
    board_keys = reps["_board_key"].tolist()
    rng = __import__("random").Random(seed + hash(spot_class) % 10000)
    rng.shuffle(board_keys)
    chosen_boards = board_keys[:n_boards]
    if len(chosen_boards) < n_boards:
        raise RuntimeError(
            f"Only {len(chosen_boards)} distinct boards for {spot_class}; need {n_boards}"
        )

    heldout_idx: list[int] = []
    for bk in chosen_boards:
        row = reps[reps["_board_key"] == bk].iloc[0]
        heldout_idx.append(int(row["_idx"]))
    return sorted(heldout_idx)


def split_prod600(
    input_parquet: Path,
    *,
    seed: int = DEFAULT_SEED,
    vanilla_out: Path,
    bunching_out: Path,
    train_out: Path,
    manifest_out: Path,
) -> dict:
    df = pd.read_parquet(input_parquet)
    vanilla_idx = _select_heldout_indices(df, "hu_postflop_2p", HELDOUT_PER_CLASS, seed)
    bunching_idx = _select_heldout_indices(df, "pseudo_hu_bunching", HELDOUT_PER_CLASS, seed + 1)
    heldout_idx = set(vanilla_idx + bunching_idx)

    heldout_vanilla = df.loc[vanilla_idx].reset_index(drop=True)
    heldout_bunching = df.loc[bunching_idx].reset_index(drop=True)
    train = df.drop(index=list(heldout_idx)).reset_index(drop=True)

    vanilla_out.parent.mkdir(parents=True, exist_ok=True)
    heldout_vanilla.to_parquet(vanilla_out, index=False)
    heldout_bunching.to_parquet(bunching_out, index=False)
    train.to_parquet(train_out, index=False)

    def _manifest_rows(indices: list[int]) -> list[dict]:
        rows = []
        for idx in indices:
            m = _extract_meta(df.loc[idx])
            rows.append(
                {
                    "row_index": int(idx),
                    "row_key": m["row_key"],
                    "spot_id": m["spot_id"],
                    "board": m["board"],
                    "board_key": m["board_key"],
                    "hero_hole": m["hero_hole"],
                    "spot_class": df.loc[idx, "spot_class"],
                }
            )
        return rows

    manifest = {
        "seed": seed,
        "input_parquet": str(input_parquet),
        "heldout_vanilla_parquet": str(vanilla_out),
        "heldout_bunching_parquet": str(bunching_out),
        "train_parquet": str(train_out),
        "heldout_vanilla": _manifest_rows(vanilla_idx),
        "heldout_bunching": _manifest_rows(bunching_idx),
        "counts": {
            "input_total": len(df),
            "train_total": len(train),
            "heldout_vanilla": len(heldout_vanilla),
            "heldout_bunching": len(heldout_bunching),
            "train_vanilla": int((train["spot_class"] == "hu_postflop_2p").sum()),
            "train_bunching": int((train["spot_class"] == "pseudo_hu_bunching").sum()),
        },
        "heldout_boards_vanilla": sorted({r["board_key"] for r in _manifest_rows(vanilla_idx)}),
        "heldout_boards_bunching": sorted({r["board_key"] for r in _manifest_rows(bunching_idx)}),
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Split prod600 into train580 + held-out")
    p.add_argument("--input", default="data/solver/train_prod600.parquet")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--train-out", default="data/solver/train_prod580.parquet")
    p.add_argument("--heldout-vanilla-out", default="data/solver/heldout_prod600_vanilla.parquet")
    p.add_argument("--heldout-bunching-out", default="data/solver/heldout_prod600_bunching.parquet")
    p.add_argument("--manifest-out", default="data/solver/heldout_prod600_manifest.json")
    args = p.parse_args()

    manifest = split_prod600(
        Path(args.input),
        seed=args.seed,
        vanilla_out=Path(args.heldout_vanilla_out),
        bunching_out=Path(args.heldout_bunching_out),
        train_out=Path(args.train_out),
        manifest_out=Path(args.manifest_out),
    )
    c = manifest["counts"]
    print(
        f"Split complete: train={c['train_total']} "
        f"(vanilla={c['train_vanilla']}, bunching={c['train_bunching']}) "
        f"heldout={c['heldout_vanilla']}+{c['heldout_bunching']}"
    )
    print(f"Wrote {args.manifest_out}")


if __name__ == "__main__":
    main()
