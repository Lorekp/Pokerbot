#!/usr/bin/env python3
"""Bootstrap behavioral-cloning Parquet from PokerBench CSV (not solver-verified)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pokerbot.data.bc_parquet import LABEL_SOURCE_BC, make_bc_policy_from_label
from pokerbot.data.pokerbench_csv import iter_csv_rows
from pokerbot.encoding.csv_encoder import encode_csv_row
from pokerbot.encoding.weva_encoder import WEVAMapper, weva_id_from_tensors


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bootstrap BC Parquet from PokerBench CSV (NOT solver-verified)"
    )
    p.add_argument("--output", type=Path, default=Path("data/bc/train.parquet"))
    p.add_argument("--weva-index", type=Path, default=Path("data/weva"))
    p.add_argument("--preflop-limit", type=int, default=5000)
    p.add_argument("--postflop-limit", type=int, default=5000)
    p.add_argument("--cache-dir", type=Path, default=Path("data/pokerbench"))
    args = p.parse_args()

    mapper = WEVAMapper.from_directory(args.weva_index)
    rows: list[dict] = []

    for stage, limit in (("preflop", args.preflop_limit), ("postflop", args.postflop_limit)):
        if limit <= 0:
            continue
        try:
            samples = iter_csv_rows(stage, "train", limit=limit, cache_dir=args.cache_dir)
        except FileNotFoundError as exc:
            print(f"Warning: skipping {stage} — {exc}")
            continue
        for sample in samples:
            enc = encode_csv_row(sample.row, sample.stage)
            weva_id = weva_id_from_tensors(enc["cards"], enc["position"], mapper=mapper)
            label = int(enc["label"])
            legal = enc["legal_mask"]
            rows.append(
                {
                    "weva_bucket_id": weva_id,
                    "actions": enc["actions"].tolist(),
                    "position": enc["position"].tolist(),
                    "legal_mask": legal.tolist(),
                    "bc_label": label,
                    "bc_policy": make_bc_policy_from_label(label, legal).tolist(),
                    "label_source": LABEL_SOURCE_BC,
                    "stage": stage,
                }
            )

    if not rows:
        raise ValueError(
            "No BC rows generated. Download PokerBench CSV to data/pokerbench/ "
            "or reduce limits."
        )

    import pandas as pd

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(
        f"Wrote {len(rows)} BC rows to {args.output} "
        f"(label_source={LABEL_SOURCE_BC!r} — NOT solver-verified)"
    )


if __name__ == "__main__":
    main()
