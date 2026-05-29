#!/usr/bin/env python3
"""Repair solver Parquet: action alignment, cards, and spot_scope encoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pokerbot.engine.cards import Card
from pokerbot.encoding.spot_scope import encode_spot_scope
from pokerbot.encoding.tensors import encode_cards_from_hole_board
from pokerbot.encoding.toolpoker_encoder import (
    action_distribution_to_policy,
    solver_action_keys_to_legal_mask,
)


def repair_row(raw_json: str, row: dict) -> dict:
    data = json.loads(raw_json)
    dist = data.get("action_distribution") or {}
    if not dist:
        raise ValueError("missing action_distribution in raw_solver_json")

    response = data.get("solver_response") or {}
    avail = response.get("available_actions") or list(dist.keys())
    legal_mask = solver_action_keys_to_legal_mask([str(a) for a in avail])
    policy = action_distribution_to_policy(dist, legal_mask)
    solver_label = int(np.argmax(policy))

    hole = [Card.from_str(c) for c in data["hero_hole"]]
    board = [Card.from_str(c) for c in data["solver_request"]["board"]]
    cards = encode_cards_from_hole_board(hole, board)
    spot_scope = encode_spot_scope(
        str(row.get("spot_class", "")),
        str(row.get("oop_range", "")),
        str(row.get("ip_range", "")),
    )

    return {
        "legal_mask": legal_mask.tolist(),
        "solver_policy": policy.tolist(),
        "solver_label": solver_label,
        "cards": cards.tolist(),
        "spot_scope": spot_scope.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Repair solver Parquet rows")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None, help="Default: overwrite input")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path
    df = pd.read_parquet(in_path)

    fixed = 0
    for i, row in df.iterrows():
        if row.get("label_source") != "postflop_solver":
            continue
        raw = row.get("raw_solver_json")
        if not raw:
            continue
        repaired = repair_row(str(raw), row.to_dict())
        changed = False
        for key, val in repaired.items():
            old = row.get(key)
            if key == "solver_label":
                differs = old != val
            elif old is None:
                differs = True
            else:
                differs = list(old) != list(val)
            if differs:
                if key not in df.columns:
                    df[key] = pd.Series([None] * len(df), dtype=object)
                df.at[i, key] = val
                changed = True
        if changed:
            fixed += 1

    df.to_parquet(out_path, index=False)
    print(f"Repaired {fixed}/{len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
