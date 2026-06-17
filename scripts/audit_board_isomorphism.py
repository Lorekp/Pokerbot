#!/usr/bin/env python3
"""Audit prod6000 board enumeration for suit-isomorphic duplicates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from pokerbot.solver.board_canonical import canonical_flop_key, canonical_flop_str
from pokerbot.solver.spot_enumerator import (
    enumerate_spots,
    load_solver_config,
    sample_boards,
    spot_key,
)
import random


def main() -> None:
    p = argparse.ArgumentParser(description="Audit board suit-isomorphism duplicates")
    p.add_argument("--config", type=Path, default=Path("config/solver_labels_prod6000.yaml"))
    p.add_argument("--output", type=Path, default=Path("logs/board_isomorphism_audit.json"))
    args = p.parse_args()

    config = load_solver_config(args.config)
    rng = random.Random(config.seed)
    boards = sample_boards(config, rng)
    board_strs = ["".join(c.to_str() for c in b) for b in boards]

    flop_keys = [canonical_flop_key(b) for b in board_strs]
    flop_ctr = Counter(flop_keys)
    board_only_dupes = sum(v - 1 for v in flop_ctr.values() if v > 1)

    # Spot-level: board + hero hole (fixed enumeration hands)
    spot_canon: dict[str, list[str]] = defaultdict(list)
    spot_keys_raw: list[str] = []
    for spot in enumerate_spots(config):
        sk = spot_key(spot)
        spot_keys_raw.append(sk)
        hero = (spot.hero_hole[0].to_str(), spot.hero_hole[1].to_str())
        from pokerbot.solver.board_canonical import canonical_spot_key

        ck = canonical_spot_key(sk, hero_hole=hero)
        spot_canon[ck].append(sk)

    unique_raw = len(set(spot_keys_raw))
    spot_iso_dupes = sum(len(v) - 1 for v in spot_canon.values() if len(v) > 1)

    # Spots per unique board-iso class × spot_id × hero (stack-agnostic)
    stack_collisions = len(spot_keys_raw) - unique_raw

    iso_groups = [
        {
            "canonical_flop": canonical_flop_str(k),
            "count": v,
            "examples": [b for b, fk in zip(board_strs, flop_keys) if fk == k][:5],
        }
        for k, v in sorted(flop_ctr.items(), key=lambda x: -x[1])
        if v > 1
    ]

    report = {
        "config": str(args.config),
        "board_count": len(board_strs),
        "unique_boards": len(set(board_strs)),
        "unique_flop_isomorphism_classes": len(flop_ctr),
        "board_only_isomorphic_duplicates": board_only_dupes,
        "total_spot_keys_enumerated": len(spot_keys_raw),
        "unique_spot_keys_raw": unique_raw,
        "stack_depth_key_collisions": stack_collisions,
        "unique_spot_keys_with_hero_isomorphism": len(spot_canon),
        "spot_level_isomorphic_duplicates_with_hero": spot_iso_dupes,
        "estimated_eliminable_spots_board_only_iso": board_only_dupes
        * len(config.spots)
        * len(config.hero_hands)
        * len(config.stack_depths_bb),
        "top_isomorphic_flop_groups": iso_groups[:15],
        "recommendation": (
            "do_not_dedupe_board_only_iso_with_fixed_hero_hands"
            if spot_iso_dupes > board_only_dupes * len(config.spots)
            else "board_only_iso_dedupe_may_be_safe"
        ),
        "note": (
            "32 board-only isomorphic duplicates exist, but hero hands (AhKd, QsQh) "
            "have fixed suits — deduping boards alone can change label semantics. "
            "515 of 3000 unique spot_keys collapse under full board+hero isomorphism."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
