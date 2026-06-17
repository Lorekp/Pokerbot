#!/usr/bin/env python3
"""Before/after proof for MC sampler fix (Stage 1 follow-up)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_cfr.buckets import LABELS_169
from aof_cfr.terminal_ev import _exact_nway_shares_numba
from aof_cfr.trio_eval_mc import build_live_deck, mc_nway_shares
from aof_cfr.trio_payoff_tensor import (
    _all_combo_triples,
    _avg_shares_for_triple_legacy_per_combo,
    exact_shares_for_triple_oracle,
)
from aof_equity import warmup
from aof_equity_grids import _card_str_to_id


def bucket_indices(a: str, b: str, c: str) -> tuple[int, int, int]:
    idx = {lab: i for i, lab in enumerate(LABELS_169)}
    return idx[a], idx[b], idx[c]


def triple_share_error(i: int, j: int, k: int, *, mc_boards: int) -> dict:
    s0, s1, s2, _, _, _, _ = _avg_shares_for_triple_legacy_per_combo(i, j, k, mc_boards=mc_boards)
    o0, o1, o2, _, _, _, _ = exact_shares_for_triple_oracle(i, j, k)
    mc = np.array([s0, s1, s2], dtype=np.float64)
    exact = np.array([o0, o1, o2], dtype=np.float64)
    return {
        "mc": mc.tolist(),
        "exact": exact.tolist(),
        "max_error": float(np.max(np.abs(mc - exact))),
    }


def combo_mc_vs_exact(
    i: int, j: int, k: int, *, mc_boards: int, seed: int
) -> tuple[float, float]:
    combos = _all_combo_triples(i, j, k)
    if not combos:
        return 0.0, 0.0
    combo = combos[0]
    flat = np.array([_card_str_to_id(c) for c in sum(combo, ())], dtype=np.int32)
    exact = _exact_nway_shares_numba(flat, 3)
    mc = mc_nway_shares(flat, 3, mc_boards, seed)
    return float(np.max(np.abs(mc - exact))), float(np.std(mc - exact))


def four_max_collision_spot(*, mc_boards: int, seed: int = 0) -> dict:
    cards = ["2c", "2d", "2h", "2s", "3c", "3d", "4c", "5d"]
    ids = [_card_str_to_id(c) for c in cards]
    flat = np.array(ids, dtype=np.int32)
    deck, nd = build_live_deck(flat, 4)
    assert nd == 44, nd
    exact = _exact_nway_shares_numba(flat, 4)
    mc = mc_nway_shares(flat, 4, mc_boards, seed)
    return {
        "cards": cards,
        "deck_size": nd,
        "exact": exact.tolist(),
        "mc": mc.tolist(),
        "max_error": float(np.max(np.abs(mc - exact))),
    }


def deck_probe_22_22_33() -> dict:
    combos = _all_combo_triples(0, 0, 1)
    combo = combos[0]
    flat = np.array([_card_str_to_id(c) for c in sum(combo, ())], dtype=np.int32)
    deck, nd = build_live_deck(flat, 3)
    return {
        "combo": combo,
        "hole_ids": flat.tolist(),
        "deck_size": nd,
        "deck_ok": nd == 46,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Prove MC sampler fix")
    p.add_argument("--mc-boards", type=int, default=8192)
    p.add_argument("--out", default="data/aof/cfr_base/mc_sampler_fix_proof.json")
    args = p.parse_args()

    warmup()

    triples = [
        ("22", "22", "33"),
        ("22", "42o", "97s"),
        ("AKs", "AQs", "KQo"),
    ]
    report: dict = {
        "deck_probe_22_22_33": deck_probe_22_22_33(),
        "triples": [],
        "four_max_collision": {},
    }

    print("Deck probe 22/22/33:", report["deck_probe_22_22_33"])

    print(f"\n3-max triple shares (mc_boards={args.mc_boards} per combo):")
    for a, b, c in triples:
        i, j, k = bucket_indices(a, b, c)
        row = {"label": f"{a}/{b}/{c}", **triple_share_error(i, j, k, mc_boards=args.mc_boards)}
        report["triples"].append(row)
        print(
            f"  {row['label']} exact={row['exact']} mc={row['mc']} "
            f"max_err={row['max_error']:.6f}"
        )

    spot = four_max_collision_spot(mc_boards=args.mc_boards)
    report["four_max_collision"] = spot
    print(f"\n4-max collision {spot['cards']}")
    print(f"  exact={spot['exact']} mc={spot['mc']} max_err={spot['max_error']:.6f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
