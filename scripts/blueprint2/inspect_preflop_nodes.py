#!/usr/bin/env python3
"""Inspect the trained PREFLOP strategy at canonical nodes directly from a checkpoint.

Uses the exact fingerprint message construction (b"w" + weva + b"S" + betting) to look up
RFI and vs-open nodes for every position and reports jam/raise/fold/call frequencies,
range-aggregated over all 169 preflop classes. Pure stdlib.

  python scripts/blueprint2/inspect_preflop_nodes.py \
      --checkpoint runs/native_cfr/bp_emd_k160_1.nlcfr
"""
from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from f32_regret_strategy_delta import avg_strategy, load_checkpoint  # noqa: E402

ACTION_NAMES = ["FOLD", "CHECK", "CALL", "B1/3p", "B1/2p", "B.66p",
                "B.75p", "B1.0p", "B1.5p", "B2.0p", "ALLIN"]
# pos_enum: UTG=0 HJ=1 CO=2 BTN=3 SB=4 BB=5 (fingerprint.rs POS_BY_OFFSET semantics)
POS = {"UTG": 0, "HJ": 1, "CO": 2, "BTN": 3, "SB": 4, "BB": 5}


def key_for(weva: int, betting: bytes) -> bytes:
    msg = b"w" + struct.pack("<i", weva) + b"S" + betting
    return hashlib.blake2b(msg, digest_size=16).digest()


def agg_node(strat, pos_enum: int, betting: bytes):
    """Aggregate avg-strategy over all 169 classes at (pos, betting-state). Returns
    (mean action probs over classes found, n_found)."""
    tot = [0.0] * 11
    n = 0
    for cls in range(169):
        weva = pos_enum * 169 + cls
        row = strat.get(key_for(weva, betting))
        if row is None:
            continue
        p = avg_strategy(row)
        for a in range(11):
            tot[a] += p[a]
        n += 1
    if n:
        tot = [x / n for x in tot]
    return tot, n


def show(label: str, probs, n):
    tops = sorted(range(11), key=lambda a: -probs[a])
    line = "  ".join(f"{ACTION_NAMES[a]} {100*probs[a]:.1f}%" for a in tops if probs[a] > 0.005)
    print(f"{label:34} ({n:3}/169 classes)  {line}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    ck = load_checkpoint(args.checkpoint)
    strat = ck["strat"]
    print(f"{args.checkpoint}: iters={ck['iters_done']} rows={len(strat)}\n")

    # betting[6] = [street, pot_b, tocall_b, spr_b, is_button, facing_agg]
    # RFI (unopened): pot = 1.5bb -> pot_b=0 (edges 3,6,12,25,50,100,200);
    # to_call = 1bb / 1.5bb = 0.667 -> tocall_b: edges (.01,.34,.67,1.01,2.01) -> 0.667 < .67 -> bucket 2
    # BB facing limps/unopened: to_call 0 -> bucket 0.
    # spr = 100 / 1.5 = 66 -> edges (1,2.5,5,10,20) -> bucket 5.
    print("=== RFI-ish nodes (unopened pot ~1.5bb, facing_agg=0) ===")
    for name, pe in POS.items():
        btn = 1 if name == "BTN" else 0
        toc = 0 if name == "BB" else 2
        b = bytes((0, 0, toc, 5, btn, 0))
        probs, n = agg_node(strat, pe, b)
        if n:
            show(f"RFI {name}", probs, n)

    # vs a ~2-2.5bb open: pot ~ 3.5-4bb -> pot_b=1 (3<=pot<6); to_call 2.5/4 ~ 0.6 -> tocall_b=2;
    # spr ~ 100/4 = 25 -> bucket 5; facing_agg=1.
    print("\n=== facing an open (pot 3-6bb, facing_agg=1) ===")
    for name, pe in POS.items():
        btn = 1 if name == "BTN" else 0
        b = bytes((0, 1, 2, 5, btn, 1))
        probs, n = agg_node(strat, pe, b)
        if n:
            show(f"vs-open {name}", probs, n)

    # vs 3bet/bigger pot: pot 6-12bb -> pot_b=2, tocall_b=3 (to_call/pot ~ .67-1), spr bucket 4-5.
    print("\n=== facing 3bet-ish (pot 6-12bb, tocall_b=3, facing_agg=1) ===")
    for name, pe in POS.items():
        btn = 1 if name == "BTN" else 0
        for spr_b in (4, 5):
            b = bytes((0, 2, 3, spr_b, btn, 1))
            probs, n = agg_node(strat, pe, b)
            if n:
                show(f"vs-3bet {name} spr_b={spr_b}", probs, n)

    # Big jam pots: pot 25-50 (pot_b=4) or 50-100 (pot_b=5), facing_agg=1, to_call huge -> tocall_b 4-5.
    print("\n=== facing jam-ish (pot_b 4-5, tocall_b 4-5, facing_agg=1) ===")
    for name, pe in POS.items():
        btn = 1 if name == "BTN" else 0
        for pot_b in (4, 5):
            for toc in (4, 5):
                b = bytes((0, pot_b, toc, 1, btn, 1))
                probs, n = agg_node(strat, pe, b)
                if n and n > 30:
                    show(f"vs-jam {name} pot_b={pot_b} toc={toc}", probs, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
