#!/usr/bin/env python3
"""Exact per-street decomposition of an NLCFR001 blueprint checkpoint.

The trainer keys infosets with blake2b-16 over a SMALL structured message space
(fingerprint.rs::fingerprint_from_bucket / fingerprint_fast_depth_postflop_bucket):

  preflop : b"w" + weva_id(i32 LE, 0..1013)          + b"S" + betting[6]
  postflop: b"B" + bucket(i32 LE, 0..num_buckets-1)  + b"S" + betting[6]
  betting[6] = [street&7, pot_bucket 0..7, tocall_bucket 0..5, spr_bucket 0..5,
                is_button 0..1, facing_agg 0..1]
  streets: 0=preflop 1=flop 2=turn 3=river (core.rs STREET_*)

That is ~1.17M preflop + ~num_buckets*4*1152 postflop candidate digests — cheap to
enumerate exhaustively, which INVERTS the hash: every checkpoint key is classified
exactly (street, bucket/weva, betting state). Reports per-street row counts and
strat_sum visit-mass (the training-coverage signal the 92.5%-preflop-termination
question needs).

Usage:
  python scripts/blueprint2/classify_checkpoint_streets.py \
      --checkpoint runs/native_cfr/bp_emd_k160_1.nlcfr --num-buckets 160 \
      [--out artifacts/bp2_street_decomp_emd.json]

Pure stdlib + the NLCFR001 parser in scripts/f32_regret_strategy_delta.py.
No pokerbot import (safe to run anywhere).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from f32_regret_strategy_delta import load_checkpoint  # noqa: E402

NUM_ACTIONS = 11
ACTION_NAMES = ["FOLD", "CHECK", "CALL", "BET_1/3p", "BET_1/2p", "BET_0.66p",
                "BET_0.75p", "BET_1.0p", "BET_1.5p", "BET_2.0p", "ALL_IN"]
STREET_NAMES = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}


def _digest(msg: bytes) -> bytes:
    return hashlib.blake2b(msg, digest_size=16).digest()


def build_inverse_table(num_buckets: int, depth_edges_len: int = 0) -> dict[bytes, tuple]:
    """Enumerate every possible fingerprint message -> classification tuple."""
    inv: dict[bytes, tuple] = {}
    t0 = time.time()
    # Betting-state space (shared tail).
    betting = []
    for street in (0, 1, 2, 3):
        for pot_b in range(8):
            for tocall_b in range(6):
                for spr_b in range(6):
                    for btn in (0, 1):
                        for fac in (0, 1):
                            betting.append((street, bytes((street, pot_b, tocall_b, spr_b, btn, fac))))
    # Preflop b"w" branch (street byte 0 only; weva 0..1013 = 6 pos * 169 classes).
    n_pre = 0
    pre_betting = [b for b in betting if b[0] == 0]
    for weva in range(6 * 169):
        head = b"w" + struct.pack("<i", weva)
        for _street, bs in pre_betting:
            inv[_digest(head + b"S" + bs)] = ("preflop", weva, bs)
            n_pre += 1
    # Postflop b"B" branch (any street byte -- river rows exist via made-hand mirror bucket).
    n_post = 0
    for bucket in range(num_buckets):
        head = b"B" + struct.pack("<i", bucket)
        for street, bs in betting:
            if street == 0:
                continue  # bucket >= 0 never occurs preflop (board < 3 cards)
            inv[_digest(head + b"S" + bs)] = (STREET_NAMES[street], bucket, bs)
            n_post += 1
    print(f"[inv] enumerated {n_pre} preflop + {n_post} postflop candidate keys "
          f"in {time.time()-t0:.1f}s", flush=True)
    return inv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-buckets", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ck = load_checkpoint(args.checkpoint)
    strat = ck["strat"]
    cum = ck["cum"]
    print(f"[ck] {args.checkpoint}: iters={ck['iters_done']} strat_rows={len(strat)} "
          f"cum_rows={len(cum)}", flush=True)

    inv = build_inverse_table(args.num_buckets)

    per_street = {}
    unclassified = 0
    unclassified_examples = []
    for key, row in strat.items():
        cls = inv.get(key)
        if cls is None:
            unclassified += 1
            if len(unclassified_examples) < 5:
                unclassified_examples.append(key.hex())
            continue
        street = cls[0]
        d = per_street.setdefault(street, {
            "rows": 0, "strat_mass": 0.0, "action_mass": [0.0] * NUM_ACTIONS,
            "distinct_buckets": set(), "facing_agg_rows": 0,
        })
        d["rows"] += 1
        m = sum(row)
        d["strat_mass"] += m
        for a in range(NUM_ACTIONS):
            d["action_mass"][a] += row[a]
        d["distinct_buckets"].add(cls[1])
        if cls[2][5] == 1:
            d["facing_agg_rows"] += 1

    total_rows = len(strat)
    total_mass = sum(d["strat_mass"] for d in per_street.values())
    print(f"\n=== per-street decomposition ({args.checkpoint}) ===")
    print(f"{'street':8} {'rows':>8} {'rows%':>7} {'visit-mass%':>12} {'buckets':>8} {'facing%':>8}")
    out = {"checkpoint": args.checkpoint, "iters": ck["iters_done"],
           "num_buckets": args.num_buckets, "total_rows": total_rows,
           "unclassified": unclassified, "per_street": {}}
    for street in ("preflop", "flop", "turn", "river"):
        d = per_street.get(street)
        if d is None:
            continue
        rows_pct = 100.0 * d["rows"] / max(1, total_rows)
        mass_pct = 100.0 * d["strat_mass"] / max(1e-12, total_mass)
        print(f"{street:8} {d['rows']:>8} {rows_pct:>6.2f}% {mass_pct:>11.2f}% "
              f"{len(d['distinct_buckets']):>8} "
              f"{100.0*d['facing_agg_rows']/max(1,d['rows']):>7.1f}%")
        am = d["action_mass"]
        s = sum(am)
        top = sorted(range(NUM_ACTIONS), key=lambda a: -am[a])[:4]
        print("         top actions by visit-mass: " +
              ", ".join(f"{ACTION_NAMES[a]} {100.0*am[a]/max(1e-12,s):.1f}%" for a in top))
        out["per_street"][street] = {
            "rows": d["rows"], "rows_pct": rows_pct, "visit_mass_pct": mass_pct,
            "distinct_buckets": len(d["distinct_buckets"]),
            "facing_agg_rows": d["facing_agg_rows"],
            "action_mass_pct": [100.0 * a / max(1e-12, s) for a in am],
        }
    if unclassified:
        print(f"\nWARNING: {unclassified} keys unclassified (unexpected message space?): "
              f"{unclassified_examples}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
