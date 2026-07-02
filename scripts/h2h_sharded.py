"""Shard a deck-paired H2H run across worker processes (CRN-exact).

Both engine_ab_h2h.py and external_ladder_h2h.py derive every per-deck RNG purely
from (base_seed, deck_index), so splitting the deck range across processes and
concatenating the per-deck nets is bit-identical to one sequential run.

Usage:
  python scripts/h2h_sharded.py --shards 4 --decks 300 --out artifacts/merged.json \
      -- scripts/engine_ab_h2h.py --arm-a '{}' --arm-b '{"iterations":3000}' --seed 7331 --label K3000

The wrapped command must NOT contain --decks/--deck-start/--out (the driver owns them).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time


def main() -> None:
    p = argparse.ArgumentParser(description="CRN-exact sharded H2H driver")
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--rayon-per-shard", type=int, default=3,
                   help="RAYON_NUM_THREADS per shard (0 = inherit; NEVER 1 — nested-rayon deadlock)")
    p.add_argument("--deck-offset", type=int, default=0,
                   help="Global deck-index offset (extend a prior run: offset=N covers decks [N, N+decks))")
    p.add_argument("--decks", type=int, required=True, help="TOTAL decks across all shards")
    p.add_argument("--out", required=True, help="Merged output JSON")
    p.add_argument("cmd", nargs=argparse.REMAINDER,
                   help="-- <script.py> <args...> (without --decks/--deck-start/--out)")
    args = p.parse_args()

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        p.error("missing wrapped command after --")
    for forbidden in ("--decks", "--deck-start", "--out"):
        if forbidden in cmd:
            p.error(f"{forbidden} belongs to the driver, remove it from the wrapped command")

    n, k = args.decks, args.shards
    base_out = os.path.splitext(args.out)[0]
    procs, shard_outs = [], []
    t0 = time.perf_counter()
    env = dict(os.environ)
    if args.rayon_per_shard > 0:
        env["RAYON_NUM_THREADS"] = str(args.rayon_per_shard)
    for i in range(k):
        # Round-robin (strided) deck assignment: shard i gets decks i, i+k, i+2k, ...
        # so expensive flop-decks distribute evenly across shards.
        cnt = (n - i + k - 1) // k
        if cnt <= 0:
            break
        so = f"{base_out}.shard{i}.json"
        shard_outs.append(so)
        log = open(f"{base_out}.shard{i}.log", "w")
        procs.append((subprocess.Popen(
            [sys.executable, "-u"] + cmd
            + ["--deck-start", str(args.deck_offset + i), "--deck-stride", str(k),
               "--decks", str(cnt), "--out", so],
            stdout=log, stderr=subprocess.STDOUT, env=env), log, so))
        print(f"shard {i}: decks {i}, {i + k}, ... ({cnt} decks)  -> {so}", flush=True)

    # Heartbeat: poll shard checkpoints for combined progress.
    done = [False] * len(procs)
    while not all(done):
        time.sleep(20)
        total_done = 0
        for i, (pr, _, so) in enumerate(procs):
            if pr.poll() is not None:
                done[i] = True
            try:
                total_done += len(json.load(open(so)).get("per_deck_nets", []))
            except Exception:
                pass
        el = time.perf_counter() - t0
        eta = el / max(total_done, 1) * (n - total_done) / 60
        print(f"  progress {total_done}/{n} decks | {el/60:.1f}min ETA {eta:.1f}min", flush=True)
    for pr, log, _ in procs:
        pr.wait()
        log.close()
    fails = [so for (pr, _, so), _d in zip(procs, done) if pr.returncode != 0]
    if fails:
        raise SystemExit(f"shard(s) FAILED (rc!=0): {fails} — see .log files")

    # Merge: shards are strided (shard i holds decks i, i+k, ...), so interleave
    # round-robin to restore global deck order.
    shard_lists, flopdecks = [], 0
    first = json.load(open(shard_outs[0]))
    for so in shard_outs:
        d = json.load(open(so))
        shard_lists.append(list(d["per_deck_nets"]))
        flopdecks += int(d.get("flopdecks", 0))
    per_deck = []
    for j in range(max(len(x) for x in shard_lists)):
        for sl in shard_lists:
            if j < len(sl):
                per_deck.append(sl[j])
    m = sum(per_deck) / len(per_deck)
    se = (sum((x - m) ** 2 for x in per_deck) / (len(per_deck) - 1)) ** 0.5 / math.sqrt(len(per_deck))
    merged = dict(first)
    merged.update({
        "decks": len(per_deck),
        "per_deck_nets": per_deck,
        "flopdecks": flopdecks,
        "ci95": [m - 1.96 * se, m + 1.96 * se],
        "wall_s": time.perf_counter() - t0,
        "shards": len(shard_outs),
    })
    for key in ("a_minus_b_per_deck", "ours_minus_them_per_deck"):
        if key in merged:
            merged[key] = m
    json.dump(merged, open(args.out, "w"), indent=1)
    print(f"MERGED {len(per_deck)} decks: mean/deck {m:+.2f}  CI95 [{m-1.96*se:+.1f}, {m+1.96*se:+.1f}]  "
          f"flopdecks {flopdecks}  wall {(time.perf_counter()-t0)/60:.1f}min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
