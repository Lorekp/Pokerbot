#!/usr/bin/env python3
"""Precompute the SHARED train/serve FROZEN ``(hole,board)->EMD-bucket`` lookup for FLOP+TURN.

The deployed blueprint buckets postflop by MADE-HAND RANK only (draw-blind). The draw-aware EMD bucket
(``emd_buckets_*.json`` centroids) fixes that, but computing it per-node in the CFR+ hot loop is a
non-starter (a ~200-runout Monte Carlo per spot, AND the native xoshiro MC diverges from numpy's PCG64
so a per-node recompute would NOT match the serve encoder — a silent train/serve key skew).

THE FIX (speed + correctness): precompute ONCE, offline, the EMD bucket for every canonical (hole,board)
on FLOP+TURN, freeze it into a dense table, and have BOTH the trainer key and the serve encoder read that
SAME blob. The hot path collapses to a suit-canonicalize + one hash probe (≈ the made-hand cost — in fact
faster, since it skips the phevaluator eval on a hit), and the buckets are bit-identical train/serve.

RIVER stays made-hand: at the river the board is complete, there are no draws, so the made-hand rank is
already EXACT. The frozen table covers flop+turn only (~6.5M canonical spots, ~tens of MB as u16).

This is HEAVY (millions of canonical spots × a ~200-runout MC, rayon-parallel) — run it on a CLOUD spot
box (e.g. c6a.32xlarge), NOT the user's PC. It uses the native ``battery_native.emd_frozen_build`` kernel.

    python scripts/precompute_emd_frozen_lookup.py \
        --centroids data/abstraction/emd_buckets_200.json \
        --out data/abstraction/emd_frozen_flop_turn_200.emdfrzn \
        --n-runouts 200

Then point BOTH sides at the result:
    train: pass emd_frozen_path=<out> to NlheCFRTrainer (the launcher --emd-frozen flag)
    serve: POKERBOT_POSTFLOP_MODE=emd POKERBOT_POSTFLOP_EMD_FROZEN=<out>
The blob is content-hash-stamped; ship_blueprint.py's serve-skew gate asserts train-table == serve-table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--centroids", default="data/abstraction/emd_buckets_200.json",
                    help="EMDBucketer.save() centroids JSON (the K×n_bins clusters)")
    ap.add_argument("--out", default="data/abstraction/emd_frozen_flop_turn_200.emdfrzn",
                    help="output frozen-lookup binary (EMDFRZN1)")
    ap.add_argument("--n-bins", type=int, default=0,
                    help="override the centroids' n_bins (0 = use the file's)")
    ap.add_argument("--n-runouts", type=int, default=0,
                    help="board completions per spot (0 = use the centroids' n_runouts)")
    a = ap.parse_args()

    d = json.loads(Path(a.centroids).read_text(encoding="utf-8"))
    centroids = d["centroids"]
    if centroids is None:
        raise SystemExit(f"centroids file has no fitted clusters: {a.centroids}")
    n_bins = a.n_bins or int(d.get("n_bins", 20))
    n_runouts = a.n_runouts or int(d.get("n_runouts", 200))
    k = len(centroids)
    print(f"centroids: K={k} n_bins={n_bins} n_runouts={n_runouts} from {a.centroids}", flush=True)

    # The native kernel: enumerates all canonical flop+turn (hole,board), buckets each, writes the blob.
    try:
        import battery_native as bn
    except Exception as e:  # noqa: BLE001
        # Fall back to the native_solver loader (aliases the cdylib -> .pyd and puts it on sys.path).
        from pokerbot.search import native_solver  # noqa: PLC0415
        bn = native_solver._try_load()
        if bn is None:
            print(f"could not import battery_native: {e!r}\n"
                  "build it first: (cd tools/battery_native && "
                  "cargo build --release --features python)", file=sys.stderr)
            raise SystemExit(1) from e

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("building frozen flop+turn lookup (rayon-parallel; minutes on a many-core box)…", flush=True)
    count, stamp = bn.emd_frozen_build(centroids, a.out, n_bins, n_runouts)
    dt = time.time() - t0
    size = Path(a.out).stat().st_size
    print(f"wrote {a.out}  ({count} canonical spots, {size/1e6:.1f} MB, stamp {stamp})  in {dt:.0f}s",
          flush=True)

    # SHA256 content stamp + sidecar. The Rust stamp is blake2b16 (fingerprint, fast); this is
    # sha256 (cryptographic, single-shot) of the WHOLE blob, written to a sidecar file. ship_blueprint.py
    # cross-checks the SHA256 of the blob the serve side loads against this sidecar BEFORE activating.
    blob_bytes = Path(a.out).read_bytes()
    sha256 = hashlib.sha256(blob_bytes).hexdigest()
    sidecar = Path(str(a.out) + ".sha256")
    sidecar.write_text(sha256 + "\n", encoding="ascii")
    print(f"wrote sidecar -> {sidecar}  sha256={sha256}  blake2b16={stamp}", flush=True)

    print("Set for train: emd_frozen_path=" + a.out, flush=True)
    print(f"Set for serve: POKERBOT_POSTFLOP_MODE=emd POKERBOT_POSTFLOP_EMD_FROZEN={a.out}", flush=True)


if __name__ == "__main__":
    main()
