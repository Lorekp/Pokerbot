#!/usr/bin/env python3
"""STREAMING NLCFR001 -> serve-pkl converter with a row-mass filter (bounded RAM).

Purpose: a fullhist+presum blueprint checkpoint has ~150M strat rows (~28 GB f64 on disk) — far
beyond any practical serve table (the dict->PyDict->HashMap serve path would need >80 GB RSS).
The practical SHIP OBJECT is the mass-filtered f32 snapshot this script emits: rows are kept by
descending strat_sum mass (linear-CFR reach-weighted) until BOTH the target mass coverage and the
row cap are satisfied. Dropped rows were (a) rarely reached in training, so they are rarely hit at
serve, and (b) noise-dominated averages anyway; a dropped-row lookup falls back exactly like an
untrained infoset (uniform prior — measured by the h2h_arm_worker miss counter, never guessed).

Layout (little-endian): magic "NLCFR001" (8) | iters u64 | cum: n u64 + n*[key16|11*f64] |
strat: n u64 + n*[key16|11*f64].  The cum (regret) section is STREAM-SKIPPED — averaged serve
(serve_final_iterate=False, the deployed default) reads only strat.

Output pkl mirrors scripts/ship_blueprint.py install(): {"strat_sum": {key16: np.float32[11]
row-normalized}, "abstraction": <stamp dict>, "filter": <this filter's provenance>}.

Usage:
  python scripts/nlcfr_stream_filter.py IN.nlcfr OUT.pkl \
      --mass 0.999 --max-rows 25000000 [--curve-only] [--stamp-json '{...}']
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import struct
import sys
import time

import numpy as np

NUM_ACTIONS = 11
ROW = 16 + NUM_ACTIONS * 8
MAGIC = b"NLCFR001"
CHUNK_ROWS = 262144  # 27 MB read chunks


def _open_at_strat(path):
    f = open(path, "rb", buffering=1 << 22)
    if f.read(8) != MAGIC:
        raise ValueError(f"{path}: bad magic")
    (iters,) = struct.unpack("<Q", f.read(8))
    (n_cum,) = struct.unpack("<Q", f.read(8))
    f.seek(n_cum * ROW, os.SEEK_CUR)  # stream-skip the regret section
    (n_strat,) = struct.unpack("<Q", f.read(8))
    return f, iters, n_cum, n_strat


def _iter_chunks(f, n_rows):
    done = 0
    while done < n_rows:
        take = min(CHUNK_ROWS, n_rows - done)
        buf = f.read(take * ROW)
        if len(buf) < take * ROW:
            raise EOFError(f"short read at row {done}")
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(take, ROW)
        keys = arr[:, :16]
        vals = arr[:, 16:].copy().view("<f8").reshape(take, NUM_ACTIONS)
        yield keys, vals
        done += take


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--mass", type=float, default=0.999,
                    help="keep rows (by descending strat_sum mass) covering this fraction of total")
    ap.add_argument("--max-rows", type=int, default=25_000_000,
                    help="hard row cap (RAM bound); binds over --mass when smaller")
    ap.add_argument("--curve-only", action="store_true", help="print the mass curve and exit")
    ap.add_argument("--stamp-json", default="", help="abstraction stamp dict (JSON) embedded in pkl")
    ap.add_argument("--arrays", action="store_true",
                    help="write the ARRAYS format: {'keys': bytes(n*16), 'vals': f32[n,11], ...} — "
                         "vectorized pass 2 + O(seconds) save/load (150M-row scale); loaders rebuild "
                         "the dict at load time (engine_ab_h2h load_snapshot handles both formats)")
    args = ap.parse_args()

    t0 = time.time()
    f, iters, n_cum, n_strat = _open_at_strat(args.input)
    print(f"[filter] iters={iters} cum_rows={n_cum} strat_rows={n_strat} "
          f"(pass 1: mass scan)", flush=True)

    # PASS 1: per-row mass (f32; ~4B/row) + total.
    masses = np.empty(n_strat, dtype=np.float32)
    pos = 0
    for keys, vals in _iter_chunks(f, n_strat):
        s = vals.sum(axis=1)
        masses[pos:pos + len(s)] = s.astype(np.float32)
        pos += len(s)
        if pos % (CHUNK_ROWS * 32) < CHUNK_ROWS:
            print(f"  [pass1] {pos}/{n_strat} rows ({100.0*pos/n_strat:.1f}%) "
                  f"t={time.time()-t0:.0f}s", flush=True)
    f.close()
    total = float(masses.sum(dtype=np.float64))
    srt = np.sort(masses)[::-1].astype(np.float64)
    csum = np.cumsum(srt)
    print(f"[filter] total mass={total:.6e}  t={time.time()-t0:.0f}s", flush=True)
    for k in (1e5, 1e6, 5e6, 1e7, 2e7, 5e7, 1e8):
        k = int(k)
        if k <= n_strat:
            print(f"  top {k:>11,} rows -> {100.0*csum[k-1]/total:8.4f}% of mass", flush=True)

    n_mass = int(np.searchsorted(csum, args.mass * total) + 1)
    n_keep = min(n_mass, args.max_rows, n_strat)
    cutoff = float(srt[n_keep - 1])
    kept_mass = float(csum[n_keep - 1]) / total
    print(f"[filter] target mass={args.mass} -> {n_mass:,} rows; cap={args.max_rows:,} "
          f"=> KEEP {n_keep:,} rows (cutoff={cutoff:.6e}, kept mass={100*kept_mass:.4f}%)",
          flush=True)
    if args.curve_only:
        return 0
    if not args.output:
        print("FATAL: output path required unless --curve-only", flush=True)
        return 2

    # PASS 2: keep rows >= cutoff (ties may add a few rows past n_keep; harmless). Vectorized:
    # chunk-select, chunk-normalize, append to contiguous buffers (no per-row Python).
    del srt, csum
    f, _, _, _ = _open_at_strat(args.input)
    key_chunks = []
    val_chunks = []
    kept = 0
    pos = 0
    for keys, vals in _iter_chunks(f, n_strat):
        s = vals.sum(axis=1)
        sel = np.flatnonzero(s >= cutoff)
        if sel.size:
            kv = vals[sel]
            tot = kv.sum(axis=1, keepdims=True)
            uniform = np.full((1, NUM_ACTIONS), 1.0 / NUM_ACTIONS)
            norm = np.where(tot > 0, kv / np.where(tot > 0, tot, 1.0), uniform)
            key_chunks.append(keys[sel].copy())
            val_chunks.append(norm.astype(np.float32))
            kept += sel.size
        pos += len(s)
        if pos % (CHUNK_ROWS * 32) < CHUNK_ROWS:
            print(f"  [pass2] {pos}/{n_strat} rows, kept={kept:,} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    f.close()
    all_keys = np.concatenate(key_chunks) if key_chunks else np.zeros((0, 16), np.uint8)
    all_vals = np.concatenate(val_chunks) if val_chunks else np.zeros((0, NUM_ACTIONS), np.float32)
    del key_chunks, val_chunks
    print(f"[filter] kept {kept:,} rows; writing {args.output} "
          f"(format={'arrays' if args.arrays else 'dict'})", flush=True)
    if args.arrays:
        out = {"keys": all_keys.tobytes(), "vals": all_vals, "format": "arrays_v1"}
    else:
        out = {}
        for i in range(all_keys.shape[0]):
            out[all_keys[i].tobytes()] = all_vals[i]
        del all_keys, all_vals

    stamp = json.loads(args.stamp_json) if args.stamp_json else {}
    payload = {
        "strat_sum": out,  # dict {key16: f32[11]} OR arrays_v1 {"keys","vals"} per --arrays
        "abstraction": stamp,
        "filter": {
            "source": os.path.basename(args.input),
            "iters": iters,
            "strat_rows_total": n_strat,
            "rows_kept": int(kept),
            "mass_target": args.mass,
            "mass_kept": kept_mass,
            "cutoff": cutoff,
            "max_rows": args.max_rows,
        },
    }
    with open(args.output, "wb") as fo:
        pickle.dump(payload, fo, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[filter] DONE {args.output} ({os.path.getsize(args.output)/1e9:.2f} GB) "
          f"t={time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
