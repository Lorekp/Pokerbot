#!/usr/bin/env python3
"""Build a SORTED KEY INDEX (+ optional aligned row store) from an NLCFR001 checkpoint's strat
section.

Purpose (serve-coverage root-cause work, 2026-07-11): classifying a serve-time blueprint lookup
MISS requires asking "is this key in the FULL trained table?" without loading the 28 GB checkpoint
into a dict. The index is the strat section's 16-byte digests as a lexicographically SORTED
``np.dtype('S16')`` array (n*16 bytes on disk); membership is ``np.searchsorted`` (O(log n), mmap-
friendly). A key present here but absent from a mass-filtered serve pkl is a TRUNCATION miss
(dropped row); a key absent here was never stored by the trainer (unvisited infoset or key-space
divergence).

With ``--with-rows`` the strat rows are ALSO written, f32, permuted into the same sorted order
(``rows_sorted.f32.bin``, shape (n, 11)) — a memory-mappable FULL-TABLE serve object
(``SortedSnapshot`` in scripts/serve_miss_probe.py): O(log n) binary-search lookup, ~0 RSS.
This is the practical alternative to the top-25M dict extract whose row cap is the measured
self-play miss driver (mass_kept=90.9% at 25M rows for the 50k finemenu checkpoint).

Usage:
  python scripts/nlcfr_key_index.py IN.nlcfr OUT_DIR [--with-rows]
  # writes OUT_DIR/keys_sorted.bin [+ rows_sorted.f32.bin] + meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import time

import numpy as np

NUM_ACTIONS = 11
ROW = 16 + NUM_ACTIONS * 8
MAGIC = b"NLCFR001"
CHUNK_ROWS = 1 << 20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("out_dir")
    ap.add_argument("--with-rows", action="store_true",
                    help="also write rows_sorted.f32.bin (aligned with keys_sorted.bin)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    f = open(args.input, "rb", buffering=1 << 22)
    if f.read(8) != MAGIC:
        raise ValueError(f"{args.input}: bad magic")
    (iters,) = struct.unpack("<Q", f.read(8))
    (n_cum,) = struct.unpack("<Q", f.read(8))
    f.seek(n_cum * ROW, os.SEEK_CUR)  # stream-skip the regret section
    (n_strat,) = struct.unpack("<Q", f.read(8))
    print(f"[keyidx] iters={iters} cum_rows={n_cum} strat_rows={n_strat}", flush=True)

    # RAM-bounded two-pass layout (a 241M-row merged table would not fit keys+vals+order at once):
    # pass 1 reads keys only, argsorts, writes keys_sorted, FREES the key array; pass 2 re-reads the
    # rows as f32 and writes them permuted in sorted order (only vals + order held).
    strat_off = f.tell()
    keys = np.empty(n_strat, dtype="S16")
    done = 0
    while done < n_strat:
        take = min(CHUNK_ROWS, n_strat - done)
        buf = f.read(take * ROW)
        if len(buf) < take * ROW:
            raise EOFError(f"short read at row {done}")
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(take, ROW)
        keys[done:done + take] = arr[:, :16].copy().view("S16").reshape(take)
        done += take
        if done % (CHUNK_ROWS * 16) < CHUNK_ROWS:
            print(f"  [scan keys] {done}/{n_strat} ({100.0 * done / n_strat:.1f}%) "
                  f"t={time.time() - t0:.0f}s", flush=True)
    print(f"[keyidx] keys read t={time.time() - t0:.0f}s; sorting...", flush=True)
    order = np.argsort(keys, kind="stable")
    with open(os.path.join(args.out_dir, "keys_sorted.bin"), "wb") as kf:
        step = 16_000_000
        for lo in range(0, n_strat, step):
            kf.write(keys[order[lo:lo + step]].tobytes())
    del keys
    print(f"[keyidx] keys sorted+written t={time.time() - t0:.0f}s", flush=True)

    if args.with_rows:
        f.seek(strat_off)
        vals = np.empty((n_strat, NUM_ACTIONS), dtype=np.float32)
        done = 0
        while done < n_strat:
            take = min(CHUNK_ROWS, n_strat - done)
            buf = f.read(take * ROW)
            if len(buf) < take * ROW:
                raise EOFError(f"short read at row {done}")
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(take, ROW)
            vals[done:done + take] = arr[:, 16:].copy().view("<f8").reshape(
                take, NUM_ACTIONS).astype(np.float32)
            done += take
            if done % (CHUNK_ROWS * 16) < CHUNK_ROWS:
                print(f"  [scan rows] {done}/{n_strat} ({100.0 * done / n_strat:.1f}%) "
                      f"t={time.time() - t0:.0f}s", flush=True)
        out_rows = os.path.join(args.out_dir, "rows_sorted.f32.bin")
        with open(out_rows, "wb") as rf:
            step = 8_000_000
            for lo in range(0, n_strat, step):
                rf.write(vals[order[lo:lo + step]].tobytes())
        print(f"[keyidx] rows written ({os.path.getsize(out_rows):,} bytes) "
              f"t={time.time() - t0:.0f}s", flush=True)
    f.close()
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as m:
        json.dump({"source": os.path.abspath(args.input), "iters": int(iters),
                   "n_strat": int(n_strat), "num_actions": NUM_ACTIONS,
                   "format": "keys: S16 sorted ascending; rows: f32 (n,11) aligned",
                   "with_rows": bool(vals is not None),
                   "built_at": time.strftime("%F %T")}, m, indent=1)
    print(f"[keyidx] done t={time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
