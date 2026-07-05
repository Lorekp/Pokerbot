#!/usr/bin/env python3
"""Streaming sorted key->offset index over an NLSNAP01 blob (v3 coverage diagnostic).

NLSNAP01 format: magic(8) + iters_done(u64) + n_rows(u64) + [16-byte key + 11*f32 probs] * n_rows
Row size = 60 bytes (vs NLCFR001 strat rows at 104 bytes).

This is the snap-file equivalent of bpv4_nlcfr_index.py — same A/B/O index, same binary-search
lookup contract, so _IndexedSnapFile (below) can be plugged in place of _IndexedSnap in the
coverage probe.

Usage:
  python scripts/blueprint2/bpv4_nlsnap_index.py --blob artifacts/bpv4_tmp/snap_020000.nlsnap \
      --out-dir artifacts/bpv4_tmp/idx_v3_snap20k [--check 2000]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
import time

import numpy as np

NUM_ACTIONS = 11
ROW = 16 + NUM_ACTIONS * 4  # 60 bytes: 16-byte key + 11 f32 probs
CHUNK_ROWS = 1_000_000

# strided view over raw records: two big-endian u64s at key bytes 0-7, 8-15; rest skipped
DT_SCAN = np.dtype({"names": ["a", "b"], "formats": [">u8", ">u8"],
                    "offsets": [0, 8], "itemsize": ROW})
DT_REC = np.dtype([("a", "<u8"), ("b", "<u8"), ("o", "<u8")])


def _plog(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build(blob: str, out_dir: str, check_n: int) -> int:
    os.makedirs(out_dir, exist_ok=True)
    size = os.path.getsize(blob)
    f = open(blob, "rb")
    hdr = f.read(24)
    assert hdr[:8] == b"NLSNAP01", f"{blob}: bad magic (got {hdr[:8]!r})"
    (iters,) = struct.unpack("<Q", hdr[8:16])
    (n_rows,) = struct.unpack("<Q", hdr[16:24])
    data_base = 24
    expected_size = data_base + n_rows * ROW
    assert size == expected_size, f"size mismatch: got {size} want {expected_size} (n_rows={n_rows:,})"
    _plog(f"NLSNAP01 iters={iters} n_rows={n_rows:,} row_bytes={ROW} data@{data_base}")

    # pass 1: sequential scan -> 256 bucket files by top key byte
    tmp = os.path.join(out_dir, "_buckets")
    os.makedirs(tmp, exist_ok=True)
    buckets = [open(os.path.join(tmp, f"b{i:03d}.bin"), "wb") for i in range(256)]
    done = 0
    t0 = time.time()
    last_hb = t0
    while done < n_rows:
        n = min(CHUNK_ROWS, n_rows - done)
        raw = f.read(n * ROW)
        assert len(raw) == n * ROW, "short read"
        keys = np.frombuffer(raw, dtype=DT_SCAN, count=n)
        rec = np.empty(n, dtype=DT_REC)
        rec["a"] = keys["a"]
        rec["b"] = keys["b"]
        rec["o"] = data_base + (done + np.arange(n, dtype=np.uint64)) * ROW
        top = (rec["a"] >> np.uint64(56)).astype(np.uint16)
        order = np.argsort(top, kind="stable")
        rec = rec[order]
        top = top[order]
        bounds = np.searchsorted(top, np.arange(257, dtype=np.uint16))
        for i in range(256):
            lo, hi = int(bounds[i]), int(bounds[i + 1])
            if hi > lo:
                buckets[i].write(rec[lo:hi].tobytes())
        done += n
        now = time.time()
        if now - last_hb > 30:
            last_hb = now
            pct = 100.0 * done / n_rows
            eta = (now - t0) / done * (n_rows - done)
            _plog(f"pass1 {done:,}/{n_rows:,} ({pct:.1f}%) eta {eta/60:.1f}m")
    for b in buckets:
        b.close()
    f.close()
    _plog(f"pass1 done in {(time.time()-t0)/60:.1f}m")

    # pass 2: per-bucket in-RAM sort -> final sorted arrays
    t1 = time.time()
    fa = open(os.path.join(out_dir, "A.bin"), "wb")
    fb = open(os.path.join(out_dir, "B.bin"), "wb")
    fo = open(os.path.join(out_dir, "O.bin"), "wb")
    total = 0
    for i in range(256):
        p = os.path.join(tmp, f"b{i:03d}.bin")
        raw = open(p, "rb").read()
        arr = np.frombuffer(raw, dtype=DT_REC).copy()
        arr.sort(order=("a", "b"))
        fa.write(np.ascontiguousarray(arr["a"]).tobytes())
        fb.write(np.ascontiguousarray(arr["b"]).tobytes())
        fo.write(np.ascontiguousarray(arr["o"]).tobytes())
        total += len(arr)
        os.remove(p)
        if i % 64 == 63:
            _plog(f"pass2 bucket {i+1}/256 ({total:,} rows)")
    fa.close(); fb.close(); fo.close()
    os.rmdir(tmp)
    assert total == n_rows, f"pass2 total {total} != n_rows {n_rows}"
    _plog(f"pass2 done in {(time.time()-t1)/60:.1f}m")

    meta = {"n_rows": int(n_rows), "iters": int(iters), "blob": os.path.abspath(blob),
            "blob_bytes": size, "row_bytes": ROW, "format": "NLSNAP01",
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    # self-check: random records round-trip
    A = np.memmap(os.path.join(out_dir, "A.bin"), dtype="<u8", mode="r")
    B = np.memmap(os.path.join(out_dir, "B.bin"), dtype="<u8", mode="r")
    O = np.memmap(os.path.join(out_dir, "O.bin"), dtype="<u8", mode="r")
    rng = random.Random(13)
    blob_f = open(blob, "rb")
    bad = 0
    for _ in range(check_n):
        j = rng.randrange(n_rows)
        off = data_base + j * ROW
        blob_f.seek(off)
        key = blob_f.read(16)
        a = np.uint64(int.from_bytes(key[:8], "big"))
        b = np.uint64(int.from_bytes(key[8:16], "big"))
        lo = int(np.searchsorted(A, a, "left"))
        hi = int(np.searchsorted(A, a, "right"))
        k = lo + int(np.searchsorted(B[lo:hi], b, "left"))
        if not (k < hi and int(B[k]) == b and int(O[k]) == off):
            bad += 1
    blob_f.close()
    if bad:
        _plog(f"SELF-CHECK FAILED: {bad}/{check_n} records did not round-trip — deleting meta.json")
        os.remove(os.path.join(out_dir, "meta.json"))
        return 1
    _plog(f"self-check OK ({check_n} random records round-trip); index -> {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--check", type=int, default=2000)
    args = ap.parse_args()
    return build(args.blob, args.out_dir, args.check)


if __name__ == "__main__":
    sys.exit(main())
