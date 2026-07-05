#!/usr/bin/env python3
"""Streaming sorted key->offset index over the STRAT section of an NLCFR001 blob.

WHY: blueprint-v4 Stage-0 G5 (design v2 §6.1/§7.1.4) must run the forced-response probe against the
CLOUD fullhist+M-RR checkpoint (bpv4_h200 @20k: 347.7M strat rows, 55.3 GB). The probe's historical
path (`f32_regret_strategy_delta.load_checkpoint` -> full in-RAM `{key: row}` dict) needs >100 GB of
RAM for that blob — impossible on the 31 GB local box. This script builds, in TWO streaming passes,
an on-disk sorted index the probe can mmap and binary-search per lookup (O(log n) page touches,
page-cache shared across shard processes):

  A.bin  n * u8 LE  — first 8 key bytes as a BIG-ENDIAN-interpreted uint64 (so numeric order ==
                      lexicographic byte order), sorted ascending (ties broken by B)
  B.bin  n * u8 LE  — last 8 key bytes, same interpretation, co-sorted
  O.bin  n * u8 LE  — byte offset of the RECORD START (the 16-byte key; row at +16) in the blob
  meta.json         — {n_strat, iters, blob, blob_bytes, built_at}

Pass 1 streams the strat section sequentially (never loads the blob) and partitions (a, b, offset)
records into 256 bucket files by the top key byte; pass 2 sorts each bucket in RAM (~1.4M records)
and appends to the final arrays — a bucketed radix + in-RAM sort, peak RAM ~a few hundred MB.

Self-check: after the build, N random strat records re-read from the blob must round-trip through
the index (lookup -> same offset -> same row bytes); any mismatch deletes meta.json and exits 1.

Usage:
  python scripts/blueprint2/bpv4_nlcfr_index.py --blob artifacts/bpv4_ladder/bpv4_h200_iter20000.nlcfr \
      --out-dir artifacts/bpv4_ladder/idx_h200_20k [--check 2000]
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
ROW = 16 + NUM_ACTIONS * 8  # 104
CHUNK_ROWS = 1_000_000

# strided view over raw records: two big-endian u64s at the key, row skipped
DT_SCAN = np.dtype({"names": ["a", "b"], "formats": [">u8", ">u8"],
                    "offsets": [0, 8], "itemsize": ROW})
DT_REC = np.dtype([("a", "<u8"), ("b", "<u8"), ("o", "<u8")])


def _plog(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build(blob: str, out_dir: str, check_n: int) -> int:
    os.makedirs(out_dir, exist_ok=True)
    size = os.path.getsize(blob)
    f = open(blob, "rb")
    assert f.read(8) == b"NLCFR001", f"{blob}: bad magic"
    (iters,) = struct.unpack("<Q", f.read(8))
    (n_cum,) = struct.unpack("<Q", f.read(8))
    strat_hdr = 24 + n_cum * ROW
    f.seek(strat_hdr)
    (n_strat,) = struct.unpack("<Q", f.read(8))
    strat_base = strat_hdr + 8
    assert size == strat_base + n_strat * ROW, "size mismatch — truncated blob?"
    _plog(f"iters={iters} n_cum={n_cum:,} n_strat={n_strat:,} strat@{strat_base}")

    # ---- pass 1: sequential scan -> 256 bucket files by top key byte --------------------------
    tmp = os.path.join(out_dir, "_buckets")
    os.makedirs(tmp, exist_ok=True)
    buckets = [open(os.path.join(tmp, f"b{i:03d}.bin"), "wb") for i in range(256)]
    done = 0
    t0 = time.time()
    last_hb = t0
    while done < n_strat:
        n = min(CHUNK_ROWS, n_strat - done)
        raw = f.read(n * ROW)
        assert len(raw) == n * ROW, "short read"
        keys = np.frombuffer(raw, dtype=DT_SCAN, count=n)
        rec = np.empty(n, dtype=DT_REC)
        rec["a"] = keys["a"]  # value-preserving byteswap: big-endian interpretation, native store
        rec["b"] = keys["b"]
        rec["o"] = strat_base + (done + np.arange(n, dtype=np.uint64)) * ROW
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
            pct = 100.0 * done / n_strat
            eta = (now - t0) / done * (n_strat - done)
            _plog(f"pass1 {done:,}/{n_strat:,} ({pct:.1f}%) eta {eta/60:.1f}m")
    for b in buckets:
        b.close()
    f.close()
    _plog(f"pass1 done in {(time.time()-t0)/60:.1f}m")

    # ---- pass 2: per-bucket in-RAM sort -> final sorted arrays --------------------------------
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
    assert total == n_strat, f"pass2 total {total} != n_strat {n_strat}"
    _plog(f"pass2 done in {(time.time()-t1)/60:.1f}m")

    meta = {"n_strat": int(n_strat), "iters": int(iters), "blob": os.path.abspath(blob),
            "blob_bytes": size, "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    # ---- self-check: random records must round-trip blob -> index -> same offset --------------
    A = np.memmap(os.path.join(out_dir, "A.bin"), dtype="<u8", mode="r")
    B = np.memmap(os.path.join(out_dir, "B.bin"), dtype="<u8", mode="r")
    O = np.memmap(os.path.join(out_dir, "O.bin"), dtype="<u8", mode="r")
    rng = random.Random(13)
    blob_f = open(blob, "rb")
    bad = 0
    for _ in range(check_n):
        j = rng.randrange(n_strat)
        off = strat_base + j * ROW
        blob_f.seek(off)
        key = blob_f.read(16)
        # np.uint64 casts are LOAD-BEARING (same as _IndexedSnap.get): a Python-int scalar degrades
        # searchsorted to a full-array object-comparison scan (~0.9 s/call on 348M rows).
        a = np.uint64(int.from_bytes(key[:8], "big"))
        b = np.uint64(int.from_bytes(key[8:16], "big"))
        lo = int(np.searchsorted(A, a, "left"))
        hi = int(np.searchsorted(A, a, "right"))
        k = lo + int(np.searchsorted(B[lo:hi], b, "left"))
        # duplicates of the same 16-byte key cannot exist (it's a hash-map dump), so the found
        # slot must hold exactly this record's offset among the (a,b)-equal run of length 1
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
