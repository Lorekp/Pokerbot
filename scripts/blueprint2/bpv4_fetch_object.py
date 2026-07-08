#!/usr/bin/env python3
"""Fast parallel whole-object S3 download (chunked range GETs) — much faster than `aws s3 cp` here
(~100 MB/s vs ~38 MB/s through the SSL-inspecting proxy). Verify=False + silenced warning (repo
convention). Preallocates the dest, writes chunks at their offsets under a lock.

Usage:
  python scripts/blueprint2/bpv4_fetch_object.py --key <s3-key> --out <path> [--workers 16]
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import bpv4_kill_gate as kg  # noqa: E402

CHUNK = 128 * 1024 * 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=kg.DEFAULT_BUCKET)
    ap.add_argument("--key", required=True)
    ap.add_argument("--region", default=kg.DEFAULT_REGION)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    import boto3
    s3 = boto3.client("s3", region_name=args.region, verify=False)
    size = int(s3.head_object(Bucket=args.bucket, Key=args.key)["ContentLength"])
    print(f"[fetch-obj] {args.key} = {size/1e9:.1f} GB -> {args.out}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        f.truncate(size)
    fh = open(args.out, "r+b")
    wlock = threading.Lock()
    done = [0]
    t0 = time.time()

    def _get(off, length):
        end = off + length - 1
        for att in range(6):
            try:
                r = s3.get_object(Bucket=args.bucket, Key=args.key, Range=f"bytes={off}-{end}")
                return r["Body"].read()
            except Exception:
                if att == 5:
                    raise
                time.sleep(0.5 * (att + 1))

    def _chunk(ci):
        off = ci * CHUNK
        length = min(CHUNK, size - off)
        data = _get(off, length)
        with wlock:
            fh.seek(off)
            fh.write(data)
            done[0] += len(data)
            el = max(1e-6, time.time() - t0)
            pct = 100.0 * done[0] / size
            rate = done[0] / el / 1e6
            eta = (size - done[0]) / max(1, done[0] / el)
            sys.stdout.write(f"\r[fetch-obj] {done[0]/1e9:6.1f}/{size/1e9:.1f} GB ({pct:5.1f}%) "
                             f"{rate:6.0f} MB/s ETA {eta/60:5.1f} min   ")
            sys.stdout.flush()

    n_chunks = (size + CHUNK - 1) // CHUNK
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(_chunk, range(n_chunks)))
    fh.close()
    print(f"\n[fetch-obj] done {os.path.getsize(args.out)/1e9:.1f} GB in {(time.time()-t0)/60:.1f} min",
          flush=True)
    assert os.path.getsize(args.out) == size, "size mismatch"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
