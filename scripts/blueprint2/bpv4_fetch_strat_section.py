#!/usr/bin/env python3
"""Fetch ONLY the strat (average-strategy) section of a huge NLCFR001 checkpoint -> a preflop-only
.nlcfr, without downloading the (up to ~500 GB) cum section.

Under snapshot-avg the strat section is PREFLOP-ONLY and small (~41.5 GB for arm-3 15k vs the 537 GB
blob). The check-call EV probe (bp3_forced_response_probe --postflop-field checkcall) only consults
the AVERAGE preflop strategy, so a valid NLCFR001 with an EMPTY cum section + the real strat section
serves it exactly. Layout written:
    "NLCFR001"(8) | iters(8) | cum_len=0 (8) | strat_len(8) | <strat rows copied verbatim from S3>
The strat rows are byte-identical to the source (same 104 B rows), so scan_checkpoint / the mmap
index / make_blueprint_policy all read it unchanged.

Usage:
  python scripts/blueprint2/bpv4_fetch_strat_section.py --iter 15000 --out artifacts/bpv4_ladder/arm3_15k_stratonly.nlcfr
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
import bpv4_kill_gate as kg  # noqa: E402

ROW = 104
HDR_OUT = 32  # magic(8)+iters(8)+cum_len(8)+strat_len(8)
CHUNK = 256 * 1024 * 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=kg.DEFAULT_BUCKET)
    ap.add_argument("--prefix", default=kg.DEFAULT_PREFIX)
    ap.add_argument("--run-subdir", default="bpv4_emd160_arm3")
    ap.add_argument("--region", default=kg.DEFAULT_REGION)
    ap.add_argument("--iter", type=int, default=None)
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
    ladder = kg._list_ladder(s3, args.bucket, f"{args.prefix}/{args.run_subdir}")
    target = args.iter if args.iter is not None else ladder[-1][0]
    key = next(k for i, k in ladder if i == target)
    size = int(s3.head_object(Bucket=args.bucket, Key=key)["ContentLength"])

    def _get(off, length):
        end = off + length - 1
        for att in range(6):
            try:
                r = s3.get_object(Bucket=args.bucket, Key=key, Range=f"bytes={off}-{end}")
                return r["Body"].read()
            except Exception as e:
                if att == 5:
                    raise
                time.sleep(0.5 * (att + 1))

    hdr = _get(0, 24)
    assert hdr[:8] == b"NLCFR001", f"bad magic {hdr[:8]!r}"
    (iters,) = struct.unpack_from("<Q", hdr, 8)
    (n_cum,) = struct.unpack_from("<Q", hdr, 16)
    strat_len_off = 24 + n_cum * ROW
    (n_strat,) = struct.unpack_from("<Q", _get(strat_len_off, 8), 0)
    strat_rows_off = strat_len_off + 8
    strat_bytes = n_strat * ROW
    assert strat_rows_off + strat_bytes == size, \
        f"layout mismatch: {strat_rows_off}+{strat_bytes} != {size}"
    print(f"[fetch] iter {iters}: n_cum={n_cum:,} n_strat={n_strat:,} strat={strat_bytes/1e9:.1f} GB "
          f"(src blob {size/1e9:.1f} GB) -> {args.out}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total_out = HDR_OUT + strat_bytes
    with open(args.out, "wb") as f:
        f.truncate(total_out)
        f.seek(0)
        f.write(b"NLCFR001")
        f.write(struct.pack("<Q", iters))
        f.write(struct.pack("<Q", 0))          # cum_len = 0 (empty cum section)
        f.write(struct.pack("<Q", n_strat))    # strat_len

    wlock = threading.Lock()
    done = [0]
    t0 = time.time()
    fh = open(args.out, "r+b")

    def _chunk(ci):
        src = strat_rows_off + ci * CHUNK
        length = min(CHUNK, strat_bytes - ci * CHUNK)
        data = _get(src, length)
        with wlock:
            fh.seek(HDR_OUT + ci * CHUNK)
            fh.write(data)
            done[0] += len(data)
            el = max(1e-6, time.time() - t0)
            pct = 100.0 * done[0] / strat_bytes
            rate = done[0] / el / 1e6
            eta = (strat_bytes - done[0]) / max(1, done[0] / el)
            sys.stdout.write(f"\r[fetch] {done[0]/1e9:5.1f}/{strat_bytes/1e9:.1f} GB ({pct:5.1f}%) "
                             f"{rate:6.0f} MB/s ETA {eta/60:4.1f} min   ")
            sys.stdout.flush()

    n_chunks = (strat_bytes + CHUNK - 1) // CHUNK
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(_chunk, range(n_chunks)))
    fh.close()
    print(f"\n[fetch] wrote {args.out} ({os.path.getsize(args.out)/1e9:.1f} GB) in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    assert os.path.getsize(args.out) == total_out, "size mismatch after write"

    # sanity: scan the local preflop-only blob for a known node (rfi_BTN AA) via the trusted mmap reader
    import bpv4_stage0_gates as g0
    import preflop_root_diag as prd
    prd.set_pre_menu("arm3")
    nl = g0.build_nodes("fullhist")
    keys = g0.node_keys(nl, "fullhist")
    want = {keys[("rfi_BTN", c)] for c in range(169)}
    it, nc, ns, cum, strat = g0.scan_checkpoint(args.out, want)
    print(f"[fetch] sanity: local blob iters={it} n_cum={nc} n_strat={ns:,} rfi_BTN strat hits={len(strat)}",
          flush=True)
    assert nc == 0 and ns == n_strat and len(strat) > 0, "sanity scan failed"
    print("[fetch] OK: preflop-only strat blob is valid and readable.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
