#!/usr/bin/env python3
"""Ranged (partial) reader for a SORTED NLCFR001 checkpoint -- pull the ~few-thousand wanted node
rows WITHOUT downloading the (up to ~500 GB) blob.

Motivation: under FULLHIST keying an arm-3 checkpoint is 400-500 GB and no longer fits on local
disk (the 2026-07-08 arm-3 stop left a 499.8 GiB iter_015000.nlcfr that cannot be pulled onto a
283 GB drive). But the trainer serializes every section GLOBALLY SORTED by the 16-byte key
interpreted as a big-endian u128 == plain lexicographic byte order (trainer.rs
`gather_sorted_all` -> `par_sort_unstable_by_key(u128::from_be_bytes)`). So the ~2k canonical
preflop node keys can be found by BINARY SEARCH over small ranged reads (HTTP Range GETs on S3, or
pread on a local file) -- a few hundred MB and a few thousand requests instead of the whole blob.

`scan_ranged(reader, wanted)` returns the SAME 5-tuple as
`bpv4_stage0_gates.scan_checkpoint(path, wanted)`:
    (iters, n_cum, n_strat, cum_hits, strat_hits)
so EVERY downstream gate function (extract_nodes / evaluate / neg-control / menu-guard) is reused
byte-for-byte -- this module only replaces the "get the rows off disk" step. The self-test
(`python bpv4_ranged_scan.py --self-test`) builds a synthetic NLCFR001 blob and asserts the ranged
reader returns EXACTLY what the trusted mmap `scan_checkpoint` returns on the same file.

NLCFR001 layout (trainer.rs write_sections_to_file):
    magic "NLCFR001" (8) | iters u64 LE (8) | cum_len u64 LE (8) | cum rows | strat_len u64 LE (8) | strat rows
    each row = key (16 B) + 11 * f64 LE (88 B) = 104 B, rows ASCENDING by lexicographic key.
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

MAGIC = b"NLCFR001"
NUM_ACTIONS = 11
ROW = 16 + NUM_ACTIONS * 8          # 104 -- MUST match bpv4_stage0_gates.ROW / trainer NLCFR_ROW_BYTES
HEADER = 16                          # magic(8) + iters(8)
_ROW_FMT = struct.Struct("<11d")     # 11 f64 LE, the value payload after the 16-byte key


# ------------------------------------------------------------------------------------------------
# Readers: a uniform .read(off, length) -> bytes over S3 (Range GET) or a local file (pread).
# ------------------------------------------------------------------------------------------------
class LocalRangedReader:
    """pread over a local file -- used by the self-test and for already-downloaded blobs."""

    def __init__(self, path: str):
        self._f = open(path, "rb")
        self._lock = threading.Lock()
        self._size = os.path.getsize(path)
        self.n_get = 0
        self.n_bytes = 0

    def size(self) -> int:
        return self._size

    def read(self, off: int, length: int) -> bytes:
        # seek+read under a lock (portable; os.pread is POSIX-only -- the same reason the trainer's
        # write path has a Windows seek_write fallback). One lock is fine: local reads are not the
        # bottleneck and this keeps the self-test identical across platforms.
        with self._lock:
            self._f.seek(off)
            data = self._f.read(length)
            self.n_get += 1
            self.n_bytes += len(data)
            return data

    def close(self):
        self._f.close()


class S3RangedReader:
    """HTTP Range GETs against one S3 object. Reaches S3 through the box's SSL-inspecting proxy the
    same way the kill gate does (boto3 verify=False + silenced InsecureRequestWarning)."""

    def __init__(self, bucket: str, key: str, region: str):
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            warnings.filterwarnings("ignore", message="Unverified HTTPS request")
        import boto3
        self._s3 = boto3.client("s3", region_name=region, verify=False)
        self._bucket = bucket
        self._key = key
        self._size = int(self._s3.head_object(Bucket=bucket, Key=key)["ContentLength"])
        self._lock = threading.Lock()
        self.n_get = 0
        self.n_bytes = 0

    def size(self) -> int:
        return self._size

    def read(self, off: int, length: int) -> bytes:
        end = min(self._size, off + length) - 1
        if end < off:
            return b""
        rng = f"bytes={off}-{end}"
        last = None
        for attempt in range(5):
            try:
                r = self._s3.get_object(Bucket=self._bucket, Key=self._key, Range=rng)
                data = r["Body"].read()
                with self._lock:
                    self.n_get += 1
                    self.n_bytes += len(data)
                return data
            except Exception as e:  # transient proxy / throttle -- backoff and retry
                last = e
                time.sleep(0.25 * (attempt + 1))
        raise RuntimeError(f"S3 Range GET failed after retries ({rng}): {last}")


# ------------------------------------------------------------------------------------------------
# Core: INTERPOLATION-search the sorted wanted keys inside one section.
#
# The keys are 128-bit fingerprints -> uniformly distributed, so interpolation search converges in
# ~O(log log n) ~= 5 probes vs binary search's ~31 -- decisive when each probe is an S3 Range GET.
# A bisection fallback (degenerate top-8 ties / no progress / iter cap) guarantees termination, and
# once the interval is small the whole [lo,hi] span is pulled in ONE GET and scanned locally, so the
# per-key request count is a handful regardless of how sparse the wanted keys are in the section.
# ------------------------------------------------------------------------------------------------
TAIL_ROWS = 2048          # once hi-lo <= this, one GET (~213 KB) fetches the span; local search
_ITER_CAP = 96            # interpolation/bisection iters before forcing the tail read (safety net)


def _top8(k: bytes) -> int:
    return int.from_bytes(k[:8], "big")   # monotone position proxy under the serializer's BE order


def _search_section(reader, base: int, n_rows: int,
                    wanted_sorted: list[bytes], workers: int) -> dict[bytes, tuple]:
    """Return {key: (11 f64)} for every wanted key present in the [base, base+n_rows*ROW) section."""
    hits: dict[bytes, tuple] = {}
    if n_rows <= 0 or not wanted_sorted:
        return hits
    hlock = threading.Lock()
    kcache: dict[int, bytes] = {}
    kclock = threading.Lock()

    def _key_at(idx: int) -> bytes:
        with kclock:
            hit = kcache.get(idx)
        if hit is not None:
            return hit
        k = reader.read(base + idx * ROW, 16)
        with kclock:
            kcache[idx] = k
        return k

    k0, kN = _key_at(0), _key_at(n_rows - 1)     # section bounds (shared across all keys)
    v0, vN = _top8(k0), _top8(kN)

    def _one(target: bytes) -> None:
        if target < k0 or target > kN:           # provably absent -- no probe needed
            return
        t = _top8(target)
        lo, hi = 0, n_rows - 1                    # inclusive; target is always in [lo, hi]
        vlo, vhi = v0, vN
        for it in range(_ITER_CAP):
            if hi - lo <= TAIL_ROWS:
                break
            if vhi <= vlo:                        # degenerate (top-8 tie across interval)
                mid = (lo + hi) >> 1
            else:
                frac = (t - vlo) / (vhi - vlo)
                frac = 0.0 if frac < 0.0 else 1.0 if frac > 1.0 else frac
                mid = lo + int(frac * (hi - lo))
                if mid <= lo:
                    mid = lo + 1
                elif mid >= hi:
                    mid = hi - 1
            k = _key_at(mid)
            if k == target:
                vals = _ROW_FMT.unpack(reader.read(base + mid * ROW + 16, ROW - 16))
                with hlock:
                    hits[target] = vals
                return
            if k < target:
                lo, vlo = mid + 1, _top8(k)
            else:
                hi, vhi = mid - 1, _top8(k)
        # tail: [lo, hi] inclusive fits in one GET; binary-search it locally on full 16-byte keys.
        if lo > hi:
            return
        span = hi - lo + 1
        data = reader.read(base + lo * ROW, span * ROW)
        L, R = 0, span
        while L < R:
            m = (L + R) >> 1
            k = data[m * ROW:m * ROW + 16]
            if k == target:
                vals = _ROW_FMT.unpack(data[m * ROW + 16:m * ROW + ROW])
                with hlock:
                    hits[target] = vals
                return
            if k < target:
                L = m + 1
            else:
                R = m

    if workers <= 1:
        for t in wanted_sorted:
            _one(t)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_one, wanted_sorted))
    return hits


def scan_ranged(reader, wanted: set[bytes], *, want_cum: bool = True, want_strat: bool = True,
                workers: int = 16, progress: bool = True):
    """Drop-in ranged twin of bpv4_stage0_gates.scan_checkpoint.

    Returns (iters, n_cum, n_strat, cum_hits, strat_hits). Sections not requested return {} and their
    count is still read from the header (so n_cum / n_strat are always the true section sizes)."""
    head = reader.read(0, HEADER + 8)                 # magic + iters + cum_len
    if head[:8] != MAGIC:
        raise ValueError(f"bad magic {head[:8]!r} (expected {MAGIC!r}) -- not an NLCFR001 blob")
    (iters,) = struct.unpack_from("<Q", head, 8)
    (n_cum,) = struct.unpack_from("<Q", head, 16)
    cum_base = HEADER + 8                              # 24
    strat_len_off = cum_base + n_cum * ROW
    (n_strat,) = struct.unpack_from("<Q", reader.read(strat_len_off, 8), 0)
    strat_base = strat_len_off + 8

    ws = sorted(wanted)
    t0 = time.time()
    cum_hits = (_search_section(reader, cum_base, n_cum, ws, workers) if want_cum else {})
    if progress:
        print(f"[ranged] cum section: {n_cum:,} rows, {len(cum_hits):,}/{len(ws):,} keys hit "
              f"({reader.n_get:,} GETs, {reader.n_bytes/1e6:.0f} MB, {time.time()-t0:.0f}s)",
              flush=True)
    strat_hits = (_search_section(reader, strat_base, n_strat, ws, workers) if want_strat else {})
    if progress:
        print(f"[ranged] strat section: {n_strat:,} rows, {len(strat_hits):,}/{len(ws):,} keys hit "
              f"(cumulative {reader.n_get:,} GETs, {reader.n_bytes/1e6:.0f} MB, "
              f"{time.time()-t0:.0f}s)", flush=True)
    return iters, n_cum, n_strat, cum_hits, strat_hits


# ------------------------------------------------------------------------------------------------
# Self-test: synthetic NLCFR001 blob; ranged reader must MATCH the trusted mmap scan_checkpoint.
# ------------------------------------------------------------------------------------------------
def _self_test() -> int:
    import random
    import tempfile

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bpv4_stage0_gates as g0  # the trusted mmap reader we validate against

    rng = random.Random(1234567)
    n = 20000

    def _mkrows(salt: int):
        keys = set()
        while len(keys) < n:
            keys.add(bytes(rng.randrange(256) for _ in range(16)))
        rows = []
        for k in sorted(keys):                        # lexicographic == the serializer's order
            vals = [float((int.from_bytes(k[:4], "big") ^ (salt * 2654435761)) % 100000) + a
                    for a in range(NUM_ACTIONS)]
            rows.append((k, vals))
        return rows

    cum_rows = _mkrows(1)
    strat_rows = _mkrows(2)

    def _section(rows):
        b = bytearray()
        b += struct.pack("<Q", len(rows))
        for k, vals in rows:
            b += k
            b += struct.pack("<11d", *vals)
        return bytes(b)

    blob = bytearray()
    blob += MAGIC
    blob += struct.pack("<Q", 15000)                  # iters
    blob += _section(cum_rows)
    blob += _section(strat_rows)

    tmp = os.path.join(tempfile.gettempdir(), "bpv4_ranged_selftest.nlcfr")
    with open(tmp, "wb") as f:
        f.write(blob)

    # Wanted = a mix of present (from BOTH sections + shared) and absent keys.
    present = [cum_rows[i][0] for i in (0, 1, 7, 99, 5000, 12345, n - 2, n - 1)]
    present += [strat_rows[i][0] for i in (3, 42, 8000, n - 3)]
    absent = [bytes([0] * 16), bytes([255] * 16), bytes(rng.randrange(256) for _ in range(16))]
    wanted = set(present) | set(absent)

    ok = True
    for workers in (1, 16):
        rr = LocalRangedReader(tmp)
        it_r, nc_r, ns_r, cum_r, strat_r = scan_ranged(rr, wanted, workers=workers, progress=False)
        it_m, nc_m, ns_m, cum_m, strat_m = g0.scan_checkpoint(tmp, wanted)
        checks = {
            "iters": it_r == it_m == 15000,
            "n_cum": nc_r == nc_m == n,
            "n_strat": ns_r == ns_m == n,
            "cum_hits_equal": cum_r == cum_m,
            "strat_hits_equal": strat_r == strat_m,
            "no_false_positives": set(cum_r) | set(strat_r) <= wanted,
        }
        for name, passed in checks.items():
            print(f"  [{workers:2d} workers] {name:22s} {'PASS' if passed else 'FAIL'}")
            ok &= passed
        rr.close()
    os.remove(tmp)
    print(f"\nSELF-TEST: {'PASS -- ranged reader matches mmap scan_checkpoint' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Ranged NLCFR001 reader (self-test + ad-hoc probe).")
    ap.add_argument("--self-test", action="store_true", help="run the synthetic-blob equivalence test")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
