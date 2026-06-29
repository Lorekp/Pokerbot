#!/usr/bin/env python3
"""Compare two NLCFR001 trainer-state blobs for numerical equivalence.

Format (little-endian):
  magic[8] = 'NLCFR001'
  iters_done : u64 LE
  map(cum)   : n:u64 LE, then n * (16-byte key + 11 * f64 LE)
  map(strat) : n:u64 LE, then n * (16-byte key + 11 * f64 LE)

Reports: same iters_done, same key sets per map, count of keys only in A / only in B,
and the max abs diff over the 11-wide rows on the SHARED keys, for cum and strat.
Compares against the production gate atol = 1e-9.
"""
import sys
import struct

MAGIC = b"NLCFR001"
NACT = 11
ROW_BYTES = 16 + NACT * 8  # 16-byte key + 11 f64


def parse_map(buf, off):
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    d = {}
    for _ in range(n):
        key = bytes(buf[off:off + 16])
        off += 16
        row = struct.unpack_from("<" + "d" * NACT, buf, off)
        off += NACT * 8
        d[key] = row
    return d, off


def parse_blob(path):
    with open(path, "rb") as f:
        buf = f.read()
    assert buf[:8] == MAGIC, f"bad magic in {path}: {buf[:8]!r}"
    off = 8
    (iters_done,) = struct.unpack_from("<Q", buf, off)
    off += 8
    cum, off = parse_map(buf, off)
    strat, off = parse_map(buf, off)
    assert off == len(buf), f"trailing bytes in {path}: off={off} len={len(buf)}"
    return iters_done, cum, strat


def cmp_map(a, b, name):
    ka, kb = set(a), set(b)
    only_a = ka - kb
    only_b = kb - ka
    shared = ka & kb
    max_diff = 0.0
    max_key = None
    for k in shared:
        ra, rb = a[k], b[k]
        for i in range(NACT):
            d = abs(ra[i] - rb[i])
            if d > max_diff:
                max_diff = d
                max_key = k
    # Also: among keys only in one side, what is the max abs value (how "real"
    # are the missing rows?)
    def max_abs(d, keys):
        m = 0.0
        for k in keys:
            for v in d[k]:
                if abs(v) > m:
                    m = abs(v)
        return m
    max_only_a = max_abs(a, only_a) if only_a else 0.0
    max_only_b = max_abs(b, only_b) if only_b else 0.0
    print(f"[{name}] |A|={len(a)} |B|={len(b)} shared={len(shared)} "
          f"only_A={len(only_a)} only_B={len(only_b)}")
    print(f"[{name}] max abs diff on SHARED rows = {max_diff:.6e}  (key={max_key.hex() if max_key else None})")
    print(f"[{name}] max abs VALUE among only-A rows = {max_only_a:.6e}; only-B rows = {max_only_b:.6e}")
    return max_diff, len(only_a), len(only_b), max_only_a, max_only_b


def main():
    if len(sys.argv) != 3:
        print("usage: A.bin B.bin", file=sys.stderr)
        sys.exit(2)
    pa, pb = sys.argv[1], sys.argv[2]
    ia, ca, sa = parse_blob(pa)
    ib, cb, sb = parse_blob(pb)
    print(f"A = {pa}")
    print(f"B = {pb}")
    print(f"iters_done: A={ia} B={ib} {'MATCH' if ia == ib else 'DIFFER'}")
    cd = cmp_map(ca, cb, "cum_regret")
    sd = cmp_map(sa, sb, "strat_sum")
    GATE = 1e-9
    worst = max(cd[0], sd[0])
    only_total = cd[1] + cd[2] + sd[1] + sd[2]
    print("-" * 60)
    print(f"production gate atol = {GATE:.0e}")
    print(f"worst max-abs-diff on shared rows = {worst:.6e} -> "
          f"{'WITHIN gate' if worst <= GATE else 'EXCEEDS gate'}")
    print(f"total non-shared keys (A-only + B-only, both maps) = {only_total}")
    if only_total == 0 and worst <= GATE:
        print("VERDICT: numerically EQUIVALENT (key sets identical, shared rows within atol)")
    elif only_total > 0:
        print("VERDICT: key sets DIFFER -> not bit-identical; equivalence depends on "
              "whether missing rows are negligible (see max-abs-VALUE above)")
    else:
        print("VERDICT: key sets match but shared rows EXCEED atol")


if __name__ == "__main__":
    main()
