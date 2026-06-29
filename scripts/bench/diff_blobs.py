"""Diff two bench_trainer blobs row-by-row.
Blobs: magic 'NLCFR001' (8B) | iters_done (u64 LE) | cum_regret map | strat_sum map
map: n (u64 LE) | sorted-by-key rows: key (16B) + NUM_ACTIONS f64 (each 8B LE)
NUM_ACTIONS = 11.
"""
import struct
from pathlib import Path

NUM_ACTIONS = 11

def load(path):
    b = Path(path).read_bytes()
    assert b[:8] == b"NLCFR001", f"bad magic in {path}: {b[:8]!r}"
    p = 8
    iters_done = struct.unpack_from("<Q", b, p)[0]; p += 8
    def read_map():
        nonlocal p
        n = struct.unpack_from("<Q", b, p)[0]; p += 8
        m = {}
        for _ in range(n):
            key = bytes(b[p:p+16]); p += 16
            row = struct.unpack_from(f"<{NUM_ACTIONS}d", b, p); p += 8 * NUM_ACTIONS
            m[key] = row
        return m
    cum_regret = read_map()
    strat_sum = read_map()
    return iters_done, cum_regret, strat_sum

def diff_maps(a, b):
    """Return (max_abs_diff, n_keys_only_in_a, n_keys_only_in_b, n_common, n_action_elements_checked)."""
    keys_a, keys_b = set(a.keys()), set(b.keys())
    common = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    max_diff = 0.0
    where = None
    n_elem = 0
    for k in common:
        ra, rb = a[k], b[k]
        for ai in range(NUM_ACTIONS):
            d = abs(ra[ai] - rb[ai])
            n_elem += 1
            if d > max_diff:
                max_diff = d
                where = (k.hex(), ai, ra[ai], rb[ai])
    return max_diff, len(only_a), len(only_b), len(common), n_elem, where

base = Path(r"./artifacts/fast_merge_prod_shape")
for i in range(1, 6):
    sp = base / f"serial_1024_{i}.bin"
    fp = base / f"fast_1024_{i}.bin"
    si, scr, sss = load(sp)
    fi, fcr, fss = load(fp)
    print(f"--- run {i}: serial iters={si}, fast iters={fi} ---")
    d, oa, ob, nc, ne, where = diff_maps(scr, fcr)
    print(f"  cum_regret: max_abs_diff={d:.3e}  only_in_serial={oa}  only_in_fast={ob}  common={nc}  elements={ne}")
    if where:
        print(f"    worst: key={where[0]}  action={where[1]}  serial={where[2]:.6e}  fast={where[3]:.6e}  rel_diff={abs(where[2]-where[3])/max(abs(where[2]),1e-30):.3e}")
    d, oa, ob, nc, ne, where = diff_maps(sss, fss)
    print(f"  strat_sum : max_abs_diff={d:.3e}  only_in_serial={oa}  only_in_fast={ob}  common={nc}  elements={ne}")
    if where:
        print(f"    worst: key={where[0]}  action={where[1]}  serial={where[2]:.6e}  fast={where[3]:.6e}  rel_diff={abs(where[2]-where[3])/max(abs(where[2]),1e-30):.3e}")