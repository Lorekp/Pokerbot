#!/usr/bin/env python3
"""Fit (or re-fit) the blueprint-v2 river-OCHS table + sidecar + parity verification.

Wraps the native fit binary (tools/battery_native/examples/river_ochs_fit.rs), writes the
.sha256 sidecar the on-box verify + ship gate require, and closes the cross-language parity
chain by re-deriving every golden row with the pure-Python twin.

  python scripts/blueprint2/fit_river_ochs.py            # default: K=160, 250k spots, seed 7
  python scripts/blueprint2/fit_river_ochs.py --spots 500000 --seed 11

Deterministic given --seed. Runtime: ~seconds (native, rayon). Artifacts:
  data/abstraction/river_ochs_160.rochs          (~10 KB ROCHS001 table)
  data/abstraction/river_ochs_160.rochs.sha256   (sidecar)
  data/abstraction/river_ochs_160.golden.tsv     (512 parity rows, both test suites pin it)

NOTE: refitting CHANGES the table -> any blueprint trained under the old table must keep the
old table to serve. Do not refit between train and serve of the same run.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CRATE = os.path.join(ROOT, "tools", "battery_native")
OUT = os.path.join(ROOT, "data", "abstraction", "river_ochs_160.rochs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=160)
    ap.add_argument("--spots", type=int, default=250_000)
    ap.add_argument("--kmeans-iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--class-mc", type=int, default=60_000)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    out_abs = os.path.abspath(args.out)
    print(f"[fit] building + running native fit (K={args.k}, spots={args.spots}, seed={args.seed})",
          flush=True)
    cmd = ["cargo", "run", "--release", "--example", "river_ochs_fit", "--",
           "--k", str(args.k), "--spots", str(args.spots),
           "--kmeans-iters", str(args.kmeans_iters), "--seed", str(args.seed),
           "--class-mc", str(args.class_mc), "--out", out_abs]
    r = subprocess.run(cmd, cwd=CRATE)
    if r.returncode != 0:
        raise SystemExit(f"native fit failed (rc={r.returncode})")

    sha = hashlib.sha256(open(out_abs, "rb").read()).hexdigest()
    with open(out_abs + ".sha256", "w", encoding="ascii") as f:
        f.write(sha + "\n")
    print(f"[fit] sha256 sidecar written: {sha}", flush=True)

    # Cross-language parity: re-derive every golden row with the pure-Python twin.
    sys.path.insert(0, ROOT)
    from pokerbot.encoding.river_ochs import RiverOchs  # noqa: E402

    golden = out_abs[: -len(".rochs")] + ".golden.tsv" if out_abs.endswith(".rochs") \
        else out_abs + ".golden.tsv"
    t = RiverOchs.load(out_abs)
    n = 0
    for line in open(golden, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = [int(x) for x in line.split()]
        got = t.bucket((f[0], f[1]), f[2:7])
        if got != f[7]:
            raise SystemExit(f"PARITY FAIL at golden row {n}: {f} -> python {got}")
        n += 1
    print(f"[fit] python golden parity: {n}/{n} rows OK", flush=True)
    print(f"[fit] DONE: {out_abs}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
