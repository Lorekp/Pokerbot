#!/usr/bin/env python3
"""Interleaved A/B driver: alternate two bench_subgame builds on the same states/K/seeds and report
per-class and per-state paired ratios (medians over passes). Only relative ratios are trustworthy on
a busy box, which is exactly what alternation gives.

Usage: python bench/interleave.py A.exe B.exe --k 2000 --passes 5 [--states i,j,..] [--csv out.csv]
"""
import argparse
import statistics as st
import subprocess
import sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--k", type=int, default=2000)
ap.add_argument("--passes", type=int, default=5)
ap.add_argument("--states", default=None)
ap.add_argument("--data", default="bench_data")
args = ap.parse_args()


def run(exe, seed):
    cmd = [exe, args.data, "--k", str(args.k), "--repeats", "1", "--seed", str(seed), "--quiet-probs"]
    if args.states:
        cmd += ["--states", args.states]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    walls = {}
    cls = {}
    for line in out.splitlines():
        if line.startswith("RESULT"):
            _, sid, c, rep, ns, _ = line.split("\t")
            walls[sid] = int(ns)
            cls[sid] = c
    return walls, cls


wa = defaultdict(list)
wb = defaultdict(list)
cls_of = {}
for p in range(args.passes):
    seed = 4242 + p * 101
    for tag, exe, sink in (("A", args.a, wa), ("B", args.b, wb)):
        walls, cls = run(exe, seed)
        cls_of.update(cls)
        for sid, ns in walls.items():
            sink[sid].append(ns)
        print(f"[pass {p} {tag}] total={sum(walls.values())/1e9:.1f}s", flush=True)

# paired per-state ratios (median-of-passes each side)
per_class = defaultdict(list)
print("\nPER-STATE (A/B ratio = speedup of B over A):")
for sid in sorted(wa):
    ma = st.median(wa[sid])
    mb = st.median(wb[sid])
    r = ma / mb
    per_class[cls_of[sid]].append(r)
    print(f"  {sid:>24s} {cls_of[sid]:<10s} A={ma/1e6:9.1f}ms B={mb/1e6:9.1f}ms x{r:.3f}")
print("\nPER-CLASS median speedup (B vs A):")
for c, rs in sorted(per_class.items()):
    rs.sort()
    print(f"  {c:<10s} n={len(rs)} median=x{rs[len(rs)//2]:.3f} min=x{min(rs):.3f} max=x{max(rs):.3f}")
