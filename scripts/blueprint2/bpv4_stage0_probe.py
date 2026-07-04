#!/usr/bin/env python3
"""blueprint-v4 Stage-0 G5 driver: forced-response probe per arm (design doc §7.1.4).

Runs bp3_forced_response_probe.py `rollout` (8 shards x 5000 deals, seed 13, hands AA and K9s)
against one arm's checkpoint with the arm's env (key mode / postflop mode / K), then `merge`.
Shards run as parallel subprocesses (each is single-threaded Python). Adapter fallback rate MUST
be 0.0000 in every shard (else the serve twin is broken — abort, fix parity, rerun).

Usage:
  python scripts/blueprint2/bpv4_stage0_probe.py --arm H200 \
      --ckpt runs/native_cfr/bpv4_stage0/H200/iter_040000.nlcfr \
      --key-mode fullhist --postflop-mode fast --num-buckets 200 \
      [--shards 8] [--deals 5000] [--parallel 6] [--hands AA,K9s]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
PROBE = os.path.join(_HERE, "bp3_forced_response_probe.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--key-mode", choices=("prefhist", "fullhist", "summary"), required=True)
    ap.add_argument("--postflop-mode", choices=("emd", "fast"), required=True)
    ap.add_argument("--num-buckets", type=int, required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--deals", type=int, default=5000)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--hands", default="AA,K9s")
    args = ap.parse_args()

    hands = [h.strip() for h in args.hands.split(",") if h.strip()]
    jobs = []
    outs = []
    for hand in hands:
        for s in range(args.shards):
            out = os.path.join(_ROOT, "artifacts",
                               f"_bpv4_probe_{args.arm}_{hand}.shard{s}.json")
            outs.append(out)
            if os.path.exists(out):
                continue  # kill-safe resume: completed shards are kept
            cmd = [sys.executable, PROBE, "rollout",
                   "--blueprint", args.ckpt, "--hand", hand,
                   "--start", str(s * args.deals * 3), "--deals", str(args.deals),
                   "--seed", str(args.seed),
                   "--key-mode", args.key_mode, "--postflop-mode", args.postflop_mode,
                   "--num-buckets", str(args.num_buckets),
                   "--out", out]
            jobs.append((hand, s, cmd, out))

    print(f"[{args.arm}] {len(jobs)} shard jobs to run ({len(outs)-len(jobs)} already done)",
          flush=True)
    running: list[tuple] = []
    t0 = time.time()
    fails = []
    while jobs or running:
        while jobs and len(running) < args.parallel:
            hand, s, cmd, out = jobs.pop(0)
            logf = open(out.replace(".json", ".out"), "w")
            p = subprocess.Popen(cmd, cwd=_ROOT, stdout=logf, stderr=subprocess.STDOUT)
            running.append((hand, s, p, out, logf))
            print(f"[{args.arm}] launched {hand} shard {s} (pid {p.pid})", flush=True)
        time.sleep(10)
        still = []
        for hand, s, p, out, logf in running:
            if p.poll() is None:
                still.append((hand, s, p, out, logf))
                continue
            logf.close()
            if p.returncode != 0 or not os.path.exists(out):
                fails.append((hand, s, p.returncode))
                print(f"[{args.arm}] FAIL {hand} shard {s} rc={p.returncode}", flush=True)
            else:
                d = json.load(open(out, encoding="utf-8"))
                fb = d["adapter_fallback_rate"]
                print(f"[{args.arm}] done {hand} shard {s}: {d['deals']} deals "
                      f"fallback={fb:.4f} wall={d['wall_s']:.0f}s", flush=True)
                if fb > 0.0:
                    fails.append((hand, s, f"fallback {fb}"))
        running = still
    if fails:
        print(f"[{args.arm}] PROBE FAILED shards: {fails} — twin broken? fix parity, rerun",
              flush=True)
        return 1

    merged = os.path.join(_ROOT, "artifacts", f"bpv4_stage0_probe_{args.arm}.json")
    rc = subprocess.run([sys.executable, PROBE, "merge", "--out", merged] + outs,
                        cwd=_ROOT).returncode
    if rc != 0:
        return rc
    d = json.load(open(merged, encoding="utf-8"))
    print(f"\n[{args.arm}] G5 verdict inputs (paired vs limp, bb/100):")
    for hand, h in d["hands"].items():
        for k, v in h["paired_vs_limp"].items():
            print(f"  {hand} {k}: {v['delta_bb100']:+.1f} ± {v['se_bb100']:.1f}")
        for a, s in h["arms"].items():
            print(f"  {hand} EV[{a}] = {s['ev_bb100']:+.1f} ± {s['se_bb100']:.1f}")
    print(f"[{args.arm}] wall {time.time()-t0:.0f}s -> {merged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
