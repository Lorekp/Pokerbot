#!/usr/bin/env python3
"""blueprint-v4 Stage-0 G5 driver: forced-response probe per arm (design doc §7.1.4).

Runs bp3_forced_response_probe.py `rollout` (8 shards x 5000 deals, seed 13, hands AA and K9s)
against one arm's checkpoint with the arm's env (key mode / postflop mode / K), then `merge`.
Shards run as parallel subprocesses (each is single-threaded Python).

TWO sanity gates per shard:
  * ``adapter_fallback_rate == 0`` — catches a BROKEN serve twin (bridge/lookup EXCEPTION or an
    all-zero row). NOTE: this does NOT detect a fingerprint MISS — a miss returns a valid
    uniform-legal distribution, so it is silently absorbed and never increments fallback_count.
  * ``postflop miss-rate`` (flop/turn/river fingerprint misses, POKERBOT_BP_COVERAGE) <=
    ``--max-postflop-miss`` — the REAL train/probe skew signal. Under the vast fullhist key space the
    postflop TAIL misses far more than the summary control and is served as uniform-random play, which
    biases EV(open)-EV(limp) in an arm-dependent way (the exact axis A1/A2 rest on). A shard whose
    postflop miss-rate exceeds the bar FAILS: the G5 verdict from that arm is NOT trustworthy until the
    probe is made MRR-aware / the arm is better visited.

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
    ap.add_argument("--mrr", action="store_true",
                    help="forward --mrr to every shard (M-RR thinned serve menus, POKERBOT_MRR=1) — "
                         "set for every FULLHIST+M-RR-trained arm; leave off for summary controls.")
    ap.add_argument("--snapshot-index", default=None,
                    help="dir built by bpv4_nlcfr_index.py — forwarded to every shard so the 55 GB "
                         "cloud fullhist blob is served via mmap binary-search instead of the full "
                         "in-RAM load (which needs >100 GB). Default None = in-RAM path.")
    ap.add_argument("--max-postflop-miss", type=float, default=0.02,
                    help="fail a shard whose flop/turn/river fingerprint miss-rate exceeds this (the "
                         "fullhist train/probe skew signal adapter_fallback_rate cannot see). Default "
                         "0.02 (2%%). Raise deliberately only if the differential vs the summary control "
                         "is understood.")
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
            if args.mrr:
                cmd.append("--mrr")
            if args.snapshot_index:
                cmd += ["--snapshot-index", args.snapshot_index]
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
                cov = d.get("bp_coverage", {}) or {}
                pf_h = sum(cov.get(st, {}).get("hits", 0) for st in ("flop", "turn", "river"))
                pf_m = sum(cov.get(st, {}).get("misses", 0) for st in ("flop", "turn", "river"))
                pf_miss = pf_m / max(1, pf_h + pf_m)
                print(f"[{args.arm}] done {hand} shard {s}: {d['deals']} deals "
                      f"fallback={fb:.4f} postflop_miss={pf_miss:.4f} [{pf_m}/{pf_h + pf_m}] "
                      f"wall={d['wall_s']:.0f}s", flush=True)
                if fb > 0.0:
                    fails.append((hand, s, f"fallback {fb}"))
                # The fingerprint-miss gate adapter_fallback_rate cannot provide: a high postflop
                # miss-rate means much of this arm's downstream play was uniform-random (train/probe
                # abstraction skew), so its EV(open)-EV(limp) is not trustworthy.
                if not cov:
                    fails.append((hand, s, "no bp_coverage recorded (POKERBOT_BP_COVERAGE off?)"))
                elif pf_miss > args.max_postflop_miss:
                    fails.append((hand, s, f"postflop miss-rate {pf_miss:.4f} > "
                                           f"{args.max_postflop_miss:.4f}"))
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
        pf = h.get("postflop_miss_rate", 0.0)
        print(f"  {hand} postflop_miss_rate={pf:.4f} (bp_coverage={h.get('bp_coverage', {})})")
        for k, v in h["paired_vs_limp"].items():
            print(f"  {hand} {k}: {v['delta_bb100']:+.1f} ± {v['se_bb100']:.1f}")
        for a, s in h["arms"].items():
            print(f"  {hand} EV[{a}] = {s['ev_bb100']:+.1f} ± {s['se_bb100']:.1f}")
    print(f"[{args.arm}] wall {time.time()-t0:.0f}s -> {merged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
