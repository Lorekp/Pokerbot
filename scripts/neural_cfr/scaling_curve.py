"""Throughput-vs-cores scaling curve for the VR-DeepPDCFR+ trainer.

Runs bench_saturation.py in a FRESH subprocess per topology point (the XLA device count is frozen at
jax import, so each point must be its own process), collects the JSON, and prints a scaling table:
traversals/sec and net-updates/sec as a function of compute-thread budget, plus efficiency (% of linear
extrapolated from the 1-thread point).

Example:
    python -m scripts.neural_cfr.scaling_curve --game nlhe --points 1,2,4,6 --traversals 96
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def run_point(game: str, topology: str, vcpus: int, devices: int | None, intra_op: int | None,
              traversals: int, warmup: int, measure: int) -> dict:
    cmd = [
        sys.executable, "-m", "scripts.neural_cfr.bench_saturation",
        "--game", game, "--topology", topology, "--vcpus", str(vcpus),
        "--traversals", str(traversals), "--warmup", str(warmup), "--measure", str(measure),
        "--json",
    ]
    if devices is not None:
        cmd += ["--devices", str(devices)]
    if intra_op is not None:
        cmd += ["--intra-op", str(intra_op)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # the JSON line is the last non-empty stdout line
    out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    if not out_lines:
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
        raise RuntimeError(f"no JSON from point devices={devices} intra_op={intra_op}")
    return json.loads(out_lines[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=["nlhe", "leduc"], default="nlhe")
    ap.add_argument("--topology", default="fine", choices=["fine", "coarse", "single"],
                    help="fine=N single-thread devices; single=1 device w/ N-thread Eigen")
    ap.add_argument("--vcpus", type=int, default=None)
    ap.add_argument("--points", default="1,2,4,6", help="comma list of device (fine) or thread (single) counts")
    ap.add_argument("--traversals", type=int, default=96)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--measure", type=int, default=3)
    args = ap.parse_args()

    pts = [int(x) for x in args.points.split(",")]
    rows = []
    for p in pts:
        if args.topology == "single":
            r = run_point(args.game, "single", args.vcpus or 64, None, p,
                          args.traversals, args.warmup, args.measure)
        else:
            r = run_point(args.game, args.topology, args.vcpus or 64, p, None,
                          args.traversals, args.warmup, args.measure)
        rows.append(r)
        print(f"  point {p}: threads={r['compute_threads']} "
              f"trav/s={r['traversals_per_sec']:.1f} upd/s={r['net_updates_per_sec']:.1f} "
              f"trav_cores~{r['trav_mean_cores']:.1f} grad_cores~{r['grad_mean_cores']:.1f}", flush=True)

    base = rows[0]
    print("\n=== scaling curve ({}, {}) ===".format(args.game, args.topology))
    print(f"{'threads':>8} {'trav/s':>10} {'speedup':>8} {'eff%':>6} "
          f"{'upd/s':>9} {'trav_cores':>11} {'grad_cores':>11}")
    for r in rows:
        thr = r["compute_threads"]
        sp = r["traversals_per_sec"] / base["traversals_per_sec"] if base["traversals_per_sec"] else 0
        base_thr = base["compute_threads"]
        eff = (sp / (thr / base_thr) * 100) if thr else 0
        print(f"{thr:>8} {r['traversals_per_sec']:>10.1f} {sp:>7.2f}x {eff:>5.0f}% "
              f"{r['net_updates_per_sec']:>9.1f} {r['trav_mean_cores']:>11.1f} {r['grad_mean_cores']:>11.1f}")


if __name__ == "__main__":
    main()
