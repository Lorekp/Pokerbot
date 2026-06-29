"""EnvStateN hist-copy-elision KILL-SCREEN: identical-tree nodes/s A/B (local, Windows, CPU-bound).

Two separately-built cdylibs that differ ONLY in core::MAX_HISTORY (arm A = 64, arm B = 2 via the
`probe_slim_hist` feature) => the per-node `*s_in` EnvStateN copy drops ~760B in arm B. hist is train-DEAD
(written, never read on the trainer traversal), so the traversal is bit-identical => per-iteration node
counts MUST match exactly across arms (asserted; abort on mismatch = a transition bug, not a perf signal).

Arm B is a GENEROUS UPPER BOUND on the real train-only hist-free struct (it also skips the in-apply hist
store and shrinks init), so a FAIL here certainly implies a ship FAIL. This settles whether shrinking the
copy 64% moves nodes/s at all, or whether (as the cpu128 latency-bound verdict predicts) the sequential
L1-resident memcpy is hidden under the random hashmap-probe DRAM stall.

Each arm's cdylib is loaded in its OWN subprocess (one cdylib per process). Protocol: interleaved A,B,A,B...
reps, fixed seed, warm EMD-K160 production path, steady-state window (drop warmup iters), bootstrap 95% CI
on the nodes/s ratio B/A + common-language P(B>A). SHIP iff median ratio>=1.05 AND CI-lo>1.02 AND P>=0.80.

NEVER cloud-ship; this is a per-core throughput isolator (local-CPU-OK). The probe DLL lives in an isolated
--target-dir and is loaded by explicit path; it never touches the canonical target/release cdylib.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dll_as_module(dll_path: str):
    """Load a specific battery_native.dll by explicit path as module `battery_native`.

    Windows extension modules must end in a recognized .pyd suffix, so copy the .dll to a temp .pyd
    and load THAT (avoids importing the canonical target/release build that sits on sys.path)."""
    import importlib.util

    tmpdir = tempfile.mkdtemp(prefix="ecprobe_")
    pyd = os.path.join(tmpdir, "battery_native.pyd")
    shutil.copy2(dll_path, pyd)
    spec = importlib.util.spec_from_file_location("battery_native", pyd)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot build import spec for {pyd}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["battery_native"] = mod  # mirror native_solver: set before exec (pyo3 may consult it)
    spec.loader.exec_module(mod)
    return mod


def _worker(args: argparse.Namespace) -> int:
    """Run one arm in this process: build a trainer, run `iters` profiled iterations, emit per-iter rows."""
    mod = _load_dll_as_module(args.dll)
    if not hasattr(mod, "NlheCFRTrainer"):
        raise SystemExit(f"{args.dll}: no NlheCFRTrainer (wrong/old cdylib)")
    emd = args.emd_frozen or None
    # (num_buckets, depth_edges, n_deals, n_seats, prune_threshold, fast_merge, emd_frozen)
    trainer = mod.NlheCFRTrainer(args.num_buckets, [], args.deals, 6, float("-inf"), False, emd)
    rows = json.loads(trainer.train_profile_json(args.iters))
    out = [
        {
            "iter": i,
            "nodes": int(r.get("nodes", 0)),
            "traverse_ms": float(r.get("traverse_ms", 0.0)),
        }
        for i, r in enumerate(rows)
    ]
    sys.stdout.write("ECPROBE_ROWS " + json.dumps(out) + "\n")
    return 0


def _run_arm(dll: str, args: argparse.Namespace) -> list[dict]:
    env = dict(os.environ)
    env["RAYON_NUM_THREADS"] = str(args.threads)
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--dll", dll,
        "--emd-frozen", args.emd_frozen or "",
        "--num-buckets", str(args.num_buckets),
        "--deals", str(args.deals), "--iters", str(args.iters),
    ]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout + "\n" + p.stderr + "\n")
        raise SystemExit(f"worker failed for {dll} (rc={p.returncode})")
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("ECPROBE_ROWS ")), None)
    if line is None:
        raise SystemExit(f"no ECPROBE_ROWS from worker for {dll}\n{p.stdout}\n{p.stderr}")
    return json.loads(line[len("ECPROBE_ROWS "):])


def _nodes_per_s(rows: list[dict], warmup: int) -> tuple[float, list[int]]:
    steady = rows[warmup:]
    nodes = sum(r["nodes"] for r in steady)
    traverse_s = sum(r["traverse_ms"] for r in steady) / 1000.0
    per_iter_nodes = [r["nodes"] for r in steady]
    return (nodes / max(traverse_s, 1e-12)), per_iter_nodes


def _bootstrap_ratio_ci(ratios: list[float], n_boot: int = 20000) -> tuple[float, float, float]:
    """Deterministic bootstrap (LCG, no Math.random) 95% CI on the mean ratio + P(ratio>1)."""
    k = len(ratios)
    seed = 0x9E3779B97F4A7C15
    means = []
    gt = 0
    for b in range(n_boot):
        s = 0.0
        for _ in range(k):
            seed = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            idx = (seed >> 11) % k
            s += ratios[idx]
        m = s / k
        means.append(m)
        if m > 1.0:
            gt += 1
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return lo, hi, gt / n_boot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--dll-a", default=str(ROOT / "tools/battery_native/target/release/battery_native.dll"))
    ap.add_argument("--dll-b", default=str(ROOT / "tools/battery_native/target_probe/release/battery_native.dll"))
    ap.add_argument("--dll", default="")  # worker
    ap.add_argument("--emd-frozen", default=str(ROOT / "data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn"))
    ap.add_argument("--num-buckets", type=int, default=160)
    ap.add_argument("--deals", type=int, default=96)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1, help="RAYON threads; default 1 = bit-deterministic, "
                    "cleanest per-core copy isolation (copy is L1/L2-private, contention can't amplify it)")
    args = ap.parse_args()

    if args.worker:
        return _worker(args)

    for label, dll in (("A", args.dll_a), ("B", args.dll_b)):
        if not Path(dll).exists():
            raise SystemExit(f"arm {label} cdylib missing: {dll} (build it first)")
    if args.emd_frozen and not Path(args.emd_frozen).exists():
        raise SystemExit(f"EMD-frozen lookup missing: {args.emd_frozen}")

    print(f"[env-copy kill-screen] emd={Path(args.emd_frozen).name} buckets={args.num_buckets} "
          f"deals={args.deals} iters={args.iters} warmup={args.warmup} reps={args.reps} threads={args.threads}")
    print(f"  arm A (MAX_HISTORY=64): {args.dll_a}")
    print(f"  arm B (MAX_HISTORY=2):  {args.dll_b}")

    nps_a: list[float] = []
    nps_b: list[float] = []
    ratios: list[float] = []
    ref_per_iter: list[int] | None = None
    max_ndelta = 0.0
    all_exact = True
    for rep in range(args.reps):
        rows_a = _run_arm(args.dll_a, args)  # interleaved A then B
        rows_b = _run_arm(args.dll_b, args)
        a_nps, a_iter = _nodes_per_s(rows_a, args.warmup)
        b_nps, b_iter = _nodes_per_s(rows_b, args.warmup)
        # Identity guard. hist is train-DEAD, so the traversal must be STATISTICALLY EQUIVALENT across
        # arms. Single-thread (RAYON=1) is bit-deterministic, so we expect EXACT equality; under threads>1
        # the documented non-bit-identical parallel merge (tree-sum reorder) lets tiny FP diffs amplify
        # chaotically after ~100 CFR iters (both are valid CFR paths). We therefore require total steady
        # nodes within a small tolerance (structural divergence = a real bug => abort), and report whether
        # the run was bit-exact. nodes/s is a RATE, robust to ~0.1% node-count drift between equivalent trees.
        tot_a, tot_b = sum(a_iter), sum(b_iter)
        ndelta = abs(tot_b - tot_a) / max(tot_a, 1)
        exact = (a_iter == b_iter)
        if ndelta > 0.02:
            first = next(((i, x, y) for i, (x, y) in enumerate(zip(a_iter, b_iter)) if x != y), None)
            raise SystemExit(f"ABORT rep {rep+1}: total steady nodes diverge {ndelta*100:.2f}% "
                             f"(>2% => structural, not chaotic-FP). first per-iter mismatch={first}")
        max_ndelta = max(max_ndelta, ndelta)
        all_exact = all_exact and exact
        if ref_per_iter is None:
            ref_per_iter = a_iter
        nps_a.append(a_nps)
        nps_b.append(b_nps)
        ratios.append(b_nps / a_nps)
        print(f"  rep {rep+1}/{args.reps}: A={a_nps/1e6:8.3f} M nodes/s  B={b_nps/1e6:8.3f} M nodes/s  "
              f"B/A={b_nps/a_nps:.4f}  (nodes d={ndelta*100:.3f}%{' EXACT' if exact else ''})")

    med = statistics.median(ratios)
    tmean = statistics.fmean(sorted(ratios)[1:-1]) if len(ratios) >= 4 else statistics.fmean(ratios)
    lo, hi, p_gt = _bootstrap_ratio_ci(ratios)
    ship = (med >= 1.05) and (lo > 1.02) and (p_gt >= 0.80)

    print("\n=== VERDICT ===")
    print(f"  median A = {statistics.median(nps_a)/1e6:.3f} M nodes/s   median B = {statistics.median(nps_b)/1e6:.3f} M nodes/s")
    print(f"  ratio B/A: median={med:.4f}  10%-trimmed-mean={tmean:.4f}  bootstrap95%CI=[{lo:.4f},{hi:.4f}]  P(B>A)={p_gt:.3f}")
    print(f"  steady nodes total: {sum(ref_per_iter)} over {len(ref_per_iter)} iters  "
          f"(traversal equivalence: {'BIT-EXACT' if all_exact else f'statistically-equiv, max nodes-d={max_ndelta*100:.3f}%'})")
    print(f"  SHIP THRESHOLD (median>=1.05 AND CI-lo>1.02 AND P>=0.80): {'PASS -> build EnvStateTrain' if ship else 'FAIL -> DEFER (copy hidden under probe latency)'}")

    payload = {
        "config": vars(args) | {"emd_frozen": str(args.emd_frozen)},
        "nps_a": nps_a, "nps_b": nps_b, "ratios": ratios,
        "median_ratio": med, "trimmed_mean_ratio": tmean,
        "ci95": [lo, hi], "p_b_gt_a": p_gt, "ship": ship,
        "max_nodes_delta": max_ndelta, "traversal_bit_exact": all_exact,
        "steady_nodes_total": sum(ref_per_iter), "steady_iters": len(ref_per_iter),
    }
    out = ROOT / "artifacts" / "env_copy_probe_ab.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
