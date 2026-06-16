"""Local overnight driver for the 6-max LBR vs blueprint+SEARCH exploitability measurement.

Runs the SAME per-cell adversary as the cloud path (scripts/lbr6_audit.py --target blueprint-search) but
on this machine, fanned across a capped process pool, then pools the independent seeds into ONE tight CI
per menu (+ per-seat) exactly like the cloud SUMMARY. Use when the cloud box is unavailable / oversubscribed.

WHY THIS EXISTS / the oversubscription fix: each lbr6_audit subprocess imports JAX, which spins its OWN
XLA CPU thread pool regardless of OMP_NUM_THREADS. Many such subprocesses then fight for cores and every
cell crawls. This runner forces every child SINGLE-THREADED (OMP/OPENBLAS/MKL/NUMEXPR + XLA eigen off)
so N workers really means N cores, not N*cores threads.

  python scripts/run_local_lbr6_search.py                       # 3 menus x 20 seeds x 700 hands, ~10 workers
  python scripts/run_local_lbr6_search.py --hands 500 --seeds 12 --workers 8
  python scripts/run_local_lbr6_search.py --menus fc            # fc floor only (the clean headline)

Resumable: a cell whose JSON already exists (non-empty) is skipped, so a crash/Ctrl-C just resumes.
Pools at the end into <out-dir>/SUMMARY.json and prints it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BP = r"checkpoints/bp6max_dcfrvr_iter200.pkl"
MENUS = ("fc", "fcpa", "full")
CKPT = 4   # re-pool + atomically rewrite SUMMARY.json every CKPT completed cells (rolling checkpoint)


def _child_env() -> dict:
    """Single-thread EVERY pool a child might spawn (the anti-oversubscription contract)."""
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", NPY_NUM_THREADS="1", JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_cpu_multi_thread_eigen=false "
                  "--xla_force_host_platform_device_count=1",
    )
    return env


def _pool_stats(triples):
    """Pool (mean_bb_per_hand, std_bb_per_hand, n) -> bb/100 + 95% CI via total-variance decomposition."""
    triples = [t for t in triples if t[2]]
    if not triples:
        return None
    N = sum(n for _, _, n in triples)
    M = sum(m * n for m, _, n in triples) / N
    ss = sum(n * (sd ** 2) + n * (m - M) ** 2 for m, sd, n in triples)
    stderr = math.sqrt((ss / N) / N)
    return {"bb_per_100": round(M * 100, 2), "ci_lo": round((M - 1.96 * stderr) * 100, 2),
            "ci_hi": round((M + 1.96 * stderr) * 100, 2), "stderr_bb_per_100": round(stderr * 100, 2),
            "total_hands": N, "k_seeds": len(triples)}


def _pool_menu(out_dir: Path, menu: str):
    cells = []
    for f in out_dir.glob(f"lbr6_{menu}_s*.json"):
        try:
            cells.append(json.loads(f.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            continue
    cells = [c for c in cells if c.get("hands")]
    if not cells:
        return None
    overall = _pool_stats([(c["ev_bb_per_hand"], c["std_bb_per_hand"], c["hands"]) for c in cells])
    per_seat = {}
    for s in range(6):
        trip = [(ps["bb_per_100"] / 100.0, ps.get("std_bb_per_hand", 0.0), ps["n"])
                for c in cells for sk, ps in (c.get("detail", {}).get("per_seat", {}) or {}).items()
                if int(sk) == s]
        ps_pooled = _pool_stats(trip)
        if ps_pooled:
            per_seat[s] = {k: ps_pooled[k] for k in ("bb_per_100", "ci_lo", "ci_hi")}
    return {**overall, "per_seat": per_seat}


def _read_one(out_dir: Path, fn: str):
    f = out_dir / fn
    if not f.exists():
        return None
    try:
        r = json.loads(f.read_text(encoding="utf-8"))
        return {"bb_per_100": round(r["bb_per_100"], 1), "hands": r["hands"]}
    except (ValueError, OSError, KeyError):
        return None


def _build_summary(out_dir: Path, menus, args, elapsed_min: float, partial: bool) -> dict:
    """Pool whatever cells have landed -> the SUMMARY dict (cloud-compatible shape). Safe to call mid-run:
    pools only the JSONs present so far, so it is a true rolling checkpoint."""
    pooled = {m: _pool_menu(out_dir, m) for m in menus}
    pooled = {m: v for m, v in pooled.items() if v}
    cal = {t: _read_one(out_dir, f"lbr6_cal_{t}_s0.json") for t in ("foldbot", "station")}
    cal_ok = all(c and c["bb_per_100"] > 50 for c in cal.values())
    beatable = any(v["ci_lo"] > 0 for v in pooled.values())
    worst = max((v["bb_per_100"] for v in pooled.values()), default=0.0)
    if not pooled:
        verdict = "PENDING: no menu cells pooled yet"
    elif not cal_ok and not partial:
        verdict = "CALIBRATION-FAILED (adversary not detecting leaks)"
    elif beatable:
        verdict = (f"BEATABLE: value best-responder profits vs blueprint+search by up to {worst:+.1f} "
                   f"bb/100 (a menu CI excludes 0)")
    else:
        verdict = "ROBUST at this CI: no menu's value best-responder proves a positive exploit of blueprint+search"
    return {"mode": "lbr6_local", "target": "blueprint-search", "method": "analytic_1ply_lbr",
            "value_only_raises": True, "partial": partial, "search": {"leaf": args.leaf, "iters": args.iters},
            "hands": args.hands, "seeds": args.seeds, "menus": pooled, "calibration": cal,
            "verdict": verdict, "elapsed_min": round(elapsed_min, 1)}


def _write_summary(out_dir: Path, menus, args, t0: float, partial: bool) -> dict:
    summary = _build_summary(out_dir, menus, args, (time.time() - t0) / 60.0, partial)
    tmp = out_dir / "SUMMARY.json.tmp"
    tmp.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    tmp.replace(out_dir / "SUMMARY.json")   # atomic checkpoint (never a half-written file on disk)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--hands", type=int, default=700)
    ap.add_argument("--iters", type=int, default=10, help="fixed CFR+ iters per opponent search solve")
    ap.add_argument("--leaf", choices=("uniform", "gadget"), default="uniform")
    ap.add_argument("--menus", default="fc,fcpa,full", help="comma list subset of fc,fcpa,full")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--bp", default=DEFAULT_BP)
    ap.add_argument("--out-dir", default=str(REPO / "runs" / "local_search"))
    args = ap.parse_args()

    menus = tuple(m for m in args.menus.split(",") if m in MENUS)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _child_env()

    # Job list: 2 fast calibration cells (raw blueprint foldbot/station -> must be >>0) + the search cells.
    jobs = []
    for tgt in ("foldbot", "station"):
        out = out_dir / f"lbr6_cal_{tgt}_s0.json"
        jobs.append((f"cal_{tgt}", [sys.executable, "scripts/lbr6_audit.py", "--target", tgt,
                     "--menu", "full", "--hands", str(min(args.hands, 400)), "--seed", "0",
                     "--unpaired", "--out", str(out)], out))
    for menu in menus:
        for seed in range(args.seeds):
            out = out_dir / f"lbr6_{menu}_s{seed}.json"
            jobs.append((f"{menu}_s{seed}", [sys.executable, "scripts/lbr6_audit.py",
                         "--target", "blueprint-search", "--blueprint-snapshot", args.bp,
                         "--menu", menu, "--hands", str(args.hands), "--seed", str(seed),
                         "--search-iters", str(args.iters), "--search-leaf", args.leaf,
                         "--out", str(out)], out))

    def _done(o: Path) -> bool:
        # A cell counts as done only if its JSON parses AND carries hands -- a cell truncated by an
        # interruption (size>0 but unparseable) is re-run, not silently skipped by both resume and pooling.
        if not (o.exists() and o.stat().st_size > 0):
            return False
        try:
            return bool(json.loads(o.read_text(encoding="utf-8")).get("hands"))
        except (ValueError, OSError):
            return False

    todo = [(n, c, o) for (n, c, o) in jobs if not _done(o)]
    print(f"local lbr6 vs BP+search: {len(jobs)} cells ({len(jobs) - len(todo)} already done), "
          f"{args.workers} workers, {args.hands} hands x {args.seeds} seeds, menus={menus}", flush=True)
    print(f"  out-dir: {out_dir}", flush=True)
    print(f"  rolling checkpoint -> {out_dir / 'SUMMARY.json'} (refreshed every {CKPT} cells)", flush=True)
    t0 = time.time()
    done = [0]
    lock = threading.Lock()

    def run(job):
        name, cmd, out = job
        log = out.with_suffix(".log")
        with open(log, "w", encoding="utf-8") as lf:
            rc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=lf,
                                stderr=subprocess.STDOUT).returncode
        ok = out.exists() and out.stat().st_size > 0
        with lock:
            done[0] += 1
            n = done[0]
            el = (time.time() - t0) / 60.0
            print(f"  [{n}/{len(todo)}] {name:14s} rc={rc} {'OK' if ok else 'FAIL'}  "
                  f"({el:.1f} min elapsed)", flush=True)
            if n % CKPT == 0:   # rolling checkpoint: re-pool + atomically rewrite SUMMARY.json
                s = _write_summary(out_dir, menus, args, t0, partial=True)
                head = "  ".join(f"{m}={s['menus'][m]['bb_per_100']:+.0f}" for m in s["menus"])
                print(f"     [ckpt] {head}   ({el:.1f} min)" if head else "     [ckpt] (no menu cells yet)",
                      flush=True)
        return name, rc

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(run, todo))

    summary = _write_summary(out_dir, menus, args, t0, partial=False)   # final pooled SUMMARY
    print("\n=== LOCAL LBR6 vs BP+SEARCH SUMMARY ===\n" + json.dumps(summary, indent=1), flush=True)
    print(f"\nwrote {out_dir / 'SUMMARY.json'}", flush=True)


if __name__ == "__main__":
    main()
