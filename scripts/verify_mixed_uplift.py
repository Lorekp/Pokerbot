#!/usr/bin/env python3
"""High-N re-verification of the EXPLOIT-OVERLAY net value on MIXED (heterogeneous) calibrated fields.

Background: the only prior mixed-field uplift numbers (runs/skill_ladder_*.json) were N=400, on HS-bot
CARICATURE fields (not the IRC-calibrated real_mixed_*), and within ~1 sigma of zero. The rotated
duplicate harness can't measure mixed fields (it needs a homogeneous field), so the overlay's value on a
realistic mixed table has never been measured at scale. This driver fixes that.

It shards ``run_field_absolute`` (the CRN-paired on/off path: ``--field`` WITHOUT ``--rotate-seats``,
``--per-seat-profiles``) across (field x seed) processes and pools the result. The CRN uplift
(``uplift_bb_per_hand`` = chips_on - chips_off on a SHARED deck) is trustworthy even though AIVAT's
absolute adj-mean is broken on the search path -- a hand the overlay doesn't change cancels exactly, so
the paired uplift has tiny variance. Absolute base/exploit are reported from the RAW sample means
(``raw_mean_bb``), NOT the AIVAT adj-mean (which has negative variance reduction here).

Run:  python scripts/verify_mixed_uplift.py --fields real_mixed_soft,real_mixed_tough,real_pro_table,real_recreational_table
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SNAP = r"checkpoints/bp6max_lossless2_iter200.pkl"


def _pool(rows):
    """Pool independent (n, mean, std) estimates -> (N, pooled_mean, pooled_se)."""
    n_tot = sum(n for n, _, _ in rows)
    if n_tot == 0:
        return 0, 0.0, 0.0
    mean = sum(n * m for n, m, _ in rows) / n_tot
    ss = sum((n - 1) * s * s for n, _, s in rows if n > 1)
    ss += sum(n * (m - mean) ** 2 for n, m, _ in rows)
    var = ss / (n_tot - 1) if n_tot > 1 else 0.0
    return n_tot, mean, math.sqrt(var / n_tot)


def _run_one(field: str, seed: int, hands: int, iters: int, snap: str, out: Path) -> tuple[str, int, int]:
    env = dict(os.environ)
    env["POKERBOT_PREFLOP_LOSSLESS"] = "1"
    env["POKERBOT_SEARCH_PREFLOP"] = "1"
    env["POKERBOT_NATIVE_SUBGAME"] = "1"
    env["POKERBOT_NATIVE_ROLLOUT"] = "1"
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    cmd = [sys.executable, str(_ROOT / "scripts" / "eval_exploitation_battery.py"),
           "--hero", "pluribus", "--blueprint-snapshot", snap,
           "--field", field, "--per-seat-profiles", "--allin-ev",
           "--iterations", str(iters), "--seed", str(seed), "--hands", str(hands),
           "--no-aivat-leafq", "--out", str(out)]
    p = subprocess.run(cmd, env=env, cwd=str(_ROOT), capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(f"[shard {field} s{seed}] rc={p.returncode}\n{p.stderr[-800:]}\n")
    return field, seed, p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="real_mixed_soft,real_mixed_tough,real_pro_table,real_recreational_table")
    ap.add_argument("--shards-per-field", type=int, default=12, help="independent seeds per field")
    ap.add_argument("--hands", type=int, default=1500, help="hands per shard")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--snapshot", default=_SNAP)
    ap.add_argument("--outdir", default="runs/mixed_uplift")
    args = ap.parse_args()

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = [(f, s, outdir / f"{f}_s{s}.json")
            for f in fields for s in range(args.shards_per_field)]
    print(f"launching {len(jobs)} shards ({len(fields)} fields x {args.shards_per_field} seeds x "
          f"{args.hands} hands) on {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, f, s, args.hands, args.iters, args.snapshot, o): (f, s)
                for f, s, o in jobs}
        done = 0
        for fut in as_completed(futs):
            f, s, rc = fut.result()
            done += 1
            print(f"  [{done}/{len(jobs)}] {f} s{s} rc={rc}", flush=True)

    print("\n=== EXPLOIT-OVERLAY net value (CRN-paired uplift), pooled ===", flush=True)
    hdr = (f"{'field':28s} {'base(raw)':>11s} {'exploit(raw)':>13s} "
           f"{'UPLIFT':>9s} {'95% CI':>20s} {'hands':>7s} {'zero%':>6s}")
    print(hdr, flush=True)
    summary = {}
    for f in fields:
        up_rows, off_rows, on_rows = [], [], []
        zero_fracs = []
        for s in range(args.shards_per_field):
            jp = outdir / f"{f}_s{s}.json"
            if not jp.exists():
                continue
            try:
                r = json.loads(jp.read_text(encoding="utf-8")).get("results", {}).get(f)
            except Exception:  # noqa: BLE001
                continue
            if not r:
                continue
            n = int(r.get("hands", args.hands))
            up_m = float(r["uplift_bb_per_hand"])
            up_se = float(r["uplift_se"])
            up_rows.append((n, up_m, up_se * math.sqrt(n)))           # se -> std
            off = r.get("off_aivat", {}); on = r.get("on_aivat", {})
            if "raw_mean_bb" in off:
                off_rows.append((n, float(off["raw_mean_bb"]), float(off.get("raw_std", 0.0))))
            if "raw_mean_bb" in on:
                on_rows.append((n, float(on["raw_mean_bb"]), float(on.get("raw_std", 0.0))))
            zero_fracs.append(float(r.get("uplift_zero_frac", 0.0)))
        if not up_rows:
            print(f"{f:28s}  (no results)", flush=True)
            continue
        nN, um, use = _pool(up_rows)
        _, om, _ = _pool(off_rows) if off_rows else (0, 0.0, 0.0)
        _, onm, _ = _pool(on_rows) if on_rows else (0, 0.0, 0.0)
        lo, hi = um - 1.96 * use, um + 1.96 * use
        verdict = "+EV" if lo > 0 else ("-EV" if hi < 0 else "~0")
        zf = sum(zero_fracs) / len(zero_fracs) if zero_fracs else 0.0
        print(f"{f:28s} {om:>+11.2f} {onm:>+13.2f} {um:>+9.3f} "
              f"[{lo:>+8.3f},{hi:>+8.3f}] {nN:>7d} {100 * zf:>5.0f}%  {verdict}", flush=True)
        summary[f] = {"hands": nN, "uplift_bb_per_hand": um, "uplift_se": use,
                      "uplift_ci95": [lo, hi], "uplift_bb100": 100 * um,
                      "base_raw_bb_per_hand": om, "exploit_raw_bb_per_hand": onm,
                      "verdict": verdict}
    (outdir / "pooled_uplift.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nwrote {outdir / 'pooled_uplift.json'}", flush=True)
    print("UPLIFT is bb/HAND (x100 for bb/100). +EV/-EV = 95% CI excludes 0; ~0 = straddles.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
