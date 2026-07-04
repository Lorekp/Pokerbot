#!/usr/bin/env python3
"""Run ONLY the two dyn_k200 arms (honest and mixed) with sufficient timeout.

These are the Python dynamic exploit tail arms at K=200.
At ~0.24 h/s for 1000 hands each, these take ~70 min per arm.
We set a 120-min timeout (generous), running both sequentially.

Writes to:
  artifacts/native_exploit_value_ab/honest_dyn_k200.json
  artifacts/native_exploit_value_ab/mixed_dyn_k200.json

And appends/updates the ab_summary.json with corrected dyn results.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "artifacts" / "native_exploit_value_ab"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT = str(REPO / "scripts" / "adaptive_field_h2h.py")

HANDS = 1000
SEED = 42
WINDOWS = 10

TIMEOUT_SECS = 120 * 60  # 2-hour hard cap (dyn_k200 at ~0.24 h/s = ~70 min for 1000 hands)

FIELDS = [
    ("honest", '{"honest": 3}'),
    ("mixed",  '{"honest": 2, "fish": 1, "random": 1}'),
]


def build_env() -> dict:
    env = os.environ.copy()
    env["POKERBOT_LIVE_NATIVE_EXPLOIT"] = "0"
    env["ADAPTIVE_PROD_K"] = "200"
    env.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
    env.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
    env.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
    env.setdefault("POKERBOT_POSTFLOP_MODE", "emd")
    env.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "160")
    emdfrzn = str(REPO / "data" / "abstraction" / "emd_frozen_flop_turn_160_r400.emdfrzn")
    env.setdefault("POKERBOT_POSTFLOP_EMD_FROZEN", emdfrzn)
    env.setdefault("POKERBOT_EMD_FROZEN_MIRROR_FALLBACK", "1")
    env.setdefault("POKERBOT_NATIVE_SUBGAME", "1")
    env.setdefault("JAX_PLATFORMS", "cpu")
    return env


def run_arm(field_label: str, field_json: str) -> dict:
    out_json = OUT_DIR / f"{field_label}_dyn_k200.json"
    run_label = f"{field_label}/dyn_k200"

    cmd = [
        sys.executable, "-u", SCRIPT,
        "--field", field_json,
        "--hands", str(HANDS),
        "--seed", str(SEED),
        "--exploit", "on",
        "--windows", str(WINDOWS),
        "--hero", "production",
        "--diag",
        "--out", str(out_json),
        "--progress-every", "25",
    ]

    env = build_env()

    print(f"\n{'='*70}", flush=True)
    print(f"[DYN_K200] STARTING: {run_label}", flush=True)
    print(f"[DYN_K200]   POKERBOT_LIVE_NATIVE_EXPLOIT=0  ADAPTIVE_PROD_K=200", flush=True)
    print(f"[DYN_K200]   field={field_json}  hands={HANDS}  seed={SEED}", flush=True)
    print(f"[DYN_K200]   timeout={TIMEOUT_SECS/60:.0f} min  output -> {out_json}", flush=True)
    print(f"{'='*70}\n", flush=True)

    t0 = time.perf_counter()
    last_heartbeat = t0

    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    native_active_confirmed = False

    while True:
        line = proc.stdout.readline()
        if line == "" and proc.poll() is not None:
            break
        if line:
            line = line.rstrip("\n")
            print(f"  [{run_label}] {line}", flush=True)
            if "NATIVE EXPLOIT serve ACTIVE" in line:
                native_active_confirmed = True
            now = time.perf_counter()
            elapsed = now - t0
            if now - last_heartbeat > 120:
                last_heartbeat = now
                print(f"[DYN_K200] heartbeat: {run_label} — {elapsed/60:.1f} min elapsed", flush=True)
            if elapsed > TIMEOUT_SECS:
                print(f"[DYN_K200] TIMEOUT exceeded for {run_label}. Killing.", flush=True)
                proc.kill()
                break

    rc = proc.wait()
    wall_s = time.perf_counter() - t0
    print(f"\n[DYN_K200] {run_label} done. rc={rc}  wall={wall_s/60:.1f} min", flush=True)

    result = {
        "run_label": run_label,
        "field_label": field_label,
        "arm_label": "dyn_k200",
        "field_json": field_json,
        "native_exploit_env": "0",
        "prod_k": "200",
        "out_json": str(out_json),
        "returncode": rc,
        "wall_s": round(wall_s, 1),
        "native_active_confirmed": native_active_confirmed,
        "anomaly": None,
    }

    if rc != 0:
        result["anomaly"] = f"non-zero returncode={rc}"

    if out_json.exists():
        with open(out_json) as fh:
            data = json.load(fh)
        ov = data.get("overall", {})
        result["bb_per_100"]   = round(ov.get("bb_per_100", 0.0), 1)
        result["ci95_lo"]      = round(ov.get("ci95_lo", 0.0) * 100, 1)
        result["ci95_hi"]      = round(ov.get("ci95_hi", 0.0) * 100, 1)
        result["hands_per_s"]  = data.get("hands_per_second", 0.0)
        result["served_source"] = data.get("served_source", {})
        result["diag_loss"]    = data.get("diag_loss_attribution", {})
        result["diag_exploit"] = data.get("diag_exploit", {})
    else:
        result["anomaly"] = (result["anomaly"] or "") + "; output JSON missing"
        result["bb_per_100"] = result["ci95_lo"] = result["ci95_hi"] = result["hands_per_s"] = None

    return result


def main():
    print(f"\n{'#'*70}", flush=True)
    print("# DYN_K200 ARMS — corrected runner (120-min timeout each)", flush=True)
    print(f"# Hands: {HANDS}  Seed: {SEED}", flush=True)
    print(f"{'#'*70}\n", flush=True)

    results = []
    for field_label, field_json in FIELDS:
        r = run_arm(field_label, field_json)
        results.append(r)
        print(f"\n[DYN_K200] Arm {r['run_label']}: bb/100={r.get('bb_per_100')} "
              f"CI95=[{r.get('ci95_lo')},{r.get('ci95_hi')}] "
              f"h/s={r.get('hands_per_s')} anomaly={r.get('anomaly')}", flush=True)

    # Merge into existing ab_summary.json if it exists
    summary_path = OUT_DIR / "ab_summary.json"
    if summary_path.exists():
        with open(summary_path) as fh:
            existing = json.load(fh)
        # Replace any existing dyn_k200 entries
        existing = [r for r in existing if r.get("arm_label") != "dyn_k200"]
        existing.extend(results)
        with open(summary_path, "w") as fh:
            json.dump(existing, fh, indent=2)
        print(f"\n[DYN_K200] Updated {summary_path} with corrected dyn_k200 results.", flush=True)
    else:
        # Write standalone summary
        with open(summary_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n[DYN_K200] Wrote {summary_path}", flush=True)

    for r in results:
        print(f"\n[DYN_K200] Final: {r['run_label']}", flush=True)
        print(f"  bb/100={r.get('bb_per_100')}  CI95=[{r.get('ci95_lo')},{r.get('ci95_hi')}]", flush=True)
        print(f"  h/s={r.get('hands_per_s')}  anomaly={r.get('anomaly')}", flush=True)


if __name__ == "__main__":
    main()
