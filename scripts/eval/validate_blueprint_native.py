"""In-depth native validation of a Pluribus 6-max blueprint -- LBR6 exploitability at high throughput.

The pure-Python `lbr6_exploitability` is GIL-bound at ~1-2 hands/s, so the published fc-floor came back
+134 bb/100 with CI [-40, +308] (1500 hands) -- directional only. This CLI runs the fully-native,
rayon-parallel LBR (`pokerbot/eval/lbr_native.py`) at ~100x throughput, so we can afford the hands to
tighten the CI by ~10x and get a verdict on whether the blueprint is actually exploitable.

It prints the calibration self-checks FIRST (always-fold must hit ~ +128 bb/100; station must blow up),
so a miscalibrated harness is caught before any real number is trusted. Then it pools the blueprint LBR6
over several seeds for a tight estimate + per-seat breakdown.

Run:
  POKERBOT_PREFLOP_LOSSLESS=1 python scripts/eval/validate_blueprint_native.py \
    --checkpoint C:/path/to/blueprint.pkl --hands 8000 --seeds 12
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")

import numpy as np  # noqa: E402

from pokerbot.eval.lbr6 import LBR6Config  # noqa: E402
from pokerbot.eval import lbr_native as L  # noqa: E402


def load_blueprint(path: str):
    """Return (snapshot dict[bytes->ndarray], depth_edges list). Accepts a trainer checkpoint (with
    `strat_sum`) or a raw snapshot dict."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "strat_sum" in obj:
        snap = {bytes(k): np.asarray(v, dtype=np.float64) for k, v in obj["strat_sum"].items()}
        edges = list(obj.get("abstraction", {}).get("depth_edges", []) or [])
        return snap, edges
    # raw snapshot (bytes -> array)
    snap = {bytes(k): np.asarray(v, dtype=np.float64) for k, v in obj.items()}
    return snap, []


def pooled(fn, seeds):
    vals, t0 = [], time.time()
    for s in seeds:
        r = fn(s)
        vals.append(r.bb_per_100)
    v = np.array(vals)
    se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return v.mean(), se, time.time() - t0, vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="blueprint checkpoint (.pkl) or raw snapshot")
    ap.add_argument("--hands", type=int, default=8000, help="hands per seed")
    ap.add_argument("--seeds", type=int, default=12, help="number of seeds to pool")
    ap.add_argument("--n-seats", type=int, default=6)
    ap.add_argument("--value-only", action="store_true",
                    help="value-only raise valuation (no fold-equity credit; lower variance, conservative)")
    ap.add_argument("--no-calibrate", action="store_true", help="skip the fold/station self-checks")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if not L.available():
        print("native battery_native not built. Build with:\n"
              "  PYO3_PYTHON=$(python -c 'import sys;print(sys.executable)') \\\n"
              "  cargo build --release --features python --lib --manifest-path tools/battery_native/Cargo.toml")
        return 1

    snap, edges = load_blueprint(args.checkpoint)
    print(f"blueprint: {len(snap):,} infosets  depth_edges={edges or 'none'}")

    # CRN reseed per replication (SCRIPTS-EVAL-AUDIT HIGH). The previous
    # ``seeds = list(range(args.seeds))`` lets the table_id + deck sequence
    # COLLIDE across replications (same shape as the eval_emd_vs_iter62000
    # hypothesis B bug — the k500 verdict was produced by exactly this
    # pattern). Reuse the documented XOR-with-base-seed pattern from
    # ``scripts/eval/eval_emd_vs_iter62000.py:90-109`` so all readers agree
    # on the stream. We duplicate the 8 lines here rather than import
    # because that module is heavy (it pulls JAX / battery_native) and
    # validate_blueprint_native is a smoke-gate.
    BASE_SEED = 0x5EEDBAB0  # mirror scripts/eval/eval_emd_vs_iter62000.py
    seeds = [BASE_SEED ^ i for i in range(args.seeds)]
    # CRN-collision assertion — belt-and-suspenders (the XOR trivially
    # gives N distinct non-zero values; asserting it forces a runtime
    # check the day someone changes BASE_SEED to 0 or breaks the XOR).
    assert len(set(seeds)) == args.seeds, f"CRN reseed produced collisions: {seeds}"

    def cfg(seed):
        return LBR6Config(hands=args.hands, seed=seed, n_seats=args.n_seats,
                          value_only_raises=args.value_only)

    if not args.no_calibrate:
        # The +128 always-fold anchor is defined for the FULL LBR (fold-equity on), independent of the
        # --value-only flag used for the real measurement -- so calibrate with the full config.
        calib = LBR6Config(hands=args.hands, seed=0, n_seats=args.n_seats)
        print("\n--- calibration self-checks (harness must hit these before the real number is trusted) ---")
        rf = L.lbr6_exploitability_native(None, calib, policy="fold")
        print(f"  always-fold : {rf.bb_per_100:+7.1f} bb/100  (anchor ~ +128; {'OK' if 110 <= rf.bb_per_100 <= 145 else 'OFF!'})")
        rs = L.lbr6_exploitability_native(snap, calib, policy="nit")
        print(f"  nit         : {rs.bb_per_100:+7.1f} bb/100  (a tight value bot; sanity only)")

    print(f"\n--- blueprint LBR6 ({args.seeds} seeds x {args.hands} = {args.seeds*args.hands:,} hands"
          f"{', value-only' if args.value_only else ''}) ---")
    mean, se, wall, vals = pooled(
        lambda s: L.lbr6_exploitability_native(snap, cfg(s), policy="blueprint", depth_edges=edges), seeds)
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    verdict = ("EXPLOITABLE (CI > 0)" if lo > 0 else
               "NOT provably exploitable (CI crosses 0)" if hi > 0 else "robust (CI <= 0)")
    print(f"  exploitability = {mean:+.1f} bb/100   95% CI [{lo:+.1f}, {hi:+.1f}]")
    print(f"  per-seed: {[round(x,0) for x in vals]}")
    print(f"  wall = {wall:.1f}s  ({args.seeds*args.hands/max(wall,1e-9):.0f} hands/s)")
    print(f"  VERDICT: {verdict}")

    # per-seat (from the last seed's breakdown -- the structure, not a pooled CI)
    last = L.lbr6_exploitability_native(snap, cfg(seeds[-1]), policy="blueprint", depth_edges=edges)
    print("  per-seat (last seed): " +
          ", ".join(f"s{k}={v['bb_per_100']:+.0f}" for k, v in sorted(last.detail["per_seat"].items())))

    if args.output:
        # Atomic JSON write (SCRIPTS-EVAL-AUDIT CRIT) — torn-tail-safe.
        # Import lazily so a smoke run without --output doesn't pay the cost.
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
        from _atomic_io import write_json_atomic  # noqa: PLC0415
        write_json_atomic(args.output, {
            "infosets": len(snap), "depth_edges": edges, "hands_per_seed": args.hands,
            "seeds": args.seeds, "value_only": args.value_only,
            "bb_per_100": mean, "ci95": [lo, hi], "per_seed": vals,
            "verdict": verdict, "wall_s": wall,
        }, indent=2)
        print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
