#!/usr/bin/env python3
"""Honest strategic-quality re-measurement: ABSOLUTE per-archetype bb/100 for the DEPLOYED blueprint,
served RAW, vs each field archetype as a HOMOGENEOUS 5-handed table (CRN-paired seat-rotated duplicate).

The field-weighted combined number is ~95% responsive_station by construction, so it launders strategic
defects into "~0". This prints the per-archetype breakdown -- crucially the TOUGH TAIL (responsive_reg,
pro_balanced) -- so we can see whether the strategy actually beats regs/pros or only farms stations.

  POKERBOT_* env (deployed EMD-K160) set BELOW before importing pokerbot.
  python scripts/audit_deployed_vs_tail.py [decks=2000] [seed=0] [workers=10]

NOTE: heavy body is under `if __name__ == "__main__"` so the rotated-duplicate worker pool (Windows spawn
re-imports this module) does NOT re-run the measurement in every worker.
"""
import math
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EMD = os.path.join(REPO, "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn")
# Abstraction flags MUST be set before pokerbot import (baked at encoder import, incl. in each spawned
# worker's re-import). setdefault is idempotent + worker-safe.
for _k, _v in {
    "JAX_PLATFORMS": "cpu", "POKERBOT_BETTING_ABSTRACTION": "1", "POKERBOT_POSTFLOP_BUCKETS": "1",
    "POKERBOT_POSTFLOP_NUM_BUCKETS": "160", "POKERBOT_POSTFLOP_MODE": "emd",
    "POKERBOT_POSTFLOP_EMD_FROZEN": EMD, "POKERBOT_EMD_FROZEN_MIRROR_FALLBACK": "1",
    "POKERBOT_PREFLOP_LOSSLESS": "1", "POKERBOT_BP_COVERAGE": "1",
}.items():
    os.environ.setdefault(_k, _v)

import pickle  # noqa: E402

import numpy as np  # noqa: E402
from pokerbot.eval import blueprint_ranking as br  # noqa: E402


def run() -> int:
    decks = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    os.environ["POKERBOT_ROTATED_WORKERS"] = str(workers)

    dep = sys.argv[4] if len(sys.argv) > 4 else os.path.join(REPO, "checkpoints", "serve_blueprint_6max.pkl")
    raw = pickle.load(open(dep, "rb"))
    snap = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw
    snap = {bytes(k): np.asarray(v, dtype=np.float64) for k, v in snap.items()}
    cov = br.serve_coverage_probe(snap)
    print(f"blueprint={os.path.basename(dep)}: {len(snap):,} infosets  serve-coverage {((cov or 0)*100):.1f}% postflop  "
          f"decks={decks} seed={seed} workers={workers}", flush=True)
    if cov is not None and cov < 0.5:
        raise SystemExit(f"FATAL coverage {cov*100:.1f}% < 50% -- served uniform, numbers meaningless")

    roster = br.default_pool_roster()
    print(f"roster: {[(n, round(w,3)) for n,w in roster]}", flush=True)
    print(f"\n{'archetype':22s} {'weight':>7s} {'bb/100':>9s} {'+-95%CI':>9s} {'decks':>7s}", flush=True)
    comb = None
    for name, w in roster:
        a = np.asarray(br._rotated_seat_means(snap, villain_name=name, decks=decks, seed=seed),
                       dtype=np.float64)
        mean = float(a.mean()) * 100.0
        ci = 1.96 * float(a.std(ddof=1) / math.sqrt(a.size)) * 100.0 if a.size > 1 else 0.0
        flag = "BEATS" if (mean - ci) > 0 else ("LOSES" if (mean + ci) < 0 else "~0/noisy")
        print(f"{name:22s} {w:7.3f} {mean:+9.1f} {ci:9.1f} {a.size:7d}   {flag}", flush=True)
        comb = (w * a) if comb is None else comb + w * a
    cm = float(comb.mean()) * 100.0
    cci = 1.96 * float(comb.std(ddof=1) / math.sqrt(comb.size)) * 100.0
    print(f"{'FIELD-WEIGHTED':22s} {'1.000':>7s} {cm:+9.1f} {cci:9.1f} {comb.size:7d}", flush=True)
    print("\nNOTE: field-weighted is ~95% station by construction. Judge strategic quality by the TOUGH-"
          "TAIL rows (responsive_reg, pro_balanced), not the combined.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
