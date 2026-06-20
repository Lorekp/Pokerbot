"""In-depth native validation of the HU neural-CFR net -- LBR exploitability across a DEPTH SWEEP.

The Python HU-LBR could only afford ~150 hands (+1765 +/-5196 mbb/g -- noise), and only at the engine
default 100bb, which is the net's LEAST-trained depth. This CLI runs the fully-native, rayon-parallel HU
LBR (`pokerbot/eval/lbr_native.py`, neural policy ported to Rust + golden-cross-checked) and sweeps the
stack depth, so we finally measure exploitability in the SHALLOW tournament regime the net actually plays
(final-2 of an MTT = high blinds = ~5-50bb), with tight CIs.

Run (the checkpoint stamps POKERBOT_INFO_SUITS; the native serve is env-independent so no flag needed):
  python scripts/neural_cfr/validate_hu_neural_native.py \
    --checkpoint checkpoints/sd_cfr_hu_neural/hu_vrpdcfr_v2.pkl --hands 4000 --seeds 6
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from pokerbot.eval.lbr import LBRConfig
from pokerbot.eval import lbr_native as L


def pooled_mbb(checkpoint, depth, ante, hands, seeds):
    vals = []
    for s in range(seeds):
        r = L.lbr_hu_neural_native(
            checkpoint, LBRConfig(hands=hands, seed=s, stack_bb=float(depth), ante_bb=float(ante)))
        vals.append(r.mbb_per_g)
    v = np.array(vals)
    se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return v.mean(), se, vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/sd_cfr_hu_neural/hu_vrpdcfr_v2.pkl")
    ap.add_argument("--hands", type=int, default=4000, help="hands per seed per depth")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--depths", default="5,10,20,50,100", help="comma list of stack depths (BB)")
    ap.add_argument("--ante-bb", type=float, default=0.125, help="per-player ante (tournament structure)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if not L.available():
        print("native battery_native not built (see lbr_native.py build hint).")
        return 1

    _, meta = L.build_neural_harness(args.checkpoint)
    print(f"checkpoint: {args.checkpoint}")
    print(f"  hidden_dim={meta['hidden_dim']} info_suits={meta['info_suits']} iteration={meta['iteration']}")
    print(f"\n--- HU neural LBR depth sweep (ante {args.ante_bb}bb; "
          f"{args.seeds} seeds x {args.hands} = {args.seeds*args.hands:,} hands each) ---")
    print("  reference: always-fold ~ +750 mbb/g; positive = exploitable lower bound; LOWER = stronger.")
    depths = [int(x) for x in args.depths.split(",")]
    rows = []
    t0 = time.time()
    for depth in depths:
        mean, se, vals = pooled_mbb(args.checkpoint, depth, args.ante_bb, args.hands, args.seeds)
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
        print(f"  {depth:3d}bb: {mean:+7.0f} mbb/g   95% CI [{lo:+.0f}, {hi:+.0f}]")
        rows.append({"depth_bb": depth, "mbb_per_g": mean, "ci95": [lo, hi], "per_seed": vals})
    print(f"\n  wall = {time.time()-t0:.1f}s")
    print("  read: exploitability typically RISES with depth (the net is best-trained shallow, where it "
          "plays); the shallow rows are the tournament-relevant number.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "meta": meta, "ante_bb": args.ante_bb,
                       "hands_per_seed": args.hands, "seeds": args.seeds, "sweep": rows}, f, indent=2)
        print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
