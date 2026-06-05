"""Solve and ship the 4-max AoF GTO base across the stack-depth grid (vectorized FP).

Uses aof_cfr.fp_4max_vec (the vectorized solver): one fixed real-card deal batch reused across
depths; showdown equities are depth-invariant (cached after the first depth), so only the cheap
per-depth payoff arithmetic + fast numpy FP rounds repeat. Records the independent sample-split
MC legal gap per depth, then ships cfr_base_4p.npz (infoset key v2) + manifest.

Run:  python scripts/solve_4max_base.py [--depths 8] [--deals 15000] [--rounds 400]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_cfr.exploit_mc import estimate_nash_conv_mc
from aof_cfr.export import export_manifest, export_policy_npz
from aof_cfr.fp_4max_vec import precompute_payoffs, sample_deals, solve_4max_fp_vec
from aof_cfr.infoset import STACK_DEPTH_BUCKETS
from aof_cfr.solve import SolveResult

CFR_DIR = Path("data/aof/cfr_base")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=str, default=",".join(str(d) for d in STACK_DEPTH_BUCKETS))
    ap.add_argument("--deals", type=int, default=15000)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--gate-samples", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",")]
    print(f"4-max base grid: depths={depths} deals={args.deals} rounds={args.rounds}")

    print("sampling deal batch (once) ...")
    buckets, holes = sample_deals(args.deals, args.seed)

    all_rows: dict[str, object] = {}
    depth_gap: dict[int, float] = {}
    for d in depths:
        t = time.monotonic()
        payoffs = precompute_payoffs(holes, float(d))  # showdown cache warm after 1st depth
        res = solve_4max_fp_vec(
            buckets=buckets, payoffs=payoffs, max_rounds=args.rounds,
            seed=args.seed, stack_bb=float(d),
        )
        mc, _imps, _m = estimate_nash_conv_mc(
            4, res["rows"], samples=args.gate_samples, seed=args.seed + 7, stack_bb=float(d)
        )
        depth_gap[d] = mc * 1000.0
        all_rows.update(res["rows"])
        print(f"  4max d{d}: independent legal gap {mc*1000:.1f} mbb/g "
              f"(vec internal {res['nash_conv_bb']*1000:.0f}) ({time.monotonic()-t:.0f}s)")

    result = SolveResult(
        num_players=4, solver_name="fictitious_play_4max_vec", iterations=args.rounds,
        seed=args.seed, exploitability_bb_per_hand=depth_gap.get(8, 0.0) / 1000.0,
        exploitability_mbb_g=depth_gap.get(8, 0.0), br_method="mc_combo_br_legal_split",
        player_improvements=None, policy_rows=all_rows,
        nash_disclaimer="N=4 vectorized legal FP; per-depth independent MC gap in manifest.depth_grid_4max.",
    )
    export_policy_npz(CFR_DIR / "cfr_base_4p.npz", result)
    export_manifest(
        CFR_DIR / "cfr_base.manifest.json", [result], merge_existing=True,
        extra={"depth_grid_4max": {str(d): v for d, v in depth_gap.items()},
               "fp_4max_vec": {"deals": args.deals, "rounds": args.rounds, "seed": args.seed}},
    )
    print(f"\nwrote cfr_base_4p.npz ({len(all_rows)} infosets across {len(depths)} depths) + manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
