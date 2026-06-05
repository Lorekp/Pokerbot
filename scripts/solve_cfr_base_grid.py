"""Solve the AoF CFR base across the effective-stack-depth grid and ship v2 artifacts.

Reuses the depth-invariant equity (HU equity matrix / trio tensor): only the per-depth
payoff layer changes. Produces depth-keyed infoset-key-v2 policies and merges all depths
per player-count into ``cfr_base_{n}p.npz``. 4-max base is NOT solved here — a converged
4-max FP equilibrium is perf-blocked in pure Python (~133 s/round); the corrected 4-max
machinery is ready for a vectorized/C++ solve later.

Run:  python scripts/solve_cfr_base_grid.py [--depths 4,6,8,10,12,15,20] [--rounds 1000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_cfr.export import export_manifest, export_policy_npz
from aof_cfr.fp_br import StabilityTracker
from aof_cfr.hu_export import hu_policy_rows_from_freqs
from aof_cfr.hu_payoff_matrix import build_hu_bucket_equity_matrix
from aof_cfr.infoset import STACK_DEPTH_BUCKETS
from aof_cfr.solve import SolveResult, _bb_to_mbbg
from aof_cfr.trio_payoff_tensor import DEFAULT_TENSOR, load_trio_tensor
from pokerbot.solver.hu_preflop_dcfr import solve_hu_pushfold
from pokerbot.solver.trio_preflop_fp import solve_3max_fp, trio_fp_policy_rows

CFR_DIR = Path("data/aof/cfr_base")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=str, default=",".join(str(d) for d in STACK_DEPTH_BUCKETS))
    ap.add_argument("--rounds", type=int, default=1000, help="3-max FP rounds per depth")
    ap.add_argument("--hu-iters", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",")]
    print(f"depth grid: {depths}")

    depth_expl: dict[int, dict[int, float]] = {2: {}, 3: {}}

    # ---- HU (2-max): equity matrix is depth-invariant; solve per depth ----
    print("\nbuilding HU equity matrix (once) ...")
    eq = build_hu_bucket_equity_matrix(progress=False)
    hu_rows: dict[str, "object"] = {}
    for d in depths:
        t = time.monotonic()
        r = solve_hu_pushfold(stack_bb=float(d), iterations=args.hu_iters, seed=0, eq_matrix=eq)
        rows = hu_policy_rows_from_freqs(r.sb_push_freq, r.bb_call_freq, stack_bb=float(d))
        hu_rows.update(rows)
        depth_expl[2][d] = r.exploitability_bb * 1000.0
        print(f"  HU d{d}: {r.exploitability_bb*1000:.2f} mbb/g ({time.monotonic()-t:.0f}s)")

    hu_result = SolveResult(
        num_players=2, solver_name="cfr_plus_dcfr", iterations=args.hu_iters, seed=0,
        exploitability_bb_per_hand=depth_expl[2][8] / 1000.0 if 8 in depth_expl[2] else 0.0,
        exploitability_mbb_g=depth_expl[2].get(8, 0.0),
        br_method="exact", player_improvements=None, policy_rows=hu_rows,
        nash_disclaimer="HU 2p0s: exact exploitability; depth grid in manifest.depth_grid.",
    )
    export_policy_npz(CFR_DIR / "cfr_base_2p.npz", hu_result)
    print(f"  wrote cfr_base_2p.npz ({len(hu_rows)} infosets across {len(depths)} depths)")

    # ---- 3-max: trio tensor is depth-invariant; FP per depth ----
    print("\nloading trio tensor (once) ...")
    shares, pair_eq, deal_w = load_trio_tensor(DEFAULT_TENSOR)
    trio_rows: dict[str, "object"] = {}
    for d in depths:
        t = time.monotonic()
        res = solve_3max_fp(
            max_rounds=args.rounds, damping=1.0, seed=args.seed,
            shares=shares, pair_eq=pair_eq, deal_w=deal_w, stack_bb=float(d),
            stability=StabilityTracker(gap_floor_mbb_g=None, policy_tol=0.0),
        )
        trio_rows.update(trio_fp_policy_rows(res))
        depth_expl[3][d] = res.exploitability_bb * 1000.0
        print(f"  3max d{d}: {res.exploitability_bb*1000:.2f} mbb/g rounds={res.rounds} ({time.monotonic()-t:.0f}s)")

    trio_result = SolveResult(
        num_players=3, solver_name="fictitious_play_3max", iterations=args.rounds, seed=args.seed,
        exploitability_bb_per_hand=depth_expl[3].get(8, 0.0) / 1000.0,
        exploitability_mbb_g=depth_expl[3].get(8, 0.0),
        br_method="exact_legal_marginalized", player_improvements=None, policy_rows=trio_rows,
        nash_disclaimer="N=3 legal marginalized BR; per-depth exploitability in manifest.depth_grid.",
    )
    export_policy_npz(CFR_DIR / "cfr_base_3p.npz", trio_result)
    print(f"  wrote cfr_base_3p.npz ({len(trio_rows)} infosets across {len(depths)} depths)")

    export_manifest(
        CFR_DIR / "cfr_base.manifest.json",
        [hu_result, trio_result],
        merge_existing=True,
        extra={"depth_grid": {str(n): {str(d): v for d, v in depth_expl[n].items()} for n in (2, 3)},
               "stack_depth_buckets": list(STACK_DEPTH_BUCKETS)},
    )
    print("\nwrote manifest with per-depth exploitability grid.")
    print("4-max base NOT solved (perf-blocked); machinery ready for vectorized/C++ solve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
