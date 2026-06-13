"""Re-measure the shipped HU base's exploitability with the FIXED deal-weighted metric and
update the manifest in place.

The policy artifacts are untouched (the CFR+ training loop was always deal_w-correct); only
the gap RECORDED at solve time was computed by the old metric whose SB terms used uniform 1/n
weights — correct in the v1 uniform game, a phantom ~10 mbb/g floor under v2 combo weights.

Run:  python scripts/remeasure_hu_gap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_cfr.hu_payoff_matrix import hu_deal_weights
from aof_cfr.hu_br_legal import _pot_called, load_hu_eq
from aof_cfr.hu_validate import load_policy_freqs
from aof_cfr.infoset import STACK_DEPTH_BUCKETS
from pokerbot.solver.hu_preflop_dcfr import _exploitability

CFR_DIR = Path("data/aof/cfr_base")


def main() -> int:
    eq = load_hu_eq()
    w = hu_deal_weights(eq.shape[0])

    grid: dict[str, float] = {}
    for d in STACK_DEPTH_BUCKETS:
        sb, bb = load_policy_freqs(CFR_DIR / "cfr_base_2p.npz", depth=float(d))
        gap = _exploitability(eq, sb, bb, float(d), _pot_called(float(d)), 0.5, 1.0, w)
        grid[str(int(d))] = gap * 1000.0
        print(f"  HU d{d}: {gap*1000:.3f} mbb/g (deal-weighted metric)")

    mpath = CFR_DIR / "cfr_base.manifest.json"
    m = json.loads(mpath.read_text(encoding="utf-8"))
    for t in m.get("tables", []):
        if int(t.get("num_players", 0)) == 2:
            t["exploitability_mbb_g"] = grid.get("8", 0.0)
            t["exploitability_bb_per_hand"] = grid.get("8", 0.0) / 1000.0
            t["gap_metric"] = "deal_weighted_v2 (remeasured; solve-time metric had uniform SB weights)"
    m.setdefault("depth_grid", {})["2"] = grid
    mpath.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"updated {mpath} (2-max gap @8bb -> {grid.get('8', 0.0):.3f} mbb/g)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
