"""How exploitable is the shipped base (solved @ fee 0.12bb) when the REAL game fee differs?

CoinPoker AoF fee is stake-dependent: 0.10/0.25 -> 0.16bb, 0.25/0.50..10/20 -> 0.12bb, and
PLAY-MONEY -> 0.0. The base npz bakes in the 0.12 ranges. This measures the best-response gap of
those fixed 0.12-ranges *evaluated under* each candidate fee (HU + 3-max, exact analytic — no MC),
i.e. the real-world cost of the fee mismatch. The env AOF_FLAT_FEE_BB overrides the fee used by the
payoff/BR math (takes precedence over the manifest), while the policy rows stay the shipped 0.12 ones.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FEES = [0.0, 0.12, 0.16, 0.22]
DEPTHS = [6.0, 8.0, 10.0]


def gap_under_fee(n: int, depth: float, fee: float) -> float:
    """mbb/g best-response gap of the SHIPPED base rows evaluated under `fee`.

    Re-imports the fee-dependent modules under a fresh AOF_FLAT_FEE_BB so cached pots/eq reflect
    the fee. Policy rows come from the shipped npz (solved @ 0.12) and are NOT re-solved."""
    os.environ["AOF_FLAT_FEE_BB"] = str(fee)
    # Drop cached fee/game-model + any fee-baked modules so the new fee takes effect.
    import importlib

    import aof_cfr.pushfold_payoffs as pp
    importlib.reload(pp)
    import aof_cfr.hu_br_legal as hbl
    importlib.reload(hbl)
    import aof_cfr.fp_br as fpb
    importlib.reload(fpb)
    import aof_cfr.exploit_router as er
    importlib.reload(er)
    import aof_cfr.evalsuite.nash_gates as ng
    importlib.reload(ng)
    import aof_cfr.query as q
    q.load_cfr_base()
    r = ng.nash_gap_mbbg(n, stack_bb=depth)
    return r["mbbg"]


def main() -> None:
    print("FEE-SENSITIVITY of the shipped base (rows solved @ 0.12bb), exact BR, mbb/g")
    print("fee=0.0 ~ play-money; 0.16 ~ 0.10/0.25 stake; 0.22 ~ DC2 capture estimate\n")
    for n in (2, 3):
        print(f"[{n}-max]")
        header = "  depth |" + "".join(f"  fee={f:<5}" for f in FEES)
        print(header)
        for d in DEPTHS:
            cells = []
            for f in FEES:
                try:
                    cells.append(f"{gap_under_fee(n, d, f):8.3f}")
                except Exception as e:  # noqa: BLE001
                    cells.append(f"  ERR:{type(e).__name__}")
            print(f"  {d:>4.0f}bb |" + "".join(f"  {c}" for c in cells))
        print()


if __name__ == "__main__":
    main()
