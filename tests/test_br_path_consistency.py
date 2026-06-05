"""Cross-path drift guard: the analytic tensor BR (fp_br) and the MC combo BR (exploit_mc)
must agree that the shipped 3-max base is near-optimal.

Both the GTO solve and the per-player-exploitability metric depend on these two paths staying
consistent. A chip-model regression (pot 24 vs 25.5) or a return of the clairvoyance bug would
blow one path up to hundreds/thousands of mbb/g on the (correct) shipped policy — this test
catches that. It deliberately checks the NEAR-EQUILIBRIUM shipped policy (where the 169-bucket
abstraction error between the paths is negligible), not an extreme policy (where legitimate
abstraction error would cause false alarms).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aof_cfr.buckets import LABELS_169
from aof_cfr.exploit_mc import estimate_nash_conv_mc
from aof_cfr.fp_br import load_3max_payoffs, nash_gap_3max_legal
from aof_cfr.infoset import encode_infoset_key
from aof_cfr.pushfold_payoffs import BB_HIST, SB_HIST
from aof_cfr.trio_payoff_tensor import DEFAULT_TENSOR, load_trio_tensor
import aof_cfr.query as q

_ART = Path("data/aof/cfr_base/cfr_base_3p.npz")
_TOL_MBBG = 30.0  # both paths must agree the shipped base is well under this (trivial ~3000)


@pytest.mark.slow
@pytest.mark.skipif(not (_ART.is_file() and Path(DEFAULT_TENSOR).is_file()),
                    reason="3-max base / tensor not built")
def test_shipped_base_near_optimal_both_paths() -> None:
    q.load_cfr_base()
    probs, _keys, index = q._LOADED[3]

    def p(pos, lab, hist):
        return float(probs[index[encode_infoset_key(pos, lab, hist, 8.0)]][1])

    btn = np.array([p("BTN", l, "-") for l in LABELS_169])
    sb = np.stack([[p("SB", l, h) for l in LABELS_169] for h in SB_HIST])
    bb = np.stack([[p("BB", l, h) for l in LABELS_169] for h in BB_HIST])

    shares, pair_eq, dw = load_trio_tensor(DEFAULT_TENSOR)
    st, dwn = load_3max_payoffs(shares, pair_eq, dw, stack_bb=8.0)
    analytic_mbbg = nash_gap_3max_legal(st, dwn, btn, sb, bb)[0] * 1000.0

    rows = {k: probs[i] for k, i in index.items() if k.endswith("|d8")}
    mc_mbbg = estimate_nash_conv_mc(3, rows, samples=6000, seed=9, stack_bb=8.0)[0] * 1000.0

    assert analytic_mbbg < _TOL_MBBG, f"analytic legal gap {analytic_mbbg:.1f} mbb/g too high"
    assert mc_mbbg < _TOL_MBBG, f"MC legal gap {mc_mbbg:.1f} mbb/g too high (chip/clairvoyance drift?)"
    # Near eq, the two paths agree within abstraction error + MC noise.
    assert abs(analytic_mbbg - mc_mbbg) < _TOL_MBBG, (
        f"analytic {analytic_mbbg:.1f} vs MC {mc_mbbg:.1f} mbb/g — BR paths diverged"
    )
