"""Head-to-head bb/100 evaluation (3-max, exact analytic) — empirical proof of the goal.

Measures hero's expected chip-EV (bb/100) at BTN against:
  * a GTO ("Pluribus-level") table  — our own near-equilibrium base, and
  * a weak table of over-folders     — call jams with premiums only,
for hero playing the GTO base vs hero playing the confidence-gated exploit router (fully
profiled). Exact on-policy values (``fp_br.seat_values_3max``) over the bucket deal — zero
sampling variance.

Claims demonstrated:
  * Exploit > base vs the weak table  -> the engine exploits weak opponents for real EV.
  * Exploit == base vs the GTO table  -> it never deviates (loses nothing) vs strong play.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.fp_br import _legal_marginals_3max, load_3max_payoffs, seat_values_3max
from aof_cfr.trio_payoff_tensor import DEFAULT_TENSOR, load_trio_tensor

try:
    from aof import MARGINAL_EDGE_BB
except Exception:
    MARGINAL_EDGE_BB = 0.30

PREMIUM = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}


def _base_arrays(stack_bb: float):
    import aof_cfr.query as q
    q.load_cfr_base()
    n = len(LABELS_169)
    btn = np.array([q.base_strategy(3, "BTN", "-", h, stack_bb) for h in LABELS_169])
    sb = np.stack([[q.base_strategy(3, "SB", hh, h, stack_bb) for h in LABELS_169] for hh in ("F", "A")])
    bb = np.stack([[q.base_strategy(3, "BB", hh, h, stack_bb) for h in LABELS_169] for hh in ("FA", "AF", "AA")])
    return btn, sb, bb


def _weak_overfolders(base_sb, base_bb):
    """Blinds that call a jam with premiums only (massive over-folding); open as base."""
    prem = np.array([1.0 if h in PREMIUM else 0.0 for h in LABELS_169])
    sb = base_sb.copy(); bb = base_bb.copy()
    sb[1] = prem            # SB facing BTN jam: call premiums only
    bb[1] = prem            # BB|AF (BTN jam, SB fold): call premiums only
    bb[2] = prem            # BB|AA (both jam): call premiums only
    return sb, bb


def _router_btn(base_btn, opp_sb, opp_bb, st, dw, *, confidence=1.0):
    """Vectorized router BTN policy vs (profiled) opponents: margin-gated confidence blend."""
    m = _legal_marginals_3max(st, dw, base_btn, opp_sb, opp_bb)
    e_push, e_fold, _w = m["btn"]
    base_ev = (1.0 - base_btn) * e_fold + base_btn * e_push
    gain = np.maximum(e_fold, e_push) - base_ev
    exploit_pure = (e_push > e_fold).astype(np.float64)
    gated = (gain >= MARGINAL_EDGE_BB)
    blended = (1.0 - confidence) * base_btn + confidence * exploit_pure
    return np.where(gated, blended, base_btn)


def evaluate(stack_bb: float = 8.0) -> dict:
    shares, pair_eq, deal_w = load_trio_tensor(DEFAULT_TENSOR)
    st, dw = load_3max_payoffs(shares, pair_eq, deal_w, stack_bb=stack_bb)
    base_btn, base_sb, base_bb = _base_arrays(stack_bb)
    weak_sb, weak_bb = _weak_overfolders(base_sb, base_bb)

    exploit_vs_weak = _router_btn(base_btn, weak_sb, weak_bb, st, dw)
    exploit_vs_gto = _router_btn(base_btn, base_sb, base_bb, st, dw)

    def btn_bb100(btn, sb, bb):
        return seat_values_3max(st, dw, btn, sb, bb)[0] * 100.0

    return {
        "stack_bb": stack_bb,
        "base_vs_gto": btn_bb100(base_btn, base_sb, base_bb),
        "exploit_vs_gto": btn_bb100(exploit_vs_gto, base_sb, base_bb),
        "base_vs_weak": btn_bb100(base_btn, weak_sb, weak_bb),
        "exploit_vs_weak": btn_bb100(exploit_vs_weak, weak_sb, weak_bb),
        "n_btn_deviations": int(np.sum(np.abs(exploit_vs_weak - base_btn) > 1e-6)),
    }


def evaluate_4max(stack_bb: float = 8.0, n_deals: int = 8000, seed: int = 99) -> dict:
    """4-max UTG bb/100: exploit (router) vs over-folding table vs GTO table (vectorized)."""
    import numpy as np
    from aof_cfr.fp_4max_vec import (_br_marginals, _infoset_index, onpolicy_seat_ev,
                                     precompute_payoffs, rows_to_table, sample_deals)
    from aof_cfr.profiler import base_policy_rows

    buckets, holes = sample_deals(n_deals, seed)
    payoffs = precompute_payoffs(holes, stack_bb)
    infosets = _infoset_index(buckets)
    base_tab = rows_to_table(base_policy_rows(4, stack_bb), stack_bb)

    # Weak opponents (BTN, SB, BB) call any jam with premiums only; UTG (seat 0) plays base.
    prem = np.array([1.0 if h in PREMIUM else 0.0 for h in LABELS_169])
    weak_tab = rows_to_table(base_policy_rows(4, stack_bb), stack_bb)
    for (i, h) in list(weak_tab.table):
        if i != 0 and "A" in h:  # facing at least one jam -> over-fold (call premiums only)
            weak_tab.table[(i, h)] = prem.copy()

    # UTG exploit policy vs the weak table: margin-gated pure BR at UTG infoset ("-").
    np_push, np_fold, den = _br_marginals(weak_tab, buckets, payoffs, infosets)
    e_push = np.where(den[(0, "-")] > 0, np_push[(0, "-")], 0.0)
    e_fold = np.where(den[(0, "-")] > 0, np_fold[(0, "-")], 0.0)
    base_utg = base_tab.table[(0, "-")]
    base_ev = (1.0 - base_utg) * e_fold + base_utg * e_push
    gain = np.maximum(e_push, e_fold) - base_ev
    exploit_utg = np.where(gain >= MARGINAL_EDGE_BB, (e_push > e_fold).astype(float), base_utg)

    weak_exploit = rows_to_table(base_policy_rows(4, stack_bb), stack_bb)
    weak_exploit.table.update({k: v for k, v in weak_tab.table.items()})
    weak_exploit.table[(0, "-")] = exploit_utg
    weak_base = rows_to_table(base_policy_rows(4, stack_bb), stack_bb)
    weak_base.table.update({k: v for k, v in weak_tab.table.items()})

    return {
        "stack_bb": stack_bb,
        "base_vs_gto": onpolicy_seat_ev(base_tab, buckets, payoffs)[0] * 100.0,
        "base_vs_weak": onpolicy_seat_ev(weak_base, buckets, payoffs)[0] * 100.0,
        "exploit_vs_weak": onpolicy_seat_ev(weak_exploit, buckets, payoffs)[0] * 100.0,
        "n_utg_deviations": int(np.sum(np.abs(exploit_utg - base_utg) > 1e-6)),
    }


def main() -> int:
    for d in (6, 8, 12):
        r = evaluate(float(d))
        print(f"\n=== BTN bb/100 @ {d}bb (3-max) ===")
        print(f"  vs GTO table:   base {r['base_vs_gto']:+.2f}   exploit {r['exploit_vs_gto']:+.2f}  (deviations vs GTO: 0 expected)")
        print(f"  vs WEAK table:  base {r['base_vs_weak']:+.2f}   exploit {r['exploit_vs_weak']:+.2f}")
        print(f"  exploit uplift vs weak: {r['exploit_vs_weak'] - r['base_vs_weak']:+.2f} bb/100  "
              f"({r['n_btn_deviations']} BTN hands deviated)")
    r4 = evaluate_4max(8.0)
    print(f"\n=== UTG bb/100 @ 8bb (4-max) ===")
    print(f"  vs GTO table:   base {r4['base_vs_gto']:+.2f}")
    print(f"  vs WEAK table:  base {r4['base_vs_weak']:+.2f}   exploit {r4['exploit_vs_weak']:+.2f}")
    print(f"  exploit uplift vs weak: {r4['exploit_vs_weak'] - r4['base_vs_weak']:+.2f} bb/100  "
          f"({r4['n_utg_deviations']} UTG hands deviated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
