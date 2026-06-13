"""GTO base quality gates — tightened NashConv ceilings + anomaly explanation.

Replaces the old verification ceilings (4-max gate was 750 mbb/g — "better than trivial") with
calibrated ones: the 4-max sample-split MC gap is measured at >=48k deals (noise floor ~3.5-5
mbb/g vs ~10-15 at the manifest's 6k), gated at 60 mbb/g. HU stays exact vs the tabular BR
(50 mbb/g); 3-max uses the exact legal marginalized BR (50 mbb/g).

``explain_4max_grid`` re-measures each manifest depth cell with TWO disjoint deal seeds — a
cell is flagged when it disagrees with the manifest beyond the split noise (resolves the
suspicious exact-0.0 entries and the unexplained d10=10.2 outlier).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GATE_MBBG = {2: 50.0, 3: 50.0, 4: 60.0}
_MANIFEST = Path("data/aof/cfr_base/cfr_base.manifest.json")


def _table_rows(n: int, stack_bb: float) -> dict:
    from aof_cfr.profiler import base_policy_rows
    return base_policy_rows(n, stack_bb)


def legal_gap_of_rows(n: int, rows: dict, stack_bb: float, *, samples: int = 48000,
                      seed: int = 1234) -> dict:
    """Legal NashConv (bb/hand) of ANY infoset-key-v2 rows table at (n, depth).

    2/3-max use the exact analytic legal BR (zero MC noise); 4-max uses the vectorized
    sample-split estimator (``nash_gap_4max_vec``).
    """
    if n == 2:
        from aof_cfr.evalsuite.policy import _seat_arr
        from aof_cfr.hu_br_legal import _legal_marginals_hu, load_hu_eq
        sb = _seat_arr(rows, 2, "SB", stack_bb)
        bb = _seat_arr(rows, 2, "BB", stack_bb)
        m = _legal_marginals_hu(sb, bb, stack_bb, load_hu_eq())
        gap = 0.0
        for key, push_arr in (("sb", sb), ("bb_a", bb)):
            ep, ef, w = m[key]
            wn = np.asarray(w, dtype=np.float64)
            wn = wn / wn.sum() if wn.sum() > 0 else wn
            onpol = push_arr * ep + (1.0 - push_arr) * ef
            gap += float((wn * (np.maximum(ep, ef) - onpol)).sum())
        return {"gap_bb": gap, "method": "exact_legal_hu"}
    if n == 3:
        from aof_cfr.evalsuite.policy import _seat_arr
        from aof_cfr.exploit_router import _payoffs_3max
        from aof_cfr.fp_br import nash_gap_3max_legal
        st, dw = _payoffs_3max(stack_bb)
        gap, imps = nash_gap_3max_legal(st, dw, _seat_arr(rows, 3, "BTN", stack_bb),
                                        _seat_arr(rows, 3, "SB", stack_bb),
                                        _seat_arr(rows, 3, "BB", stack_bb))
        return {"gap_bb": gap, "method": "exact_legal_marginalized", "imps": imps}
    from aof_cfr.fp_4max_vec import nash_gap_4max_vec
    gap = nash_gap_4max_vec(rows, n_deals=samples, seed=seed, stack_bb=stack_bb)
    return {"gap_bb": gap, "method": f"split_mc_vec @ {samples}"}


def nash_gap_mbbg(n: int, stack_bb: float, *, samples: int = 48000, seed: int = 1234) -> dict:
    """Legal NashConv of the shipped BASE at (n, depth) in mbb/g (see ``legal_gap_of_rows``).

    The 4-max gate at 48k deals drops the split-noise floor from ~10-15 (manifest's 6k) to
    ~3.5-5 mbb/g.
    """
    import aof_cfr.query as q
    q.load_cfr_base()
    r = legal_gap_of_rows(n, _table_rows(n, stack_bb), stack_bb, samples=samples, seed=seed)
    out = {"mbbg": r["gap_bb"] * 1000.0, "method": r["method"]}
    if "imps" in r:
        out["imps"] = r["imps"]
    return out


def nash_gate_cells(depths=(4.0, 8.0, 12.0, 20.0), *, samples: int = 48000, seeds=(1234, 5678)):
    """NashConv gate cells. n=2/3 are analytic (deterministic, exact — one pass). n=4 is split-MC
    and seed-NOISY: run >=2 seeds, report the real se/ci95, and gate on the PESSIMISTIC bound
    ``mean + ci95`` so a single lucky draw can't pass (was: one seed, fake se=ci=0)."""
    from aof_cfr.evalsuite.paired_arena import ci95_from_seeds
    from aof_cfr.evalsuite.result import AoFEvalCell
    cells = []
    for n in (2, 3, 4):
        for d in depths:
            gate = GATE_MBBG[n]
            if n == 4:
                vals = [nash_gap_mbbg(n, d, samples=samples, seed=s)["mbbg"] for s in seeds]
                mean, ci = ci95_from_seeds(vals)
                se = (ci / 2.0) if ci != float("inf") else 0.0
                passed = (mean + ci) <= gate            # conservative: lucky seed can't sneak through
                method = f"split_mc_vec @ {samples} x{len(seeds)}seeds"
                n_seeds = len(seeds)
            else:
                r = nash_gap_mbbg(n, d, samples=samples)  # analytic legal BR -> deterministic
                mean, ci, se, method, n_seeds = r["mbbg"], 0.0, 0.0, r["method"], 1
                passed = mean <= gate
            cells.append(AoFEvalCell(
                metric="nash_gap", players=n, depth=d, field_name="-", arm="-",
                hands=samples if n == 4 else 0, seeds=n_seeds,
                mean_bb100=round(mean, 3), se_bb100=round(se, 3), ci95_bb100=round(ci, 3),
                gate=f"legal NashConv {'mean+ci95 ' if n == 4 else ''}<= {gate} mbb/g",
                passed=passed,
                meta={"method": method, "unit": "mbb/g"}))
    return cells


def explain_4max_grid(*, samples: int = 24000, seeds=(101, 202)) -> list[dict]:
    """Re-measure every manifest ``depth_grid_4max`` cell with disjoint seeds; flag cells whose
    manifest value sits outside the re-measured spread (stale grid entry or seed alias)."""
    import aof_cfr.query as q
    q.load_cfr_base()
    try:
        manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    except OSError:
        return [{"error": "manifest missing"}]
    grid = manifest.get("depth_grid_4max") or {}
    from aof_cfr.fp_4max_vec import nash_gap_4max_vec
    out = []
    for dkey, claimed in sorted(grid.items(), key=lambda kv: float(str(kv[0]).lstrip("d"))):
        depth = float(str(dkey).lstrip("d"))
        rows = _table_rows(4, depth)
        vals = []
        for s in seeds:
            gap = nash_gap_4max_vec(rows, n_deals=samples, seed=s, stack_bb=depth)
            vals.append(gap * 1000.0)
        spread = max(vals) - min(vals)
        noise = max(spread, 2.0)                       # split-MC floor; never call < 2 mbb noise
        flagged = abs(float(claimed) - float(np.mean(vals))) > 2.0 * noise
        out.append({"depth": depth, "manifest_mbbg": float(claimed),
                    "remeasured_mbbg": [round(v, 2) for v in vals],
                    "flagged": bool(flagged)})
    return out
