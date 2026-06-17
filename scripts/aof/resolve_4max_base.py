"""Production re-solve of the 4-max AoF base — fix the under-sampled shipped base.

The shipped cfr_base_4p.npz was solved with solve_4max_fp(max_rounds=30, br_samples=4000): far
too few training deals, so it OVERFITS its tiny sample (in-sample gap ~0) yet is ~14 mbb/g
exploitable OUT-OF-SAMPLE. The experiment scripts/aof/resolve_4max_experiment.py proved that a
vectorized FP solve on ~40k deals drops d=8 out-of-sample exploitability to ~1 mbb/g (the 3-max
floor). This re-solves EVERY depth bucket that way, verifies before/after out-of-sample on held-out
deals, backs up the old artifact, and writes a freshly-stamped cfr_base_4p.npz.

Per-depth work runs in parallel across cores (precompute is the bottleneck). Run as a background
job; ~15-20 min wall on a 12-core box.

    python scripts/aof/resolve_4max_base.py            # solve + verify + write
    python scripts/aof/resolve_4max_base.py --dry-run  # solve + verify, DO NOT overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEPTHS = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
TRAIN_N = 50000
ROUNDS = 2000
EVAL_N = 30000
EVAL_SEEDS = (8001, 8002)
OUT = Path("data/aof/cfr_base/cfr_base_4p.npz")
LOG = Path("runs/aof_audit/resolve_4max_base.log")


def _solve_one(depth: float) -> dict:
    """Full per-depth job (runs in a worker process): train-solve + held-out before/after gaps."""
    import numpy as np

    from aof_cfr.fp_4max_vec import (
        _infoset_index, legal_nash_gap_4max, precompute_payoffs, rows_to_table,
        sample_deals, solve_4max_fp_vec)
    from aof_cfr.profiler import base_policy_rows
    import aof_cfr.query as q
    q.load_cfr_base()                                   # OLD shipped base (for 'before')

    shipped_rows = base_policy_rows(4, depth)
    t0 = time.time()
    bt, ht = sample_deals(TRAIN_N, 1000 + int(depth))
    pt = precompute_payoffs(ht, depth)
    pol = solve_4max_fp_vec(buckets=bt, payoffs=pt, max_rounds=ROUNDS, stack_bb=depth)
    new_rows = pol["rows"]
    t_solve = time.time() - t0

    before, after = [], []
    for es in EVAL_SEEDS:
        be, he = sample_deals(EVAL_N, es)
        pe = precompute_payoffs(he, depth)
        inf = _infoset_index(be)
        before.append(legal_nash_gap_4max(rows_to_table(shipped_rows, depth), be, pe, inf)[0] * 1000)
        after.append(legal_nash_gap_4max(rows_to_table(new_rows, depth), be, pe, inf)[0] * 1000)
    return {
        "depth": depth,
        "rows": {k: v.tolist() for k, v in new_rows.items()},
        "before": before, "after": after,
        "before_mean": float(np.mean(before)), "after_mean": float(np.mean(after)),
        "secs": time.time() - t0, "t_solve": t_solve,
    }


def emit(s: str) -> None:
    print(s, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(s + "\n")


def main() -> None:
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="solve+verify but do NOT overwrite npz")
    ap.add_argument("--workers", type=int, default=min(7, len(DEPTHS)))
    args = ap.parse_args()

    emit("=" * 76)
    emit(f"4-MAX BASE RE-SOLVE  train_n={TRAIN_N} rounds={ROUNDS} eval_n={EVAL_N} "
         f"seeds={EVAL_SEEDS} workers={args.workers}")
    emit("=" * 76)

    t0 = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_solve_one, DEPTHS):
            results.append(r)
            emit(f"  d={r['depth']:>4.0f}bb  before={r['before_mean']:7.3f}  "
                 f"after={r['after_mean']:7.3f} mbb/g   before={[round(x,2) for x in r['before']]} "
                 f"after={[round(x,2) for x in r['after']]}  ({r['secs']:.0f}s)")
    results.sort(key=lambda d: d["depth"])
    emit(f"\nAll depths solved in {time.time()-t0:.0f}s wall.")

    b = np.mean([r["before_mean"] for r in results])
    a = np.mean([r["after_mean"] for r in results])
    emit(f"OVERALL out-of-sample 4-max exploitability:  before={b:.3f}  after={a:.3f} mbb/g")

    # merge rows: START from the full shipped base (every key) then OVERLAY re-solved rows, so a
    # rare infoset absent from the sampled deals keeps its shipped value (never silently -> 0.5).
    shipped = np.load(OUT, allow_pickle=True)
    s_keys = [str(k) for k in shipped["infoset_keys"]]
    s_probs = np.asarray(shipped["action_probs"], dtype=np.float64)
    merged: dict[str, np.ndarray] = {k: s_probs[i] for i, k in enumerate(s_keys)}
    n_base = len(merged)
    n_new = 0
    for r in results:
        for k, v in r["rows"].items():
            if k not in merged:
                n_new += 1
            merged[k] = np.asarray(v, dtype=np.float64)
    n_overlaid = sum(len(r["rows"]) for r in results) - n_new
    emit(f"merged rows: {n_base} shipped base keys, overlaid {n_overlaid} re-solved + "
         f"{n_new} new = {len(merged)} total")

    if args.dry_run:
        emit("\n--dry-run: NOT overwriting the base npz.")
        return

    # backup + export (export stamps with default game_model/fee/spec)
    from aof_cfr.export import export_policy_npz
    from aof_cfr.solve import SolveResult
    if OUT.is_file():
        bak = OUT.with_name("cfr_base_4p_preresolve.npz")
        bak.write_bytes(OUT.read_bytes())
        emit(f"backed up {OUT.name} -> {bak.name}")
    res = SolveResult(
        num_players=4, solver_name="fictitious_play_4max_vec_resolved",
        iterations=ROUNDS, seed=42,
        exploitability_bb_per_hand=a / 1000.0, exploitability_mbb_g=a,
        br_method=f"split_mc_vec out-of-sample @ {EVAL_N}x{len(EVAL_SEEDS)}seeds",
        player_improvements=None, policy_rows=merged,
        nash_disclaimer=(f"N=4 re-solved {TRAIN_N} deals x{ROUNDS} rounds/depth; gap = held-out "
                         f"out-of-sample split-MC mean over depths."))
    export_policy_npz(OUT, res)
    emit(f"wrote {OUT}")

    # verify it loads through the stamp guard
    import aof_cfr.query as q
    q.load_cfr_base()
    emit("re-loaded via query.load_cfr_base() OK (stamp verified)")

    # update manifest's table + grid for n=4
    mpath = Path("data/aof/cfr_base/cfr_base.manifest.json")
    try:
        man = json.loads(mpath.read_text(encoding="utf-8"))
        man["depth_grid_4max"] = {str(int(r["depth"])): round(r["after_mean"], 4) for r in results}
        for t in man.get("tables", []):
            if int(t.get("num_players", 0)) == 4:
                t["solver"] = "fictitious_play_4max_vec_resolved"
                t["iterations"] = ROUNDS
                t["exploitability_mbb_g"] = round(a, 4)
                t["exploitability_bb_per_hand"] = a / 1000.0
                t["br_method"] = f"out-of-sample split-MC @ {EVAL_N}x{len(EVAL_SEEDS)}"
                t["nash_disclaimer"] = ("N=4 re-solved; exploitability is held-out OUT-OF-SAMPLE "
                                        "(in-sample is meaningless for sampled FP).")
        mpath.write_text(json.dumps(man, indent=2), encoding="utf-8")
        emit("updated manifest depth_grid_4max + n=4 table (honest out-of-sample numbers)")
    except (OSError, ValueError) as e:
        emit(f"manifest update skipped: {e}")

    emit("\nDONE.")


if __name__ == "__main__":
    main()
