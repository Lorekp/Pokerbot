#!/usr/bin/env python3
"""Sweep exact-enum fraction on row 0 vs combo-count threshold (analytic + optional probe)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.cpu_pool import pin_compute_threads
from aof_cfr.trio_eval_mc import build_live_deck, mc_nway_shares_adaptive
from aof_cfr.trio_payoff_tensor import (
    DEFAULT_TRIPLE_SAMPLE_BUDGET,
    MC_SHARE_STDERR_GATE,
    MIN_BOARDS_PER_COMBO,
    _all_combo_triples,
    _combo_seed,
)
from aof_equity import warmup
from aof_equity_grids import _card_str_to_id

N = 169


def _analytic_row(row: int, threshold: int, budget: int) -> dict[str, float | int]:
    exact_by_threshold = 0
    all_combos_runout_exact = 0
    any_combo_runout_exact = 0
    mc_triples = 0
    total = N * N
    max_combo_count = 0

    for j in range(N):
        for k in range(N):
            combos = _all_combo_triples(row, j, k)
            count = len(combos)
            if count == 0:
                continue
            max_combo_count = max(max_combo_count, count)
            if count <= threshold:
                exact_by_threshold += 1
                continue
            boards_per_combo = max(MIN_BOARDS_PER_COMBO, budget // count)
            runout_hits = 0
            for (c0, c1), (c2, c3), (c4, c5) in combos:
                flat = np.array(
                    [
                        _card_str_to_id(c0),
                        _card_str_to_id(c1),
                        _card_str_to_id(c2),
                        _card_str_to_id(c3),
                        _card_str_to_id(c4),
                        _card_str_to_id(c5),
                    ],
                    dtype=np.int32,
                )
                _, nd = build_live_deck(flat, 3)
                if nd >= 5 and math.comb(nd, 5) <= boards_per_combo:
                    runout_hits += 1
            if runout_hits == count:
                all_combos_runout_exact += 1
            elif runout_hits > 0:
                any_combo_runout_exact += 1
            else:
                mc_triples += 1

    return {
        "threshold": threshold,
        "total_triples": total,
        "exact_by_threshold": exact_by_threshold,
        "all_combos_runout_le_budget": all_combos_runout_exact,
        "some_combo_runout_le_budget": any_combo_runout_exact,
        "mc_only_triples": mc_triples,
        "frac_exact_by_threshold": exact_by_threshold / total,
        "frac_all_combos_runout_exact": all_combos_runout_exact / total,
        "frac_mc_only": mc_triples / total,
        "frac_any_exact_path": (exact_by_threshold + all_combos_runout_exact) / total,
        "max_combo_count_seen": max_combo_count,
    }


def _probe_adaptive_triples(row: int, threshold: int, budget: int, *, limit: int) -> dict[str, int]:
    """Optional: run adaptive MC on first ``limit`` MC-only triples to count exact_enum hits."""
    hits = 0
    probed = 0
    for j in range(N):
        for k in range(N):
            combos = _all_combo_triples(row, j, k)
            count = len(combos)
            if count <= threshold or count == 0:
                continue
            boards_per_combo = max(MIN_BOARDS_PER_COMBO, budget // count)
            all_small = True
            for (c0, c1), (c2, c3), (c4, c5) in combos:
                flat = np.array(
                    [
                        _card_str_to_id(c0),
                        _card_str_to_id(c1),
                        _card_str_to_id(c2),
                        _card_str_to_id(c3),
                        _card_str_to_id(c4),
                        _card_str_to_id(c5),
                    ],
                    dtype=np.int32,
                )
                _, nd = build_live_deck(flat, 3)
                if nd < 5 or math.comb(nd, 5) > boards_per_combo:
                    all_small = False
                    break
            if all_small:
                continue
            probed += 1
            sub_seed = _combo_seed(row, j, k)
            used_exact = False
            for n, ((c0, c1), (c2, c3), (c4, c5)) in enumerate(combos):
                flat = np.array(
                    [
                        _card_str_to_id(c0),
                        _card_str_to_id(c1),
                        _card_str_to_id(c2),
                        _card_str_to_id(c3),
                        _card_str_to_id(c4),
                        _card_str_to_id(c5),
                    ],
                    dtype=np.int32,
                )
                _, _, _, method = mc_nway_shares_adaptive(
                    flat,
                    3,
                    sub_seed + n,
                    stderr_gate=MC_SHARE_STDERR_GATE,
                    max_boards=boards_per_combo,
                    min_boards=MIN_BOARDS_PER_COMBO,
                )
                if method.startswith("exact"):
                    used_exact = True
                    break
            if used_exact:
                hits += 1
            if probed >= limit:
                return {"probed": probed, "exact_hits": hits}
    return {"probed": probed, "exact_hits": hits}


def main() -> None:
    p = argparse.ArgumentParser(description="Exact-enum fraction sweep on one bucket row")
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--budget", type=int, default=DEFAULT_TRIPLE_SAMPLE_BUDGET)
    p.add_argument("--thresholds", default="0,8,16,36")
    p.add_argument("--probe-limit", type=int, default=0, help="Adaptive MC probe on N MC-only triples")
    p.add_argument("--out", default="data/aof/cfr_base/exact_enum_fraction_sweep.json")
    args = p.parse_args()

    warmup()
    pin_compute_threads(1)

    thresholds = [int(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    results = []
    for th in thresholds:
        print(f"threshold={th} analytic sweep (row={args.row}, budget={args.budget}) ...", flush=True)
        row = _analytic_row(args.row, th, args.budget)
        print(
            f"  threshold={th}: exact_by_threshold={row['exact_by_threshold']} "
            f"({100.0 * row['frac_exact_by_threshold']:.2f}%) "
            f"mc_only={row['mc_only_triples']} max_combo={row['max_combo_count_seen']}",
            flush=True,
        )
        if args.probe_limit > 0 and th == thresholds[-1]:
            row["adaptive_probe"] = _probe_adaptive_triples(
                args.row, th, args.budget, limit=args.probe_limit
            )
        results.append(row)

    out = {
        "row": args.row,
        "budget": args.budget,
        "stderr_gate": MC_SHARE_STDERR_GATE,
        "note": (
            "frac_exact_by_threshold = triples with combo_count <= threshold (full exact enum). "
            "frac_mc_only = triples needing MC on all combos (runouts > per-combo budget). "
            "Stale 15% figure used threshold=36 on full tensor; re-measure here on row 0."
        ),
        "thresholds": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
