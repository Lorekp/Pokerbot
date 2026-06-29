"""Bounded native NLHE trainer profiling smoke.

This script is local-safe by default: it only runs a few CFR+ iterations with a small deal count and refuses
large iteration*deal products unless explicitly overridden. It does not start cloud jobs, evals, pilots, or
full training runs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_ext():
    try:
        import battery_native as ext

        if hasattr(ext, "NlheCFRTrainer"):
            ext_path = getattr(ext, "__file__", None)
            release_dll = ROOT / "tools" / "battery_native" / "target" / "release" / "battery_native.dll"
            if ext_path and release_dll.exists() and Path(ext_path).exists():
                if release_dll.stat().st_mtime > Path(ext_path).stat().st_mtime:
                    raise ImportError(f"stale battery_native extension {ext_path}; release dll is newer")
            return ext
    except Exception:
        pass

    from pokerbot.search import native_solver

    ext = native_solver._try_load()
    if ext is None or not hasattr(ext, "NlheCFRTrainer"):
        raise SystemExit(
            "battery_native with NlheCFRTrainer is not importable; build tools/battery_native first"
        )
    return ext


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float:
        return float(statistics.fmean(float(r.get(key, 0.0)) for r in rows)) if rows else 0.0

    def total(key: str) -> float:
        return float(sum(float(r.get(key, 0.0)) for r in rows))

    traverse_s = total("traverse_ms") / 1000.0
    nodes = total("nodes")
    fingerprint_calls = total("fingerprint_calls")
    fp_new = total("fp_memo_new_entries")
    emd_queries = total("emd_bucket_queries")
    emd_hits = total("emd_bucket_hits")
    emd_fp_hits = total("emd_fp_memo_hits")
    emd_fp_misses = total("emd_fp_memo_misses")

    return {
        "iterations": len(rows),
        "iters_done": rows[-1]["iters_done"] if rows else 0,
        "n_deals": rows[-1]["n_deals"] if rows else 0,
        "rayon_threads": rows[-1]["rayon_threads"] if rows else 0,
        "fast_merge": rows[-1]["fast_merge"] if rows else False,
        "final_cum_regret_rows": rows[-1]["cum_regret_rows"] if rows else 0,
        "final_strat_sum_rows": rows[-1]["strat_sum_rows"] if rows else 0,
        "avg_total_ms": avg("total_ms"),
        "avg_traverse_ms": avg("traverse_ms"),
        "avg_merge_ms": avg("merge_ms"),
        "avg_floor_ms": avg("floor_ms"),
        "merge_share": avg("merge_ms") / max(avg("total_ms"), 1e-9),
        "floor_share": avg("floor_ms") / max(avg("total_ms"), 1e-9),
        "avg_nodes": avg("nodes"),
        "avg_decision_nodes": avg("decision_nodes"),
        "avg_hero_nodes": avg("hero_nodes"),
        "avg_opponent_nodes": avg("opponent_nodes"),
        "avg_chance_nodes": avg("chance_nodes"),
        "avg_terminal_nodes": avg("terminal_nodes"),
        "nodes_per_second": nodes / max(traverse_s, 1e-9),
        "avg_legal_actions_per_decision": total("legal_actions_total") / max(total("decision_nodes"), 1.0),
        "fp_memo_new_entries": int(fp_new),
        "fp_memo_miss_rate": fp_new / max(fingerprint_calls, 1.0),
        "emd_bucket_queries": int(emd_queries),
        "emd_bucket_hit_rate": emd_hits / max(emd_queries, 1.0),
        "emd_bucket_misses": int(total("emd_bucket_misses")),
        "emd_bucket_fallbacks": int(total("emd_bucket_fallbacks")),
        "emd_fp_memo_hits": int(emd_fp_hits),
        "emd_fp_memo_misses": int(emd_fp_misses),
        "emd_fp_memo_hit_rate": emd_fp_hits / max(emd_fp_hits + emd_fp_misses, 1.0),
        "emd_blake2b_calls_avoided_vs_old_path": int(emd_fp_hits),
        "max_depth": int(max((int(r.get("max_depth", 0)) for r in rows), default=0)),
    }


def _run_case(
    ext: Any,
    *,
    num_buckets: int,
    deals: int,
    iterations: int,
    fast_merge: bool,
    emd_frozen: str | None,
) -> dict[str, Any]:
    trainer = ext.NlheCFRTrainer(num_buckets, [], deals, 6, float("-inf"), fast_merge, emd_frozen)
    rows = json.loads(trainer.train_profile_json(iterations))
    return {"summary": _summarize(rows), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--deals", type=int, default=16)
    parser.add_argument("--num-buckets", type=int, default=200)
    parser.add_argument("--emd-frozen", default=None, help="optional .emdfrzn lookup path to profile the EMD trainer path")
    parser.add_argument("--threads", type=int, default=0, help="sets RAYON_NUM_THREADS before import")
    parser.add_argument(
        "--mode",
        choices=("serial", "fast", "both"),
        default="both",
        help="serial=byte-identical merge, fast=tree-reduce merge",
    )
    parser.add_argument(
        "--max-work",
        type=int,
        default=2048,
        help="maximum iterations*deals per mode without --allow-larger",
    )
    parser.add_argument("--allow-larger", action="store_true")
    args = parser.parse_args()

    if args.iterations <= 0 or args.deals <= 0:
        raise SystemExit("--iterations and --deals must be positive")
    work = args.iterations * args.deals
    if work > args.max_work and not args.allow_larger:
        raise SystemExit(
            f"refusing unbounded microbench: iterations*deals={work} > max_work={args.max_work}"
        )
    if args.threads > 0:
        os.environ["RAYON_NUM_THREADS"] = str(args.threads)

    ext = _load_ext()
    modes = [False] if args.mode == "serial" else [True] if args.mode == "fast" else [False, True]
    payload = {
        "config": {
            "iterations": args.iterations,
            "deals": args.deals,
            "num_buckets": args.num_buckets,
            "emd_frozen": args.emd_frozen,
            "requested_threads": args.threads,
            "mode": args.mode,
        },
        "cases": {},
    }
    for fast_merge in modes:
        name = "fast_merge" if fast_merge else "serial_merge"
        payload["cases"][name] = _run_case(
            ext,
            num_buckets=args.num_buckets,
            deals=args.deals,
            iterations=args.iterations,
            fast_merge=fast_merge,
            emd_frozen=args.emd_frozen,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
