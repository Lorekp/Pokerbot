#!/usr/bin/env python3
"""CERT9 verdict assembler: merge engine_ab_h2h shard JSONs into VERDICT_FINAL.json.

Re-runnable after every section (the cert7 pattern): a cap-stop / spot reclaim always leaves the
freshest partial verdict. A shard contributes money ONLY if its search-arm proof-of-search gate
passes (cert9_gate.evaluate): decisions_fallback == 0 AND 100% native-served postflop decisions
AND the wall floor — otherwise the shard is listed under invalid_shards and its decks are dropped.

Sections merged (by filename convention in --out-dir):
  probe_ab.json                       pre-spend probe (evidence block, not money)
  cert9_ab_seed*_shard*.json          PRIMARY: tier-1 SEARCH (arm a) vs tier-1 RAW (arm b), CRN-paired
  cert9_inc_seed*_shard*.json         SECONDARY: tier-1 SEARCH (arm a) vs incumbent K160 (arm b)

chips/deck -> bb/100 uses the established factor 6 (cert5/cert7 continuity: -832 chips/deck was
reported as ~-139 bb/100)."""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cert9_gate import evaluate  # noqa: E402

CHIPS_PER_DECK_PER_BB100 = 6.0


def _pool(shards: list[dict], *, require_gate: bool, min_mean_wall_s: float) -> dict:
    per_shard, valid, invalid = [], [], []
    nets: list[float] = []
    agg = {"decisions_total": 0, "decisions_fallback": 0, "postflop_decisions": 0,
           "postflop_search_native": 0}
    streets: dict[str, dict] = {}
    for path, res in shards:
        gate = evaluate(res, arm="a", min_postflop=1, min_mean_wall_s=min_mean_wall_s,
                        check_wall=require_gate)
        pd = [float(x) for x in res.get("per_deck_nets", [])]
        row = {
            "file": os.path.basename(path),
            "label": res.get("label"),
            "decks_done": len(pd),
            "mean_chips_per_deck": (sum(pd) / len(pd)) if pd else None,
            "wall_s": res.get("wall_s", res.get("wall_s_so_far")),
            "seed": res.get("seed"),
            "gate_pass": bool(gate["pass"]) if require_gate else None,
            "gate_reasons": gate.get("reasons", []),
            "decisions_fallback": gate.get("decisions_fallback"),
            "postflop_search_native": gate.get("postflop_search_native"),
            "postflop_decisions": gate.get("postflop_decisions"),
            "mean_postflop_wall_s": gate.get("mean_postflop_wall_s"),
        }
        per_shard.append(row)
        if require_gate and not gate["pass"]:
            invalid.append(row)
            continue
        valid.append(row)
        nets.extend(pd)
        ss = (res.get("arm_a_runtime_stats") or {}).get("search_stats") or {}
        for k in agg:
            agg[k] += int(ss.get(k, 0) or 0)
        for st, srow in (ss.get("by_street") or {}).items():
            d = streets.setdefault(st, {"n": 0, "wall_weighted_sum_s": 0.0,
                                        "p50_range_s": [None, None], "p95_range_s": [None, None],
                                        "served": {}})
            n = int(srow.get("n", 0))
            d["n"] += n
            if srow.get("wall_mean_s") is not None:
                d["wall_weighted_sum_s"] += float(srow["wall_mean_s"]) * n
            for key, tgt in (("wall_p50_s", "p50_range_s"), ("wall_p95_s", "p95_range_s")):
                v = srow.get(key)
                if v is not None:
                    lo, hi = d[tgt]
                    d[tgt] = [v if lo is None else min(lo, v), v if hi is None else max(hi, v)]
            for name, c in (srow.get("served") or {}).items():
                d["served"][name] = d["served"].get(name, 0) + int(c)
    for st, d in streets.items():
        d["wall_mean_s"] = (d["wall_weighted_sum_s"] / d["n"]) if d["n"] else None
        d.pop("wall_weighted_sum_s", None)

    n = len(nets)
    mean = (sum(nets) / n) if n else None
    if n > 1:
        var = sum((x - mean) ** 2 for x in nets) / (n - 1)
        se = math.sqrt(var / n)
        ci = [mean - 1.96 * se, mean + 1.96 * se]
    else:
        se, ci = None, None
    return {
        "n_shards": len(per_shard), "n_shards_valid": len(valid), "n_shards_invalid": len(invalid),
        "decks": n,
        "mean_chips_per_deck": mean,
        "ci95_chips_per_deck": ci,
        "ci_half_width_chips_per_deck": (1.96 * se) if se is not None else None,
        "mean_bb100": (mean / CHIPS_PER_DECK_PER_BB100) if mean is not None else None,
        "ci95_bb100": ([c / CHIPS_PER_DECK_PER_BB100 for c in ci] if ci else None),
        "verdict": (None if not n or ci is None else
                    ("A STRONGER (CI excludes 0)" if ci[0] > 0 else
                     ("B STRONGER (CI excludes 0)" if ci[1] < 0 else "INCONCLUSIVE (CI includes 0)"))),
        "proof_of_search": {**agg, "by_street_pooled": streets},
        "per_shard": per_shard,
        "invalid_shards": invalid,
    }


def _load(pattern: str) -> list[tuple[str, dict]]:
    out = []
    for p in sorted(glob.glob(pattern)):
        try:
            out.append((p, json.load(open(p))))
        except Exception as e:  # noqa: BLE001
            print(f"[assemble] skip {p}: {e!r}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default="cert9")
    ap.add_argument("--min-mean-wall", type=float, default=0.1)
    args = ap.parse_args()
    od = args.out_dir

    probe = None
    pth = os.path.join(od, "probe_ab.json")
    if os.path.exists(pth):
        probe_res = json.load(open(pth))
        probe = evaluate(probe_res, arm="a", min_postflop=10,
                         min_mean_wall_s=args.min_mean_wall)

    ab = _pool(_load(os.path.join(od, "cert9_ab_seed*_shard*.json")),
               require_gate=True, min_mean_wall_s=args.min_mean_wall)
    inc = _pool(_load(os.path.join(od, "cert9_inc_seed*_shard*.json")),
                require_gate=True, min_mean_wall_s=args.min_mean_wall)

    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        rev = "unknown"

    verdict = {
        "run_id": args.run_id,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_rev": rev,
        "methodology": "CRN-paired seat-rotated engine_ab_h2h; arm A = tier-1 126k serve index + "
                       "K=2000 native search (strict, fail-closed); arm B = same index, search off. "
                       "chips/deck -> bb/100 factor 6.",
        "probe_pre_spend": probe,
        "primary_search_minus_raw": ab,
        "secondary_search_vs_incumbent": inc if inc["n_shards"] else None,
        "proof_of_search_required": {
            "decisions_fallback": 0,
            "postflop_search_native_frac": 1.0,
            "min_mean_postflop_wall_s": args.min_mean_wall,
        },
    }
    out_path = os.path.join(od, "VERDICT_FINAL.json")
    json.dump(verdict, open(out_path, "w"), indent=1)
    print(json.dumps({k: verdict[k] for k in ("run_id", "generated_utc")}, indent=1))
    m = ab.get("mean_chips_per_deck")
    print(f"[assemble] PRIMARY search-raw: {m if m is None else round(m, 2)} chips/deck "
          f"CI {ab.get('ci95_chips_per_deck')} over {ab.get('decks')} decks "
          f"({ab.get('n_shards_valid')}/{ab.get('n_shards')} shards valid)", flush=True)
    print(f"[assemble] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
