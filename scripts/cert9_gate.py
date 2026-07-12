#!/usr/bin/env python3
"""CERT9 proof-of-search gate: validate that an engine_ab_h2h shard/probe JSON shows the search
arm GENUINELY running native K=2000 search — the mandatory pre-spend gate and the per-shard
validity criterion.

WHY THIS EXISTS: in cert7/cert8 the "with search" arms never searched — the native engine build
crashed on the mmap-index blueprint and make_search_policy_fn's except silently served the RAW
blueprint (cert7's search-h2h shard did 200 decks in 53 s wall). This gate makes that failure mode
a hard, machine-checked verdict on every shard:

  PASS requires (on the designated search arm, default 'a'):
    * runtime stats present, collected under POKERBOT_SEARCH_STRICT=1;
    * native caps include "subgame_index_dir" (the 033a172 index-dir engine ctor);
    * decisions_fallback == 0 (no raw-blueprint serve, ever);
    * postflop decisions >= --min-postflop, 100% of them served "search_native";
    * decision-count-weighted mean postflop wall >= --min-mean-wall (default 0.1 s: local quiet-box
      reference at K=2000 on the tier-1 index is flop p50 ~5.6 s / turn ~1.8 s / river ~0.02 s,
      mean ~5 s — a 0.01 s/decision arm is NOT searching).

Exit 0 = PASS (prints a JSON verdict), exit 1 = FAIL (probe: terminate the box, INFRA_FAIL;
shard: mark INVALID, not a money number).
"""
from __future__ import annotations

import argparse
import json
import sys


def evaluate(res: dict, *, arm: str = "a", min_postflop: int = 20,
             min_mean_wall_s: float = 0.1, check_wall: bool = True) -> dict:
    reasons: list[str] = []
    stats = res.get(f"arm_{arm}_runtime_stats") or {}
    ss = stats.get("search_stats")
    if ss is None:
        reasons.append(f"arm_{arm} has no search_stats (not the instrumented search policy)")
        return {"pass": False, "reasons": reasons}

    if stats.get("strict_env", ss.get("strict")) not in ("1", True):
        reasons.append("search arm is not running strict mode (POKERBOT_SEARCH_STRICT=1)")

    caps = stats.get("native_caps") or []
    if "subgame_index_dir" not in caps:
        reasons.append(f"native caps missing 'subgame_index_dir' (caps={caps})")

    fb = int(ss.get("decisions_fallback", -1))
    if fb != 0:
        reasons.append(f"decisions_fallback={fb} (must be 0)")

    post = int(ss.get("postflop_decisions", 0))
    post_native = int(ss.get("postflop_search_native", 0))
    if post < min_postflop:
        reasons.append(f"postflop_decisions={post} < required {min_postflop}")
    if post_native != post:
        reasons.append(f"postflop_search_native={post_native}/{post} (must be 100%)")

    # Decision-count-weighted mean postflop wall from the per-street summaries.
    wsum, n = 0.0, 0
    for st in ("1", "2", "3"):
        row = (ss.get("by_street") or {}).get(st)
        if row and row.get("wall_mean_s") is not None:
            wsum += float(row["wall_mean_s"]) * int(row["n"])
            n += int(row["n"])
    mean_wall = (wsum / n) if n else 0.0
    if check_wall and mean_wall < min_mean_wall_s:
        reasons.append(f"mean postflop wall {mean_wall:.4f}s < {min_mean_wall_s}s — search is "
                       "NOT genuinely running (fallback-speed decisions)")

    decks = int(res.get("decks_done", res.get("decks", 0)) or 0)
    wall_s = float(res.get("wall_s", res.get("wall_s_so_far", 0.0)) or 0.0)
    return {
        "pass": not reasons,
        "reasons": reasons,
        "arm": arm,
        "strict": stats.get("strict_env", ss.get("strict")),
        "native_caps_has_index_dir": "subgame_index_dir" in caps,
        "decisions_total": int(ss.get("decisions_total", 0)),
        "decisions_fallback": fb,
        "postflop_decisions": post,
        "postflop_search_native": post_native,
        "mean_postflop_wall_s": mean_wall,
        "by_street": ss.get("by_street"),
        "decks": decks,
        "wall_s": wall_s,
        "sec_per_deck": (wall_s / decks) if decks else None,
        "bp_misses": stats.get("bp_misses"),
        "bp_hits": stats.get("bp_hits"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", help="engine_ab_h2h output JSON (probe or shard)")
    ap.add_argument("--arm", default="a", choices=("a", "b"))
    ap.add_argument("--min-postflop", type=int, default=20)
    ap.add_argument("--min-mean-wall", type=float, default=0.1)
    ap.add_argument("--no-wall-check", action="store_true",
                    help="skip the wall floor (shard-validity mode on a contended box)")
    ap.add_argument("--out", default=None, help="write the gate verdict JSON here too")
    args = ap.parse_args()

    res = json.load(open(args.result_json))
    v = evaluate(res, arm=args.arm, min_postflop=args.min_postflop,
                 min_mean_wall_s=args.min_mean_wall, check_wall=not args.no_wall_check)
    print(json.dumps(v, indent=1))
    if args.out:
        json.dump(v, open(args.out, "w"), indent=1)
    if v["pass"]:
        print("[cert9_gate] PASS — search arm is genuinely native-searching", flush=True)
        return 0
    print("[cert9_gate] FAIL — " + "; ".join(v["reasons"]), flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
