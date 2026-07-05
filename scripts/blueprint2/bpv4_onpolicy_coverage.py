#!/usr/bin/env python3
"""blueprint-v4 Stage-0 diagnostic: ON-POLICY per-street fingerprint coverage of a checkpoint.

Plays N full 6-max hands where EVERY seat samples the checkpoint's own average strategy
(TabularBlueprintAgent — the same serve twin the G5 probe uses), and reports the per-street
blueprint hit/miss counts (POKERBOT_BP_COVERAGE). This separates the two readings of a low
probe coverage number:

  * forced-line starvation — the probe forces opens the arm rarely plays, so ITS postflop
    subtrees are untrained, but the arm's OWN lines are covered (on-policy hit-rate HIGH);
  * table-wide postflop starvation / preflop row domination — even the arm's own lines miss
    (on-policy hit-rate LOW): the fullhist postflop space is so large that 20k iters left the
    postflop tail essentially unpopulated everywhere.

Usage (H200):
  python scripts/blueprint2/bpv4_onpolicy_coverage.py --blueprint <ckpt> --hands 400 \
      --key-mode fullhist --postflop-mode fast --num-buckets 200 --mrr \
      --snapshot-index artifacts/bpv4_ladder/idx_h200_20k --out artifacts/_bpv4_onpolicy_H200.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import bp3_forced_response_probe as probe  # noqa: E402  (reuses _configure_env + _IndexedSnap)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--hands", type=int, default=400)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--key-mode", choices=("prefhist", "fullhist", "summary"), default="fullhist")
    ap.add_argument("--postflop-mode", choices=("emd", "fast"), default="fast")
    ap.add_argument("--num-buckets", type=int, default=200)
    ap.add_argument("--mrr", action="store_true")
    ap.add_argument("--emd-frozen", default="data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn")
    ap.add_argument("--river-ochs", default="data/abstraction/river_ochs_160.rochs")
    ap.add_argument("--snapshot-index", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    probe._configure_env(args)  # env BEFORE pokerbot imports (standing rule)
    import random

    import numpy as np

    from pokerbot.engine.game import GameState, play_hand
    from pokerbot.eval import lbr as _lbr
    from pokerbot.eval.blueprint_ranking import TabularBlueprintAgent

    if args.snapshot_index:
        snap = probe._IndexedSnap(args.snapshot_index)
    else:
        from f32_regret_strategy_delta import avg_strategy, load_checkpoint
        ck = load_checkpoint(args.blueprint)
        snap = {bytes(k): np.asarray(avg_strategy(row), dtype=np.float64)
                for k, row in ck["strat"].items()}
        del ck
    print(f"snapshot {len(snap):,} rows; key-mode={args.key_mode} mrr={args.mrr}", flush=True)

    agents = {s: TabularBlueprintAgent(snap, random.Random(1000 + s)) for s in range(6)}
    t0 = time.time()
    end_street = {}
    for h in range(args.hands):
        rng = random.Random(args.seed * 77 + h)
        state = GameState.new_hand(rng=rng)
        for s, ag in agents.items():
            ag.reseed(args.seed * 1_000_003 + h * 31 + s)
        final, _ = play_hand({s: agents[s] for s in range(6)}, state=state)
        es = len(final.board)
        end_street[str(es)] = end_street.get(str(es), 0) + 1

    cov = _lbr.bp_coverage_report()
    fb = sum(a.fallback_count for a in agents.values())
    dc = sum(a.decision_count for a in agents.values())
    pf_h = sum(cov.get(s, {}).get("hits", 0) for s in ("flop", "turn", "river"))
    pf_m = sum(cov.get(s, {}).get("misses", 0) for s in ("flop", "turn", "river"))
    res = {"mode": "onpolicy_coverage", "blueprint": args.blueprint, "hands": args.hands,
           "seed": args.seed, "key_mode": args.key_mode, "mrr": bool(args.mrr),
           "adapter_fallback_rate": fb / max(1, dc),
           "bp_coverage": cov,
           "postflop_miss_rate": pf_m / max(1, pf_h + pf_m),
           "end_street_board_cards": end_street,
           "wall_s": round(time.time() - t0, 1)}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("adapter_fallback_rate", "postflop_miss_rate",
                                          "bp_coverage", "end_street_board_cards")}, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
