#!/usr/bin/env python3
"""
Diagnostic: which dimension of the infoset fingerprint actually drives table cardinality?

The fingerprint is (weva_id, cards.tobytes(), actions.tobytes(), policy_position.tobytes()).
We run a few smoke iterations and decompose the keys of `_cum_regret` to see, per iteration,
how many DISTINCT values each component has and whether the count saturates. This tells us what
to bucket (cards vs actions) before building an abstraction — profile, don't assume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer


def _component_counts(keys) -> dict:
    full = set()
    weva = set()
    cards = set()
    actions = set()
    pos = set()
    weva_cards = set()  # (weva_id, cards) — the "hand identity" incl. exact holes+board
    cards_actions = set()
    for k in keys:
        full.add(k)
        # key is (weva_id, cards_bytes, actions_bytes, pos_bytes)
        if isinstance(k, tuple) and len(k) >= 4:
            weva.add(k[0])
            cards.add(k[1])
            actions.add(k[2])
            pos.add(k[3])
            weva_cards.add((k[0], k[1]))
            cards_actions.add((k[1], k[2]))
    return {
        "full": len(full),
        "distinct_weva_id": len(weva),
        "distinct_cards": len(cards),
        "distinct_actions": len(actions),
        "distinct_position": len(pos),
        "distinct_weva_x_cards": len(weva_cards),
        "distinct_cards_x_actions": len(cards_actions),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--traversals", type=int, default=4)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="logs/infoset_cardinality.json")
    args = p.parse_args()

    cfg = SDCFRNLHEConfig(
        iterations=1, traversals_per_iter=args.traversals, deal_sweeps=1,
        train_batch_size=256, sft_checkpoint=None, seed=args.seed,
        max_traverse_depth=args.depth,
    )
    trainer = SDCFRNLHETrainer(cfg)
    rows = []
    print(f"{'iter':>4} {'full':>10} {'weva':>8} {'cards':>10} {'actions':>10} "
          f"{'weva*cards':>11} {'cards*act':>11}", flush=True)
    for n in range(1, args.iters + 1):
        np.random.seed(cfg.seed + n - 1)
        trainer.train_iteration(float(n))
        c = _component_counts(trainer._cum_regret.keys())
        c["iter"] = n
        rows.append(c)
        print(f"{n:>4} {c['full']:>10} {c['distinct_weva_id']:>8} {c['distinct_cards']:>10} "
              f"{c['distinct_actions']:>10} {c['distinct_weva_x_cards']:>11} "
              f"{c['distinct_cards_x_actions']:>11}", flush=True)

    # Growth deltas to see what's NOT saturating.
    print("\nper-iter delta (new keys added):", flush=True)
    print(f"{'iter':>4} {'d_full':>10} {'d_cards':>10} {'d_actions':>10} {'d_weva':>8}", flush=True)
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        print(f"{b['iter']:>4} {b['full']-a['full']:>10} {b['distinct_cards']-a['distinct_cards']:>10} "
              f"{b['distinct_actions']-a['distinct_actions']:>10} "
              f"{b['distinct_weva_id']-a['distinct_weva_id']:>8}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)
    # Interpretation hint
    last = rows[-1]
    print("\nINTERPRETATION:", flush=True)
    print(f"  full keys = {last['full']}; distinct cards = {last['distinct_cards']} "
          f"({100*last['distinct_cards']/max(last['full'],1):.0f}% of full); "
          f"distinct actions = {last['distinct_actions']}", flush=True)
    print("  If distinct_cards ~ full and keeps growing -> CARDS (hole+board) drive cardinality "
          "-> bucket hands (A2).", flush=True)
    print("  If distinct_actions ~ full and cards saturate -> betting history drives it -> "
          "discretize action key (A1).", flush=True)


if __name__ == "__main__":
    main()
