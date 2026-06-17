"""Local convergence comparison: CFR+ vs DCFR vs VR-MCCFR on the 6-max NLHE abstraction.

Runs each variant for the SAME traversal budget (multi-round reseed so regret CONTINUES across rounds),
then measures a convergence proxy: TabularBlueprintBot bb/100 vs a FIXED weak field (same field+seed for
all variants, so the only difference is the training variant). A more-converged blueprint exploits the
field harder. This is the data that informs whether a DCFR/VR cloud RERUN is worth it — NOT a substitute
for the real cloud A/B, but a cheap directional signal.

Honest caveat: local budgets are tiny (the blueprint is extremely coarse for ALL variants), so a null /
noisy result is itself informative ("local scale can't distinguish -> don't restart blind").

Usage: python scripts/compare_cfr_variants.py [--rounds 5 --traversals 20 --eval-hands 1200]
"""

from __future__ import annotations

import argparse
import os
import time


def _accumulate(variant: str, rounds: int, traversals: int, capacity: int, seed: int):
    from pokerbot.training.sd_cfr_nlhe_hogwild import dcfr_discount_snapshots, run_nlhe_hogwild

    reg, strat = None, None
    use_baseline = variant == "vr"
    cfr_plus = variant != "dcfr"  # DCFR keeps signed regret + discounts; CFR+/VR floor at 0
    for t in range(1, rounds + 1):
        res = run_nlhe_hogwild(
            traversals=traversals, num_workers=1, capacity=capacity,
            occupancy=(True,) * 6, client_bet_sizing=True, cfr_plus=cfr_plus, seed=seed + t,
            init_strat=strat, init_regret=reg, use_baseline=use_baseline,
        )
        reg, strat = res["regret"], res["strat"]
        if variant == "dcfr":
            dcfr_discount_snapshots(reg, strat, t)
    return strat, res["n_infosets"]


def main() -> None:
    os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--traversals", type=int, default=20)
    p.add_argument("--capacity", type=int, default=1 << 18)
    p.add_argument("--eval-hands", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import random

    from pokerbot.eval.bots import BOT_REGISTRY
    from pokerbot.eval.multiway_table import run_multiway
    from pokerbot.training.sd_cfr_nlhe_hogwild import TabularBlueprintBot

    # FIXED weak field (same instances/seed for every variant -> controlled comparison).
    field = [BOT_REGISTRY["random"](random.Random(7 + i)) for i in range(6)]

    print(f"budget = {args.rounds} rounds x {args.traversals} traversals (x6 seats); eval {args.eval_hands} "
          f"hands vs fixed random field\n", flush=True)
    results = []
    for variant in ("cfr_plus", "dcfr", "vr"):
        t0 = time.time()
        snap, n_info = _accumulate(variant, args.rounds, args.traversals, args.capacity, args.seed)
        train_s = time.time() - t0
        bot = TabularBlueprintBot(snapshot=snap, client_bet_sizing=True)
        r = run_multiway(bot, field_bots=list(field), hands=args.eval_hands, seed=99, duplicate=True)
        results.append((variant, n_info, r, train_s))
        print(f"{variant:9s} infosets={n_info:6d} train={train_s:5.1f}s  -> {r.summary()}", flush=True)

    print("\n=== CONVERGENCE PROXY (bb/100 vs fixed random field; higher = better exploitation) ===", flush=True)
    base = next(r.bb_per_100 for v, _, r, _ in results if v == "cfr_plus")
    for v, n, r, _ in results:
        delta = r.bb_per_100 - base
        print(f"  {v:9s} {r.bb_per_100:+9.2f} +/- {r.ci95_bb_per_100:8.2f} bb/100   "
              f"(delta vs CFR+ {delta:+.1f})", flush=True)
    print("\nInterpretation: a Δ inside the CIs => indistinguishable at local scale (expected; don't restart "
          "blind). A clear, CI-separated improvement => a controlled cloud rerun may be justified.", flush=True)


if __name__ == "__main__":
    main()
