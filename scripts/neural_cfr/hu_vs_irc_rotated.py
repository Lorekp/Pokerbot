"""Seat-rotated duplicate HU win-rate of a HU policy vs the IRC HU field (Python eval, NO Rust crate).

Closes the Workstream-E harness gap: a seat-rotated duplicate HU win-rate cell that, given ANY HU
policy_fn, reports bb/100 + CI vs an IRC-calibrated heads-up villain. Plugs in:
  * --policy neural  : the trained AverageStrategyNet (hu_lbr_neural.make_neural_policy)
  * --policy uniform : harness calibration (uniform-legal; should be a CLEAR loss vs a real villain)
  * --policy fold    : harness calibration (always-fold; should lose ~ the blinds)

The number for the dev/checkpoint net is a CURRENT-POLICY baseline -- NOT the converged target (which
is gated on the ~24h converged training run, see docs/hu_engine_status.md).

Run (WSL Debian; CPU jax is fastest for single-example serves):
  source ~/jaxgpu-venv/bin/activate
  cd .
  PYTHONPATH=. POKERBOT_INFO_SUITS=1 JAX_PLATFORMS=cpu \
    python scripts/neural_cfr/hu_vs_irc_rotated.py \
      --checkpoint checkpoints/sd_cfr_hu_neural/hu_vrpdcfr_v2.pkl \
      --villain irc_station --decks 600 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time

from pokerbot.eval.hu_irc_field import (
    HUFieldConfig,
    hu_vs_irc_rotated,
    list_archetypes,
)
from pokerbot.eval.lbr import always_fold_policy, uniform_policy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/sd_cfr_hu_neural/hu_vrpdcfr_v2.pkl")
    ap.add_argument("--policy", choices=["neural", "uniform", "fold"], default="neural")
    ap.add_argument("--villain", choices=list_archetypes(), default="irc_station")
    ap.add_argument("--all-villains", action="store_true",
                    help="run vs every IRC archetype (station/lag/nit/tag) in turn.")
    ap.add_argument("--decks", type=int, default=600,
                    help="distinct deals; each costs 2 hero hands (seat-rotated duplicate).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stack-bb", type=float, default=None,
                    help="deal at this effective stack depth (BB). Default = engine 100bb. "
                         "Set e.g. 20 for the tournament-shallow regime.")
    ap.add_argument("--ante-bb", type=float, default=0.0)
    ap.add_argument("--villain-rollouts", type=int, default=80,
                    help="phevaluator runout samples for the villain's equity gate (speed/variance).")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.policy == "neural":
        from scripts.neural_cfr.hu_lbr_neural import make_neural_policy
        hero_policy, meta = make_neural_policy(args.checkpoint)
        label = f"neural:{args.checkpoint}"
    elif args.policy == "uniform":
        hero_policy, meta, label = uniform_policy, {"policy": "uniform"}, "uniform"
    else:
        hero_policy, meta, label = always_fold_policy, {"policy": "always_fold"}, "fold"

    villains = list_archetypes() if args.all_villains else [args.villain]
    print(f"HU-vs-IRC seat-rotated duplicate win-rate")
    print(f"  hero policy = {label}  meta={meta}")
    print(f"  decks={args.decks} seed={args.seed} stack_bb={args.stack_bb} ante_bb={args.ante_bb} "
          f"villain_rollouts={args.villain_rollouts}")
    print(f"  (CURRENT-policy baseline unless the net is the CONVERGED blueprint -- see "
          f"docs/hu_engine_status.md)\n")

    out_rows = []
    for v in villains:
        cfg = HUFieldConfig(decks=args.decks, seed=args.seed, villain=v, stack_bb=args.stack_bb,
                            ante_bb=args.ante_bb, villain_rollouts=args.villain_rollouts)
        t0 = time.time()
        res = hu_vs_irc_rotated(hero_policy, cfg)
        dt = time.time() - t0
        print(f"  vs {v:<12} bb/100 = {res.bb_per_100:+8.2f}  +/- {res.bb_per_100_ci95:6.2f} (95% CI)"
              f"   [decks={res.decks} hero_hands={res.hero_hands} wall={dt:.1f}s]")
        out_rows.append({
            "villain": v, "bb_per_100": res.bb_per_100, "bb_per_100_ci95": res.bb_per_100_ci95,
            "bb_per_hand": res.bb_per_hand, "stderr_bb_per_hand": res.stderr_bb_per_hand,
            "decks": res.decks, "hero_hands": res.hero_hands, "wall_s": dt, "detail": res.detail,
        })

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"policy": label, "meta": meta, "seed": args.seed, "decks": args.decks,
                       "results": out_rows}, f, indent=2)
        print(f"\n  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
