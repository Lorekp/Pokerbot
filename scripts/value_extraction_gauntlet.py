#!/usr/bin/env python3
"""Phase-2 value-extraction gauntlet: can the UNOPENED value-bet exploit be turned ON on the
blueprint+search base WITHOUT harming play vs aggressive opponents?

The validated facing-branch exploit is do-no-harm (per-seat targeting shipped). The VALUE-EXTRACTION
(unopened) branch — value-bet stations thinner/more, the actual "extract from fish" lever — is OFF
because it couldn't be gated do-no-harm against the hs_* bots (their aggression collapsed to forced
checks, so the read couldn't tell a station from a lag). This gauntlet re-tests that branch against the
realistic IRC villains (pokerbot.eval.irc_villains), which express aggression as real sized bets and so
give the read a true station-vs-lag signal.

ARMS (CRN-paired, per-seat profiles = live behaviour):
  OFF = pure blueprint+search base (opponent_model=None).
  ON  = base + full exploit (solver opponent tail + root mixer with the UNOPENED branch enabled).

VERDICT (both must hold to ship the unopened branch):
  * VALUE: ON - OFF > 0 on the station-heavy fields (irc_soft / irc_station) beyond noise.
  * DO-NO-HARM: ON - OFF >= ~-0.05 bb/100 on the lag/tag fields (irc_tough / irc_lag / irc_tag).
It also prints the per-seat learned init_agg so you can SEE the station-vs-lag separation the gate needs.

  python scripts/value_extraction_gauntlet.py --blueprint-snapshot checkpoints/blueprint_bounded_full_iter200.pkl \
      --hands 200 --seeds 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
from pathlib import Path

# Blueprint serve keys must match the trained table postflop (S1-FIX): set before any pokerbot import.
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "emd")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "160")
# EMD-K160 deployed regime (promoted 2026-07-02, certified +32.89 bb/100 vs made-hand iter62000):
# frozen flop/turn EMD lookup + train-mirrored made-hand fallback on the river (the river is not
# in the blob; without the mirror flag the river frozen-miss serves a UNIFORM policy).
os.environ.setdefault("POKERBOT_POSTFLOP_EMD_FROZEN", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn")))
os.environ.setdefault("POKERBOT_EMD_FROZEN_MIRROR_FALLBACK", "1")

# Station-heavy = value target; lag/tag-heavy = do-no-harm protection check.
FIELDS = {
    "irc_soft": ["irc_station", "irc_station", "irc_station", "irc_nit", "irc_tag"],
    "irc_mixed": ["irc_station", "irc_station", "irc_lag", "irc_nit", "irc_tag"],
    "irc_tough": ["irc_lag", "irc_lag", "irc_tag", "irc_tag", "irc_nit"],
    "irc_station": ["irc_station"] * 5,
    "irc_lag": ["irc_lag"] * 5,
    "irc_pros": ["irc_tag"] * 5,   # all near-solid TAGs: the professional-table do-no-harm test
}
WEAK_FIELDS = ("irc_soft", "irc_station", "irc_mixed")
TOUGH_FIELDS = ("irc_tough", "irc_lag", "irc_pros")


def _pool(xs):
    k = len(xs)
    if k == 0:
        return 0.0, 0.0
    m = sum(xs) / k
    se = (math.sqrt(sum((x - m) ** 2 for x in xs) / (k - 1)) / math.sqrt(k)) if k > 1 else 0.0
    return m, se


def make_factory(snap, args):
    from pokerbot.eval.production_hero import ProductionHero
    from pokerbot.eval.search_agent import SearchAgent

    def factory(*, exploit: bool):
        agent = SearchAgent(
            use_pluribus_search=True,
            blueprint_snapshot=snap,
            pluribus_leaf_select=args.leaf_select,
            time_budget_ms=10 ** 9,
            transformer_checkpoint=None,
            hero_seat=0,
            root_exploit_mixer=True,        # full exploit on the ON arm (inert when profiles=None)
            root_mixer_unopened=True,        # <-- the value-extraction branch under test
            aivat_leaf_q=False,
        )
        agent.solver.config.max_iterations = args.iterations
        agent.solver.config.time_budget_ms = 10 ** 9
        return ProductionHero(agent, random.Random(args.seed), exploit=exploit, hero_seat=0, n_seats=6)

    return factory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint-snapshot", required=True)
    ap.add_argument("--hands", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--leaf-select", default="uniform", choices=("uniform", "gadget"))
    ap.add_argument("--fields", default=",".join(FIELDS))
    ap.add_argument("--out", default="runs/value_extraction_gauntlet.json")
    args = ap.parse_args()

    from pokerbot.eval.exploitation_battery import run_field_absolute
    from pokerbot.eval.irc_villains import build_irc_villain

    with open(args.blueprint_snapshot, "rb") as f:
        snap_full = pickle.load(f)
    snap = snap_full["strat_sum"] if isinstance(snap_full, dict) and "strat_sum" in snap_full else snap_full
    print(f"blueprint: {len(snap)} infosets; iters={args.iterations} leaf={args.leaf_select}", flush=True)
    factory = make_factory(snap, args)

    results = {}
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    print(f"{'field':12s} {'base bb/100':>12s} {'+expl bb/100':>13s} {'value(on-off)':>14s} {'k':>3s}")
    for fname in fields:
        spec = [(b, None) for b in FIELDS[fname]]
        offs, ons = [], []
        for s in range(args.seeds):
            r = run_field_absolute(spec, hands=args.hands, seed=args.seed + s, hero_factory=factory,
                                   allin_ev=True, per_seat_profiles=True, bot_builder=build_irc_villain,
                                   progress_every=0)
            offs.append(r["off_aivat"]["adj_mean_bb"])
            ons.append(r["on_aivat"]["adj_mean_bb"])
        om, ose = _pool(offs)
        nm, nse = _pool(ons)
        results[fname] = {"off": om, "off_se": ose, "on": nm, "on_se": nse, "k": len(offs),
                          "value": nm - om}
        tag = "weak" if fname in WEAK_FIELDS else ("tough" if fname in TOUGH_FIELDS else "")
        print(f"{fname:12s} {om:>+8.2f}+-{ose:>4.2f} {nm:>+9.2f}+-{nse:>4.2f} "
              f"{nm - om:>+14.2f} {len(offs):>3d}  {tag}", flush=True)

    weak_val = [results[f]["value"] for f in fields if f in WEAK_FIELDS]
    tough_val = [results[f]["value"] for f in fields if f in TOUGH_FIELDS]
    summary = {
        "value_vs_weak": (sum(weak_val) / len(weak_val)) if weak_val else None,
        "harm_vs_tough": (min(tough_val)) if tough_val else None,
        "fields": results,
        "config": {"hands": args.hands, "seeds": args.seeds, "iterations": args.iterations,
                   "leaf_select": args.leaf_select},
    }
    if weak_val:
        print(f"\nMEAN value extracted vs WEAK = {summary['value_vs_weak']:+.2f} bb/100")
    if tough_val:
        worst = summary["harm_vs_tough"]
        print(f"WORST harm vs TOUGH          = {worst:+.2f} bb/100 "
              f"({'DO-NO-HARM OK' if worst >= -0.05 else 'HARMS — keep unopened OFF'})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
