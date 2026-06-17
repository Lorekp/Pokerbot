"""P3 driver: measure each layer of the Pluribus 6-max foundation — blueprint < +search < +exploit.

Loads a trained 6-max blueprint snapshot (``strat_sum`` from a hogwild checkpoint) and runs the multiway
harness (bb/100 ± 95% CI, seat rotation + duplicate dealing) for three hero configs:
  (i)   blueprint-alone   — TabularBlueprintBot (no search)
  (ii)  +Pluribus search  — SearchAgent(use_pluribus_search) with the blueprint, exploit OFF
  (iii) +exploit          — same, with an opponent_model so the firewalled exploit tail activates

The blueprint-alone config is fast (run many hands → a real signal: does it beat random/nit?, the P0 gate).
The search configs are ~seconds/decision, so default to a SMALL hand count — this validates the pipeline
end-to-end on real trained data; statistical significance on the final blueprint uses AIVAT + many hands.

Usage:
  python scripts/eval_pluribus_layers.py --checkpoint <path.pkl|s3://...> [--bp-hands 1200]
         [--search-hands 12] [--leaf-select uniform] [--aivat]
"""

from __future__ import annotations

import argparse
import os
import pickle
import random


def _load_snapshot(path: str) -> dict:
    if path.startswith("s3://"):
        import subprocess
        import tempfile

        dst = os.path.join(tempfile.gettempdir(), "pluribus_bp.pkl")
        subprocess.run(["aws", "s3", "cp", path, dst, "--region", "eu-central-1"], check=True)
        path = dst
    with open(path, "rb") as f:
        ck = pickle.load(f)
    snap = ck["strat_sum"] if isinstance(ck, dict) and "strat_sum" in ck else ck
    return snap


def _exploit_profiles(n_seats: int):
    """A leaky (calling-station-ish) profile per seat so the exploit tail activates (pipeline check)."""
    from pokerbot.opponent.stats import OpponentStats

    prof = OpponentStats(hands=120)
    # Loose-passive markers -> archetype() == 'calling_station' -> apply_opponent_bias tilts to value.
    for attr, val in (("vpip", 0.62), ("pfr", 0.06), ("aggression_calls", 80), ("aggression_bets_raises", 8)):
        if hasattr(prof, attr):
            try:
                setattr(prof, attr, val)
            except Exception:  # noqa: BLE001
                pass
    return {s: prof for s in range(n_seats)}


def main() -> None:
    os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="local .pkl path or s3:// URL")
    p.add_argument("--bp-hands", type=int, default=1200, help="hands for the fast blueprint-only config")
    p.add_argument("--search-hands", type=int, default=12, help="hands for the slow search configs")
    p.add_argument("--leaf-select", choices=["uniform", "gadget"], default="uniform")
    p.add_argument("--time-budget-ms", type=int, default=800)
    p.add_argument("--aivat", action="store_true", help="apply the AIVAT decision+chance CV to search configs")
    p.add_argument("--no-aivat-chance", action="store_true", help="AIVAT decision term only (skip the board-reveal chance CV)")
    p.add_argument("--aivat-flop", action="store_true", help="also apply the (sampled) flop chance term; default is turn+river only")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import numpy as np  # noqa: F401  (engine import side effects honor the env var set above)

    from pokerbot.eval.multiway_table import run_multiway
    from pokerbot.eval.search_agent import SearchAgent
    from pokerbot.engine.game import NUM_PLAYERS
    from pokerbot.training.sd_cfr_nlhe_hogwild import TabularBlueprintBot

    snap = _load_snapshot(args.checkpoint)
    print(f"Loaded blueprint snapshot: {len(snap)} infosets from {args.checkpoint}", flush=True)

    rows = []

    # (i) blueprint-alone — fast, many hands.
    bp_bot = TabularBlueprintBot(snapshot=snap, client_bet_sizing=True)
    r_bp = run_multiway(bp_bot, hands=args.bp_hands, seed=args.seed, duplicate=True)
    rows.append(("blueprint-alone", r_bp))
    print(f"[i]  blueprint-alone : {r_bp.summary()}", flush=True)

    # (ii) +Pluribus search, exploit OFF.
    hero_search = SearchAgent(
        use_pluribus_search=True, blueprint_snapshot=snap,
        time_budget_ms=args.time_budget_ms, pluribus_leaf_select=args.leaf_select,
    )
    aivat_kw = dict(
        aivat=args.aivat,
        aivat_chance=not args.no_aivat_chance,
        aivat_flop=args.aivat_flop,
    )
    r_se = run_multiway(hero_search, hands=args.search_hands, seed=args.seed, duplicate=True, **aivat_kw)
    rows.append(("+search(exploitOFF)", r_se))
    vr_msg = ""
    if args.aivat:
        vr = r_se.detail.get("variance_reduction", 0.0)
        vr_d = r_se.detail.get("variance_reduction_decision_only", 0.0)
        vr_msg = f"  varRed(dec+chance)={vr:+.2f} (dec-only={vr_d:+.2f})"
    print(f"[ii] +search exploitOFF: {r_se.summary()}" + vr_msg, flush=True)

    # (iii) +exploit — opponent_model on (firewalled exploit tail).
    hero_exp = SearchAgent(
        use_pluribus_search=True, blueprint_snapshot=snap,
        time_budget_ms=args.time_budget_ms, pluribus_leaf_select=args.leaf_select,
    )
    hero_exp.solver.opponent_model = _exploit_profiles(NUM_PLAYERS)
    r_ex = run_multiway(hero_exp, hands=args.search_hands, seed=args.seed, duplicate=True, **aivat_kw)
    rows.append(("+exploit", r_ex))
    print(f"[iii]+exploit          : {r_ex.summary()}", flush=True)

    print("\n=== LAYER SUMMARY (bb/100 ± 95% CI) ===", flush=True)
    for name, r in rows:
        flag = "  <-- CI>0" if r.beats_zero else ""
        print(f"  {name:22s} {r.bb_per_100:+9.2f} ± {r.ci95_bb_per_100:8.2f}  (n={r.hands}){flag}", flush=True)
    print("\nNOTE: search configs use few hands (pipeline validation, not significance). The final blueprint "
          "+ AIVAT + many hands produces the real layered numbers.", flush=True)


if __name__ == "__main__":
    main()
