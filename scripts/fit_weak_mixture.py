#!/usr/bin/env python3
"""Fit the weak_mixture gauntlet (G3): map IRC per-context PERCENTILES onto HS-bot thresholds.

For each weak archetype x percentile level, BISECTS the identity-defining threshold until the
bot's REALIZED frequency (measured by simulation with the battery's exact context/bin machinery)
hits the IRC population target:

  hs_station x {p10,p25,p50 postflop-facing FOLD}   -> call_hs   (the sticky tail of real fields)
  hs_nit     x {p75,p90 fold} + agg p25 cap         -> call_hs   (the foldy tail)
  hs_lag     x {p50,p75,p90 postflop-unopened AGGRO} -> bet_hs   (the aggro tail)

Output: data/player_profiles/irc_weak_mixture.json — consumed by run_archetype("weak_mixture")
(one seed-deterministic entry per replica ⇒ pooled replicas measure uplift vs the fitted
weak-player DISTRIBUTION). Costs ~a minute; no battery.

Usage: python scripts/fit_weak_mixture.py [--hands 600] [--out data/player_profiles/irc_weak_mixture.json]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

REPO = Path(__file__).resolve().parents[1]
PCT_PATH = REPO / "data" / "player_profiles" / "irc_population_percentiles.json"

_POSTFLOP_FACING = [f"{st}|{tx}|facing_bet" for st in ("flop", "turn", "river")
                    for tx in ("dry", "wet", "paired")]
_POSTFLOP_UNOPENED = [f"{st}|{tx}|unopened" for st in ("flop", "turn", "river")
                      for tx in ("dry", "wet", "paired")]


def _pooled_target(pct: dict, keys: list[str], bin_name: str, level: str,
                   extra_bin: str | None = None) -> float:
    """n_players-weighted mean of a percentile level over a context family."""
    num = den = 0.0
    for ck in keys:
        e = pct.get(ck)
        if not e:
            continue
        w = e["n_players"]
        v = e["bins"][bin_name][level]
        if extra_bin:
            v += e["bins"][extra_bin][level]
        num += w * v
        den += w
    return num / den if den else 0.0


def realized(bot_name: str, params: dict, hands: int, family: str) -> float:
    """Simulate; return the bot's pooled postflop facing-FOLD or unopened-AGGRO frequency."""
    from pokerbot.engine.game import GameState
    from pokerbot.eval.bots import build_bot
    from pokerbot.eval.exploitation_battery import ExploitHero, _action_to_bin, _ctx_key_for, _play

    want_facing = family == "facing_fold"
    hit = tot = 0
    for h in range(hands):
        hand_seed = 31337 + h
        hero = ExploitHero(random.Random(hand_seed), exploit=False)
        deck = GameState.new_hand(button_seat=3, rng=random.Random(hand_seed))
        bot = build_bot(bot_name, random.Random(5000 + hand_seed), params=params)
        counts = [0, 0]

        def bw(state, _bot=bot, _c=counts):
            a = _bot(state)
            ck = _ctx_key_for(state)
            parts = ck.split("|")
            if parts[0] != "preflop":
                b = _action_to_bin(state, a)
                if want_facing and parts[-1] == "facing_bet":
                    _c[0] += int(b == 0)
                    _c[1] += 1
                elif not want_facing and parts[-1] == "unopened":
                    # PROPENSITY, not raw frequency: count only nodes where a BET/RAISE is
                    # actually legal. The NL abstraction strands committed stacks on
                    # (ALL_IN|CHECK|FOLD) menus where _pick_bet checks regardless of intent —
                    # the IRC targets come from LIMIT, where betting is always available, so the
                    # comparable quantity is P(bet | bet available).
                    from pokerbot.engine.actions import ActionType as _AT

                    if any(x.action_type in (_AT.BET, _AT.RAISE) for x in state.legal_actions()):
                        _c[0] += int(b >= 2)
                        _c[1] += 1
            return a

        _play(0, hero, bw, deck)
        hit += counts[0]
        tot += counts[1]
    return hit / max(1, tot)


def bisect_param(bot_name: str, base: dict, param: str, target: float, family: str,
                 lo: float, hi: float, hands: int, increasing: bool) -> float:
    """1-D bisection of ``param`` so realized(family) hits ``target``. ``increasing`` = realized
    frequency grows with the param value."""
    for _ in range(7):
        mid = 0.5 * (lo + hi)
        r = realized(bot_name, {**base, param: mid}, hands, family)
        if (r < target) == increasing:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=600)
    ap.add_argument("--out", default="data/player_profiles/irc_weak_mixture.json")
    args = ap.parse_args()

    pct = json.loads(PCT_PATH.read_text(encoding="utf-8"))["percentiles"]
    entries = []

    # hs_station: the STICKY tail — facing-fold targets at p10/p25/p50. call_hs UP => folds MORE.
    for level in ("p10", "p25", "p50"):
        tgt = _pooled_target(pct, _POSTFLOP_FACING, "fold", level)
        v = bisect_param("hs_station", {}, "call_hs", tgt, "facing_fold",
                         lo=0.05, hi=0.65, hands=args.hands, increasing=True)
        r = realized("hs_station", {"call_hs": v}, args.hands, "facing_fold")
        entries.append({"bot": "hs_station", "level": level, "target_fold": round(tgt, 4),
                        "realized_fold": round(r, 4), "params": {"call_hs": v}})
        print(f"hs_station {level}: target fold {tgt:.3f} -> call_hs={v} (realized {r:.3f})", flush=True)

    # hs_nit: the FOLDY tail — facing-fold targets at p75/p90. call_hs UP => folds MORE.
    for level in ("p75", "p90"):
        tgt = _pooled_target(pct, _POSTFLOP_FACING, "fold", level)
        v = bisect_param("hs_nit", {}, "call_hs", tgt, "facing_fold",
                         lo=0.30, hi=0.95, hands=args.hands, increasing=True)
        r = realized("hs_nit", {"call_hs": v}, args.hands, "facing_fold")
        entries.append({"bot": "hs_nit", "level": level, "target_fold": round(tgt, 4),
                        "realized_fold": round(r, 4), "params": {"call_hs": v}})
        print(f"hs_nit {level}: target fold {tgt:.3f} -> call_hs={v} (realized {r:.3f})", flush=True)

    # hs_lag: the AGGRO tail — unopened-aggression targets at p50/p75/p90. bet_hs DOWN => bets MORE.
    for level in ("p50", "p75", "p90"):
        tgt = _pooled_target(pct, _POSTFLOP_UNOPENED, "agg_small", level, extra_bin="agg_large")
        v = bisect_param("hs_lag", {}, "bet_hs", tgt, "unopened_aggro",
                         lo=0.10, hi=0.90, hands=args.hands, increasing=False)
        r = realized("hs_lag", {"bet_hs": v}, args.hands, "unopened_aggro")
        entries.append({"bot": "hs_lag", "level": level, "target_aggro": round(tgt, 4),
                        "realized_aggro": round(r, 4), "params": {"bet_hs": v}})
        print(f"hs_lag {level}: target aggro {tgt:.3f} -> bet_hs={v} (realized {r:.3f})", flush=True)

    payload = {"source": "IRC percentile fit (scripts/fit_weak_mixture.py)",
               "targets": "postflop facing fold / unopened aggro, n_players-weighted",
               "entries": entries}
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(entries)} entries)", flush=True)


if __name__ == "__main__":
    main()
