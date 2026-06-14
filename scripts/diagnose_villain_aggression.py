#!/usr/bin/env python3
"""Phase 2 diagnostic (read-only): does each eval archetype's OWN-INITIATIVE (unopened) aggression
read actually SEPARATE a bluffy lag from a value-only station?

The value-extraction exploit (value-bet stations) is OFF on the search base because it can't be gated
do-no-harm: root_mixer.py reports `init_agg` (the villain's unopened aggression EXCESS vs the IRC Nash
reference) is negative for EVERY archetype, so "defer vs aggressors, value-bet vs passive callers" has no
signal. The suspected cause is task #12: `_pick_bet` returns CHECK when the legal menu offers no SIZED
bet (only ALL_IN) — so an aggressive bot that wants to bet silently checks, flattening init_agg for all.

This script plays homogeneous self-play per archetype, accumulates each villain's actions into an
OpponentStats with the SAME context keys / 4-bin map the live engine uses, and prints per archetype:
  * unopened bin distribution [fold, passive, agg_small, agg_large] and aggression frequency,
  * init_agg = (post.agg) - (pop.agg)  (the exact gate signal; >0 = reads aggressive),
  * forced_check_rate = fraction of decisions where the bot took CHECK while the ONLY legal aggressive
    action was ALL_IN (the task #12 collapse — aggression the abstraction couldn't express).
If init_agg does NOT separate lag from station, Phase 2 must fix _pick_bet (jam instead of check) so the
own-initiative signal returns. Run before AND after the fix to quantify the gain.

  python scripts/diagnose_villain_aggression.py --hands 400
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from pokerbot.engine.actions import ActionType
from pokerbot.engine.game import GameState
from pokerbot.eval.bots import build_bot
from pokerbot.eval.exploitation_battery import _action_to_bin, _ctx_key_for
from pokerbot.opponent.bayes_exploit import aggregate_read
from pokerbot.opponent.population_prior import context_prior
from pokerbot.opponent.stats import OpponentStats
from pokerbot.training.self_play import _legal_mask

ARCHETYPES = ("hs_station", "hs_nit", "hs_lag", "hs_tag")


def _only_allin_aggressive(state: GameState, seat: int) -> bool:
    """True iff the only legal AGGRESSIVE action is ALL_IN (no sized BET/RAISE) — the task #12 spot
    where _pick_bet falls through to CHECK."""
    legal = state.legal_actions(seat)
    has_sized = any(a.action_type in (ActionType.BET, ActionType.RAISE) for a in legal)
    has_allin = any(a.action_type == ActionType.ALL_IN for a in legal)
    return has_allin and not has_sized


def diagnose(name: str, hands: int, seed: int, n_seats: int = 6) -> dict:
    prof = OpponentStats()
    forced_check = 0
    unopened_decisions = 0
    for h in range(hands):
        hs = 7000 + seed * 1_000_000 + h
        state = GameState.new_hand(button_seat=3, rng=random.Random(hs))
        bots = {s: build_bot(name, random.Random(5000 + hs + 17 * s)) for s in range(n_seats)}
        buf: list[tuple[str, int]] = []
        guard = 0
        while state.street.name != "SHOWDOWN" and guard < 400:
            guard += 1
            seat = state.action_seat
            if not state.legal_actions(seat):  # all-in/folded seat (engine runout) — stop collecting
                break
            legal_mask = np.asarray(_legal_mask(state, seat), dtype=bool)
            check_legal = bool(legal_mask.size > 1 and legal_mask[1])
            only_allin = _only_allin_aggressive(state, seat)
            action = bots[seat](state)
            if check_legal:  # an UNOPENED decision (CHECK legal ⇒ no outstanding bet)
                unopened_decisions += 1
                buf.append((_ctx_key_for(state), _action_to_bin(state, action)))
                if only_allin and action.action_type == ActionType.CHECK:
                    forced_check += 1
            state.apply_action(action, seat)
        prof.hands += 1
        seen: dict[str, list[int]] = {}
        for ck, b in buf:
            seen.setdefault(ck, [0, 0, 0, 0])[b] += 1
        for ck, bins in seen.items():
            prof.add_ctx(ck, bins)
            prof.decay_and_add(ck, bins, context_prior(ck), prof.hands)

    read = aggregate_read(prof, facing="unopened")
    if read is None:
        return {"name": name, "post": None, "init_agg": None, "forced_check_rate": 0.0}
    post = np.asarray(read.post, dtype=float)
    pop = np.asarray(read.pop, dtype=float)
    init_agg = float((post[2] + post[3]) - (pop[2] + pop[3]))
    return {
        "name": name,
        "post": post.tolist(),
        "agg_freq": float(post[2] + post[3]),
        "init_agg": init_agg,
        "conf": float(read.conf),
        "forced_check_rate": forced_check / max(1, unopened_decisions),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"{'archetype':12s} {'fold':>6s} {'pass':>6s} {'aggS':>6s} {'aggL':>6s} "
          f"{'aggFreq':>8s} {'init_agg':>9s} {'forcedCHK':>10s}")
    rows = []
    for a in ARCHETYPES:
        r = diagnose(a, args.hands, args.seed)
        rows.append(r)
        if r["post"] is None:
            print(f"{a:12s}  (no read)")
            continue
        p = r["post"]
        print(f"{a:12s} {p[0]:>6.2f} {p[1]:>6.2f} {p[2]:>6.2f} {p[3]:>6.2f} "
              f"{r['agg_freq']:>8.2f} {r['init_agg']:>+9.3f} {r['forced_check_rate']:>9.1%}")

    valid = [r for r in rows if r.get("init_agg") is not None]
    if len(valid) >= 2:
        lag = next((r for r in valid if r["name"] == "hs_lag"), None)
        sta = next((r for r in valid if r["name"] == "hs_station"), None)
        if lag and sta:
            sep = lag["init_agg"] - sta["init_agg"]
            print(f"\nSEPARATION lag.init_agg - station.init_agg = {sep:+.3f} "
                  f"({'SIGNAL EXISTS' if sep > 0.05 else 'NO USABLE SIGNAL — gate is blind'})")
        spread = max(r["init_agg"] for r in valid) - min(r["init_agg"] for r in valid)
        print(f"init_agg spread across archetypes = {spread:.3f}")
        print(f"mean forced-check rate = {np.mean([r['forced_check_rate'] for r in valid]):.1%}")


if __name__ == "__main__":
    main()
