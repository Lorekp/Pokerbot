#!/usr/bin/env python3
"""Bluff-ratio learnability probe (Phase 2 pre-battery gate): quantify, vs HSLagBot, (a) the TRUE
bluff ratio among ALL aggressive actions (sim omniscient), (b) the OBSERVED ratio among showdown-
revealed ones (the live information set — selection bias = a−b), and (c) the Beta posterior's
convergence (bluff_read) at checkpoints. Costs seconds, no search hero, no battery spend.

Usage: python scripts/check_bluff_learnability.py [--hands 4000] [--bot hs_lag] [--seed 0]
"""

from __future__ import annotations

import argparse
import os
import random

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=4000)
    ap.add_argument("--bot", default="hs_lag")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from pokerbot.engine.game import GameState
    from pokerbot.eval.bots import build_bot
    from pokerbot.eval.exploitation_battery import (
        ExploitHero,
        _action_to_bin,
        _ctx_key_for,
        _hand_strength_at,
        _play,
    )
    from pokerbot.opponent.bayes_exploit import bluff_read
    from pokerbot.opponent.stats import OpponentStats

    stats = OpponentStats()
    true_b = true_v = 0       # ALL aggressive actions (omniscient)
    obs_b = obs_v = 0         # showdown-revealed only (the live info set), undiscounted tallies
    checkpoints = {250, 500, 1000, 2000, args.hands}

    print(f"bot={args.bot} hands={args.hands} seed={args.seed}")
    print(f"{'hands':>6s} {'true_ratio':>10s} {'obs_ratio':>10s} {'bias':>7s} "
          f"{'post_flop_unop':>14s} {'post_global':>11s} {'obs_n':>6s}")
    for h in range(args.hands):
        hand_seed = 1000 + args.seed * 1_000_000 + h
        hero = ExploitHero(random.Random(hand_seed), exploit=False)
        deck = GameState.new_hand(button_seat=3, rng=random.Random(hand_seed))
        bot = build_bot(args.bot, random.Random(5000 + hand_seed))
        buf: list[tuple] = []

        def bw(state, _bot=bot, _buf=buf):
            a = _bot(state)
            b = _action_to_bin(state, a)
            hs = _hand_strength_at(state) if b >= 2 else None
            _buf.append((_ctx_key_for(state), b, state.action_seat, hs))
            return a

        _play(0, hero, bw, deck)
        stats.hands += 1
        contenders = [p.seat for p in deck.players if not p.folded]
        sd = len(contenders) >= 2
        for ck, b, seat, hs in buf:
            if b >= 2 and hs is not None:
                if hs < 0.45:
                    true_b += 1
                elif hs > 0.60:
                    true_v += 1
                if sd and seat in contenders:
                    if hs < 0.45:
                        obs_b += 1
                        stats.bluff_decay_and_add(ck, True, stats.hands)
                    elif hs > 0.60:
                        obs_v += 1
                        stats.bluff_decay_and_add(ck, False, stats.hands)
        if (h + 1) in checkpoints:
            tr = true_b / max(1, true_b + true_v)
            orr = obs_b / max(1, obs_b + obs_v)
            fu = bluff_read(stats, "flop|dry|unopened")
            gl = bluff_read(stats, "river|paired|facing_bet")  # sparse key ⇒ shows the hierarchy
            print(f"{h + 1:>6d} {tr:>10.3f} {orr:>10.3f} {orr - tr:>+7.3f} "
                  f"{(f'{fu[0]:.3f}/c{fu[1]:.2f}' if fu else '—'):>14s} "
                  f"{(f'{gl[0]:.3f}/c{gl[1]:.2f}' if gl else '—'):>11s} "
                  f"{obs_b + obs_v:>6d}", flush=True)

    print("\nverdict: the consumption gate fires when conf >= ~0.3 (~2 effective reveals via the "
          "hierarchy); 'bias' is the showdown-selection skew the posterior inherits (negative = "
          "bluffs under-observed because they fold the field).")


if __name__ == "__main__":
    main()
