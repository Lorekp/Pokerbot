#!/usr/bin/env python3
"""Read-accuracy convergence rig (G4): how many hands until the Bayesian read identifies each
fair bot's coded weakness, in the right DIRECTION, with how much confidence?

Ground truth = the bot's EMPIRICAL long-run profile (--truth-hands, disjoint seed): per context,
the dominant deviation bin vs the near-Nash reference and its sign. For each N in --checkpoints,
simulates N hands of online learning (the exact battery ingestion loop), freezes the profile, and
scores whether each truth context's small-N read recovers the true DIRECTION.

Output: direction accuracy@N (context-weighted), mean confidence, mean TV. Pure profile math —
seconds per archetype, no search hero, no battery spend. This is the measured replacement for
the repo's previously untested "98.4% read hit rate" lore.

Usage: python scripts/eval_read_convergence.py [--checkpoints 10,25,50,100,200] [--seeds 5]
"""

from __future__ import annotations

import argparse
import os
import random

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

import numpy as np

# Ground truth is EMPIRICAL, not hand-derived: each bot's long-run (--truth-hands) profile defines,
# per context, the dominant deviation bin vs the near-Nash reference and its sign. accuracy@N then
# measures whether the small-N read recovers that direction — a pure convergence metric, immune to
# wrong armchair derivations (first version of this rig hand-coded "the nit over-folds vs a raise";
# empirically its hs<0.52 continue threshold folds LESS than the tight near-Nash facing line, so the
# correct exploit direction there is the opposite — exactly the kind of error the model itself never
# makes, because it reads the data).
TRUTH_MIN_OBS = 30.0
TV_GATE = 0.05  # contexts where the bot's true deviation is ~0 carry no direction to score


def _truth_directions(stats) -> dict[str, tuple[int, float]]:
    """{context_key: (dominant_bin, sign)} from a long-run profile — the bot's TRUE style."""
    from pokerbot.opponent.bayes_exploit import context_read

    out = {}
    for ck in stats.ctx_fast:
        read = context_read(stats, ck)
        if read is None or read.n_obs < TRUTH_MIN_OBS or read.tv < TV_GATE:
            continue
        dev = read.post - read.pop
        d = int(abs(dev).argmax())
        out[ck] = (d, 1.0 if dev[d] > 0 else -1.0)
    return out


def run(arch: str, n_hands: int, seed: int):
    """Simulate ``n_hands`` of the battery's exact online-learning loop; return the profile."""
    from pokerbot.engine.game import GameState
    from pokerbot.eval.bots import build_bot
    from pokerbot.eval.exploitation_battery import ExploitHero, _action_to_bin, _ctx_key_for, _play
    from pokerbot.opponent.population_prior import context_prior
    from pokerbot.opponent.stats import OpponentStats

    stats = OpponentStats()
    for h in range(n_hands):
        hand_seed = 1000 + seed * 1_000_000 + h
        hero = ExploitHero(random.Random(hand_seed), exploit=False)
        deck = GameState.new_hand(button_seat=3, rng=random.Random(hand_seed))
        bot = build_bot(arch, random.Random(5000 + hand_seed))
        buf: list[tuple] = []

        def bw(state, _bot=bot, _buf=buf):
            a = _bot(state)
            _buf.append((_ctx_key_for(state), _action_to_bin(state, a)))
            return a

        _play(0, hero, bw, deck)
        if buf:
            stats.hands += 1
            seen: dict[str, list[int]] = {}
            for ck, b in buf:
                seen.setdefault(ck, [0, 0, 0, 0])[b] += 1
            for ck, bins in seen.items():
                stats.add_ctx(ck, bins)
                stats.decay_and_add(ck, bins, context_prior(ck), stats.hands)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="10,25,50,100,200")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--truth-hands", type=int, default=4000)
    ap.add_argument("--archetypes", default="hs_station,hs_nit,hs_lag,hs_tag")
    args = ap.parse_args()
    checkpoints = [int(x) for x in args.checkpoints.split(",")]

    from pokerbot.opponent.bayes_exploit import context_read

    print(f"{'archetype':12s} {'N':>5s} {'dir_acc':>8s} {'n_ctx':>6s} {'mean_conf':>10s} {'mean_tv':>8s}")
    for arch in args.archetypes.split(","):
        # Empirical ground truth: the bot's long-run per-context deviation directions (truth seed
        # is disjoint from every probe seed).
        truth = _truth_directions(run(arch, args.truth_hands, seed=97))
        for n in checkpoints:
            accs, confs, tvs, n_ctxs = [], [], [], []
            for seed in range(args.seeds):
                stats = run(arch, n, seed)
                ok = tot = 0
                for ck, (d, sign) in truth.items():
                    read = context_read(stats, ck)
                    if read is None or read.n_obs < 1:
                        continue
                    confs.append(read.conf)
                    tvs.append(read.tv)
                    tot += 1
                    if float(read.post[d] - read.pop[d]) * sign > 0:
                        ok += 1
                n_ctxs.append(tot)
                if tot:
                    accs.append(ok / tot)
            acc_s = f"{np.mean(accs):8.3f}" if accs else "       —"
            print(f"{arch:12s} {n:>5d} {acc_s} {np.mean(n_ctxs):>6.1f} "
                  f"{np.mean(confs) if confs else 0:>10.3f} {np.mean(tvs) if tvs else 0:>8.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
