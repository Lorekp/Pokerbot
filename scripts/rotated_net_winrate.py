"""NET-of-rake seat-rotated win-rate (the DOLLAR metric, not chip-EV).

The rotated eval reports GROSS chip-EV (rake-free). This re-runs the same seat-rotated duplicate loop
but also reads the RESOLVED GameState to charge the hero its contributed rake share
(rake x hero_committed/pot, the validated low-variance attribution), net of CoinRewards rakeback, for
each stake band. net = gross - contributed_rake*(1 - rakeback). Read-only on pot resolution (no engine
change). Reports per (field, stake-band): gross / rake / net bb/100 with 95% CI on net.

Run: python scripts/rotated_net_winrate.py
"""
from __future__ import annotations

import os
import pickle
import random
import sys
from pathlib import Path

os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
os.environ.setdefault("POKERBOT_SEARCH_PREFLOP", "1")
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

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np

from pokerbot.economics.rake import STAKES, contributed_rake_bb, rakeback_fraction
from pokerbot.engine.game import play_hand
from pokerbot.engine.game import GameState
from pokerbot.eval.exploitation_battery import _allin_ev_settle, build_calibrated_field_bot
from pokerbot.eval.production_hero import ProductionHero
from pokerbot.eval.search_agent import SearchAgent

DECKS = 60
BUTTON = 3
ITERS = 20
# One representative stake per CoinPoker rakeback band (micro=100% modeled, low/mid/high tiers).
BANDS = {"micro": "0.01/0.02", "low": "0.10/0.25", "mid": "0.50/1", "high": "5/10"}
# (display label, homogeneous tier bot name passed to build_calibrated_field_bot)
FIELDS = [("recreational", "recreational"), ("solid_reg", "solid_reg")]


def make_hero():
    with open(r"checkpoints/bp6max_lossless2_iter200.pkl", "rb") as f:
        full = pickle.load(f)
    snap = full["strat_sum"] if isinstance(full, dict) and "strat_sum" in full else full
    ag = SearchAgent(use_pluribus_search=True, blueprint_snapshot=snap, pluribus_leaf_select="uniform",
                     time_budget_ms=10 ** 9, transformer_checkpoint=None, hero_seat=0,
                     root_exploit_mixer=True, root_mixer_unopened=True, aivat_leaf_q=False)
    ag.solver.config.max_iterations = ITERS
    ag.solver.config.time_budget_ms = 10 ** 9
    return ProductionHero(ag, random.Random(0), exploit=False, hero_seat=0, n_seats=6)


def _stake_key(label):
    # STAKES is keyed directly by the "sb/bb" label string (e.g. "0.10/0.25").
    if label in STAKES:
        return label
    raise KeyError(f"{label} not in STAKES ({sorted(STAKES)})")


def run_field(field_bot_name):
    hero = make_hero()
    # per-deck seat-mean of: gross, and contributed rake (in BB, cap-independent part) per band
    deck_gross, deck_rake = [], {b: [] for b in BANDS}
    n_dealt = 6
    for d in range(DECKS):
        deck_seed = 2000 + d
        settle = lambda st, nb, _s=deck_seed: _allin_ev_settle(st, nb, 7777 + _s)
        seat_g = []
        seat_r = {b: [] for b in BANDS}
        for hs in range(6):
            bots = {s: build_calibrated_field_bot(field_bot_name, random.Random(5000 + deck_seed + 17 * s))
                    for s in range(6) if s != hs}
            hero.hero_seat = hs
            hero.reseed(deck_seed)
            hero.on_new_hand()
            st = GameState.new_hand(button_seat=BUTTON, rng=random.Random(deck_seed))
            policies = {s: (hero if s == hs else (lambda state, _b=bots: _b[state.action_seat](state)))
                        for s in range(6)}
            final, deltas = play_hand(policies, state=st, allin_settle=settle)
            seat_g.append(float(deltas.get(hs, 0.0)))
            pot = float(final.pot_bb)
            committed = float(final.players[hs].committed_total_bb)
            ended_pf = len(final.board) == 0
            for band, label in BANDS.items():
                sk = _stake_key(label)
                seat_r[band].append(contributed_rake_bb(pot, committed, n_dealt, ended_pf, sk))
        deck_gross.append(float(np.mean(seat_g)))
        for b in BANDS:
            deck_rake[b].append(float(np.mean(seat_r[b])))
    return np.asarray(deck_gross), {b: np.asarray(v) for b, v in deck_rake.items()}


def main():
    print(f"NET-of-rake seat-rotated win-rate ({DECKS} decks, {ITERS} iters, exploit ON). "
          f"net = gross - rake*(1-rakeback). bb/100.\n")
    for label, botname in FIELDS:
        g, rk = run_field(botname)
        gross = 100 * g.mean()
        gse = 100 * g.std(ddof=1) / np.sqrt(len(g))
        print(f"=== {label}  (gross {gross:+.1f} +/- {1.96*gse:.0f} bb/100) ===")
        print(f"  {'band':6s} {'stake':10s} {'rake':>8s} {'rakeback':>9s} {'NET(floor)':>11s} {'NET(modeled)':>13s}")
        for band, label in BANDS.items():
            sk = _stake_key(label)
            rake100 = 100 * rk[band].mean()
            rb_floor = rakeback_fraction(sk, "floor")
            rb_model = rakeback_fraction(sk, "modeled")
            # paired per-deck net (low-variance): gross_deck - rake_deck*(1-rb)
            net_floor = 100 * (g - rk[band] * (1 - rb_floor)).mean()
            net_model = 100 * (g - rk[band] * (1 - rb_model)).mean()
            print(f"  {band:6s} {label:10s} {rake100:>7.1f} {rb_model*100:>7.0f}% "
                  f"{net_floor:>+11.1f} {net_model:>+13.1f}")
        print()


if __name__ == "__main__":
    main()
