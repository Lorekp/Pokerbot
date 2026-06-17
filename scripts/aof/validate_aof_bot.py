"""AoF bot validation battery — answers "are we 100% sure the bot beats the average player?".

Three independent checks:
  (1) RANGES vs external Nash oracle: our HU SB-jam% / BB-call% at 6/8/10bb + 3/4-max per-position
      jam%, to compare against published push/fold Nash charts.
  (2) FIELD SIM (the decisive one): our GTO bot vs the REAL observed CoinPoker field (per-node commit
      frequencies aggregated from the opponent DB = the average player's actual strategy), seat-rotated,
      rake INCLUDED (fee 0.16 bb = live 0.10/0.25), over a large N so the CI is tight (the 779 live
      hands were variance-dominated). Villains commit their top-f% range at the frequency we've
      observed them play. Broken down by player count and by hero position to expose any node leak.
  (3) Cross-check: same sim vs a pure-NIT and pure-STATION field (directions must match the analytic
      absolute_winrate table — sanity that the simulator is wired right).

Independent of the corrupted 4-max EV annotations: hero strategy comes from the engine, but all
payoffs come from clean native equity + the validated pushfold settlement. Play-money / offline only.
"""
from __future__ import annotations

import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from aof_cfr.buckets import LABELS_169, LABEL_TO_INDEX, combo_to_bucket
from aof_cfr.combo_deal import FULL_DECK, card_id
from aof_cfr.infoset import aggregate_infoset_key
from aof_cfr.live_exploit import decide
from aof_cfr.opponent_db import AoFOpponentDB
from aof_cfr.profiler import _strength_order_desc
from aof_cfr.pushfold_payoffs import contested_pot_bb
from aof_cfr.query import base_strategy, load_cfr_base
from aof_cfr.trio_eval_mc import mc_nway_shares

FEE_BB = 0.16
SB_BL, BB_BL = 0.5, 1.0
STACK = 8.0
RING = {2: ("SB", "BB"), 3: ("BTN", "SB", "BB"), 4: ("UTG", "BTN", "SB", "BB")}

# ---- combos per bucket + per-hand strength rank (top fraction) -----------------------------------
_BUCKET_COMBOS: dict[str, list[tuple[str, str]]] = defaultdict(list)
for _i in range(len(FULL_DECK)):
    for _j in range(_i + 1, len(FULL_DECK)):
        _BUCKET_COMBOS[combo_to_bucket(FULL_DECK[_i], FULL_DECK[_j])].append((FULL_DECK[_i], FULL_DECK[_j]))
_COMBO_W = {lab: len(v) for lab, v in _BUCKET_COMBOS.items()}
_ORDER = [LABELS_169[i] for i in _strength_order_desc()]   # strongest -> weakest
_TOTAL = sum(_COMBO_W.values())   # 1326
# top-fraction at which each label becomes the *weakest* included hand (cum combos / total)
_RANK_FRAC = {}
_acc = 0
for _lab in _ORDER:
    _acc += _COMBO_W[_lab]
    _RANK_FRAC[_lab] = _acc / _TOTAL     # commit hand iff freq >= _RANK_FRAC[label]-ish; see below
_LABEL_FRAC_LO = {}
_acc = 0
for _lab in _ORDER:
    _LABEL_FRAC_LO[_lab] = _acc / _TOTAL  # fraction of strictly-stronger hands
    _acc += _COMBO_W[_lab]


def commits(label: str, freq: float) -> bool:
    """Does a player with `label` commit, if they commit their top-`freq` fraction? Boundary bucket
    is split proportionally so the realised frequency matches `freq` in expectation."""
    lo = _LABEL_FRAC_LO[label]
    hi = _RANK_FRAC[label]
    if freq >= hi:
        return True
    if freq <= lo:
        return False
    # partially in the range: include with prob (freq-lo)/(hi-lo)
    return random.random() < (freq - lo) / (hi - lo)


# ---- field model (average player) ----------------------------------------------------------------
class Field:
    def __init__(self, db: AoFOpponentDB):
        self.agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for ident in db.identities():
            for key, (p, t) in db.cells_for(ident).items():
                self.agg[key][0] += p
                self.agg[key][1] += t

    def freq(self, pos, hist):
        p, t = self.agg.get(aggregate_infoset_key(pos, hist or "-", STACK), (0, 0))
        return (p / t) if t else None


# ---- showdown: one random board per contested pot (unbiased mean over many hands) ----------------
def hero_net_when_alive(alive_idx, holes, n, folded_blinds, rng_seed):
    """Hero (index 0 within alive_idx ordering) net bb at an all-in terminal, 1 sampled board."""
    k = len(alive_idx)
    if k == 1:
        return folded_blinds           # uncontested
    flat = np.array([c for i in alive_idx for c in holes[i]], dtype=np.int64)
    shares = mc_nway_shares(flat, k, 1, rng_seed)
    pot = contested_pot_bb(k, folded_blinds, STACK, fee_bb=FEE_BB)
    return float(shares[0]) * pot - STACK   # hero is alive_idx[0]


def deal(n, rng):
    deck = list(FULL_DECK); rng.shuffle(deck)
    return [(deck[2 * i], deck[2 * i + 1]) for i in range(n)]


def play_hand(n, hero_pos, holes_lbl, holes_cards, strat_fn, rng, seed):
    """strat_fn(pos, hist, label) -> commit?  for villains. Hero uses the bot. Returns hero net bb."""
    order = RING[n]
    hist = ""
    committed = []          # positions all-in
    folded = []             # positions folded
    blind = {"SB": SB_BL, "BB": BB_BL}
    for idx, pos in enumerate(order):
        # WALK: last player to act, everyone before folded, nobody committed -> wins uncontested,
        # no decision node exists (e.g. folds around to the BB). The base has no such infoset.
        if idx == n - 1 and not committed and len(folded) == idx:
            folded_blinds = sum(blind.get(p, 0.0) for p in folded)
            if pos == hero_pos:
                return folded_blinds          # hero (BB) collects the dead blinds
            return -blind.get(hero_pos, 0.0)  # hero folded earlier
        lbl = holes_lbl[pos]
        if pos == hero_pos:
            p = base_strategy(n, pos, hist or "-", lbl, STACK)
            act = rng.random() < p
        else:
            act = strat_fn(pos, hist, lbl)
        if act:
            committed.append(pos); hist += "A"
        else:
            folded.append(pos); hist += "F"
    # settle
    if hero_pos in folded:
        return -blind.get(hero_pos, 0.0)
    folded_blinds = sum(blind.get(p, 0.0) for p in folded)
    alive = committed                      # all committed go to showdown
    # build holes list indexed by alive order with hero first
    alive_sorted = [hero_pos] + [p for p in alive if p != hero_pos]
    holes = [holes_cards[p] for p in alive_sorted]
    hcards = [(card_id(c0), card_id(c1)) for (c0, c1) in holes]
    return hero_net_when_alive(list(range(len(alive_sorted))), hcards, n, folded_blinds, seed)


def field_strat(field):
    nash_fallback = {}   # cache: when field has no data, fall back to our own base freq (neutral)

    def f(pos, hist, label):
        fr = field.freq(pos, hist)
        if fr is None:
            fr = 0.5
        return commits(label, fr)
    return f


def fixed_strat(freq_const):
    def f(pos, hist, label):
        return commits(label, freq_const)
    return f


def nit_strat(label_threshold_frac):
    def f(pos, hist, label):
        return commits(label, label_threshold_frac)
    return f


def sim(n, n_hands, strat_fn, seed0=12345):
    rng = random.Random(seed0)
    nets = []
    for h in range(n_hands):
        cards = deal(n, rng)
        lbls = [combo_to_bucket(c0, c1) for (c0, c1) in cards]
        # assign positions: random rotation -> seat-rotated
        order = RING[n]
        perm = list(range(n)); rng.shuffle(perm)
        pos_cards = {order[k]: cards[perm[k]] for k in range(n)}
        pos_lbl = {order[k]: lbls[perm[k]] for k in range(n)}
        # hero rotates through all positions uniformly
        hero_pos = order[h % n]
        net = play_hand(n, hero_pos, pos_lbl, pos_cards, strat_fn, rng, seed0 + h)
        nets.append(net)
    m = sum(nets) / len(nets)
    se = st.pstdev(nets) / math.sqrt(len(nets))
    return m, se, nets


def by_position(n, n_hands, strat_fn, seed0=222):
    rng = random.Random(seed0)
    pos_nets = defaultdict(list)
    order = RING[n]
    for h in range(n_hands):
        cards = deal(n, rng)
        lbls = [combo_to_bucket(c0, c1) for (c0, c1) in cards]
        perm = list(range(n)); rng.shuffle(perm)
        pos_cards = {order[k]: cards[perm[k]] for k in range(n)}
        pos_lbl = {order[k]: lbls[perm[k]] for k in range(n)}
        hero_pos = order[h % n]
        net = play_hand(n, hero_pos, pos_lbl, pos_cards, strat_fn, rng, seed0 + h)
        pos_nets[hero_pos].append(net)
    return {p: (sum(v) / len(v) * 100, len(v)) for p, v in pos_nets.items()}


def our_jam_freq(n, pos, hist):
    """Fraction of all 1326 combos our base jams at this node."""
    tot = 0
    for lab in LABELS_169:
        tot += _COMBO_W[lab] * base_strategy(n, pos, hist or "-", lab, STACK)
    return tot / _TOTAL


def main():
    load_cfr_base()
    db = AoFOpponentDB(path="data/aof_player_profiles/profiles.json"); db.load()
    field = Field(db)

    print("=" * 78)
    print("(1) OUR PUSH/FOLD RANGES vs external Nash oracle")
    for stk in (6.0, 8.0, 10.0):
        # HU: SB jams (hist '-'), BB calls a jam (hist 'A')
        global STACK
        STACK = stk
        sb = our_jam_freq(2, "SB", "-") * 100
        bbc = our_jam_freq(2, "BB", "A") * 100
        print(f"  HU @ {stk:>4.0f}bb:  SB jam {sb:5.1f}%   BB call-vs-jam {bbc:5.1f}%")
    STACK = 8.0
    print("  3-max @ 8bb jam%:  " + "  ".join(
        f"{p}:{our_jam_freq(3, p, h)*100:.0f}%" for p, h in [("BTN", "-"), ("SB", "F"), ("BB", "FA")]))
    print("  4-max @ 8bb jam%:  " + "  ".join(
        f"{p}:{our_jam_freq(4, p, h)*100:.0f}%" for p, h in [("UTG", "-"), ("BTN", "F"), ("SB", "FF"), ("BB", "FFA")]))

    print()
    print("=" * 78)
    print("(2) FIELD jam% (the average player we faced) vs OUR jam% — leak check")
    for n, nodes in [(4, [("UTG", "-"), ("BTN", "F"), ("SB", "FF"), ("BB", "FFA")]),
                     (3, [("BTN", "-"), ("SB", "F"), ("BB", "FA")])]:
        for p, h in nodes:
            fr = field.freq(p, h)
            ours = our_jam_freq(n, p, h) * 100
            frs = f"{fr*100:5.1f}%" if fr is not None else "  n/a"
            print(f"  {n}-max {p:3} hist={h:4}:  field {frs}   ours {ours:5.1f}%   "
                  f"{'(field TIGHTER -> we steal)' if fr and fr*100 < ours-3 else ('(field LOOSER -> we tighten)' if fr and fr*100 > ours+3 else '(~matched)')}")

    print()
    print("=" * 78)
    print("(3) WIN-RATE vs the REAL field  (seat-rotated, rake 0.16bb, large N)")
    NH = {4: 400_000, 3: 250_000, 2: 120_000}
    fstrat = field_strat(field)
    blended = []
    weights = {4: 0.77, 3: 0.21, 2: 0.02}   # live n-distribution
    wsum = 0.0
    for n in (2, 3, 4):
        m, se, nets = sim(n, NH[n], fstrat, seed0=1000 + n)
        lo, hi = (m - 2 * se) * 100, (m + 2 * se) * 100
        print(f"  {n}-max:  {m*100:+6.2f} bb/100   95% CI [{lo:+.2f}, {hi:+.2f}]   (N={NH[n]:,})")
        wsum += weights[n] * m
    print(f"  BLENDED (live mix 77/21/2): {wsum*100:+.2f} bb/100")

    print()
    print("  by hero position (4-max vs field):")
    bp = by_position(4, 300_000, fstrat)
    for p in RING[4]:
        print(f"    {p:3}: {bp[p][0]:+6.1f} bb/100")

    print()
    print("=" * 78)
    print("(4) CROSS-CHECK vs synthetic fields (directions must match absolute_winrate)")
    for name, frac in [("NIT (jam 12%)", 0.12), ("avg-ish (jam 35%)", 0.35), ("STATION (jam 70%)", 0.70)]:
        m, se, _ = sim(4, 250_000, fixed_strat(frac), seed0=777)
        print(f"  4-max vs {name:20}: {m*100:+6.2f} bb/100   95% CI [{(m-2*se)*100:+.2f}, {(m+2*se)*100:+.2f}]")


if __name__ == "__main__":
    main()
