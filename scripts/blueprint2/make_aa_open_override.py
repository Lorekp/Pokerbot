#!/usr/bin/env python3
"""Produce a MINIMAL AA-forced-open variant of the deployed blueprint: identical to
checkpoints/serve_blueprint_6max.pkl EXCEPT the AA folded-to-hero (RFI) rows are overwritten to open a
normal raise (bucket 7 = pot ~= 2.5x). Everything else -- postflop, all other preflop -- is byte-unchanged.
Used to measure the AA-limp's ISOLATED value vs the pool (deployed vs this, CRN-paired: rank_cross_abstraction).

  POKERBOT_* env (deployed EMD-K160) is set below BEFORE importing pokerbot/battery_native.
  python scripts/blueprint2/make_aa_open_override.py <out.pkl>
"""
import os
import pickle
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EMD = os.path.join(REPO, "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn")
for _k, _v in {
    "JAX_PLATFORMS": "cpu", "POKERBOT_BETTING_ABSTRACTION": "1", "POKERBOT_POSTFLOP_BUCKETS": "1",
    "POKERBOT_POSTFLOP_NUM_BUCKETS": "160", "POKERBOT_POSTFLOP_MODE": "emd",
    "POKERBOT_POSTFLOP_EMD_FROZEN": EMD, "POKERBOT_EMD_FROZEN_MIRROR_FALLBACK": "1",
    "POKERBOT_PREFLOP_LOSSLESS": "1",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np  # noqa: E402
import jax  # noqa: E402
from pokerbot.engine.jax_engine import init_hand  # noqa: E402
from pokerbot.engine.np_engine import (  # noqa: E402
    apply_action_np, current_player_np, is_terminal_np, legal_action_mask_np, to_numpy_state,
)
from pokerbot.search import native_solver as ns  # noqa: E402
import battery_native as bn  # noqa: E402

FOLD, OPEN_ACTION = 0, 7  # 7 = 1.0x pot (~2.5bb open); the standard RFI size in the deployed menu

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "artifacts", "_ccleaf", "deployed_AAopen.pkl")
DEP = os.path.join(REPO, "checkpoints", "serve_blueprint_6max.pkl")
raw = pickle.load(open(DEP, "rb"))
snap = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw
snap = {bytes(k): np.asarray(v, dtype=np.float64) for k, v in snap.items()}
print(f"deployed: {len(snap)} infosets", flush=True)


def seat_of(btn, pos):
    return (btn + {"BTN": 0, "SB": 1, "BB": 2, "UTG": 3, "HJ": 4, "CO": 5}[pos]) % 6


def preflop_order(btn):
    return [(btn + 3 + i) % 6 for i in range(6)]


def rig_hero_aces(holes, hero):
    holes = holes.copy()
    for i, card in enumerate([48, 49]):  # Ac, Ad
        loc = None
        for s in range(holes.shape[0]):
            for sl in range(2):
                if int(holes[s, sl]) == card:
                    loc = (s, sl)
        if loc == (hero, i) or loc is None:
            if loc is None:
                holes[hero, i] = card
            continue
        tmp = int(holes[hero, i]); holes[hero, i] = int(holes[loc[0], loc[1]]); holes[loc[0], loc[1]] = tmp
    return holes


occ = jax.numpy.array([True] * 6)
overrides = {}   # key -> (pos, action)
missing = 0
for d in range(300):
    env0 = to_numpy_state(init_hand(jax.random.PRNGKey(910001 + d), occupancy=occ))
    btn = int(env0.button)
    order = preflop_order(btn)
    for pos in ("UTG", "HJ", "CO", "BTN", "SB"):
        pi = ["UTG", "HJ", "CO", "BTN", "SB"].index(pos)
        e = env0
        ok = True
        for s in order[:pi]:
            if is_terminal_np(e) or current_player_np(e) != s:
                ok = False
                break
            e = apply_action_np(e, FOLD)
        if not ok or is_terminal_np(e):
            continue
        hero = seat_of(btn, pos)
        if current_player_np(e) != hero:
            continue
        e = e._replace(holes=rig_hero_aces(np.asarray(e.holes), hero))
        flat = ns._to_flat(e).tobytes()
        key = bytes(bn.fingerprint_fast_emd(flat, hero, 160, [], EMD, True))
        if key in overrides:
            continue
        legal = legal_action_mask_np(e, hero)
        a = OPEN_ACTION if legal[OPEN_ACTION] else next((i for i in range(3, 11) if legal[i]), 2)
        row = np.zeros(11, dtype=np.float64)
        row[a] = 1.0
        in_dep = key in snap
        if not in_dep:
            missing += 1
        # deployed AA-RFI action BEFORE override (for the report)
        before = snap.get(key)
        bstr = ("limp" if before is not None and before[2] == before.max() else
                ("jam" if before is not None and before[10] == before.max() else "raise/other")) if before is not None else "ABSENT"
        snap[key] = row
        overrides[key] = (pos, int(a), bstr, in_dep)

print(f"overrode {len(overrides)} AA-RFI keys (missing-from-deployed={missing}):", flush=True)
for k, (pos, a, bstr, indep) in overrides.items():
    print(f"  {pos}: was {bstr} -> now open action {a}  (key_in_deployed={indep})", flush=True)

with open(OUT, "wb") as f:
    pickle.dump({"strat_sum": snap}, f)
print(f"SAVED {OUT} ({len(snap)} infosets)", flush=True)
