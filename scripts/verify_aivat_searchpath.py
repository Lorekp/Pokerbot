#!/usr/bin/env python3
"""Verify the AIVAT search-path fix: the NEW per-action baseline (the solve's own cached ROOT CFVs,
returned by the patched ``PluribusReBeLSolver._leaf_q``) must REDUCE the hero's decision-term variance,
where the OLD 1-ply continuation-gadget baseline INCREASED it (negative variance reduction).

Exact PAIRED test (no field-desync, no two-run noise): play N seat-rotated duplicate hands ONCE; at every
hero search decision capture (probs, taken_idx, bv_new, bv_old) on IDENTICAL play -- bv_new = the cached
root CFVs, bv_old = the legacy 1-ply gadget (re-derived by hiding the cache for one call). Then for each
hand form the AIVAT-adjusted score with each baseline and compare the realized variance reduction
VR = 1 - Var(adj)/Var(raw). PASS = VR_new > 0 AND VR_new > VR_old (the fix flips/raises the decision term).

Run (serve env must match the blueprint, like the ranking harness):
  POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_PREFLOP_LOSSLESS=1 \
    python scripts/verify_aivat_searchpath.py --blueprint <path> --hands 180 --iters 20
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(path: str) -> dict:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    inner = obj["strat_sum"] if (isinstance(obj, dict) and "strat_sum" in obj) else obj
    return {bytes(k): np.asarray(v, dtype=np.float64) for k, v in inner.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", default=r"checkpoints/bp6max_lossless2_iter200.pkl")
    ap.add_argument("--hands", type=int, default=180)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--search-preflop", action="store_true",
                    help="also search preflop (slower; default = blueprint-served preflop, postflop search)")
    args = ap.parse_args()

    from pokerbot.engine.game import NUM_PLAYERS, play_hand
    from pokerbot.eval.aivat import decision_correction
    from pokerbot.eval.bots import build_bot
    from pokerbot.eval.search_agent import SearchAgent

    snap = _load(args.blueprint)
    print(f"[aivat-verify] blueprint {len(snap):,} infosets  hands={args.hands} iters={args.iters} "
          f"search_preflop={args.search_preflop}", flush=True)

    class CaptureHero(SearchAgent):
        """Records (probs, idx, bv_new, bv_old) per search decision on the SAME play."""

        def __init__(self, **kw):
            super().__init__(**kw)
            self.records: list[tuple] = []        # flat list across the current hand
            self.solver._aivat_capture = True      # arm the root-CFV cache (forces the Python CFR loop)

        def __call__(self, state):
            from pokerbot.eval.search_agent import _index_to_action
            from pokerbot.search.state_bridge import build_subgame_root

            seat = state.action_seat
            hero = self._state.hero_seat
            if len(self._state.trajectory.token_types) == 0:
                self._init_pbs(state)
            self._state.trajectory.on_decision(state, seat)
            self._sync_pbs(state, seat=None, action=None)
            pbs = self._state.pbs
            root = build_subgame_root(state, pbs, hero)
            result = self.solver.solve(root, pbs)
            probs = np.asarray(result.action_probs, dtype=float)
            if self._decision_rng is not None and probs.sum() > 0:
                w = np.clip(probs, 0.0, None)
                idx = int(self._decision_rng.choices(range(len(w)), weights=w.tolist())[0])
            else:
                idx = int(probs.argmax())
            # bv_new = cached root CFVs; bv_old = legacy 1-ply gadget (hide cache for one call).
            try:
                bv_new = np.asarray(self.solver._leaf_q(root.env), dtype=float).reshape(-1)
                _save = self.solver._root_av_sum
                self.solver._root_av_sum = None
                bv_old = np.asarray(self.solver._leaf_q(root.env), dtype=float).reshape(-1)
                self.solver._root_av_sum = _save
                # only a real search decision (cache populated) is a fix-affected node
                if _save is not None:
                    self.records.append((probs.copy(), idx, bv_new, bv_old))
            except Exception:  # noqa: BLE001
                pass
            action = _index_to_action(state, seat, idx)
            self._state.trajectory.on_action(state, seat, action)
            return action

    def make_hero():
        h = CaptureHero(use_pluribus_search=True, blueprint_snapshot=snap, aivat_leaf_q=True,
                        search_preflop=args.search_preflop, hero_seat=0)
        h.solver.config.max_iterations = args.iters
        h.solver.config.time_budget_ms = 10 ** 9
        h._decision_rng = random.Random(args.seed ^ 0xA1A7)
        return h

    hero = make_hero()
    raw, c_new, c_old = [], [], []           # per-hand: raw chips, summed corrections (new / old baseline)
    groups = max(1, math.ceil(args.hands / NUM_PLAYERS))
    t0 = time.time()
    n_dec = 0
    for g in range(groups):
        for hero_seat in range(NUM_PLAYERS):
            hero.reset_hand(hero_seat)
            hero.records = []
            # fresh field bots per hand (deterministic) so play is reproducible; deck = seed+g (duplicate)
            field = {s: build_bot("calling_station", random.Random(5000 + (args.seed + g) + 17 * s))
                     for s in range(NUM_PLAYERS) if s != hero_seat}
            field[hero_seat] = hero
            _final, d = play_hand(field, rng=random.Random(args.seed + g))
            v = float(d.get(hero_seat, 0.0))
            cn = sum(decision_correction(p, bvn, i) for (p, i, bvn, _bvo) in hero.records)
            co = sum(decision_correction(p, bvo, i) for (p, i, _bvn, bvo) in hero.records)
            raw.append(v); c_new.append(cn); c_old.append(co); n_dec += len(hero.records)
        if (g + 1) % 5 == 0:
            print(f"[aivat-verify] {(g+1)*NUM_PLAYERS}/{groups*NUM_PLAYERS} hands  "
                  f"{n_dec} search decisions  ({time.time()-t0:.0f}s)", flush=True)

    raw = np.asarray(raw); adj_new = raw - np.asarray(c_new); adj_old = raw - np.asarray(c_old)
    vr = lambda a: (1.0 - np.var(a, ddof=1) / np.var(raw, ddof=1)) if np.var(raw, ddof=1) > 0 else 0.0
    VRn, VRo = vr(adj_new), vr(adj_old)
    print("\n=== AIVAT search-path decision-term variance reduction (paired, identical play) ===")
    print(f"  hands={len(raw)}  search-decisions={n_dec}  raw_std={np.std(raw, ddof=1):.1f} bb")
    print(f"  OLD (1-ply gadget _leaf_q):  VR = {VRo:+.4f}   adj_mean={adj_old.mean():+.2f} bb "
          f"(raw {raw.mean():+.2f})")
    print(f"  NEW (cached root CFVs):      VR = {VRn:+.4f}   adj_mean={adj_new.mean():+.2f} bb "
          f"(raw {raw.mean():+.2f})")
    verdict = "PASS" if (VRn > 0 and VRn > VRo) else ("PARTIAL (improved, still <=0)" if VRn > VRo else "FAIL")
    print(f"  -> {verdict}: NEW {'>' if VRn>VRo else '<='} OLD, NEW {'reduces' if VRn>0 else 'does not reduce'} variance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
