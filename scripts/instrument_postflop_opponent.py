"""INSTRUMENT one postflop spot: does the MODELED opponent continue too often (loose),
making hero over-bet (the same mechanism as the preflop over-jam)?

Mechanism under test (from pluribus_search.py):
  * _cfv OPPONENT branch samples ONE action from _opponent_strategy = (1-mix)*uniform_regret_match
    + mix*blueprint (mix=0.5 default). Opponent regrets NEVER update.
  * If the opponent's facing-a-bet infoset is served LOOSE (uniform-on-miss / poor blueprint
    coverage), the modeled caller continues too often regardless of blueprint_mix, so hero
    best-responds by betting/jamming.

We monkeypatch _opponent_strategy to log, at every opponent facing-a-bet node encountered during
the solve, the served fold-vs-continue mass and whether the blueprint row was a real hit or a
uniform fallback. We also report the served opponent strategy at mix=0.5 vs mix=1.0.
"""
from __future__ import annotations

import os
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import pickle
from collections import Counter
import numpy as np

from pokerbot.engine.np_engine import (
    ALL_IN, BET_BUCKET_START, FOLD, CHECK, CALL, legal_action_mask_np,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="artifacts/search_roots.pkl")
    ap.add_argument("--snapshot", default="checkpoints/serve_blueprint_6max.pkl")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--mix", type=float, default=0.5)
    ap.add_argument("--node", type=int, default=-1, help="specific root idx; -1 = first facing-bet river/turn node")
    args = ap.parse_args()

    with open(args.roots, "rb") as f:
        roots = pickle.load(f)
    with open(args.snapshot, "rb") as f:
        obj = pickle.load(f)
    snap = obj["strat_sum"] if isinstance(obj, dict) and "strat_sum" in obj else obj

    from pokerbot.search import pluribus_search as PS

    # pick a node: prefer a facing-bet node (FOLD legal) on turn/river so hero can over-raise
    if args.node >= 0:
        pick = args.node
    else:
        pick = None
        for i, r in enumerate(roots):
            env = r["env"]
            seat = int(r["hero_seat"])
            legal = legal_action_mask_np(env, seat).astype(bool)
            if legal[FOLD] and int(r["street"]) >= 2:
                pick = i
                break
        if pick is None:
            for i, r in enumerate(roots):
                if int(roots[i]["street"]) >= 2:
                    pick = i
                    break
    r = roots[pick]
    env = r["env"]
    seat = int(r["hero_seat"])
    legal = legal_action_mask_np(env, seat).astype(bool)
    print(f"chosen node idx={pick} street={r['street']} hero_seat={seat} "
          f"facing_bet(FOLD legal)={bool(legal[FOLD])} legal={np.flatnonzero(legal).tolist()}")

    # ---- instrument the opponent branch ----
    log = []  # each entry: (is_facing_bet, fold_mass, continue_mass, bp_was_hit)
    orig = PS.PluribusReBeLSolver._opponent_strategy

    def patched(self, env_, actor, legal_, regrets, key):
        strat = orig(self, env_, actor, legal_, regrets, key)
        lg = np.asarray(legal_, dtype=bool)
        facing = bool(lg[FOLD])
        fold_mass = float(strat[FOLD]) if facing else 0.0
        cont_mass = float(strat[CALL] + strat[BET_BUCKET_START:].sum())
        # detect if blueprint row was a real hit: _cached_blueprint returns uniform-legal on miss
        bp = self._cached_blueprint(env_, actor, key)
        bp = np.asarray(bp, dtype=np.float64)
        lgf = lg.astype(np.float64)
        uni = lgf / max(lgf.sum(), 1)
        # a "hit" means the blueprint row differs meaningfully from uniform-over-legal
        is_uniform = bool(np.allclose(bp * lg / max((bp * lg).sum(), 1e-12), uni, atol=1e-6))
        if facing:
            log.append((fold_mass, cont_mass, is_uniform))
        return strat

    PS.PluribusReBeLSolver._opponent_strategy = patched

    from pokerbot.eval.solver_exploitability import make_search_policy_fn
    from pokerbot.search.pluribus_search import PluribusReBeLConfig

    # patch default mix if requested
    if args.mix != 0.5:
        _orig_init = PluribusReBeLConfig.__init__
        def _init(self, *a, **k):
            _orig_init(self, *a, **k)
            self.blueprint_mix = args.mix
        PluribusReBeLConfig.__init__ = _init

    pol = make_search_policy_fn(snap, leaf_select="uniform", seed=0,
                                iterations=args.iters, search_preflop=False)
    probs = pol(env, seat)
    print(f"\nHERO served policy at this node (K={args.iters}, mix={args.mix}):")
    print(f"  fold={probs[FOLD]:.3f} check={probs[CHECK]:.3f} call={probs[CALL]:.3f} "
          f"bet_mass={probs[BET_BUCKET_START:].sum():.3f} all_in={probs[ALL_IN]:.3f}")

    print(f"\nOPPONENT facing-a-bet nodes visited during solve: {len(log)}")
    if log:
        folds = np.array([x[0] for x in log])
        conts = np.array([x[1] for x in log])
        uniform_frac = np.mean([x[2] for x in log])
        print(f"  mean opponent FOLD mass when facing hero bet : {folds.mean():.3f}")
        print(f"  mean opponent CONTINUE mass (call+raise)      : {conts.mean():.3f}")
        print(f"  fraction of opp facing-bet infosets served UNIFORM (blueprint MISS): {uniform_frac:.3f}")
        print(f"  --> a HIGH continue mass + HIGH uniform frac == the modeled caller is LOOSE")


if __name__ == "__main__":
    main()
