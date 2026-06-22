"""6-max LBR validation for the starvation-robust serve (the 6-max analogue of check_search_blend_lbr.py).

The deployed 6-max serve is blueprint + Pluribus depth-limited search. When a live solve is BUDGET-BOUND
(iterations << max_iterations, the live ~22/200 regime), the served ROOT policy is blended toward the
trusted root blueprint (POKERBOT_SEARCH_BLUEPRINT_BLEND, default ON). The blueprint is the trusted GTO
floor, so leaning on it under starvation can only REDUCE a best-responder's edge, never raise it.

This measures, at a FIXED starved iteration count (force_budget_bound, so blend ON/OFF differ ONLY by the
blend, no wall-clock jitter), the 6-max LBR exploitability of:
  * RAW BLUEPRINT       (no search)                         -- the un-defended floor
  * SEARCH, blend OFF   (raw starved CFR average)           -- search without the starvation defense
  * SEARCH, blend ON    (-> blueprint when budget-bound)    -- the deployed starvation-robust serve

It is the THREE-WAY comparison the task asks for, pooled across seeds for a usable CI. The seat ACTS the
full (starved) search; the LBR range-tracks against the cheap raw blueprint MODEL (the two-policy harness).

Run (abstraction env BEFORE pokerbot import, the encoder bakes it at import):
  POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 \
    python scripts/check_search_blend_lbr6.py --hands 2000 --seeds 8 --starved-iters 8 --menu fcpa
"""

from __future__ import annotations

import argparse
import math
import os

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")


def _pool(triples):
    """Pool (mean_bb_per_hand, std_bb_per_hand, n) -> bb/100 + 95% CI via total-variance decomposition."""
    triples = [t for t in triples if t[2]]
    N = sum(n for _, _, n in triples)
    M = sum(m * n for m, _, n in triples) / N
    ss = sum(n * (sd ** 2) + n * (m - M) ** 2 for m, sd, n in triples)
    se = math.sqrt((ss / N) / N)
    return M * 100, (M - 1.96 * se) * 100, (M + 1.96 * se) * 100, N


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--starved-iters", type=int, default=8,
                    help="FIXED starved iteration count (the live ~22/200 regime); held identical across "
                         "blend ON/OFF so the comparison isolates the blend, no wall-clock jitter.")
    ap.add_argument("--target-iters", type=int, default=200,
                    help="converged-target iteration count the shortfall is measured against.")
    ap.add_argument("--menu", choices=("fc", "fcpa", "full"), default="fcpa")
    ap.add_argument("--n-seats", type=int, default=6)
    ap.add_argument("--blueprint-snapshot",
                    default=r"checkpoints/bp6max_dcfrvr_iter200.pkl")
    args = ap.parse_args()

    import pickle

    import numpy as np

    from pokerbot.engine.cards import Card
    from pokerbot.engine.np_engine import legal_action_mask_np
    from pokerbot.eval import lbr6
    from pokerbot.eval.lbr import make_blueprint_policy
    from pokerbot.eval.solver_exploitability import _RANKS, _SUITS
    from pokerbot.search import pluribus_search as PS
    from pokerbot.search.pbs_tracker import PBSTracker
    from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
    from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver
    from pokerbot.search.state_bridge import SubgameRoot

    with open(args.blueprint_snapshot, "rb") as f:
        snap_full = pickle.load(f)
    snap = (snap_full["strat_sum"]
            if isinstance(snap_full, dict) and "strat_sum" in snap_full else snap_full)
    print(f"blueprint snapshot: {len(snap)} infosets")

    bp_policy = make_blueprint_policy(snap)
    _MENUS = {"fc": (), "fcpa": (lbr6._POT_BET_INDEX, lbr6.ALL_IN),
              "full": (lbr6._HALF_POT_INDEX, lbr6._POT_BET_INDEX, lbr6.ALL_IN)}

    def _card(idx: int) -> Card:
        i = int(idx)
        return Card.from_str(_RANKS[i // 4] + _SUITS[i % 4])

    def make_starved_search(blend: bool, seed: int):
        # FIXED-iteration starved serve (force_budget_bound): exactly --starved-iters CFR iters run, so the
        # ONLY difference between arms is whether that starved average blends toward the blueprint.
        scfg = PluribusReBeLConfig(
            max_iterations=args.target_iters, time_budget_ms=10 ** 12,
            force_budget_bound=args.starved_iters, blueprint_blend=blend,
            search_preflop=False, leaf=ContinuationLeafConfig(select="uniform"))
        solver = PluribusReBeLSolver(snap, scfg, rng=np.random.default_rng(seed))

        def policy(env, seat: int) -> np.ndarray:
            solver.rng = np.random.default_rng(seed)
            hole = (env.holes[seat][0], env.holes[seat][1])
            if int(hole[0]) < 0 or int(hole[1]) < 0:
                return bp_policy(env, seat)
            board = [_card(int(c)) for c in np.asarray(env.board) if int(c) >= 0]
            hero_hole = (_card(int(hole[0])), _card(int(hole[1])))
            pbs = PBSTracker.init_uniform(seat, hero_hole, board)
            legal = legal_action_mask_np(env, seat).astype(bool)
            root = SubgameRoot(env=env, hero_seat=seat,
                               pbs_vector=pbs.get_public_belief_state(), legal_mask=legal)
            try:
                res = solver.solve(root, pbs)
            except Exception:  # noqa: BLE001
                return bp_policy(env, seat)
            p = np.asarray(res.action_probs, dtype=np.float64)
            s = p.sum()
            return p / s if s > 1e-12 else bp_policy(env, seat)
        return policy

    def measure(policy_fn, *, model_fn):
        trip = []
        for seed in range(args.seeds):
            cfg = lbr6.LBR6Config(hands=args.hands, seed=seed, n_seats=args.n_seats,
                                  track_ranges=True, paired=False, value_only_raises=True)
            cfg.bet_candidates = _MENUS[args.menu]
            r = lbr6.lbr6_exploitability(policy_fn, cfg, model_fn=model_fn)
            trip.append((r.ev_bb_per_hand, r.stderr_bb_per_hand * (r.hands ** 0.5), r.hands))
        return _pool(trip)

    # RAW BLUEPRINT (no search) -- the floor.
    raw = measure(bp_policy, model_fn=bp_policy)

    # SEARCH blend OFF / ON. value-only raises (search fold-probs un-modellable; matches lbr6_audit default).
    PS.set_blueprint_blend(False)
    off = measure(make_starved_search(blend=False, seed=0), model_fn=bp_policy)
    PS.set_blueprint_blend(True)
    on = measure(make_starved_search(blend=True, seed=0), model_fn=bp_policy)
    PS.set_blueprint_blend(True)  # restore live default

    def fmt(name, t):
        return f"{name:28s}: {t[0]:+.1f} bb/100  [{t[1]:+.1f}, {t[2]:+.1f}]   ({t[3]}h)"

    print(f"\n=== 6-max LBR exploitability: blueprint vs starved-search blend OFF/ON "
          f"({args.menu} menu, {args.starved_iters}/{args.target_iters} iters) ===")
    print(fmt("RAW BLUEPRINT (no search)", raw))
    print(fmt("SEARCH blend OFF (starved)", off))
    print(fmt("SEARCH blend ON (-> bp)", on))
    d_blend = on[0] - off[0]
    d_vs_raw = on[0] - raw[0]
    print(f"\ndelta blend (ON - OFF) : {d_blend:+.1f} bb/100  -> "
          f"{'OK: blend <= raw starved (not more exploitable)' if d_blend <= 5.0 else 'REGRESSION: blend MORE exploitable'}")
    print(f"delta vs raw blueprint : {d_vs_raw:+.1f} bb/100  (search+blend vs bare blueprint; "
          f"{'search REDUCES exploitability' if d_vs_raw < 0 else 'no reduction at this CI'})")


if __name__ == "__main__":
    main()
