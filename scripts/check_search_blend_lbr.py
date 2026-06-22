"""LBR validation for the starvation-robust serve: the BLEND-to-blueprint (when budget-bound) must be
<= the raw budget-bound serve in exploitability — the blueprint is the trusted GTO floor, so leaning on it
under starvation can only REDUCE a Local-Best-Response adversary's edge, never raise it.

Wraps PluribusReBeLSolver via eval.solver_exploitability.make_search_policy_fn with a TINY wall-clock budget
so postflop solves are BUDGET-BOUND (the live ~22/200 regime), then measures HU LBR exploitability with the
blend OFF (raw starved average) vs ON (blend toward the root blueprint). Deterministic given the seed.

Run (set the abstraction env BEFORE pokerbot import, as the encoder bakes it at import):
  POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 python scripts/check_search_blend_lbr.py --hands 120
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--starved-iters", type=int, default=8,
                    help="FIXED starved iteration count (the live ~22/200 regime); held identical across "
                         "blend ON/OFF arms so the comparison isolates the blend, no wall-clock jitter")
    ap.add_argument("--target-iters", type=int, default=200,
                    help="the converged-target iteration count the shortfall is measured against")
    ap.add_argument("--real-blueprint", action="store_true",
                    help="load the deployed blueprint snapshot (the trusted GTO floor) instead of empty")
    args = ap.parse_args()

    import numpy as np

    from pokerbot.engine.np_engine import legal_action_mask_np
    from pokerbot.eval.lbr import LBRConfig, lbr_exploitability
    from pokerbot.eval.solver_exploitability import make_blueprint_policy_fn, _RANKS, _SUITS
    from pokerbot.engine.cards import Card
    from pokerbot.search.pbs_tracker import PBSTracker
    from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver
    from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
    from pokerbot.search.state_bridge import SubgameRoot
    from pokerbot.search import pluribus_search as PS

    snap: dict = {}
    if args.real_blueprint:
        from pokerbot.eval.bots import _load_blueprint_snapshot
        snap = _load_blueprint_snapshot()
        if isinstance(snap, dict) and "strat_sum" in snap:
            snap = snap["strat_sum"]
        print(f"loaded blueprint snapshot: {len(snap)} keys")

    cfg = LBRConfig(hands=args.hands, seed=args.seed)
    bp_policy = make_blueprint_policy_fn(snap)

    def _card(idx: int) -> Card:
        i = int(idx)
        return Card.from_str(_RANKS[i // 4] + _SUITS[i % 4])

    def make_policy(blend: bool):
        # FIXED-iteration starved serve (force_budget_bound): both arms run EXACTLY starved_iters CFR
        # iterations, so the ONLY difference is whether that starved average blends toward the blueprint.
        scfg = PluribusReBeLConfig(
            max_iterations=args.target_iters, time_budget_ms=10 ** 12,
            force_budget_bound=args.starved_iters, blueprint_blend=blend,
            search_preflop=False, leaf=ContinuationLeafConfig(select="uniform"))
        solver = PluribusReBeLSolver(snap, scfg, rng=np.random.default_rng(args.seed))

        def policy(env, seat: int) -> np.ndarray:
            solver.rng = np.random.default_rng(args.seed)
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
            except Exception:
                return bp_policy(env, seat)
            p = np.asarray(res.action_probs, dtype=np.float64)
            s = p.sum()
            return p / s if s > 1e-12 else bp_policy(env, seat)
        return policy

    def run(blend: bool) -> "object":
        PS.set_blueprint_blend(blend)
        return lbr_exploitability(make_policy(blend), cfg)

    off = run(blend=False)
    on = run(blend=True)
    PS.set_blueprint_blend(True)  # restore live default

    print(f"hands={args.hands} starved_iters={args.starved_iters}/{args.target_iters} seed={args.seed}")
    print(f"BLEND OFF (raw starved):  {off.mbb_per_g:+.1f} mbb/g  "
          f"({off.bb_per_100:+.2f} bb/100, +/-{off.stderr_bb_per_hand * 100:.1f})")
    print(f"BLEND ON  (-> blueprint): {on.mbb_per_g:+.1f} mbb/g  "
          f"({on.bb_per_100:+.2f} bb/100, +/-{on.stderr_bb_per_hand * 100:.1f})")
    delta = on.mbb_per_g - off.mbb_per_g
    print(f"delta (ON - OFF): {delta:+.1f} mbb/g  -> "
          f"{'OK: blend <= raw (not more exploitable)' if delta <= 0 + 1e-6 else 'REGRESSION: blend MORE exploitable'}")


if __name__ == "__main__":
    main()
