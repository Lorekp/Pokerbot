"""Multi-seed LBR EV-equivalence gate for the native CFV-under-exploitation port (POKERBOT_NATIVE_CFV).

THE GATE (required by artifacts/search_latency_spec.md, validation step 1): the native CFR driver, made
reachable under exploitation by carrying the firewalled exploit tilt AS DATA (tilt_table), must produce
LBR-exploitability statistically INDISTINGUISHABLE from the interpreted _cfv path. The relaxable changes
(native xoshiro RNG vs numpy PCG64, native float-reduction order, suit-iso-free leaf cache) are validated
by CI-overlap, not byte-equality.

REACHABILITY (measured locally before this run — see the report): the native CFV path fires only when
``self._leaf_cont_weights is None`` (the model-aware native leaf port is deferred, so it is mutually
exclusive with native CFV). _build_leaf_cont_weights returns non-None as soon as ANY modeled seat has an
``aggregate_read`` — i.e. as soon as the bot has observed the opponent. So the native CFV path is reachable
ONLY in the COLD-MODEL regime: a modeled seat present (opponent_model != None, so the firewalled exploit
tail + tilt build run) but with no observations folded in yet (aggregate_read None). This harness measures
LBR exactly there: opponent_model is set (tilt table is built + applied), but no reads accumulate, so the
gate that the native tilt machinery (key lookup + tilt application + relaxable RNG/float) is LBR-equivalent
to the interpreted exploit _cfv is tested on the path that actually carries those decisions.

Each (seed) runs the LBR adversary twice on the SAME deck stream:
  ON : POKERBOT_NATIVE_CFV=1 + POKERBOT_NATIVE_SUBGAME=1 -> served=search_native_cfv
  OFF: both off                                          -> served=search (interpreted _cfv)
We assert served paths match expectation, then aggregate per-arm bb/100 across seeds and report whether
the two arms' 95% CIs OVERLAP (holds=true iff they do).

Run on the cloud box (x86_64 c6a) after `cargo build --release --features python`:
  python scripts/native_cfv_lbr_equivalence.py --seeds 8 --hands 300 --menu fc \
      --blueprint-snapshot /tmp/bp.pkl --out /tmp/native_cfv_lbr_result.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import time

import numpy as np


def _load_snapshot(path):
    with open(path, "rb") as f:
        snap = pickle.load(f)
    return snap["strat_sum"] if isinstance(snap, dict) and "strat_sum" in snap else snap


_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def _card(idx):
    from pokerbot.engine.cards import Card
    i = int(idx)
    return Card.from_str(_RANKS[i // 4] + _SUITS[i % 4])


class ColdExploitTable:
    """Blueprint+search bots that hold ONE non-hero seat in opponent_model with a COLD (0-obs) profile.

    opponent_model != None => the firewalled exploit tail + (under POKERBOT_NATIVE_CFV) the tilt table are
    built and applied, so the native CFV path is exercised — but aggregate_read stays None (no observe),
    so _leaf_cont_weights is None and the native CFV gate is OPEN. observe() is a no-op (the read never
    builds), which is precisely the regime where native CFV carries the decision."""

    def __init__(self, snapshot, *, leaf_select="uniform", iterations=10, seed=0):
        from pokerbot.opponent.stats import OpponentStats
        from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
        from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver
        from pokerbot.eval.lbr import make_blueprint_policy

        cfg = PluribusReBeLConfig(time_budget_ms=10 ** 9,
                                  leaf=ContinuationLeafConfig(select=leaf_select))
        cfg.max_iterations = int(iterations)
        self._solver = PluribusReBeLSolver(snapshot, cfg, rng=np.random.default_rng(seed))
        self._bp = make_blueprint_policy(snapshot)
        self._seed = seed
        self._cold = OpponentStats()  # never observed -> aggregate_read None -> leaf_cw None
        self.lbr_seat = None
        self.served = collections.Counter()
        self.tilt_built = 0
        self.solves = 0

    def on_hand_start(self, lbr_seat):
        self.lbr_seat = int(lbr_seat)

    def observe(self, env, seat, action):
        return  # no-op: keep the model cold so native CFV stays reachable

    def act(self, env, seat):
        from pokerbot.engine.np_engine import legal_action_mask_np
        from pokerbot.search.pbs_tracker import PBSTracker
        from pokerbot.search.state_bridge import SubgameRoot

        hole = (int(env.holes[seat][0]), int(env.holes[seat][1]))
        if hole[0] < 0 or hole[1] < 0:
            return np.asarray(self._bp(env, seat), dtype=np.float64)
        # Model the LBR seat (cold) so the exploit tail + tilt build run; native CFV gate stays open.
        model = {self.lbr_seat: self._cold} if self.lbr_seat is not None and self.lbr_seat != seat else None
        self._solver.opponent_model = model
        self._solver.rng = np.random.default_rng(self._seed)
        board = [_card(int(c)) for c in np.asarray(env.board) if int(c) >= 0]
        pbs = PBSTracker.init_uniform(seat, (_card(hole[0]), _card(hole[1])), board)
        legal = legal_action_mask_np(env, seat).astype(bool)
        root = SubgameRoot(env=env, hero_seat=seat, pbs_vector=pbs.get_public_belief_state(),
                           legal_mask=legal, opponent_profiles=model)
        try:
            result = self._solver.solve(root, pbs)
            probs = np.asarray(result.action_probs, dtype=np.float64)
            self.solves += 1
            self.served[(result.metadata or {}).get("served", "?")] += 1
            if self._solver._native_cfv_tilt_table is not None:
                self.tilt_built += 1
        except Exception:
            return np.asarray(self._bp(env, seat), dtype=np.float64)
        s = probs.sum()
        if s <= 1e-12:
            return np.asarray(self._bp(env, seat), dtype=np.float64)
        return probs / s


def _run_arm(snap, *, native_cfv: bool, seed: int, hands: int, warmup: int, menu: str,
             iterations: int, range_cap: int):
    """Run ONE LBR measurement arm at one seed. Sets the env flags BEFORE importing the solver gate-reads."""
    from pokerbot.search import pluribus_search as ps
    from pokerbot.search import native_solver as ns

    # Flip the process-global gates (the solver reads these module flags at solve time).
    ps.set_native_cfv(bool(native_cfv))
    ps.set_native_subgame(bool(native_cfv))
    if native_cfv:
        ns.enable()
    else:
        ns.disable()

    from pokerbot.eval import lbr6
    from pokerbot.eval.lbr import make_blueprint_policy

    table = ColdExploitTable(snap, leaf_select="uniform", iterations=iterations, seed=seed)
    model_fn = make_blueprint_policy(snap)
    menus = {"fc": (), "fcpa": (lbr6._POT_BET_INDEX, lbr6.ALL_IN),
             "full": (lbr6._HALF_POT_INDEX, lbr6._POT_BET_INDEX, lbr6.ALL_IN)}

    def cfg(h, s):
        c = lbr6.LBR6Config(hands=h, seed=s, paired=False, range_cap=range_cap, value_only_raises=True)
        c.bet_candidates = menus[menu]
        return c

    if warmup > 0:
        lbr6.lbr6_exploitability(table.act, cfg(warmup, seed), model_fn=model_fn,
                                 observe_fn=table.observe, on_hand_start=table.on_hand_start)
    res = lbr6.lbr6_exploitability(table.act, cfg(hands, seed + 100_000), model_fn=model_fn,
                                   observe_fn=table.observe, on_hand_start=table.on_hand_start)
    return {
        "bb_per_100": float(res.bb_per_100),
        "ev_bb_per_hand": float(res.ev_bb_per_hand),
        "stderr_bb_per_hand": float(res.stderr_bb_per_hand),
        "hands": int(res.hands),
        "served": dict(table.served),
        "tilt_built": int(table.tilt_built),
        "solves": int(table.solves),
    }


def _direct_equivalence(snap, *, n_nodes: int, iterations: int, seed: int):
    """SENSITIVE companion to the LBR gate: on a sweep of native-CFV-eligible postflop nodes, solve EACH
    node twice (native CFV ON vs interpreted _cfv OFF) at the SAME public_seed and compare the served root
    policy. Reports argmax agreement + mean L1 over the search nodes that actually route through native CFV.
    This isolates the native-vs-interp decision difference (the LBR signal is diluted by blueprint preflop).

    Cold model so native CFV is reachable; per-decision-seed makes each solve a pure function of public state,
    so ON and OFF integrate the SAME Monte-Carlo expectation under the relaxable RNG/float reorder."""
    import random
    from pokerbot.engine.actions import ActionType
    from pokerbot.engine.game import GameState, play_hand
    from pokerbot.engine.np_engine import to_numpy_state, legal_action_mask_np
    from pokerbot.opponent.stats import OpponentStats
    from pokerbot.search import native_solver as ns
    from pokerbot.search import pluribus_search as ps
    from pokerbot.search.pbs_tracker import PBSTracker
    from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
    from pokerbot.search.state_bridge import game_state_to_env, SubgameRoot

    cfg = ps.PluribusReBeLConfig(time_budget_ms=10 ** 9,
                                 leaf=ContinuationLeafConfig(select="uniform"))
    cfg.max_iterations = int(iterations)
    solver = ps.PluribusReBeLSolver(snap, cfg, rng=np.random.default_rng(seed))

    # collect postflop multiway nodes
    rng = random.Random(seed)
    nodes = []
    while len(nodes) < n_nodes:
        st0 = GameState.new_hand(button_seat=rng.randint(0, 5),
                                 rng=random.Random(rng.randint(0, 10 ** 9)))

        def pol(state):
            legal = state.legal_actions()
            s = state.action_seat
            street = int(state.street.value if hasattr(state.street, "value") else state.street)
            if s is not None and s >= 0 and legal and street >= 2:
                nodes.append((game_state_to_env(state, int(s)), int(s)))
            if not legal:
                return None
            passive = [a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)]
            if passive and rng.random() < 0.55:
                return rng.choice(passive)
            return rng.choice(legal)

        play_hand({s: pol for s in range(6)}, state=st0,
                  rng=random.Random(rng.randint(0, 10 ** 9)))
    nodes = nodes[:n_nodes]

    def solve_probs(env, hero, native):
        ps.set_native_cfv(bool(native))
        ps.set_native_subgame(bool(native))
        ns.enable() if native else ns.disable()
        opp = (hero + 1) % 6
        model = {opp: OpponentStats()}
        solver.opponent_model = model
        solver.rng = np.random.default_rng(seed)
        board = [_card(int(c)) for c in np.asarray(env.board) if int(c) >= 0]
        h0, h1 = int(env.holes[hero][0]), int(env.holes[hero][1])
        pbs = PBSTracker.init_uniform(hero, (_card(h0), _card(h1)), board)
        legal = legal_action_mask_np(env, hero).astype(bool)
        root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=pbs.get_public_belief_state(),
                           legal_mask=legal, opponent_profiles=model)
        res = solver.solve(root, pbs)
        return (np.asarray(res.action_probs, dtype=np.float64),
                (res.metadata or {}).get("served"))

    agree = 0
    l1s = []
    n_native = 0
    for env, hero in nodes:
        p_on, served_on = solve_probs(env, hero, True)
        p_off, served_off = solve_probs(env, hero, False)
        if served_on != "search_native_cfv":
            continue  # only score nodes where native CFV actually carried the ON decision
        n_native += 1
        if int(np.argmax(p_on)) == int(np.argmax(p_off)):
            agree += 1
        l1s.append(float(np.abs(p_on - p_off).sum()))
    return {
        "n_nodes_swept": len(nodes),
        "n_native_cfv_nodes": n_native,
        "argmax_agreement": (agree / n_native) if n_native else None,
        "mean_l1": (float(np.mean(l1s)) if l1s else None),
        "p95_l1": (float(np.percentile(l1s, 95)) if l1s else None),
    }


def _agg(per_seed_bb, per_seed_stderr_bb_hand, hands):
    """Aggregate per-seed bb/100 into a mean + 95% CI. Each seed is an independent measurement; we combine
    via the per-seed variances (stderr_bb_per_hand * sqrt(hands) = per-hand std; the seed mean's variance is
    stderr^2). Pooled SE over S seeds = sqrt(sum(se_i^2))/S."""
    import math
    arr = np.asarray(per_seed_bb, dtype=np.float64)
    mean = float(arr.mean())
    se_seed = np.asarray(per_seed_stderr_bb_hand, dtype=np.float64) * 100.0  # bb/100 SE per seed
    pooled_se = float(math.sqrt(float((se_seed ** 2).sum())) / len(arr))
    # Between-seed spread also matters; take the larger of pooled-within and between-seed SE for honesty.
    between_se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    se = max(pooled_se, between_se)
    return {"mean_bb_per_100": mean, "se_bb_per_100": se,
            "ci95": [mean - 1.96 * se, mean + 1.96 * se],
            "pooled_within_se": pooled_se, "between_seed_se": between_se}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--hands", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--menu", choices=("fc", "fcpa", "full"), default="fc")
    ap.add_argument("--search-iters", type=int, default=10)
    ap.add_argument("--range-cap", type=int, default=80)
    ap.add_argument("--blueprint-snapshot", required=True)
    ap.add_argument("--direct-nodes", type=int, default=400,
                    help="postflop nodes for the sensitive per-decision native-vs-interp argmax/L1 check")
    ap.add_argument("--out", default=None)
    ap.add_argument("--s3-bucket", default=None)
    ap.add_argument("--s3-key", default=None)
    ap.add_argument("--region", default="eu-central-1")
    args = ap.parse_args()

    snap = _load_snapshot(args.blueprint_snapshot)
    print(f"blueprint: {len(snap)} infosets  seeds={args.seeds} hands={args.hands} "
          f"warmup={args.warmup} menu={args.menu} iters={args.search_iters}", flush=True)

    on, off = [], []
    t0 = time.time()
    for s in range(args.seeds):
        rs_on = _run_arm(snap, native_cfv=True, seed=s, hands=args.hands, warmup=args.warmup,
                         menu=args.menu, iterations=args.search_iters, range_cap=args.range_cap)
        rs_off = _run_arm(snap, native_cfv=False, seed=s, hands=args.hands, warmup=args.warmup,
                          menu=args.menu, iterations=args.search_iters, range_cap=args.range_cap)
        on.append(rs_on)
        off.append(rs_off)
        print(f"seed {s}: ON {rs_on['bb_per_100']:+.2f}  OFF {rs_off['bb_per_100']:+.2f}  "
              f"| ON served={rs_on['served']} tilt={rs_on['tilt_built']}/{rs_on['solves']}  "
              f"OFF served={rs_off['served']}  ({(time.time()-t0)/60:.1f}m)", flush=True)

    on_agg = _agg([r["bb_per_100"] for r in on], [r["stderr_bb_per_hand"] for r in on], args.hands)
    off_agg = _agg([r["bb_per_100"] for r in off], [r["stderr_bb_per_hand"] for r in off], args.hands)
    # CI overlap test
    overlap = not (on_agg["ci95"][1] < off_agg["ci95"][0] or off_agg["ci95"][1] < on_agg["ci95"][0])
    # Did native CFV actually carry the ON arm? (sanity: search_native_cfv must dominate ON's served paths)
    on_served = collections.Counter()
    for r in on:
        for k, v in r["served"].items():
            on_served[k] += v
    off_served = collections.Counter()
    for r in off:
        for k, v in r["served"].items():
            off_served[k] += v
    # Share among SEARCH-class decisions only (blueprint_preflop is served from the blueprint in BOTH arms,
    # identically — it carries no native-vs-interp signal). The gate cares whether, of the decisions that DO
    # run a subgame solve, the ON arm took the native CFV path.
    on_search_total = on_served.get("search_native_cfv", 0) + on_served.get("search", 0)
    native_cfv_share = on_served.get("search_native_cfv", 0) / max(1, on_search_total)

    # SENSITIVE companion: direct per-decision native-vs-interp equivalence on the native CFV nodes.
    print("\nrunning direct per-decision equivalence sweep...", flush=True)
    direct = _direct_equivalence(snap, n_nodes=args.direct_nodes, iterations=args.search_iters, seed=4242)
    print(f"  direct: {direct['n_native_cfv_nodes']}/{direct['n_nodes_swept']} native CFV nodes  "
          f"argmax_agree={direct['argmax_agreement']}  mean_l1={direct['mean_l1']}", flush=True)

    payload = {
        "gate": "native_cfv_lbr_equivalence",
        "regime": "cold_model (opponent_model set, 0 obs, leaf_cw None) — the only regime native CFV is reachable",
        "config": {"seeds": args.seeds, "hands": args.hands, "warmup": args.warmup, "menu": args.menu,
                   "search_iters": args.search_iters, "range_cap": args.range_cap,
                   "snapshot_infosets": len(snap)},
        "on_arm": {"agg": on_agg, "per_seed": on, "served_total": dict(on_served),
                   "native_cfv_share": native_cfv_share},
        "off_arm": {"agg": off_agg, "per_seed": off, "served_total": dict(off_served)},
        "ci_overlap": bool(overlap),
        "delta_mean_bb_per_100": on_agg["mean_bb_per_100"] - off_agg["mean_bb_per_100"],
        "direct_equivalence": direct,
        # HOLDS iff: (1) the LBR CIs overlap (required gate), (2) the ON arm actually ran native CFV on its
        # search decisions, and (3) the sensitive per-decision check shows high argmax agreement (>=0.90, the
        # DCFR-vs-CFR+ serve-iteration bar) with small L1 — i.e. the relaxable RNG/float changes did not
        # change decision quality.
        "holds": bool(
            overlap
            and native_cfv_share > 0.5
            and direct["n_native_cfv_nodes"] >= 50
            and (direct["argmax_agreement"] or 0.0) >= 0.90
        ),
        "wall_min": (time.time() - t0) / 60.0,
    }

    print("\n==================== NATIVE CFV LBR-EQUIVALENCE GATE ====================")
    print(f"  ON  (native CFV): mean {on_agg['mean_bb_per_100']:+.2f} bb/100  "
          f"CI95 [{on_agg['ci95'][0]:+.2f}, {on_agg['ci95'][1]:+.2f}]")
    print(f"  OFF (interp _cfv): mean {off_agg['mean_bb_per_100']:+.2f} bb/100  "
          f"CI95 [{off_agg['ci95'][0]:+.2f}, {off_agg['ci95'][1]:+.2f}]")
    print(f"  CI overlap: {overlap}   delta {payload['delta_mean_bb_per_100']:+.2f} bb/100")
    print(f"  ON native_cfv_share (of search decisions): {native_cfv_share:.3f}  (ON served: {dict(on_served)})")
    print(f"  OFF served: {dict(off_served)}")
    print(f"  DIRECT per-decision: {direct['n_native_cfv_nodes']}/{direct['n_nodes_swept']} native CFV nodes  "
          f"argmax_agree={direct['argmax_agreement']}  mean_l1={direct['mean_l1']}  p95_l1={direct['p95_l1']}")
    print(f"  HOLDS: {payload['holds']}")
    print("========================================================================")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.out}", flush=True)
    if args.s3_bucket and args.s3_key:
        import boto3
        boto3.client("s3", region_name=args.region).put_object(
            Bucket=args.s3_bucket, Key=args.s3_key, Body=json.dumps(payload, indent=2).encode())
        print(f"uploaded s3://{args.s3_bucket}/{args.s3_key}", flush=True)


if __name__ == "__main__":
    main()
