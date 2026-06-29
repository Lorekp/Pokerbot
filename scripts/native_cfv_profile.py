"""Per-decision LATENCY profile for the native CFV-under-exploitation port (POKERBOT_NATIVE_CFV).

THE PROFILE (the deliverable of the native-cfv port workflow, alongside the LBR-equivalence gate in
``native_cfv_lbr_equivalence.py``): on a sweep of native-CFV-eligible postflop decision nodes, solve EACH
node twice at the LIVE budget and time it —

  ON  : POKERBOT_NATIVE_CFV=1 + POKERBOT_NATIVE_SUBGAME=1  -> served=search_native_cfv
  OFF : both off                                           -> served=search (interpreted _cfv)

and report p50/p95 wall-clock per decision for each arm, the speedup, and a per-COMPONENT cost breakdown of
the ON arm (the dominant cost decomposition the optimization pass targets):

  * tilt_build_ms   — the Python ``_build_native_cfv_tilt_table`` dry pass (the firewalled exploit tail run
                      ONCE per decision; the contract's dominant ON cost — the native solve is ~60x the
                      interpreted _cfv, so the build, not the solve, sets the ON floor).
  * native_solve_ms — the single ``eng.solve(...)`` FFI call (marshal-in + native CFR + marshal-out).
  * other_ms        — preflop gate / leaf-cont-weights build / blueprint / bookkeeping (ON total minus the
                      two measured spans).

REACHABILITY: identical to the LBR gate — the native CFV path fires only in the COLD-MODEL regime
(opponent_model set so the exploit tail + tilt build run, but no observation folded in yet so
_leaf_cont_weights is None and the gate is OPEN). We profile exactly there. Each node is filtered to those
the ON arm actually serves via search_native_cfv (else the timing carries no native-vs-interp signal).

This is the AMDAHL-AWARE companion to the LBR gate: the LBR gate proves the decision is unchanged; this
proves the decision is FASTER and, via the component split, identifies the residual bottleneck (so the
"why was the last cost not optimized further" question is answered from data, not assumed).

Run on the cloud box (x86_64 c6a) after ``cargo build --release --features python``:
  python scripts/native_cfv_profile.py --nodes 600 --search-iters 200 \
      --blueprint-snapshot /tmp/bp.pkl --out /tmp/native_cfv_profile.json
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np


def _load_snapshot(path):
    import pickle
    with open(path, "rb") as f:
        snap = pickle.load(f)
    return snap["strat_sum"] if isinstance(snap, dict) and "strat_sum" in snap else snap


_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def _card(idx):
    from pokerbot.engine.cards import Card
    i = int(idx)
    return Card.from_str(_RANKS[i // 4] + _SUITS[i % 4])


def _pct(xs, p):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p)) if xs else None


def _collect_postflop_nodes(n_nodes: int, seed: int):
    """Play random hands; capture postflop (street>=2) multiway decision nodes — the regime native CFV
    serves. Mirrors native_cfv_lbr_equivalence._direct_equivalence's collector exactly."""
    from pokerbot.engine.actions import ActionType
    from pokerbot.engine.game import GameState, play_hand
    from pokerbot.search.state_bridge import game_state_to_env

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
    return nodes[:n_nodes]


def profile(snap, *, n_nodes: int, iterations: int, seed: int, repeats: int, budget_ms: int):
    """Time native-CFV ON vs interpreted-_cfv OFF over the collected nodes; component-split the ON arm.

    Each eligible node is solved ``repeats`` times per arm (the first run warms the per-adapter native
    engine + JIT caches; we report both incl- and excl-warmup percentiles). The component split times the
    Python tilt build (snapshot/restore-wrapped, as in solve()) and the single native eng.solve call
    separately, the same spans the production solve() pays."""
    import numpy as _np
    from pokerbot.engine.jax_weva_encode import _POSTFLOP_NUM_BUCKETS
    from pokerbot.engine.np_engine import legal_action_mask_np
    from pokerbot.opponent.stats import OpponentStats
    from pokerbot.search import native_solver as ns
    from pokerbot.search import pluribus_search as ps
    from pokerbot.search.pbs_tracker import PBSTracker
    from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
    from pokerbot.search.state_bridge import SubgameRoot

    cfg = ps.PluribusReBeLConfig(time_budget_ms=int(budget_ms),
                                 leaf=ContinuationLeafConfig(select="uniform"))
    cfg.max_iterations = int(iterations)
    solver = ps.PluribusReBeLSolver(snap, cfg, rng=_np.random.default_rng(seed))
    num_buckets = int(_POSTFLOP_NUM_BUCKETS)

    nodes = _collect_postflop_nodes(n_nodes, seed)

    def setup(env, hero, native):
        ps.set_native_cfv(bool(native))
        ps.set_native_subgame(bool(native))
        ns.enable() if native else ns.disable()
        opp = (hero + 1) % 6
        model = {opp: OpponentStats()}
        solver.opponent_model = model
        solver.rng = _np.random.default_rng(seed)
        board = [_card(int(c)) for c in _np.asarray(env.board) if int(c) >= 0]
        h0, h1 = int(env.holes[hero][0]), int(env.holes[hero][1])
        pbs = PBSTracker.init_uniform(hero, (_card(h0), _card(h1)), board)
        legal = legal_action_mask_np(env, hero).astype(bool)
        root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=pbs.get_public_belief_state(),
                           legal_mask=legal, opponent_profiles=model)
        return root, pbs

    on_ms, off_ms = [], []
    on_ms_warm, off_ms_warm = [], []
    build_ms, solve_ms, total_split_ms = [], [], []
    tilt_rows = []
    n_native = 0

    # Identify the eligible nodes first (ON arm must serve search_native_cfv).
    eligible = []
    for env, hero in nodes:
        root, pbs = setup(env, hero, True)
        res = solver.solve(root, pbs)
        served = (res.metadata or {}).get("served")
        if served == "search_native_cfv":
            eligible.append((env, hero))
            if solver._native_cfv_tilt_table is not None:
                tilt_rows.append(len(solver._native_cfv_tilt_table))
    n_native = len(eligible)

    for env, hero in eligible:
        # ON arm timings (repeats; first is warmup).
        for r in range(repeats):
            root, pbs = setup(env, hero, True)
            t0 = time.perf_counter()
            solver.solve(root, pbs)
            dt = (time.perf_counter() - t0) * 1000.0
            on_ms.append(dt)
            if r > 0:
                on_ms_warm.append(dt)
        # OFF arm timings.
        for r in range(repeats):
            root, pbs = setup(env, hero, False)
            t0 = time.perf_counter()
            solver.solve(root, pbs)
            dt = (time.perf_counter() - t0) * 1000.0
            off_ms.append(dt)
            if r > 0:
                off_ms_warm.append(dt)

        # COMPONENT SPLIT (ON arm): time the tilt build + the single native solve span the way solve()
        # pays them. We re-derive the exact inputs solve() uses.
        root, pbs = setup(env, hero, True)
        base_state = root.env
        # 1) leaf-cont-weights (must be None for native CFV to fire — confirm + cheap).
        solver._leaf_cont_weights = solver._build_leaf_cont_weights(hero)
        # 2) tilt build (snapshot/restore the RNG as solve() does).
        _rng_state = solver.rng.bit_generator.state
        tb0 = time.perf_counter()
        try:
            tilt = solver._build_native_cfv_tilt_table(base_state, hero)
        finally:
            solver.rng.bit_generator.state = _rng_state
        build = (time.perf_counter() - tb0) * 1000.0
        # 3) native solve span (the single FFI call: marshal-in + native CFR + marshal-out).
        sv0 = time.perf_counter()
        ns.native_subgame_solve(
            base_state, hero, solver.adapter,
            public_seed=solver._public_seed(base_state, hero),
            max_iterations=cfg.max_iterations, max_depth=cfg.max_depth,
            blueprint_mix=cfg.blueprint_mix,
            rollout_samples=cfg.leaf.rollout_samples,
            resample_board=cfg.leaf.resample_board,
            max_playout_depth=cfg.leaf.max_playout_depth,
            num_buckets=num_buckets,
            linear_averaging=bool(cfg.linear_averaging),
            dcfr=bool(cfg.dcfr), dcfr_alpha=float(cfg.dcfr_alpha),
            dcfr_beta=float(cfg.dcfr_beta), dcfr_gamma=float(cfg.dcfr_gamma),
            tilt_table=tilt,
        )
        solve_native = (time.perf_counter() - sv0) * 1000.0
        build_ms.append(build)
        solve_ms.append(solve_native)
        total_split_ms.append(build + solve_native)

    def arm(xs):
        return {"n": len(xs), "p50_ms": _pct(xs, 50), "p95_ms": _pct(xs, 95),
                "mean_ms": float(np.mean(xs)) if xs else None,
                "min_ms": float(np.min(xs)) if xs else None}

    on, off = arm(on_ms), arm(off_ms)
    on_w, off_w = arm(on_ms_warm), arm(off_ms_warm)
    speed_p50 = (off["p50_ms"] / on["p50_ms"]) if (on["p50_ms"] and on["p50_ms"] > 0) else None
    speed_p95 = (off["p95_ms"] / on["p95_ms"]) if (on["p95_ms"] and on["p95_ms"] > 0) else None
    speed_p50_w = (off_w["p50_ms"] / on_w["p50_ms"]) if (on_w["p50_ms"] and on_w["p50_ms"] > 0) else None
    speed_p95_w = (off_w["p95_ms"] / on_w["p95_ms"]) if (on_w["p95_ms"] and on_w["p95_ms"] > 0) else None

    comp = {
        "tilt_build": arm(build_ms),
        "native_solve": arm(solve_ms),
        "build_plus_solve": arm(total_split_ms),
        "tilt_build_share_of_split_p50": (
            (_pct(build_ms, 50) / _pct(total_split_ms, 50))
            if (total_split_ms and _pct(total_split_ms, 50)) else None),
        "tilt_rows_p50": _pct(tilt_rows, 50),
        "tilt_rows_p95": _pct(tilt_rows, 95),
    }

    return {
        "profile": "native_cfv_per_decision_latency",
        "regime": "cold_model (opponent_model set, 0 obs, leaf_cw None) — the only regime native CFV is reachable",
        "config": {"nodes_requested": n_nodes, "n_eligible_native_cfv": n_native,
                   "search_iters": iterations, "repeats": repeats, "budget_ms": budget_ms,
                   "snapshot_infosets": len(snap), "seed": seed},
        "interpreted_off": off, "native_cfv_on": on,
        "interpreted_off_warm": off_w, "native_cfv_on_warm": on_w,
        "speedup": {"p50": speed_p50, "p95": speed_p95,
                    "p50_warm": speed_p50_w, "p95_warm": speed_p95_w},
        "on_component_breakdown_ms": comp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=600,
                    help="postflop decision nodes to sweep (eligible subset is profiled)")
    ap.add_argument("--search-iters", type=int, default=200,
                    help="CFR iterations (the live converged target; 200 = production)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="solves per node per arm (first = warmup, excluded from *_warm percentiles)")
    ap.add_argument("--budget-ms", type=int, default=800,
                    help="time_budget_ms at the live budget (so the OFF arm reflects the budget-bound serve)")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--blueprint-snapshot", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--s3-bucket", default=None)
    ap.add_argument("--s3-key", default=None)
    ap.add_argument("--region", default="eu-central-1")
    args = ap.parse_args()

    snap = _load_snapshot(args.blueprint_snapshot)
    print(f"blueprint: {len(snap)} infosets  nodes={args.nodes} iters={args.search_iters} "
          f"repeats={args.repeats} budget_ms={args.budget_ms}", flush=True)

    t0 = time.time()
    payload = profile(snap, n_nodes=args.nodes, iterations=args.search_iters, seed=args.seed,
                      repeats=args.repeats, budget_ms=args.budget_ms)
    payload["wall_min"] = (time.time() - t0) / 60.0

    off, on = payload["interpreted_off"], payload["native_cfv_on"]
    onw, offw = payload["native_cfv_on_warm"], payload["interpreted_off_warm"]
    sp = payload["speedup"]
    comp = payload["on_component_breakdown_ms"]
    print("\n==================== NATIVE CFV PER-DECISION PROFILE ====================")
    print(f"  eligible native-CFV nodes: {payload['config']['n_eligible_native_cfv']}/{args.nodes}")
    print(f"  OFF (interpreted _cfv): p50 {off['p50_ms']:.2f}ms  p95 {off['p95_ms']:.2f}ms")
    print(f"  ON  (native CFV)      : p50 {on['p50_ms']:.2f}ms  p95 {on['p95_ms']:.2f}ms")
    print(f"  SPEEDUP (incl warmup) : p50 {sp['p50']:.2f}x  p95 {sp['p95']:.2f}x")
    if onw["p50_ms"] and offw["p50_ms"]:
        print(f"  OFF warm: p50 {offw['p50_ms']:.2f}ms  p95 {offw['p95_ms']:.2f}ms")
        print(f"  ON  warm: p50 {onw['p50_ms']:.2f}ms  p95 {onw['p95_ms']:.2f}ms")
        print(f"  SPEEDUP (warm)        : p50 {sp['p50_warm']:.2f}x  p95 {sp['p95_warm']:.2f}x")
    print(f"  ON component p50: tilt_build {comp['tilt_build']['p50_ms']:.2f}ms  "
          f"native_solve {comp['native_solve']['p50_ms']:.2f}ms  "
          f"(build share {comp['tilt_build_share_of_split_p50']:.2%})")
    print(f"  tilt rows/decision p50={comp['tilt_rows_p50']}  p95={comp['tilt_rows_p95']}")
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
