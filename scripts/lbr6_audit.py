"""6-max Local Best Response audit -- certified lower bound on the exploitability of the 6-max base.

Run the greedy MC-rollout best-responder (pokerbot.eval.lbr6) against a table of 5 copies of a target
policy and report how much it wins (bb/100). Calibration targets (always-fold / uniform / nit) MUST come
back hugely positive -- they are trivially exploitable -- which validates the harness before trusting the
blueprint number.

Examples
--------
  # Calibration (fast, should be large positive):
  python scripts/lbr6_audit.py --target foldbot --hands 200 --rollouts 12
  python scripts/lbr6_audit.py --target uniform --hands 200 --rollouts 12

  # The real measurement: the deployed 6-max DCFR+VR blueprint foundation
  python scripts/lbr6_audit.py --target blueprint --hands 800 --rollouts 20 \
      --blueprint-snapshot checkpoints/bp6max_dcfrvr_iter200.pkl
"""

from __future__ import annotations

import argparse
import pickle
import time

# Abstraction env must be set before pokerbot encoders import; lbr6 sets the same defaults at import.
from pokerbot.eval import lbr6
from pokerbot.eval.lbr import (always_fold_policy, make_blueprint_policy, make_nit_policy,
                               uniform_policy)


def _load_snapshot(path):
    # cert7: a DIRECTORY is the nlcfr_key_index.py --with-rows mmap index (SortedSnapshotDir, the
    # full snap_avg-merged serve table). make_blueprint_policy touches the snapshot only via
    # snapshot.get(key) (lbr.py), which SortedSnapshotDir duck-types -- so the pure-python LBR6
    # serves the full index directly (no 25M row cap, no pkl materialization). Default (a file)
    # path is byte-identical to the prior behavior.
    import os as _os
    if _os.path.isdir(path):
        import sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from engine_ab_h2h import load_snapshot as _load_dir
        return _load_dir(path)
    with open(path, "rb") as f:
        snap_full = pickle.load(f)
    return (snap_full["strat_sum"]
            if isinstance(snap_full, dict) and "strat_sum" in snap_full else snap_full)


def _station_policy(env, seat):
    """Calling station: never folds, never bets -- call when facing a bet, else check. The classic
    max-payoff leak; LBR must crush it (value-betting competence check)."""
    import numpy as np
    from pokerbot.engine.np_engine import CALL, CHECK, FOLD, NUM_ACTIONS, legal_action_mask_np
    mask = legal_action_mask_np(env, seat).astype(bool)
    p = np.zeros(NUM_ACTIONS, dtype=np.float64)
    if mask[CALL]:
        p[CALL] = 1.0
    elif mask[CHECK]:
        p[CHECK] = 1.0
    elif mask[FOLD]:
        p[FOLD] = 1.0
    else:
        p[int(np.argmax(mask))] = 1.0
    return p


def _build_policy(args):
    """Return (act_fn, label, model_fn). ``act_fn`` is what the table seats actually play (the strategy
    whose exploitability we bound); ``model_fn`` (or None) is the cheap white-box the LBR uses internally
    for range tracking + fold-equity reads -- only differs from act_fn for the expensive search target."""
    if args.target == "uniform":
        return uniform_policy, "uniform-legal", None
    if args.target == "foldbot":
        return always_fold_policy, "always-fold", None
    if args.target == "station":
        return _station_policy, "calling-station", None
    if args.target == "nit":
        return make_nit_policy(), "nit (premium-only)", None
    if args.target == "blueprint":
        snap = _load_snapshot(args.blueprint_snapshot)
        print(f"blueprint snapshot: {len(snap)} infosets from {args.blueprint_snapshot}")
        return make_blueprint_policy(snap), f"blueprint ({len(snap)} infosets)", None
    if args.target == "blueprint-search":
        # The DEPLOYED base: blueprint + Pluribus depth-limited search. Seats ACT via search; the LBR
        # range-tracks against the cheap raw blueprint (model_fn) so the run stays tractable (one search
        # solve per opponent decision instead of O(range_cap) per decision). Valid lower bound regardless.
        from pokerbot.eval.solver_exploitability import make_search_policy_fn
        snap = _load_snapshot(args.blueprint_snapshot)
        print(f"blueprint+search: {len(snap)} infosets, leaf={args.search_leaf}, "
              f"iters={args.search_iters}, budget={args.search_budget_ms}ms")
        act = make_search_policy_fn(snap, leaf_select=args.search_leaf,
                                    time_budget_ms=args.search_budget_ms, seed=args.seed,
                                    iterations=args.search_iters,
                                    search_preflop=getattr(args, "search_preflop", False))
        model = make_blueprint_policy(snap)
        cfgtxt = (f"iter{args.search_iters}" if args.search_iters > 0 else f"{args.search_budget_ms}ms")
        return act, f"blueprint+search ({args.search_leaf} leaf, {cfgtxt})", model
    raise ValueError(args.target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target",
                    choices=("uniform", "foldbot", "station", "nit", "blueprint", "blueprint-search"),
                    default="blueprint",
                    help="blueprint = RAW tabular lookup (uniform-on-miss). blueprint-search = the DEPLOYED "
                         "base (blueprint + Pluribus depth-limited search), i.e. what the bot actually "
                         "plays. Use blueprint-search to measure whether search repairs the raw leak.")
    ap.add_argument("--hands", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollouts", type=int, default=20)
    ap.add_argument("--n-seats", type=int, default=6)
    ap.add_argument("--no-track", action="store_true",
                    help="disable Bayesian range tracking (fast uniform-belief path; calibration only)")
    ap.add_argument("--unpaired", action="store_true",
                    help="report LBR's raw absolute EV (no CRN base-arm pairing; much higher variance)")
    ap.add_argument("--passive-cont", action="store_true",
                    help="use the passive call-down continuation (valid but weaker) instead of the default "
                         "valid belief-based competent continuation")
    ap.add_argument("--menu", choices=("fc", "fcpa", "full"), default="full",
                    help="LBR action menu (Lisy-Bowling graded protocol): fc=fold/call only (the robustness "
                         "FLOOR -- if this can't exploit you, you're very robust), fcpa=+pot+all-in, "
                         "full=+half-pot. Escalate fc->fcpa->full.")
    ap.add_argument("--rollout", action="store_true",
                    help="use the LEGACY MC-rollout LBR instead of the analytic 1-ply LBR (default). The "
                         "rollout adversary can bleed below fold vs strong opponents; analytic cannot.")
    ap.add_argument("--no-allin", action="store_true",
                    help="(rollout legacy) drop all-in from the bet menu; prefer --menu for the graded set")
    ap.add_argument("--range-cap", type=int, default=80)
    ap.add_argument("--search-leaf", choices=("uniform", "gadget"), default="uniform",
                    help="(blueprint-search) Pluribus continuation leaf. uniform = deployed/floor config; "
                         "gadget = the stronger adversarial leaf (much slower).")
    ap.add_argument("--search-iters", type=int, default=10,
                    help="(blueprint-search) FIXED CFR+ iterations per opponent solve (box-speed "
                         "independent, reproducible). 10 = the config the +9.1 basestrength base was "
                         "measured at. 0 => use --search-budget-ms wall clock instead.")
    ap.add_argument("--search-budget-ms", type=int, default=300,
                    help="(blueprint-search) per-decision wall budget when --search-iters 0 (live uses "
                         "~2000ms; NOT reproducible across boxes/parallelism -- prefer --search-iters).")
    ap.add_argument("--search-preflop", action="store_true",
                    help="(blueprint-search) ALSO search preflop (default: preflop from blueprint, the "
                         "Pluribus default). Tests whether searching the preflop fixes the preflop leak.")
    ap.add_argument("--search-fold-equity", action="store_true",
                    help="(blueprint-search) credit fold equity on raises by reading search's fold prob "
                         "from the cheap blueprint MODEL. Off by default: that model error makes the LBR "
                         "suicide-bluff into search (-600 bb/100). Default = value-only raises (robust).")
    ap.add_argument("--out", default=None,
                    help="write the result as JSON here (poolable across independent seed-runs: combine "
                         "by hand-weighted mean + pooled variance to shrink the CI like a single large N)")
    ap.add_argument("--blueprint-snapshot",
                    default=r"checkpoints/bp6max_dcfrvr_iter200.pkl")
    args = ap.parse_args()

    policy_fn, label, model_fn = _build_policy(args)
    cfg = lbr6.LBR6Config(hands=args.hands, seed=args.seed, rollouts=args.rollouts, n_seats=args.n_seats,
                          track_ranges=not args.no_track, range_cap=args.range_cap,
                          paired=not args.unpaired, analytic=not args.rollout)
    _MENUS = {"fc": (), "fcpa": (lbr6._POT_BET_INDEX, lbr6.ALL_IN),
              "full": (lbr6._HALF_POT_INDEX, lbr6._POT_BET_INDEX, lbr6.ALL_IN)}
    cfg.bet_candidates = _MENUS[args.menu]
    if args.passive_cont:
        cfg.smart_continuation = False
    if args.no_allin:
        cfg.bet_candidates = (lbr6._HALF_POT_INDEX, lbr6._POT_BET_INDEX)
    if args.target == "blueprint-search" and cfg.paired:
        # The base arm would run search at the LBR seat too -> doubles solves for no gain (LBR and base
        # diverge too much to correlate; pairing never cut variance here). Force the raw unpaired estimator.
        cfg.paired = False
        print("blueprint-search: forcing --unpaired (paired base arm would double search cost, no benefit)")
    # Value-only raises vs SEARCH: the LBR cannot cheaply model search's fold probs, so a blueprint-read
    # fold-equity term makes it suicide-bluff into search (-600 bb/100). Default to value-only raises (bet
    # only on showdown value, no fold-equity credit) for the search target; --search-fold-equity opts back
    # into the (untrustworthy here) white-box fold read.
    if args.target == "blueprint-search" and not args.search_fold_equity:
        cfg.value_only_raises = True
        print("blueprint-search: value-only raises ON (no fold-equity credit; robust to un-modellable search)")

    t0 = time.time()
    res = lbr6.lbr6_exploitability(policy_fn, cfg, model_fn=model_fn)
    dt = time.time() - t0

    lo = res.bb_per_100 - 1.96 * res.stderr_bb_per_hand * 100.0
    hi = res.bb_per_100 + 1.96 * res.stderr_bb_per_hand * 100.0
    mode = "paired (LBR - base copy, CRN)" if res.detail.get("paired") else "unpaired (raw absolute)"
    print(f"\n=== LBR6 vs {label}  ({args.n_seats}-max, {args.hands} hands x {args.rollouts} rollouts) ===")
    print(f"  estimator  : {mode}")
    print(f"  LBR edge   : {res.bb_per_100:+.2f} bb/100   (95% CI [{lo:+.2f}, {hi:+.2f}])")
    if res.detail.get("paired"):
        print(f"  (raw BR EV : {res.detail['br_raw_bb_per_100']:+.2f} bb/100, high-variance reference)")
    print(f"  mbb/g      : {res.mbb_per_g:+.1f}")
    print(f"  stderr     : {res.stderr_bb_per_hand * 100.0:.2f} bb/100")
    print(f"  decisions  : {res.detail['lbr_decisions']}   ({dt:.1f}s, {dt / max(args.hands,1):.2f}s/hand)")
    verdict = ("BEATABLE: a best-responder profits" if lo > 0
               else "robust at this CI (LBR cannot prove a win)")
    print(f"  verdict    : {verdict}")
    print(f"\n  {res.note}")

    if args.out:
        import json
        std = res.stderr_bb_per_hand * (args.hands ** 0.5)  # per-hand std (for pooling)
        payload = {
            "target": args.target, "label": label, "n_seats": args.n_seats,
            "hands": res.hands, "seed": args.seed, "rollouts": args.rollouts,
            "ev_bb_per_hand": res.ev_bb_per_hand, "bb_per_100": res.bb_per_100,
            "stderr_bb_per_hand": res.stderr_bb_per_hand, "std_bb_per_hand": std,
            "detail": res.detail,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote      : {args.out}")


if __name__ == "__main__":
    main()
