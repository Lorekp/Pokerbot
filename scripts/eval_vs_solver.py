#!/usr/bin/env python
"""Measure OUR policy's POSTFLOP exploitability vs the in-repo near-exact GTO solver.

This is the GTO-Wizard-style low-variance quality metric: how much EV a best response gains
against us (lower = closer to GTO), measured EXACTLY via the postflop-solver crate's node
locking — not the high-variance win-rate bb/100. For each representative HU postflop spot it
reports, in BOTH % of pot and bb/100:

  (a) the solver's OWN GTO-solution exploitability (sanity: ~ target, confirms wiring),
  (b) our BLUEPRINT-alone exploitability,
  (c) our BLUEPRINT+SEARCH exploitability,

and the headline: does the SEARCH layer measurably REDUCE exploitability vs blueprint-alone?

Run (git-bash; ALWAYS export the abstraction env first):

    POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_POSTFLOP_MODE=emd \
    POKERBOT_POSTFLOP_NUM_BUCKETS=160 \
    POKERBOT_POSTFLOP_EMD_FROZEN=<abs>/data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn \
    POKERBOT_EMD_FROZEN_MIRROR_FALLBACK=1 python scripts/eval_vs_solver.py \
        --config config/solver_labels_smoke.yaml \
        --blueprint "$LOCALAPPDATA/Temp/bp6max_dcfrvr_iter200.pkl" \
        --max-spots 6 --leaf uniform

By default only the blueprint is evaluated (fast). Pass --with-search to add the
blueprint+search column (much slower: a depth-limited CFR solve per private hand per node).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

# Abstraction env MUST be set before importing the engine/encoders (fingerprint parity).
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

import numpy as np  # noqa: E402

from pokerbot.eval import solver_exploitability as se  # noqa: E402
from pokerbot.solver.spot_enumerator import (  # noqa: E402
    SolverSpot,
    enumerate_valid_spots,
    load_solver_config,
)


def _load_blueprint(path: Path) -> dict[bytes, np.ndarray]:
    obj = pickle.load(open(path, "rb"))
    snap = obj["strat_sum"] if isinstance(obj, dict) and "strat_sum" in obj else obj
    return snap


def _hero_tag(spot: SolverSpot) -> str:
    return "OOP" if spot.hero_seat == spot.oop_seat else "IP"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/solver_labels_smoke.yaml")
    ap.add_argument("--blueprint",
                    default=str(Path.home() / "AppData/Local/Temp/bp6max_dcfrvr_iter200.pkl"))
    ap.add_argument("--max-spots", type=int, default=6,
                    help="cap on number of spots (each solve is bounded but not free).")
    ap.add_argument("--street", default="river", choices=["flop", "turn", "river"],
                    help="subtree to evaluate. 'river' (default) derives a single-street RIVER spot "
                         "from each flop spot: fully enumerable, solves to a TIGHT target fast, so the "
                         "GTO residual is << the bp-vs-search effect. 'flop' is the full tree (slow; "
                         "residual too large to resolve small deltas).")
    ap.add_argument("--max-board-len", type=int, default=None,
                    help="override deepest locked street (3=flop,4=turn,5=river). Default: from --street.")
    ap.add_argument("--with-search", action="store_true",
                    help="also evaluate blueprint+search (slow).")
    ap.add_argument("--leaf", default="uniform", choices=["uniform", "gadget"],
                    help="pluribus depth-limit leaf-select strategy.")
    ap.add_argument("--search-budget-ms", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-pct", type=float, default=None,
                    help="override solver target_exploitability_pct (default: config value).")
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--big-blind-bb", type=float, default=1.0)
    args = ap.parse_args(argv)

    cfg = load_solver_config(args.config)
    spots, excluded = enumerate_valid_spots(cfg)
    if not spots:
        print(f"No valid spots in {args.config} (excluded={dict(excluded)})", file=sys.stderr)
        return 2
    spots = spots[: args.max_spots]
    # Reduce each flop spot to the chosen single-street subtree (default RIVER) for a TIGHT,
    # trustworthy residual; 'flop' keeps the full (slow) tree.
    spots = [se.derive_street_spot(s, args.street) for s in spots]
    max_board_len = (args.max_board_len if args.max_board_len is not None
                     else {"flop": 3, "turn": 4, "river": 5}[args.street])

    bt = cfg.bet_tree
    sv = cfg.solver
    # Default to a TIGHT target on single-street subtrees (they converge fast): the residual must
    # be << the bp-vs-search effect. Full-flop trees keep the (loose) config target since a tight
    # one is intractable there. --target-pct overrides either.
    if args.target_pct is not None:
        target_pct = float(args.target_pct)
    elif args.street == "flop":
        target_pct = float(sv.get("target_exploitability_pct", 1.0))
    else:
        target_pct = 0.1
    kw = dict(
        bet_fractions=[float(x) for x in bt.get("pot_fractions", [0.5])],
        raise_multiplier=float(bt.get("raise_multiplier", 2.5)),
        max_iterations=int(args.max_iters or sv.get("max_iterations", 2000)),
        target_exploitability_pct=target_pct,
        chip_unit_bb=float(sv.get("chip_unit_bb", 100)),
    )

    bp_path = Path(args.blueprint)
    if not bp_path.is_file():
        print(f"Blueprint not found: {bp_path}", file=sys.stderr)
        return 2
    snap = _load_blueprint(bp_path)
    bp_policy = se.make_blueprint_policy_fn(snap)
    search_policy = None
    if args.with_search:
        search_policy = se.make_search_policy_fn(
            snap, leaf_select=args.leaf, time_budget_ms=args.search_budget_ms, seed=args.seed
        )

    print(f"# Postflop exploitability vs in-repo GTO solver")
    print(f"# config={args.config}  spots={len(spots)}  street={args.street} max_board_len={max_board_len}")
    print(f"# bet_tree pot_fractions={kw['bet_fractions']} target={kw['target_exploitability_pct']}% "
          f"max_iters={kw['max_iterations']}  leaf={args.leaf if args.with_search else 'n/a'}")
    print()

    hdr = ["spot_id", "board", "hero", "gto_%pot", "bp_%pot", "bp_bb/100"]
    if search_policy is not None:
        hdr += ["srch_%pot", "srch_bb/100", "d%pot(s-bp)"]
    print("  ".join(f"{h:>14}" for h in hdr))
    print("  ".join("-" * 14 for _ in hdr))

    gto_vals, bp_vals, srch_vals = [], [], []
    gate_ok = True
    for spot in spots:
        gb = se.gto_baseline(spot, **kw)
        anchor = gb.opp_equilibrium_value_chips  # share the free-solve GTO anchor with policies
        bp = se.policy_exploitability(spot, bp_policy, opp_equilibrium_value_chips=anchor,
                                      max_board_len=max_board_len,
                                      big_blind_bb=args.big_blind_bb, **kw)
        row = [spot.spot_id[:14], bp.board, _hero_tag(spot),
               f"{gb.exploitability_pct_of_pot:.3f}",
               f"{bp.exploitability_pct_of_pot:.3f}",
               f"{bp.exploitability_bb_per_100:.2f}"]
        gto_vals.append(gb.exploitability_pct_of_pot)
        bp_vals.append(bp.exploitability_pct_of_pot)
        # Killer gate: GTO must be the LEAST exploitable (within solve tolerance ~ target%).
        tol = kw["target_exploitability_pct"]
        if not bp.error and bp.exploitability_pct_of_pot + tol < gb.exploitability_pct_of_pot:
            gate_ok = False
        if search_policy is not None:
            sr = se.policy_exploitability(spot, search_policy, opp_equilibrium_value_chips=anchor,
                                          max_board_len=max_board_len,
                                          big_blind_bb=args.big_blind_bb, **kw)
            srch_vals.append(sr.exploitability_pct_of_pot)
            delta = sr.exploitability_pct_of_pot - bp.exploitability_pct_of_pot
            row += [f"{sr.exploitability_pct_of_pot:.3f}",
                    f"{sr.exploitability_bb_per_100:.2f}", f"{delta:+.3f}"]
            if sr.error:
                row[-1] = f"ERR:{sr.error[:20]}"
            elif sr.exploitability_pct_of_pot + tol < gb.exploitability_pct_of_pot:
                gate_ok = False
        if bp.error:
            row[4] = f"ERR:{bp.error[:18]}"
        print("  ".join(f"{c:>14}" for c in row))

    print()
    print(f"## Summary (mean ONE-SIDED exploitability, % of pot over {len(spots)} spots)")
    mgto = float(np.mean(gto_vals)) if gto_vals else 0.0
    # Solve noise floor: the worst-case GTO residual (an upper bound on per-spot solve noise).
    noise = float(np.max(gto_vals)) if gto_vals else kw["target_exploitability_pct"]
    print(f"  solver GTO floor      : {mgto:.3f} %pot  (one-sided residual; solve noise ~ "
          f"max {noise:.3f} %pot)")
    print(f"  blueprint-alone       : {np.mean(bp_vals):.3f} %pot")
    if srch_vals:
        msrch = float(np.mean(srch_vals))
        mbp = float(np.mean(bp_vals))
        print(f"  blueprint+search      : {msrch:.3f} %pot")
        delta = msrch - mbp
        # Honest verdict: the per-spot effect must clear the solve noise to count, not 1e-3.
        if delta < -noise:
            verdict = "SEARCH REDUCES exploitability (margin clears solve noise)"
        elif delta > noise:
            verdict = "SEARCH INCREASES exploitability (margin clears solve noise)"
        else:
            verdict = "WASH: |delta| within solve noise => no resolvable difference"
        print(f"  d (search - bp)       : {delta:+.3f} %pot  =>  {verdict}")
    print()
    # KILLER GATE report: GTO must be the minimum on every spot (within tolerance).
    print(f"## Killer gate (GTO <= every policy, tol={kw['target_exploitability_pct']}%pot): "
          f"{'PASS' if gate_ok else 'FAIL — metric still inconsistent'}")
    print()
    print("Notes: ONE-SIDED metric = opponent's best-response value vs our LOCKED hero MINUS the "
          "opponent's EQUILIBRIUM value (free solve), so GTO/blueprint/search are the SAME quantity "
          "and GTO is the provable minimum. HU postflop vs 6-max blueprint (reconstructed HU states); "
          "ranges from the config; solver bet sizes bucketed to our 11-action abstraction. Exact for "
          "the locked region (hero nodes up to the chosen street; deeper streets solve freely == GTO).")
    return 0 if gate_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
