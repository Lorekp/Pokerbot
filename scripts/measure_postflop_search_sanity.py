"""MEASURE: is postflop search sane, or does it inherit the one-sided-CFR over-aggression?

For each captured postflop root (artifacts/search_roots.pkl, ~60 flop/turn/river nodes) we
compare the RAW blueprint distribution vs blueprint+search at several iteration counts.

Metrics per node & aggregated by street:
  * all_in_mass  = probs[ALL_IN]
  * bet_mass     = sum(probs[BET_BUCKET_START:])           # all aggressive (bet/raise incl all-in)
  * aggr_mass    = bet_mass (facing nothing) OR includes RAISE over CALL when facing a bet
  * argmax shift = did search's argmax action differ from blueprint's argmax?
  * L1 distance  = 0.5 * sum|search - blueprint|
Growth-with-iterations: does aggression rise K200 -> K2000 -> K5000 (the preflop degeneracy signature)?
"""
from __future__ import annotations

import os
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "emd")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "160")
# EMD-K160 deployed regime (promoted 2026-07-02, certified +32.89 bb/100 vs made-hand iter62000):
# frozen flop/turn EMD lookup + train-mirrored made-hand fallback on the river (the river is not
# in the blob; without the mirror flag the river frozen-miss serves a UNIFORM policy).
os.environ.setdefault("POKERBOT_POSTFLOP_EMD_FROZEN", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn")))
os.environ.setdefault("POKERBOT_EMD_FROZEN_MIRROR_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import json
import pickle
import sys
import time
import numpy as np

from pokerbot.engine.np_engine import (
    ALL_IN, BET_BUCKET_START, NUM_ACTIONS, FOLD, CHECK, CALL,
    legal_action_mask_np,
)
from pokerbot.eval.lbr import make_blueprint_policy
from pokerbot.eval.solver_exploitability import make_search_policy_fn

STREET_NAME = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}


def facing_bet(env, seat) -> bool:
    """True if hero faces a live bet (CALL is legal & costs chips / FOLD legal)."""
    legal = legal_action_mask_np(env, seat).astype(bool)
    # FOLD legal at a non-check spot => there is a bet to call
    return bool(legal[FOLD])


def summarize(probs, legal):
    p = np.asarray(probs, dtype=np.float64)
    all_in = float(p[ALL_IN])
    bet = float(p[BET_BUCKET_START:].sum())         # all bet/raise buckets incl all-in
    call = float(p[CALL])
    check = float(p[CHECK])
    fold = float(p[FOLD])
    argmax = int(np.argmax(np.where(legal, p, -1.0)))
    return dict(all_in=all_in, bet=bet, call=call, check=check, fold=fold, argmax=argmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="artifacts/search_roots.pkl")
    ap.add_argument("--snapshot", default="checkpoints/serve_blueprint_6max.pkl")
    ap.add_argument("--iters", type=int, nargs="+", default=[200, 2000, 5000])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="limit #roots (0=all)")
    ap.add_argument("--jsonl", default="", help="write per-node records here (line-buffered)")
    args = ap.parse_args()

    jf = open(args.jsonl, "w", buffering=1) if args.jsonl else None

    with open(args.roots, "rb") as f:
        roots = pickle.load(f)
    if args.limit:
        roots = roots[: args.limit]

    with open(args.snapshot, "rb") as f:
        obj = pickle.load(f)
    snap = obj["strat_sum"] if isinstance(obj, dict) and "strat_sum" in obj else obj
    print(f"snapshot: {len(snap)} infosets ({args.snapshot})", flush=True)
    print(f"roots: {len(roots)}", flush=True)

    bp_policy = make_blueprint_policy(snap)
    search_policies = {
        K: make_search_policy_fn(snap, leaf_select="uniform", seed=args.seed,
                                 iterations=K, search_preflop=False)
        for K in args.iters
    }

    # rows: per-root records
    records = []
    t0 = time.time()
    for ri, r in enumerate(roots):
        env = r["env"]
        seat = int(r["hero_seat"])
        street = int(r["street"])
        legal = legal_action_mask_np(env, seat).astype(bool)
        fb = facing_bet(env, seat)
        bp = summarize(bp_policy(env, seat), legal)
        rec = dict(idx=ri, street=street, facing_bet=fb, legal=legal.copy(), bp=bp, search={})
        for K in args.iters:
            sp = summarize(search_policies[K](env, seat), legal)
            rec["search"][K] = sp
        records.append(rec)
        if jf is not None:
            dump = dict(idx=ri, street=street, facing_bet=fb,
                        bp={k: v for k, v in bp.items()},
                        search={str(K): rec["search"][K] for K in args.iters})
            jf.write(json.dumps(dump) + "\n")
        print(f"  node {ri+1}/{len(roots)} street={street} "
              f"bp_ai={bp['all_in']:.3f} bp_bet={bp['bet']:.3f} "
              + " ".join(f"K{K}_ai={rec['search'][K]['all_in']:.3f} K{K}_bet={rec['search'][K]['bet']:.3f}"
                         for K in args.iters)
              + f"  ({(time.time()-t0):.0f}s)", flush=True)
        sys.stdout.flush()

    # ---- aggregate report ----
    def agg(recs, key):
        return np.mean([x[key] for x in recs]) if recs else float("nan")

    print("\n" + "=" * 78)
    print("POSTFLOP SEARCH vs RAW BLUEPRINT — aggression comparison")
    print("=" * 78)

    for st in (1, 2, 3):
        sub = [r for r in records if r["street"] == st]
        if not sub:
            continue
        print(f"\n--- {STREET_NAME[st].upper()} ({len(sub)} nodes) ---")
        bp_all_in = agg([r["bp"] for r in sub], "all_in")
        bp_bet = agg([r["bp"] for r in sub], "bet")
        print(f"  BLUEPRINT    : all_in={bp_all_in:.3f}  bet_mass={bp_bet:.3f}")
        for K in args.iters:
            s_all_in = agg([r["search"][K] for r in sub], "all_in")
            s_bet = agg([r["search"][K] for r in sub], "bet")
            # argmax shifts
            shifts = sum(1 for r in sub if r["search"][K]["argmax"] != r["bp"]["argmax"])
            # argmax -> more aggressive?
            more_aggr = sum(
                1 for r in sub
                if r["search"][K]["argmax"] >= BET_BUCKET_START and r["bp"]["argmax"] < BET_BUCKET_START
            )
            to_allin = sum(
                1 for r in sub
                if r["search"][K]["argmax"] == ALL_IN and r["bp"]["argmax"] != ALL_IN
            )
            print(f"  SEARCH K={K:<5d}: all_in={s_all_in:.3f} ({s_all_in-bp_all_in:+.3f})  "
                  f"bet_mass={s_bet:.3f} ({s_bet-bp_bet:+.3f})  "
                  f"argmax_shift={shifts}/{len(sub)}  ->more_aggr={more_aggr}  ->all_in={to_allin}")

    # ---- growth with iterations (degeneracy signature) ----
    print("\n" + "=" * 78)
    print("GROWTH WITH ITERATIONS (preflop degeneracy signature = aggression RISES with K)")
    print("=" * 78)
    for st in (1, 2, 3):
        sub = [r for r in records if r["street"] == st]
        if not sub:
            continue
        series_ai = [agg([r["search"][K] for r in sub], "all_in") for K in args.iters]
        series_bet = [agg([r["search"][K] for r in sub], "bet") for K in args.iters]
        print(f"  {STREET_NAME[st]:6s}  all_in: " +
              " -> ".join(f"K{K}={v:.3f}" for K, v in zip(args.iters, series_ai)))
        print(f"          bet   : " +
              " -> ".join(f"K{K}={v:.3f}" for K, v in zip(args.iters, series_bet)))

    # ---- facing-a-bet subset (does search over-continue / over-raise) ----
    print("\n" + "=" * 78)
    print("FACING-A-BET subset (does hero over-call/over-raise a bet, analogous to over-jam?)")
    print("=" * 78)
    fb_sub = [r for r in records if r["facing_bet"]]
    nofb_sub = [r for r in records if not r["facing_bet"]]
    print(f"  facing-bet nodes: {len(fb_sub)}   not-facing (can check): {len(nofb_sub)}")
    for label, sub in (("FACING BET", fb_sub), ("NOT FACING (open)", nofb_sub)):
        if not sub:
            continue
        bp_bet = agg([r["bp"] for r in sub], "bet")
        bp_fold = agg([r["bp"] for r in sub], "fold")
        print(f"  [{label}] BLUEPRINT bet={bp_bet:.3f} fold={bp_fold:.3f}")
        for K in args.iters:
            s_bet = agg([r["search"][K] for r in sub], "bet")
            s_fold = agg([r["search"][K] for r in sub], "fold")
            print(f"           SEARCH K={K:<5d} bet={s_bet:.3f} ({s_bet-bp_bet:+.3f}) "
                  f"fold={s_fold:.3f} ({s_fold-bp_fold:+.3f})")

    # ---- per-node worst offenders (biggest all_in jump at max K) ----
    Kmax = max(args.iters)
    print("\n" + "=" * 78)
    print(f"WORST all_in OVER-SHIFTS at K={Kmax} (search all_in - blueprint all_in)")
    print("=" * 78)
    deltas = sorted(records, key=lambda r: r["search"][Kmax]["all_in"] - r["bp"]["all_in"], reverse=True)
    for r in deltas[:8]:
        d = r["search"][Kmax]["all_in"] - r["bp"]["all_in"]
        print(f"  node {r['idx']:2d} {STREET_NAME[r['street']]:6s} facing_bet={r['facing_bet']} "
              f"bp_all_in={r['bp']['all_in']:.3f} search_all_in={r['search'][Kmax]['all_in']:.3f} "
              f"(d{d:+.3f}) bp_argmax={r['bp']['argmax']} search_argmax={r['search'][Kmax]['argmax']}")

    print(f"\ntotal wall: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
