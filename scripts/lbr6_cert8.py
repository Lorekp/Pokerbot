"""cert8 LBR6 driver -- HONEST exploitability lower bound with a BETTING menu, per-hand dump.

Fixes the cert7 LBR defects:
  * cert7 used ``--menu fc`` (fold/call only) -> NOT a best-responder (cannot bet, so it can only
    realise showdown value by calling; it cannot punish over-folding). cert8 uses a BETTING menu
    (default ``full`` = half-pot + pot + all-in, the richest graded Lisy-Bowling set lbr6 supports).
  * cert7's headline used the DEFAULT ``paired`` estimator (LBR - base copy). In >=3-player poker the
    LBR and base arms play different lines, so they are UNCORRELATED and pairing ADDS the base arm's
    variance instead of cancelling it. The base arm's per-hand reward has expectation 0 by table
    symmetry but at N<=800 carries +/-200..520 bb/100 of pure sampling noise, which the paired diff
    subtracts -- producing a large spurious sign (incumbent: paired -170.69 = raw +30.60 - base +201.3;
    new: paired -625.33 = raw -102.82 - base +522.5). cert8 ALWAYS uses the RAW/unpaired estimator and
    reduces variance HONESTLY via (a) large N, (b) seat rotation (built in), and (c) CRN ACROSS ARMS
    (same seeds -> new-vs-incumbent and search-vs-raw contrasts share the deck luck; done in the
    assembler by matching (seed, hand_idx)).

Emits BOTH a pooled summary JSON and a per-hand JSONL (seed, idx, seat, reward, streets, folded) so
the assembler can (1) pool shards into one large-N CI, (2) form the CRN paired contrasts, and (3) run
the per-street / per-position profit attribution (where the best-responder makes its money).

This is the trustworthy instrument: an ``--unpaired`` lbr6_audit and a ``--raw`` lbr6_attribute run at
the same seed/hands/menu are BIT-IDENTICAL to this driver's raw EV (same rng, same policy, paired=False)
-- the cert8 bootstrap runs that reconciliation as stage 0 and asserts agreement before spending.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

from pokerbot.eval import lbr6
from pokerbot.eval.lbr import make_blueprint_policy


def _load_snapshot(path):
    """A DIRECTORY = the nlcfr_key_index mmap index (SortedSnapshotDir, duck-typed .get); a FILE = pkl."""
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


def _build(args):
    """Return (act_fn, model_fn, label, value_only). RAW blueprint: act=model=blueprint, full menu valid
    (white-box fold-equity reads). blueprint-SEARCH: seats ACT via Pluribus search, LBR range-tracks +
    reads fold equity from the cheap blueprint MODEL; value-only raises forced (search fold probs are not
    cheaply modellable -> a blueprint-read fold-equity term makes the LBR suicide-bluff into search)."""
    snap = _load_snapshot(args.blueprint_snapshot)
    n = len(snap)
    model = make_blueprint_policy(snap)
    if args.target == "blueprint":
        return model, model, f"raw-blueprint({n})", (args.menu == "fc")
    if args.target == "blueprint-search":
        from pokerbot.eval.solver_exploitability import make_search_policy_fn
        act = make_search_policy_fn(snap, leaf_select=args.search_leaf, iterations=args.search_iters,
                                    seed=args.seed, search_preflop=False)
        return act, model, f"blueprint+search({args.search_leaf},iter{args.search_iters})", True
    raise ValueError(args.target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("blueprint", "blueprint-search"), default="blueprint")
    ap.add_argument("--menu", choices=("fc", "fcpa", "full"), default="full",
                    help="LBR bet menu. full = half-pot+pot+all-in (richest graded set, the strongest "
                         "standard config). fcpa = pot+all-in. fc = fold/call floor (NOT a best-responder).")
    ap.add_argument("--hands", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--range-cap", type=int, default=80)
    ap.add_argument("--search-iters", type=int, default=2000)
    ap.add_argument("--search-leaf", choices=("uniform", "gadget"), default="uniform")
    ap.add_argument("--blueprint-snapshot", required=True)
    ap.add_argument("--out", required=True, help="pooled summary JSON")
    ap.add_argument("--jsonl", default=None, help="per-hand JSONL dump (seed,idx,seat,reward,...)")
    args = ap.parse_args()

    act, model, label, value_only = _build(args)
    print(f"cert8 LBR: target={label} menu={args.menu} hands={args.hands} seed={args.seed} "
          f"value_only={value_only}", flush=True)

    _MENUS = {"fc": (), "fcpa": (lbr6._POT_BET_INDEX, lbr6.ALL_IN),
              "full": (lbr6._HALF_POT_INDEX, lbr6._POT_BET_INDEX, lbr6.ALL_IN)}
    cfg = lbr6.LBR6Config(hands=args.hands, seed=args.seed, paired=False, value_only_raises=value_only,
                          range_cap=args.range_cap)
    cfg.bet_candidates = _MENUS[args.menu]

    records: list[dict] = []
    t0 = time.time()
    res = lbr6.lbr6_exploitability(act, cfg, model_fn=model, hand_sink=records.append)
    dt = time.time() - t0

    # Per-hand JSONL: idx == hand index in this shard; seat = idx % n_seats (= offset from button).
    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for i, r in enumerate(records):
                f.write(json.dumps({
                    "seed": args.seed, "idx": i, "seat": i % cfg.n_seats,
                    "reward": r["reward"], "last_call_street": r.get("last_call_street", -1),
                    "end_street": r.get("end_street", -1), "folded": bool(r.get("folded", False)),
                }) + "\n")

    lo = res.bb_per_100 - 1.96 * res.stderr_bb_per_hand * 100.0
    hi = res.bb_per_100 + 1.96 * res.stderr_bb_per_hand * 100.0
    payload = {
        "target": args.target, "label": label, "menu": args.menu, "value_only_raises": value_only,
        "hands": res.hands, "seed": args.seed, "range_cap": args.range_cap,
        "estimator": "unpaired_raw", "bb_per_100": res.bb_per_100, "mbb_per_g": res.mbb_per_g,
        "ev_bb_per_hand": res.ev_bb_per_hand, "stderr_bb_per_hand": res.stderr_bb_per_hand,
        "ci95_bb_per_100": [lo, hi], "seconds": dt, "detail": res.detail,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"cert8 LBR {label}: {res.bb_per_100:+.2f} bb/100  CI95[{lo:+.1f},{hi:+.1f}]  "
          f"N={res.hands}  ({dt:.0f}s, {dt/max(res.hands,1):.2f}s/hand)  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
