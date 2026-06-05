"""Phase A — adjudicate the 4-max AoF FP push/fold policy and conditionally ship it.

Mirrors scripts/phase_a_fp_adjudicate.py (3-max) for the 4-player sampled path. The BR
and exploitability are the LEGAL (opponent-hands-marginalized) versions in
aof_cfr/exploit_mc.py (clairvoyance + chip-model bugs fixed). 4-max has no materializable
169^4 tensor, so equities come from combo-deal sampling with exact runout showdowns; the
Nash gap and per-infoset improvement are Monte-Carlo estimates (the spec allows an
approximate 4-max metric).

Gate (same spirit as 3-max): ship iff fixed-point stable AND max reach-relevant per-infoset
legal BR improvement < 0.30 bb (the indifference margin). NashConv recorded, not gated.

Run:  python scripts/phase_a_4max_adjudicate.py [--max-rounds N --br-samples M --gate-samples K]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.exploit_mc import estimate_nash_conv_mc, per_infoset_legal_improvement_mc
from aof_cfr.export import export_manifest, export_policy_npz
from aof_cfr.fp_4max import solve_4max_fp
from aof_cfr.infoset import decode_infoset_key, encode_infoset_key
from aof_cfr.solve import SolveResult
from aof_ranges import active_seats

MARGIN_BB = 0.30
RELEVANT_REACH_FRAC = 1e-3  # MC: ignore very-rarely-reached infosets (noise-dominated)
CFR_DIR = Path("data/aof/cfr_base")
REPORT_PATH = CFR_DIR / "phase_a_4max_adjudicate.json"
ART_4P = CFR_DIR / "cfr_base_4p.npz"
MANIFEST = CFR_DIR / "cfr_base.manifest.json"

# 4-max opening/aggression sanity spots (UTG first, then BTN, SB, BB). History tokens are
# the prior players' F/A actions in seat order.
CHART_SPOTS = [
    ("UTG", "-", "AA", 0.80, "push"),
    ("UTG", "-", "72o", 0.15, "fold"),
    ("BTN", "F", "AA", 0.80, "push"),        # UTG folded
    ("BTN", "F", "72o", 0.25, "fold"),
    ("SB", "FF", "AA", 0.80, "push"),        # UTG, BTN folded (SB vs BB)
    ("BB", "AFF", "AA", 0.80, "call"),       # UTG jammed, BTN+SB folded
    ("BB", "AFF", "72o", 0.20, "fold"),
]


def _policy_arrays(rows: dict[str, np.ndarray]) -> dict[str, float]:
    return {k: float(v[1]) for k, v in rows.items()}


def _chart_sanity(push: dict[str, float], stack_bb: float) -> dict:
    checks = []
    ok = True
    for seat, hist, hand, bound, kind in CHART_SPOTS:
        key = encode_infoset_key(seat, hand, hist, stack_bb)
        p = push.get(key)
        if p is None:
            checks.append({"spot": f"{seat}|{hist}|{hand}", "kind": kind, "p_push": None, "passed": None})
            continue
        passed = (p >= bound) if kind in ("push", "call") else (p <= bound)
        ok = ok and passed
        checks.append({"spot": f"{seat}|{hist}|{hand}", "kind": kind, "p_push": p, "bound": bound, "passed": passed})
    return {"checks": checks, "passed": ok}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rounds", type=int, default=40)
    ap.add_argument("--br-samples", type=int, default=6000)
    ap.add_argument("--gate-samples", type=int, default=12000)
    ap.add_argument("--stack-bb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-ship", action="store_true")
    args = ap.parse_args()

    seats = active_seats(4)
    print(f"4-max seats (acting order): {seats}")
    print(f"solving 4-max FP (max_rounds={args.max_rounds}, br_samples={args.br_samples}, seed={args.seed}) ...")
    t0 = time.monotonic()
    result = solve_4max_fp(
        max_rounds=args.max_rounds, br_samples=args.br_samples, seed=args.seed,
        stack_bb=args.stack_bb,
    )
    print(
        f"  rounds={result.rounds} converged={result.converged} "
        f"solve_gap={result.exploitability_bb * 1000:.1f} mbb/g method={result.br_method} "
        f"({time.monotonic() - t0:.0f}s)"
    )
    rows = result.policy_rows
    push = _policy_arrays(rows)

    # Final legal NashConv (higher samples than the per-round measure).
    print(f"measuring legal NashConv (gate_samples={args.gate_samples}) ...")
    nash_conv, imps, method = estimate_nash_conv_mc(
        4, rows, samples=args.gate_samples, seed=args.seed + 1, stack_bb=args.stack_bb,
        show_progress=True,
    )
    print(f"  legal NashConv={nash_conv * 1000:.1f} mbb/g per-seat={[round(x*1000,1) for x in imps]}")

    # Per-infoset legal improvement gate.
    print("computing per-infoset legal improvements ...")
    pi = per_infoset_legal_improvement_mc(4, rows, samples=args.gate_samples, seed=args.seed + 2, stack_bb=args.stack_bb)
    by_seat: dict[str, list] = {s: [] for s in seats}
    max_all = 0.0
    max_rel = 0.0
    worst_all = None
    worst_rel = None
    for key, d in pi.items():
        seat, hand, hist, _depth = decode_infoset_key(key)
        rec = {"key": key, "imp_bb": d["imp_bb"], "reach": d["reach"], "fp_push": push.get(key, 0.5)}
        by_seat.setdefault(seat, []).append(rec)
        if d["imp_bb"] > max_all:
            max_all = d["imp_bb"]; worst_all = rec
        if d["reach"] >= RELEVANT_REACH_FRAC and d["imp_bb"] > max_rel:
            max_rel = d["imp_bb"]; worst_rel = rec

    seat_tops = {}
    for s, recs in by_seat.items():
        recs.sort(key=lambda r: r["imp_bb"], reverse=True)
        seat_tops[s] = recs[:8]

    chart = _chart_sanity(push, args.stack_bb)
    stable = bool(result.converged)

    print(f"max per-infoset legal imp (all):      {max_all:.4f} bb at {worst_all}")
    print(f"max per-infoset legal imp (relevant): {max_rel:.4f} bb at {worst_rel}")
    print(f"chart sanity passed: {chart['passed']}")
    for c in chart["checks"]:
        print(f"   {c['spot']:<16} {c.get('kind'):<5} p_push={c.get('p_push')} -> {c.get('passed')}")

    br_consistent = max_rel < MARGIN_BB
    clean_pass = stable and br_consistent
    verdict = "PASS_clean" if clean_pass else ("HOLD_unstable" if br_consistent else "FAIL")
    ship = clean_pass and not args.no_ship

    report = {
        "reference": "internal_legal_br_equilibrium_mc",
        "margin_bb": MARGIN_BB,
        "solver": {"name": "fictitious_play_4max", "rounds": result.rounds,
                    "br_samples": args.br_samples, "gate_samples": args.gate_samples,
                    "seed": args.seed, "converged": result.converged},
        "nash_conv": {"metric": "legal_marginalized_mc", "total_mbb_g": nash_conv * 1000.0,
                       "per_seat_bb": dict(zip(seats, imps)),
                       "note": "MC estimate; recorded, not gated"},
        "br_consistency": {"max_imp_bb_all": max_all, "max_imp_bb_relevant": max_rel,
                            "worst_all": worst_all, "worst_relevant": worst_rel,
                            "relevant_reach_frac_floor": RELEVANT_REACH_FRAC,
                            "br_consistent": br_consistent},
        "stability": {"converged": result.converged},
        "chart_sanity": chart,
        "seat_top_improvements": seat_tops,
        "verdict": verdict,
        "ship": ship,
    }
    CFR_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nverdict: {verdict}   ship: {ship}\nreport -> {REPORT_PATH}")

    if not ship:
        print("\nNot shipping. See seat_top_improvements / chart_sanity in the report.")
        return 0 if verdict != "FAIL" else 1

    # ---- SHIP ----
    print("\nclean pass -> shipping FP policy to cfr_base_4p.npz")
    solve_result = SolveResult(
        num_players=4,
        solver_name="fictitious_play_4max",
        iterations=result.rounds,
        seed=args.seed,
        exploitability_bb_per_hand=nash_conv,
        exploitability_mbb_g=nash_conv * 1000.0,
        br_method=method,
        player_improvements=list(imps),
        policy_rows=rows,
        nash_disclaimer=(
            "N=4 general-sum: legal (opponent-hands-marginalized) MC best-response NashConv; "
            "no unique/low-gap Nash, recorded not gated. Shipped via BR-consistency: max "
            f"reach-relevant per-infoset legal improvement {max_rel:.4f} bb < {MARGIN_BB} bb."
        ),
    )
    export_policy_npz(ART_4P, solve_result)
    export_manifest(MANIFEST, [solve_result], merge_existing=True,
                    extra={"fp_4max": {"rounds": result.rounds, "br_samples": args.br_samples,
                                        "gate_samples": args.gate_samples, "seed": args.seed}})
    print(f"  wrote {ART_4P} ({len(rows)} infosets); updated {MANIFEST}")

    import aof_cfr.query as q
    q.load_cfr_base()
    for seat, hist, hand in [("UTG", "-", "AA"), ("UTG", "-", "72o"), ("BB", "AFF", "AA")]:
        got = q.base_strategy(4, seat, hist, hand, args.stack_bb)
        print(f"  base_strategy(4,{seat},{hist},{hand})={got:.4f}")
    print("\n4-max FP base shipped and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
