"""Phase A — adjudicate the 3-max FP push/fold policy and conditionally ship it.

Reference = the project's OWN validated BR machinery (internal BR-equilibrium), not a
remembered/intuition chart. The discriminator is a per-hand BR-improvement decomposition
of ``nash_gap_3max`` (the same probe that resolved the 1547 investigation): for every
(seat, history, hand) we measure the bb-EV an opponent gains from FP playing its current
frequency instead of the pure best response. If FP is BR-consistent within the project's
~0.30 bb indifference margin AND a stable fixed point, the FP policy is shipped to
``cfr_base_3p.npz`` (manifest solver=fictitious_play_3max). Otherwise nothing is written
and the offending seats/hands are reported.

The aggregate Nash gap (~1800 mbb/g) is RECORDED, not gated: n>=3 general-sum has no
unique/low-gap Nash to reach.

Run:  python scripts/phase_a_fp_adjudicate.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.export import export_manifest, export_policy_npz
from aof_cfr.fp_br import (
    StabilityTracker,
    best_responses_3max_legal,
    load_3max_payoffs,
    nash_gap_3max,  # clairvoyant, recorded for comparison only
    nash_gap_3max_legal,
    per_hand_legal_improvement_3max,
)
from aof_cfr.infoset import encode_infoset_key
from aof_cfr.solve import SolveResult
from aof_cfr.trio_payoff_tensor import DEFAULT_TENSOR, load_trio_tensor
from pokerbot.solver.trio_preflop_fp import solve_3max_fp, trio_fp_policy_rows

TRUSTED_SHA = "babe82516019cb9fada1f64f61d5d80bad00f98ffbc414dc0c53ebf2d4dca3ed"
MARGIN_BB = 0.30  # invariant 4: ~0.30 bb indifference margin (do not deviate for less)
TENSOR_BPC = 192  # boards/combo of the trusted tensor (phase_a_full_tensor_validate.json)
TENSOR_SEED_SCHEME = "combo_sha256_v1"

CFR_DIR = Path("data/aof/cfr_base")
REPORT_PATH = CFR_DIR / "phase_a_fp_adjudicate.json"
ART_3P = CFR_DIR / "cfr_base_3p.npz"
MANIFEST = CFR_DIR / "cfr_base.manifest.json"

# Reconciliation / stability thresholds (mirror StabilityTracker semantics).
RECON_TOL = 1e-8
GAP_SPREAD_TOL = 0.5  # mbb/g, tail window
POLICY_DELTA_TOL = 1e-3
TAIL_WINDOW = 5
RELEVANT_REACH_FRAC = 1e-4  # a hand is "strategically relevant" above this reach share

# (seat label, history key, frequency-array selector) for ladder/probe iteration.
SEATS = (
    ("BTN", "-", ("btn", None)),
    ("SB", "F", ("sb", 0)),
    ("SB", "A", ("sb", 1)),
    ("BB", "FA", ("bb", 0)),
    ("BB", "AF", ("bb", 1)),
    ("BB", "AA", ("bb", 2)),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sel(arr: np.ndarray, hist_idx) -> np.ndarray:
    """Select the (n,) row for a seat: BTN arrays are 1-D, SB/BB are (h, n)."""
    return arr if hist_idx is None else arr[hist_idx]


def _policy_arrays_from_artifact(path: Path, labels: list[str]):
    """Reconstruct (btn (n,), sb (2,n), bb (3,n)) push freqs from a cfr_base_Np.npz."""
    data = np.load(path, allow_pickle=True)
    keys = [str(k) for k in data["infoset_keys"]]
    probs = np.asarray(data["action_probs"], dtype=np.float64)
    idx = {k: i for i, k in enumerate(keys)}
    n = len(labels)

    def p(pos, lab, hist):
        return float(probs[idx[encode_infoset_key(pos, lab, hist)]][1])

    btn = np.array([p("BTN", lab, "-") for lab in labels], dtype=np.float64)
    sb = np.zeros((2, n))
    bb = np.zeros((3, n))
    for bi, lab in enumerate(labels):
        sb[0, bi] = p("SB", lab, "F")
        sb[1, bi] = p("SB", lab, "A")
        bb[0, bi] = p("BB", lab, "FA")
        bb[1, bi] = p("BB", lab, "AF")
        bb[2, bi] = p("BB", lab, "AA")
    return btn, sb, bb


def _cfr_plus_relook(st, deal_w, labels):
    """Re-score the existing (stale) cfr_base_3p policy under both metrics.

    Run BEFORE any ship (ship overwrites the artifact). Answers whether CFR+'s sharper
    thresholds were actually closer to legally correct than FP's clairvoyant-graded ladder.
    """
    if not ART_3P.is_file():
        return {"available": False, "reason": "cfr_base_3p.npz missing"}
    try:
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        t3 = next((t for t in man.get("tables", []) if t["num_players"] == 3), {})
        solver = t3.get("solver", "unknown")
    except Exception:
        solver = "unknown"
    cb, cs, cbb = _policy_arrays_from_artifact(ART_3P, labels)
    legal_total, legal_imps = nash_gap_3max_legal(st, deal_w, cb, cs, cbb)
    clair_total, clair_imps = nash_gap_3max(st, deal_w, cb, cs, cbb)
    return {
        "available": True,
        "artifact_solver": solver,
        "legal_total_mbb_g": legal_total * 1000.0,
        "legal_per_seat_bb": {"btn": legal_imps[0], "sb": legal_imps[1], "bb": legal_imps[2]},
        "clairvoyant_total_mbb_g": clair_total * 1000.0,
        "premium_thresholds": {
            "BTN_AA": float(cb[labels.index("AA")]),
            "BTN_77": float(cb[labels.index("77")]),
            "BB_AA_call_99": float(cbb[2, labels.index("99")]),
            "BB_AA_call_TT": float(cbb[2, labels.index("TT")]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-ship",
        action="store_true",
        help="run adjudication only; never overwrite cfr_base_3p.npz even on a clean pass",
    )
    args = parser.parse_args()

    tensor_path = Path(DEFAULT_TENSOR)
    if not tensor_path.is_file():
        raise FileNotFoundError(f"Missing trusted tensor {tensor_path}")

    sha = _sha256(tensor_path)
    print(f"tensor: {tensor_path}\n  sha256={sha}")
    if sha != TRUSTED_SHA:
        raise SystemExit(
            f"REFUSE: tensor sha {sha} != trusted {TRUSTED_SHA}. "
            "Adjudication must run on the validated tensor."
        )
    print("  sha matches trusted tensor (locked).")

    shares, pair_eq, deal_w_raw = load_trio_tensor(tensor_path)
    n = int(shares.shape[0])
    labels = list(LABELS_169[:n])

    # Solve FP pinned to the validated ground-truth config. policy_tol=0.0 disables the
    # plateau early-exit so we run the full max_rounds (shipped policy == ground truth),
    # while gap_floor_mbb_g=None drops the unjustified 640 mbb/g floor. The tracker
    # records the full gap / policy-delta trajectory for the stability check.
    tracker = StabilityTracker(gap_floor_mbb_g=None, policy_tol=0.0)
    print(
        f"\nsolving FP (max_rounds={args.max_rounds}, damping={args.damping}, "
        f"seed={args.seed}) ..."
    )
    result = solve_3max_fp(
        max_rounds=args.max_rounds,
        damping=args.damping,
        seed=args.seed,
        shares=shares,
        pair_eq=pair_eq,
        deal_w=deal_w_raw,
        stability=tracker,
    )
    btn, sb, bb = result.btn_push_freq, result.sb_push_freq, result.bb_push_freq

    # Reconstruct the exact (st, normalized deal_w) the solver used for the probes.
    st, deal_w = load_3max_payoffs(shares, pair_eq, deal_w_raw)

    # LEGAL (marginalized) metric is the gate. Clairvoyant gap recorded for comparison.
    total_gap, imps = nash_gap_3max_legal(st, deal_w, btn, sb, bb)
    clair_gap, clair_imps = nash_gap_3max(st, deal_w, btn, sb, bb)
    ph = per_hand_legal_improvement_3max(st, deal_w, btn, sb, bb)
    br_btn, br_sb, br_bb = best_responses_3max_legal(st, deal_w, btn, sb, bb)

    # ---- Drift guard: per-hand decomposition must reconcile to nash_gap_3max ----
    recon = {
        "btn": (float(np.sum(ph["btn"] * ph["btn_w"])), imps[0]),
        "sb": (float(np.sum(ph["sb"] * ph["sb_w"])), imps[1]),
        "bb": (float(np.sum(ph["bb"] * ph["bb_w"])), imps[2]),
    }
    recon_ok = True
    for seat, (got, want) in recon.items():
        d = abs(got - want)
        ok = d < RECON_TOL
        recon_ok = recon_ok and ok
        print(f"  reconcile {seat}: per-hand sum={got:.12f}  nash_gap={want:.12f}  absdiff={d:.2e}  {'OK' if ok else 'FAIL'}")
    if not recon_ok:
        raise SystemExit("REFUSE: per-hand decomposition does not reconcile to nash_gap_3max.")

    # ---- Stability (fixed-point) check from the recorded trajectory ----
    gaps = list(tracker._gaps)  # mbb/g per round
    deltas = list(tracker._btn_delta)  # |Δ BTN policy| per round (from round 2)
    tail_gaps = gaps[-TAIL_WINDOW:]
    tail_deltas = deltas[-TAIL_WINDOW:] if deltas else [float("inf")]
    gap_spread = float(max(tail_gaps) - min(tail_gaps))
    max_tail_delta = float(max(tail_deltas))
    stable = gap_spread < GAP_SPREAD_TOL and max_tail_delta < POLICY_DELTA_TOL
    print(
        f"\nstability: tail_gap_spread={gap_spread:.4f} mbb/g (<{GAP_SPREAD_TOL}) "
        f"max_tail_policy_delta={max_tail_delta:.2e} (<{POLICY_DELTA_TOL}) "
        f"rounds={result.rounds} -> {'STABLE' if stable else 'NOT STABLE'}"
    )

    # ---- BR-consistency: per-hand improvement vs the indifference margin ----
    seat_reports = []
    max_imp_all = 0.0
    max_imp_relevant = 0.0
    worst_all = None
    worst_relevant = None
    for seat_label, hist, (arr_name, hist_idx) in SEATS:
        imp_v = _sel(ph[arr_name], hist_idx)
        w_v = _sel(ph[arr_name + "_w"], hist_idx)
        fp_v = _sel({"btn": btn, "sb": sb, "bb": bb}[arr_name], hist_idx)
        br_v = _sel({"btn": br_btn, "sb": br_sb, "bb": br_bb}[arr_name], hist_idx)
        total_w = float(np.sum(w_v))
        reach_frac = w_v / total_w if total_w > 0 else np.zeros_like(w_v)

        order = np.argsort(imp_v)[::-1]
        top = []
        for bi in order[:10]:
            top.append(
                {
                    "hand": labels[bi],
                    "imp_bb": float(imp_v[bi]),
                    "fp_push": float(fp_v[bi]),
                    "br_push": float(br_v[bi]),
                    "reach_frac": float(reach_frac[bi]),
                    "direction_disagree": bool((fp_v[bi] - 0.5) * (br_v[bi] - 0.5) < 0),
                }
            )
        seat_max = float(np.max(imp_v)) if imp_v.size else 0.0
        seat_max_relevant = float(np.max(np.where(reach_frac >= RELEVANT_REACH_FRAC, imp_v, 0.0)))
        n_over_margin = int(np.sum(imp_v >= MARGIN_BB))
        n_over_margin_relevant = int(np.sum((imp_v >= MARGIN_BB) & (reach_frac >= RELEVANT_REACH_FRAC)))

        if seat_max > max_imp_all:
            max_imp_all = seat_max
            worst_all = (seat_label, hist, labels[int(order[0])], seat_max)
        if seat_max_relevant > max_imp_relevant:
            max_imp_relevant = seat_max_relevant
            ri = int(np.argmax(np.where(reach_frac >= RELEVANT_REACH_FRAC, imp_v, 0.0)))
            worst_relevant = (seat_label, hist, labels[ri], seat_max_relevant)

        seat_reports.append(
            {
                "seat": seat_label,
                "history": hist,
                "max_imp_bb": seat_max,
                "max_imp_bb_relevant": seat_max_relevant,
                "n_hands_over_margin": n_over_margin,
                "n_hands_over_margin_relevant": n_over_margin_relevant,
                "top_hands_by_imp": top,
            }
        )

    print(f"\nLEGAL per-seat improvements (bb): BTN={imps[0]:.4f} SB={imps[1]:.4f} BB={imps[2]:.4f}")
    print(f"LEGAL total gap:       {total_gap * 1000:.2f} mbb/g (recorded, not gated)")
    print(f"clairvoyant total gap: {clair_gap * 1000:.2f} mbb/g (diagnostic; over-states ~Jensen)")
    print(f"max per-hand LEGAL BR improvement (all hands):       {max_imp_all:.4f} bb  at {worst_all}")
    print(f"max per-hand LEGAL BR improvement (reach-relevant):  {max_imp_relevant:.4f} bb  at {worst_relevant}")

    # CFR+ re-look under both metrics (BEFORE any ship overwrites cfr_base_3p.npz).
    cfr_relook = _cfr_plus_relook(st, deal_w, labels)
    if cfr_relook.get("available"):
        print(
            f"\nCFR+ re-look ({cfr_relook['artifact_solver']}): legal {cfr_relook['legal_total_mbb_g']:.1f} mbb/g "
            f"vs FP legal {total_gap * 1000:.1f} | clairvoyant {cfr_relook['clairvoyant_total_mbb_g']:.1f} "
            f"| BB|AA call 99={cfr_relook['premium_thresholds']['BB_AA_call_99']:.3f} TT={cfr_relook['premium_thresholds']['BB_AA_call_TT']:.3f}"
        )

    br_consistent_all = max_imp_all < MARGIN_BB
    br_consistent_relevant = max_imp_relevant < MARGIN_BB

    # Clean pass -> auto-ship. Relevant-only pass (failures confined to dust hands) is a
    # conservative HOLD for user judgment, not an auto-ship.
    clean_pass = stable and recon_ok and br_consistent_all
    if clean_pass:
        verdict = "PASS_clean"
    elif stable and recon_ok and br_consistent_relevant:
        verdict = "HOLD_dust_only"  # only negligible-reach hands exceed the margin
    else:
        verdict = "FAIL"
    ship = clean_pass and not args.no_ship

    report = {
        "tensor_path": str(tensor_path),
        "tensor_sha256": sha,
        "trusted_sha256": TRUSTED_SHA,
        "reference": "internal_br_equilibrium",
        "margin_bb": MARGIN_BB,
        "solver": {
            "name": "fictitious_play_3max",
            "max_rounds": args.max_rounds,
            "rounds_run": result.rounds,
            "damping": args.damping,
            "seed": args.seed,
        },
        "nash_gap": {
            "metric": "legal_marginalized",
            "total_bb": total_gap,
            "total_mbb_g": total_gap * 1000.0,
            "per_seat_bb": {"btn": imps[0], "sb": imps[1], "bb": imps[2]},
            "clairvoyant_total_mbb_g": clair_gap * 1000.0,
            "clairvoyant_per_seat_bb": {"btn": clair_imps[0], "sb": clair_imps[1], "bb": clair_imps[2]},
            "trivial_baselines_mbb_g": result.trivial_baselines_mbb_g,
            "note": "legal gap recorded, not gated; n>=3 has no unique/low-gap Nash. clairvoyant is diagnostic (per-combo max, over-states by the Jensen gap).",
        },
        "cfr_plus_relook": cfr_relook,
        "reconcile_per_hand_vs_nash_gap": {
            k: {"per_hand_sum": v[0], "nash_gap": v[1], "abs_diff": abs(v[0] - v[1])}
            for k, v in recon.items()
        },
        "reconcile_ok": recon_ok,
        "stability": {
            "tail_gap_spread_mbb_g": gap_spread,
            "max_tail_policy_delta": max_tail_delta,
            "gap_spread_tol": GAP_SPREAD_TOL,
            "policy_delta_tol": POLICY_DELTA_TOL,
            "tail_window": TAIL_WINDOW,
            "stable": stable,
        },
        "br_consistency": {
            "max_imp_bb_all": max_imp_all,
            "max_imp_bb_relevant": max_imp_relevant,
            "worst_all": list(worst_all) if worst_all else None,
            "worst_relevant": list(worst_relevant) if worst_relevant else None,
            "relevant_reach_frac_floor": RELEVANT_REACH_FRAC,
            "br_consistent_all": br_consistent_all,
            "br_consistent_relevant": br_consistent_relevant,
        },
        "seats": seat_reports,
        "verdict": verdict,
        "ship": ship,
    }
    CFR_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nverdict: {verdict}   ship: {ship}")
    print(f"report -> {REPORT_PATH}")

    if not ship:
        if verdict == "HOLD_dust_only":
            print(
                "\nHOLD: stable + BR-consistent except on negligible-reach hands. Not "
                "auto-shipping; review br_consistency.worst_all in the report."
            )
        elif verdict == "FAIL":
            print(
                "\nFAIL: not shipping. Offending seats/hands above (per-seat "
                "top_hands_by_imp). If failures are concentrated -> structural; if thin "
                "across many marginal hands -> try more rounds / damping."
            )
        return 0 if verdict != "FAIL" else 1

    # ---------------- SHIP ----------------
    print("\nclean pass -> shipping FP policy to cfr_base_3p.npz")
    rows = trio_fp_policy_rows(result)
    solve_result = SolveResult(
        num_players=3,
        solver_name="fictitious_play_3max",
        iterations=result.rounds,
        seed=args.seed,
        exploitability_bb_per_hand=result.exploitability_bb,
        exploitability_mbb_g=result.exploitability_bb * 1000.0,
        br_method="exact_legal_marginalized",
        player_improvements=list(result.player_improvements),
        policy_rows=rows,
        nash_disclaimer=(
            "N=3 general-sum: exploitability is the exact LEGAL (opponent-hands "
            "marginalized) tabular BR gain over the bucket deal; multiplayer has no "
            "unique/low-gap Nash, so the gap is recorded, not gated. Shipped via "
            "BR-consistency: max per-hand legal BR improvement "
            f"{max_imp_all:.4f} bb < {MARGIN_BB} bb indifference margin, at a stable fixed "
            f"point (tail gap spread {gap_spread:.3f} mbb/g)."
        ),
    )
    export_policy_npz(ART_3P, solve_result)
    export_manifest(
        MANIFEST,
        [solve_result],
        merge_existing=True,
        extra={
            "tensor_provenance": {
                "tensor_sha256": sha,
                "boards_per_combo": TENSOR_BPC,
                "seed_scheme": TENSOR_SEED_SCHEME,
                "fp_seed": args.seed,
                "fp_rounds": result.rounds,
                "fp_damping": args.damping,
            }
        },
    )
    print(f"  wrote {ART_3P} ({len(rows)} infosets)")
    print(f"  updated {MANIFEST}")

    # ---------------- Runtime verification ----------------
    print("\nruntime verification (fresh load):")
    import aof_cfr.query as q
    from aof_cfr.infoset import infoset_key_spec_hash

    q.load_cfr_base()
    checks = [
        (3, "BTN", "-", "AA"),
        (3, "BTN", "-", "72o"),
        (3, "SB", "F", "72o"),
        (3, "BB", "FA", "72o"),
        (3, "SB", "A", "AA"),
    ]
    for np_, pos, hist, hand in checks:
        bi = labels.index(hand)
        if pos == "BTN":
            expect = float(btn[bi])
        elif pos == "SB":
            expect = float(sb[0 if hist == "F" else 1, bi])
        else:
            expect = float(bb[{"FA": 0, "AF": 1, "AA": 2}[hist], bi])
        got = q.base_strategy(np_, pos, hist, hand)
        ok = abs(got - expect) < 1e-4
        print(f"  base_strategy(3,{pos},{hist},{hand})={got:.4f}  fp={expect:.4f}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit("REFUSE: runtime value does not match shipped FP policy.")
    probs, keys, _ = q._LOADED[3]
    print(f"  3p infosets loaded: {len(keys)}")
    spec = infoset_key_spec_hash()
    print(f"  infoset_key_spec_hash: {spec} (expect 81f2a6a86d2cc429)")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    t3 = next(t for t in man["tables"] if t["num_players"] == 3)
    print(f"  manifest 3p solver: {t3['solver']}  exploitability_mbb_g: {t3['exploitability_mbb_g']:.2f}")
    print("\n3-max FP base shipped and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
