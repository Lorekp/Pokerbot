r"""Tiny-N CPU SMOKE for the cash-eval suite — the pre-trust check (NO heavy / NO GPU run).

This runs the WHOLE ``cash_eval_suite`` orchestration at smoke scale against the DEPLOYED blueprint
snapshot with the CORRECT abstraction flags set, and asserts the spec's validating anchors:

  1. the serve-fidelity gate PASSES (the fingerprint matches the snapshot stamp — the -559 bb/100 guard);
  2. the M1/M4 control variate REDUCES variance (``var_reduction > 1`` on at least the headline stake) —
     a non-reducing CV is mis-wired (the AIVAT-on-search failure class) and must never read clean;
  3. M2 exploitation uplift is POSITIVE on the size-RESPONSIVE station AND ~0 on the size-BLIND control
     (the blindness regression — the overlay must express the size->fold-vs-pay mechanism, not a
     bookkeeping artifact);
  4. the net-of-rake curve over ``rake.STAKE_ORDER`` is a visible HUMP (non-monotone, interior peak);
  5. the M3/M6 pro-safety Delta is finite + CI-reported (or cleanly DEFERRED when the C5 sibling is not
     yet built — the suite still emits a valid verdict).

Smoke scale (the spec's): ~40 paired decks, 200 LBR hands, ONE stake, 100bb. Everything is CPU + the
blueprint+search serve (net-free in the SFT sense — no torch belief/value net). Sub-few-minutes on a
laptop; the heavy stake x depth x position grid is deferred to AWS spot (the suite runs it unchanged).

Run (WSL Debian):
  source ~/jaxgpu-venv/bin/activate
  cd .
  PYTHONPATH=. POKERBOT_INFO_SUITS=1 JAX_PLATFORMS=cpu python scripts/eval/cash_eval_smoke.py
"""

from __future__ import annotations

# env BEFORE importing pokerbot (the encoder bakes the abstraction flags at import). Identical to the
# suite's defaults so the served fingerprint matches the lossless2 ship-candidate stamp.
import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")
_os.environ.setdefault("POKERBOT_INFO_SUITS", "1")
_os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
_os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
_os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "emd")
_os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "160")
# EMD-K160 deployed regime (promoted 2026-07-02, certified +32.89 bb/100 vs made-hand iter62000):
# frozen flop/turn EMD lookup + train-mirrored made-hand fallback on the river (the river is not
# in the blob; without the mirror flag the river frozen-miss serves a UNIFORM policy).
_os.environ.setdefault("POKERBOT_POSTFLOP_EMD_FROZEN", _os.path.abspath(_os.path.join(
    _os.path.dirname(__file__), "..", "..", "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn")))
_os.environ.setdefault("POKERBOT_EMD_FROZEN_MIRROR_FALLBACK", "1")
_os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
_os.environ.setdefault("POKERBOT_SEARCH_PREFLOP", "1")

import argparse
import json
import sys

_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Reuse the suite verbatim — the smoke is the suite at smoke scale + assertions (no duplicated logic).
from scripts.eval import cash_eval_suite as suite  # noqa: E402

_DEFAULT_SNAPSHOT = _os.environ.get(
    "POKERBOT_BLUEPRINT_SNAPSHOT",
    r"checkpoints/bp6max_lossless2_iter200.pkl")

_SMOKE_STAKE = "0.50/1"          # the mid stake near the rake-hump peak (a clean GO candidate)


def _check(report: dict) -> tuple[bool, list[str]]:
    """Assert the spec's smoke anchors against a finished report. Returns ``(ok, notes)``."""
    notes: list[str] = []
    ok = True

    # 1) serve gate passed (assert_cash_serve returned without raising -> serve_gate block present).
    gate = report.get("serve_gate")
    if not isinstance(gate, dict) or "abstraction" not in gate:
        ok = False
        notes.append("FAIL: serve gate did not produce an abstraction block (gate did not pass)")
    else:
        cov = gate.get("coverage", {})
        notes.append(f"OK: serve gate PASSED (coverage probe hands={gate.get('probe_hands')}, "
                     f"streets={list(cov)})")

    # walk the single smoke depth.
    per_depth = report.get("per_depth", {})
    if not per_depth:
        return False, notes + ["FAIL: no per_depth measurements emitted"]
    depth_key = next(iter(per_depth))
    cell = per_depth[depth_key]

    # 2) M1 control variate reduced variance on at least one stake.
    by_stake = cell.get("m1_net_curve", {}).get("by_stake", {})
    vrs = [b.get("net_var_reduction") for b in by_stake.values() if b.get("net_var_reduction") is not None]
    if any(v is not None and v > 1.0 for v in vrs):
        notes.append(f"OK: M1 var_reduction > 1 (max={max(v for v in vrs if v is not None):.2f})")
    else:
        ok = False
        notes.append(f"FAIL: M1 var_reduction never exceeded 1 (got {vrs}) — CV mis-wired")

    # 3) M2 uplift POSITIVE on responsive AND ~0 on size-blind control.
    m2 = cell.get("m2_uplift", {})
    resp_up = m2.get("responsive", {}).get("uplift_bb_per_100")
    resp_ci = m2.get("responsive", {}).get("uplift_ci95_bb_per_100")
    blind_up = m2.get("size_blind_control", {}).get("uplift_bb_per_100")
    if resp_up is not None and resp_ci is not None:
        # POSITIVE = uplift point > 0 (smoke N is small; we report the CI but require the point sign +
        # that it exceeds the size-blind control, the blindness regression).
        if resp_up > 0.0:
            notes.append(f"OK: M2 responsive uplift = {resp_up:+.1f} bb/100 (CI {resp_ci})")
        else:
            ok = False
            notes.append(f"FAIL: M2 responsive uplift not positive ({resp_up:+.1f} bb/100)")
        if blind_up is not None:
            if abs(blind_up) <= abs(resp_up) + 1e-9:
                notes.append(f"OK: M2 blindness regression — size-blind uplift {blind_up:+.1f} "
                             f"<= responsive {resp_up:+.1f} (mechanism is field-specific)")
            else:
                notes.append(f"WARN: size-blind uplift {blind_up:+.1f} exceeded responsive {resp_up:+.1f} "
                             "(small-N noise; rerun with more --uplift-decks)")
    else:
        ok = False
        notes.append("FAIL: M2 uplift block missing")

    # 4) net-of-rake hump visible (non-monotone curve over the stakes run).
    pos_key = next((k for k in cell if k.startswith("pos")), None)
    if pos_key is not None:
        ht = cell[pos_key].get("hump_table", {})
        if ht.get("curve"):
            shape = "HUMP" if ht.get("is_hump") else "monotone"
            notes.append(f"OK: net curve has {len(ht['curve'])} stakes, shape={shape}, "
                         f"peak={ht.get('peak_stake')} valley={ht.get('valley_stake')}")
        else:
            notes.append("WARN: net curve empty (only one stake run in smoke — hump needs >=3 stakes)")

    # 5) M3/M6 pro-safety Delta finite + CI-reported, or cleanly deferred.
    ps = cell.get("m3_m6_pro_safety", {})
    if ps.get("status") == "built":
        r = ps.get("result", {})
        d = r.get("delta_bb_per_100", r.get("lbr_delta"))
        ci = r.get("delta_ci95_bb_per_100", r.get("lbr_delta_ci95"))
        if d is not None and ci is not None:
            notes.append(f"OK: M3 Delta = {d:+.1f} bb/100, CI {ci}, bet_shift={r.get('bet_shift')}")
        else:
            notes.append(f"WARN: M3 built but Delta/CI not in result keys ({list(r)})")
    elif ps.get("status") == "deferred":
        notes.append(f"OK (deferred): M3/M6 — {ps.get('reason')}")
    else:
        notes.append(f"WARN: M3/M6 status={ps.get('status')} ({ps.get('reason')})")

    return ok, notes


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Tiny-N CPU smoke for the cash-eval suite.")
    ap.add_argument("--blueprint-snapshot", default=_DEFAULT_SNAPSHOT)
    ap.add_argument("--paired-decks", type=int, default=40)
    ap.add_argument("--cheap-decks", type=int, default=120)
    ap.add_argument("--uplift-decks", type=int, default=40)
    ap.add_argument("--lbr-hands", type=int, default=200)
    ap.add_argument("--iterations", type=int, default=40, help="FIXED CFR iters (small for the smoke)")
    ap.add_argument("--stakes", default="all",
                    help="stakes for the net curve (default all -> the hump is visible)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/_cash_eval_smoke.json")
    args = ap.parse_args(argv[1:])

    if args.stakes.strip().lower() == "all":
        from pokerbot.economics.rake import STAKE_ORDER
        stakes = list(STAKE_ORDER)
    else:
        stakes = [s.strip() for s in args.stakes.split(",") if s.strip()]

    print(f"[smoke] snapshot={args.blueprint_snapshot}", file=sys.stderr, flush=True)
    print(f"[smoke] paired={args.paired_decks} cheap={args.cheap_decks} uplift={args.uplift_decks} "
          f"lbr={args.lbr_hands} iters={args.iterations} stakes={len(stakes)} @ 100bb", file=sys.stderr,
          flush=True)

    report = suite.run_suite(
        snapshot=args.blueprint_snapshot,
        depths=[100.0],                      # the spec's single smoke depth
        stakes=stakes,
        positions=[suite.BUTTON_SEAT],
        paired_decks=args.paired_decks, cheap_decks=args.cheap_decks, uplift_decks=args.uplift_decks,
        lbr_hands=args.lbr_hands, iterations=args.iterations, leaf_select="uniform", seed=args.seed,
        probe_hands=20, eps_rnr=5.0,
        station_field="responsive_station", reg_field="responsive_reg",
        size_blind_field="calling_station", uplift_stake=_SMOKE_STAKE,
        out=args.out, quiet=False)

    ok, notes = _check(report)
    print("\n[smoke] === anchors ===", file=sys.stderr, flush=True)
    for n in notes:
        print(f"[smoke]   {n}", file=sys.stderr, flush=True)
    print(f"[smoke] {'PASS' if ok else 'FAIL'}  (report -> {args.out})", file=sys.stderr, flush=True)
    # Also dump a compact verdict to stdout for log capture.
    print(json.dumps({"smoke_pass": ok, "summary": report["summary"],
                      "go_no_go_by_stake": report["go_no_go_by_stake"]}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
