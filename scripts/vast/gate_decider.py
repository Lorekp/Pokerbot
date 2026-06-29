#!/usr/bin/env python3
"""Decide whether the 7-max neural-CFR net has hit the LBR convergence gate.

Reads ONE tournament-eval-suite JSON (the output of
``scripts/tournament/eval_tournament_suite.py --policy net ...``) and prints a single-line
verdict + exit code so a shell supervisor (the vast.ai onstart loop / the babysitter) can
``stop training when the gate passes``.

The gate is the certified always-fold anchor comparison, NOT an absolute mbb/g (multiplayer +
ICM has no Nash-exploitability bound -- every repo note calls this a behavioral guardrail). Per
the RECON gate spec there are two tiers, both expressed against the SAME-CELL always-fold floor
(``anchor_chip_exploit_at_cell_bb_per_100``), measured at each cell's exact geometry:

  TIER 1 (minimum / hard stop-gate): on EVERY evaluated cell the net's
          ``chip_exploit_bb_per_100`` is below the same-cell always-fold floor, i.e.
          ``less_exploitable_than_always_fold == true``. "A best-responder extracts less from the
          net than from a pure folder."

  TIER 2 (quality target): tier 1 AND chip-LBR is a clear margin under the floor
          (chip_exploit <= margin * floor, default margin 0.5) AND the $-LBR ``verdict`` is not
          POSITIVE (ci95 lower bound <= 0) on the headline NORMAL cells.

This module is pure-stdlib so it runs on the rented GPU box with no repo imports. It is invoked
both on-box (by onstart.sh) and locally (by the watcher) with the same JSON contract.

CLI:
  python gate_decider.py --json artifacts/tournament_eval_net.json --tier 1
  python gate_decider.py --json artifacts/tournament_eval_net.json --tier 2 --margin 0.5 \
      --cells NORMAL:deep NORMAL:shallow NORMAL:bubble

Exit codes: 0 = GATE MET, 2 = not met, 3 = unusable JSON (no built cells / parse error).
The single stdout line is machine-greppable: starts with "GATE_MET" or "GATE_NOT_MET" or
"GATE_UNUSABLE".
"""
from __future__ import annotations

import argparse
import json
import sys

# Default headline cells the gate is decided on (a cheap, decision-sufficient subset; do NOT need
# the full 24-cell grid to decide convergence -- these three span deep / short / ICM-pressure).
DEFAULT_CELLS = ["NORMAL:deep", "NORMAL:shallow", "NORMAL:bubble"]


def _cell_result(suite: dict, cell_key: str) -> dict | None:
    """Return the E1_E2_lbr_icm.result block for a cell, or None if absent/not built."""
    cells = suite.get("cells", {})
    cell = cells.get(cell_key)
    if not isinstance(cell, dict):
        return None
    pillars = cell.get("pillars", {})
    e12 = pillars.get("E1_E2_lbr_icm", {})
    if e12.get("status") != "built":
        return None
    res = e12.get("result")
    return res if isinstance(res, dict) else None


def decide(suite: dict, *, tier: int, margin: float, cells: list[str]) -> dict:
    """Pure decision over a parsed suite JSON. Returns a verdict dict (also used by the watcher)."""
    rows: list[dict] = []
    evaluated = 0
    for ck in cells:
        res = _cell_result(suite, ck)
        if res is None:
            rows.append({"cell": ck, "present": False})
            continue
        evaluated += 1
        chip = res.get("chip_exploit_bb_per_100")
        floor = res.get("anchor_chip_exploit_at_cell_bb_per_100")
        lt_fold = bool(res.get("less_exploitable_than_always_fold"))
        verdict = res.get("verdict")
        ci95 = res.get("ci95") or [None, None]
        ci_lo = ci95[0] if isinstance(ci95, (list, tuple)) and len(ci95) == 2 else None
        # tier-1 condition for this cell
        t1_ok = lt_fold
        # tier-2 condition: clear margin under the floor + $-LBR CI does not exclude 0-positive.
        margin_ok = (
            chip is not None and floor is not None and floor > 0 and chip <= margin * floor
        )
        dollar_ok = (verdict != "POSITIVE") or (ci_lo is not None and ci_lo <= 0)
        rows.append({
            "cell": ck, "present": True,
            "chip": chip, "floor": floor, "less_than_fold": lt_fold,
            "verdict": verdict, "ci_lo": ci_lo,
            "t1_ok": t1_ok, "margin_ok": margin_ok, "dollar_ok": dollar_ok,
            "ratio": (chip / floor) if (chip is not None and floor) else None,
        })

    if evaluated == 0:
        return {"status": "UNUSABLE", "met": False, "evaluated": 0, "rows": rows,
                "reason": "no built E1_E2_lbr_icm cells in the requested subset"}

    present = [r for r in rows if r.get("present")]
    t1 = all(r["t1_ok"] for r in present)
    if tier <= 1:
        met = t1
    else:
        t2 = t1 and all(r["margin_ok"] and r["dollar_ok"] for r in present)
        met = t2
    return {"status": "MET" if met else "NOT_MET", "met": met, "tier": tier,
            "margin": margin, "evaluated": evaluated, "rows": rows}


def _num(v) -> str:
    """Format a possibly-None numeric for display (the net-free smoke leaves anchor=None)."""
    return f"{v:.0f}" if isinstance(v, (int, float)) else "?"


def _fmt_rows(rows: list[dict]) -> str:
    parts = []
    for r in rows:
        if not r.get("present"):
            parts.append(f"{r['cell']}=<absent>")
            continue
        ratio = r.get("ratio")
        ratio_s = f"{ratio:.2f}x" if ratio is not None else "?x"
        flag = "OK" if r["t1_ok"] else "HIGH"
        parts.append(
            f"{r['cell']}: chip={_num(r.get('chip'))} floor={_num(r.get('floor'))} ({ratio_s} {flag}) "
            f"$verdict={r['verdict']}"
        )
    return " | ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="7-max neural-CFR LBR gate decider")
    ap.add_argument("--json", required=True, help="tournament-eval-suite JSON (--policy net output)")
    ap.add_argument("--tier", type=int, default=1, choices=[1, 2],
                    help="1=hard stop-gate (less_exploitable_than_always_fold on every cell); "
                         "2=quality target (margin under floor + non-POSITIVE $-LBR)")
    ap.add_argument("--margin", type=float, default=0.5,
                    help="tier-2 only: chip-LBR must be <= margin * same-cell anchor floor")
    ap.add_argument("--cells", nargs="*", default=DEFAULT_CELLS,
                    help="cell subset to decide on (default NORMAL deep/shallow/bubble)")
    args = ap.parse_args(argv)

    try:
        with open(args.json, "r", encoding="utf-8") as fh:
            suite = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"GATE_UNUSABLE could not read/parse {args.json}: {exc}", flush=True)
        return 3

    out = decide(suite, tier=args.tier, margin=args.margin, cells=args.cells)
    if out["status"] == "UNUSABLE":
        print(f"GATE_UNUSABLE {out['reason']} (evaluated={out['evaluated']})", flush=True)
        return 3

    tag = "GATE_MET" if out["met"] else "GATE_NOT_MET"
    print(f"{tag} tier={out['tier']} evaluated={out['evaluated']} :: {_fmt_rows(out['rows'])}",
          flush=True)
    return 0 if out["met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
