#!/usr/bin/env python3
"""Apply the PRE-REGISTERED base-routing decision rule to a ``--mode basestrength`` SUMMARY.

The stated goal: *extract maximum value from weak players while protecting against strong players, with
the focus on value extraction.* The basestrength run measures, for the first time, each candidate
base's ABSOLUTE AIVAT bb/hand:

  * SFT net (``sft``)          — the data-trained, human-exploitative base (the current value vehicle).
  * blueprint+search (``bp``)  — the near-equilibrium CFR base (the candidate protection base).

vs three field classes:

  * VALUE (weak):   realistic_table, soft_table, weak_mixture, hs_station, hs_lag
  * PROTECT (strong): tough_table, blueprint, hs_tag

This script reads the pooled absolute EV and decides:

  1. VALUE base — the base used by default (weak/normal tables). Chosen to MAXIMIZE value vs the weak
     fields. The focus, so it wins ties.
  2. PROTECT base — whether the table-toughness router should swap to the blueprint base on confirmed
     regs. Enabled ONLY if the blueprint loses materially less than the value base vs the strong
     fields (otherwise swapping buys nothing and we just damp exploitation).

A "material" difference = point gap beyond the combined 95% interval (pooled SEs) AND > MIN_GAP_BB.
Output: a printed verdict + ``data/player_profiles/base_routing.json`` (the BaseRouter config the live
wiring reads). Pure ``decide(summary)`` so it is unit-testable; ``--selftest`` exercises it offline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

VALUE_CELLS = ("realistic_table", "soft_table", "weak_mixture", "hs_station", "hs_lag")
PROTECT_CELLS = ("tough_table", "blueprint", "hs_tag")
PRIMARY_VALUE = "realistic_table"   # the modal real table — the headline value metric
PRIMARY_PROTECT = "tough_table"     # the reg-heavy table — the headline protection metric
MIN_GAP_BB = 0.10                   # ignore base differences smaller than this (bb/hand)
OUT = Path("data/player_profiles/base_routing.json")


def _cell(bases: dict, base: str, cell: str) -> dict | None:
    return (bases.get(base) or {}).get(cell)


def _mean(bases: dict, base: str, cells) -> tuple[float, float] | None:
    """Mean base_bb (and pooled SE) over the available cells for one base. None if no data."""
    xs, ses = [], []
    for c in cells:
        d = _cell(bases, base, c)
        if d and "base_bb" in d:
            xs.append(float(d["base_bb"]))
            ses.append(float(d.get("base_se", 0.0)))
    if not xs:
        return None
    m = sum(xs) / len(xs)
    se = math.sqrt(sum(s * s for s in ses)) / len(xs)
    return m, se


def _materially_greater(a: tuple[float, float] | None, b: tuple[float, float] | None) -> bool:
    """a > b beyond the combined 95% interval AND by at least MIN_GAP_BB."""
    if a is None or b is None:
        return False
    gap = a[0] - b[0]
    ci = 1.96 * math.sqrt(a[1] ** 2 + b[1] ** 2)
    return gap > MIN_GAP_BB and gap > ci


def decide(summary: dict) -> dict:
    bases = summary.get("bases", {})
    sft_val = _mean(bases, "sft", VALUE_CELLS)
    bp_val = _mean(bases, "bp", VALUE_CELLS)
    sft_prot = _mean(bases, "sft", PROTECT_CELLS)
    bp_prot = _mean(bases, "bp", PROTECT_CELLS)

    # (1) VALUE base — maximize value vs weak; SFT (the focus / current vehicle) wins ties.
    if _materially_greater(bp_val, sft_val):
        value_base = "bp"
        value_reason = "blueprint extracts materially MORE from weak fields than the SFT net"
    else:
        value_base = "sft"
        value_reason = ("SFT net is >= blueprint vs weak fields (expected: the data-trained base is "
                        "exploitative; the blueprint is GTO/non-exploitative) - keep the value focus")

    # (2) PROTECT base — only swap to blueprint on tough tables if it loses materially less there.
    use_blueprint_router = _materially_greater(bp_prot, sft_prot)
    if use_blueprint_router:
        protect_reason = "blueprint loses materially less than the SFT base vs strong fields"
    else:
        protect_reason = ("blueprint is NOT materially stronger vs tough fields - protect by DAMPING "
                          "exploitation only (no base swap)")

    def _fmt(t):
        return None if t is None else {"mean_bb": round(t[0], 3), "se": round(t[1], 3)}

    return {
        "value_base": value_base,
        "value_reason": value_reason,
        "protect_base": "bp" if use_blueprint_router else value_base,
        "protect_reason": protect_reason,
        # BaseRouter runtime config: enable the router whenever we protect; only attach the blueprint
        # provider (base swap) when the data says it helps — else the router just damps exploit on regs.
        "router": {
            "enabled": True,
            "use_blueprint_provider": bool(use_blueprint_router),
            "tough_high": 0.5,
        },
        "evidence": {
            "value": {"sft": _fmt(sft_val), "bp": _fmt(bp_val)},
            "protect": {"sft": _fmt(sft_prot), "bp": _fmt(bp_prot)},
            "primary_value_cell": {
                "sft": _cell(bases, "sft", PRIMARY_VALUE),
                "bp": _cell(bases, "bp", PRIMARY_VALUE),
            },
            "primary_protect_cell": {
                "sft": _cell(bases, "sft", PRIMARY_PROTECT),
                "bp": _cell(bases, "bp", PRIMARY_PROTECT),
            },
        },
    }


def _print(verdict: dict) -> None:
    print("=== BASE-ROUTING DECISION (pre-registered rule) ===")
    print(f"VALUE base   : {verdict['value_base']}  -- {verdict['value_reason']}")
    print(f"PROTECT base : {verdict['protect_base']}  -- {verdict['protect_reason']}")
    print(f"router       : {json.dumps(verdict['router'])}")
    ev = verdict["evidence"]
    print("\nabsolute base bb/hand (pure base, exploit OFF):")
    print(f"  vs WEAK fields   sft={ev['value']['sft']}  bp={ev['value']['bp']}")
    print(f"  vs STRONG fields sft={ev['protect']['sft']}  bp={ev['protect']['bp']}")


_SELFTEST = {
    "bases": {
        # SFT: strong vs weak (the focus), weak vs strong. BP: weaker vs weak, stronger vs strong.
        "sft": {
            "realistic_table": {"base_bb": 4.0, "base_se": 0.3, "exploit_bb": 6.0, "uplift_bb": 2.0},
            "soft_table": {"base_bb": 7.0, "base_se": 0.3, "exploit_bb": 10.0, "uplift_bb": 3.0},
            "weak_mixture": {"base_bb": 3.0, "base_se": 0.3, "exploit_bb": 4.0, "uplift_bb": 1.0},
            "hs_station": {"base_bb": 5.0, "base_se": 0.3, "exploit_bb": 8.0, "uplift_bb": 3.0},
            "hs_lag": {"base_bb": 2.0, "base_se": 0.3, "exploit_bb": 3.0, "uplift_bb": 1.0},
            "tough_table": {"base_bb": -3.0, "base_se": 0.3, "exploit_bb": -3.1, "uplift_bb": -0.1},
            "blueprint": {"base_bb": -4.0, "base_se": 0.3, "exploit_bb": -4.0, "uplift_bb": 0.0},
            "hs_tag": {"base_bb": -1.0, "base_se": 0.3, "exploit_bb": -1.0, "uplift_bb": 0.0},
        },
        "bp": {
            "realistic_table": {"base_bb": 1.5, "base_se": 0.3, "exploit_bb": 1.6, "uplift_bb": 0.1},
            "soft_table": {"base_bb": 3.0, "base_se": 0.3, "exploit_bb": 3.1, "uplift_bb": 0.1},
            "weak_mixture": {"base_bb": 1.0, "base_se": 0.3, "exploit_bb": 1.0, "uplift_bb": 0.0},
            "hs_station": {"base_bb": 2.0, "base_se": 0.3, "exploit_bb": 2.1, "uplift_bb": 0.1},
            "hs_lag": {"base_bb": 0.5, "base_se": 0.3, "exploit_bb": 0.5, "uplift_bb": 0.0},
            "tough_table": {"base_bb": -0.5, "base_se": 0.3, "exploit_bb": -0.5, "uplift_bb": 0.0},
            "blueprint": {"base_bb": 0.0, "base_se": 0.3, "exploit_bb": 0.0, "uplift_bb": 0.0},
            "hs_tag": {"base_bb": 0.2, "base_se": 0.3, "exploit_bb": 0.2, "uplift_bb": 0.0},
        },
    }
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", nargs="?", help="path to the basestrength SUMMARY.json")
    ap.add_argument("--out", default=str(OUT), help="where to write the BaseRouter config")
    ap.add_argument("--selftest", action="store_true", help="run the rule on a synthetic summary")
    args = ap.parse_args()

    if args.selftest:
        summary = _SELFTEST
    elif args.summary:
        summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    else:
        ap.error("provide a SUMMARY.json path or --selftest")

    verdict = decide(summary)
    _print(verdict)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    if args.selftest:
        assert verdict["value_base"] == "sft", verdict
        assert verdict["protect_base"] == "bp" and verdict["router"]["use_blueprint_provider"], verdict
        print("SELFTEST OK: value=sft (focus), protect=bp (blueprint loses less vs strong)")


if __name__ == "__main__":
    main()
