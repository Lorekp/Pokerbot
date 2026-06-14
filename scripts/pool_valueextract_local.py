#!/usr/bin/env python3
"""Pool the Phase-2 value-extraction gauntlet results locally from a dir of ve_*.json cells.

Works on a PARTIAL / in-progress run. Each cell (written by scripts/value_extraction_gauntlet.py for one
field + one seed) has {"fields": {<field>: {"off":..., "on":..., "value":...}}}, named ve_<field>_s<seed>.json.
Pools off/on per field across seeds and prints the verdict: does turning the UNOPENED value-bet exploit ON
extract value vs WEAK (station-heavy) fields without harming TOUGH (lag/tag) ones?

  python scripts/pool_valueextract_local.py runs\\cloud_ve1
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

WEAK = ("irc_soft", "irc_station", "irc_mixed")
TOUGH = ("irc_tough", "irc_lag")
_FIELD_RE = re.compile(r"^ve_(.+)_s\d+\.json$")


def _pool(xs):
    k = len(xs)
    if k == 0:
        return 0.0, 0.0, 0
    m = sum(xs) / k
    se = (math.sqrt(sum((x - m) ** 2 for x in xs) / (k - 1)) / math.sqrt(k)) if k > 1 else 0.0
    return m, se, k


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/cloud_ve1")
    files = sorted(d.glob("ve_*.json"))
    if not files:
        raise SystemExit(f"no ve_*.json in {d}")

    per_field: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"off": [], "on": []})
    for f in files:
        m = _FIELD_RE.match(f.name)
        if not m:
            continue
        field = m.group(1)
        try:
            fd = (json.loads(f.read_text(encoding="utf-8")).get("fields") or {}).get(field)
        except (ValueError, OSError):
            continue
        if fd:
            per_field[field]["off"].append(float(fd["off"]))
            per_field[field]["on"].append(float(fd["on"]))

    print(f"{'field':12s} {'base bb/100':>14s} {'+expl bb/100':>14s} {'value':>9s} {'k':>3s}")
    weak_vals, tough_vals = [], []
    order = WEAK + TOUGH + tuple(f for f in per_field if f not in WEAK + TOUGH)
    for field in order:
        d2 = per_field.get(field)
        if not d2:
            continue
        om, ose, k = _pool(d2["off"])
        nm, nse, _ = _pool(d2["on"])
        tag = "weak" if field in WEAK else ("tough" if field in TOUGH else "")
        print(f"{field:12s} {om:>+9.2f}+-{ose:>4.2f} {nm:>+9.2f}+-{nse:>4.2f} {nm - om:>+9.2f} {k:>3d}  {tag}")
        (weak_vals if field in WEAK else tough_vals if field in TOUGH else []).append(nm - om)

    if weak_vals:
        print(f"\nMEAN value extracted vs WEAK = {sum(weak_vals) / len(weak_vals):+.2f} bb/100 "
              f"({len(weak_vals)} fields)")
    if tough_vals:
        worst = min(tough_vals)
        ok = worst >= -0.05
        print(f"WORST harm vs TOUGH          = {worst:+.2f} bb/100 "
              f"({'DO-NO-HARM OK -> flip unopened ON (gated)' if ok else 'HARMS -> keep unopened OFF'})")


if __name__ == "__main__":
    main()
