#!/usr/bin/env python3
"""Pool absolute base-strength results locally from a directory of bs_*.json result files.

Works on a PARTIAL / in-progress run (just pools whatever cells are present), so you can read the
coverage-fixed uniform base strength without waiting for the slow gadget tail or the cloud SUMMARY.

Each result JSON (written by scripts/eval_exploitation_battery.py) has:
  {"results": {<cell>: {"off_aivat": {"adj_mean_bb": ...}, "on_aivat": {...}, ...}}, "bp_coverage": {...}}
and is named  bs_<basetag>_<cell>_s<seed>.json  (basetag e.g. bp_i10_uniform / bp_i50_gadget / sft).

Usage:
  python scripts/pool_basestrength_local.py runs\\cloud_bs_sweep1
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

WEAK = ("realistic_table", "soft_table", "weak_mixture", "hs_station", "hs_lag")
STRONG = ("tough_table", "blueprint", "hs_tag")
_SEED_RE = re.compile(r"_s(\d+)\.json$")


def _pool(xs: list[float]) -> tuple[float, float, int]:
    k = len(xs)
    if k == 0:
        return 0.0, 0.0, 0
    m = sum(xs) / k
    se = (math.sqrt(sum((x - m) ** 2 for x in xs) / (k - 1)) / math.sqrt(k)) if k > 1 else 0.0
    return m, se, k


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/cloud_bs_sweep1")
    files = sorted(d.glob("bs_*.json"))
    if not files:
        raise SystemExit(f"no bs_*.json in {d}")

    # (basetag, cell) -> {"off": [...], "on": [...]}; basetag -> per-street [hits, misses]
    cells: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: {"off": [], "on": []})
    cov: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    for f in files:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        results = doc.get("results") or {}
        sm = _SEED_RE.search(f.name)
        for cell, r in results.items():
            if not (isinstance(r.get("off_aivat"), dict) and isinstance(r.get("on_aivat"), dict)):
                continue
            # basetag = filename between 'bs_' and '_<cell>_s<seed>.json'
            suffix = f"_{cell}_s{sm.group(1)}.json" if sm else f"_{cell}.json"
            base = f.name[len("bs_"):].removesuffix(suffix)
            cells[(base, cell)]["off"].append(float(r["off_aivat"]["adj_mean_bb"]))
            cells[(base, cell)]["on"].append(float(r["on_aivat"]["adj_mean_bb"]))
        # Coverage is per-file (one base+cell+seed); attribute it once to this file's basetag.
        bp_cov = doc.get("bp_coverage") or {}
        if bp_cov and results:
            cell0 = next(iter(results))
            suffix = f"_{cell0}_s{sm.group(1)}.json" if sm else f"_{cell0}.json"
            base = f.name[len("bs_"):].removesuffix(suffix)
            for street, c in bp_cov.items():
                cov[base][street][0] += int(c.get("hits", 0))
                cov[base][street][1] += int(c.get("misses", 0))

    bases = sorted({b for (b, _c) in cells})
    for base in bases:
        print(f"\n=== base: {base} ===")
        print(f"{'cell':16s} {'base bb/100':>14s} {'+exploit bb/100':>16s} {'k':>3s}")
        weak_means, strong_means = [], []
        for cell in WEAK + STRONG + tuple(
                c for (b, c) in cells if b == base and c not in WEAK + STRONG):
            d2 = cells.get((base, cell))
            if not d2:
                continue
            om, ose, k = _pool(d2["off"])
            nm, nse, _ = _pool(d2["on"])
            tag = "weak" if cell in WEAK else ("strong" if cell in STRONG else "")
            print(f"{cell:16s} {om:>+9.2f}+-{ose:>4.2f} {nm:>+11.2f}+-{nse:>4.2f} {k:>3d}  {tag}")
            (weak_means if cell in WEAK else strong_means if cell in STRONG else []).append(om)
        if weak_means:
            print(f"  MEAN vs WEAK   base = {sum(weak_means)/len(weak_means):+.2f} bb/100 "
                  f"({len(weak_means)} cells)")
        if strong_means:
            print(f"  MEAN vs STRONG base = {sum(strong_means)/len(strong_means):+.2f} bb/100 "
                  f"({len(strong_means)} cells)")
        if base in cov:
            cstr = "  ".join(f"{s}={h/(h+m)*100:.0f}%" for s, (h, m) in cov[base].items() if (h + m))
            print(f"  blueprint coverage: {cstr}")


if __name__ == "__main__":
    main()
