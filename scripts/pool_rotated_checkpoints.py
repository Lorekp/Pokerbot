#!/usr/bin/env python3
"""Pool every rotated-eval shard checkpoint in a directory into a running bb/100 estimate + 95% CI.

Each shard JSON (written by ``eval_exploitation_battery.py --rotate-seats --out ...``) holds
``results[<field>][off_base|on_exploit] = {decks, mean_bb, std}`` -- one INDEPENDENT seat-rotated
duplicate eval over a disjoint deck-seed range. Pooling the per-shard ``(n, mean, std)`` estimates with
the standard pooled-variance formula (within-group SS + between-group SS about the pooled mean) gives an
unbiased running win-rate whose CI tightens as shards accumulate. ``off_base`` is the headline absolute
win-rate (pure GTO base vs the homogeneous field); ``on_exploit`` is base + the exploitation overlay.

The shard JSONs are immutable, append-only checkpoints: re-running this pooler at any time (mid-run,
after a kill, on resume) reflects every shard completed so far. Used by scripts/overnight_winrate.ps1.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path


def _pool(rows):
    """Pool independent (n, mean, std) estimates -> (N, pooled_mean, pooled_se)."""
    n_tot = sum(n for n, _, _ in rows)
    if n_tot == 0:
        return 0, 0.0, 0.0
    mean = sum(n * m for n, m, _ in rows) / n_tot
    ss = sum((n - 1) * s * s for n, _, s in rows if n > 1)
    ss += sum(n * (m - mean) ** 2 for n, m, _ in rows)
    var = ss / (n_tot - 1) if n_tot > 1 else 0.0
    return n_tot, mean, math.sqrt(var / n_tot)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="directory of per-shard *.json checkpoints")
    ap.add_argument("--out", default="", help="write the pooled summary JSON here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.shards, "*.json")))
    fields: dict[str, dict] = {}
    bad = 0
    for fp in files:
        try:
            d = json.loads(Path(fp).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 -- a shard still being written / truncated: skip, retry next round
            bad += 1
            continue
        for f, r in (d.get("results", {}) or {}).items():
            if not isinstance(r, dict):
                continue
            slot = fields.setdefault(f, {"off": [], "on": [], "shards": 0})
            slot["shards"] += 1
            ob, oe = r.get("off_base"), r.get("on_exploit")
            if ob and ob.get("decks", 0) > 0:
                slot["off"].append((ob["decks"], ob["mean_bb"], ob["std"]))
            if oe and oe.get("decks", 0) > 0:
                slot["on"].append((oe["decks"], oe["mean_bb"], oe["std"]))

    summary: dict[str, dict] = {}
    lines: list[str] = []
    for f in sorted(fields):
        off, on = fields[f]["off"], fields[f]["on"]
        nN, om, ose = _pool(off)
        eN, em, ese = _pool(on) if on else (0, 0.0, 0.0)
        ohw, ehw = 1.96 * 100 * ose, 1.96 * 100 * ese
        olo, ohi = 100 * om - ohw, 100 * om + ohw
        summary[f] = {
            "shards": fields[f]["shards"], "decks": nN, "hero_hands": nN * 6,
            "off_base_bb100": 100 * om, "off_base_halfwidth95": ohw,
            "off_base_ci95": [olo, ohi], "off_base_se_bb100": 100 * ose,
            "on_exploit_bb100": 100 * em, "on_exploit_halfwidth95": ehw,
        }
        lines.append(
            f"{f:24s} base {100 * om:+8.1f} +/-{ohw:6.1f} bb/100  "
            f"[{olo:+8.1f},{ohi:+8.1f}]  exploit {100 * em:+8.1f}  "
            f"decks={nN} (~{nN * 6} hero-hands) shards={fields[f]['shards']}"
        )

    out_txt = "\n".join(lines) if lines else "(no completed shards yet)"
    if bad:
        out_txt += f"\n({bad} shard file(s) unreadable this pass -- likely mid-write, will fold in next round)"
    if not args.quiet:
        print(out_txt)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
