"""AoF opponent & population analytics report (CLI).

Thin wrapper over ``aof_cfr.opponent_analytics`` — reads the recorded hands and prints fold-to-jam,
open/call widths, showdown range shape, timing tells, population drift and repeat-encounter rate, per
villain and in aggregate. These are the field reads that size every exploit the engine runs.

Usage:
  python scripts/aof/opponent_report.py [--hands-dir runs/live_hands] [--since YYYY-MM-DD]
                                        [--min-dec 12] [--top 25]
Play-money / own-account evaluation only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:  # Windows consoles default to cp1252 and choke on the report's box/dash glyphs.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aof_cfr.opponent_analytics import analyze, load_records, render


def main() -> None:
    ap = argparse.ArgumentParser(description="AoF opponent & population analytics")
    ap.add_argument("--hands-dir", default="runs/live_hands")
    ap.add_argument("--since", default=None, help="only files dated >= YYYY-MM-DD")
    ap.add_argument("--min-dec", type=int, default=12, help="min decisions for a per-opponent row")
    ap.add_argument("--top", type=int, default=25, help="how many villains to list")
    args = ap.parse_args()

    records = load_records(args.hands_dir, args.since)
    if not records:
        print(f"no AoF hand records found under {args.hands_dir}")
        return
    an = analyze(records)
    print(render(an, min_dec=args.min_dec, top=args.top))


if __name__ == "__main__":
    main()
