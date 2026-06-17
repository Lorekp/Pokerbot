"""Per-STAKE population analysis for AoF — tests the hypothesis "different stakes, different field".

The hand database (runs/live_hands/*.jsonl) already records the stake (table.sb / table.bb) of every
hand. This script groups the recorded hands by stake and aggregates the OPPONENT POPULATION at each
stake (every non-hero push/fold, reusing the validated ``observations_from_record`` parser), then
reports per-stake field tendencies side by side so you can see whether (and how) the population
differs by stake — and optionally emits a per-stake population PRIOR the engine can cold-start from.

Why a separate per-stake population (vs the per-player profiles in profiles.json): a specific player
(by user_id) is the same person at every stake, but the FIELD COMPOSITION differs — lower stakes are
typically looser/weaker. A per-stake prior lets the exploit engine start a brand-new opponent at the
right baseline for the table they're sitting at.

Usage:
  python scripts/aof/population_by_stake.py [--hands-dir runs/live_hands] [--emit-priors]
Play-money / own-account evaluation only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aof_cfr.live_exploit import observations_from_record


def stake_label(record: dict) -> str | None:
    """Canonical 'sb/bb' label for a hand, or None if the stake isn't recorded."""
    t = record.get("table") or {}
    sb, bb = t.get("sb"), t.get("bb")
    if bb is None:
        return None
    if sb is None:
        return f"?/{bb:g}"
    return f"{float(sb):g}/{float(bb):g}"


def _position_of(key: str) -> str:
    # aggregate infoset key is 'POS|*|HIST|dX' (or per-bucket 'POS|BUCKET|HIST|dX')
    return key.split("|", 1)[0]


def collect(hands_dir: Path) -> dict:
    """stake -> {'push':int,'total':int, 'by_pos':{pos:[push,total]}, 'players':set, 'hands':int}."""
    agg: dict = defaultdict(lambda: {"push": 0, "total": 0,
                                     "by_pos": defaultdict(lambda: [0, 0]),
                                     "players": set(), "hands": 0})
    files = sorted(hands_dir.glob("*.jsonl"))
    for fp in files:
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            stake = stake_label(rec)
            if stake is None:
                continue
            obs = observations_from_record(rec)   # [(identity, infoset_key, pushed)] or None
            if not obs:
                continue
            a = agg[stake]
            a["hands"] += 1
            # count each aggregate (pos|*|...) decision once; per-bucket showdown rows are extra detail
            for ident, key, pushed in obs:
                if "|*|" not in key:
                    continue
                a["players"].add(ident)
                a["total"] += 1
                a["push"] += 1 if pushed else 0
                pc = a["by_pos"][_position_of(key)]
                pc[0] += 1 if pushed else 0
                pc[1] += 1
    return agg


def report(agg: dict) -> None:
    stakes = sorted(agg, key=lambda s: (len(agg[s]["players"]), agg[s]["total"]), reverse=True)
    print("=" * 72)
    print(" AoF POPULATION BY STAKE  (field push-frequency = how loose the table jams)")
    print("=" * 72)
    print(f"\n {'stake':>10} {'hands':>7} {'players':>8} {'obs':>8} {'push%':>7}   by position")
    positions = ["UTG", "BTN", "SB", "BB"]
    for s in stakes:
        a = agg[s]
        if not a["total"]:
            continue
        pf = 100.0 * a["push"] / a["total"]
        byp = "  ".join(
            f"{p} {100.0*a['by_pos'][p][0]/a['by_pos'][p][1]:.0f}%" if a["by_pos"].get(p) and a["by_pos"][p][1]
            else f"{p} —" for p in positions)
        print(f" {s:>10} {a['hands']:>7} {len(a['players']):>8} {a['total']:>8} {pf:>6.1f}%   {byp}")
    # hypothesis read: spread in field push% across stakes with a meaningful sample
    samp = [(s, 100.0 * agg[s]["push"] / agg[s]["total"]) for s in stakes if agg[s]["total"] >= 200]
    if len(samp) >= 2:
        lo, hi = min(samp, key=lambda x: x[1]), max(samp, key=lambda x: x[1])
        print(f"\n field push% spread (n>=200 stakes): {hi[1]-lo[1]:.1f} pts "
              f"(loosest {hi[0]} {hi[1]:.1f}% vs tightest {lo[0]} {lo[1]:.1f}%)")
        print(" => different stakes DO show a different field" if hi[1]-lo[1] >= 3.0
              else " => stakes look similar so far (need more data, or the field is homogeneous)")
    else:
        print("\n (need >=200 observations at 2+ stakes to judge the hypothesis)")
    print("=" * 72)


def emit_priors(agg: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for s, a in agg.items():
        if a["total"] < 200:
            continue
        prior = {"stake": s, "n_obs": a["total"], "n_players": len(a["players"]),
                 "push_freq": a["push"] / a["total"],
                 "push_freq_by_pos": {p: (c[0] / c[1]) for p, c in a["by_pos"].items() if c[1]}}
        fn = out_dir / f"population_prior_stake_{s.replace('/', '-')}.json"
        fn.write_text(json.dumps(prior, indent=2), encoding="utf-8")
        print(f"  wrote {fn}")


def main() -> None:
    ap = argparse.ArgumentParser(description="AoF per-stake population analysis")
    ap.add_argument("--hands-dir", default="runs/live_hands")
    ap.add_argument("--emit-priors", action="store_true",
                    help="write per-stake population priors to data/aof/")
    args = ap.parse_args()
    agg = collect(Path(args.hands_dir))
    if not agg:
        print(f"no stake-tagged AoF hands found under {args.hands_dir}")
        return
    report(agg)
    if args.emit_priors:
        emit_priors(agg, Path("data/aof"))


if __name__ == "__main__":
    main()
