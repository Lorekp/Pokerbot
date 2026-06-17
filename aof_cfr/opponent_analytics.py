"""Opponent & population analytics for AoF - the read side of the exploit engine.

The exploit engine ADAPTS to opponents live (opponent_db + profiler). This module is the offline
REPORT over the recorded hands (``runs/live_hands/*.jsonl``) that tells us what the field actually
does, per villain and in aggregate - the numbers that size every exploit we run. It reuses the
validated ``live_exploit.observations_from_record`` parser (same push/fold reconstruction the live
profiler uses), so the report can never disagree with what the engine learned.

Six measurements (all derivable from the existing hand log; timing needs the new per-action ``dt``):

  1. Fold-to-jam % by position / opponent - directly sizes our open-jam steal EV (#1 AoF lever).
  2. Open-jam width & call-vs-jam width - RFI vs defending frequency, per villain and population.
  3. Showdown range SHAPE - from revealed cards (``shown``): which hands inside the width actually
     jam / call, not just the frequency.
  4. Timing tells - snap-jam vs tank correlated with the action (uses the per-action ``dt``).
  5. Population drift over time - is the field tightening week-over-week (are old reads going stale)?
  6. Repeat-encounter rate - how often we re-see a villain (the ROI of keeping a profile on them).

Play-money / own-account evaluation use only.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from aof_cfr.infoset import decode_infoset_key
from aof_cfr.live_exploit import _norm_token, _PUSH_TOKENS, observations_from_record
from aof_cfr.opponent_db import identity_key

_POSITIONS = ("UTG", "BTN", "SB", "BB")
SNAP_S = 1.5     # acted faster than this = "snap" (pre-decided / pre-clicked-feel)
TANK_S = 4.0     # slower than this = "tank" (deliberated)


# --------------------------------------------------------------------------------------------------
def _facing_jam(hist: str) -> bool:
    return "A" in (hist or "")


def _first_in(hist: str) -> bool:
    """True at an unopened (first-to-act) node: empty history or only folds ahead."""
    h = hist or "-"
    return h == "-" or set(h) <= {"F"}


def _iso_week(ts: str) -> str:
    """ISO year-week label (e.g. 2026-W24) from a record timestamp; '?' if unparseable."""
    try:
        d = datetime.fromisoformat((ts or "").replace("Z", "+00:00"))
        y, w, _ = d.isocalendar()
        return f"{y}-W{w:02d}"
    except ValueError:
        return "?"


# --------------------------------------------------------------------------------------------------
@dataclass
class OppAgg:
    """Per-villain (or population) accumulators. Frequencies are derived in the report."""
    name: str = "?"
    hands: int = 0
    days: set = field(default_factory=set)
    decisions: int = 0
    pushes: int = 0
    by_pos: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))  # pos -> [dec, push]
    open: list = field(default_factory=lambda: [0, 0])         # first-in: [total, jam]
    facing_jam: list = field(default_factory=lambda: [0, 0])   # facing >=1 jam: [total, call]
    # showdown range shape: bucket -> [seen, acted-aggressively] split by context
    shape_open: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    shape_call: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    # timing (seconds) by the action taken (villain only)
    t_jam: list = field(default_factory=list)
    t_fold: list = field(default_factory=list)

    def fold_to_jam(self) -> float | None:
        t, c = self.facing_jam
        return (t - c) / t if t else None

    def call_vs_jam_freq(self) -> float | None:
        t, c = self.facing_jam
        return c / t if t else None

    def open_jam_freq(self) -> float | None:
        t, j = self.open
        return j / t if t else None


@dataclass
class Analysis:
    opps: dict                       # identity -> OppAgg
    pop: OppAgg                      # population aggregate
    drift: dict                      # period -> [dec, push, open_total, open_jam, faj_total, faj_call]
    n_records: int = 0
    n_usable: int = 0                # records that parsed to observations
    timing_records: int = 0          # records carrying any per-action dt


def _add_obs(agg: OppAgg, pos: str, hist: str, pushed: bool) -> None:
    agg.decisions += 1
    agg.pushes += 1 if pushed else 0
    bp = agg.by_pos[pos]
    bp[0] += 1
    bp[1] += 1 if pushed else 0
    if _facing_jam(hist):
        agg.facing_jam[0] += 1
        agg.facing_jam[1] += 1 if pushed else 0
    elif _first_in(hist):
        agg.open[0] += 1
        agg.open[1] += 1 if pushed else 0


def analyze(records: list[dict]) -> Analysis:
    opps: dict[str, OppAgg] = {}
    pop = OppAgg(name="POPULATION")
    drift: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    n_usable = 0
    timing_records = 0

    for rec in records:
        obs = observations_from_record(rec)
        if not obs:
            continue
        n_usable += 1
        period = _iso_week(rec.get("ts", ""))
        day = (rec.get("ts", "") or "")[:10]
        seen_idents_this_hand: set[str] = set()

        for ident, key, pushed in obs:
            try:
                pos, bucket, hist, _depth = decode_infoset_key(key)
            except ValueError:
                continue
            o = opps.get(ident)
            if o is None:
                o = opps[ident] = OppAgg(name=ident)
            if bucket == "*":  # aggregate (frequency) row - one per villain decision
                _add_obs(o, pos, hist, pushed)
                _add_obs(pop, pos, hist, pushed)
                seen_idents_this_hand.add(ident)
                dr = drift[period]
                dr[0] += 1
                dr[1] += 1 if pushed else 0
                if _facing_jam(hist):
                    dr[4] += 1
                    dr[5] += 1 if pushed else 0
                elif _first_in(hist):
                    dr[2] += 1
                    dr[3] += 1 if pushed else 0
            else:               # per-bucket showdown reveal - range SHAPE
                target = o.shape_call if _facing_jam(hist) else o.shape_open
                target[bucket][0] += 1
                target[bucket][1] += 1 if pushed else 0
                ptarget = pop.shape_call if _facing_jam(hist) else pop.shape_open
                ptarget[bucket][0] += 1
                ptarget[bucket][1] += 1 if pushed else 0

        for ident in seen_idents_this_hand:
            o = opps[ident]
            o.hands += 1
            if day:
                o.days.add(day)
        if day:
            pop.days.add(day)
        pop.hands += 1

        # --- timing (#4): walk the raw actions; villains only, preflop, dt present ---
        hero_seat = (rec.get("hero") or {}).get("seat")
        names = {p.get("seat"): p.get("name") for p in (rec.get("players") or [])}
        had_dt = False
        for a in rec.get("actions") or []:
            if not str(a.get("street") or "preflop").lower().startswith("pre"):
                continue
            dt = a.get("dt")
            if dt is None or a.get("seat") == hero_seat:
                continue
            had_dt = True
            ident = identity_key(a.get("user_id"), a.get("name") or names.get(a.get("seat")))
            if ident is None:
                continue
            pushed = _norm_token(a.get("type")) in _PUSH_TOKENS
            bucket_list = (opps.setdefault(ident, OppAgg(name=ident)).t_jam if pushed
                           else opps[ident].t_fold)
            bucket_list.append(float(dt))
            (pop.t_jam if pushed else pop.t_fold).append(float(dt))
        if had_dt:
            timing_records += 1

    # backfill display names from the player rosters (identity -> last-seen username)
    name_by_ident = _name_map(records)
    for ident, o in opps.items():
        o.name = name_by_ident.get(ident, ident)

    return Analysis(opps=opps, pop=pop, drift=drift, n_records=len(records),
                    n_usable=n_usable, timing_records=timing_records)


def _name_map(records: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rec in records:
        for p in rec.get("players") or []:
            ident = identity_key(p.get("user_id"), p.get("name"))
            if ident and p.get("name"):
                out[ident] = p["name"]
    return out


# --------------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------------
def _pct(x: float | None) -> str:
    return f"{100*x:4.0f}%" if x is not None else "  - "


def _mean(xs: list) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def _top_buckets(shape: dict, n: int = 12) -> str:
    """Most-jammed/called buckets (by aggressive frequency, min 2 sightings)."""
    rows = [(b, c[1] / c[0], c[0]) for b, c in shape.items() if c[0] >= 2]
    rows.sort(key=lambda r: (-r[1], -r[2]))
    return "  ".join(f"{b}{int(100*f)}%" for b, f, _ in rows[:n]) or "(<2 reveals)"


def render(an: Analysis, *, min_dec: int = 12, top: int = 25) -> str:
    out: list[str] = ["=" * 78, " AoF OPPONENT & POPULATION ANALYTICS", "=" * 78,
                      f" records: {an.n_records}   usable (parsed): {an.n_usable}"
                      f"   with-timing: {an.timing_records}"]
    p = an.pop
    out += ["", " POPULATION", "-" * 78,
            f"   decisions {p.decisions}   distinct villains {len(an.opps)}"
            f"   field push {_pct(p.pushes / p.decisions if p.decisions else None)}",
            f"   open-jam width      {_pct(p.open_jam_freq())}   (n={p.open[0]})",
            f"   fold-to-jam         {_pct(p.fold_to_jam())}   "
            f"call-vs-jam {_pct(p.call_vs_jam_freq())}   (n={p.facing_jam[0]})"]
    byp = "   ".join(f"{pos} {_pct(p.by_pos[pos][1]/p.by_pos[pos][0] if p.by_pos.get(pos) and p.by_pos[pos][0] else None)}"
                     for pos in _POSITIONS)
    out.append(f"   push% by position:  {byp}")
    if p.t_jam or p.t_fold:
        out += [f"   timing  jam {_fmt_t(p.t_jam)}   fold {_fmt_t(p.t_fold)}   "
                f"(snap<{SNAP_S}s tank>{TANK_S}s)"]
    if p.shape_open or p.shape_call:
        out += [f"   open-jam range shape (revealed): {_top_buckets(p.shape_open)}",
                f"   call-vs-jam range  (revealed):   {_top_buckets(p.shape_call)}"]

    # repeat-encounter (#6)
    multi = [o for o in an.opps.values() if len(o.days) > 1]
    heavy = [o for o in an.opps.values() if o.hands >= 30]
    nopp = max(len(an.opps), 1)
    out += ["", " REPEAT ENCOUNTERS  (profile ROI)", "-" * 78,
            f"   villains: {len(an.opps)}   mean hands/villain {sum(o.hands for o in an.opps.values())/nopp:.1f}"
            f"   seen on >1 day: {len(multi)} ({100*len(multi)/nopp:.0f}%)"
            f"   >=30 hands: {len(heavy)} ({100*len(heavy)/nopp:.0f}%)"]

    # drift (#5)
    if len(an.drift) >= 2:
        out += ["", " POPULATION DRIFT  (is the field tightening?)", "-" * 78,
                f"   {'period':>9} {'dec':>6} {'push%':>7} {'open-jam%':>10} {'fold-to-jam%':>13}"]
        for per in sorted(an.drift):
            d = an.drift[per]
            push = d[1] / d[0] if d[0] else None
            opn = d[3] / d[2] if d[2] else None
            f2j = (d[4] - d[5]) / d[4] if d[4] else None
            out.append(f"   {per:>9} {d[0]:>6} {_pct(push):>7} {_pct(opn):>10} {_pct(f2j):>13}")

    # per-opponent table (#1, #2) - only villains with a meaningful sample
    ranked = sorted((o for o in an.opps.values() if o.decisions >= min_dec),
                    key=lambda o: -o.decisions)
    out += ["", f" PER-OPPONENT  (>= {min_dec} decisions; top {top})", "-" * 78,
            f"   {'villain':<18} {'dec':>5} {'hnds':>5} {'push%':>6} {'open%':>6}"
            f" {'f2jam%':>7} {'call%':>6} {'jamT':>6} {'foldT':>6}"]
    for o in ranked[:top]:
        out.append(
            f"   {o.name[:18]:<18} {o.decisions:>5} {o.hands:>5}"
            f" {_pct(o.pushes/o.decisions if o.decisions else None):>6}"
            f" {_pct(o.open_jam_freq()):>6} {_pct(o.fold_to_jam()):>7}"
            f" {_pct(o.call_vs_jam_freq()):>6}"
            f" {_t_short(o.t_jam):>6} {_t_short(o.t_fold):>6}")
    if len(ranked) > top:
        out.append(f"   ... +{len(ranked)-top} more villains with >= {min_dec} decisions")
    if not ranked:
        out.append("   (no villain has reached the sample threshold yet)")
    out += ["", "=" * 78,
            " fold-to-jam sizes our steal EV; open/call widths + range shape size our value jams.",
            "=" * 78]
    return "\n".join(out)


def _fmt_t(xs: list) -> str:
    m = _mean(xs)
    if m is None:
        return "-"
    snap = sum(1 for x in xs if x < SNAP_S) / len(xs)
    tank = sum(1 for x in xs if x > TANK_S) / len(xs)
    return f"{m:.1f}s (snap {100*snap:.0f}% / tank {100*tank:.0f}%, n={len(xs)})"


def _t_short(xs: list) -> str:
    m = _mean(xs)
    return f"{m:.1f}s" if m is not None else "-"


# --------------------------------------------------------------------------------------------------
def load_records(hands_dir: str | Path, since: str | None = None) -> list[dict]:
    out: list[dict] = []
    d = Path(hands_dir)
    if not d.exists():
        return out
    for fp in sorted(d.glob("*.jsonl")):
        if since and fp.stem < since:
            continue
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out
