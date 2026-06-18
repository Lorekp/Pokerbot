"""Per-opponent 6-max NLHE report from live hand records (the cash mirror of the AoF opponent report).

Reads v2 hand records (``runs/live_hands/**/*.jsonl[.gz]`` from the live coach) and, per opponent,
reports the signal the new capture pipeline unlocks:

  * derived stats        — VPIP / PFR / WTSD / aggression (from the action stream);
  * server HUD           — the client's whole-history VPIP/PFR/WTSD/3bet/cbet (when captured);
  * TIMING reconciled with HOLDINGS — for hands the villain SHOWED, their aggressive-action decision
    time (`dt`) split by whether the revealed hand was a bluff (hs<0.45) or value (hs>0.60). A villain
    who tanks before bluffing and snaps value (or vice-versa) is exploitable; this is the tell that the
    decision-time + showdown-reveal capture makes measurable;
  * bet sizing           — mean bet/raise size as a fraction of the pot (a fish/reg discriminator);
  * net won/lost         — chips netted vs us over the sample (a direct table-selection signal).

Play-money / own-account analysis only.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

from pokerbot.opponent.player_profiles import _STREET_BOARD_LEN, _player_key, _static_hand_strength

_AGGR = frozenset({"bet", "raise", "all_in"})
_VPIP = frozenset({"call", "bet", "raise", "all_in"})


def _read(paths):
    for p in paths:
        op = gzip.open if str(p).endswith(".gz") else open
        try:
            with op(p, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except ValueError:
                            continue
        except OSError:
            continue


class Agg:
    __slots__ = ("name", "hands", "vpip", "pfr", "wtsd", "saw_flop", "aggr", "calls",
                 "net", "dt", "bluff_dt", "value_dt", "sizes", "hud", "shown")

    def __init__(self) -> None:
        self.name = None
        self.hands = self.vpip = self.pfr = self.wtsd = self.saw_flop = 0
        self.aggr = self.calls = 0
        self.net = 0.0
        self.dt: list[float] = []
        self.bluff_dt: list[float] = []
        self.value_dt: list[float] = []
        self.sizes: list[float] = []
        self.hud: dict | None = None
        self.shown = 0


def _reconcile(rec: dict, aggs: dict[str, Agg]) -> None:
    """Fold one hand into the per-opponent aggregates, reconciling timing with revealed holdings."""
    actions = rec.get("actions") or []
    board = rec.get("board") or []
    # revealed holes by player key (for the timing<->holdings reconciliation)
    holes: dict[str, list] = {}
    for srec in (rec.get("shown") or []):
        key = _player_key(srec.get("user_id"), srec.get("name"))
        cards = [c for c in (srec.get("cards") or []) if c]
        if key and len(cards) >= 2:
            try:
                from pokerbot.engine.cards import Card
                holes[key] = [Card.from_str(cards[0]), Card.from_str(cards[1])]
            except Exception:  # noqa: BLE001
                pass
    board_cards = []
    for code in board:
        try:
            from pokerbot.engine.cards import Card
            board_cards.append(Card.from_str(code))
        except Exception:  # noqa: BLE001
            board_cards.append(None)

    # per-player flags this hand
    by_player: dict[str, list[dict]] = defaultdict(list)
    for a in actions:
        key = _player_key(a.get("user_id"), a.get("name"))
        if key:
            by_player[key].append(a)
    pot = 0.0
    for a in actions:  # running pot for the sizing fraction
        pot += float(a.get("amount") or 0.0)

    for key, acts in by_player.items():
        g = aggs[key]
        g.hands += 1
        g.name = acts[-1].get("name") or g.name
        pre = [a for a in acts if a.get("street") == "preflop"]
        if any(a.get("type") in _VPIP for a in pre):
            g.vpip += 1
        if any(a.get("type") in ("raise", "bet", "all_in") for a in pre):
            g.pfr += 1
        if len(board) >= 5 and rec.get("winners") and not any(a.get("type") == "fold" for a in acts):
            g.wtsd += 1
        for a in acts:
            t = a.get("type")
            if t in _AGGR:
                g.aggr += 1
                amt = float(a.get("amount") or 0.0)
                if pot > 0 and amt > 0:
                    g.sizes.append(amt / pot)
            elif t == "call":
                g.calls += 1
            dt = a.get("dt")
            if dt is not None:
                g.dt.append(float(dt))

    # timing <-> holdings reconciliation (only for shown players)
    run_pot = 0.0
    cur_street = None
    for a in actions:
        st = a.get("street")
        if st != cur_street:
            cur_street = st
        t = a.get("type")
        if t in _AGGR:
            key = _player_key(a.get("user_id"), a.get("name"))
            hole = holes.get(key)
            dt = a.get("dt")
            if hole is not None and dt is not None:
                n_board = _STREET_BOARD_LEN.get(st, 0)
                bslice = [c for c in board_cards[:n_board] if c is not None]
                if len(bslice) == n_board:
                    hs = _static_hand_strength(hole, bslice)
                    if hs < 0.45:
                        aggs[key].bluff_dt.append(float(dt))
                    elif hs > 0.60:
                        aggs[key].value_dt.append(float(dt))
        run_pot += float(a.get("amount") or 0.0)

    # server HUD snapshot + reveal count + net
    for uid, stats in (rec.get("hud") or {}).items():
        aggs[str(uid)].hud = stats
    for srec in (rec.get("shown") or []):
        key = _player_key(srec.get("user_id"), srec.get("name"))
        if key:
            aggs[key].shown += 1
    for w in (rec.get("winners") or []):
        key = _player_key(w.get("user_id"), w.get("name"))
        if key and w.get("amount") is not None:
            aggs[key].net += float(w["amount"])


TIMING_HANDS_TARGET = 1000   # timed hands needed before wiring the dt-based timing-tell model
_MIN_RECON_PER_VILLAIN = 10  # showdown-reconciled samples per villain for a usable per-player tell


def compute_timing_readiness(records, *, target_timed: int = TIMING_HANDS_TARGET,
                             min_per_villain: int = _MIN_RECON_PER_VILLAIN) -> dict:
    """Scan NLHE hand records for whether we have enough TIMED data to wire the dt-based timing-tell
    model. The headline gate is ``timed_hands >= target_timed`` (a 'timed hand' has >=1 action carrying
    a decision time ``dt``). Also reports showdown-reconciled coverage — aggressive dt-actions by a
    villain whose hole we later learned at showdown — since the tell (dt split by bluff/value) can only
    be FIT on those; 1000 timed hands with no showdowns still can't train it."""
    n_hands = n_timed = n_showdown = 0
    recon: dict[str, int] = defaultdict(int)
    for rec in records:
        if (rec.get("table") or {}).get("game_mode") == "aof":
            continue  # AoF has its own report + push/fold has no postflop timing tell
        n_hands += 1
        actions = rec.get("actions") or []
        if any(a.get("dt") is not None for a in actions):
            n_timed += 1
        shown_keys = {_player_key(s.get("user_id"), s.get("name")) for s in (rec.get("shown") or [])}
        shown_keys.discard(None)
        if shown_keys:
            n_showdown += 1
            for a in actions:
                if a.get("type") in _AGGR and a.get("dt") is not None:
                    k = _player_key(a.get("user_id"), a.get("name"))
                    if k in shown_keys:
                        recon[k] += 1
    return {
        "hands": n_hands,
        "timed_hands": n_timed,
        "target_timed": target_timed,
        "showdown_hands": n_showdown,
        "reconciled_samples": sum(recon.values()),
        "villains_reconciled_ge_min": sum(1 for v in recon.values() if v >= min_per_villain),
        "min_per_villain": min_per_villain,
        "ready": n_timed >= target_timed,
    }


def print_timing_readiness(rd: dict) -> None:
    tgt, timed = rd["target_timed"], rd["timed_hands"]
    pct = (100.0 * timed / tgt) if tgt else 100.0
    verdict = "READY" if rd["ready"] else "NOT READY"
    print(f"\nTIMING READINESS (gate to wire dt-based timing tells)  ->  {verdict}")
    print(f"  timed hands (>=1 dt action): {timed:>6} / {tgt}   [{pct:3.0f}% of target]")
    print(f"  showdown hands (reveals):    {rd['showdown_hands']:>6}")
    print(f"  reconciled timing samples:   {rd['reconciled_samples']:>6}   "
          f"(aggressive dt-actions by a villain we saw show down)")
    print(f"  villains >= {rd['min_per_villain']} reconciled:        "
          f"{rd['villains_reconciled_ge_min']:>6}")
    if rd["ready"]:
        if rd["reconciled_samples"] == 0:
            print("  => 1000-timed-hand gate MET, but 0 showdown-reconciled samples — the tell needs "
                  "reveals to fit. Keep collecting until reconciled samples accrue, THEN wire timing.")
        else:
            print("  => gate MET. Wire the dt -> bluff/value timing model and battery/replay-validate it.")
    else:
        need = tgt - timed
        print(f"  => collect {need} more timed hand(s) before wiring timing (target {tgt}).")


def _pct(num: int, den: int) -> str:
    return f"{100.0 * num / den:4.0f}" if den else "   -"


def _med(xs: list[float]) -> str:
    return f"{statistics.median(xs):4.1f}" if xs else "   -"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="runs/live_hands", help="directory of live hand records")
    ap.add_argument("--min-hands", type=int, default=5, help="only print opponents with >= this many hands")
    ap.add_argument("--timing-target", type=int, default=TIMING_HANDS_TARGET,
                    help="timed-hand count that flags the timing-tell model as ready to wire")
    args = ap.parse_args()

    paths = sorted(glob.glob(str(Path(args.dir) / "**" / "*.jsonl"), recursive=True)
                   + glob.glob(str(Path(args.dir) / "**" / "*.jsonl.gz"), recursive=True))
    records = list(_read(paths))  # materialize once: feeds both the opponent table and the readiness scan
    aggs: dict[str, Agg] = defaultdict(Agg)
    n_rec = 0
    for rec in records:
        if (rec.get("table") or {}).get("game_mode") == "aof":
            continue  # this is the NLHE/cash report; AoF has its own (opponent_report.py)
        _reconcile(rec, aggs)
        n_rec += 1

    rows = [(k, g) for k, g in aggs.items() if g.hands >= args.min_hands]
    rows.sort(key=lambda kv: -kv[1].hands)
    print(f"\n6-max NLHE opponent report  ({n_rec} hands, {len(rows)} opponents >= {args.min_hands}h)\n")
    hdr = (f"{'player':<18}{'n':>4}{'VPIP':>5}{'PFR':>5}{'WTSD':>5}{'AF':>5}{'size':>6}"
           f"{'net':>8}{'tBluff':>7}{'tValue':>7}{'HUD vpip/pfr/wtsd':>20}")
    print(hdr)
    print("-" * len(hdr))
    for key, g in rows:
        af = g.aggr / g.calls if g.calls else (g.aggr if g.aggr else 0.0)
        size = f"{statistics.mean(g.sizes):4.2f}" if g.sizes else "   -"
        hud = ("" if not g.hud
               else f"{g.hud.get('vpip','-')}/{g.hud.get('pfr','-')}/{g.hud.get('wtsd','-')}")
        name = (g.name or key)[:17]
        print(f"{name:<18}{g.hands:>4}{_pct(g.vpip, g.hands):>5}{_pct(g.pfr, g.hands):>5}"
              f"{_pct(g.wtsd, g.hands):>5}{af:>5.1f}{size:>6}{g.net:>8.0f}"
              f"{_med(g.bluff_dt):>7}{_med(g.value_dt):>7}{hud:>20}")
    print("\ntBluff/tValue = median decision-time (s) on aggressive actions later shown as bluff/value "
          "(the timing tell). 'size' = mean bet/raise as a fraction of pot. 'net' = chips netted vs us.")
    print("HUD = server whole-history stats (blank if not captured this sample).")

    print_timing_readiness(compute_timing_readiness(records, target_timed=args.timing_target))


if __name__ == "__main__":
    main()
