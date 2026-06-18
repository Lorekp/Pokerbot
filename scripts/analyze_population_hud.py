#!/usr/bin/env python3
"""Population HUD-stat analysis + exploitable-leak detector for a real-player hand dataset.

Computes, PER PLAYER, the full CoinPoker server-HUD stat suite — vpip, pfr, three_bet,
fold_to_three_bet, cbet, fold_to_cbet, wtsd, wsd (won-at-showdown), wwsf (won-when-saw-flop),
check_raise, steal, fold_to_steal, af (aggression factor), fold_to_bet (fold-to-aggression) — by
replaying each hand's betting in true action order with proper OPPORTUNITY denominators (a stat is
``made / chances``, not ``made / hands``). Then it aggregates the population distribution
(percentiles) and runs a rule-based **exploitable-pattern detector** that maps population leaks to
concrete value-extraction adjustments.

Data source: the IRC poker database (Maurer format) — the same ~1 GB ``IRCdata.tgz`` the per-context
prior fit streams (`scripts/analyze_irc_population.py`). This sibling computes the HUD SUITE +
leak report rather than the context Dirichlet. It streams the two-level tarball member-by-member
(no disk explosion).

  PDB line: nick ts n_players pos <preflop> [<flop> <turn> <river>] bankroll wagered won [cards...]
  HDB line: ts gameset# game# #players flop(#p/$) turn(#p/$) river(#p/$) showdn(#p/$) board...
  Action codes: B=blind f=fold k=check b=bet c=call r=raise A=allin -=no-action Q=quit K=kicked

CAVEATS recorded in the output: (1) IRC hold'em is mostly LIMIT (fixed sizes) — frequency stats
(3bet/cbet/WTSD/fold-to-X) are robust, but bet-SIZE elasticity is not observable; (2) the 1995-2001
field is even looser/more passive than a modern micro field, so treat magnitudes as a directional
upper bound on softness; (3) position (steal) uses pos-order button inference and is lower-confidence
than the position-independent stats.

Run: python scripts/analyze_population_hud.py data/datasets/IRCdata.tgz --min-hands 100
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_ACTION_RE = re.compile(r"^[BfkbcrAQK-]+$")
_INT_RE = re.compile(r"^-?\d+$")
_STREET_BOARD_N = (0, 3, 4, 5)
_AGGR_CHARS = frozenset("brA")   # bet / raise / all-in = aggressive
_VOL_CHARS = frozenset("kbcrA")  # voluntary decision chars (k=check is voluntary but not money)


def _new_stat() -> dict:
    # Each metric is (made, opportunities). A rate is made/opportunities; opportunities=0 => undefined.
    keys = ("vpip", "pfr", "three_bet", "fold_to_three_bet", "cbet", "fold_to_cbet",
            "wtsd", "wsd", "wwsf", "check_raise", "steal", "fold_to_steal", "fold_to_bet")
    s = {f"{k}_n": 0 for k in keys}
    s.update({f"{k}_d": 0 for k in keys})
    s.update(hands=0, bets=0, raises=0, calls=0, won_chips=0, saw_flop=0)
    return s


def parse_pdb_line_full(line: str):
    """(nick, ts, pos, [street strings], won:int|None, n_cards:int) or None. Robustly splits the
    action strings from the numeric tail (bankroll wagered won) and the optional 2 hole cards (shown
    only at showdown — so n_cards==2 is a clean WENT-TO-SHOWDOWN signal)."""
    t = line.split()
    if len(t) < 7:
        return None
    nick, ts = t[0], t[1]
    if not (_INT_RE.match(t[2]) and _INT_RE.match(t[3])):
        return None
    pos = int(t[3])
    streets, i = [], 4
    while i < len(t) and len(streets) < 4 and _ACTION_RE.match(t[i]) and not _INT_RE.match(t[i]):
        streets.append(t[i])
        i += 1
    if not streets:
        return None
    # numeric tail: bankroll wagered won  (3 ints). won is the 3rd.
    nums = []
    while i < len(t) and _INT_RE.match(t[i]) and len(nums) < 3:
        nums.append(int(t[i]))
        i += 1
    won = nums[2] if len(nums) == 3 else None
    n_cards = sum(1 for tok in t[i:] if len(tok) == 2 and not _INT_RE.match(tok))
    return nick, ts, pos, streets, won, n_cards


def parse_hdb_board(line: str):
    t = line.split()
    if len(t) < 8 or not _INT_RE.match(t[0]):
        return None
    board = [c for c in t[8:] if len(c) == 2 and not _INT_RE.match(c)]
    return t[0], board


def linear_actions(rows: list[tuple[int, str, list[str]]], board: list[str]):
    """Replay ONE hand round-robin in position order; yield (nick, street_idx, char, aggr_before)
    for every voluntary action (k/b/c/r/A and f), in true global order. ``aggr_before`` is the count
    of aggressive actions earlier in THAT street (preflop: = raises before => 3bet depth; postflop:
    = bets/raises before => facing depth). Mirrors the validated replay in analyze_irc_population.

    The k-th char of a player's street string is their k-th action that street; betting cycles the
    table in seat order, so consuming char k of every live player in position order reconstructs the
    sequence. Folds/quits drop a player from later rounds; leading 'B' = blind; '-' = no action."""
    rows = sorted(rows, key=lambda r: r[0])
    out = set()  # folded/quit nicks
    for st_idx in range(4):
        live = [(pos, nick, streets[st_idx]) for pos, nick, streets in rows
                if st_idx < len(streets) and nick not in out]
        if not live:
            continue
        if len(board) < _STREET_BOARD_N[st_idx]:
            break
        aggr = 0
        maxlen = max((len(s) for _, _, s in live), default=0)
        for k in range(maxlen):
            for _pos, nick, s in live:
                if nick in out or k >= len(s):
                    continue
                ch = s[k]
                if ch in ("Q", "K"):
                    out.add(nick)
                    continue
                if ch in ("B", "-"):
                    continue
                yield nick, st_idx, ch, aggr
                if ch in _AGGR_CHARS:
                    aggr += 1
                if ch == "f":
                    out.add(nick)


def process_hand(rows, board, meta, players: dict) -> None:
    """Fold one hand into per-player HUD accumulators. ``rows``=[(pos,nick,streets)];
    ``meta``={nick: (won, n_cards)}; ``players``=global {nick: stat}."""
    acts = list(linear_actions(rows, board))
    if not acts:
        return
    n_players_hand = len(rows)
    nicks = {nick for _p, nick, _s in rows}
    button = max((p for p, _n, _s in rows), default=0)  # highest pos acts latest preflop (button)
    # steal-eligible "late" positions: button + the seat before it (CO). pos-order inference (caveat).
    late = {button, button - 1}

    # ---- preflop pass: opener, 3bet, fold-to-3bet, steal ----
    pre = [(nick, ch, ab) for nick, st, ch, ab in acts if st == 0]
    opener = None
    three_done = False
    opened = False                  # any voluntary call/raise yet (blinds don't count)
    had_3bet_opp: set[str] = set()
    steal_counted: set[str] = set()
    folded_pre: set[str] = set()
    pos_by_nick = {nick: pos for pos, nick, _s in rows}
    for nick, ch, ab in pre:
        s = players[nick]
        # 3-bet: first live decision facing >=1 prior raise (opp = denominator, raise = numerator)
        if ch in _VOL_CHARS and ab >= 1 and nick not in had_3bet_opp:
            had_3bet_opp.add(nick)
            s["three_bet_d"] += 1
            if ch == "r":
                s["three_bet_n"] += 1
        # STEAL: folded to a late-position player in an UNOPENED pot (their chance) — denominator is
        # the opportunity, numerator is "raised". Checked BEFORE marking the pot opened so the
        # stealer's own raise still counts as their own chance.
        if (not opened and nick not in steal_counted
                and pos_by_nick.get(nick) in late and ch in ("f", "c", "r", "A")):
            steal_counted.add(nick)
            s["steal_d"] += 1
            if ch in ("r", "A"):
                s["steal_n"] += 1
        if ch in ("r", "A") and ab == 0 and opener is None:
            opener = nick
        if ch in ("c", "r", "A"):
            opened = True
        if ch == "f":
            folded_pre.add(nick)
        # fold-to-3bet for the opener: their response to a re-raise (decision facing >=2 raises)
        if nick == opener and ab >= 2 and not three_done:
            s["fold_to_three_bet_d"] += 1
            if ch == "f":
                s["fold_to_three_bet_n"] += 1
            three_done = True
    pf_aggressor = None
    # last preflop raiser = the postflop initiative holder
    for nick, ch, ab in pre:
        if ch in ("r", "A"):
            pf_aggressor = nick

    # fold-to-steal: a blind (lowest two pos) facing exactly the steal-raise and folding
    if opener is not None and pos_by_nick.get(opener) in late:
        blinds = sorted(rows, key=lambda r: r[0])[:2]
        for pos, nick, _s in blinds:
            if nick == opener:
                continue
            # their first preflop response after the steal
            resp = next((c for n, c, ab in pre if n == nick and ab >= 1), None)
            if resp is not None:
                players[nick]["fold_to_steal_d"] += 1
                if resp == "f":
                    players[nick]["fold_to_steal_n"] += 1

    # ---- flop pass: cbet, fold-to-cbet ----
    flop = [(nick, ch, ab) for nick, st, ch, ab in acts if st == 1]  # (nick, ch, aggr_before) on flop
    saw_flop_nicks = {nick for nick, _c, _ab in flop} | {
        n for n in nicks if n not in folded_pre and len(board) >= 3}
    if pf_aggressor is not None and pf_aggressor in saw_flop_nicks and len(board) >= 3:
        s = players[pf_aggressor]
        s["cbet_d"] += 1
        first_flop = next((ch for nick, ch, ab in flop if nick == pf_aggressor), None)
        cbet_made = first_flop == "b"
        if cbet_made:
            s["cbet_n"] += 1
        # fold-to-cbet for everyone else facing that bet
        if cbet_made:
            for nick in saw_flop_nicks:
                if nick == pf_aggressor:
                    continue
                resp = next((ch for n, ch, ab in flop if n == nick and ab >= 1), None)
                if resp is not None:
                    players[nick]["fold_to_cbet_d"] += 1
                    if resp == "f":
                        players[nick]["fold_to_cbet_n"] += 1

    # ---- per-player aggregate: vpip/pfr/af/fold-to-bet/check-raise/wtsd/wsd/wwsf ----
    by_player_street: dict[tuple[str, int], list[str]] = defaultdict(list)
    for nick, st, ch, ab in acts:
        by_player_street[(nick, st)].append(ch)
    seen_nicks = {nick for nick, _s in by_player_street}
    for nick in seen_nicks:
        s = players[nick]
        pre_chars = by_player_street.get((nick, 0), [])
        s["hands"] += 1
        if any(c in "bcrA" for c in pre_chars):
            s["vpip_n"] += 1
        s["vpip_d"] += 1
        if "r" in pre_chars:
            s["pfr_n"] += 1
        s["pfr_d"] += 1
        saw = nick in saw_flop_nicks
        if saw:
            s["saw_flop"] += 1
        won, n_cards = meta.get(nick, (None, 0))
        wtsd = n_cards == 2  # cards logged only at showdown
        if saw:
            s["wtsd_d"] += 1
            if wtsd:
                s["wtsd_n"] += 1
            s["wwsf_d"] += 1
            if won and won > 0:
                s["wwsf_n"] += 1
        if wtsd:
            s["wsd_d"] += 1
            if won and won > 0:
                s["wsd_n"] += 1
        # aggression + fold-to-bet + check-raise across all streets
        for st in range(4):
            chars = by_player_street.get((nick, st), [])
            s["bets"] += chars.count("b") + chars.count("A")
            s["raises"] += chars.count("r")
            s["calls"] += chars.count("c")
            # check-raise: checked then later raised same street
            if "k" in chars and "r" in chars and chars.index("k") < chars.index("r"):
                s["check_raise_n"] += 1
            if "k" in chars:
                s["check_raise_d"] += 1
        if won is not None:
            s["won_chips"] += won
    # fold-to-bet (fold-to-aggression): any decision facing >=1 aggressive action this street
    faced: dict[str, list[bool]] = defaultdict(list)  # nick -> [folded?]
    for nick, st, ch, ab in acts:
        if ab >= 1 and ch in ("f", "k", "c", "r", "b", "A"):
            faced[nick].append(ch == "f")
    for nick, outcomes in faced.items():
        s = players[nick]
        s["fold_to_bet_d"] += len(outcomes)
        s["fold_to_bet_n"] += sum(outcomes)


def _rate(s, key):
    d = s[f"{key}_d"]
    return (s[f"{key}_n"] / d) if d else None


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def detect_leaks(pop: dict) -> list[dict]:
    """Map population HUD distribution → concrete exploitable adjustments. Each rule cites the
    triggering stat (population median), the exploit, and a one-line EV rationale."""
    def med(k):
        m = pop.get(k, {})
        return m.get("p50")
    leaks = []
    ftc = med("fold_to_cbet")
    if ftc is not None and ftc < 0.45:
        leaks.append({"leak": "under-folds to flop c-bet", "stat": f"fold_to_cbet p50={ftc:.0%}",
                      "exploit": "value-bet flop relentlessly (cbet thin), but DON'T bluff-barrel a non-folder; "
                                 "size up turn/river — they arrive capped",
                      "ev": "MDF at 66% pot = 60%; if they fold <45% our value cbets print, and bluffs lose"})
    elif ftc is not None and ftc > 0.6:
        leaks.append({"leak": "over-folds to flop c-bet", "stat": f"fold_to_cbet p50={ftc:.0%}",
                      "exploit": "auto-profit cbet bluff any two on most textures",
                      "ev": "fold>60% beats the 0EV threshold for a 50%-pot bluff (need >33%)"})
    wtsd = med("wtsd")
    vpip = med("vpip")
    if wtsd is not None and wtsd > 0.32:
        leaks.append({"leak": "calls down too light (high WTSD)", "stat": f"wtsd p50={wtsd:.0%}",
                      "exploit": "thin-value bigger on turn/river; rarely bluff rivers",
                      "ev": "they call >MDF with worse — over-betting value extracts more per call"})
    wsd = med("wsd")
    if wsd is not None and wsd < 0.48:
        leaks.append({"leak": "wins <half at showdown (pays off value)", "stat": f"wsd p50={wsd:.0%}",
                      "exploit": "merge value range down (2nd pair/weak TP bets for value)",
                      "ev": "showdown range is weak -> our marginal made hands are ahead when called"})
    f3 = med("fold_to_three_bet")
    if f3 is not None and f3 > 0.62:
        leaks.append({"leak": "over-folds to 3-bet", "stat": f"fold_to_three_bet p50={f3:.0%}",
                      "exploit": "3-bet light for the steal (esp. vs late opens)",
                      "ev": "fold>62% -> a 3bet to ~3x risking R wins the pot >R/(R+pot) of the time"})
    tb = med("three_bet")
    if tb is not None and tb < 0.04:
        leaks.append({"leak": "almost never 3-bets", "stat": f"three_bet p50={tb:.0%}",
                      "exploit": "open wider / fold less to their rare 3bet (it's nutted)",
                      "ev": "their 3bet range is value-only -> tighten vs it, widen opens otherwise"})
    fb = med("fold_to_bet")
    if fb is not None and fb < 0.42:
        leaks.append({"leak": "sticky vs aggression (low fold-to-bet)", "stat": f"fold_to_bet p50={fb:.0%}",
                      "exploit": "cut bluff frequency hard; extract via larger value sizing",
                      "ev": "bluffs need fold>= bet/(bet+pot); a sub-42% folder makes most bluffs -EV"})
    cr = med("check_raise")
    if cr is not None and cr < 0.04:
        leaks.append({"leak": "never check-raises", "stat": f"check_raise p50={cr:.0%}",
                      "exploit": "bet strong hands OOP for value (no trap needed); barrel freely",
                      "ev": "no check-raise risk -> checking strong hands only forfeits value"})
    if vpip is not None and vpip > 0.40 and (med("pfr") or 0) < 0.14:
        leaks.append({"leak": "loose-passive limp/caller (big VPIP/PFR gap)",
                      "stat": f"vpip p50={vpip:.0%}, pfr p50={med('pfr'):.0%}",
                      "exploit": "iso-raise larger (4-5bb) to play HU vs the fish; never multiway-bluff",
                      "ev": "price out others, keep the station; HU value extraction is far higher-EV"})
    return leaks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tgz")
    ap.add_argument("--min-hands", type=int, default=100)
    ap.add_argument("--games", default="holdem,holdem1,holdem2,holdem3,holdemii,holdempot,nolimit")
    ap.add_argument("--max-archives", type=int, default=0, help="0=all; cap monthly archives for a quick run")
    ap.add_argument("--out", default="data/player_profiles/population_hud_stats.json")
    args = ap.parse_args()

    games = set(args.games.split(","))
    players: dict[str, dict] = defaultdict(_new_stat)
    n_arch = n_replayed = 0

    with tarfile.open(args.tgz, "r:gz") as outer:
        for im in outer:
            if not (im.isfile() and im.name.endswith(".tgz")):
                continue
            base = im.name.rsplit("/", 1)[-1]
            if base.split(".")[0] not in games:
                continue
            n_arch += 1
            if args.max_archives and n_arch > args.max_archives:
                break
            inner_f = outer.extractfile(im)
            if inner_f is None:
                continue
            boards: dict[str, list[str]] = {}
            hand_rows: dict[str, list] = defaultdict(list)
            hand_meta: dict[str, dict] = defaultdict(dict)
            try:
                with tarfile.open(fileobj=inner_f, mode="r:gz") as inner:
                    for m in inner:
                        mb = m.name.rsplit("/", 1)[-1]
                        if not m.isfile():
                            continue
                        f = inner.extractfile(m)
                        if f is None:
                            continue
                        if mb == "hdb":
                            for raw in f:
                                hb = parse_hdb_board(raw.decode("latin-1", "replace"))
                                if hb is not None:
                                    boards[hb[0]] = hb[1]
                        elif mb.startswith("pdb."):
                            for raw in f:
                                p = parse_pdb_line_full(raw.decode("latin-1", "replace"))
                                if p is None:
                                    continue
                                nick, ts, pos, streets, won, n_cards = p
                                hand_rows[ts].append((pos, nick, streets))
                                hand_meta[ts][nick] = (won, n_cards)
            except (tarfile.TarError, OSError, EOFError) as e:
                print(f"  skip remainder of {base}: {e}", flush=True)
            for ts, rows in hand_rows.items():
                process_hand(rows, boards.get(ts, []), hand_meta.get(ts, {}), players)
                n_replayed += 1
            if n_arch % 25 == 0:
                print(f"  ...{n_arch} archives, {n_replayed} hands, {len(players)} players", flush=True)

    # ---- per-player rates for qualified players, then population percentiles ----
    rate_keys = ("vpip", "pfr", "three_bet", "fold_to_three_bet", "cbet", "fold_to_cbet",
                 "wtsd", "wsd", "wwsf", "check_raise", "steal", "fold_to_steal", "fold_to_bet")
    qualified = []
    for nick, s in players.items():
        if s["hands"] < args.min_hands:
            continue
        af = (s["bets"] + s["raises"]) / max(s["calls"], 1)
        row = {"hands": s["hands"], "af": round(af, 3),
               "bb_won_per_hand": round(s["won_chips"] / max(s["hands"], 1), 2)}
        for k in rate_keys:
            r = _rate(s, k)
            row[k] = round(r, 4) if r is not None else None
        qualified.append(row)

    pop = {}
    for k in rate_keys + ("af",):
        xs = [r[k] for r in qualified if r.get(k) is not None]
        if xs:
            import statistics
            pop[k] = {"mean": round(statistics.mean(xs), 4),
                      "p10": round(_pct(xs, .1), 4), "p25": round(_pct(xs, .25), 4),
                      "p50": round(_pct(xs, .5), 4), "p75": round(_pct(xs, .75), 4),
                      "p90": round(_pct(xs, .9), 4), "n_players": len(xs)}
    leaks = detect_leaks(pop)

    payload = {
        "source": f"IRC poker DB HUD-suite fit ({n_replayed} hands, {len(players)} players, "
                  f"{len(qualified)} qualified >= {args.min_hands} hands, games={sorted(games)})",
        "caveats": ["IRC hold'em is mostly LIMIT: frequencies robust, bet-SIZE elasticity unobservable",
                    "1995-2001 field is looser/more passive than modern micro (directional upper bound)",
                    "fold_to_three_bet ~0 is a LIMIT artifact (a 3bet is one cheap bet -> nobody folds); "
                    "do NOT read it as 'never folds to 3bet' in NL — it is unobservable here",
                    "steal/fold_to_steal use pos-order button inference (lower confidence)"],
        "population": pop,
        "exploitable_leaks": leaks,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))

    # ---- printed report ----
    print("\n" + "=" * 72)
    print(f"POPULATION HUD STATS  ({len(qualified)} players >= {args.min_hands} hands, "
          f"{n_replayed} hands)")
    print("=" * 72)
    hdr = f"{'stat':<20}{'mean':>8}{'p10':>8}{'p25':>8}{'p50':>8}{'p75':>8}{'p90':>8}"
    print(hdr)
    for k in rate_keys + ("af",):
        if k not in pop:
            continue
        m = pop[k]
        def f(v):
            return f"{v:>8.1%}" if (k != "af" and v is not None) else (f"{v:>8.2f}" if v is not None else f"{'-':>8}")
        print(f"{k:<20}{f(m['mean'])}{f(m['p10'])}{f(m['p25'])}{f(m['p50'])}{f(m['p75'])}{f(m['p90'])}")
    print("\n" + "-" * 72)
    print(f"EXPLOITABLE LEAKS ({len(leaks)} detected)")
    print("-" * 72)
    for i, lk in enumerate(leaks, 1):
        print(f"{i}. {lk['leak'].upper()}  [{lk['stat']}]")
        print(f"     -> exploit: {lk['exploit']}")
        print(f"        EV: {lk['ev']}")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
