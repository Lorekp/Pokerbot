#!/usr/bin/env python3
"""Stream-parse the IRC poker database (Maurer format) → population priors for opponent exploitation.

Reads the ~1 GB ``IRCdata.tgz`` tarball member-by-member WITHOUT extracting (no small-file disk
explosion). TWO analysis layers:

1. GLOBAL (original): per-player VPIP/PFR/aggression/fold from ``pdb`` files alone → population
   distribution + archetype-threshold validation. Written to --summary-out.
2. PER-CONTEXT (Phase 3): joins ``hdb`` (board cards) with every player's per-street action strings,
   replays each hand round-robin in position order, and recovers the EXACT locked 22-context key for
   every voluntary action via ``pokerbot.opponent.context.context_key`` (zero spec skew by
   construction). Output: FITTED per-context Dirichlet pseudocounts in the exact shape
   ``population_prior.load_prior`` consumes (--prior-out) + per-player percentile tables
   (--percentiles-out) for the fitted eval-bot gauntlet.

SIZE CAVEAT: IRC hold'em is mostly LIMIT — ``b``/``r`` carry no amount, so they map to AGG_SMALL and
``A`` (all-in) to AGG_LARGE for every game type; the small/large split therefore under-represents
big sizings. Recorded in the prior file's meta.

PDB line: nick ts n_players pos <preflop> [<flop> <turn> <river>] bankroll wagered won [cards...]
HDB line: ts gameset# game# #players flop(#p/$) turn(#p/$) river(#p/$) showdn(#p/$) board...
Action codes: B=blind f=fold k=check b=bet c=call r=raise A=allin -=no-action Q=quit K=kicked
Run: python scripts/analyze_irc_population.py /path/to/IRCdata.tgz [--min-hands 50]
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
# Our archetype thresholds (mirror pokerbot/opponent/stats.py OpponentStats.archetype()).
def archetype(vpip: float, pfr: float, af: float, hands: int) -> str:
    if hands < 5:
        return "unknown"
    if vpip < 0.15 and pfr < 0.12:
        return "nit"
    if vpip > 0.45 and af < 1.5:
        return "calling_station"
    if vpip > 0.35 and af > 2.5:
        return "lag"
    if 0.18 <= vpip <= 0.28 and pfr >= 0.15:
        return "tag"
    return "unknown"


def parse_pdb_line(line: str):
    """Return (nick, [action_strings]) or None. Splits action strings from the numeric tail robustly."""
    t = line.split()
    if len(t) < 6:
        return None
    nick = t[0]
    acts = []
    i = 4  # fields 0..3 = nick, ts, n_players, pos
    while i < len(t) and _ACTION_RE.match(t[i]) and not _INT_RE.match(t[i]):
        acts.append(t[i])
        i += 1
    if not acts:
        return None
    return nick, acts


def accumulate(acts: list[str], st: dict) -> None:
    pre = acts[0]
    st["hands"] += 1
    # VPIP: voluntary preflop money (exclude the forced blind 'B').
    if any(c in pre for c in "bcrA"):
        st["vpip"] += 1
    if "r" in pre:
        st["pfr"] += 1
    for a in acts:
        st["bets"] += a.count("b") + a.count("A")
        st["raises"] += a.count("r")
        st["calls"] += a.count("c")
        st["folds"] += a.count("f")


def parse_pdb_line_full(line: str):
    """(nick, ts, pos, [street action strings]) or None — the per-context replay needs position
    and the per-street split, not just the flattened action list."""
    t = line.split()
    if len(t) < 7:
        return None
    nick, ts = t[0], t[1]
    if not (_INT_RE.match(t[2]) and _INT_RE.match(t[3])):
        return None
    pos = int(t[3])
    streets = []
    i = 4
    while i < len(t) and len(streets) < 4 and _ACTION_RE.match(t[i]) and not _INT_RE.match(t[i]):
        streets.append(t[i])
        i += 1
    if not streets:
        return None
    return nick, ts, pos, streets


_STREET_BOARD_N = (0, 3, 4, 5)
_CHAR_BIN = {"f": 0, "k": 1, "c": 1, "b": 2, "r": 2, "A": 3}  # limit data: b/r=small, all-in=large
_CHAR_AGGR = frozenset("brA")


def replay_hand_contexts(rows: list[tuple[int, str, list[str]]], board: list[str]):
    """Replay ONE hand's betting round-robin in position order; yield (nick, context_key, bin) for
    every voluntary decision. ``rows`` = [(pos, nick, street_strings)]; ``board`` = card strings.

    The k-th character of a player's street string is their k-th action that street, and betting
    cycles the table in seat order — so consuming character k of every live player in position
    order reconstructs the true sequence (blinds 'B' appear as leading preflop chars, keeping the
    alignment). Folds/quits drop a player from later rounds; '-' = no action. The running
    aggression count before each action is the locked spec's facing depth."""
    from pokerbot.opponent.context import context_key

    rows = sorted(rows, key=lambda r: r[0])
    out_players = set()  # folded/quit
    for st_idx in range(4):
        live = [(pos, nick, streets[st_idx]) for pos, nick, streets in rows
                if st_idx < len(streets) and nick not in out_players]
        if not live:
            continue
        bslice = board[: _STREET_BOARD_N[st_idx]]
        if len(bslice) < _STREET_BOARD_N[st_idx]:
            break  # board ended (no flop dealt etc.)
        aggr = 0
        maxlen = max(len(s) for _, _, s in live)
        for k in range(maxlen):
            for pos, nick, s in live:
                if nick in out_players or k >= len(s):
                    continue
                ch = s[k]
                if ch in ("Q", "K"):
                    out_players.add(nick)
                    continue
                if ch in ("B", "-"):
                    continue
                b = _CHAR_BIN.get(ch)
                if b is None:
                    continue
                yield nick, context_key(st_idx, bslice, aggr), b
                if ch in _CHAR_AGGR:
                    aggr += 1
                if ch == "f":
                    out_players.add(nick)


def parse_hdb_board(line: str) -> tuple[str, list[str]] | None:
    """(ts, [board card strings]) from an hdb line; board may be empty (no flop)."""
    t = line.split()
    if len(t) < 8 or not _INT_RE.match(t[0]):
        return None
    board = [c for c in t[8:] if len(c) == 2 and not _INT_RE.match(c)]
    return t[0], board


def fit_prior(ctx_player: dict, ctx_global: dict, strength: float,
              min_obs_per_player: int = 20, min_players: int = 30):
    """Per-context Dirichlet pseudocounts + percentile tables from the per-player frequencies.

    Robust fit: per-bin MEDIAN over qualifying players (>= min_obs_per_player observations at the
    context), renormalized — volume players can't drag the prior. Contexts with too few qualifying
    players fall back to the action-weighted aggregate."""
    from pokerbot.opponent.context import NUM_BINS, all_contexts

    prior, percentiles = {}, {}
    for ck in all_contexts():
        players = ctx_player.get(ck, {})
        freqs = []
        for counts in players.values():
            n = sum(counts)
            if n >= min_obs_per_player:
                freqs.append([c / n for c in counts])
        if len(freqs) >= min_players:
            med = [sorted(f[b] for f in freqs)[len(freqs) // 2] for b in range(NUM_BINS)]
            tot = sum(med) or 1.0
            dist = [m / tot for m in med]
            src = f"median of {len(freqs)} players"
        else:
            g = ctx_global.get(ck, [0] * NUM_BINS)
            tot = sum(g)
            dist = [c / tot for c in g] if tot else [1.0 / NUM_BINS] * NUM_BINS
            src = f"aggregate ({int(tot)} actions, {len(freqs)} qualifying players)"
        # FLOOR each bin at 2%: limit-era data fits agg_large to exactly 0 (no per-action sizes,
        # all-ins rare), and a ZERO Dirichlet pseudocount lets one observed action zero-collapse /
        # TV-explode the posterior. A small floor keeps every bin live without moving the medians.
        dist = [max(d, 0.02) for d in dist]
        tot = sum(dist)
        dist = [d / tot for d in dist]
        prior[ck] = [round(strength * d, 4) for d in dist]
        if freqs:
            def pct(b, p):
                xs = sorted(f[b] for f in freqs)
                return round(xs[min(len(xs) - 1, int(p * len(xs)))], 4)
            percentiles[ck] = {
                "n_players": len(freqs),
                "bins": {name: {f"p{int(p * 100)}": pct(b, p) for p in (0.1, 0.25, 0.5, 0.75, 0.9)}
                         for b, name in enumerate(("fold", "passive", "agg_small", "agg_large"))},
                "fit_source": src,
            }
    return prior, percentiles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tgz")
    ap.add_argument("--min-hands", type=int, default=50)
    ap.add_argument("--summary-out", "--out", dest="summary_out",
                    default="data/player_profiles/irc_population_summary.json",
                    help="global VPIP/PFR/AF summary (NO LONGER the prior file — see --prior-out)")
    ap.add_argument("--prior-out", default="data/player_profiles/irc_population_prior.json",
                    help="fitted per-context Dirichlet pseudocounts (population_prior.load_prior shape)")
    ap.add_argument("--percentiles-out", default="data/player_profiles/irc_population_percentiles.json")
    ap.add_argument("--games", default="holdem,holdem1,holdem2,holdem3,holdemii,holdempot,nolimit",
                    help="comma list of IRC game-type prefixes to include (hold'em betting family)")
    ap.add_argument("--max-archives", type=int, default=0, help="0=all; else cap inner monthly archives")
    args = ap.parse_args()

    from pokerbot.opponent.context import NUM_BINS, context_spec_hash
    from pokerbot.opponent.population_prior import PRIOR_STRENGTH

    games = set(args.games.split(","))
    players: dict[str, dict] = defaultdict(lambda: dict(hands=0, vpip=0, pfr=0, bets=0, raises=0, calls=0, folds=0))
    ctx_global: dict[str, list[int]] = defaultdict(lambda: [0] * NUM_BINS)
    ctx_player: dict[str, dict[str, list[int]]] = defaultdict(dict)
    n_arch = n_pdb = n_lines = n_replayed = 0

    # The outer tgz holds IRCdata/<game>.<yearmonth>.tgz inner tarballs; each inner tarball holds the
    # per-month hdb/hroster + pdb/pdb.<player> files. Stream both levels, never extracting to disk.
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
            hand_rows: dict[str, list[tuple[int, str, list[str]]]] = defaultdict(list)
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
                            n_pdb += 1
                            for raw in f:
                                n_lines += 1
                                line = raw.decode("latin-1", "replace")
                                p = parse_pdb_line(line)
                                if p is not None:
                                    accumulate(p[1], players[p[0]])
                                pf = parse_pdb_line_full(line)
                                if pf is not None:
                                    nick, ts, pos, streets = pf
                                    hand_rows[ts].append((pos, nick, streets))
            except (tarfile.TarError, OSError, EOFError) as e:
                # EOFError: a truncated inner gzip (the 199808 archives are known-corrupt in the
                # distributed tarball) ends mid-member — keep whatever parsed, skip the rest.
                print(f"  skip remainder of {base}: {e}", flush=True)
            # join + replay this month's hands (board known ⇒ exact texture; missing ts ⇒ preflop-only)
            for ts, rows in hand_rows.items():
                board = boards.get(ts, [])
                n_replayed += 1
                for nick, ck, b in replay_hand_contexts(rows, board):
                    ctx_global[ck][b] += 1
                    prow = ctx_player[ck].get(nick)
                    if prow is None:
                        prow = ctx_player[ck][nick] = [0] * NUM_BINS
                    prow[b] += 1
            if n_arch % 25 == 0:
                print(f"  ...{n_arch} archives, {n_pdb} player files, {n_lines} hands, "
                      f"{n_replayed} replayed, {len(players)} players", flush=True)

    # Per-player derived stats for players with enough sample.
    rows = []
    for nick, s in players.items():
        h = s["hands"]
        if h < args.min_hands:
            continue
        vpip = s["vpip"] / h
        pfr = s["pfr"] / h
        af = (s["bets"] + s["raises"]) / max(s["calls"], 1)
        fold = s["folds"] / max(s["bets"] + s["raises"] + s["calls"] + s["folds"], 1)
        rows.append((vpip, pfr, af, fold, h, archetype(vpip, pfr, af, h)))

    import statistics as st
    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0
    vpips = [r[0] for r in rows]; pfrs = [r[1] for r in rows]; afs = [r[2] for r in rows]; folds = [r[3] for r in rows]
    arche = defaultdict(int)
    for r in rows:
        arche[r[5]] += 1
    n = len(rows)
    summary = {
        "n_players_total": len(players),
        "n_players_qualified": n,
        "min_hands": args.min_hands,
        "total_hands_parsed": n_lines,
        "population_prior": {
            "vpip": {"mean": st.mean(vpips) if vpips else 0, "p10": pct(vpips, .1), "p50": pct(vpips, .5), "p90": pct(vpips, .9)},
            "pfr": {"mean": st.mean(pfrs) if pfrs else 0, "p10": pct(pfrs, .1), "p50": pct(pfrs, .5), "p90": pct(pfrs, .9)},
            "aggression_factor": {"mean": st.mean(afs) if afs else 0, "p10": pct(afs, .1), "p50": pct(afs, .5), "p90": pct(afs, .9)},
            "fold_freq": {"mean": st.mean(folds) if folds else 0, "p50": pct(folds, .5), "p90": pct(folds, .9)},
        },
        "archetype_distribution": {k: {"count": v, "pct": round(100 * v / max(n, 1), 1)} for k, v in sorted(arche.items(), key=lambda x: -x[1])},
    }
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nwrote {args.summary_out}", flush=True)

    # ---- Phase 3: fitted per-context prior + percentiles (the load_prior-consumable artifacts) ----
    prior, percentiles = fit_prior(ctx_player, ctx_global, PRIOR_STRENGTH)
    prior_payload = {
        "source": f"IRC empirical per-context fit ({n_replayed} hands replayed, "
                  f"{len(players)} players, games={sorted(games)})",
        "context_spec_hash": context_spec_hash(),
        "prior_strength": PRIOR_STRENGTH,
        "bins": ["fold", "passive", "agg_small", "agg_large"],
        "size_split_caveat": "IRC hold'em is mostly LIMIT: b/r have no amount and map to agg_small; "
                             "A (all-in) to agg_large — big sizings under-represented.",
        "prior": prior,
    }
    Path(args.prior_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.prior_out).write_text(json.dumps(prior_payload, indent=1))
    Path(args.percentiles_out).write_text(json.dumps(
        {"context_spec_hash": context_spec_hash(), "percentiles": percentiles}, indent=1))
    print(f"wrote {args.prior_out} ({len(prior)} contexts) and {args.percentiles_out}", flush=True)


if __name__ == "__main__":
    main()
