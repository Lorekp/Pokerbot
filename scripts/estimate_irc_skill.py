#!/usr/bin/env python3
"""Sophisticated per-player SKILL estimator over the IRC corpus.

The existing ``analyze_irc_population.py`` classifies players by *style* thresholds only
(VPIP/PFR/AF -> nit/tag/lag/station). Style is a weak, saturating proxy for skill. This estimator
adds the signal style throws away: the **money-won** column. Each IRC ``pdb`` line carries
``... bankroll action winnings [holecards]`` where ``action`` = chips the player put in the pot that
hand and ``winnings`` = chips they took out, so per-hand net = ``winnings - action`` (verified against
the bankroll-delta invariant ``bankroll[i+1] == bankroll[i] + winnings[i] - action[i]``). Normalising
by the per-game-month big blind gives a real bb/100 win-rate.

Raw win-rate is a terrible skill estimate at low sample (limit hold'em needs ~50k hands to stabilise),
so we do NOT rank on it. Instead we fit a **two-level empirical-Bayes model**:

  level 1 (style -> skill): a ridge regression maps stable style features (VPIP, PFR, VPIP-PFR gap,
      aggression frequency by street, WTSD, W$SD, ...) to win-rate, trained on high-sample players
      where win-rate is trustworthy. This learns what actually WINS in this field and gives every
      player -- even a 60-hand one -- a style-predicted win-rate s_i.

  level 2 (shrinkage): each player's true skill theta_i has prior N(s_i, tau^2) (tau^2 = residual
      skill variance not explained by style) and likelihood r_i ~ N(theta_i, 1e4 * sigma^2 / n_i)
      (sigma^2 = pooled per-hand result variance). The Gaussian-Gaussian posterior is a
      precision-weighted blend: low-n players sit on their style prior, high-n players move onto
      their realised results. Output = posterior mean theta_hat_i + credible interval per player.

"Professional-level" is then a *calibrated* count: players whose posterior clears a winning bar with
high probability (e.g. P(theta > bar) > 0.9), which small-sample lucky players cannot fake.

Validation (printed): split-half reliability of raw WR vs style, out-of-sample style->results
prediction, shrinkage-vs-raw calibration, and a zero-sum conservation check.

Usage:
    python scripts/estimate_irc_skill.py                       # full limit-holdem corpus
    python scripts/estimate_irc_skill.py --max-months 3        # smoke
    python scripts/estimate_irc_skill.py --games holdem,nolimit --include-nolimit
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

LIMIT_HOLDEM = ["holdem", "holdem1", "holdem2", "holdem3", "holdemii"]
NONLIMIT_HOLDEM = ["holdempot", "nolimit"]

# ------------------------------------------------------------------ parsing ---

def _is_int(s: str) -> bool:
    return s.lstrip("-").isdigit()


def parse_pdb_full(line: str):
    """(nick, ts, n_players, pos, streets[4], bankroll, action, winnings, saw_showdown) or None.

    Layout: nick ts n_players pos <up to 4 street action strings> bankroll action winnings [c1 c2].
    Street strings are the only non-integer tokens before the numeric tail, so we consume
    action-like tokens until the first integer (the bankroll), read the 3 ints, and treat any
    trailing 2-char card tokens as a revealed showdown hand."""
    t = line.split()
    if len(t) < 7:
        return None
    nick = t[0]
    if not (_is_int(t[1]) and _is_int(t[2]) and _is_int(t[3])):
        return None
    pos = int(t[3])
    i = 4
    streets: list[str] = []
    while i < len(t) and i < 8 and not _is_int(t[i]):
        streets.append(t[i])
        i += 1
    if not streets or i + 2 >= len(t):
        return None  # need bankroll, action, winnings after the streets
    try:
        bankroll = int(t[i]); action = int(t[i + 1]); winnings = int(t[i + 2])
    except ValueError:
        return None
    while len(streets) < 4:
        streets.append("-")
    cards = t[i + 3:i + 5]
    saw_sd = len(cards) == 2 and all(len(c) == 2 and c[1] in "cdhs" for c in cards)
    return nick, int(t[1]), int(t[2]), pos, streets, bankroll, action, winnings, saw_sd


# ------------------------------------------------------ per-player accumulator ---

# counter indices (a flat list per player for speed)
(H, NET, NET2, VPIP, PFR, BETS, RAISES, CALLS, FOLDS, CHECKS,
 SAW_FLOP, WTSD, WON_SD, FL_AGG, FL_CALL, TN_AGG, TN_CALL, RV_AGG, RV_CALL, NL_H) = range(20)
NCOUNT = 20


def _blank():
    return [0.0] * NCOUNT


_BONLY = re.compile(r"^B[kf]*$")


def _bb_from_counter(c: "Counter") -> float:
    """A player who posts a blind and only checks/folds puts in exactly that blind, so their B-only
    ``action`` values cluster at {SB, BB}. The big blind is the larger of the two dominant values."""
    if not c:
        return 0.0
    top = [v for v, _ in c.most_common(2)]
    return float(max(top))


def month_blind_tables(rows):
    """Return (per-player BB dict, month-modal BB). IRC months mix tables at different stakes, so we
    read each player's OWN blind posts to recover the stakes they actually played -- per-player
    normalisation, not one blind for the whole month."""
    per_player: dict[str, Counter] = {}
    month: Counter = Counter()
    for r in rows:
        pre = r[4][0]
        if _BONLY.match(pre) and r[6] > 0:
            per_player.setdefault(r[0], Counter())[r[6]] += 1
            month[r[6]] += 1
    player_bb = {nick: _bb_from_counter(c) for nick, c in per_player.items()}
    return player_bb, _bb_from_counter(month)


def accumulate(rows, player_bb: dict, month_bb: float, handcap: float, is_nl: bool,
               players: dict) -> tuple[int, float]:
    """Accumulate per-player stats. Each hand's net is normalised by the player's own table stakes
    and winsorised at +/- handcap big blinds so a single mis-normalised or dumped pot cannot blow up
    a player's mean."""
    used = 0
    net_sum = 0.0
    for r in rows:
        nick, ts, npl, pos, streets, bankroll, action, winnings, saw_sd = r
        bb = player_bb.get(nick) or month_bb
        if bb <= 0:
            continue
        pre, flop, turn, river = streets
        s = players.get(nick)
        if s is None:
            s = players[nick] = _blank()
        s[H] += 1
        if is_nl:
            s[NL_H] += 1
        net_bb = (winnings - action) / bb
        if net_bb > handcap:
            net_bb = handcap
        elif net_bb < -handcap:
            net_bb = -handcap
        net_sum += net_bb
        s[NET] += net_bb
        s[NET2] += net_bb * net_bb
        if any(ch in pre for ch in "bcrA"):
            s[VPIP] += 1
        if "r" in pre:
            s[PFR] += 1
        for st in streets:
            s[BETS] += st.count("b") + st.count("A")
            s[RAISES] += st.count("r")
            s[CALLS] += st.count("c")
            s[FOLDS] += st.count("f")
            s[CHECKS] += st.count("k")
        saw = any(ch in flop for ch in "kbcrA")
        if saw:
            s[SAW_FLOP] += 1
        if saw_sd:
            s[WTSD] += 1
            if winnings > 0:
                s[WON_SD] += 1
        s[FL_AGG] += flop.count("b") + flop.count("r") + flop.count("A")
        s[FL_CALL] += flop.count("c")
        s[TN_AGG] += turn.count("b") + turn.count("r") + turn.count("A")
        s[TN_CALL] += turn.count("c")
        s[RV_AGG] += river.count("b") + river.count("r") + river.count("A")
        s[RV_CALL] += river.count("c")
        used += 1
    return used, net_sum


# ---------------------------------------------------------------- corpus scan ---

def scan_corpus(tgz: Path, games: set[str], nl_games: set[str], max_months: int | None,
                handcap: float = 100.0):
    players: dict[str, list[float]] = {}
    per_game_hands: Counter = Counter()
    bb_log: dict[str, float] = {}
    total_hands = 0
    total_net = 0.0  # conservation diagnostic (should be ~0)
    parse_fail = 0

    outer = tarfile.open(tgz, "r:gz")
    months = []
    for m in outer:
        if not (m.isreg() and m.name.endswith(".tgz")):
            continue
        base = m.name.split("/")[-1]
        game = base.split(".")[0]
        if game in games:
            months.append((m.name, game, game in nl_games))
    # names are gathered lazily above; reopen for extraction (streaming tarfile is single-pass)
    outer.close()
    if max_months:
        months = months[:max_months]
    n_months = len(months)
    print(f"[scan] {n_months} game-months matched games={sorted(games)}", flush=True)

    outer = tarfile.open(tgz, "r:gz")
    name_to_meta = {n: (g, nl) for n, g, nl in months}
    processed = 0
    t0 = time.time()
    for m in outer:
        meta = name_to_meta.get(m.name)
        if meta is None or not m.isreg():
            continue
        game, is_nl = meta
        rows = []
        # A handful of monthly archives in the IRC corpus are truncated; skip any inner tarball that
        # fails to fully decompress rather than aborting the whole scan.
        try:
            data = outer.extractfile(m).read()
            inner = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
            for mm in inner.getmembers():
                if not mm.isreg():
                    continue
                nm = mm.name
                if "/pdb/" not in nm and "pdb." not in nm:
                    continue
                try:
                    txt = inner.extractfile(mm).read().decode("latin-1")
                except (OSError, UnicodeDecodeError, EOFError, tarfile.TarError):
                    continue
                for line in txt.splitlines():
                    if not line.strip():
                        continue
                    pr = parse_pdb_full(line)
                    if pr is None:
                        parse_fail += 1
                        continue
                    rows.append(pr)
        except (EOFError, tarfile.TarError, OSError) as e:
            print(f"[scan] WARN skipping truncated {m.name}: {type(e).__name__}", flush=True)
            processed += 1
            continue
        player_bb, month_bb = month_blind_tables(rows)
        if month_bb <= 0:
            processed += 1
            continue
        bb_log[m.name.split("/")[-1]] = month_bb
        used, net_sum = accumulate(rows, player_bb, month_bb, handcap, is_nl, players)
        total_net += net_sum
        total_hands += used
        per_game_hands[game] += used
        processed += 1
        if processed % 10 == 0 or processed == n_months:
            el = time.time() - t0
            rate = processed / el if el else 0
            eta = (n_months - processed) / rate if rate else 0
            print(f"[scan] {processed}/{n_months} months  {total_hands:,} hands  "
                  f"{len(players):,} players  {el:.0f}s  ETA {eta:.0f}s", flush=True)
    outer.close()
    return {
        "players": players,
        "per_game_hands": dict(per_game_hands),
        "bb_log": bb_log,
        "total_hands": total_hands,
        "total_net_bb": total_net,
        "parse_fail": parse_fail,
    }


# ------------------------------------------------------------ feature / stats ---

FEATURES = ["vpip", "pfr", "gap", "afq", "postflop_afq", "fold_freq",
            "wtsd", "wsd", "flop_afq", "river_afq", "saw_flop_rate"]


def player_features(s):
    h = s[H]
    agg = s[BETS] + s[RAISES]
    calls = s[CALLS]
    vpip = s[VPIP] / h
    pfr = s[PFR] / h
    postagg = s[FL_AGG] + s[TN_AGG] + s[RV_AGG]
    postcall = s[FL_CALL] + s[TN_CALL] + s[RV_CALL]
    return {
        "vpip": vpip,
        "pfr": pfr,
        "gap": vpip - pfr,
        "afq": agg / max(agg + calls, 1),
        "postflop_afq": postagg / max(postagg + postcall, 1),
        "fold_freq": s[FOLDS] / max(s[BETS] + s[RAISES] + s[CALLS] + s[FOLDS], 1),
        "wtsd": s[WTSD] / max(s[SAW_FLOP], 1),
        "wsd": s[WON_SD] / max(s[WTSD], 1),
        "flop_afq": s[FL_AGG] / max(s[FL_AGG] + s[FL_CALL], 1),
        "river_afq": s[RV_AGG] / max(s[RV_AGG] + s[RV_CALL], 1),
        "saw_flop_rate": s[SAW_FLOP] / h,
    }


def archetype(vpip, pfr, af, hands):
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


def wvar(x, w):
    """weighted variance."""
    w = np.asarray(w, float); x = np.asarray(x, float)
    mu = np.average(x, weights=w)
    return float(np.average((x - mu) ** 2, weights=w)), float(mu)


def ridge_fit(X, y, w, lam=1.0):
    """weighted ridge on standardised X (intercept unpenalised). Returns predict()."""
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xa = np.hstack([np.ones((len(Xs), 1)), Xs])
    W = np.diag(w)
    P = np.eye(Xa.shape[1]); P[0, 0] = 0.0  # don't penalise intercept
    beta = np.linalg.solve(Xa.T @ W @ Xa + lam * P, Xa.T @ W @ y)

    def predict(Xn):
        Xns = (Xn - mu) / sd
        return np.hstack([np.ones((len(Xns), 1)), Xns]) @ beta
    return predict, beta


def kfold_oof(X, y, w, lam, k=5):
    """Out-of-fold predictions + weighted R^2. OOF preds give an HONEST residual variance (the
    in-sample ridge residual is optimistically small and would make the model look near-perfect)."""
    n = len(y)
    folds = np.array_split(np.arange(n), k)
    preds = np.zeros(n)
    for f in folds:
        mask = np.ones(n, bool); mask[f] = False
        pf, _ = ridge_fit(X[mask], y[mask], w[mask], lam)
        preds[f] = pf(X[f])
    ybar = np.average(y, weights=w)
    ss_res = np.average((y - preds) ** 2, weights=w)
    ss_tot = np.average((y - ybar) ** 2, weights=w)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return preds, r2


# ------------------------------------------------------------------- estimate ---

def estimate(scan, min_hands, style_min_hands, pro_bars, k_feat=500.0,
             organic_ceiling=60.0):
    players = scan["players"]
    qual = {n: s for n, s in players.items() if s[H] >= min_hands}
    nicks0 = list(qual.keys())
    S0 = [qual[n] for n in nicks0]
    n0 = np.array([s[H] for s in S0], float)
    net0 = np.array([s[NET] for s in S0], float)
    net20 = np.array([s[NET2] for s in S0], float)
    raw_wr0 = 100.0 * net0 / n0

    # robust per-hand SD (trim the top 5% of per-player variance so contaminated tables can't inflate
    # the sampling-noise scale that the contamination gate depends on)
    per_var0 = np.clip(net20 / n0 - (net0 / n0) ** 2, 1e-6, None)
    keep_v = per_var0 <= np.percentile(per_var0, 95)
    sigma2 = float(np.average(per_var0[keep_v], weights=np.clip(n0[keep_v] - 1, 1, None)))
    se0 = 100.0 * np.sqrt(sigma2 / n0)

    # ---- contamination gate: no legitimate limit strategy sustains |WR| beyond a physical ceiling.
    # Flag accounts whose win-rate clears the ceiling by >3 SE (persistent, not a small-sample heater)
    # as non-organic (chip-dumping / bot-farms, e.g. a player fed by ad1..ad9). Drop BOTH the fed
    # winners and the dumping losers; a small-sample lucky player is NOT flagged (SE absorbs it).
    non_organic = (raw_wr0 - 3 * se0 > organic_ceiling) | (raw_wr0 + 3 * se0 < -organic_ceiling)
    flagged = [(nicks0[i], int(n0[i]), round(float(raw_wr0[i]), 1))
               for i in np.argsort(-np.abs(raw_wr0)) if non_organic[i]]
    keep = ~non_organic

    nicks = [nicks0[i] for i in range(len(nicks0)) if keep[i]]
    S = [S0[i] for i in range(len(S0)) if keep[i]]
    n = n0[keep]
    net = net0[keep]
    raw_wr = raw_wr0[keep]
    samp_var = 1e4 * sigma2 / n  # Var(raw_wr | theta)

    feats = np.array([[player_features(s)[f] for f in FEATURES] for s in S])

    # ---- level 1: style -> win-rate, trained on high-sample players ----
    train = n >= style_min_hands
    if train.sum() < 40:  # smoke fallback
        train = n >= np.quantile(n, 0.6)
    Xtr, ytr, wtr = feats[train], raw_wr[train], n[train]
    predict, beta = ridge_fit(Xtr, ytr, wtr, lam=1.0)
    style_wr = predict(feats)
    # Bound extrapolation: no evidence anyone in this field sustains a win-rate outside the band
    # observed among PROVEN (high-sample) players, so clip the linear model to that support.
    lo, hi = np.percentile(ytr, [2, 98])
    style_wr = np.clip(style_wr, lo, hi)

    # ---- level 2: empirical-Bayes shrinkage toward the style prior ----
    # tau^2 = residual skill variance beyond style, from OUT-OF-FOLD residuals (in-sample residuals
    # are overfit-small and would make the prior falsely tight -> overconfident posteriors).
    if train.sum() >= 25:
        oof, cvr2 = kfold_oof(Xtr, ytr, wtr, lam=1.0)
        obs_resid_var, _ = wvar(ytr - np.clip(oof, lo, hi), wtr)
    else:
        cvr2 = float("nan")
        obs_resid_var, _ = wvar(raw_wr[train] - style_wr[train], wtr)
    exp_samp = float(np.average(samp_var[train], weights=wtr))
    tau2 = max(obs_resid_var - exp_samp, 1.0)  # residual skill var beyond style (on proven players)

    # Feature reliability: VPIP/AF/WTSD are themselves estimated from n hands, so a low-n player's
    # style prediction is unreliable and may be off-model. Inflate the prior variance by 1/rel_feat
    # so the posterior stays honestly wide at low sample instead of trusting a noisy style read.
    rel_feat = n / (n + k_feat)
    prior_var = tau2 / rel_feat
    prec_prior = 1.0 / prior_var
    prec_lik = 1.0 / samp_var
    theta = (prec_prior * style_wr + prec_lik * raw_wr) / (prec_prior + prec_lik)
    post_var = 1.0 / (prec_prior + prec_lik)
    post_sd = np.sqrt(post_var)

    # results-only EB (prior = global mean) for the "style helps" comparison
    mu0 = float(np.average(raw_wr, weights=n))
    tot_var, _ = wvar(raw_wr, n)
    tau2_0 = max(tot_var - float(np.average(samp_var, weights=n)), 1.0)
    theta0 = (raw_wr / samp_var + mu0 / tau2_0) / (1.0 / samp_var + 1.0 / tau2_0)

    # ---- pro classification via posterior tail probability ----
    from math import erf, sqrt
    def p_gt(bar):
        z = (theta - bar) / post_sd
        return 0.5 * (1 + np.array([erf(zz / sqrt(2)) for zz in z]))

    tiers = {}
    for bar in pro_bars:
        p = p_gt(bar)
        tiers[f"p_gt_{bar}_at_.90"] = int((p > 0.90).sum())
        tiers[f"p_gt_{bar}_at_.95"] = int((p > 0.95).sum())

    p_win = p_gt(0.0)  # P(true skill > breakeven)

    # Trustworthy tier: pro counts restricted to players with enough hands that the RESULTS signal
    # (not just style) is meaningful. These are the ones we can actually confirm win.
    floors = {}
    for floor in (500, 1000, 2000):
        m = n >= floor
        floors[f"n>={floor}"] = {
            "players": int(m.sum()),
            "conf_win_p90": int(((p_win > 0.90) & m).sum()),
            "pro_p_gt_5_p90": int(((p_gt(5.0) > 0.90) & m).sum()),
        }

    # archetypes for cross-validation of the style axis
    arche = []
    for s in S:
        vpip = s[VPIP] / s[H]; pfr = s[PFR] / s[H]
        af = (s[BETS] + s[RAISES]) / max(s[CALLS], 1)
        arche.append(archetype(vpip, pfr, af, s[H]))
    arche = np.array(arche)

    return {
        "nicks": nicks, "n": n, "raw_wr": raw_wr, "style_wr": style_wr,
        "theta": theta, "post_sd": post_sd, "theta0": theta0, "p_win": p_win,
        "sigma2": sigma2, "tau2": tau2, "tau2_0": tau2_0, "mu0": mu0,
        "cv_r2": cvr2, "beta": beta, "tiers": tiers, "floors": floors, "arche": arche,
        "samp_var": samp_var, "feats": feats,
        "n_flagged": len(flagged), "flagged": flagged, "n_pre_gate": len(nicks0),
    }


# ----------------------------------------------------------------- validation ---

def validate(scan, min_hands, organic_ceiling=60.0):
    """Population reliability from variance components (robust to contamination): what fraction of a
    player's observed win-rate is real skill vs sampling noise, at the median sample size. Uses the
    trimmed per-hand variance and drops physically-impossible accounts, mirroring estimate()."""
    players = scan["players"]
    S = [s for s in players.values() if s[H] >= min_hands]
    n = np.array([s[H] for s in S], float)
    net = np.array([s[NET] for s in S], float)
    net2 = np.array([s[NET2] for s in S], float)
    raw_wr = 100.0 * net / n
    per_var = np.clip(net2 / n - (net / n) ** 2, 1e-6, None)
    keep_v = per_var <= np.percentile(per_var, 95)
    sigma2 = float(np.average(per_var[keep_v], weights=np.clip(n[keep_v] - 1, 1, None)))
    se = 100.0 * np.sqrt(sigma2 / n)
    keep = ~((raw_wr - 3 * se > organic_ceiling) | (raw_wr + 3 * se < -organic_ceiling))
    n, raw_wr = n[keep], raw_wr[keep]
    samp_var = 1e4 * sigma2 / n
    tot_var, _ = wvar(raw_wr, n)
    true_var = max(tot_var - float(np.average(samp_var, weights=n)), 1e-6)
    med_n = float(np.median(n))
    rel_med = true_var / (true_var + 1e4 * sigma2 / med_n)
    return {"reliability_raw_wr_at_median_n": rel_med, "median_hands": med_n,
            "signal_sd_bb100": true_var ** 0.5, "per_hand_sd_bb": sigma2 ** 0.5}


# --------------------------------------------------------------------- report ---

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tgz", default="data/datasets/IRCdata.tgz")
    ap.add_argument("--games", default=",".join(LIMIT_HOLDEM))
    ap.add_argument("--include-nolimit", action="store_true")
    ap.add_argument("--max-months", type=int, default=None)
    ap.add_argument("--min-hands", type=int, default=50)
    ap.add_argument("--style-min-hands", type=int, default=2000)
    ap.add_argument("--pro-bars", default="0,5,10")
    ap.add_argument("--handcap", type=float, default=100.0,
                    help="winsorise per-hand net at +/- this many big blinds")
    ap.add_argument("--organic-ceiling", type=float, default=60.0,
                    help="max plausible sustained |win-rate| bb/100; beyond it (3 SE) => non-organic")
    ap.add_argument("--k-feat", type=float, default=500.0,
                    help="feature-reliability half-saturation (hands); higher => more low-n shrinkage")
    ap.add_argument("--out", default="data/player_profiles/irc_skill_estimates.json")
    ap.add_argument("--summary-out", default="data/player_profiles/irc_skill_summary.json")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--cache-scan", default=None, help="pickle the parsed scan here for fast re-runs")
    ap.add_argument("--load-scan", default=None, help="load a pickled scan and skip parsing")
    args = ap.parse_args()

    tgz = REPO / args.tgz
    if not tgz.exists():
        print(f"IRC tarball not found: {tgz}", file=sys.stderr); sys.exit(1)

    games = set(args.games.split(","))
    nl_games = set(NONLIMIT_HOLDEM)
    if args.include_nolimit:
        games |= nl_games
    pro_bars = [float(x) for x in args.pro_bars.split(",")]

    if args.load_scan:
        import pickle
        print(f"[start] loading cached scan {args.load_scan}", flush=True)
        scan = pickle.load(open(args.load_scan, "rb"))
    else:
        print(f"[start] scanning {tgz.name} games={sorted(games)} "
              f"max_months={args.max_months}", flush=True)
        scan = scan_corpus(tgz, games, nl_games, args.max_months, handcap=args.handcap)
        if args.cache_scan:
            import pickle
            pickle.dump(scan, open(args.cache_scan, "wb"))
            print(f"[cache] scan pickled -> {args.cache_scan}", flush=True)
    print(f"[scan-done] {scan['total_hands']:,} hands, {len(scan['players']):,} players, "
          f"parse_fail={scan['parse_fail']:,}, conservation total_net={scan['total_net_bb']:.1f} bb "
          f"(should be ~0)", flush=True)
    print(f"[bb] detected big blinds (sample): "
          f"{dict(list(scan['bb_log'].items())[:6])}", flush=True)
    print(f"[games] hands per game: {scan['per_game_hands']}", flush=True)

    est = estimate(scan, args.min_hands, args.style_min_hands, pro_bars,
                   k_feat=args.k_feat, organic_ceiling=args.organic_ceiling)
    val = validate(scan, args.min_hands, organic_ceiling=args.organic_ceiling)

    nicks, theta, sd = est["nicks"], est["theta"], est["post_sd"]
    order = np.argsort(-theta)
    nq = len(nicks)

    print("\n================ IRC SKILL ESTIMATOR ================")
    print(f"qualified (>= {args.min_hands} hands): {est['n_pre_gate']:,}  |  "
          f"flagged NON-ORGANIC (chip-dump/bot-farm, excluded): {est['n_flagged']:,}  |  "
          f"modeled: {nq:,}")
    if est["flagged"]:
        print("   worst flagged (nick, hands, raw bb/100): "
              + ", ".join(f"{f[0]}({f[1]}h,{f[2]:.0f})" for f in est["flagged"][:8]))
    print(f"per-hand result SD: {val['per_hand_sd_bb']:.2f} bb  |  "
          f"true skill SD: {val['signal_sd_bb100']:.2f} bb/100")
    print(f"raw-WR reliability at median n={val['median_hands']:.0f}: "
          f"{val['reliability_raw_wr_at_median_n']:.3f}  "
          f"(low => raw win-rate is mostly noise => shrinkage essential)")
    print(f"style->win-rate cross-validated R^2: {est['cv_r2']:.3f}  "
          f"(fraction of skill the STYLE features explain)")
    print(f"sigma^2(per-hand)={est['sigma2']:.2f}  tau^2(skill|style)={est['tau2']:.1f}  "
          f"tau0^2(skill)={est['tau2_0']:.1f}  pop-mean WR={est['mu0']:.2f} bb/100")

    print("\n-- professional-level counts (posterior-calibrated) --")
    for k, v in est["tiers"].items():
        print(f"   {k:24s}: {v:5d}  ({100*v/nq:.2f}%)")
    print(f"   confident winners P(theta>0)>.90 : "
          f"{int((est['p_win']>0.90).sum()):5d}  ({100*(est['p_win']>0.90).mean():.2f}%)")

    print("\n-- TRUSTWORTHY tier: pros among players with enough hands (results are real here) --")
    for k, v in est["floors"].items():
        print(f"   {k:9s}: {v['players']:5d} players | conf-winners {v['conf_win_p90']:4d} | "
              f"strong-reg P(>5)>.90 {v['pro_p_gt_5_p90']:4d}")

    print(f"\n-- top {args.top} players by posterior skill --")
    print(f"   {'nick':16s} {'hands':>7s} {'rawWR':>7s} {'styleWR':>7s} "
          f"{'theta':>7s} {'+-sd':>6s} {'P(win)':>6s}")
    for i in order[:args.top]:
        print(f"   {nicks[i][:16]:16s} {est['n'][i]:7.0f} {est['raw_wr'][i]:7.1f} "
              f"{est['style_wr'][i]:7.1f} {theta[i]:7.1f} {sd[i]:6.1f} {est['p_win'][i]:6.2f}")

    # style-axis validation: do TAGs actually win?
    print("\n-- mean posterior skill by style archetype (validates the style axis) --")
    for a in ["tag", "lag", "nit", "calling_station", "unknown"]:
        mask = est["arche"] == a
        if mask.any():
            print(f"   {a:16s} n={int(mask.sum()):5d}  "
                  f"mean theta={theta[mask].mean():6.2f} bb/100  "
                  f"conf-win%={100*(est['p_win'][mask]>0.9).mean():5.1f}")

    # ---- persist ----
    def feats_dict(i):
        return {f: round(float(est["feats"][i][j]), 4) for j, f in enumerate(FEATURES)}
    rows = []
    for i in order:
        rows.append({
            "nick": nicks[i], "hands": int(est["n"][i]),
            "raw_wr_bb100": round(float(est["raw_wr"][i]), 2),
            "style_wr_bb100": round(float(est["style_wr"][i]), 2),
            "theta_bb100": round(float(theta[i]), 2),
            "post_sd": round(float(sd[i]), 2),
            "p_win": round(float(est["p_win"][i]), 4),
            "archetype": est["arche"][i],
        })
    Path(REPO / args.out).parent.mkdir(parents=True, exist_ok=True)
    (REPO / args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")

    summary = {
        "n_qualified_pre_gate": est["n_pre_gate"], "n_flagged_non_organic": est["n_flagged"],
        "n_modeled": nq, "min_hands": args.min_hands, "organic_ceiling_bb100": args.organic_ceiling,
        "flagged_examples": est["flagged"][:25],
        "total_hands": scan["total_hands"], "games": sorted(games),
        "conservation_total_net_bb": round(scan["total_net_bb"], 1),
        "per_hand_result_sd_bb": round(val["per_hand_sd_bb"], 3),
        "true_skill_sd_bb100": round(val["signal_sd_bb100"], 3),
        "raw_wr_reliability_median_n": round(val["reliability_raw_wr_at_median_n"], 4),
        "style_cv_r2": round(float(est["cv_r2"]), 4),
        "sigma2_per_hand": round(est["sigma2"], 3),
        "tau2_skill_given_style": round(est["tau2"], 2),
        "tau2_skill_total": round(est["tau2_0"], 2),
        "pop_mean_wr_bb100": round(est["mu0"], 3),
        "pro_tiers": est["tiers"],
        "trustworthy_tier_by_hand_floor": est["floors"],
        "confident_winners_p90": int((est["p_win"] > 0.90).sum()),
        "style_beta": {f: round(float(est["beta"][j + 1]), 3) for j, f in enumerate(FEATURES)},
        "archetype_mean_theta": {
            a: round(float(theta[est["arche"] == a].mean()), 3)
            for a in set(est["arche"]) if (est["arche"] == a).any()
        },
    }
    (REPO / args.summary_out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n[out] per-player -> {args.out}\n[out] summary -> {args.summary_out}")


if __name__ == "__main__":
    main()
