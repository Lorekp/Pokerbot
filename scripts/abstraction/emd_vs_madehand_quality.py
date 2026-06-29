"""Validation harness: PROVE the potential-aware EMD postflop abstraction is strictly better than
the made-hand-only abstraction on the two ULTRACODE Workstream-F acceptance properties.

  (1) draws separated from made hands of equal CURRENT rank, and
  (2) EMD vs made-hand-only shows a measurable thin-value EV gain.

This is a NEW, READ-ONLY harness over `pokerbot.encoding.postflop_bucketer`. It does not modify the
bucketer, the engines, the action menu, or any native crate file. It prefers the native
`battery_native` EMD kernel for the Monte-Carlo histogram sampling (much faster) and falls back to the
pure-Python `strength_histogram` if native isn't importable.

---------------------------------------------------------------------------------------------------
DELIVERABLE 1 -- draw/made separation at EQUAL current rank
---------------------------------------------------------------------------------------------------
We sample many legal FLOP and TURN spots (RIVER is EXCLUDED from this metric: a 5-card board has no
runouts, so `strength_histogram` collapses to a single spike at the made-hand percentile and EMD is
by construction == made-hand bucketing there -- there is no potential left to separate). We group
spots by `fast_made_hand_bucket` (made-hand-only -- ~equal CURRENT rank) and measure how differently
the EMD table treats spots that share a made-hand bucket:

  - A1 = fraction of equal-made-rank spot PAIRS that EMD assigns to DIFFERENT buckets.
  - A2 = mean EMD distance (_emd_1d) between the strength histograms of equal-made-rank pairs.
  - A worked example: a made hand (e.g. two pair / middle pair) and a strong DRAW (flush + straight
    draw) that share a made-hand bucket but get different EMD buckets, with their EMD distance.

---------------------------------------------------------------------------------------------------
DELIVERABLE 2 -- measurable thin-value EV gain (the load-bearing metric)
---------------------------------------------------------------------------------------------------
EV functional (stated precisely so it can be reviewed):

  Spot: a TURN decision. Hero either makes a THIN VALUE BET or CHECKs. Hero's value depends ONLY on the
  river-strength DISTRIBUTION = `strength_histogram` (the potential the made-hand bucket throws away).

  Villain CALLS the bet with a fixed MEDIUM strength BAND [CALL_LO, CALL_HI] of river strength (a thin
  value bet targets a calling range of MEDIANS -- the nuts would raise, trash folds). Conditional on a
  call, villain's river strength is ~Uniform[CALL_LO, CALL_HI], so hero of realized river percentile p
  wins the showdown with probability
      wc(p) = clip((p - CALL_LO) / (CALL_HI - CALL_LO), 0, 1).
  Call frequency CF = CALL_HI - CALL_LO (the band width). Bet size B*POT. The value of betting is the
  EXTRA chips moved against the calling band; CHECKING gives up the bet (baseline 0). Per realized
  river percentile p:
      EV_bet(p)   = CF * B*POT * (2 * wc(p) - 1)     # +B when ahead of the called villain, -B when behind
      EV_check(p) = 0                                 # give up the thin value option

  This payoff is CONVEX / NUT-PEAKED in river strength: it pays only when hero lands ABOVE the calling
  band, which DRAWS reach (top histogram bins) but stagnant made hands of the same current rank do NOT.
  So the optimal action is NOT monotone in current made-hand rank: a drawing hand (low current rank,
  high river potential) should BET while a marginal made hand stuck inside the calling band should
  CHECK. That order REVERSAL is the thin-value distinction the made-hand-only bucket -- which orders by
  current rank -- structurally cannot make, and which EMD (clustering by histogram SHAPE) can.

  EV_a(h) = E_{p ~ hist(h)}[ EV_a(p) ] for a in {bet, check}.

  Abstraction EV under a bucketing B (you are forced to play ONE action per bucket):
      abstraction_EV(B) = mean_h [ EV_{a*(bucket(h))}(h) ],
      where a*(b) = argmax_{a in {bet,check}} mean_{h in b}[ EV_a(h) ]      (best single action per bucket)
  The unabstracted optimum:
      EV_optimal = mean_h [ max_a EV_a(h) ].
  (Both are per-hand averages -> directly comparable; gap-to-optimal measures the abstraction's EV
  loss. EMD recovers EV the made-hand bucket cannot because it groups by histogram shape, so hands
  that should bet are not pooled with hands that should check. The EMD table for D2 is fit on the SAME
  TURN-only distribution it is evaluated on, the honest apples-to-apples setup for a turn decision.)

Run:
    python scripts/abstraction/emd_vs_madehand_quality.py [--samples N]
Writes artifacts/emd_quality_report.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pokerbot.encoding.postflop_bucketer import (
    EMDBucketer,
    _emd_1d,
    card_id_to_str,
    fast_made_hand_bucket,
    strength_histogram,
)

# --------------------------------------------------------------------------------------------------
# Native kernel (optional but preferred for sampling speed).
# --------------------------------------------------------------------------------------------------
def _load_native():
    try:
        from pokerbot.search import native_solver

        ext = native_solver._try_load()
        if ext is not None and hasattr(ext, "emd_strength_histograms_batch"):
            return ext
    except Exception:  # noqa: BLE001
        pass
    return None


_EXT = _load_native()


# --------------------------------------------------------------------------------------------------
# Spot sampling.
# --------------------------------------------------------------------------------------------------
def sample_legal_spot(rng: np.random.Generator, *, streets=(3, 4)):
    """A random legal postflop spot: 2 hole + (3 flop | 4 turn) board, all distinct. RIVER (5) is
    deliberately excluded from the potential-aware metrics (no runouts -> EMD == made-hand there)."""
    n_board = int(rng.choice(streets))
    cards = rng.choice(52, size=2 + n_board, replace=False)
    hole = (int(cards[0]), int(cards[1]))
    board = tuple(int(c) for c in cards[2:])
    return hole, board


def histograms_for_spots(spots, *, n_bins, n_runouts, seed_base):
    """(N, n_bins) strength histograms for a list of (hole, board) spots. Native batch if available,
    else the Python oracle. Both are normalized rows; either is a valid sample of the same runout
    distribution (native uses a fast RNG -> statistically equivalent at these n_runouts)."""
    if _EXT is not None:
        nat_spots = [([int(c) for c in h], [int(c) for c in b]) for (h, b) in spots]
        out = _EXT.emd_strength_histograms_batch(nat_spots, int(n_bins), int(n_runouts), int(seed_base))
        return np.asarray(out, dtype=np.float64)
    rows = [
        strength_histogram(h, b, n_bins=n_bins, n_runouts=n_runouts, seed=seed_base + i)
        for i, (h, b) in enumerate(spots)
    ]
    return np.asarray(rows, dtype=np.float64)


# --------------------------------------------------------------------------------------------------
# DELIVERABLE 1 -- draw/made separation at EQUAL current rank.
# --------------------------------------------------------------------------------------------------
def deliverable1(rng, *, n_spots, n_made_buckets, n_emd_buckets, n_bins, n_runouts, fit_sample, max_pairs):
    """Returns a dict with A1, A2, per-bucket detail, and the worked example."""
    # 1. Fit the EMD table over a representative sample of flop+turn histograms.
    fit_spots = [sample_legal_spot(rng) for _ in range(fit_sample)]
    fit_hists = histograms_for_spots(fit_spots, n_bins=n_bins, n_runouts=n_runouts, seed_base=10_000)
    emd = EMDBucketer(num_buckets=n_emd_buckets, n_bins=n_bins, n_runouts=n_runouts)
    emd.fit(fit_hists, iters=12, seed=0)

    # 2. Sample evaluation spots; compute made-hand bucket, EMD bucket, and histogram for each.
    spots = [sample_legal_spot(rng) for _ in range(n_spots)]
    hists = histograms_for_spots(spots, n_bins=n_bins, n_runouts=n_runouts, seed_base=20_000)
    made = np.array([fast_made_hand_bucket(h, b, n_made_buckets) for (h, b) in spots], dtype=int)
    # EMD bucket from the SAME histograms we already sampled (nearest centroid), so D1 and D2 share data.
    cc = np.cumsum(emd.centroids, axis=1)
    Hc = np.cumsum(hists, axis=1)
    emd_b = np.abs(Hc[:, None, :] - cc[None, :, :]).sum(axis=2).argmin(axis=1)

    # 3. Within each made-hand bucket, enumerate pairs (capped) and measure A1/A2.
    diff_pairs = 0
    total_pairs = 0
    emd_dists = []
    pair_rng = np.random.default_rng(424242)
    by_made: dict[int, list[int]] = {}
    for i, m in enumerate(made):
        by_made.setdefault(int(m), []).append(i)

    for m, idxs in by_made.items():
        if len(idxs) < 2:
            continue
        idxs = np.array(idxs)
        # Enumerate all pairs if few; else sample up to max_pairs random pairs without replacement.
        npairs_full = len(idxs) * (len(idxs) - 1) // 2
        if npairs_full <= max_pairs:
            pairs = [(idxs[a], idxs[b]) for a in range(len(idxs)) for b in range(a + 1, len(idxs))]
        else:
            pairs = []
            seen = set()
            tries = 0
            while len(pairs) < max_pairs and tries < max_pairs * 8:
                a, b = pair_rng.choice(len(idxs), size=2, replace=False)
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    pairs.append((idxs[a], idxs[b]))
                tries += 1
        for ia, ib in pairs:
            total_pairs += 1
            if emd_b[ia] != emd_b[ib]:
                diff_pairs += 1
            emd_dists.append(_emd_1d(hists[ia], hists[ib]))

    a1 = diff_pairs / total_pairs if total_pairs else 0.0
    a2 = float(np.mean(emd_dists)) if emd_dists else 0.0

    # 4. Worked example: search for a clear made-hand vs strong-draw pair that shares a made-hand
    #    bucket but lands in different EMD buckets. We hand-construct two spots that we KNOW are a
    #    made two-pair and a big combo draw of similar current rank, then verify.
    worked = _worked_example(emd, n_made_buckets, n_bins, n_runouts)

    return {
        "A1_separation_fraction": a1,
        "A2_mean_emd_distance": a2,
        "n_spots": n_spots,
        "n_made_buckets": n_made_buckets,
        "n_emd_buckets": n_emd_buckets,
        "n_pairs_compared": total_pairs,
        "n_made_buckets_populated": int(sum(1 for v in by_made.values() if len(v) >= 2)),
        "worked_example": worked,
    }


# Card-id helper: rank 0..12 (2..A), suit 0..3 (c,d,h,s). card_id = rank*4 + suit.
def _cid(rank: int, suit: int) -> int:
    return rank * 4 + suit


def _worked_example(emd: EMDBucketer, n_made_buckets, n_bins, n_runouts):
    """Construct a MADE hand and a big DRAW that share a made-hand bucket but differ under EMD.

    MADE: two pair on a dry, rainbow board, no draws (e.g. 9c9d on Ah9h... no -> keep it dry).
      Hero 8c 7c? we want pure made. Use 9s 9c (set of nines) on Ad 9h 4c board... that is very strong.
    DRAW: a flush + open-ended straight draw with NO made hand yet (air now, big river upside).

    We try a small handful of hand-built candidates and pick the first pair sharing a made-hand
    bucket but in different EMD buckets, with a sizeable histogram EMD distance. If none of the
    hand-built candidates collide on the made bucket, we fall back to a random search over sampled
    spots for the same property.
    """
    ranks = "23456789TJQKA"  # index 0..12
    suits = "cdhs"

    # Candidate MADE hands (decent but capped current rank, little/no draw potential): middle/second
    # pair or two pair on dryish boards.
    made_candidates = [
        # second pair, no draw: 8 7 offsuit-ish on K 8 3 rainbow
        ((_cid(6, 0), _cid(5, 1)), (_cid(11, 2), _cid(6, 3), _cid(1, 0))),
        # middle pair Tens-ish: T 9 on Q T 4 rainbow
        ((_cid(8, 0), _cid(7, 1)), (_cid(10, 2), _cid(8, 3), _cid(2, 0))),
        # bottom two pair: 6 5 on 6 5 K rainbow
        ((_cid(4, 0), _cid(3, 1)), (_cid(4, 2), _cid(3, 3), _cid(11, 0))),
    ]
    # Candidate DRAWS (no made hand, big combo draw): suited connectors on a two-tone connected board.
    draw_candidates = [
        # 9h 8h on 7h 6h 2c: flush draw + open-ended straight draw (monster draw, no made hand)
        ((_cid(7, 2), _cid(6, 2)), (_cid(5, 2), _cid(4, 2), _cid(0, 0))),
        # Jh Th on 9h 8c 2h: flush draw + OESD
        ((_cid(9, 2), _cid(8, 2)), (_cid(7, 2), _cid(6, 0), _cid(0, 2))),
        # Kh Qh on Jh Th 3c: flush draw + OESD (with overcards)
        ((_cid(11, 2), _cid(10, 2)), (_cid(9, 2), _cid(8, 2), _cid(1, 0))),
    ]

    def _describe(hole, board):
        return {
            "hole": [card_id_to_str(c) for c in hole],
            "board": [card_id_to_str(c) for c in board],
            "made_bucket": fast_made_hand_bucket(hole, board, n_made_buckets),
            "emd_bucket": emd.bucket(hole, board),
        }

    for made in made_candidates:
        md = _describe(*made)
        for draw in draw_candidates:
            dd = _describe(*draw)
            if md["made_bucket"] == dd["made_bucket"] and md["emd_bucket"] != dd["emd_bucket"]:
                hist_made = strength_histogram(made[0], made[1], n_bins=n_bins, n_runouts=n_runouts, seed=1)
                hist_draw = strength_histogram(draw[0], draw[1], n_bins=n_bins, n_runouts=n_runouts, seed=2)
                return {
                    "found": True,
                    "source": "hand_built",
                    "made_hand": {**md, "histogram": hist_made.round(4).tolist()},
                    "draw_hand": {**dd, "histogram": hist_draw.round(4).tolist()},
                    "shared_made_bucket": md["made_bucket"],
                    "emd_distance": _emd_1d(hist_made, hist_draw),
                }

    # Fallback: random search for any equal-made-bucket pair in different EMD buckets.
    rng = np.random.default_rng(999)
    bucket_to_spot: dict[int, tuple] = {}
    for _ in range(4000):
        hole, board = sample_legal_spot(rng)
        mb = fast_made_hand_bucket(hole, board, n_made_buckets)
        eb = emd.bucket(hole, board)
        if mb in bucket_to_spot:
            (phole, pboard, peb) = bucket_to_spot[mb]
            if peb != eb:
                hist_a = strength_histogram(phole, pboard, n_bins=n_bins, n_runouts=n_runouts, seed=1)
                hist_b = strength_histogram(hole, board, n_bins=n_bins, n_runouts=n_runouts, seed=2)
                return {
                    "found": True,
                    "source": "random_search",
                    "made_hand": {
                        "hole": [card_id_to_str(c) for c in phole],
                        "board": [card_id_to_str(c) for c in pboard],
                        "made_bucket": mb,
                        "emd_bucket": peb,
                        "histogram": hist_a.round(4).tolist(),
                    },
                    "draw_hand": {
                        "hole": [card_id_to_str(c) for c in hole],
                        "board": [card_id_to_str(c) for c in board],
                        "made_bucket": mb,
                        "emd_bucket": eb,
                        "histogram": hist_b.round(4).tolist(),
                    },
                    "shared_made_bucket": int(mb),
                    "emd_distance": _emd_1d(hist_a, hist_b),
                }
        else:
            bucket_to_spot[mb] = (hole, board, eb)
    return {"found": False}


# --------------------------------------------------------------------------------------------------
# DELIVERABLE 2 -- measurable thin-value EV gain.
# --------------------------------------------------------------------------------------------------
def _bin_percentiles(n_bins: int) -> np.ndarray:
    """Midpoint river percentile of each histogram bin. strength_histogram bins by
    floor(pct * n_bins) into [0, n_bins); bin b covers [b/n_bins, (b+1)/n_bins), midpoint = (b+.5)/n_bins."""
    return (np.arange(n_bins) + 0.5) / n_bins


def ev_check_per_bin(n_bins: int, *, pot: float) -> np.ndarray:
    """EV_check(p) = 0 for every bin: checking gives up the thin-value option (the baseline). Kept as a
    function of n_bins/pot for a uniform interface with ev_bet_per_bin."""
    return np.zeros(int(n_bins), dtype=np.float64)


def ev_bet_per_bin(n_bins: int, *, call_lo: float, call_hi: float, pot: float, bet_frac: float) -> np.ndarray:
    """EV_bet(p) = CF * B*pot * (2*wc(p) - 1), the thin-value bet payoff vs a MEDIUM calling band.

      wc(p) = clip((p - call_lo) / (call_hi - call_lo), 0, 1)   # P(hero wins | called), v ~ U[call_lo, call_hi]
      CF    = call_hi - call_lo                                  # villain's call frequency (band width)
      B     = bet_frac * pot
    Convex / nut-peaked in river strength: positive only ABOVE the calling band (where draws complete),
    negative inside/below it. See the module docstring for why this is the load-bearing thin-value
    metric and why the made-hand bucket cannot capture it."""
    width = call_hi - call_lo
    bet = bet_frac * pot
    p = _bin_percentiles(n_bins)
    wc = np.clip((p - call_lo) / width, 0.0, 1.0)
    return width * bet * (2.0 * wc - 1.0)


def per_hand_ev(hists: np.ndarray, ev_bet_bins: np.ndarray, ev_check_bins: np.ndarray):
    """Returns (EV_bet, EV_check) per hand = expectation of the per-bin EV under each histogram."""
    ev_bet = hists @ ev_bet_bins
    ev_check = hists @ ev_check_bins
    return ev_bet, ev_check


def abstraction_ev(buckets: np.ndarray, ev_bet: np.ndarray, ev_check: np.ndarray):
    """abstraction_EV(B) = mean_h EV_{a*(bucket(h))}(h), a*(b)=argmax_a mean_{h in b} EV_a(h).
    Returns (abstraction_EV, per_bucket_action dict). Forced to play ONE action per bucket."""
    n = len(ev_bet)
    total = 0.0
    actions = {}
    for b in np.unique(buckets):
        mask = buckets == b
        mean_bet = ev_bet[mask].mean()
        mean_check = ev_check[mask].mean()
        if mean_bet >= mean_check:
            actions[int(b)] = "bet"
            total += ev_bet[mask].sum()
        else:
            actions[int(b)] = "check"
            total += ev_check[mask].sum()
    return total / n, actions


def deliverable2(rng, *, n_spots, n_buckets, n_bins, n_runouts, call_lo, call_hi, pot, bet_frac,
                 fit_sample, bootstrap=2000):
    """EV_optimal, EV_madehand, EV_emd at the SAME bucket count, the disagreement breakdown, and a
    bootstrap CI on the per-hand EMD-minus-madehand EV difference.

    The EMD table is fit on the SAME TURN-only distribution it is evaluated on (apples-to-apples for a
    turn decision; fitting on flop+turn shifts the centroids off-distribution and adds noise)."""
    # Fit the EMD table over a representative sample of TURN spots (matches the eval distribution).
    fit_spots = [sample_legal_spot(rng, streets=(4,)) for _ in range(fit_sample)]
    fit_hists = histograms_for_spots(fit_spots, n_bins=n_bins, n_runouts=n_runouts, seed_base=30_000)
    emd = EMDBucketer(num_buckets=n_buckets, n_bins=n_bins, n_runouts=n_runouts)
    emd.fit(fit_hists, iters=15, seed=0)

    # Evaluation spots (TURN ONLY -- the decision is a turn bet/check; turn has river runouts).
    spots = [sample_legal_spot(rng, streets=(4,)) for _ in range(n_spots)]
    hists = histograms_for_spots(spots, n_bins=n_bins, n_runouts=n_runouts, seed_base=40_000)

    made = np.array([fast_made_hand_bucket(h, b, n_buckets) for (h, b) in spots], dtype=int)
    cc = np.cumsum(emd.centroids, axis=1)
    Hc = np.cumsum(hists, axis=1)
    emd_b = np.abs(Hc[:, None, :] - cc[None, :, :]).sum(axis=2).argmin(axis=1)

    ev_bet_bins = ev_bet_per_bin(n_bins, call_lo=call_lo, call_hi=call_hi, pot=pot, bet_frac=bet_frac)
    ev_check_bins = ev_check_per_bin(n_bins, pot=pot)
    ev_bet, ev_check = per_hand_ev(hists, ev_bet_bins, ev_check_bins)

    ev_optimal = float(np.maximum(ev_bet, ev_check).mean())
    ev_made, made_actions = abstraction_ev(made, ev_bet, ev_check)
    ev_emd, emd_actions = abstraction_ev(emd_b, ev_bet, ev_check)

    # bb conversion: pot is expressed in bb (we pass pot in bb units), so EV is already in bb.
    gain_chips = ev_emd - ev_made
    gain_bb = gain_chips

    # Per-hand realized EV under each abstraction (forced action of that hand's bucket).
    made_forced = np.array([1 if made_actions[int(m)] == "bet" else 0 for m in made])
    emd_forced = np.array([1 if emd_actions[int(e)] == "bet" else 0 for e in emd_b])
    optimal_action = np.where(ev_bet >= ev_check, 1, 0)  # 1=bet, 0=check
    per_hand_made = np.where(made_forced == 1, ev_bet, ev_check)
    per_hand_emd = np.where(emd_forced == 1, ev_bet, ev_check)

    # Disagreement breakdown: spots where the made-hand bucket forces the WRONG (suboptimal) action ->
    # the thin-value / semibluff spots where current rank and river potential DISAGREE.
    made_wrong = optimal_action != made_forced
    emd_right_where_made_wrong = made_wrong & (optimal_action == emd_forced)
    recovered_on_disagree = float((per_hand_emd - per_hand_made)[made_wrong].sum() / n_spots)

    # Bootstrap CI on the mean per-hand (EMD - madehand) realized-EV difference: resample hands, recompute
    # the mean diff. Forced actions are held at the full-sample optimum; we bootstrap the realized-EV
    # difference, the quantity whose sign is the claim.
    diff = per_hand_emd - per_hand_made
    ci_lo = ci_hi = None
    if bootstrap and bootstrap > 0:
        boot_rng = np.random.default_rng(12345)
        boots = np.empty(int(bootstrap), dtype=np.float64)
        for i in range(int(bootstrap)):
            idx = boot_rng.integers(0, n_spots, size=n_spots)
            boots[i] = diff[idx].mean()
        ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    return {
        "EV_optimal": ev_optimal,
        "EV_madehand": ev_made,
        "EV_emd": ev_emd,
        "gain_chips": gain_chips,
        "gain_bb": gain_bb,
        "gain_bb_ci95": [ci_lo, ci_hi],
        "gap_to_optimal_madehand": ev_optimal - ev_made,
        "gap_to_optimal_emd": ev_optimal - ev_emd,
        "frac_gap_closed_by_emd": (
            (gain_chips / (ev_optimal - ev_made)) if (ev_optimal - ev_made) > 1e-12 else None
        ),
        "n_spots": n_spots,
        "n_buckets": n_buckets,
        "call_band": [call_lo, call_hi],
        "pot_bb": pot,
        "bet_frac": bet_frac,
        "n_made_forces_wrong_action": int(made_wrong.sum()),
        "n_emd_right_where_made_wrong": int(emd_right_where_made_wrong.sum()),
        "ev_recovered_on_disagreement_spots": recovered_on_disagree,
    }


def sensitivity_sweep(rng_seed, *, n_spots, n_buckets, n_bins, n_runouts, pot, bet_frac, fit_sample, bands):
    """EV gain across a sweep of the villain calling BAND (a stated caveat: the gain magnitude depends
    on the band; we show it across a defensible range of medium calling ranges, with the SAME spots)."""
    out = []
    for lo, hi in bands:
        rng = np.random.default_rng(rng_seed)  # SAME spots across bands for a clean comparison
        d2 = deliverable2(
            rng, n_spots=n_spots, n_buckets=n_buckets, n_bins=n_bins, n_runouts=n_runouts,
            call_lo=lo, call_hi=hi, pot=pot, bet_frac=bet_frac, fit_sample=fit_sample, bootstrap=0,
        )
        out.append({"call_band": [lo, hi], "gain_bb": d2["gain_bb"], "EV_emd": d2["EV_emd"],
                    "EV_madehand": d2["EV_madehand"], "EV_optimal": d2["EV_optimal"],
                    "frac_gap_closed": d2["frac_gap_closed_by_emd"]})
    return out


def functional_micro_ev_tasks(
    rng,
    *,
    n_spots: int = 800,
    n_buckets: int = 30,
    n_bins: int = 20,
    n_runouts: int = 200,
    fit_sample: int = 600,
    pot: float = 10.0,
) -> dict[str, dict]:
    """Multi-task functional EV screen beyond the single thin-value spot.

    Tasks probe draw-continue / semi-bluff, bluff-catcher check-call, and a wide-band continue spot.
    Each task reuses deliverable2 with task-specific calling bands and bet fractions."""
    specs = {
        "draw_continue": {"call_lo": 0.25, "call_hi": 0.80, "bet_frac": 0.66},
        "semi_bluff": {"call_lo": 0.30, "call_hi": 0.65, "bet_frac": 0.75},
        "bluff_catcher_check_call": {"call_lo": 0.40, "call_hi": 0.55, "bet_frac": 0.33},
        "wide_band_continue": {"call_lo": 0.20, "call_hi": 0.85, "bet_frac": 0.50},
    }
    out: dict[str, dict] = {}
    for name, kw in specs.items():
        task_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        d2 = deliverable2(
            task_rng,
            n_spots=n_spots,
            n_buckets=n_buckets,
            n_bins=n_bins,
            n_runouts=n_runouts,
            call_lo=kw["call_lo"],
            call_hi=kw["call_hi"],
            pot=pot,
            bet_frac=kw["bet_frac"],
            fit_sample=fit_sample,
            bootstrap=500,
        )
        out[name] = {
            "gain_bb": d2["gain_bb"],
            "gain_bb_ci95": d2["gain_bb_ci95"],
            "EV_emd": d2["EV_emd"],
            "EV_madehand": d2["EV_madehand"],
            "frac_gap_closed_by_emd": d2["frac_gap_closed_by_emd"],
            "n_made_forces_wrong_action": d2["n_made_forces_wrong_action"],
            "call_band": [kw["call_lo"], kw["call_hi"]],
            "bet_frac": kw["bet_frac"],
        }
    return out


def river_fallback_sanity(*, n_bins: int = 20, n_runouts: int = 200, n_buckets: int = 30) -> dict:
    """River spots have no runouts left; histograms should collapse (no potential spread)."""
    rng = np.random.default_rng(4242)
    spots = [sample_legal_spot(rng, streets=(5,)) for _ in range(120)]
    hists = histograms_for_spots(spots, n_bins=n_bins, n_runouts=n_runouts, seed_base=50_000)
    peak_mass = hists.max(axis=1)
    spread = float(np.var(hists, axis=1).mean())
    return {
        "river_peak_mass_min": float(peak_mass.min()),
        "river_hist_variance_mean": spread,
        "n_spots": len(spots),
    }


# --------------------------------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=6000, help="evaluation spots per deliverable")
    ap.add_argument("--fit-sample", type=int, default=4000, help="histograms used to fit the EMD table")
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--n-runouts", type=int, default=400)
    ap.add_argument("--made-buckets", type=int, default=100, help="num_buckets for the made-hand metric (D1)")
    ap.add_argument("--emd-buckets", type=int, default=100, help="num_buckets for the EMD table (D1)")
    ap.add_argument("--ev-buckets", type=int, default=30, help="bucket count for the EV comparison (D2)")
    ap.add_argument("--call-lo", type=float, default=0.30, help="villain calling band lower river-pct")
    ap.add_argument("--call-hi", type=float, default=0.70, help="villain calling band upper river-pct")
    ap.add_argument("--pot", type=float, default=10.0, help="pot before the decision, in bb")
    ap.add_argument("--bet-frac", type=float, default=0.5, help="bet size as a fraction of pot")
    ap.add_argument("--seed", type=int, default=20260620)
    ap.add_argument("--out", type=str, default="artifacts/emd_quality_report.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print("=" * 90)
    print("EMD vs made-hand-only postflop abstraction -- quality validation (Workstream F, item 3)")
    print(f"native kernel: {'YES (battery_native)' if _EXT is not None else 'NO (python fallback)'}")
    print(f"samples={args.samples}  fit_sample={args.fit_sample}  n_bins={args.n_bins} "
          f"n_runouts={args.n_runouts}  seed={args.seed}")
    print("=" * 90)

    # ----- DELIVERABLE 1 -----
    print("\n[D1] Draw/made separation at EQUAL current rank (flop+turn; river EXCLUDED -- no potential)")
    rng = np.random.default_rng(args.seed)
    d1 = deliverable1(
        rng, n_spots=args.samples, n_made_buckets=args.made_buckets, n_emd_buckets=args.emd_buckets,
        n_bins=args.n_bins, n_runouts=args.n_runouts, fit_sample=args.fit_sample, max_pairs=200,
    )
    print(f"  A1 separation fraction (equal-made-rank pairs in DIFFERENT EMD buckets) = {d1['A1_separation_fraction']:.4f}")
    print(f"  A2 mean EMD distance between equal-made-rank histograms              = {d1['A2_mean_emd_distance']:.4f}")
    print(f"     ({d1['n_pairs_compared']} pairs across {d1['n_made_buckets_populated']} populated made-buckets, "
          f"{d1['n_made_buckets']} made / {d1['n_emd_buckets']} EMD buckets)")
    we = d1["worked_example"]
    if we.get("found"):
        m, dr = we["made_hand"], we["draw_hand"]
        print(f"  WORKED EXAMPLE ({we['source']}): both in made-bucket {we['shared_made_bucket']}, "
              f"EMD distance {we['emd_distance']:.3f}")
        print(f"    MADE  {m['hole']} on {m['board']}  -> made_bucket={m['made_bucket']}  EMD_bucket={m['emd_bucket']}")
        print(f"    DRAW  {dr['hole']} on {dr['board']}  -> made_bucket={dr['made_bucket']}  EMD_bucket={dr['emd_bucket']}")
    else:
        print("  WORKED EXAMPLE: not found (see caveats)")

    # ----- DELIVERABLE 2 -----
    print("\n[D2] Thin-value EV gain (TURN bet vs check; EV functional in module docstring)")
    rng = np.random.default_rng(args.seed + 1)
    d2 = deliverable2(
        rng, n_spots=args.samples, n_buckets=args.ev_buckets, n_bins=args.n_bins,
        n_runouts=args.n_runouts, call_lo=args.call_lo, call_hi=args.call_hi, pot=args.pot,
        bet_frac=args.bet_frac, fit_sample=args.fit_sample,
    )
    ci = d2["gain_bb_ci95"]
    print(f"  bucket count = {d2['n_buckets']}   villain calls band {d2['call_band']}   "
          f"pot = {d2['pot_bb']} bb   bet = {d2['bet_frac']:.0%} pot")
    print(f"  EV_optimal (unabstracted)  = {d2['EV_optimal']:.5f} bb")
    print(f"  EV_madehand                = {d2['EV_madehand']:.5f} bb   (gap to optimal {d2['gap_to_optimal_madehand']:.5f})")
    print(f"  EV_emd                     = {d2['EV_emd']:.5f} bb   (gap to optimal {d2['gap_to_optimal_emd']:.5f})")
    print(f"  >> EV GAIN (EMD - madehand) = {d2['gain_bb']:+.5f} bb  "
          f"95%CI [{ci[0]:+.5f}, {ci[1]:+.5f}]  "
          f"({100.0 * (d2['frac_gap_closed_by_emd'] or 0):.1f}% of the made-hand gap closed)")
    print(f"     made-bucket forces the WRONG action on {d2['n_made_forces_wrong_action']} / {d2['n_spots']} spots; "
          f"EMD recovers {d2['ev_recovered_on_disagreement_spots']:+.5f} bb there")

    # ----- SENSITIVITY -----
    print("\n[D2-sensitivity] EV gain vs villain calling BAND (same spots per band)")
    sweep = sensitivity_sweep(
        args.seed + 1, n_spots=min(args.samples, 6000), n_buckets=args.ev_buckets, n_bins=args.n_bins,
        n_runouts=args.n_runouts, pot=args.pot, bet_frac=args.bet_frac, fit_sample=args.fit_sample,
        bands=[(0.25, 0.65), (0.30, 0.70), (0.35, 0.75), (0.35, 0.85), (0.40, 0.90)],
    )
    for s in sweep:
        fc = s["frac_gap_closed"]
        print(f"    band={s['call_band']}  gain={s['gain_bb']:+.5f} bb  "
              f"(EMD {s['EV_emd']:.4f} vs made {s['EV_madehand']:.4f}, optimal {s['EV_optimal']:.4f}, "
              f"gap closed {100.0 * (fc or 0):.0f}%)")

    elapsed = time.perf_counter() - t0
    report = {
        "meta": {
            "native_kernel": _EXT is not None,
            "seed": args.seed,
            "samples": args.samples,
            "fit_sample": args.fit_sample,
            "n_bins": args.n_bins,
            "n_runouts": args.n_runouts,
            "elapsed_sec": elapsed,
            "ev_functional": (
                "TURN thin-value bet vs check. Villain calls a MEDIUM band [call_lo, call_hi] of river "
                "strength; conditional on a call v~U[call_lo,call_hi] so wc(p)=clip((p-call_lo)/"
                "(call_hi-call_lo),0,1). CF=call_hi-call_lo, B=bet_frac*pot. "
                "EV_bet(p)=CF*B*(2*wc(p)-1) (convex/nut-peaked: +B above the band, -B below), "
                "EV_check(p)=0 (give up the option). EV_a(h)=E_{p~hist(h)}[EV_a(p)]. "
                "abstraction_EV(B)=mean_h EV_{a*(bucket h)}(h), a*(b)=argmax_a mean_{h in b} EV_a(h). "
                "EV_optimal=mean_h max_a EV_a(h). EMD table for D2 fit on the SAME turn-only distribution."
            ),
            "river_excluded_note": (
                "RIVER (5-card board) is excluded from D1: no runouts -> strength_histogram is a single "
                "spike at the made-hand percentile, so EMD == made-hand bucketing there by construction."
            ),
        },
        "deliverable1_draw_made_separation": d1,
        "deliverable2_thin_value_ev_gain": d2,
        "deliverable2_sensitivity_call_band": sweep,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}  ({elapsed:.1f}s)")
    print("=" * 90)


if __name__ == "__main__":
    main()
