"""Vectorized 4-max AoF fictitious play — the perf fix for the 4-player GTO base.

The pure-Python per-deal tree walk in ``exploit_mc`` is ~tens of ms/sample, making a converged
4-max FP solve take hours. But 4-max AoF has only **15 reachable action outcomes** (each player
folds/jams in order; a terminal's payoff depends only on the *jammer set*), so we:

  1. Sample a fixed batch of N real-card deals (CRN across rounds).
  2. Precompute ONCE the per-(outcome, deal, player) terminal payoff — reusing the validated
     legal chip model in ``exploit_mc._terminal_returns`` (pot = k*stack + dead blinds, etc.).
  3. Run FP/BR as pure numpy over the (15, N, 4) payoff tensor — milliseconds/round.

This is the sampled analog of the 3-max equity tensor. The legal BR marginalizes opponents'
hidden hands (no clairvoyance): for each own-hand infoset, push iff E[u_push] >= E[u_fold] with
opponents on-policy downstream. Validated against the (slow, correct) ``exploit_mc`` path.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.combo_deal import FULL_DECK, bucket_from_card_ids, card_id
from aof_cfr.exploit_mc import _terminal_returns
from aof_cfr.infoset import encode_infoset_key
from aof_ranges import active_seats

NUM_PLAYERS = 4
N_BUCKETS = 169


def _outcomes() -> list[dict]:
    """The 15 reachable action outcomes (UTG, BTN, SB, BB act in order; FFF -> BB wins)."""
    outs: list[dict] = []
    for a0 in "FA":
        for a1 in "FA":
            for a2 in "FA":
                prefix = a0 + a1 + a2
                if prefix == "FFF":
                    acts = [(0, "-", "F"), (1, "F", "F"), (2, "FF", "F")]  # BB wins uncontested
                    outs.append({"acts": acts, "folded": frozenset({0, 1, 2})})
                    continue
                for a3 in "FA":
                    acts = [(0, "-", a0), (1, a0, a1), (2, a0 + a1, a2), (3, a0 + a1 + a2, a3)]
                    folded = frozenset(i for (i, _h, a) in acts if a == "F")
                    outs.append({"acts": acts, "folded": folded})
    return outs


_OUTCOMES = _outcomes()


def sample_deals(n_deals: int, seed: int) -> np.ndarray:
    """Return (n_deals, 4) bucket indices from real-card deals (correct combo freq + blockers).

    Also returns the raw hole ids (n_deals, 4, 2) for the payoff precompute via the cache key.
    """
    rng = random.Random(seed)
    buckets = np.empty((n_deals, NUM_PLAYERS), dtype=np.int64)
    holes: list[list[tuple[int, int]]] = []
    lab_to_idx = {lab: i for i, lab in enumerate(LABELS_169)}
    for d in range(n_deals):
        deck = list(FULL_DECK)
        rng.shuffle(deck)
        picked = deck[: 2 * NUM_PLAYERS]
        deal = [(card_id(picked[2 * i]), card_id(picked[2 * i + 1])) for i in range(NUM_PLAYERS)]
        holes.append(deal)
        for i in range(NUM_PLAYERS):
            buckets[d, i] = lab_to_idx[bucket_from_card_ids(deal[i][0], deal[i][1])]
    return buckets, holes


def precompute_payoffs(holes: list, stack_bb: float) -> np.ndarray:
    """(n_outcomes, n_deals, 4) net-bb payoff per outcome, via the validated terminal model."""
    seats = active_seats(NUM_PLAYERS)
    n = len(holes)
    pay = np.zeros((len(_OUTCOMES), n, NUM_PLAYERS), dtype=np.float64)
    for oi, o in enumerate(_OUTCOMES):
        folded = set(o["folded"])
        for d in range(n):
            pay[oi, d, :] = _terminal_returns(NUM_PLAYERS, seats, holes[d], folded, stack_bb)
    return pay


@dataclass
class _Policy:
    """push prob arrays keyed by (player, history) -> (169,)."""
    table: dict[tuple[int, str], np.ndarray] = field(default_factory=dict)

    def get(self, player: int, hist: str) -> np.ndarray:
        key = (player, hist)
        if key not in self.table:
            self.table[key] = np.full(N_BUCKETS, 0.5, dtype=np.float64)
        return self.table[key]


def _infoset_index(buckets: np.ndarray):
    """List of (player, hist) decision infosets present across outcomes."""
    seen = set()
    for o in _OUTCOMES:
        for (i, h, _a) in o["acts"]:
            seen.add((i, h))
    return sorted(seen)


def _reach_factors(pol: _Policy, buckets: np.ndarray):
    """Per outcome: list of (player, action, factor_array(N)) for acting players."""
    out = []
    for o in _OUTCOMES:
        facs = []
        for (i, h, a) in o["acts"]:
            ph = pol.get(i, h)[buckets[:, i]]
            facs.append((i, a, ph if a == "A" else 1.0 - ph))
        out.append(facs)
    return out


def _br_marginals(pol: _Policy, buckets: np.ndarray, payoffs: np.ndarray, infosets: list):
    """Reach-weighted push/fold payoff sums per (player, history) infoset (one FP pass).

    Returns (num_push, num_fold, den) dicts keyed by (player, hist) -> (169,). For each player
    the legal best response is ``num_push >= num_fold`` per own-hand bucket; ``den`` is the
    reach weight. Opponents act on-policy downstream (no clairvoyance)."""
    n = buckets.shape[0]
    facs = _reach_factors(pol, buckets)
    num_push = {k: np.zeros(N_BUCKETS) for k in infosets}
    num_fold = {k: np.zeros(N_BUCKETS) for k in infosets}
    den = {k: np.zeros(N_BUCKETS) for k in infosets}
    for oi, o in enumerate(_OUTCOMES):
        fac = facs[oi]
        for pos, (pp, hpp, app) in enumerate(o["acts"]):
            re = np.ones(n)
            for q, (_i, _a, f) in enumerate(fac):
                if q != pos:
                    re = re * f
            tgt = num_push if app == "A" else num_fold
            np.add.at(tgt[(pp, hpp)], buckets[:, pp], re * payoffs[oi, :, pp])
            np.add.at(den[(pp, hpp)], buckets[:, pp], re)
    return num_push, num_fold, den


def onpolicy_seat_ev(pol: _Policy, buckets: np.ndarray, payoffs: np.ndarray) -> list[float]:
    """Exact-ish expected chip-EV (bb/hand) per seat over the deal batch for a full table.

    ev[seat] = mean_deal sum_outcome reach(outcome) * payoff(outcome, seat). Equity is already
    expected over boards (payoff precompute), so the only residual variance is the deal sample.
    """
    n = buckets.shape[0]
    facs = _reach_factors(pol, buckets)
    seat_ev = np.zeros(NUM_PLAYERS)
    for oi in range(len(_OUTCOMES)):
        reach = np.ones(n)
        for (_i, _a, f) in facs[oi]:
            reach = reach * f
        for pp in range(NUM_PLAYERS):
            seat_ev[pp] += np.sum(reach * payoffs[oi, :, pp])
    return [float(x / n) for x in seat_ev]


def rows_to_table(rows: dict[str, np.ndarray], stack_bb: float) -> _Policy:
    """Convert an infoset-key-v2 rows dict into a vectorized _Policy table (push probs)."""
    from aof_cfr.infoset import decode_infoset_key

    seats = active_seats(NUM_PLAYERS)
    seat_idx = {s: i for i, s in enumerate(seats)}
    lab_to_idx = {lab: i for i, lab in enumerate(LABELS_169)}
    pol = _Policy()
    for key, row in rows.items():
        pos, bucket, hist, _d = decode_infoset_key(key)
        if pos not in seat_idx or bucket not in lab_to_idx:
            continue
        pol.get(seat_idx[pos], hist)[lab_to_idx[bucket]] = float(row[1])
    return pol


def solve_4max_fp_vec(
    *,
    n_deals: int = 60000,
    max_rounds: int = 1000,
    damping: float = 1.0,
    seed: int = 42,
    stack_bb: float = 8.0,
    payoffs: np.ndarray | None = None,
    buckets: np.ndarray | None = None,
) -> dict:
    """Vectorized averaged FP. Returns policy rows + legal NashConv (bb) on the deal batch."""
    if buckets is None:
        buckets, holes = sample_deals(n_deals, seed)
        payoffs = precompute_payoffs(holes, stack_bb)
    n = buckets.shape[0]
    infosets = _infoset_index(buckets)
    pol = _Policy()
    for (i, h) in infosets:
        pol.get(i, h)  # init 0.5

    gap = 0.0
    for rnd in range(1, max_rounds + 1):
        num_push, num_fold, _den = _br_marginals(pol, buckets, payoffs, infosets)
        alpha = damping / rnd
        for k in infosets:
            br = (num_push[k] >= num_fold[k]).astype(np.float64)  # legal pure best response
            pol.table[k] = (1.0 - alpha) * pol.table[k] + alpha * br

    # Final legal NashConv on the batch (sum of per-player BR gains).
    num_push, num_fold, den = _br_marginals(pol, buckets, payoffs, infosets)
    per_player_gain = [0.0] * NUM_PLAYERS
    for (pp, hpp) in infosets:
        p = pol.table[(pp, hpp)]
        ep = np.where(den[(pp, hpp)] > 0, num_push[(pp, hpp)], 0.0)
        ef = np.where(den[(pp, hpp)] > 0, num_fold[(pp, hpp)], 0.0)
        onpol = (1.0 - p) * ef + p * ep
        per_player_gain[pp] += float(np.sum(np.maximum(ep, ef) - onpol))
    gain_bb = [g / n for g in per_player_gain]
    gap = float(sum(max(0.0, g) for g in gain_bb))

    rows = _policy_to_rows(pol, stack_bb)
    return {
        "rows": rows, "nash_conv_bb": gap, "per_player_bb": gain_bb,
        "rounds": max_rounds, "n_deals": n, "stack_bb": stack_bb,
    }


def _policy_to_rows(pol: _Policy, stack_bb: float) -> dict[str, np.ndarray]:
    seats = active_seats(NUM_PLAYERS)
    rows: dict[str, np.ndarray] = {}
    for (i, h), arr in pol.table.items():
        for bi, lab in enumerate(LABELS_169):
            p = float(arr[bi])
            rows[encode_infoset_key(seats[i], lab, h, stack_bb)] = np.array([1.0 - p, p])
    return rows
