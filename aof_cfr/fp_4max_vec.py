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
from aof_cfr.exploit_mc import _N_BOARDS, _SHARE_CACHE, _seat_blind, _terminal_returns
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


def precompute_payoffs(holes: list, stack_bb: float, *, game_model: int | None = None) -> np.ndarray:
    """(n_outcomes, n_deals, 4) net-bb payoff per outcome, via the shared terminal model.

    ``game_model`` selects the contested-pot convention (None -> the shipped manifest's);
    batches written to disk embed it so the router can reject stale-model files.

    Optional native path: a bit-identical Rust kernel (``aof_cfr.marg4_native``) reproduces the
    production MC exactly (matched CPython MT19937 board RNG + winner-equivalent eval7), so its
    output is ``np.array_equal`` to the Python loop below. It is **opt-in** (env ``AOF_MARG4_NATIVE=1``)
    and OFF by default: because it must replay CPython's exact serial RNG per showdown to stay
    bit-identical, it is ~2x SLOWER per core than this Python path, and its only lever (rayon across
    deals in one process) is redundant with the production driver, which already parallelizes across
    deals via ProcessPoolExecutor (one core per worker). See bench/marg4_native.py. A future
    statistically-equivalent kernel (fast native RNG + SIMD eval7, validated to MC tolerance instead
    of bit-identity) is where a real speedup lives."""
    n = len(holes)
    nat = _precompute_payoffs_native(holes, stack_bb, game_model)
    if nat is not None:
        return nat
    fast = _precompute_payoffs_batched(holes, stack_bb, game_model)
    if fast is not None:
        return fast
    seats = active_seats(NUM_PLAYERS)
    pay = np.zeros((len(_OUTCOMES), n, NUM_PLAYERS), dtype=np.float64)
    for oi, o in enumerate(_OUTCOMES):
        folded = set(o["folded"])
        for d in range(n):
            pay[oi, d, :] = _terminal_returns(NUM_PLAYERS, seats, holes[d], folded, stack_bb,
                                              game_model=game_model)
    return pay


# --- batched showdown fast path -------------------------------------------------------------
# The Python loop above spends ~all its time in _terminal_returns -> _showdown_shares -> the native
# mc_nway_shares, one Python->native crossing per (deal, alive-subset) showdown (~11 per deal). The
# native MC math is already fast; the cost is the boundary crossings and the single-thread evaluator.
# This path collects EVERY distinct alive-subset showdown for the chunk, dedups them, runs ONE
# batched OpenMP-parallel native call (mc_nway_shares_batch, same OMPEval evaluator + Fisher-Yates
# sampler + _N_BOARDS as _showdown_shares), then assembles the (15, n, 4) tensor exactly as
# _terminal_returns does. Same per-showdown seed scheme => populates _SHARE_CACHE identically.
#
# threads default = 1: process-pool-safe. The production driver runs many chunks under a
# ProcessPoolExecutor (one core per worker), so each chunk must stay single-threaded or it
# oversubscribes all cores. A single-process caller can set AOF_MARG4_BATCH_THREADS to use OpenMP.
import os as _os  # noqa: E402

_DISABLE_BATCH = _os.environ.get("AOF_MARG4_BATCH", "1").lower() in ("0", "false", "no")


def _showdown_seed(flat: tuple[int, ...]) -> int:
    seed = 0
    for c in flat:
        seed = (seed * 131 + int(c) + 1) & 0xFFFFFFFF
    return seed


def _precompute_payoffs_batched(holes: list, stack_bb: float, game_model: int | None):
    """Native batched showdown tensor (statistically identical to the Python loop), or ``None`` to
    fall back. See the module comment above ``precompute_payoffs``'s fast-path wiring."""
    if _DISABLE_BATCH:
        return None
    try:
        from tools.ompeval_cli.native import (
            mc_nway_shares_batch_available,
            mc_nway_shares_batch_native,
        )
    except Exception:  # noqa: BLE001
        return None
    if not mc_nway_shares_batch_available():
        return None
    from aof_cfr.pushfold_payoffs import contested_pot_bb

    n = len(holes)
    seats = active_seats(NUM_PLAYERS)
    blinds = [_seat_blind(seats[i]) for i in range(NUM_PLAYERS)]
    threads = int(_os.environ.get("AOF_MARG4_BATCH_THREADS", "1"))

    # The contested outcomes only ever use a handful of distinct alive-index patterns (the 4-way set,
    # the four 3-way sets, the six 2-way pairs). Group outcomes by pattern so we build each deal's
    # showdown key ONCE per pattern (not once per outcome), and so step 3 can be a vectorized gather.
    alive_of = [tuple(i for i in range(NUM_PLAYERS) if i not in o["folded"]) for o in _OUTCOMES]
    patterns: dict[tuple[int, ...], list[int]] = {}
    for oi, alive in enumerate(alive_of):
        if len(alive) >= 2:
            patterns.setdefault(alive, []).append(oi)

    # 1) Collect every distinct alive-subset showdown across deals (deduped CHUNK-LOCALLY via
    #    key_to_idx), and record, per pattern, a per-deal index into the share table. We deliberately
    #    do NOT use the module-global _SHARE_CACHE here: over millions of distinct random precompute
    #    deals it essentially never hits and grows without bound, OOM-ing each ProcessPool worker
    #    (~1.9 GB+ observed -> BrokenProcessPool). All useful dedup is chunk-local and freed on return.
    needed: dict[tuple[int, ...], list[float] | None] = {}
    key_to_idx: dict[tuple[int, ...], int] = {}
    pat_idx: dict[tuple[int, ...], np.ndarray] = {}   # alive-pattern -> (n,) row index into `table`
    n_keys = 0
    for alive in patterns:
        idx = np.empty(n, dtype=np.int64)
        for d in range(n):
            hd = holes[d]
            flat = tuple(c for i in alive for c in hd[i])
            j = key_to_idx.get(flat)
            if j is None:
                j = n_keys
                key_to_idx[flat] = j
                n_keys += 1
                needed[flat] = None   # chunk-local only (see note below); never the global cache
            idx[d] = j
        pat_idx[alive] = idx

    try:
        # 2) One batched native call over the not-yet-cached showdowns; fill a (n_keys, 4) share table.
        table = np.zeros((max(n_keys, 1), NUM_PLAYERS), dtype=np.float64)
        to_eval = [f for f, v in needed.items() if v is None]
        if to_eval:
            holes_flat: list[int] = []
            sizes: list[int] = []
            seeds: list[int] = []
            for flat in to_eval:
                holes_flat.extend(int(c) for c in flat)
                sizes.append(len(flat) // 2)
                seeds.append(_showdown_seed(flat))
            shares = mc_nway_shares_batch_native(
                np.asarray(holes_flat, dtype=np.int32),
                np.asarray(sizes, dtype=np.int32),
                np.asarray(seeds, dtype=np.uint32),
                _N_BOARDS,
                threads,
            )
            for si, flat in enumerate(to_eval):
                k = len(flat) // 2
                needed[flat] = [float(x) for x in shares[si, :k]]
        # Pack all (cached + freshly evaluated) shares into the dense table by row index.
        for flat, j in key_to_idx.items():
            sh = needed[flat]
            table[j, : len(sh)] = sh
    except Exception:  # noqa: BLE001 — any native hiccup -> safe Python fallback
        return None

    # 3) Assemble (15, n, 4) exactly as _terminal_returns: blinds, uncontested steal, contested pot.
    #    Vectorized: per alive-pattern, gather each deal's share row (table[idx]) and apply
    #    share*pot - stack to the alive columns in one shot for every outcome sharing that pattern.
    pay = np.zeros((len(_OUTCOMES), n, NUM_PLAYERS), dtype=np.float64)
    base = np.array([-blinds[i] for i in range(NUM_PLAYERS)], dtype=np.float64)
    fold_blind_sum = [sum(blinds[j] for j in o["folded"]) for o in _OUTCOMES]
    # Uncontested (<=1 alive) outcomes: deterministic, no showdown.
    for oi, alive in enumerate(alive_of):
        if len(alive) <= 1:
            slab = np.tile(base, (n, 1))
            if len(alive) == 1:
                slab[:, alive[0]] = fold_blind_sum[oi]
            pay[oi] = slab
    # Contested outcomes, grouped by alive-pattern (one gather per pattern). The share table stores
    # each showdown's shares in ALIVE ORDER (columns 0..k-1), so gather those columns and scatter
    # them to the alive SEAT columns.
    for alive, ois in patterns.items():
        cols = np.array(alive, dtype=np.int64)
        k = len(alive)
        gathered = table[pat_idx[alive]][:, :k]      # (n, k) share per deal, in alive order
        for oi in ois:
            pot = contested_pot_bb(len(alive), fold_blind_sum[oi], stack_bb, game_model=game_model)
            slab = np.tile(base, (n, 1))
            slab[:, cols] = gathered * pot - stack_bb
            pay[oi] = slab
    return pay


def _precompute_payoffs_native(holes: list, stack_bb: float, game_model: int | None):
    """Native tensor (bit-identical), or ``None`` to fall back. Resolves the runtime game-model /
    flat-fee here so the native kernel uses the exact same contested-pot convention as the Python
    ``contested_pot_bb`` would."""
    import os
    if os.environ.get("AOF_MARG4_NATIVE", "") not in ("1", "true", "yes"):
        return None  # opt-in only: the bit-identical kernel is ~2x slower/core (see precompute_payoffs)
    try:
        from aof_cfr.marg4_native import native_available, precompute_payoffs_native
    except Exception:  # noqa: BLE001
        return None
    if not native_available():
        return None
    from aof_cfr.pushfold_payoffs import flat_fee_default, game_model_default
    gm = game_model_default() if game_model is None else int(game_model)
    fee = flat_fee_default()
    try:
        return precompute_payoffs_native(holes, stack_bb, game_model=gm, fee_bb=fee)
    except Exception:  # noqa: BLE001 — any native hiccup -> safe Python fallback
        return None


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
    infoset reach weight. Opponents act on-policy downstream (no clairvoyance).

    ``den`` accumulates over the PUSH branch only: an infoset's outcomes split into
    push-continuations and fold-continuations, and EACH branch's reach (own action factor
    excluded) sums to the full infoset reach — downstream actors' F/A factors sum to 1, and
    the collapsed FFF outcome (no BB act) mirrors the FFA·{F,A} pair exactly. Accumulating
    both branches (the old behavior) double-counted: den = 2x reach, silently HALVING every
    ``num/den`` EV downstream — the router's ev_gain gate demanded twice the configured edge
    in 4-max only, and advise()/coach() reported 4-max EVs at half scale. The solver paths
    (``legal_nash_gap_4max``, FP) never divide by den and were unaffected."""
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
            if app == "A":
                np.add.at(den[(pp, hpp)], buckets[:, pp], re)
    return num_push, num_fold, den


def marginal_sums(rows: dict[str, np.ndarray], stack_bb: float, n_deals: int, seed: int):
    """One chunk of the EXACT (true-game) 4-max BR marginal: the ADDITIVE reach-weighted sums
    ``(num_push, num_fold, den)`` over ``n_deals`` real-card deals at ``seed``.

    These are pure sums, so chunks from different seeds add: ``num/den`` over the concatenation of
    all chunks is the unbiased marginal of the full sample. That makes the high-N precompute
    embarrassingly parallel — run many chunks (different seeds) across cores/boxes and sum. Unlike
    the analytic 3-max reuse (which is biased ~0.3bb because it drops the folded UTG hand), this is
    the TRUE 4-max game (UTG present, real card removal + fold-range reach), so its only error is
    sampling variance, which → 0 as the total deal count grows.
    """
    pol = rows_to_table(rows, stack_bb)
    buckets, holes = sample_deals(n_deals, seed)
    payoffs = precompute_payoffs(holes, stack_bb)
    infosets = _infoset_index(buckets)
    return _br_marginals(pol, buckets, payoffs, infosets)


def accumulate_marginal_sums(chunks):
    """Sum a list of ``(num_push, num_fold, den)`` chunk dicts into one, then divide to EV dicts.
    Returns ``{(seat, hist): (ev_push(169,), ev_fold(169,))}`` keyed by SEAT NAME."""
    from aof_cfr.exploit_router import _SEATS_4MAX
    np_tot: dict = {}
    nf_tot: dict = {}
    den_tot: dict = {}
    for num_push, num_fold, den in chunks:
        for k in den:
            np_tot[k] = np_tot.get(k, 0.0) + num_push[k]
            nf_tot[k] = nf_tot.get(k, 0.0) + num_fold[k]
            den_tot[k] = den_tot.get(k, 0.0) + den[k]
    out: dict = {}
    for (si, hist), d in den_tot.items():
        safe = np.where(d > 0, d, 1.0)
        out[(_SEATS_4MAX[si], hist)] = (np_tot[(si, hist)] / safe, nf_tot[(si, hist)] / safe)
    return out, den_tot


_HIGHN_CACHE: dict = {}


HIGHN_MIN_DEN = 400.0   # only trust a high-N cell with >=this effective sample (std ~6/sqrt(den)<=0.3bb)
SAMPLE_MIN_DEN = 250.0  # only trust an 8k-batch cell with >=this effective sample; below it (the rare
#                         UTG-jammed/4-way nodes) the marginal is too noisy -> defer to the equilibrium


def load_highN_marginals(stack_bb: float):
    """Load the EXACT high-N precomputed 4-max marginals for a depth (see
    scripts/aof/precompute_4max_marginals.py), or ``None`` if absent / stale. Returns
    ``{(seat, hist): (ev_push(169,), ev_fold(169,), den(169,))}`` — the per-bucket effective sample
    ``den`` lets callers trust only well-sampled cells (the ultra-rare 4-way nodes need a very large
    deal count, so a cell below ``HIGHN_MIN_DEN`` should fall back to the stable analytic/sampled
    value — graceful degradation that makes ANY file safe at ANY deal count). Rejects a file built
    under a different game model / showdown fee (same guard as the router's deal batch). Cached."""
    import numpy as np

    key = round(float(stack_bb), 3)
    if key in _HIGHN_CACHE:
        return _HIGHN_CACHE[key]
    from pathlib import Path
    path = Path("data/aof/cfr_base") / f"marginals_4p_d{key:g}.npz"
    out = None
    if path.is_file():
        try:
            from aof_cfr.pushfold_payoffs import flat_fee_default, game_model_default
            data = np.load(path, allow_pickle=False)
            gm_ok = int(data["game_model"]) == game_model_default() if "game_model" in data else False
            fee_ok = abs(float(data["flat_fee"]) - flat_fee_default()) < 1e-9 if "flat_fee" in data else False
            if gm_ok and fee_ok:
                d: dict = {}
                for f in data.files:
                    if f.endswith("|push"):
                        seat, hist, _ = f.split("|")
                        den = data.get(f"{seat}|{hist}|den")
                        if den is None:
                            den = np.full(N_BUCKETS, np.inf)   # legacy file w/o den: trust all
                        d[(seat, hist)] = (data[f], data[f"{seat}|{hist}|fold"], den)
                out = d or None
            else:
                print(f"fp_4max_vec: ignoring {path.name} (game model/fee mismatch); "
                      "regenerate via scripts/aof/precompute_4max_marginals.py")
        except Exception:  # noqa: BLE001 — corrupt/old file -> treat as absent
            out = None
    _HIGHN_CACHE[key] = out
    return out


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


_UTG_FOLDED_MAP = {                       # 4-max (seat, history)  ->  3-max marginal key
    ("BTN", "F"): "btn",                  # UTG folded -> BTN opens the BTN/SB/BB sub-game
    ("SB", "FF"): "sb_f", ("SB", "FA"): "sb_a",
    ("BB", "FFA"): "bb_fa", ("BB", "FAF"): "bb_af", ("BB", "FAA"): "bb_aa",
}


def analytic_marginals_4max_utg_folded(stack_bb: float, *, btn_rows, sb_rows, bb_rows):
    """EXACT legal push/fold EVs for the UTG-FOLDED branch of 4-max, via the validated 3-max engine.

    When UTG folds, AoF leaves a BTN/SB/BB sub-game that is chip-identical to 3-max push/fold: UTG
    posts no blind, so it contributes no dead money, and the showdown equities depend only on the
    hands at war. So the legal marginals for (BTN,'F'), (SB,'FF'/'FA'), (BB,'FFA'/'FAF'/'FAA')
    equal the 3-max marginals (btn / sb_f / sb_a / bb_fa / bb_af / bb_aa) fed the **4-max** ranges
    at those nodes. This is the variance-free analytic that replaces the deal-sampled
    ``_br_marginals`` EVs — whose effective per-bucket sample collapses to ~1 at these deep nodes
    (e.g. BB facing two jams), producing EVs that contradict the equilibrium.

    ``btn_rows`` is the BTN push array at 'F' (169,). ``sb_rows`` is [SB@'FF', SB@'FA'] (2,169).
    ``bb_rows`` is [BB@'FFA', BB@'FAF', BB@'FAA'] (3,169) — the (fa, af, aa) order the 3-max
    engine expects. Returns ``{(seat, hist): (E_push(169,), E_fold(169,))}``. The opponents' ranges
    can be substituted (exploit BR) by passing exploited rows; pass the base rows for the GTO EVs.
    """
    from aof_cfr.exploit_router import _payoffs_3max
    from aof_cfr.fp_br import _legal_marginals_3max
    st, deal_w = _payoffs_3max(stack_bb)
    m = _legal_marginals_3max(st, deal_w, np.asarray(btn_rows, dtype=np.float64),
                              np.asarray(sb_rows, dtype=np.float64), np.asarray(bb_rows, dtype=np.float64))
    out: dict = {}
    for (seat, hist), key in _UTG_FOLDED_MAP.items():
        ep, ef, _w = m[key]
        ef_arr = ef if hasattr(ef, "__len__") else np.full(N_BUCKETS, float(ef))
        out[(seat, hist)] = (np.asarray(ep, dtype=np.float64), np.asarray(ef_arr, dtype=np.float64))
    return out


def _utg_folded_marginals_from_rows(rows: dict[str, np.ndarray], stack_bb: float):
    """Convenience: pull the 4-max ranges for the UTG-folded nodes out of an infoset-rows dict and
    return the exact analytic marginals (see ``analytic_marginals_4max_utg_folded``)."""
    from aof_cfr.infoset import encode_infoset_key

    def arr(seat: str, hist: str) -> np.ndarray:
        a = np.zeros(N_BUCKETS, dtype=np.float64)
        for bi, lab in enumerate(LABELS_169):
            a[bi] = float(rows[encode_infoset_key(seat, lab, hist, stack_bb)][1])
        return a

    return analytic_marginals_4max_utg_folded(
        stack_bb,
        btn_rows=arr("BTN", "F"),
        sb_rows=np.stack([arr("SB", "FF"), arr("SB", "FA")]),
        bb_rows=np.stack([arr("BB", "FFA"), arr("BB", "FAF"), arr("BB", "FAA")]),
    )


def legal_nash_gap_4max(pol: _Policy, buckets: np.ndarray, payoffs: np.ndarray, infosets: list,
                        *, split: bool = True):
    """Vectorized legal NashConv (bb/hand) + per-player BR gains for a full 4-max table.

    Each player's gain = reach-weighted ``BR_action_value - onpolicy`` over its infosets / n_deals.
    With ``split`` (default) the BR ACTION is chosen on one half of the deal batch and EVALUATED on
    the other — sample-splitting that removes the max-of-noisy-estimates bias (otherwise a
    near-equilibrium table reads as ~20 bb/100 instead of ~0). Same debiased definition as
    ``exploit_mc.estimate_nash_conv_mc``, but pure numpy over the precomputed payoff tensor.
    """
    n = buckets.shape[0]
    if not split or n < 4:
        num_push, num_fold, den = _br_marginals(pol, buckets, payoffs, infosets)
        per_player = [0.0] * NUM_PLAYERS
        for (pp, hpp) in infosets:
            p = pol.table[(pp, hpp)]
            ep, ef = num_push[(pp, hpp)], num_fold[(pp, hpp)]
            per_player[pp] += float(np.sum(np.maximum(ep, ef) - ((1.0 - p) * ef + p * ep)))
        gain_bb = [g / n for g in per_player]
        return float(sum(max(0.0, g) for g in gain_bb)), gain_bb

    h = n // 2  # split A (choose the BR action) | B (evaluate it) -> unbiased
    npA, nfA, _dA = _br_marginals(pol, buckets[:h], payoffs[:, :h, :], infosets)
    npB, nfB, _dB = _br_marginals(pol, buckets[h:], payoffs[:, h:, :], infosets)
    nB = n - h
    per_player = [0.0] * NUM_PLAYERS
    for (pp, hpp) in infosets:
        p = pol.table[(pp, hpp)]
        push_better = npA[(pp, hpp)] >= nfA[(pp, hpp)]           # BR action decided on A
        chosen_B = np.where(push_better, npB[(pp, hpp)], nfB[(pp, hpp)])  # ...its value on B
        onpol_B = (1.0 - p) * nfB[(pp, hpp)] + p * npB[(pp, hpp)]
        per_player[pp] += float(np.sum(chosen_B - onpol_B))
    gain_bb = [g / nB for g in per_player]
    return float(sum(max(0.0, g) for g in gain_bb)), gain_bb


def nash_gap_4max_vec(rows: dict[str, np.ndarray], *, n_deals: int = 8000, seed: int = 7,
                      stack_bb: float = 8.0, buckets: np.ndarray | None = None,
                      payoffs: np.ndarray | None = None) -> float:
    """Legal 4-max exploitability (bb/hand) of an infoset-key-v2 ``rows`` table.

    Pass a precomputed ``(buckets, payoffs)`` batch to amortize the showdown precompute across many
    tables (e.g. the practice-tier calibration). Drop-in fast replacement for
    ``exploit_mc.estimate_nash_conv_mc(4, rows, ...)``."""
    if buckets is None or payoffs is None:
        buckets, holes = sample_deals(n_deals, seed)
        payoffs = precompute_payoffs(holes, stack_bb)
    pol = rows_to_table(rows, stack_bb)
    infosets = _infoset_index(buckets)
    return legal_nash_gap_4max(pol, buckets, payoffs, infosets)[0]


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
    gap, gain_bb = legal_nash_gap_4max(pol, buckets, payoffs, infosets)

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
