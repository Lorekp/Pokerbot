"""AoF opponent profiler + per-player exploitability.

Turns the per-opponent push/fold counts (``aof_cfr.opponent_db``) into an estimated
strategy and a measure of how exploitable that opponent is. AoF makes this tractable: an
opponent's strategy IS their push frequency per infoset, directly observed, so the estimate
is a Bayesian posterior with the GTO base as prior:

    p_push = (push + s*base_p) / (total + s)            # s = prior strength (pseudocounts)
    confidence = total / (total + s)

Unknown / low-data infosets collapse to the GTO base (hard invariant 1). Per-player
exploitability = the EV the opponent spills (their own best-response gain) when they play
the estimated policy and the rest of the table plays the base — reusing the legal BR metric
(``exploit_mc.estimate_nash_conv_mc``), which works for any player count.
"""

from __future__ import annotations

import numpy as np

from aof_cfr.buckets import COMBOS_169, LABELS_169
from aof_cfr.exploit_mc import estimate_nash_conv_mc
from aof_cfr.fp_4max import _histories_before
from aof_cfr.infoset import aggregate_infoset_key, encode_infoset_key
from aof_cfr.opponent_db import AoFOpponentDB
from aof_ranges import active_seats

DEFAULT_PRIOR_STRENGTH = 8.0

# Combo weights = TRUE dealt-hand frequency per bucket: pairs 6, suited 4, offsuit 12.
# NOTE: ``len(COMBOS_169[i])`` is WRONG for this — those lists store suit-canonical
# REPRESENTATIVES (offsuit keeps only 4 of its 12 combos), which silently underweighted
# offsuit hands 3x in every combo-weighted aggregate (base_agg targets, range-width shifts,
# strength order). Classify by label instead.
def _true_combo_count(label: str) -> float:
    if len(label) == 2:          # pair ("AA")
        return 6.0
    return 4.0 if label.endswith("s") else 12.0


_COMBO_W = np.array([_true_combo_count(lab) for lab in LABELS_169], dtype=np.float64)
_COMBO_W_SUM = float(_COMBO_W.sum())
assert _COMBO_W_SUM == 1326.0    # 13*6 + 78*4 + 78*12 — the full deck's combo count

_STRENGTH_ORDER_CACHE: dict[str, np.ndarray] = {}


def _strength_order_desc() -> np.ndarray:
    """Bucket indices sorted strongest->weakest by combo-weighted HU all-in equity vs random.

    A player-count-agnostic strength ranking used to widen/tighten a range while keeping
    premiums in. Reuses the shipped HU equity matrix (always present with the base).
    """
    if "o" not in _STRENGTH_ORDER_CACHE:
        from aof_cfr.hu_br_legal import load_hu_eq

        eq = load_hu_eq()  # eq[h, j] = hand h equity vs hand j
        strength = (eq * _COMBO_W[None, :]).sum(axis=1) / _COMBO_W_SUM
        _STRENGTH_ORDER_CACHE["o"] = np.argsort(-strength, kind="stable")
    return _STRENGTH_ORDER_CACHE["o"]


def _query_base_push(num_players: int, position: str, history: str, hand: str, stack_bb: float) -> float:
    from aof_cfr.query import base_strategy

    return base_strategy(num_players, position, history, hand, stack_bb)


def posterior_push(push: int, total: int, base_p: float, prior_strength: float) -> float:
    s = max(float(prior_strength), 1e-9)   # guard 0/0 when prior_strength==0 and total==0
    return (push + s * base_p) / (total + s)


def infoset_confidence(total: int, prior_strength: float) -> float:
    s = max(float(prior_strength), 1e-9)   # guard 0/0 when prior_strength==0 and total==0
    return total / (total + s)


def _seat_infosets(num_players: int, seat_idx: int, stack_bb: float):
    seat = active_seats(num_players)[seat_idx]
    # The last seat with everyone before it folded wins uncontested -> no decision infoset
    # (the solvers omit it: e.g. HU BB|F, 3-max BB|FF). Skip those phantoms.
    phantom_hist = "F" * seat_idx if seat_idx == num_players - 1 else None
    for lab in LABELS_169:
        for hist in _histories_before(seat_idx):
            if hist == phantom_hist:
                continue
            yield seat, lab, hist, encode_infoset_key(seat, lab, hist, stack_bb)


_BASE_ROWS_CACHE: dict = {}
_BASE_ARR_CACHE: dict = {}


def base_policy_rows(num_players: int, stack_bb: float, *, base_push_fn=_query_base_push) -> dict[str, np.ndarray]:
    """Full table policy = GTO base at every infoset. Cached for the default base (callers
    must not mutate the returned dict — they build new structures from it)."""
    use_cache = base_push_fn is _query_base_push
    ck = (num_players, round(float(stack_bb), 3))
    if use_cache and ck in _BASE_ROWS_CACHE:
        return _BASE_ROWS_CACHE[ck]
    rows: dict[str, np.ndarray] = {}
    for si in range(num_players):
        for seat, lab, hist, key in _seat_infosets(num_players, si, stack_bb):
            p = float(base_push_fn(num_players, seat, hist, lab, stack_bb))
            rows[key] = np.array([1.0 - p, p], dtype=np.float64)
    if use_cache:
        _BASE_ROWS_CACHE[ck] = rows
    return rows


def _base_arrays_seathist(num_players: int, stack_bb: float, base_push_fn) -> dict:
    """Cached ``(seat, history) -> base push freq (169,)`` for the default base."""
    use_cache = base_push_fn is _query_base_push
    ck = (num_players, round(float(stack_bb), 3))
    if use_cache and ck in _BASE_ARR_CACHE:
        return _BASE_ARR_CACHE[ck]
    out: dict = {}
    for si in range(num_players):
        seat = active_seats(num_players)[si]
        phantom_hist = "F" * si if si == num_players - 1 else None
        for hist in _histories_before(si):
            if hist == phantom_hist:
                continue
            out[(seat, hist)] = np.array(
                [float(base_push_fn(num_players, seat, hist, lab, stack_bb)) for lab in LABELS_169],
                dtype=np.float64,
            )
    if use_cache:
        _BASE_ARR_CACHE[ck] = out
    return out


def estimate_policy_rows(
    db: AoFOpponentDB,
    identity: str,
    num_players: int,
    stack_bb: float,
    *,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    base_push_fn=_query_base_push,
) -> dict[str, np.ndarray]:
    """Bayesian posterior push policy for ``identity`` over all their infosets (GTO prior)."""
    rows: dict[str, np.ndarray] = {}
    for si in range(num_players):
        for seat, lab, hist, key in _seat_infosets(num_players, si, stack_bb):
            base_p = float(base_push_fn(num_players, seat, hist, lab, stack_bb))
            push, total = db.cell(identity, key)
            p = posterior_push(push, total, base_p, prior_strength)
            rows[key] = np.array([1.0 - p, p], dtype=np.float64)
    return rows


def _range_width_shift(base_p: np.ndarray, p_target: float) -> np.ndarray:
    """Shift the base push range along strength order to hit deal-weighted aggregate ``p_target``.

    Tighten (target < base aggregate) by removing the weakest pushers first; loosen by adding
    the strongest folders first. Preserves strength monotonicity and premiums, and collapses to
    ``base_p`` exactly when the target equals the base aggregate.
    """
    new_p = base_p.astype(np.float64).copy()
    base_agg = float((_COMBO_W * base_p).sum() / _COMBO_W_SUM)
    deficit = (float(p_target) - base_agg) * _COMBO_W_SUM  # signed push-weight to add/remove
    order = _strength_order_desc()
    if deficit < 0:  # tighten: drop weakest pushers (iterate weak -> strong)
        need = -deficit
        for h in order[::-1]:
            if need <= 1e-12:
                break
            take = min(base_p[h] * _COMBO_W[h], need)
            new_p[h] -= take / _COMBO_W[h]
            need -= take
    elif deficit > 0:  # loosen: promote strongest folders (iterate strong -> weak)
        need = deficit
        for h in order:
            if need <= 1e-12:
                break
            take = min((1.0 - base_p[h]) * _COMBO_W[h], need)
            new_p[h] += take / _COMBO_W[h]
            need -= take
    return np.clip(new_p, 0.0, 1.0)


# Hierarchical depth pooling (Phase 5): a depth cell shrinks toward the player's all-depth
# (seat, hist) pool before the pool shrinks toward base — undoing the 7-bucket fragmentation
# (~4-7x effective sample early). POOL_WEIGHT=0 + no cross-depth data collapses exactly to
# the original single-cell estimator.
POOL_PRIOR_STRENGTH = 10.0   # s2: pooled rate -> base shrinkage
POOL_WEIGHT = 0.5            # rho: pooled observations count at half weight toward eff. n


def _pooled_aggregate(cells: dict, seat: str, hist: str, stack_bb: float):
    """(k_cell, n_cell, k_pool, n_pool): own-depth aggregate cell + the OTHER depth buckets'
    cells for the same (seat, hist)."""
    from aof_cfr.infoset import STACK_DEPTH_BUCKETS
    own_key = aggregate_infoset_key(seat, hist, stack_bb)
    k_cell, n_cell = cells.get(own_key, (0, 0))
    k_pool = n_pool = 0
    for d in STACK_DEPTH_BUCKETS:
        key = aggregate_infoset_key(seat, hist, float(d))
        if key == own_key:
            continue
        p_, t_ = cells.get(key, (0, 0))
        k_pool += p_
        n_pool += t_
    return k_cell, n_cell, k_pool, n_pool


def exact_bucket_override(est_p: np.ndarray, cells: dict, seat: str, hist: str,
                          stack_bb: float, prior_strength: float) -> np.ndarray:
    """Two-level posterior: where showdown reveals produced exact-bucket observations, the
    per-bucket cell posterior overrides the range-width reconstruction (whose value becomes
    the bucket's PRIOR MEAN). No reveals -> returns ``est_p`` unchanged (byte-identical)."""
    out = est_p
    for bi, lab in enumerate(LABELS_169):
        cell = cells.get(encode_infoset_key(seat, lab, hist, stack_bb))
        if cell and cell[1] > 0:
            if out is est_p:
                out = est_p.copy()
            pb, tb = cell
            out[bi] = (pb + prior_strength * float(est_p[bi])) / (tb + prior_strength)
    return out


def estimate_policy_rows_shift(
    db: AoFOpponentDB,
    identity: str,
    num_players: int,
    stack_bb: float,
    *,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    base_push_fn=_query_base_push,
    pool_strength: float = POOL_PRIOR_STRENGTH,
) -> dict[str, np.ndarray]:
    """Hidden-card posterior policy via the range-width shift (the live-play estimator).

    Three-level read per ``(seat, history)``:
      1. depth-pooled (seat, hist) rate shrunk toward base (``pool_strength``),
      2. own-depth cell shrunk toward the pool (``prior_strength``) -> range-width shift,
      3. exact-bucket posteriors from showdown reveals override the shift where observed.
    Unknown ``(seat, history)`` -> exactly the base. Output shape matches ``estimate_policy_rows``.
    """
    rows: dict[str, np.ndarray] = {}
    base_arrs = _base_arrays_seathist(num_players, stack_bb, base_push_fn)
    cells = db.cells_for(identity)
    for (seat, hist), base_p in base_arrs.items():
        k_cell, n_cell, k_pool, n_pool = _pooled_aggregate(cells, seat, hist, stack_bb)
        if n_cell + n_pool > 0:
            base_agg = float((_COMBO_W * base_p).sum() / _COMBO_W_SUM)
            m_pool = (k_pool + pool_strength * base_agg) / (n_pool + pool_strength)
            target = (k_cell + prior_strength * m_pool) / (n_cell + prior_strength)
            est_p = _range_width_shift(base_p, target)
        else:
            est_p = base_p
        est_p = exact_bucket_override(est_p, cells, seat, hist, stack_bb, prior_strength)
        for bi, lab in enumerate(LABELS_169):
            p = float(est_p[bi])
            rows[encode_infoset_key(seat, lab, hist, stack_bb)] = np.array([1.0 - p, p], dtype=np.float64)
    return rows


def per_player_exploitability(
    db: AoFOpponentDB,
    identity: str,
    num_players: int,
    opp_seat_idx: int,
    stack_bb: float,
    *,
    samples: int = 6000,
    seed: int = 42,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    base_push_fn=_query_base_push,
) -> dict:
    """How exploitable ``identity`` is at seat ``opp_seat_idx`` (bb/hand they spill).

    Builds a table where the opponent seat plays the estimated policy and every other seat
    plays the GTO base, then measures the legal BR gain per seat. ``opp_spilled_bb`` (the
    opponent's own BR gain) is the EV they leave on the table by deviating — the size of the
    exploit available. Others-at-base isolates this opponent's contribution.
    """
    base_rows = base_policy_rows(num_players, stack_bb, base_push_fn=base_push_fn)
    est_rows = estimate_policy_rows(
        db, identity, num_players, stack_bb, prior_strength=prior_strength, base_push_fn=base_push_fn
    )
    table = dict(base_rows)
    for _seat, _lab, _hist, key in _seat_infosets(num_players, opp_seat_idx, stack_bb):
        table[key] = est_rows[key]

    nc, imps, _method = estimate_nash_conv_mc(
        num_players, table, samples=samples, seed=seed, stack_bb=stack_bb
    )
    return {
        "identity": identity,
        "opp_seat": active_seats(num_players)[opp_seat_idx],
        "opp_spilled_bb": imps[opp_seat_idx],
        "table_improvements_bb": imps,
        "nash_conv_bb": nc,
        "n_obs": db.n_obs(identity),
    }
