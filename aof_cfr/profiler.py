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

# Combo weights (dealt-hand frequency) per bucket: pairs 6, suited 4, offsuit 12.
_COMBO_W = np.array([len(c) for c in COMBOS_169], dtype=np.float64)
_COMBO_W_SUM = float(_COMBO_W.sum())

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
    return (push + prior_strength * base_p) / (total + prior_strength)


def infoset_confidence(total: int, prior_strength: float) -> float:
    return total / (total + prior_strength)


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


def estimate_policy_rows_shift(
    db: AoFOpponentDB,
    identity: str,
    num_players: int,
    stack_bb: float,
    *,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    base_push_fn=_query_base_push,
) -> dict[str, np.ndarray]:
    """Hidden-card posterior policy via the range-width shift (the live-play estimator).

    For each ``(seat, history)`` reads the single aggregate cell (hand-agnostic push/fold count),
    forms the posterior aggregate push rate ``(push + s*base_agg)/(total + s)``, and reconstructs
    a per-bucket policy by widening/tightening the GTO base to match it (premiums preserved).
    Unknown ``(seat, history)`` -> exactly the base. Output shape matches ``estimate_policy_rows``.
    """
    rows: dict[str, np.ndarray] = {}
    base_arrs = _base_arrays_seathist(num_players, stack_bb, base_push_fn)
    for (seat, hist), base_p in base_arrs.items():
        push, total = db.cell(identity, aggregate_infoset_key(seat, hist, stack_bb))
        if total > 0:
            base_agg = float((_COMBO_W * base_p).sum() / _COMBO_W_SUM)
            est_p = _range_width_shift(base_p, posterior_push(push, total, base_agg, prior_strength))
        else:
            est_p = base_p
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
