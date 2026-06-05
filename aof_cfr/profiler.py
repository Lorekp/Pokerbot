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

from aof_cfr.buckets import LABELS_169
from aof_cfr.exploit_mc import estimate_nash_conv_mc
from aof_cfr.fp_4max import _histories_before
from aof_cfr.infoset import encode_infoset_key
from aof_cfr.opponent_db import AoFOpponentDB
from aof_ranges import active_seats

DEFAULT_PRIOR_STRENGTH = 8.0


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


def base_policy_rows(num_players: int, stack_bb: float, *, base_push_fn=_query_base_push) -> dict[str, np.ndarray]:
    """Full table policy = GTO base at every infoset."""
    rows: dict[str, np.ndarray] = {}
    for si in range(num_players):
        for seat, lab, hist, key in _seat_infosets(num_players, si, stack_bb):
            p = float(base_push_fn(num_players, seat, hist, lab, stack_bb))
            rows[key] = np.array([1.0 - p, p], dtype=np.float64)
    return rows


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
