"""4-max AoF push/fold fictitious play with combo-deal sampling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.exploit_mc import estimate_nash_conv_mc, sampled_br_policy_mc
from aof_cfr.fp_br import StabilityTracker
from aof_cfr.infoset import encode_infoset_key
from aof_ranges import active_seats

NUM_PLAYERS = 4


@dataclass
class FourMaxFPSolveResult:
    rounds: int
    exploitability_bb: float
    policy_rows: dict[str, np.ndarray]
    player_improvements: list[float]
    br_samples: int
    br_method: str
    converged: bool
    manifest: dict = field(default_factory=dict)


def _histories_before(seat_idx: int) -> list[str]:
    from aof_cfr.infoset import history_after_action

    out: list[str] = []

    def rec(step: int, hist: str) -> None:
        if step >= seat_idx:
            out.append(hist if hist else "-")
            return
        rec(step + 1, history_after_action(hist, "F"))
        rec(step + 1, history_after_action(hist, "A"))

    rec(0, "")
    return out


def _init_uniform_rows(stack_bb: float) -> dict[str, np.ndarray]:
    seats = active_seats(NUM_PLAYERS)
    rows: dict[str, np.ndarray] = {}
    for si, seat in enumerate(seats):
        for lab in LABELS_169:
            for hist in _histories_before(si):
                key = encode_infoset_key(seat, lab, hist, stack_bb)
                rows[key] = np.array([0.5, 0.5], dtype=np.float64)
    return rows


def solve_4max_fp(
    *,
    max_rounds: int = 200,
    br_samples: int = 4000,
    br_per_round: int = 300,
    gap_every: int = 20,
    seed: int = 42,
    damping: float = 1.0,
    stack_bb: float = 8.0,
    stability: StabilityTracker | None = None,
) -> FourMaxFPSolveResult:
    """Averaged FP with CRN Nash-gap measurement.

    The per-deal MC tree walk is the cost (pure Python ~tens of ms/sample), so: (a) the FP
    target uses light per-round BR samples (``br_per_round``) — averaging tolerates noise; and
    (b) the (expensive, split) Nash gap is measured only every ``gap_every`` rounds and at the
    end, not every round. A converged production 4-max base still wants the vectorized/C++ BR
    (the spec's escalation); these only make the Python path tractable for moderate solves.
    """
    avg = _init_uniform_rows(stack_bb)
    tracker = stability or StabilityTracker(gap_tol=3.0, policy_tol=0.002, window=5)
    gap = 0.0
    imps: list[float] = [0.0] * NUM_PLAYERS
    method = "mc_combo_br_legal_split"
    converged = False
    per_player = br_per_round

    def _measure(rnd: int) -> bool:
        nonlocal gap, imps, method, converged
        gap, imps, method = estimate_nash_conv_mc(
            NUM_PLAYERS, avg, samples=br_samples, seed=seed + rnd, stack_bb=stack_bb
        )
        utg_keys = sorted(k for k in avg if k.startswith("UTG|"))[:169]
        utg = np.array([avg[k][1] for k in utg_keys]) if len(utg_keys) == 169 else np.zeros(169)
        if tracker.observe(gap * 1000.0, utg):
            converged = True
        return converged

    for rnd in range(1, max_rounds + 1):
        merged = dict(avg)
        for p in range(NUM_PLAYERS):
            br_p = sampled_br_policy_mc(
                NUM_PLAYERS,
                avg,
                br_player=p,
                samples=per_player,
                seed=seed + 10_000 * rnd + p,
                stack_bb=stack_bb,
            )
            for key, row in br_p.items():
                if key in merged:
                    merged[key] = 0.5 * (merged[key] + row)
                else:
                    merged[key] = row
        alpha = damping / rnd
        for key in avg:
            if key in merged:
                avg[key] = (1.0 - alpha) * avg[key] + alpha * merged[key]
        if gap_every and rnd % gap_every == 0 and _measure(rnd):
            break
    if gap == 0.0:  # ensure a final gap is always reported
        _measure(rnd)

    return FourMaxFPSolveResult(
        rounds=rnd,
        exploitability_bb=gap,
        policy_rows=avg,
        player_improvements=imps,
        br_samples=br_samples,
        br_method=method,
        converged=converged,
        manifest={
            "rounds": rnd,
            "exploitability_bb": gap,
            "br_samples": br_samples,
            "converged": converged,
        },
    )
