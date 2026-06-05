"""Legal (imperfect-information) best-response marginals for heads-up AoF push/fold.

The HU analogue of ``fp_br._legal_marginals_3max``: for each (seat, history) it returns
``(E_action, E_fold, reach_w)`` per own-hand bucket AFTER marginalizing the opponent's hidden
hand, so the legal best response is pure (act iff ``E_action > E_fold``). ``route_hu`` reads
from here.

Chip model is the project's HU convention (``aof_cfr.terminal_ev.hu_*``), the one the shipped
``cfr_base_2p`` policy was solved against, so the base is a near-fixed-point of these marginals:

  * deal weights: uniform over the 169x169 ordered bucket pairs (``solver_hu_dcfr`` deal_w);
  * ``eq[i, j]`` = SB(bucket i) equity vs BB(bucket j);
  * SB fold -> -0.5; SB jam, BB fold -> +1.0 (SB wins BB's blind);
  * both jam (showdown): SB -> ``eq*pot - stack``, BB -> ``(1-eq)*pot - stack`` with
    ``pot = 1.5 + 2*stack_bb``. (The +1.5 dead-blind term lives only on the showdown branch,
    so the table is zero-sum up to that dead money, not exactly zero-sum.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aof_cfr.hu_payoff_matrix import load_equity_matrix, solver_hu_deal_weights

DEFAULT_HU_EQ = Path("data/aof/cfr_base/hu_eq_matrix.npz")

SB_FOLD = -0.5      # SB folds the small blind
SB_PUSH_BB_FOLD = 1.0   # SB jams, BB folds: SB wins the big blind
BB_FOLD = -1.0      # BB folds the big blind facing a jam

_EQ_CACHE: dict[str, np.ndarray] = {}


def load_hu_eq(path: Path | str = DEFAULT_HU_EQ) -> np.ndarray:
    """Cached (169,169) SB-equity bucket matrix (eq[sb, bb]); consistent with cfr_base_2p."""
    key = str(path)
    if key not in _EQ_CACHE:
        _EQ_CACHE[key] = load_equity_matrix(Path(path))
    return _EQ_CACHE[key]


def _pot_called(stack_bb: float) -> float:
    return 1.5 + 2.0 * float(stack_bb)


def _legal_marginals_hu(
    sb_push: np.ndarray,
    bb_push: np.ndarray,
    stack_bb: float,
    eq: np.ndarray | None = None,
) -> dict[str, tuple[np.ndarray, float, np.ndarray]]:
    """Legal HU BR marginals. ``sb_push`` = SB jam freq (169,), ``bb_push`` = BB call freq (169,).

    Returns ``{"sb": (E_push, E_fold, reach_w), "bb_a": (E_call, E_fold, reach_w)}`` where the
    legal best response is pure: act iff ``E_action > E_fold``.
    """
    if eq is None:
        eq = load_hu_eq()
    n = int(eq.shape[0])
    w = solver_hu_deal_weights(n)            # (n,n) uniform 1/n^2
    pot = _pot_called(stack_bb)
    sb_show = eq * pot - float(stack_bb)     # (i=SB, j=BB) SB chip-EV at showdown
    bb_show = (1.0 - eq) * pot - float(stack_bb)  # BB chip-EV at showdown
    eps = 1e-12

    # SB marginal (axis 0 = SB hand i), history "-": unconditional reach over BB's hand j.
    bb_call = bb_push[None, :]               # (1, n)
    u_sb_jam = (1.0 - bb_call) * SB_PUSH_BB_FOLD + bb_call * sb_show  # (n, n)
    sb_w = np.sum(w, axis=1)                  # (n,) = 1/n
    sb_ep = np.where(sb_w > eps, np.sum(w * u_sb_jam, axis=1) / np.maximum(sb_w, eps), 0.0)

    # BB marginal (axis 1 = BB hand j), history "A": reach conditioned on SB actually jamming.
    wj = w * sb_push[:, None]                 # (n, n) weighted by SB jam reach
    bb_w = np.sum(wj, axis=0)                  # (n,)
    bb_ep = np.where(bb_w > eps, np.sum(wj * bb_show, axis=0) / np.maximum(bb_w, eps), 0.0)

    return {
        "sb": (sb_ep, float(SB_FOLD), sb_w),
        "bb_a": (bb_ep, float(BB_FOLD), bb_w),
    }


def seat_values_hu(
    sb_push: np.ndarray,
    bb_push: np.ndarray,
    stack_bb: float,
    eq: np.ndarray | None = None,
) -> list[float]:
    """Exact on-policy expected chip-EV (bb/hand) [SB, BB] for a full HU table.

    Sum is the dead-blind constant ``1.5 * P(both jam)``, not zero (the +1.5 convention),
    so use this for head-to-head bb evaluation, not as a zero-sum check.
    """
    if eq is None:
        eq = load_hu_eq()
    n = int(eq.shape[0])
    w = solver_hu_deal_weights(n)
    pot = _pot_called(stack_bb)
    sb_show = eq * pot - float(stack_bb)
    bb_show = (1.0 - eq) * pot - float(stack_bb)
    psb = sb_push[:, None]
    pbb = bb_push[None, :]

    u_sb_jam = (1.0 - pbb) * SB_PUSH_BB_FOLD + pbb * sb_show
    sb_val = float(np.sum(w * ((1.0 - psb) * SB_FOLD + psb * u_sb_jam)))

    # BB: SB folds -> BB wins SB blind (+0.5); SB jams, BB folds -> -1.0; both jam -> bb_show.
    bb_outcome = (1.0 - psb) * 0.5 + psb * ((1.0 - pbb) * BB_FOLD + pbb * bb_show)
    bb_val = float(np.sum(w * bb_outcome))
    return [sb_val, bb_val]
