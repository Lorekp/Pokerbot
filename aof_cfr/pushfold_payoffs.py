"""Shared push/fold terminal chip model (all seat counts) + 3-max static payoffs.

GAME MODEL VERSIONS — the single switch for the contested-pot convention:

  * v1 (legacy): pot = k*stack + 1.5 dead blinds, all-in cost = full stack. This DOUBLE-COUNTS
    alive players' blinds (an SB jammer's 0.5 is both inside their 8bb commitment and added as
    "dead" money), making the game non-zero-sum (+1.5 appears from nothing at every showdown)
    and the equilibrium ~3.75 equity points too loose on call thresholds.
  * v2 (correct): the posted blind counts TOWARD the jam — a jamming stack's total commitment
    is exactly ``stack_bb`` — so pot = k*stack + the FOLDED seats' blinds only (HU contested
    16bb @8bb, not 17.5). Matches the real client rule and the repo's own ``aof_multiway``
    pot math. Zero-sum identity: sum(payoffs) == -rake.

Fold-branch terminals (folders lose their blind; an uncontested survivor collects the folded
blinds) are IDENTICAL in both versions — only contested pots differ.

The runtime default tracks the SHIPPED base's manifest (payoff layer and artifacts must never
mix models); ``AOF_GAME_MODEL`` overrides for dev/rollback; solver scripts pass the model
explicitly when producing new artifacts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STACK_BB = 8.0
SB_POST = 0.5
BB_POST = 1.0
BLINDS_DEAD = SB_POST + BB_POST
POT_HU_STACKS = 2.0 * STACK_BB + BLINDS_DEAD
POT_THREE_STACKS = 3.0 * STACK_BB + BLINDS_DEAD

SB_HIST = ("F", "A")
BB_HIST = ("FA", "AF", "AA")

# The model THIS code produces for new solves (manifest stamps it into artifacts).
GAME_MODEL_VERSION = 2
_MANIFEST_PATH = Path("data/aof/cfr_base/cfr_base.manifest.json")
_GM_CACHE: dict = {}


def game_model_default() -> int:
    """Runtime game model: env ``AOF_GAME_MODEL`` override, else the shipped manifest's
    ``game_model_version`` (the model the base artifacts were SOLVED under), else 1."""
    env = os.environ.get("AOF_GAME_MODEL")
    if env not in (None, ""):
        return int(env)
    if "v" not in _GM_CACHE:
        try:
            _GM_CACHE["v"] = int(json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
                                 .get("game_model_version", 1))
        except (OSError, ValueError):
            _GM_CACHE["v"] = 1
    return _GM_CACHE["v"]


# Official CoinPoker Hold'em flat-fee schedule (re-confirmed by the user 2026-06-15 against the
# published rake table), keyed by (sb, bb) in dollars. Per DC2 capture evidence the fee is deducted
# from the CONTESTED pot at showdown (folded players pay nothing; an uncontested steal pays nothing).
# Every tier is an 8bb buy-in; the fee is the only thing that varies by stake. (Omaha is OUT of scope.)
OFFICIAL_HOLDEM_FEE_BB = {
    (0.10, 0.25): 0.16,                                            # buy-in $2
    (0.25, 0.50): 0.12, (0.50, 1.0): 0.12, (1.0, 2.0): 0.12,      # buy-in $4 / $8 / $16
    (2.0, 5.0): 0.12, (5.0, 10.0): 0.12, (10.0, 20.0): 0.12,      # buy-in $40 / $80 / $160
    (50.0, 100.0): 0.10,                                           # buy-in $800
}
FLAT_FEE_BB_OFFICIAL = 0.12  # the modal fee (0.25/0.50 through 10/20); also the solved-base fee


def holdem_fee_bb(sb, bb=None) -> float:
    """Official Hold'em flat fee (in bb) for a stake tier. Robust to float noise and to unlisted
    tiers: the schedule is uniquely keyed by the BIG BLIND (each bb has one Hold'em fee), so match on
    bb (the SB is informational); fall back to the modal 0.12 for any stake not in the table. Call as
    ``holdem_fee_bb(sb, bb)`` or ``holdem_fee_bb(bb)``."""
    if bb is None:
        bb = sb
    try:
        target = round(float(bb), 4)
    except (TypeError, ValueError):
        return FLAT_FEE_BB_OFFICIAL
    for (_s, b), fee in OFFICIAL_HOLDEM_FEE_BB.items():
        if abs(b - target) < 1e-6:
            return fee
    return FLAT_FEE_BB_OFFICIAL


def flat_fee_default() -> float:
    """Flat fee (bb) deducted from CONTESTED pots before payout. Incidence pinned by the
    DC2 capture (2026-06-12, 0.25/0.50 AoF): folded players pay nothing; the winner's
    payout is pot - fee; uncontested steals untouched. Magnitude: the OFFICIAL schedule
    (0.12BB at 0.25/0.50 Hold'em) per user direction — the captured hand's chip deltas
    showed 0.22BB retained vs the withoutRake field's 0.12BB, a discrepancy parked for
    re-examination with more captures. Env ``AOF_FLAT_FEE_BB`` overrides the shipped
    manifest's ``flat_fee_bb``; 0.0 when neither is set (legacy no-fee artifacts)."""
    env = os.environ.get("AOF_FLAT_FEE_BB")
    if env not in (None, ""):
        return float(env)
    if "fee" not in _GM_CACHE:
        try:
            _GM_CACHE["fee"] = float(json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
                                     .get("flat_fee_bb", 0.0))
        except (OSError, ValueError):
            _GM_CACHE["fee"] = 0.0
    return _GM_CACHE["fee"]


def contested_pot_bb(k_alive: int, folded_blinds_bb: float, stack_bb: float,
                     *, game_model: int | None = None, fee_bb: float | None = None) -> float:
    """Contested pot for ``k_alive`` all-in stacks with ``folded_blinds_bb`` dead, net of
    the flat showdown fee (zero-sum identity: sum(payoffs) == -fee)."""
    gm = game_model_default() if game_model is None else int(game_model)
    fee = flat_fee_default() if fee_bb is None else float(fee_bb)
    if gm >= 2:
        return k_alive * stack_bb + folded_blinds_bb - fee
    return k_alive * stack_bb + BLINDS_DEAD - fee


@dataclass(frozen=True)
class TerminalEV:
    """Pencil-and-paper chip-EV for fold / uncontested jam terminals (8bb stacks)."""

    btn_fold: float = 0.0
    sb_fold_after_btn_fold: float = -SB_POST
    bb_fold_after_btn_sb_fold: float = SB_POST
    sb_win_uncontested_after_btn_fold: float = BB_POST
    btn_win_uncontested_after_sb_fold: float = SB_POST + BB_POST
    sb_fold_after_btn_jam: float = -SB_POST
    bb_fold: float = -BB_POST


TERMINAL = TerminalEV()


def audit_terminal_ev() -> None:
    """Fail loudly if terminal constants disagree with AoF blind/post rules."""
    t = TERMINAL
    assert t.btn_fold == 0.0
    assert t.sb_fold_after_btn_fold == -0.5
    assert t.bb_fold_after_btn_sb_fold == 0.5
    assert abs(t.sb_win_uncontested_after_btn_fold - BB_POST) < 1e-12
    assert abs(t.btn_win_uncontested_after_sb_fold - BLINDS_DEAD) < 1e-12
    assert abs(POT_HU_STACKS - (2.0 * STACK_BB + BLINDS_DEAD)) < 1e-12
    assert abs(POT_THREE_STACKS - (3.0 * STACK_BB + BLINDS_DEAD)) < 1e-12
    assert t.sb_fold_after_btn_jam == -0.5
    assert t.bb_fold == -1.0


@dataclass
class StaticPayoffs:
    sb_bb_call1: np.ndarray
    sb_bb_call2: np.ndarray
    btn_bb_call0: np.ndarray
    btn_bb_call2: np.ndarray
    btn_sb_call0: np.ndarray
    btn_sb_call1: np.ndarray
    three0: np.ndarray
    three1: np.ndarray
    three2: np.ndarray


def build_static_payoffs(
    shares: np.ndarray, pair_eq: np.ndarray, *, stack_bb: float = STACK_BB,
    game_model: int | None = None,
) -> StaticPayoffs:
    """Showdown utilities from combo-averaged tensor (169^3) at effective depth ``stack_bb``.

    Equity (shares / pair_eq) is depth-invariant; only the pot size and the stack subtracted
    scale with ``stack_bb``. One tensor serves all depths. Under game model v2 the contested
    pots are history-dependent (the FOLDED seat's blind is the only dead money): sb_bb +0
    (BTN posts nothing), btn_bb +0.5, btn_sb +1.0, three-way +0.
    """
    pot_sb_bb = contested_pot_bb(2, 0.0, stack_bb, game_model=game_model)       # BTN folded
    pot_btn_bb = contested_pot_bb(2, SB_POST, stack_bb, game_model=game_model)  # SB folded
    pot_btn_sb = contested_pot_bb(2, BB_POST, stack_bb, game_model=game_model)  # BB folded
    pot_three = contested_pot_bb(3, 0.0, stack_bb, game_model=game_model)
    sb_bb = pair_eq[..., 2]
    btn_bb = pair_eq[..., 0]
    btn_sb = pair_eq[..., 1]
    sh = shares
    return StaticPayoffs(
        sb_bb_call1=sb_bb * pot_sb_bb - stack_bb,
        sb_bb_call2=(1.0 - sb_bb) * pot_sb_bb - stack_bb,
        btn_bb_call0=btn_bb * pot_btn_bb - stack_bb,
        btn_bb_call2=(1.0 - btn_bb) * pot_btn_bb - stack_bb,
        btn_sb_call0=btn_sb * pot_btn_sb - stack_bb,
        btn_sb_call1=(1.0 - btn_sb) * pot_btn_sb - stack_bb,
        three0=sh[..., 0] * pot_three - stack_bb,
        three1=sh[..., 1] * pot_three - stack_bb,
        three2=sh[..., 2] * pot_three - stack_bb,
    )


def mix(p: np.ndarray, u_fold: float, u_push: np.ndarray) -> np.ndarray:
    return (1.0 - p) * u_fold + p * u_push


def mix_into(
    out: np.ndarray,
    p: np.ndarray,
    u_fold: float | np.ndarray,
    u_push: np.ndarray,
    *,
    work: np.ndarray,
) -> np.ndarray:
    """Write ``(1-p)*u_fold + p*u_push`` into *out* without extra (n,n,n) temporaries."""
    np.multiply(1.0 - p, u_fold, out=out)
    np.multiply(p, u_push, out=work)
    out += work
    return out
