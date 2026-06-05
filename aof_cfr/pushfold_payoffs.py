"""Shared 3-max push/fold static payoffs and non-showdown terminal chip-EV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STACK_BB = 8.0
SB_POST = 0.5
BB_POST = 1.0
BLINDS_DEAD = SB_POST + BB_POST
POT_HU_STACKS = 2.0 * STACK_BB + BLINDS_DEAD
POT_THREE_STACKS = 3.0 * STACK_BB + BLINDS_DEAD

SB_HIST = ("F", "A")
BB_HIST = ("FA", "AF", "AA")


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
    shares: np.ndarray, pair_eq: np.ndarray, *, stack_bb: float = STACK_BB
) -> StaticPayoffs:
    """Showdown utilities from combo-averaged tensor (169^3) at effective depth ``stack_bb``.

    Equity (shares / pair_eq) is depth-invariant; only the pot size and the stack subtracted
    scale with ``stack_bb`` (pot = k*stack_bb + dead blinds). One tensor serves all depths.
    """
    pot_hu = 2.0 * stack_bb + BLINDS_DEAD
    pot_three = 3.0 * stack_bb + BLINDS_DEAD
    pot_sb_bb = pot_btn_bb = pot_btn_sb = pot_hu
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
