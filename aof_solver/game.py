"""HU push/fold toy game tree for CFR skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Action = Literal["F", "A"]  # fold, all-in (push/call)


@dataclass(frozen=True)
class HuGameConfig:
    stack_bb: float = 8.0
    sb_bb: float = 0.5
    bb_bb: float = 1.0


@dataclass
class HuState:
    """Public action history: SB acts first, then BB if SB pushed."""

    actions: tuple[Action, ...] = ()

    def is_terminal(self) -> bool:
        if not self.actions:
            return False
        if self.actions[-1] == "F":
            return True
        return len(self.actions) >= 2

    def current_player(self) -> Literal["SB", "BB"] | None:
        if self.is_terminal():
            return None
        if not self.actions:
            return "SB"
        if self.actions == ("A",):
            return "BB"
        return None

    def legal_actions(self) -> tuple[Action, ...]:
        if self.is_terminal():
            return ()
        return ("F", "A")

    def apply(self, action: Action) -> HuState:
        return HuState(actions=self.actions + (action,))


def terminal_utility_sb(
    state: HuState,
    *,
    sb_equity_vs_bb_call: float,
    cfg: HuGameConfig,
) -> float:
    """
    Chip EV for SB in bb units (zero-sum BB = -utility).

    Simplified: fold wins blinds; shove-fold wins pot; call uses ``sb_equity_vs_bb_call``.
    """
    pot = cfg.sb_bb + cfg.bb_bb
    if state.actions == ("F",):
        return -cfg.sb_bb
    if state.actions == ("A", "F"):
        return pot
    if state.actions == ("A", "A"):
        invest = cfg.stack_bb - cfg.sb_bb
        return sb_equity_vs_bb_call * (pot + 2 * invest) - invest
    return 0.0
