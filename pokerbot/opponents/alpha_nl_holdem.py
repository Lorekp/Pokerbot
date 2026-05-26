"""AlphaNLHoldem external opponent bridge (TensorFlow, HU 50BB)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.game import GameState


ALPHA_ROOT = Path(__file__).resolve().parents[2] / "external" / "AlphaNLHoldem"
_VENV313 = ALPHA_ROOT / ".venv313" / "Scripts" / "python.exe"
_VENV = ALPHA_ROOT / ".venv" / "Scripts" / "python.exe"
ALPHA_VENV_PY = _VENV313 if _VENV313.exists() else _VENV
INFER_SCRIPT = Path(__file__).resolve().parent / "alpha_nl_holdem_infer.py"


def resolve_league_dir(alpha_root: Path | None = None) -> Path:
    """Return league/history_agents, fixing common double-nested extract layouts."""
    root = alpha_root or ALPHA_ROOT
    expected = root / "league" / "history_agents"
    nested = expected / "league" / "history_agents"
    if (nested / "weights").exists() and not (expected / "weights").exists():
        return nested
    return expected


class AlphaNLHoldemUnavailable(Exception):
    pass


def _state_to_obs(state: GameState, seat: int) -> dict:
    player = state.players[seat]
    hole = ""
    if player.hole:
        hole = player.hole[0].to_str() + player.hole[1].to_str()
    return {
        "seat": seat,
        "street": state.street.name,
        "board": [c.to_str() for c in state.board],
        "hole": hole,
        "pot_bb": state.pot_bb,
        "current_bet_bb": state.current_bet_bb,
        "stack_bb": player.stack_bb,
        "action_history": [(s, a.label()) for s, a in state.action_history()],
    }


def _parse_action_response(data: dict, state: GameState, seat: int) -> Action:
    legal = state.legal_actions(seat)
    act = data.get("action", "fold").lower()
    amount = float(data.get("amount", 0))
    if act == "fold":
        return Action(ActionType.FOLD)
    if act == "check":
        return next((a for a in legal if a.action_type == ActionType.CHECK), legal[0])
    if act == "call":
        return next((a for a in legal if a.action_type == ActionType.CALL), legal[0])
    if act in ("bet", "raise"):
        for a in legal:
            if a.action_type in (ActionType.BET, ActionType.RAISE) and abs(a.amount_bb - amount) < 1.5:
                return a
        bets = [a for a in legal if a.action_type in (ActionType.BET, ActionType.RAISE)]
        return bets[-1] if bets else legal[0]
    if act in ("allin", "all in"):
        return next((a for a in legal if a.action_type == ActionType.ALL_IN), legal[-1])
    return legal[0]


class AlphaNLHoldemBot:
    """
    Subprocess bridge to AlphaNLHoldem when installed.
    Falls back to AlphaNLHoldemFallbackBot when unavailable.
    """

    def __init__(self, alpha_root: Path | None = None, python_exe: str | None = None) -> None:
        self.alpha_root = alpha_root or ALPHA_ROOT
        self.python_exe = python_exe or (str(ALPHA_VENV_PY) if ALPHA_VENV_PY.exists() else None)
        self._fallback = AlphaNLHoldemFallbackBot()
        self._proc: subprocess.Popen | None = None
        self._available = self._check_available()

    def _check_available(self) -> bool:
        league = resolve_league_dir(self.alpha_root)
        weights = league / "weights"
        has_ckpt = weights.exists() and any(weights.glob("*.pkl"))
        return self.alpha_root.exists() and has_ckpt and INFER_SCRIPT.exists()

    def _ensure_proc(self) -> None:
        if self._proc is not None:
            return
        if not self._available:
            raise AlphaNLHoldemUnavailable("AlphaNLHoldem not installed")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.alpha_root)
        py = self.python_exe or sys.executable
        self._proc = subprocess.Popen(
            [py, str(INFER_SCRIPT), str(self.alpha_root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

    def __call__(self, state: GameState) -> Action:
        seat = state.action_seat
        if not self._available:
            return self._fallback(state)
        try:
            self._ensure_proc()
            assert self._proc and self._proc.stdin and self._proc.stdout
            obs = _state_to_obs(state, seat)
            self._proc.stdin.write(json.dumps(obs) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                return self._fallback(state)
            return _parse_action_response(json.loads(line), state, seat)
        except Exception:
            return self._fallback(state)

    def close(self) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc = None


class AlphaNLHoldemFallbackBot:
    """Strong heuristic HU bot when AlphaNLHoldem checkpoints are missing."""

    def __call__(self, state: GameState) -> Action:
        import random

        seat = state.action_seat
        legal = state.legal_actions(seat)
        player = state.players[seat]
        if player.hole is None:
            return legal[0]
        r1, r2 = int(player.hole[0].rank), int(player.hole[1].rank)
        strength = max(r1, r2) + min(r1, r2) * 0.1 + (5 if r1 == r2 else 0)

        if state.street.name == "PREFLOP":
            if strength > 22:
                bets = [a for a in legal if a.action_type in (ActionType.RAISE, ActionType.BET)]
                if bets:
                    return bets[min(1, len(bets) - 1)]
            if strength < 18:
                return Action(ActionType.FOLD) if Action(ActionType.FOLD) in legal else legal[0]

        if strength > 24 and random.random() < 0.4:
            bets = [a for a in legal if a.action_type in (ActionType.BET, ActionType.RAISE)]
            if bets:
                return bets[-1]
        return next(
            (a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)),
            legal[0],
        )
