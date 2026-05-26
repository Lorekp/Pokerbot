"""Action types and discretized bet sizing aligned with solver buckets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class ActionType(Enum):
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    BET = auto()
    RAISE = auto()
    ALL_IN = auto()


# Solver-native pot fractions used by PokerBench / GTO solvers
BET_SIZE_BUCKETS = (0.33, 0.50, 0.66, 0.75, 1.0, 1.5, 2.0)


@dataclass(frozen=True, slots=True)
class Action:
    action_type: ActionType
    amount_bb: float = 0.0  # total chips committed this action, in BB units

    def label(self) -> str:
        if self.action_type == ActionType.FOLD:
            return "fold"
        if self.action_type == ActionType.CHECK:
            return "check"
        if self.action_type == ActionType.CALL:
            return "call"
        if self.action_type == ActionType.ALL_IN:
            return "all in"
        verb = "bet" if self.action_type == ActionType.BET else "raise"
        return f"{verb} {self.amount_bb:.1f}".rstrip("0").rstrip(".")


# Discrete action space for policy head (~12 actions)
# Index 0-5: fold, check, call, bet/raise at bucket sizes, all-in
ACTION_VOCAB = [
    "fold",
    "check",
    "call",
    "bet_0.33",
    "bet_0.50",
    "bet_0.66",
    "bet_0.75",
    "bet_1.00",
    "bet_1.50",
    "bet_2.00",
    "all_in",
]

NUM_ACTIONS = len(ACTION_VOCAB)


def action_to_index(action: Action) -> int:
    if action.action_type == ActionType.FOLD:
        return 0
    if action.action_type == ActionType.CHECK:
        return 1
    if action.action_type == ActionType.CALL:
        return 2
    if action.action_type == ActionType.ALL_IN:
        return NUM_ACTIONS - 1
    # bet or raise — map to nearest bucket by pot fraction stored in amount_bb
    best_idx = 3
    best_dist = float("inf")
    for i, frac in enumerate(BET_SIZE_BUCKETS, start=3):
        dist = abs(action.amount_bb - frac)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def index_to_bucket(index: int) -> float | None:
    if index < 3 or index >= NUM_ACTIONS - 1:
        return None
    return BET_SIZE_BUCKETS[index - 3]


def parse_pokerbench_output(text: str) -> Action:
    """Parse PokerBench label like 'check', 'fold', 'call', 'bet 5.0', 'all in'."""
    text = text.strip().lower()
    if text == "fold":
        return Action(ActionType.FOLD)
    if text == "check":
        return Action(ActionType.CHECK)
    if text == "call":
        return Action(ActionType.CALL)
    if text in ("all in", "allin", "all-in"):
        return Action(ActionType.ALL_IN)
    if text.startswith("bet "):
        return Action(ActionType.BET, float(text.split()[1]))
    if text.startswith("raise "):
        return Action(ActionType.RAISE, float(text.split()[1]))
    raise ValueError(f"Unknown action label: {text!r}")


def normalize_decision_for_eval(predicted: str, target: str) -> tuple[str, str]:
    """Normalize bet/raise labels for AA/EM comparison."""
    def norm(s: str) -> str:
        s = s.strip().lower()
        if s.startswith("bet "):
            return f"bet {float(s.split()[1]):.1f}"
        if s.startswith("raise "):
            return f"raise {float(s.split()[1]):.1f}"
        return s

    return norm(predicted), norm(target)


def action_accuracy(predicted: str, target: str) -> bool:
    pred, tgt = normalize_decision_for_eval(predicted, target)
    pred_base = pred.split()[0]
    tgt_base = tgt.split()[0]
    if pred_base in ("bet", "raise") and tgt_base in ("bet", "raise"):
        return True
    return pred_base == tgt_base


def exact_match(predicted: str, target: str) -> bool:
    pred, tgt = normalize_decision_for_eval(predicted, target)
    return pred == tgt
