"""Action types and discretized bet sizing aligned with solver buckets."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class ActionType(Enum):
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    BET = auto()
    RAISE = auto()
    ALL_IN = auto()


# Solver-native pot fractions used by PokerBench / GTO solvers.
#
# SINGLE SOURCE OF TRUTH for the bet-size abstraction menu. The engine state machine
# (``jax_engine`` / ``np_engine`` / ``core.rs``), the serve/translation path
# (``action_to_index`` / ``pseudo_harmonic_bucket``) and the trained policy head ALL key off this
# one tuple. The smallest bucket is the exact rational ``1/3`` (33.33% pot), NOT ``0.33`` — the
# engine has always sized at ``1/3``; the serve menu previously used ``0.33`` and silently
# mis-bucketed the smallest size relative to the trained policy. Unified to ``1/3`` here so
# ``jax_engine.BET_SIZE_BUCKETS is actions.BET_SIZE_BUCKETS`` (same object) and every consumer agrees.
#
# NOTE: the EXTERNAL postflop-solver CLI / CoinPoker client use a literal ``0.33`` ("33% of pot",
# see ``coinpoker_bet_sizes`` and ``solver_exploitability._BET_BUCKET_FRACS``) — that is a DISTINCT
# menu aligned to a different system and is intentionally NOT this constant.
BET_SIZE_BUCKETS = (1.0 / 3, 0.50, 0.66, 0.75, 1.0, 1.5, 2.0)


@dataclass(frozen=True, slots=True)
class Action:
    action_type: ActionType
    amount_bb: float = 0.0  # raise increment above the call, in BB units (NOT a pot fraction)
    # Resolved abstraction bucket index (into ACTION_VOCAB), set by the PRODUCER when the originating
    # pot-fraction / menu position is known (engine ``legal_actions`` knows it exactly; the live replay
    # computes it from the pot AT ACTION TIME). -1 = not set. ``action_to_index`` returns this directly
    # when >= 0, which is the FIX for the serve-path convention clash: ``amount_bb`` is an absolute BB
    # increment, but ``action_to_index``/``BET_SIZE_BUCKETS`` bucket a POT FRACTION, so bucketing a raw
    # ``amount_bb`` collapses every raise to the top bucket. The training/PokerBench path never sets this
    # (stays -1), so that path is byte-identical. Excluded from eq/hash so equality semantics are unchanged.
    bucket: int = field(default=-1, compare=False)

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
    # FIX (serve path): when the producer resolved the abstraction bucket (engine legal_actions, or the
    # live replay using pot-at-action), trust it — bucketing the raw absolute-BB ``amount_bb`` against the
    # pot-fraction BET_SIZE_BUCKETS would collapse every raise to the top bucket. Training Actions never
    # set ``bucket`` (stays -1), so the pot-fraction path below is byte-identical for training/encoding.
    if action.bucket >= 0:
        return int(action.bucket)
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


def pseudo_harmonic_bucket(x: float, buckets=BET_SIZE_BUCKETS) -> int:
    """Map an off-tree size ``x`` to a bucket index (0-based into ``buckets``) by the *pseudo-harmonic*
    mapping (Ganzfried & Sandholm 2013), not nearest-by-absolute-distance.

    Nearest-by-distance is the specifically exploitable translation the literature warns against: an
    opponent bets just inside a bucket boundary to swing our response a full bucket for a negligible
    size change. Pseudo-harmonic is the lowest-exploitability *deterministic* no-compute mapping: for a
    size ``x`` bracketed by abstraction sizes ``A < x < B`` it maps to the SMALLER size ``A`` with
    probability ``f = (B - x)(1 + A) / ((B - A)(1 + x))`` (and to ``B`` with ``1 - f``). We take the
    deterministic argmax (``A`` iff ``f >= 0.5``), which biases toward the smaller bet — exactly the
    anti-exploitation property (the attacker can't cheaply buy the bigger-bucket response).

    Out-of-range clamps to the end buckets. Pure + side-effect free; ``buckets`` must be ascending."""
    n = len(buckets)
    if n == 0:
        return 0
    if x <= buckets[0]:
        return 0
    if x >= buckets[-1]:
        return n - 1
    for i in range(n - 1):
        a, b = buckets[i], buckets[i + 1]
        if a <= x <= b:
            # probability mass the pseudo-harmonic mapping assigns to the SMALLER size A
            denom = (b - a) * (1.0 + x)
            f_a = ((b - x) * (1.0 + a) / denom) if denom > 0 else 1.0
            return i if f_a >= 0.5 else i + 1
    return n - 1  # unreachable for ascending buckets, but keep total


def action_to_index_pseudo_harmonic(action: Action) -> int:
    """Serve-time variant of ``action_to_index`` that translates an opponent's off-tree bet/raise size
    via ``pseudo_harmonic_bucket`` instead of nearest-by-absolute-distance. Fold/check/call/all-in are
    identical to ``action_to_index``. Use ONLY on the live serve path (opponent-action → belief/policy
    query); training/encoding paths MUST keep calling the raw ``action_to_index`` so the trained policy
    head's bucket fingerprint stays byte-identical."""
    if action.bucket >= 0:
        return int(action.bucket)  # producer already resolved the bucket (pot-at-action) — trust it
    if action.action_type == ActionType.FOLD:
        return 0
    if action.action_type == ActionType.CHECK:
        return 1
    if action.action_type == ActionType.CALL:
        return 2
    if action.action_type == ActionType.ALL_IN:
        return NUM_ACTIONS - 1
    return 3 + pseudo_harmonic_bucket(action.amount_bb, BET_SIZE_BUCKETS)


def bucket_index_for_bet(bet_to_bb: float, pot_before_bb: float) -> int:
    """Abstraction action index (3..NUM_ACTIONS-2) for a bet/raise whose STREET bet-TO total is
    ``bet_to_bb`` made into ``pot_before_bb`` (the whole-hand pot BEFORE this action) — matching the
    training menu ``target = max(current_bet, pot * frac)`` ⇒ ``frac = bet_to / pot_before``. Uses the
    pseudo-harmonic mapping (anti-exploitation). This is how the live replay resolves an observed
    off-tree opponent/hero size to the trained abstraction bucket (the value to put in ``Action.bucket``)."""
    frac = float(bet_to_bb) / max(float(pot_before_bb), 1e-9)
    return 3 + pseudo_harmonic_bucket(frac, BET_SIZE_BUCKETS)


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
