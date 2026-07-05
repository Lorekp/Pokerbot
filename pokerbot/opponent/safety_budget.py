"""Live gift / safety budget — banked opponent mistakes fund extra exploitation headroom.

Grounded in Ganzfried & Sandholm, *Safe Opponent Exploitation* (BEFEWP): a **gift** is the opponent
playing a strategy that is not a best response to some equilibrium — i.e. they surrendered EV. Safe
exploitation is possible exactly to the extent the opponent has given gifts; banking cumulative gifts
``k`` and deviating only as far as ``k`` funds guarantees we never risk below the value of the game.

POSTURE = gift-modulated bounded-ε (user-chosen): rather than gating deviation to ZERO until gifts
accrue (strict BEFEWP), a banked gift RELAXES the existing confidence-gated Pmax cap — we exploit a
demonstrably-leaky opponent HARDER, while the offline ``added_exploitability`` ε-proof remains the
floor. With zero banked gift the bonus is exactly 0, so behavior is bit-identical to the static cap.

Crediting: at an opponent decision the EV they surrendered is ``gift = max(0, maxₐ Q(h,a) − Q(h,
a_taken))`` from the leaf value net (available via the production hero / search). Crediting per
decision (gift may be 0) also DECAYS the budget, so it tracks RECENT exploitable headroom — a player
who leaked early then tightened up fades back toward GTO.

The register is kept OUT of the pinned ``OpponentStats`` schema (it is exploitation headroom, not a
recorded tendency); it lives parallel to the solver's ``opponent_model`` and is reset/owned by the
live driver or the battery.
"""

from __future__ import annotations

import math

# Most extra alpha a fully-banked budget may add on top of the base Pmax (bayes_exploit.DEFAULT_MAX_ALPHA
# = 0.7). ALPHA_CEILING hard-caps the TOTAL; the offline added_exploitability proof calibrates these so
# that even at MAX_BONUS the added exploitability on a population-like profile stays within ε (50 mbb).
MAX_BONUS = 0.15
GIFT_SCALE = 2.0      # bb of banked gift mapping to ~tanh(1)=0.76 of MAX_BONUS
GIFT_DECAY = 0.98     # per-credit decay ⇒ recent-headroom, not lifetime accumulation
ALPHA_CEILING = 0.85  # hard cap on total alpha (base Pmax + gift bonus)


def gift_to_alpha_bonus(available: float, *, max_bonus: float = MAX_BONUS,
                        scale: float = GIFT_SCALE) -> float:
    """Saturating, monotone map banked-gift → extra alpha headroom. 0 ⇒ 0 (no behavior change)."""
    a = max(0.0, float(available))
    return float(max_bonus * math.tanh(a / max(scale, 1e-9)))


class GiftRegister:
    """Per-opponent banked-gift budget + the alpha-cap relaxation it funds."""

    def __init__(self, *, decay: float = GIFT_DECAY, max_bonus: float = MAX_BONUS,
                 scale: float = GIFT_SCALE, ceiling: float = ALPHA_CEILING) -> None:
        self._budget: dict = {}
        self.decay = float(decay)
        self.max_bonus = float(max_bonus)
        self.scale = float(scale)
        self.ceiling = float(ceiling)
        # H7.3 — cumulative SESSION ledger (bb), the BEFEWP bound: gift-funded EXTRA deviation across the
        # whole session may never exceed the gifts actually received, so we never risk below the value of
        # the game. Cumulative (NOT decayed) unlike the per-key recent-headroom budget above.
        self._session_banked = 0.0
        self._session_spent = 0.0

    def credit(self, key, amount_bb: float = 0.0) -> None:
        """Observe one opponent decision: decay the running per-key budget then add the gift (≥0; 0 when
        the action was a best response). Per-decision decay makes the per-key budget track recent
        headroom; the cumulative session ledger also banks the raw gift (BEFEWP)."""
        amt = max(0.0, float(amount_bb or 0.0))
        self._budget[key] = self._budget.get(key, 0.0) * self.decay + amt
        self._session_banked += amt

    def available(self, key) -> float:
        return float(self._budget.get(key, 0.0))

    def spend(self, key, amount: float) -> None:
        """Debit realized gift-funded deviation from BOTH the per-key budget and the cumulative session
        ledger (so the session bound tightens as banked gifts are consumed)."""
        amt = max(0.0, float(amount))
        self._budget[key] = max(0.0, self.available(key) - amt)
        self._session_spent += amt

    def session_available(self) -> float:
        """Cumulative banked − spent gifts this session (bb). When 0, no gift-funded extra deviation is
        permitted (fall back to the base bounded-ε floor) — the hard BEFEWP session guarantee."""
        return max(0.0, self._session_banked - self._session_spent)

    def reset_session(self) -> None:
        """Start a fresh session ledger (call at session boundaries; per-key budgets are independent)."""
        self._session_banked = 0.0
        self._session_spent = 0.0

    def alpha_bonus(self, key) -> float:
        """Extra alpha headroom funded by ``key``'s banked gifts, HARD-CAPPED by the cumulative session
        budget (the BEFEWP bound). 0 when either is exhausted ⇒ base bounded-ε only. Zero gift at cold
        start ⇒ 0 ⇒ bit-identical to the static cap."""
        per_key = gift_to_alpha_bonus(self.available(key), max_bonus=self.max_bonus, scale=self.scale)
        session = gift_to_alpha_bonus(self.session_available(), max_bonus=self.max_bonus, scale=self.scale)
        return min(per_key, session)

    def capped_max_alpha(self, key, base_max_alpha: float) -> float:
        """Relax the base Pmax cap by the gift bonus, hard-capped at the ceiling. Zero gift ⇒
        returns ``base_max_alpha`` exactly (bit-identical to the static cap)."""
        return float(min(self.ceiling, float(base_max_alpha) + self.alpha_bonus(key)))


# P0 safety wiring (exploitation overhaul design §A3): one PROCESS-scoped register shared by every
# live advisor and the post-hand outcome join. A live coach process == one session (multi-table
# coaches share the same PlayerProfileStore, so cross-table sharing keyed by profile identity is
# correct — the same human's banked gifts follow them between tables). Batteries/tests that need
# isolation construct their own ``GiftRegister`` and pass it explicitly (the advisor only falls
# back to the session register when the caller passes none).
_SESSION_REGISTER: GiftRegister | None = None


def session_gift_register() -> GiftRegister:
    """The process-scoped session ``GiftRegister`` (lazily constructed)."""
    global _SESSION_REGISTER
    if _SESSION_REGISTER is None:
        _SESSION_REGISTER = GiftRegister()
    return _SESSION_REGISTER


def gift_from_qrow(q_row, taken_idx: int, legal_mask=None) -> float:
    """The EV the opponent surrendered at a decision = max legal Q − Q(taken), clamped ≥0. ``q_row`` is
    the leaf value net's per-action Q-row (see aivat.baseline_from_leaf_net / search leaf-Q)."""
    import numpy as np

    q = np.asarray(q_row, dtype=np.float64).reshape(-1)
    if q.size == 0 or not (0 <= int(taken_idx) < q.size):
        return 0.0
    if legal_mask is not None:
        lm = np.asarray(legal_mask).reshape(-1).astype(bool)
        if lm.shape == q.shape and lm.any():
            qmax = float(q[lm].max())
        else:
            qmax = float(q.max())
    else:
        qmax = float(q.max())
    return max(0.0, qmax - float(q[int(taken_idx)]))
