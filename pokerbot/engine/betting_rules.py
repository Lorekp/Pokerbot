"""Single source of truth for the NLHE reopen / minimum-raise rule (BUG-B4 unification).

Before this module the reopen rule was hand-re-derived in FIVE places. They fall into TWO legitimately
distinct conventions:

  * ENGINE-ABSTRACTION rule — ``jax_engine.apply_action`` / ``np_engine`` / ``core.rs`` (one vectorized
    rule transliterated into three languages, bit-identity-locked). For CFR/search the engine LOWERS the
    minimum raise to ``max(BB, raise_size)`` on ANY raise, including a short all-in under-raise.
  * REAL-GAME (strict TDA) rule — ``game.py`` (eval engine modelling the actual game) and
    ``live/normalized_to_state.py`` (the live serve, which must present the SAME legal-raise menu the
    real CoinPoker client enforces). Here a short all-in under-raise advances the standing bet but does
    NOT change the minimum raise: it stays at the last FULL raise increment. Lowering it would let the
    serve suggest an under-min raise the client rejects, and would mis-model the real game in eval.

``game.py`` and ``normalized_to_state`` shared the REAL-GAME rule but each hand-rolled it. This module is
the ONE canonical scalar implementation of that rule; both now consume it (behaviour unchanged, the
duplication removed). The engine triad keeps its own gated abstraction rule — the divergence on the
short-all-in-under-raise edge is DELIBERATE (abstraction vs real game), not an accident.

Canonical (real-game / strict) rule:
  * An aggressive action that pushes the street bet above the standing bet is a RAISE.
  * The standing bet advances to the new total.
  * The round REOPENS (the acted set resets to just the raiser) iff the raise is FULL, i.e.
    ``raise_size >= prior_min_raise``.
  * The minimum raise INCREMENT becomes ``max(bb, raise_size)`` only on a FULL raise; a short all-in
    under-raise leaves it at ``prior_min_raise`` (the last full raise).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReopenResult:
    raised: bool          # the action increased the standing bet
    full_raise: bool      # the raise reopens the round (resets the acted set)
    new_current_bet: float  # the new standing bet (== new_bet on a raise, else prev_bet)
    new_min_raise: float  # the minimum legal raise INCREMENT for the next raiser
    resets_acted: bool    # alias of full_raise, kept explicit for call-site readability


def reopen_step(prev_bet: float, new_bet: float, prior_min_raise: float, bb: float,
                *, eps: float = 0.0) -> ReopenResult:
    """Canonical reopen / min-raise update for an aggressive action that brought a seat's street
    commitment to ``new_bet`` total (the standing bet was ``prev_bet``; ``prior_min_raise`` is the current
    minimum raise increment; ``bb`` is one big blind in the caller's unit). Unit-agnostic — chips
    (``eps=0``, exact) or BB floats (``eps=1e-9``). Pure + side-effect free.
    """
    if new_bet <= prev_bet + eps:
        return ReopenResult(False, False, prev_bet, prior_min_raise, False)
    raise_size = new_bet - prev_bet
    full_raise = raise_size >= prior_min_raise - eps
    # Strict (real-game) rule: the min raise advances only on a FULL raise; a short all-in under-raise
    # leaves it at the last full raise. (The engine-abstraction rule instead lowers it on any raise.)
    new_min_raise = max(bb, raise_size) if full_raise else prior_min_raise
    return ReopenResult(True, full_raise, new_bet, new_min_raise, full_raise)
