"""ProductionHero — wraps the real production policy (HybridAgent or SearchAgent) in the lightweight
``ExploitHero`` protocol the exploitation battery's A/B loop expects, so the battery reports trustworthy
production EV uplift instead of the heuristic value-bettor's noisy proxy.

The battery only needs a hero to: be ``__call__(state) -> Action``, accept a single villain profile via
``set_profile(OpponentStats)``, surface per-decision ``(probs, baseline_values, taken_idx)`` for AIVAT
via ``pop_decisions()``, expose ``bet_freq``, and (new) ``on_new_hand()`` so a stateful agent can reset
its per-hand trajectory. This adapter bridges any agent that exposes ``enable_decision_capture(rng)`` +
``pop_decisions_raw()`` (added to both HybridAgent and SearchAgent).

AIVAT correctness: the agent MUST sample from its policy (not argmax) for the decision control variate to
be zero-mean — the adapter enables seeded sampling via ``enable_decision_capture(rng)``. The per-action
AIVAT baseline ``bv(a)`` is the agent's own leaf-net Q-row when it provides one (SearchAgent), else an
optional ``value_fn``, else the cheap hand-strength × commitment proxy (the estimator is unbiased for any
baseline; the baseline only affects variance).
"""

from __future__ import annotations

import numpy as np

from pokerbot.engine.game import GameState
from pokerbot.training.self_play import _legal_mask

NUM_ACTIONS = 11
# per-vocab commitment proxy (matches exploitation_battery._COMMIT) — the checkpoint-free fallback bv.
_COMMIT = np.array([0.0, 0.0, 0.30, 0.50, 0.55, 0.60, 0.65, 0.75, 0.85, 0.95, 1.0], dtype=np.float64)


def _crude_baseline(state: GameState, seat: int) -> np.ndarray:
    from pokerbot.agent import _hero_hand_strength

    hs = _hero_hand_strength(state, seat)
    hs = 0.5 if hs is None else float(np.clip(hs, 0.0, 1.0))
    return (hs - 0.5) * 2.0 * _COMMIT


class ProductionHero:
    """Adapt a production agent to the battery's hero protocol."""

    def __init__(self, agent, rng, *, exploit: bool = True, value_fn=None,
                 hero_seat: int = 0, n_seats: int = 6) -> None:
        self.agent = agent
        self.rng = rng
        self.exploit = exploit
        self.value_fn = value_fn
        self.hero_seat = hero_seat
        self.n_seats = n_seats
        self._profile = None
        self._dec: list[tuple] = []
        self.bet_mass = 0.0
        self.n_dec = 0
        # Seeded sampling ⇒ AIVAT decision correction is zero-mean. Both arms share the per-hand seed
        # (set via on_new_hand / the battery's CRN), so a decision the exploit doesn't change is identical.
        if hasattr(agent, "enable_decision_capture"):
            agent.enable_decision_capture(rng)

    @property
    def bet_freq(self) -> float:
        return self.bet_mass / self.n_dec if self.n_dec else 0.0

    def reseed(self, seed: int) -> None:
        """CRN per-hand reseed: both arms share the seed so an unchanged decision plays identically."""
        import random

        self.rng = random.Random(seed)

    def _apply_profile(self) -> None:
        if self._profile is None or not self.exploit:
            prof = None
        elif isinstance(self._profile, dict):
            # PER-SEAT profiles (run_field_absolute per-seat mode / live PlayerProfileStore): pass the
            # {seat: OpponentStats} map through verbatim so the root mixer targets the aggressor in
            # front of the hero. The single-OpponentStats branch below stays the broadcast default,
            # byte-identical to the legacy pooled measurement.
            prof = self._profile
        else:
            prof = {i: self._profile for i in range(self.n_seats)}
        # Clean A/B: the OFF arm is the PURE GTO/blueprint base (no profile, no tracker mixing).
        # HybridAgent honors set_exploit_enabled (alpha=0 ⇒ pure GTO); SearchAgent's OFF is already
        # pure (opponent_model=None ⇒ no solver-internal bias, no root mixer).
        if hasattr(self.agent, "set_exploit_enabled"):
            self.agent.set_exploit_enabled(self.exploit)
        if hasattr(self.agent, "set_opponent_profiles"):
            self.agent.set_opponent_profiles(prof)
        elif hasattr(self.agent, "solver"):
            self.agent.solver.opponent_model = prof

    def set_profile(self, profile) -> None:
        self._profile = profile
        self._apply_profile()

    def on_new_hand(self) -> None:
        if hasattr(self.agent, "reset_hand"):
            self.agent.reset_hand(self.hero_seat)
        # reset_hand may rebuild solver/tracker state; re-apply the profile + re-arm capture.
        self._apply_profile()
        if hasattr(self.agent, "enable_decision_capture"):
            self.agent.enable_decision_capture(self.rng)

    def pop_decisions(self) -> list[tuple]:
        d, self._dec = self._dec, []
        return d

    def pop_chance_corrections(self) -> list[float]:
        return []

    def __call__(self, state: GameState):
        seat = state.action_seat
        action = self.agent(state)
        raw = self.agent.pop_decisions_raw() if hasattr(self.agent, "pop_decisions_raw") else []
        for dec in raw:
            if len(dec) == 3 and dec[1] is not None:
                probs, bv, idx = dec
                bv = np.asarray(bv, dtype=np.float64).reshape(-1)
            else:
                probs, idx = (dec[0], dec[-1])
                if self.value_fn is not None:
                    legal = np.asarray(_legal_mask(state, seat), dtype=bool)
                    bv = np.asarray(self.value_fn(state, seat, legal), dtype=np.float64).reshape(-1)
                else:
                    bv = _crude_baseline(state, seat)
            probs = np.asarray(probs, dtype=np.float64).reshape(-1)
            self._dec.append((probs.copy(), bv, int(idx)))
            self.bet_mass += float(probs[3:].sum())
            self.n_dec += 1
        return action
