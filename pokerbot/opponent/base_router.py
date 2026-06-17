"""Pluggable GTO base + table-toughness router (exploitation task (b)).

The stated goal is to *extract maximum value from weak players while protecting against strong
players, with the focus on value extraction*. The validated read->mixer overlay (HybridPolicy +
HybridAgent) extracts value by deviating from a GTO base toward an opponent-specific read. Its base
is the SFT policy net, which learned human-exploitative tendencies from data and therefore tends to
extract MORE from a loose-passive field than a pure-equilibrium blueprint would (a GTO blueprint is
non-exploitative by construction). So the SFT base is the right *value* vehicle.

The risk is the opposite end: against a strong, near-equilibrium reg, deviating from GTO is exactly
what gets us counter-exploited, and the SFT base itself is below equilibrium. The protective move is
to (a) fall back to the stronger blueprint base and (b) stop deviating. This router decides, PER
DECISION and PER MODELED VILLAIN, where on that spectrum we are:

    toughness = conf * (1 - min(1, tv / TV_REF))

where ``tv`` is the total-variation distance of the villain's read from the near-Nash reference
(``ContextRead.tv``) and ``conf`` is the sample-size confidence ``N/(N+s)``. Interpreted:

  * cold start (conf ~ 0)            -> toughness ~ 0 -> SFT base + (confidence-gated) exploit. We do
                                        not protect against an unknown; the overlay is already safe
                                        when it has no read.
  * confirmed fish (tv large)        -> toughness ~ 0 -> SFT base + full exploit (the value focus).
  * confirmed reg (conf high, tv ~0) -> toughness high -> blueprint base + damped/zero exploit.

Because the decision keys on the *modeled villain's* profile (the seat the mixer is exploiting this
decision), a mixed table is handled naturally: we keep extracting from the fish and protect only when
the pot is against the confirmed reg.

Default-safe: a HybridAgent with no router attached is byte-identical to the validated SFT vehicle.
The thresholds here are deliberately conservative starting points; ``scripts/decide_base_routing.py``
re-fits them from the ``--mode basestrength`` measurement (which quantifies, for the first time, the
base's absolute strength vs weak AND strong fields for both candidate bases).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUM_ACTIONS = 11
# Villain-vs-Nash TV that counts as "fully exploitable" — matches bayes_exploit._TV_FULL (the same
# scale the mixer's own extremity uses), so toughness and exploit extremity share one yardstick.
TV_REF = 0.25
# Toughness at/above which we switch the base to the blueprint AND drive exploit to zero (full
# protection). Below it, exploit is damped linearly toward zero as toughness rises.
TOUGH_HIGH = 0.5


@dataclass(frozen=True)
class RouteDecision:
    """Per-decision routing verdict. ``exploit_scale`` multiplies the agent's exploit_strength."""

    toughness: float
    use_blueprint: bool
    exploit_scale: float


def villain_toughness(stats, *, tv_ref: float = TV_REF) -> float:
    """Confidence-weighted near-equilibrium score in [0, 1] for one modeled villain (see module
    docstring). ``None``/unreadable/no-data ⇒ 0.0 (treat as unknown ⇒ do not protect)."""
    if stats is None:
        return 0.0
    try:
        from pokerbot.opponent.bayes_exploit import aggregate_read

        read = aggregate_read(stats)
    except Exception:  # noqa: BLE001 — a read failure must never break a live decision
        read = None
    if read is None:
        return 0.0
    leak = min(1.0, float(read.tv) / max(1e-9, float(tv_ref)))
    return float(max(0.0, min(1.0, float(read.conf) * (1.0 - leak))))


class BlueprintBaseProvider:
    """Returns the CFR blueprint's mixed-strategy 11-vector for a ``(state, seat)`` — the strong GTO
    base the read->mixer falls back to on tough tables. Wraps the SAME ``TabularBlueprintAdapter`` that
    ``pokerbot.eval.bots.BlueprintBot`` uses, so the base is exactly the near-Nash strategy measured by
    the basestrength battery. ``probs()`` never raises: a fingerprint miss / bridge failure returns
    ``None`` so the caller falls back to the SFT base (do-no-harm)."""

    def __init__(self, snapshot=None, *, client_bet_sizing: bool = True) -> None:
        from pokerbot.eval.lbr import check_blueprint_serve_env
        from pokerbot.search.tabular_blueprint_adapter import TabularBlueprintAdapter

        # Warn if the abstraction env wasn't set before import (⇒ ~0% postflop blueprint coverage).
        check_blueprint_serve_env()
        if snapshot is None:
            from pokerbot.eval.bots import _load_blueprint_snapshot

            snapshot = _load_blueprint_snapshot()
        self.adapter = TabularBlueprintAdapter(snapshot, client_bet_sizing=client_bet_sizing)

    def probs(self, state, seat: int) -> np.ndarray | None:
        try:
            from pokerbot.search.state_bridge import game_state_to_env

            env = game_state_to_env(state, seat)
            p = np.asarray(self.adapter.blueprint(env, seat), dtype=np.float64).reshape(-1)
            s = float(p.sum())
            if p.shape[0] == NUM_ACTIONS and s > 0.0 and np.isfinite(p).all():
                return p / s
        except Exception:  # noqa: BLE001
            pass
        return None


class BaseRouter:
    """Per-decision base + exploit router (see module docstring).

    ``blueprint_provider``: a ``BlueprintBaseProvider`` (or anything with ``probs(state, seat) ->
    (11,) | None``). If ``None``, the router only damps exploit_strength by toughness (it never swaps
    the base) — a useful protect-only mode that needs no blueprint snapshot.

    ``enabled=False`` ⇒ a no-op router (the validated SFT vehicle, unchanged) — handy for A/B arms.
    """

    def __init__(self, blueprint_provider=None, *, tough_high: float = TOUGH_HIGH,
                 tv_ref: float = TV_REF, enabled: bool = True) -> None:
        self.blueprint = blueprint_provider
        self.tough_high = float(tough_high)
        self.tv_ref = float(tv_ref)
        self.enabled = bool(enabled)

    def decide(self, villain_stats) -> RouteDecision:
        if not self.enabled:
            return RouteDecision(0.0, False, 1.0)
        t = villain_toughness(villain_stats, tv_ref=self.tv_ref)
        use_bp = (self.blueprint is not None) and (t >= self.tough_high)
        # Exploit ramps DOWN to zero by the switch point: full exploit on fish (t=0 ⇒ 1.0), no
        # deviation once we judge the villain a reg (t≥tough_high ⇒ 0.0 ⇒ pure base = max protection).
        scale = 1.0 - (t / self.tough_high if self.tough_high > 0 else 1.0)
        return RouteDecision(t, use_bp, float(max(0.0, min(1.0, scale))))

    def base_probs(self, state, seat: int) -> np.ndarray | None:
        """The blueprint base 11-vector for this decision, or ``None`` (⇒ fall back to the SFT base)."""
        if self.blueprint is None:
            return None
        return self.blueprint.probs(state, seat)
