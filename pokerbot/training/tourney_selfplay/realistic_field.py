"""realistic_field — league-backed villain ``decide_fn``s for the MTT env.

Replaces the push/fold-quantile villains in the lifecycle env with league members (design component
#9): villain seats are sampled population-proportionally (calling_station ~47% at the FISH tier with
λ≈∞, a thin reg/pro tail at small λ) and each is labeled to a ``(clone_tier, λ)`` member. The realistic
field the hero trains against.

This module is the ADAPTER between a :class:`LiftedAgent` (which speaks "τ-logits + Q over a legal
mask") and the lifecycle ``DecideFn`` (which speaks "a ``SpotContext`` → ``SpotDecision`` threshold").
The lifecycle works at the confrontation abstraction (jam/call by strength quantile), so the adapter
maps a LiftedAgent's commit-probability into a fold-below threshold — exactly the mapping
``mtt_lifecycle.net_decide`` uses for the neural net, but driven by the clone's piKL policy instead.

Two builders:
  * :func:`lifted_agent_decide_fn(agent)` — wrap ONE LiftedAgent as a ``DecideFn``.
  * :func:`sample_field_decide_fns(league, seed, n)` — draw a population-proportional villain field
    from ``field_sampler.archetype_probabilities`` mapped onto the league's FROZEN_HUMAN tiers.

Because the lifecycle exposes only ONE ``villain_decide_fn`` (a single field policy) today, the
population draw is summarized to a representative villain ``decide_fn`` (the modal archetype's tier)
for the reference backend; the PER-SEAT heterogeneous field is the Rust sim's job (the design's
``agent_specs`` list). The full draw is still returned so a richer backend can use it.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

import numpy as np

from pokerbot.eval import field_sampler
from pokerbot.tournament import mtt_lifecycle as _life
from pokerbot.training.tourney_selfplay import pikl
from pokerbot.training.tourney_selfplay.league import League, Role
from pokerbot.training.tourney_selfplay.pikl_lift import LiftedAgent

# Map the field_sampler archetypes onto the league's clone tiers (design field composition §3).
ARCHETYPE_TIER: dict[str, str] = {
    "calling_station": "fish",
    "lag": "weak_reg",
    "tag": "strong_reg",
    "nit": "fish",          # tight-passive fish: still a faithful clone, λ≈∞
    "pro": "modern_pro",
}


def _commit_mass(agent: LiftedAgent, strength_quantile: float, role: str) -> float:
    """A representative commit-probability for a confrontation at this strength quantile.

    The lifecycle abstracts a hand to a single jam/call decision by strength quantile; the LiftedAgent
    speaks the 11-action space. We build a TWO-action proxy legal mask (fold vs commit) and a τ-logit
    proxy that increases the commit logit with the strength quantile (stronger hand → more commit),
    then run the piKL mixture so the tier's λ shapes the commit mass. This is a faithful-shape proxy
    for the reference env; the Rust sim serves the real per-street policy.
    """
    # 2-action proxy: index 0 = fold, index 1 = commit. τ-logit: commit grows with strength.
    q = np.array([0.0, 0.0])
    tau = np.array([1.0 - strength_quantile, strength_quantile]) * 4.0 - 2.0  # logit-ish
    legal = np.array([True, True])
    # Q proxy: committing a stronger hand is +$-EV; weaker is -$-EV (the lift's pull direction).
    q = np.array([0.0, (strength_quantile - 0.5) * 2.0])
    p = pikl.pikl_mixture(q, tau, legal, agent.config.lambda_set, kappa=agent.config.kappa)
    return float(p[1])


def lifted_agent_decide_fn(
    agent: LiftedAgent, *, jam_base: float = 0.45, call_base: float = 0.60
) -> _life.DecideFn:
    """Wrap a :class:`LiftedAgent` as a lifecycle ``DecideFn`` (jam/call threshold by commit mass).

    Mirrors ``mtt_lifecycle.net_decide``'s commit→threshold mapping: a higher commit mass → a lower
    fold-below threshold (a wider commit range). Centered on the chip-EV base so a 0.5 commit mass
    reproduces the chip-EV baseline.
    """

    def fn(ctx: _life.SpotContext) -> _life.SpotDecision:
        commit = _commit_mass(agent, ctx.strength_quantile, ctx.role)
        base = jam_base if ctx.role == "jam" else call_base
        thr = float(min(0.99, max(0.0, base + (0.5 - commit) * 0.8)))
        return _life.SpotDecision(threshold=thr, source=f"lifted:{agent.tier}")

    return fn


def population_field_draw(
    seed: int, n_villains: int = 5
) -> list[tuple[str, str]]:
    """Population-proportional draw of ``n_villains`` (archetype, tier) pairs (design §3 composition).

    Reuses ``field_sampler.sample_archetypes`` (the IRC archetype distribution) and maps each to a
    league clone tier. The dominant ``calling_station`` slice → the FISH tier (λ≈∞, verbatim
    loose-passive human play).
    """
    archetypes = field_sampler.sample_archetypes(seed, n_villains)
    return [(a, ARCHETYPE_TIER.get(a, "fish")) for a in archetypes]


def sample_field_decide_fns(
    league: League,
    agents_by_tier: dict[str, LiftedAgent],
    *,
    seed: int = 0,
    n_villains: int = 5,
) -> tuple[_life.DecideFn, list[tuple[str, str]]]:
    """Draw a population-proportional villain field and return (representative villain_decide_fn, draw).

    ``agents_by_tier`` maps a tier name to its :class:`LiftedAgent` (built from the league's
    FROZEN_HUMAN members). The lifecycle accepts ONE ``villain_decide_fn``, so we return the MODAL
    tier's agent as the representative field policy + the full draw (so a per-seat backend — the Rust
    sim — can use the whole list). Falls back to the FISH tier when a tier has no agent.
    """
    draw = population_field_draw(seed, n_villains)
    tiers = [t for _, t in draw]
    # modal tier
    modal = max(set(tiers), key=tiers.count) if tiers else "fish"
    agent = agents_by_tier.get(modal) or agents_by_tier.get("fish")
    if agent is None:
        raise ValueError("no LiftedAgent available for the drawn field (need at least a 'fish' tier)")
    return lifted_agent_decide_fn(agent), draw
