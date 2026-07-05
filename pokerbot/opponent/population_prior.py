"""Population prior over per-context action-bin distributions — the Dirichlet PSEUDOCOUNTS that give
new/unknown players a cold-start read AND shrink small-n individual data toward the population
(the anti-overfit regularizer).

PROVENANCE WARNING (exploitation overhaul design 2026-07-05, artifacts/exploitation_overhaul_design.md):
the numbers below (VPIP ~48%, PFR ~10%, AF ~0.65) come from the IRC corpus — 1995-2001,
predominantly FIXED-LIMIT, full-ring, pre-Moneymaker. They are NOT a measured fact about modern
online NLHE: the IRC nolimit folder (25,787 hands) has zero calibration signal (skill-estimator
reliability 0.00), and the modern micro/low-stakes 6-max field is BIMODAL (tight-aggressive reg
majority-by-seats + a loose-passive fish minority; field-mean VPIP ~22-32 per the coaching/HUD
literature cited in the design §8). The "field is loose-passive => the edge is thin value-betting"
doctrine holds for the FISH ARCHETYPE, not the field mean. The modern replacement is
``pokerbot.opponent.archetype_prior`` (a K=5 archetype mixture), gated behind
POKERBOT_ARCHETYPE_PRIOR until its pre-registered validation gate (G-P2) passes on real captured
NLHE data — until then THIS module remains the deployed default (byte-identical behavior).

Loads ``data/player_profiles/irc_population_prior.json`` if present; otherwise falls back to an
in-code generator so the engine never hard-depends on the file (mirrors mixer._load_calibrated_threshold).
``scripts/build_population_prior.py`` writes the file; ``scripts/analyze_irc_population.py`` is the
empirical regenerator if the IRC tarball is restored.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from pokerbot.opponent.context import NUM_BINS, all_contexts

# Env-overridable (import-time) so the Phase-3 prior A/B can run the FITTED candidate file against
# the shipped default on the same seeds without touching the default path mid-battery.
PRIOR_PATH = os.environ.get("POKERBOT_PRIOR_PATH", "data/player_profiles/irc_population_prior.json")
# Total pseudocounts/context ⇒ confidence n/(n+8): ~50% at 8 hands, ~75% at 24. Env-overridable
# (import-time, like POKERBOT_BETTING_ABSTRACTION) for the G4 adaptation tuning sweep ONLY —
# production keeps the default unless an 18-replicate gate certifies a better value.
PRIOR_STRENGTH = float(os.environ.get("POKERBOT_PRIOR_STRENGTH", "8.0"))
BIN_NAMES = ("fold", "passive", "agg_small", "agg_large")


def _base_dist(street: str, facing: str) -> list[float]:
    """Loose-passive base outcome-bin distribution [fold, passive, agg_small, agg_large] by
    (street-phase, facing). Sums to ~1. Texture-independent (individual data differentiates by
    texture; the population a-priori does not). Encodes: enters ~48%, raises ~10%, under-folds to
    bets, calls/checks a lot."""
    preflop = street == "preflop"
    if facing == "unopened":
        # Preflop: VPIP~48% ⇒ fold 52%; of entries PFR~10% ⇒ mostly limp/call, few raises.
        # Postflop unopened (checked-to): can't fold ⇒ mostly check, low cbet.
        return [0.52, 0.38, 0.07, 0.03] if preflop else [0.02, 0.68, 0.20, 0.10]
    if facing == "facing_bet":
        # The station signature: under-fold, over-call, rarely raise.
        return [0.45, 0.45, 0.06, 0.04] if preflop else [0.40, 0.50, 0.06, 0.04]
    if facing == "facing_raise":
        return [0.58, 0.34, 0.05, 0.03] if preflop else [0.52, 0.40, 0.05, 0.03]
    # facing_3bet+
    return [0.65, 0.28, 0.04, 0.03]


def generate_prior(strength: float = PRIOR_STRENGTH) -> dict[str, list[float]]:
    """Per-context Dirichlet pseudocounts (= strength × base distribution) for every context."""
    out: dict[str, list[float]] = {}
    for ck in all_contexts():
        street, _texture, facing = ck.split("|")
        out[ck] = [round(strength * x, 4) for x in _base_dist(street, facing)]
    return out


@lru_cache(maxsize=2)
def load_prior(path: str = PRIOR_PATH) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Load the prior as a hashable tuple (lru-cacheable). Prefer the JSON file; fall back to the
    in-code generator. Use ``prior_dict()`` for a normal dict."""
    p = Path(path)
    if p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            pri = raw.get("prior") or raw.get("context_prior")
            if pri:
                return tuple((k, tuple(float(x) for x in v)) for k, v in pri.items())
        except (OSError, ValueError, TypeError):
            pass
    return tuple((k, tuple(v)) for k, v in generate_prior().items())


def prior_dict(path: str = PRIOR_PATH) -> dict[str, list[float]]:
    return {k: list(v) for k, v in load_prior(path)}


def context_prior(context_key: str, path: str = PRIOR_PATH) -> list[float]:
    """Pseudocount vector for one context (generator fallback for an unseen key)."""
    for k, v in load_prior(path):
        if k == context_key:
            return list(v)
    parts = context_key.split("|")
    if len(parts) == 3:
        return [round(PRIOR_STRENGTH * x, 4) for x in _base_dist(parts[0], parts[2])]
    return [PRIOR_STRENGTH / NUM_BINS] * NUM_BINS
