"""Modern archetype-mixture population prior — the P2 replacement for the IRC limit-era point
prior (exploitation overhaul design §A1/§A2, artifacts/exploitation_overhaul_design.md).

WHY. The deployed prior (population_prior._base_dist: VPIP~48 / PFR~10 / AF~0.65, IRC 1995-2001)
is a LIMIT-era, full-ring artifact; the IRC corpus cannot calibrate modern NLHE (nolimit folder:
25,787 hands, skill-estimator reliability 0.00). The modern micro/low-stakes 6-max field is
BIMODAL — a tight-aggressive reg majority-by-seats plus a loose-passive fish minority that carries
the profit — with field-mean VPIP ~22-32, not 48 (design §8 citations: PokerCopilot / Hand2Note /
BlackRain79 10M+-hand DB / GipsyTeam 2026; coaching-consensus basis, no public per-context corpus
exists). A single point prior cannot represent a bimodal field; a K=5 archetype mixture can, and
it also pools evidence ACROSS contexts: a 30-hand VPIP/PFR read (fast) reshapes all 22 context
priors at once, exactly the 50-500-hand live regime where per-context counts are starved.

WHAT. Per archetype z: a per-context-class base distribution over the K=4 outcome bins (the same
analytic HUD->bins translation style as population_prior._base_dist, applied to each archetype's
modern HUD anchors — G-P2b later replaces the analytic translation with simulated bins). Per
player: a posterior w(z | vpip, pfr, hands) from Beta-binomial scalar-stat likelihoods (scalar
stats converge ~10x faster than per-context counts), then

    prior_for(stats, ck) = PRIOR_STRENGTH * sum_z w(z) * base_z(ck)

With zero hands w(z) = the field mixture weights => a coherent field-average prior (mixture-mean
preflop VPIP ~31 — inside the modern 28-32 band by construction, vs IRC's 48).

GATING. Everything here is behind ``POKERBOT_ARCHETYPE_PRIOR`` (default OFF — pre-registered gate
G-P2 must PASS before the default flips): OFF => ``prior_for`` returns exactly
``population_prior.context_prior(ck)`` (byte-identical serve/ingest). The gate script is
``scripts/exploit_prior_gate.py`` (leave-one-observation-out predictive log-loss on the
live-captured, NLHE-filtered profiles — validation on real data, never fitting to it).
"""

from __future__ import annotations

import math
import os

from pokerbot.opponent.context import NUM_BINS, all_contexts
from pokerbot.opponent.population_prior import PRIOR_STRENGTH, context_prior

# ---------------------------------------------------------------------------------------------
# Archetype anchors (modern micro/low 6-max, coaching-consensus — design §8 sources) and their
# analytic per-context-class outcome-bin distributions [fold, passive, agg_small, agg_large].
# Context classes mirror population_prior._base_dist: (preflop?, facing).
# ---------------------------------------------------------------------------------------------
# Pre-registered field mixture weights (design §A1 — LOW-confidence coaching consensus, subject
# to G-P2b refinement; the G-P2 gate validates the whole prior, weights included).
MIX_WEIGHTS: dict[str, float] = {
    "fish": 0.30,     # loose-passive recreational (VPIP 35-70 / PFR 3-12) — the profit pool
    "nit": 0.25,      # tight-passive multi-tabler (VPIP <15 / PFR <12)
    "tag": 0.25,      # solid reg (VPIP 19-28 / PFR 17-24) — the canonical winning profile
    "lag": 0.10,      # loose-aggressive skilled (VPIP 23-32 / PFR 20-28)
    "maniac": 0.10,   # hyper-loose aggressor (VPIP >45 / PFR >25)
}

# Scalar-stat anchors per archetype: (vpip_rate, pfr_rate) — the Beta-binomial likelihood centers.
ANCHORS: dict[str, tuple[float, float]] = {
    "fish": (0.48, 0.08),
    "nit": (0.13, 0.10),
    "tag": (0.23, 0.19),
    "lag": (0.30, 0.25),
    "maniac": (0.55, 0.30),
}

# Beta-binomial concentration for the archetype posterior: rates are cluster CENTERS with real
# within-archetype spread (a fish is 35-70 VPIP), so the likelihood must be fat. kappa=20 puts
# ~±10 VPIP points of sd around the center at moderate n — classification converges in ~30-50
# hands without being whipsawed by 10-hand noise.
KAPPA = 20.0

_CLASSES = (
    ("preflop", "unopened"), ("preflop", "facing_bet"),
    ("preflop", "facing_raise"), ("preflop", "facing_3bet"),
    ("postflop", "unopened"), ("postflop", "facing_bet"),
)

# Analytic HUD->bin translation per archetype (documented derivation: preflop-unopened fold =
# 1-VPIP; entry split by PFR share; facing distributions shaped by the archetype's fold/call/raise
# doctrine; postflop by AF/WTSD character). G-P2b replaces these with SIMULATED bins recorded
# through the real ingest pipeline.
_BASE: dict[str, dict[tuple[str, str], list[float]]] = {
    "fish": {
        ("preflop", "unopened"):     [0.55, 0.38, 0.05, 0.02],
        ("preflop", "facing_bet"):   [0.35, 0.55, 0.07, 0.03],   # under-folds, over-calls
        ("preflop", "facing_raise"): [0.50, 0.42, 0.05, 0.03],
        ("preflop", "facing_3bet"):  [0.60, 0.32, 0.05, 0.03],
        ("postflop", "unopened"):    [0.02, 0.75, 0.15, 0.08],   # checks a lot, low cbet
        ("postflop", "facing_bet"):  [0.35, 0.55, 0.06, 0.04],   # sticky station
    },
    "nit": {
        ("preflop", "unopened"):     [0.88, 0.04, 0.06, 0.02],
        ("preflop", "facing_bet"):   [0.70, 0.22, 0.05, 0.03],
        ("preflop", "facing_raise"): [0.80, 0.14, 0.04, 0.02],
        ("preflop", "facing_3bet"):  [0.85, 0.10, 0.03, 0.02],
        ("postflop", "unopened"):    [0.02, 0.70, 0.20, 0.08],
        ("postflop", "facing_bet"):  [0.55, 0.35, 0.06, 0.04],   # over-folds postflop
    },
    "tag": {
        ("preflop", "unopened"):     [0.77, 0.04, 0.14, 0.05],   # raise-or-fold, few limps
        ("preflop", "facing_bet"):   [0.55, 0.30, 0.10, 0.05],
        ("preflop", "facing_raise"): [0.62, 0.26, 0.08, 0.04],
        ("preflop", "facing_3bet"):  [0.70, 0.20, 0.06, 0.04],
        ("postflop", "unopened"):    [0.02, 0.55, 0.28, 0.15],   # healthy cbet
        ("postflop", "facing_bet"):  [0.45, 0.40, 0.10, 0.05],   # ~MDF defense
    },
    "lag": {
        ("preflop", "unopened"):     [0.70, 0.03, 0.18, 0.09],
        ("preflop", "facing_bet"):   [0.45, 0.32, 0.14, 0.09],
        ("preflop", "facing_raise"): [0.52, 0.28, 0.12, 0.08],
        ("preflop", "facing_3bet"):  [0.60, 0.24, 0.09, 0.07],
        ("postflop", "unopened"):    [0.02, 0.45, 0.33, 0.20],
        ("postflop", "facing_bet"):  [0.38, 0.38, 0.14, 0.10],   # plays back
    },
    "maniac": {
        ("preflop", "unopened"):     [0.45, 0.20, 0.20, 0.15],
        ("preflop", "facing_bet"):   [0.30, 0.35, 0.18, 0.17],
        ("preflop", "facing_raise"): [0.35, 0.33, 0.17, 0.15],
        ("preflop", "facing_3bet"):  [0.42, 0.30, 0.15, 0.13],
        ("postflop", "unopened"):    [0.02, 0.40, 0.30, 0.28],
        ("postflop", "facing_bet"):  [0.25, 0.40, 0.18, 0.17],   # raises/never folds
    },
}


def enabled() -> bool:
    """Call-time env gate (default OFF until the pre-registered G-P2 gate passes)."""
    return os.environ.get("POKERBOT_ARCHETYPE_PRIOR", "0").lower() not in (
        "0", "", "false", "no", "off")


def _class_of(context_key: str) -> tuple[str, str]:
    parts = context_key.split("|")
    street = "preflop" if parts[0] == "preflop" else "postflop"
    facing = parts[-1]
    if street == "postflop" and facing not in ("unopened", "facing_bet"):
        facing = "facing_bet"
    if street == "preflop" and facing not in ("unopened", "facing_bet", "facing_raise",
                                              "facing_3bet"):
        facing = "facing_bet"
    return street, facing


def archetype_base(arch: str, context_key: str) -> list[float]:
    """Archetype ``arch``'s base outcome-bin distribution at ``context_key``."""
    return list(_BASE[arch][_class_of(context_key)])


def _log_betabin(k: int, n: int, rate: float, kappa: float = KAPPA) -> float:
    """log BetaBinomial(k | n, a=rate*kappa, b=(1-rate)*kappa) up to a z-independent constant.
    The B(k+a, n-k+b) - B(a, b) term only (the n-choose-k factor cancels across archetypes)."""
    a = max(1e-6, rate * kappa)
    b = max(1e-6, (1.0 - rate) * kappa)
    return (math.lgamma(k + a) + math.lgamma(n - k + b) - math.lgamma(n + a + b)
            - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)))


def archetype_weights(stats) -> dict[str, float]:
    """Posterior w(z | player) from the SCALAR evidence (vpip/pfr counts over hands) — the
    fast-converging cross-context signal. No data (or a malformed profile) => the field mixture
    weights exactly (cold-start = field-average prior). Robust: any failure returns the mixture."""
    try:
        hands = int(getattr(stats, "hands", 0) or 0)
        if stats is None or hands <= 0:
            return dict(MIX_WEIGHTS)
        vpip = min(hands, int(getattr(stats, "vpip", 0) or 0))
        pfr = min(hands, int(getattr(stats, "pfr", 0) or 0))
        logw = {}
        for z, w0 in MIX_WEIGHTS.items():
            rv, rp = ANCHORS[z]
            logw[z] = (math.log(w0) + _log_betabin(vpip, hands, rv)
                       + _log_betabin(pfr, hands, rp))
        m = max(logw.values())
        exp = {z: math.exp(v - m) for z, v in logw.items()}
        tot = sum(exp.values())
        return {z: v / tot for z, v in exp.items()}
    except Exception:  # noqa: BLE001 — a malformed profile must never break the prior
        return dict(MIX_WEIGHTS)


def mixture_prior(weights: dict[str, float], context_key: str,
                  strength: float = PRIOR_STRENGTH) -> list[float]:
    """Pseudocount vector = strength × the weight-mixed archetype base distributions."""
    out = [0.0] * NUM_BINS
    for z, w in weights.items():
        base = archetype_base(z, context_key)
        for i in range(NUM_BINS):
            out[i] += w * base[i]
    return [round(strength * x, 6) for x in out]


def prior_for(stats, context_key: str, *, strength: float = PRIOR_STRENGTH,
              force: bool | None = None) -> list[float]:
    """THE P2 prior seam. Gated: OFF (default) => exactly ``population_prior.context_prior``
    (byte-identical ingest/serve). ON => the player-conditional archetype-mixture pseudocounts.
    ``force`` overrides the env gate (the G-P2 gate script evaluates both arms explicitly)."""
    on = enabled() if force is None else bool(force)
    if not on:
        return context_prior(context_key)
    return mixture_prior(archetype_weights(stats), context_key, strength)


def field_prior_dict(strength: float = PRIOR_STRENGTH) -> dict[str, list[float]]:
    """The cold-start (mixture-weight) prior for every locked context — the modern analog of
    ``population_prior.generate_prior`` (diagnostics / the future default-flip artifact)."""
    return {ck: mixture_prior(dict(MIX_WEIGHTS), ck, strength) for ck in all_contexts()}
