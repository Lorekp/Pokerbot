"""Population-proportional heterogeneous FIELD sampler — the realistic CoinPoker ring for M4 / M2.

WHY this module exists (the realistic-field gap)
------------------------------------------------
The deployed cash edge is value-EXTRACTION vs a SOFT field — and "soft" is not five copies of one bot.
The IRC 39.99M-hand population (``data/player_profiles/irc_population_summary.json``) is dominated by
loose-passive payoff fish: ~47.4% ``calling_station``, with a thin reg/aggressor tail (``lag`` 3.2%,
``nit`` 0.7%, ``tag`` 0.4%) and a large ``unknown`` residual (48.2%) that is itself loose-passive on the
pooled HUD (population VPIP ~0.497 / PFR ~0.140 / AF ~0.88 — a calling field). To MEASURE the bot vs the
field *distribution* (not vs a single archetype) the eval ring must be DRAWN from that archetype mix, and
the dominant ``calling_station`` slice must map to a SIZE-RESPONSIVE bot (``responsive_station``) so the
value-SIZING overlay can actually express (a size-blind station makes the overlay's uplift ~0 — see
``pokerbot.eval.responsive_station``).

What it provides
----------------
``sample_field(seed, n_villains=5)`` -> ``list[(bot_name, params|None)]``
    A SEED-DETERMINISTIC population-proportional draw of ``n_villains`` villain seats from the IRC
    archetype distribution. Each archetype maps to a faithful eval bot:
      * ``calling_station`` (~47%) -> ``responsive_station`` (the size-elastic loose-passive payoff fish;
        the value-SIZING target). Also where the loose-passive ``unknown`` residual is folded, because on
        the pooled population HUD the unknowns are a calling field, not a distinct skilled cohort.
      * ``lag``  -> ``responsive_reg`` seeded from the LAG HUD (a size-aware loose-aggressive reg).
      * ``tag``  -> ``responsive_reg`` (the disciplined size-aware reg; defends ~MDF — the do-no-harm seat).
      * ``nit``  -> ``hs_nit``        (the tight-passive folder; the IRC-fitted HS nit).
      * ``pro``  -> ``pro_balanced``  (the rare near-equilibrium reg; the counter-exploit-risk seat). The
        population JSON has no explicit ``pro`` slice, so a small fixed PRO_FRACTION is carved off the reg
        tail (the realistic "1 solid reg at the table" the pro-safety thesis must survive).
    The probabilities are normalised over the KNOWN archetypes (``unknown`` folded into the station mass);
    the same ``seed`` always yields the same ordered field, so it composes cleanly with the CRN seat-
    rotation in ``het_cv_absolute`` / ``run_field_absolute_rotated`` (the field is fixed per measurement).

``co_rotated_field(seed=0, n_villains=5)`` -> ``list[(bot_name, params|None)]``
    A FIXED, representative heterogeneous ring for ``het_cv_absolute`` to CO-ROTATE. Unlike a fresh random
    draw it pins a *canonical* realistic composition (mostly responsive stations + one reg + a nit) so the
    headline het-CV number is reproducible and dominated by the value-extraction target, while still being
    a genuine mix (the co-rotation requires a FIXED ring; a per-deck re-draw would break the duplicate
    cancellation ``het_cv_absolute`` relies on). ``seed`` only perturbs the *order* of the canonical
    multiset (so a caller can average over seat permutations); the multiset itself is fixed.

This is a PURE BUILDER — it returns field specs consumed by ``run_field_absolute(per_seat_profiles=True)``
and ``het_cv_absolute``; it plays no hands, holds no rake, and edits no harness. Firewall-safe (new file,
no engine / AoF / NUM_PLAYERS touch). Nothing here is stochastic beyond the supplied ``seed``.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

# The population summary the live profiler + calibrated opponents are fitted against (single source of
# truth for the archetype mix — do NOT hardcode the percentages, read them so a re-parse updates the field).
_POP_SUMMARY_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "player_profiles" / "irc_population_summary.json"
)

# Archetype -> eval (bot_name, params) mapping. The dominant calling-station slice routes to the
# SIZE-ELASTIC responsive_station so the value-SIZING overlay can express (a size-blind station makes its
# uplift ~0). The reg/aggressor tail routes to size-aware bots; the nit to the tight HS folder; the pro to
# the near-equilibrium reg (the counter-exploit-risk seat the pro-safety thesis must survive).
_ARCHETYPE_BOT: dict[str, tuple[str, dict | None]] = {
    # loose-passive payoff fish (the value-extraction target) — size-elastic, over-defends past MDF.
    "calling_station": ("responsive_station", None),
    # the large loose-passive UNKNOWN residual is a calling field on the pooled HUD -> fold into stations.
    "unknown": ("responsive_station", None),
    # loose-aggressive reg: size-aware, but defends nearer MDF (a reg, not a station) -> responsive_reg.
    "lag": ("responsive_reg", None),
    # tight-aggressive disciplined reg: defends ~MDF (the do-no-harm size-aware control seat).
    "tag": ("responsive_reg", None),
    # tight-passive folder: the IRC-fitted HS nit (fold-heavy; not a value-sizing target).
    "nit": ("hs_nit", None),
    # near-equilibrium solid reg (the rare pro): the counter-exploit-risk seat (pro_balanced).
    "pro": ("pro_balanced", None),
}

# A small fixed PRO slice carved off the reg tail. The IRC summary has no explicit "pro" archetype (the
# field is overwhelmingly recreational), but the realistic CoinPoker table seats roughly one solid reg, so
# the pro-safety measurement must include a near-equilibrium seat. 2% keeps the field soft while guaranteeing
# the pro seat appears often enough to matter — and stays BELOW the lag+tag reg-tail mass (~3.6% of the
# qualified field) so carving it off the tail does not zero out the realistic loose/tight-aggressive regs.
PRO_FRACTION = 0.02

# The canonical fixed multiset for the CO-ROTATED het ring (n_villains=5): mostly size-elastic stations
# (the value-extraction target the headline number should reflect), one disciplined reg (the do-no-harm
# control) and one nit (a fold seat). This is the realistic-soft composition, NOT a random draw, so the
# headline het-CV number is reproducible and dominated by the value target while staying a genuine mix.
_CANONICAL_RING_5 = [
    ("responsive_station", None),
    ("responsive_station", None),
    ("responsive_station", None),
    ("responsive_reg", None),
    ("hs_nit", None),
]


@lru_cache(maxsize=1)
def _archetype_distribution() -> dict[str, float]:
    """Normalised archetype -> probability over the KNOWN archetypes, with the ``unknown`` slice folded
    into ``calling_station`` (the pooled-HUD population is a calling field) and a fixed ``pro`` slice
    carved off the reg tail. Cached: the summary file does not change within a process."""
    data = json.loads(_POP_SUMMARY_PATH.read_text(encoding="utf-8"))
    dist = data["archetype_distribution"]
    # Raw counts per archetype (use count, not the rounded pct, for an exact normalisation).
    counts = {k: float(v.get("count", 0.0)) for k, v in dist.items()}
    # Fold the loose-passive UNKNOWN residual into the station mass (pooled HUD = a calling field).
    station = counts.get("calling_station", 0.0) + counts.get("unknown", 0.0)
    lag = counts.get("lag", 0.0)
    nit = counts.get("nit", 0.0)
    tag = counts.get("tag", 0.0)
    total = station + lag + nit + tag
    if total <= 0:
        # Degenerate summary -> an all-station field (still a valid, soft default; never raise here).
        return {"calling_station": 1.0}
    probs = {
        "calling_station": station / total,
        "lag": lag / total,
        "nit": nit / total,
        "tag": tag / total,
    }
    # Carve a fixed PRO slice off the reg/aggressor tail (proportional to lag+tag, so it comes from the
    # "skilled" mass — never off the fish). If the tail is empty, take it off the station mass instead.
    pro = PRO_FRACTION
    tail = probs["lag"] + probs["tag"]
    if tail > 1e-9:
        scale = max(0.0, (tail - pro)) / tail
        probs["lag"] *= scale
        probs["tag"] *= scale
    else:
        probs["calling_station"] = max(0.0, probs["calling_station"] - pro)
    probs["pro"] = pro
    # Renormalise (numerical safety) so the draw weights sum to 1.
    s = sum(probs.values())
    return {k: v / s for k, v in probs.items() if v > 0.0}


def archetype_probabilities() -> dict[str, float]:
    """Public view of the normalised draw distribution (archetype -> probability). Pooled ``unknown`` is
    already folded into ``calling_station``; a fixed ``pro`` slice is carved off the reg tail. Sums to 1."""
    return dict(_archetype_distribution())


def _map_archetype(archetype: str) -> tuple[str, dict | None]:
    """Map a drawn archetype to a ``(bot_name, params)`` field spec (raises on an unknown archetype so a
    distribution change can never silently route to a missing bot)."""
    if archetype not in _ARCHETYPE_BOT:
        raise ValueError(f"no eval-bot mapping for archetype {archetype!r} "
                         f"(known: {sorted(_ARCHETYPE_BOT)})")
    name, params = _ARCHETYPE_BOT[archetype]
    return name, (dict(params) if params else None)


def sample_archetypes(seed: int, n_villains: int = 5) -> list[str]:
    """SEED-DETERMINISTIC population-proportional draw of ``n_villains`` archetype labels.

    Pure function of ``(seed, n_villains)`` — the SAME seed yields the SAME ordered labels, so the field
    composes with the CRN seat-rotation in the het estimators (the field is fixed per measurement)."""
    if n_villains < 1:
        raise ValueError("n_villains must be >= 1")
    probs = _archetype_distribution()
    labels = list(probs.keys())
    weights = [probs[k] for k in labels]
    rng = random.Random(int(seed))
    return rng.choices(labels, weights=weights, k=int(n_villains))


def sample_field(seed: int, n_villains: int = 5) -> list[tuple[str, dict | None]]:
    """SEED-DETERMINISTIC population-proportional heterogeneous villain field.

    Draws ``n_villains`` archetypes from the IRC archetype distribution (``unknown`` folded into the
    station mass; a fixed ``pro`` slice carved off the reg tail) and maps each to a faithful eval-bot
    spec. Returns ``[(bot_name, params|None), ...]`` consumable by ``run_field_absolute`` /
    ``het_cv_absolute`` for the ``n_villains`` non-hero seats.

    The dominant ``calling_station`` slice maps to the SIZE-ELASTIC ``responsive_station`` so the deployed
    value-SIZING overlay can express; the reg/aggressor tail maps to size-aware regs; the rare ``pro`` to
    the near-equilibrium ``pro_balanced`` (the counter-exploit-risk seat). Deterministic in ``seed``."""
    return [_map_archetype(a) for a in sample_archetypes(seed, n_villains)]


def co_rotated_field(seed: int = 0, n_villains: int = 5) -> list[tuple[str, dict | None]]:
    """A FIXED representative heterogeneous ring for ``het_cv_absolute`` to CO-ROTATE.

    Unlike ``sample_field`` (a fresh population draw whose composition varies with the seed), this returns
    a CANONICAL realistic-soft multiset — mostly size-elastic stations (the value-extraction target the
    headline number should reflect), one disciplined reg and a nit — so the headline het-CV number is
    reproducible and dominated by the value target while remaining a genuine mix. ``het_cv_absolute``
    requires a FIXED ring (a per-deck re-draw would break the duplicate cancellation), so the MULTISET is
    fixed; ``seed`` only permutes the ORDER (the seat the ring presents each archetype at), letting a
    caller average over seat permutations without changing the composition.

    For ``n_villains == 5`` the canonical 5-ring is used; for other sizes the ring is built by drawing
    population-proportionally with a FIXED seed (0) so it is still deterministic and station-dominated."""
    if n_villains == 5:
        ring = list(_CANONICAL_RING_5)
    else:
        # Other table sizes: a fixed-seed population draw (deterministic, station-dominated) as the ring.
        ring = sample_field(seed=0, n_villains=n_villains)
    if seed:
        # Permute ONLY the order (not the multiset) so callers can average over seat permutations.
        rng = random.Random(int(seed))
        rng.shuffle(ring)
    return ring
