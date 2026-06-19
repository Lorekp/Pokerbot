"""Aggregate CoinPoker rake-FLOOR curve + per-stake net-EV recommendations (bb/100).

``pokerbot/economics/rake.py`` is the pure per-hand rake model. This module turns it into the aggregate
artifact the rest of the bot needs: the bb/100 a winning reg must beat at each stake (the "floor"), and,
given a measured gross edge vs a field, the per-stake play/avoid ranking. It lived in
``scripts/cash_rake_floor.py`` (CLI-only); it is promoted here so package code — the live table-selection
net-EV decision (`pokerbot/live/table_select.py`) — can import it without reaching into ``scripts/``. The
CLI re-exports these names, so its behaviour and tests are unchanged.

The net-rake floor is a HUMP: ~0 at micro (CoinRewards returns ~100% of fees), a ~8.5 bb/100 valley
across the low/mid stakes (fat BB cap, promos ended), ~3 at high (tiny cap). So the SAME gross edge nets
very differently by stake — which is exactly why a rake-aware table/stake decision needs this curve, not
a fixed softness cutoff (memory: coinpoker-cash-rake-floor).
"""

from __future__ import annotations

import math

import numpy as np

from pokerbot.economics import rake as rk

# Central 6-max reg operating point for the analytic floor (see scripts/cash_rake_floor.py CLI defaults):
# flopped pots/100 the hero contributes to, mean flopped pot (bb), hero contribution share, pot CV.
DEFAULT_FLOP_PER_100 = 26.0
DEFAULT_MEAN_POT_BB = 20.0
DEFAULT_HERO_SHARE = 0.45
DEFAULT_POT_CV = 1.2
_DEFAULT_SPLASH_SHARE = 1.0 / 6.0
_DEFAULT_SEED = 42
_DEFAULT_SAMPLES = 1 << 16


def _lognormal_pots(mean_bb: float, cv: float, n: int, rng: np.random.Generator) -> np.ndarray:
    sigma2 = math.log(1.0 + cv * cv)
    mu = math.log(mean_bb) - 0.5 * sigma2
    return rng.lognormal(mean=mu, sigma=math.sqrt(sigma2), size=n)


def analytic_rake_curve(flop_per_100: float, mean_pot_bb: float, hero_share: float,
                        cv: float, n_samples: int, seed: int,
                        splash_share: float = _DEFAULT_SPLASH_SHARE, *, tier: int = 5) -> dict:
    """Per-stake (rake_drag, net_floor, net_modeled, splash) bb/100 from a realistic flopped-pot model.

    ``tier`` is the cap tier (2 / 3 / 5 = heads-up / 3-4 / 5+ dealt); the BB cap binds harder short-handed
    is FALSE — fewer dealt => smaller cap, so the floor is materially LOWER 3-4 handed. Full ring => 5+."""
    rng = np.random.default_rng(seed)
    pots = _lognormal_pots(mean_pot_bb, cv, n_samples, rng)
    five_pct = rk.RAKE_PCT * pots
    out: dict[str, tuple[float, float, float, float]] = {}
    for sk in rk.STAKE_ORDER:
        cap = rk.STAKES[sk].cap_bb[tier]
        per_pot_hero = float(np.minimum(five_pct, cap).mean()) * hero_share
        gross = flop_per_100 * per_pot_hero
        splash = 100.0 * rk.SPLASH_FEE_BB * splash_share if rk.splash_eligible(sk) else 0.0
        out[sk] = (
            gross,
            gross * (1.0 - rk.rakeback_fraction(sk, "floor")),
            gross * (1.0 - rk.rakeback_fraction(sk, "modeled")),
            splash,
        )
    return out


def total_floor_modeled(curve: dict, sk: str) -> float:
    """bb/100 a winner must beat at a stake with the MODELED rakeback band: net rake (modeled) + splash."""
    return curve[sk][2] + curve[sk][3]


def total_floor(curve: dict, sk: str, mode: str = "floor") -> float:
    """Full bb/100 floor a winner must beat at a stake = net rake (chosen rakeback mode) + splash. Use
    ``mode='floor'`` (certain 15% Daily Guaranteed) for CONSERVATIVE go/no-go — the modeled CoinRewards
    band is now an optimistic expectation over variance-laden splash/race pools (post-2026-04-01)."""
    net_rake = curve[sk][1] if mode == "floor" else curve[sk][2]
    return net_rake + curve[sk][3]


def stake_recommendations(curve: dict, gross_edge_bb: float, *, mode: str = "floor") -> list[dict]:
    """Given a MEASURED gross edge (bb/100) vs a field, per-stake net + play/avoid, sorted best-net first.

    ``net`` = gross edge minus the full fee floor at that stake; ``ok`` = net-positive. The floor is a
    hump, so this naturally prefers micro (volume) and high (low cap) over the low/mid valley for the SAME
    field. Field softness still dominates; this only ranks stakes once a field's gross edge is known."""
    rows = []
    for sk in rk.STAKE_ORDER:
        floor = total_floor(curve, sk, mode)
        net = gross_edge_bb - floor
        rows.append({"stake": sk, "band": rk.STAKES[sk].tier_band, "floor": floor,
                     "net": net, "ok": net > 0.0})
    rows.sort(key=lambda r: -r["net"])
    return rows


# --- Convenience: a cached default-operating-point curve per dealt-count tier (the live path) ----------
_DEFAULT_CURVES: dict[int, dict] = {}


def default_curve(n_dealt: int = 6) -> dict:
    """The floor curve at the central reg operating point for ``n_dealt`` players (cap tier 2/3/5). Cached
    per tier so the live decision pays the lognormal sampling once, not per hand."""
    tier = rk.cap_tier(int(n_dealt))
    c = _DEFAULT_CURVES.get(tier)
    if c is None:
        c = analytic_rake_curve(DEFAULT_FLOP_PER_100, DEFAULT_MEAN_POT_BB, DEFAULT_HERO_SHARE,
                                DEFAULT_POT_CV, _DEFAULT_SAMPLES, _DEFAULT_SEED, tier=tier)
        _DEFAULT_CURVES[tier] = c
    return c


def floor_bb100(stake_key: str, *, n_dealt: int = 6, mode: str = "floor") -> float:
    """The bb/100 rake floor a winner must beat at ``stake_key`` for ``n_dealt`` players. Conservative
    (15% guaranteed rakeback) by default. Raises KeyError on an unknown stake key."""
    return total_floor(default_curve(n_dealt), stake_key, mode)


def nearest_stake_key(bb_usd: float) -> str:
    """Map a real-money big-blind size (USD) to the closest CoinPoker stake key. Useful when the live
    table's USD stake is known; chip-denominated/play-money blinds should NOT be passed here."""
    best, bestd = rk.STAKE_ORDER[0], float("inf")
    for sk in rk.STAKE_ORDER:
        d = abs(rk.STAKES[sk].bb_usd - float(bb_usd))
        if d < bestd:
            best, bestd = sk, d
    return best
