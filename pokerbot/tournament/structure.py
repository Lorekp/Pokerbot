"""CoinPoker NLHE MTT blind/ante ladder + structural rules (Workstream C/E).

Authoritative source: ``tournament_example.txt`` -- the "BLIND + ANTE STRUCTURE" table (levels
1-32 and 55-70, transcribed from live v1.6.0 client screenshots) plus the "DERIVED FACTS" section.

Key facts encoded here (all from the source):
  * Starting chips 20,000 = 200BB at Level 1 (BB=100). Deep-start, slow structure.
  * 10-minute levels throughout (every level).
  * UNIVERSAL ante rule (same for every CoinPoker MTT): per-player ante = round(BB/8) = 0.125 BB,
    rounded HALF-UP. Antes are DEAD money (port to pot_dead + committed_total). The captured display
    antes match this to display-rounding tolerance (one outlier: L18 BB=2500 shows 315 vs model 313).
  * Re-buys / late-reg / re-entries close at the END of Level 14 (LATE_REG_LAST_LEVEL = 14).
  * Final-table blind rollback ("Yes") + 2-minute final-table break.

All blinds/antes are stored in CHIPS (display K/M expanded). Levels 33-54 are NOT captured in the
source; we interpolate them with the same 10-min, ~geometric escalation between L32 and L55 (the
source notes "same 10-min, ~1.25-1.5x escalation").

Rounding convention (project-wide): round-half-up via ``int(x + 0.5)``, NEVER banker's round().
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BlindLevel",
    "CAPTURED_LEVELS",
    "INTERPOLATED_LEVELS",
    "LATE_REG_LAST_LEVEL",
    "FINAL_TABLE_BREAK_MINUTES",
    "STARTING_CHIPS",
    "LEVEL_MINUTES",
    "ante_for_bb",
    "blind_ladder",
    "level_at",
    "effective_bb_at_level",
    "rebuy_allowed",
    "late_reg_open",
    "final_table_rollback",
    "sample_spot",
]


# ---------------------------------------------------------------------------
# Constants from the source
# ---------------------------------------------------------------------------

STARTING_CHIPS: int = 20_000  # 200BB at L1 (BB=100)
LEVEL_MINUTES: int = 10  # 10-minute levels throughout
LATE_REG_LAST_LEVEL: int = 14  # re-buys / late-reg / re-entries close at END of L14
FINAL_TABLE_BREAK_MINUTES: int = 2  # "Final Table Break: Yes (2 Mins)"
MAX_LEVEL: int = 70  # last captured level


def _round_half_up(x: float) -> int:
    """Round-half-up to int (project convention). NEVER banker's round()."""
    return int(x + 0.5)


@dataclass(frozen=True)
class BlindLevel:
    """A single blind level: small/big blind + per-player ante + duration, all in CHIPS."""

    level: int
    sb: int
    bb: int
    ante: int
    minutes: int


# Captured EXACTLY from tournament_example.txt (display K/M expanded to chips).
# value = (sb, bb, ante, minutes).  minutes = 10 everywhere.
CAPTURED_LEVELS: dict[int, tuple[int, int, int, int]] = {
    # ---- Levels 1-32 ----
    1: (50, 100, 13, 10),
    2: (60, 120, 15, 10),
    3: (75, 150, 19, 10),
    4: (100, 200, 25, 10),
    5: (120, 240, 30, 10),
    6: (140, 280, 35, 10),
    7: (160, 320, 40, 10),
    8: (200, 400, 50, 10),
    9: (250, 500, 63, 10),
    10: (300, 600, 75, 10),
    11: (350, 700, 88, 10),
    12: (400, 800, 100, 10),
    13: (500, 1_000, 125, 10),
    14: (600, 1_200, 150, 10),
    15: (700, 1_400, 175, 10),
    16: (800, 1_600, 200, 10),
    17: (1_000, 2_000, 250, 10),
    18: (1_250, 2_500, 315, 10),
    19: (1_500, 3_000, 375, 10),
    20: (1_750, 3_500, 440, 10),
    21: (2_000, 4_000, 500, 10),
    22: (2_500, 5_000, 625, 10),
    23: (3_000, 6_000, 750, 10),
    24: (3_500, 7_000, 875, 10),
    25: (4_000, 8_000, 1_000, 10),
    26: (5_000, 10_000, 1_250, 10),
    27: (6_000, 12_000, 1_500, 10),
    28: (7_000, 14_000, 1_750, 10),
    29: (8_000, 16_000, 2_000, 10),
    30: (10_000, 20_000, 2_500, 10),
    31: (12_500, 25_000, 3_120, 10),  # display 3.12K
    32: (15_000, 30_000, 3_750, 10),  # display 3.75K
    # ---- Levels 55-70 ----
    55: (875_000, 1_750_000, 218_750, 10),  # 875K / 1.75M, 218.75K
    56: (1_000_000, 2_000_000, 250_000, 10),
    57: (1_250_000, 2_500_000, 312_500, 10),
    58: (1_500_000, 3_000_000, 375_000, 10),
    59: (1_750_000, 3_500_000, 437_500, 10),
    60: (2_000_000, 4_000_000, 500_000, 10),
    61: (2_500_000, 5_000_000, 625_000, 10),
    62: (3_000_000, 6_000_000, 750_000, 10),
    63: (3_500_000, 7_000_000, 875_000, 10),
    64: (4_000_000, 8_000_000, 1_000_000, 10),
    65: (5_000_000, 10_000_000, 1_250_000, 10),
    66: (6_000_000, 12_000_000, 1_500_000, 10),
    67: (7_000_000, 14_000_000, 1_750_000, 10),
    68: (8_000_000, 16_000_000, 2_000_000, 10),
    69: (9_000_000, 18_000_000, 2_250_000, 10),
    70: (10_000_000, 20_000_000, 2_500_000, 10),
}

# Filled in by blind_ladder() the first time it interpolates (levels 33-54).
INTERPOLATED_LEVELS: set[int] = set()


# ---------------------------------------------------------------------------
# Ante rule
# ---------------------------------------------------------------------------

def ante_for_bb(bb: int) -> int:
    """UNIVERSAL CoinPoker per-player ante = round_half_up(BB / 8) = 0.125 BB.

    Same for every CoinPoker MTT (not tournament-specific). Round-half-up via int(x + 0.5).
    """
    return _round_half_up(bb / 8.0)


# ---------------------------------------------------------------------------
# Ladder construction
# ---------------------------------------------------------------------------

def _interpolate_33_54() -> dict[int, tuple[int, int, int, int]]:
    """Geometric escalation of the BIG BLIND between L32 (30K) and L55 (1.75M).

    L32 bb = 30_000, L55 bb = 1_750_000; that's 23 level-steps. We place levels 33..54 on a
    geometric curve so each level multiplies the previous bb by a constant ratio, snap each bb to a
    "nice" round chip value, derive sb = bb // 2 and ante = ante_for_bb(bb). 10-min levels.
    """
    bb32 = CAPTURED_LEVELS[32][1]  # 30_000
    bb55 = CAPTURED_LEVELS[55][1]  # 1_750_000
    n_steps = 55 - 32  # 23 multiplicative steps from L32 to L55
    ratio = (bb55 / bb32) ** (1.0 / n_steps)

    out: dict[int, tuple[int, int, int, int]] = {}
    for level in range(33, 55):
        raw_bb = bb32 * (ratio ** (level - 32))
        bb = _round_to_nice(raw_bb)
        sb = bb // 2
        ante = ante_for_bb(bb)
        out[level] = (sb, bb, ante, LEVEL_MINUTES)
    return out


def _round_to_nice(x: float) -> int:
    """Snap a chip value to a tournament-friendly round number (leading 1/1.25/1.5/2/2.5/.../ rounded
    to a sensible power-of-ten unit) while preserving strict monotonicity via the caller's curve."""
    if x <= 0:
        return 1
    exp = math.floor(math.log10(x))
    unit = 10 ** max(exp - 1, 0)  # round to 2 significant figures
    return int(_round_half_up(x / unit) * unit)


def blind_ladder(max_level: int = 70) -> list[BlindLevel]:
    """Full ladder 1..max_level. CAPTURED where present; levels 33-54 INTERPOLATED.

    Populates the module-level ``INTERPOLATED_LEVELS`` set with the interpolated level numbers that
    fall within [1, max_level].
    """
    if max_level < 1:
        return []

    interp = _interpolate_33_54()
    INTERPOLATED_LEVELS.clear()

    ladder: list[BlindLevel] = []
    for level in range(1, max_level + 1):
        if level in CAPTURED_LEVELS:
            sb, bb, ante, minutes = CAPTURED_LEVELS[level]
        elif level in interp:
            sb, bb, ante, minutes = interp[level]
            INTERPOLATED_LEVELS.add(level)
        else:
            # Beyond the captured/interpolated range (level > 70): extend geometrically from the last
            # known level using the same ~doubling-every-~5-levels feel (ratio between L69 and L70).
            prev = ladder[-1]
            bb = _round_to_nice(prev.bb * 1.25)
            sb = bb // 2
            ante = ante_for_bb(bb)
            minutes = LEVEL_MINUTES
        ladder.append(BlindLevel(level=level, sb=sb, bb=bb, ante=ante, minutes=minutes))
    return ladder


# ---------------------------------------------------------------------------
# Level / depth helpers
# ---------------------------------------------------------------------------

def level_at(elapsed_minutes: float, ladder: list[BlindLevel] | None = None) -> BlindLevel:
    """The blind level in effect after ``elapsed_minutes`` of play.

    Level index = elapsed // 10 + 1 (10-min levels), clamped to [1, len(ladder)].
    """
    if ladder is None:
        ladder = blind_ladder()
    idx = int(elapsed_minutes // LEVEL_MINUTES) + 1
    idx = max(1, min(idx, len(ladder)))
    return ladder[idx - 1]


def effective_bb_at_level(level: int, start_chips: int = STARTING_CHIPS) -> float:
    """Effective stack in BIG BLINDS at ``level`` for a player holding ``start_chips``."""
    ladder = blind_ladder(max(level, 1))
    lv = ladder[max(1, min(level, len(ladder))) - 1]
    return start_chips / lv.bb


# ---------------------------------------------------------------------------
# Structural rules
# ---------------------------------------------------------------------------

def rebuy_allowed(level: int) -> bool:
    """Re-buys are unlimited until the END of Level 14."""
    return level <= LATE_REG_LAST_LEVEL


def late_reg_open(level: int) -> bool:
    """Late registration / re-entries close at the END of Level 14."""
    return level <= LATE_REG_LAST_LEVEL


def final_table_rollback(level: int, rollback_levels: int = 1) -> BlindLevel:
    """Blind level after the final-table rollback: ladder level max(1, level - rollback_levels).

    "Final Table Blind Rollback: Yes" -- blinds step back one (or more) levels when the FT forms.
    """
    target = max(1, level - rollback_levels)
    ladder = blind_ladder(max(target, level, 1))
    return ladder[target - 1]


# ---------------------------------------------------------------------------
# HU-final spot sampler (Workstream E training distribution)
# ---------------------------------------------------------------------------

def sample_spot(rng: "np.random.Generator", *, regime: str = "late") -> tuple[float, float, int]:
    """Draw a (depth_bb, ante_bb, level) training spot, deterministic given a numpy Generator.

    Returns the effective depth in BIG BLINDS and the per-player ante in BIG BLINDS (ante/bb ~= 0.125
    by the universal ante rule), plus the sampled level. The HU tournament-final regime occurs only at
    the very last, very high levels (SHALLOW), so the default ``regime="late"`` weights the level draw
    toward later levels (where effective depth in BB is small).

    The depth is the effective stack of a ~average remaining stack at that level: we model the field's
    average stack as (total chips) / (players remaining) but for the HU-final regime the relevant
    quantity is just the effective BB depth implied by the blind level for a representative late-stage
    stack. We sample a stack around the structure-implied depth band and report depth in BB.
    """
    if regime not in ("late", "mid", "uniform"):
        raise ValueError(f"unknown regime {regime!r}; expected 'late'/'mid'/'uniform'")

    ladder = blind_ladder()
    n = len(ladder)

    if regime == "uniform":
        level = int(rng.integers(1, n + 1))
    else:
        # Weight toward later levels. For 'late' use a steep weight (HU final = very high blinds);
        # 'mid' is gentler. Weight_i grows with level index.
        idx = np.arange(1, n + 1, dtype=np.float64)
        power = 3.0 if regime == "late" else 1.0
        w = idx ** power
        w /= w.sum()
        level = int(rng.choice(idx, p=w))

    lv = ladder[level - 1]

    # Representative effective depth (in BB) at this level. Late-stage stacks cluster in the
    # single-digit-to-low-tens of BB; we draw a log-uniform depth in a structure-anchored band so the
    # sampler is SHALLOW for late levels and deeper early. Anchor the band on the structure: the
    # average remaining stack falls as blinds rise; cap to a deep-early ceiling.
    deep_bb = STARTING_CHIPS / lv.bb  # 200BB at L1 -> shrinks with level
    # SHALLOW band: between ~3BB and the structure-implied deep_bb (capped at a realistic late ceiling).
    lo = 3.0
    hi = max(lo + 1e-6, min(deep_bb, 60.0))
    u = rng.uniform(math.log(lo), math.log(hi))
    depth_bb = float(math.exp(u))

    # Ante in BB by the universal rule (ante chips / bb chips). Equals ~0.125 (display-rounded).
    ante_bb = lv.ante / lv.bb

    return depth_bb, ante_bb, level
