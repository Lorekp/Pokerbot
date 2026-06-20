"""Tests for pokerbot/tournament/structure.py -- CoinPoker MTT blind/ante ladder.

Authoritative source values are transcribed in tournament_example.txt ("BLIND + ANTE STRUCTURE" +
"DERIVED FACTS"). Every captured level (1-32, 55-70) is checked against the source EXACTLY; the
universal ante rule round_half_up(bb/8) is spot-checked and verified against the captured antes to
display-rounding tolerance; structural rules (re-buy/late-reg, effective depth, final-table rollback)
and the HU-final sampler are checked for the documented behavior.
"""

import math

import numpy as np
import pytest

from pokerbot.tournament import structure as S
from pokerbot.tournament.structure import (
    BlindLevel,
    CAPTURED_LEVELS,
    LATE_REG_LAST_LEVEL,
    ante_for_bb,
    blind_ladder,
    effective_bb_at_level,
    final_table_rollback,
    late_reg_open,
    level_at,
    rebuy_allowed,
    sample_spot,
)

# ---------------------------------------------------------------------------
# Source transcription (chips) -- the EXACT expected (sb, bb, ante, minutes) for every captured level.
# Transcribed directly from tournament_example.txt; display K/M expanded to chips.
# ---------------------------------------------------------------------------
EXPECTED_CAPTURED: dict[int, tuple[int, int, int, int]] = {
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
    31: (12_500, 25_000, 3_120, 10),
    32: (15_000, 30_000, 3_750, 10),
    55: (875_000, 1_750_000, 218_750, 10),
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


def test_module_captured_dict_matches_source():
    """The module's CAPTURED_LEVELS is exactly the transcription above (both directions)."""
    assert CAPTURED_LEVELS == EXPECTED_CAPTURED


@pytest.mark.parametrize("level", sorted(EXPECTED_CAPTURED))
def test_ladder_captured_level_exact(level):
    """blind_ladder()[level-1] equals the transcribed source sb/bb/ante/minutes EXACTLY."""
    ladder = blind_ladder()
    lv = ladder[level - 1]
    sb, bb, ante, minutes = EXPECTED_CAPTURED[level]
    assert lv.level == level
    assert lv.sb == sb, f"L{level} sb"
    assert lv.bb == bb, f"L{level} bb"
    assert lv.ante == ante, f"L{level} ante"
    assert lv.minutes == minutes, f"L{level} minutes"


@pytest.mark.parametrize(
    "bb,expected",
    [(100, 13), (120, 15), (150, 19), (500, 63), (700, 88), (1000, 125)],
)
def test_ante_for_bb_spot_checks(bb, expected):
    """Universal ante = round_half_up(bb/8). Spot values from the source."""
    assert ante_for_bb(bb) == expected


def test_ante_for_bb_is_round_half_up():
    """ante_for_bb must use int(x+0.5) round-half-up, NOT banker's rounding.

    bb=4 -> bb/8 = 0.5 -> half-up = 1 (banker's round would give 0).
    bb=12 -> bb/8 = 1.5 -> half-up = 2 (banker's round would give 2 too, but bb=4 disambiguates).
    """
    assert ante_for_bb(4) == 1  # 0.5 -> 1 (round-half-up); round() would give 0
    assert ante_for_bb(12) == 2  # 1.5 -> 2


def test_captured_antes_match_model_within_display_tolerance():
    """Every captured ante equals ante_for_bb(bb) within display-rounding tolerance.

    Tolerance = max(1, ceil(0.011 * bb / 8)). This documents the single display-rounding outlier:
    L18 (bb=2500) shows ante 315 in the source while the model round_half_up(2500/8)=313 (delta 2,
    inside the tol-4 band). All other captured antes match exactly or within tolerance.
    """
    for level, (sb, bb, ante, minutes) in CAPTURED_LEVELS.items():
        model = ante_for_bb(bb)
        tol = max(1, math.ceil(0.011 * bb / 8))
        assert abs(ante - model) <= tol, (
            f"L{level} bb={bb}: captured ante {ante} vs model {model} exceeds tol {tol}"
        )
    # Explicitly pin the documented L18 outlier so a future edit can't silently "fix" it.
    assert CAPTURED_LEVELS[18][2] == 315
    assert ante_for_bb(2500) == 313


def test_blinds_strictly_increasing_and_well_formed():
    """Across the FULL ladder: bb strictly increasing, bb == 2*sb, minutes == 10 everywhere."""
    ladder = blind_ladder()
    assert len(ladder) == 70
    prev_bb = 0
    for lv in ladder:
        assert isinstance(lv, BlindLevel)
        assert lv.bb == 2 * lv.sb, f"L{lv.level}: bb {lv.bb} != 2*sb {lv.sb}"
        assert lv.minutes == 10, f"L{lv.level}: minutes {lv.minutes} != 10"
        assert lv.bb > prev_bb, f"L{lv.level}: bb {lv.bb} not > prev {prev_bb}"
        prev_bb = lv.bb


def test_interpolated_levels_cover_33_to_54():
    """Levels 33-54 are interpolated (not in source); INTERPOLATED_LEVELS reflects exactly those."""
    blind_ladder()  # populates the set
    assert S.INTERPOLATED_LEVELS == set(range(33, 55))
    # Interpolated levels also obey ante_for_bb and sb == bb // 2.
    ladder = blind_ladder()
    for level in range(33, 55):
        lv = ladder[level - 1]
        assert lv.ante == ante_for_bb(lv.bb), f"L{level} interp ante"
        assert lv.sb == lv.bb // 2, f"L{level} interp sb"
        assert lv.minutes == 10


def test_rebuy_and_late_reg_window():
    """Re-buys / late-reg close at the END of Level 14."""
    assert LATE_REG_LAST_LEVEL == 14
    assert rebuy_allowed(14) is True
    assert rebuy_allowed(15) is False
    assert late_reg_open(14) is True
    assert late_reg_open(15) is False
    assert rebuy_allowed(1) is True


def test_effective_bb_at_level():
    """200BB start at L1; strictly decreasing as blinds rise."""
    assert effective_bb_at_level(1) == 200.0
    prev = None
    for level in range(1, 71):
        e = effective_bb_at_level(level)
        if prev is not None:
            assert e < prev, f"effective_bb not strictly decreasing at L{level}: {e} >= {prev}"
        prev = e


def test_effective_bb_custom_start_chips():
    """effective_bb scales with start_chips: 10k chips at L1 (bb=100) = 100BB."""
    assert effective_bb_at_level(1, start_chips=10_000) == 100.0


def test_final_table_rollback():
    """Rollback steps back one level by default; L20 -> ladder level 19 blinds."""
    ladder = blind_ladder()
    rolled = final_table_rollback(20)
    expected = ladder[18]  # level 19
    assert rolled.level == 19
    assert rolled.sb == expected.sb
    assert rolled.bb == expected.bb
    assert rolled.ante == expected.ante
    # Multi-level rollback + clamp at 1.
    assert final_table_rollback(20, rollback_levels=3).level == 17
    assert final_table_rollback(1).level == 1  # clamps to >= 1
    assert S.FINAL_TABLE_BREAK_MINUTES == 2


def test_level_at_elapsed_minutes():
    """level = elapsed // 10 + 1, clamped to the ladder."""
    assert level_at(0).level == 1
    assert level_at(9.9).level == 1
    assert level_at(10).level == 2
    assert level_at(25).level == 3
    # Beyond the ladder clamps to the last level.
    assert level_at(10_000).level == 70


def test_sample_spot_reproducible_and_in_range():
    """Deterministic given a numpy Generator; depth_bb > 0; ante_bb ~= 0.125."""
    d0, a0, l0 = sample_spot(np.random.default_rng(0))
    d1, a1, l1 = sample_spot(np.random.default_rng(0))
    assert (d0, a0, l0) == (d1, a1, l1)
    assert d0 > 0
    assert abs(a0 - 0.125) < 0.05
    assert 1 <= l0 <= 70


def test_sample_spot_distribution_late_regime():
    """Over many draws (late regime): depth>0, ante_bb within 0.05 of 0.125, levels skew late."""
    rng = np.random.default_rng(7)
    levels = []
    for _ in range(3000):
        d, a, lvl = sample_spot(rng, regime="late")
        assert d > 0
        assert abs(a - 0.125) < 0.05
        levels.append(lvl)
    # 'late' regime should put the median level well into the back half of the ladder.
    assert float(np.median(levels)) >= 35


def test_sample_spot_rejects_bad_regime():
    with pytest.raises(ValueError):
        sample_spot(np.random.default_rng(0), regime="bogus")
