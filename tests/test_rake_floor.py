"""Aggregate rake-floor curve + per-stake net-EV recommendations (pokerbot/economics/rake_floor.py)."""

from __future__ import annotations

from pokerbot.economics import rake as rk
from pokerbot.economics.rake_floor import (
    default_curve,
    floor_bb100,
    nearest_stake_key,
    stake_recommendations,
    total_floor,
)


def test_floor_is_positive_and_modeled_le_floor():
    curve = default_curve(6)
    for sk in rk.STAKE_ORDER:
        f = total_floor(curve, sk, "floor")
        m = total_floor(curve, sk, "modeled")
        assert f >= 0.0 and m >= 0.0
        assert m <= f + 1e-9          # modeled rakeback returns >= the 15% guaranteed -> lower floor


def test_modeled_floor_is_a_hump_micro_below_valley():
    curve = default_curve(6)
    micro = total_floor(curve, "0.01/0.02", "modeled")   # ~100% rakeback -> ~0
    valley = total_floor(curve, "0.10/0.25", "modeled")  # fat cap, promos ended -> the valley
    high = total_floor(curve, "25/50", "modeled")        # tiny cap
    assert micro < valley and high < valley               # the floor is a hump peaking in the valley


def test_default_curve_is_cached_per_tier():
    assert default_curve(6) is default_curve(6)            # 5+ tier cached
    assert default_curve(3) is not default_curve(6)        # different cap tier -> different curve


def test_floor_bb100_matches_total_floor():
    assert abs(floor_bb100("0.10/0.25", n_dealt=6, mode="floor")
               - total_floor(default_curve(6), "0.10/0.25", "floor")) < 1e-9


def test_short_handed_floor_is_lower():
    # Fewer dealt => smaller BB cap tier => lower floor (3-4 handed cap < 5+ cap).
    assert floor_bb100("0.10/0.25", n_dealt=3) <= floor_bb100("0.10/0.25", n_dealt=6) + 1e-9


def test_stake_recommendations_sorted_and_flagged():
    curve = default_curve(6)
    rows = stake_recommendations(curve, gross_edge_bb=30.0, mode="floor")
    assert rows == sorted(rows, key=lambda r: -r["net"])   # best-net first
    for r in rows:
        assert r["ok"] == (r["net"] > 0.0)
        assert abs(r["net"] - (30.0 - r["floor"])) < 1e-9


def test_negative_edge_clears_nowhere():
    rows = stake_recommendations(default_curve(6), gross_edge_bb=-35.0, mode="floor")
    assert all(not r["ok"] for r in rows)                  # a pro-table edge nets nowhere


def test_nearest_stake_key():
    assert nearest_stake_key(0.25) == "0.10/0.25"          # bb_usd 0.25
    assert nearest_stake_key(2.0) == "1/2"
