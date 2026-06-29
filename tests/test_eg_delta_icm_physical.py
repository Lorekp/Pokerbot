"""ΔICM physicality + format/regime sensitivity pins for the E6 regime battery.

THE BUG these pin (regression guard)
------------------------------------
``regime_battery`` originally scored every cell against the spot-builder's OWN intrinsic payout ladder,
which was built at INCONSISTENT scales (the bubble cell's min-cash tail ~$89 pool, the final-table cell's
full top ~$1037 pool, the satellite cell's ``seat_value=100`` ~$800 pool) and from a single hardcoded
``TournamentFormat.NORMAL``. The consequences, observed in ``artifacts/tournament_eval_7max.json``:

  * ``delta_icm_mean`` for the FT / HU / satellite cells was O(20-34) — NON-PHYSICAL for a per-GAME
    ICM-$ delta (a sane one is O(0.01-1), like the analytic-ICM ``-0.094`` reference on NORMAL x bubble);
  * the value was IDENTICAL across NORMAL / SNG / PKO (the format was builder-hardcoded, so formats whose
    payout ladders genuinely differ collapsed to one number);
  * the satellite value was a single constant across all six regimes (deep-satellite carried the same
    seat-bubble pressure as the satellite bubble — the geometry ignored the regime).

THE FIX these pin
-----------------
When the suite passes its canonical per-cell context (the FORMAT-correct, REGIME-sized, pool-NORMALIZED
payout vector + depth/ante/seats from ``eval_tournament_suite._build_cell_payouts``), the battery
re-expresses every spot at that canonical scale + format (``regime_battery._canonical_payouts`` + the
regime-aware builders). So the per-game ΔICM is (1) PHYSICAL (bounded by the payout/buy-in scale),
(2) FORMAT-sensitive where the ladders differ, and (3) REGIME-sensitive (deep-satellite != satellite
bubble; ft != hu).

Net-free, CPU-only, tiny samples. The pure-scale pins use the cheap always-jam/always-fold anchors (no
ICM solve); the sensitivity pins drive the ``jam`` anchor cell through ``regime_battery_cell`` (a few
fixed-iteration solves per cell, smoke N).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from pokerbot.tournament import regime_battery as rb
from pokerbot.tournament.payouts import TournamentFormat

_REPO = Path(__file__).resolve().parents[1]


def _suite():
    """Load the suite module (for its canonical ``_build_cell_payouts`` + ``REGIMES``) by path — it lives
    under ``scripts/`` (not an importable package)."""
    path = _REPO / "scripts" / "tournament" / "eval_tournament_suite.py"
    spec = importlib.util.spec_from_file_location("_eval_suite_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SUITE = _suite()
_FORMATS = ["NORMAL", "SNG", "PKO", "SATELLITE"]
_REGIMES = ["deep", "mid", "shallow", "bubble", "ft", "hu"]


def _seats_for(regime: str) -> int:
    if regime == "hu":
        return 2
    if regime in ("ft", "bubble"):
        return 6
    return 7


def _cell_args(fmt: str, regime: str, policy: str, spots: int = 8) -> dict:
    """The suite-shaped ``cell_args`` for one (format, regime) cell — the canonical FORMAT-correct,
    REGIME-sized, pool-NORMALIZED payout vector + depth/ante/seats, exactly as the driver builds it."""
    seats = _seats_for(regime)
    _fmt, pay, ante = _SUITE._build_cell_payouts(fmt, seats, regime)
    return {
        "n_seats": seats,
        "depth_bb": _SUITE.REGIMES[regime]["depth_bb"],
        "ante_bb": ante,
        "payouts": [float(p) for p in pay],
        "seed": 0,
        "smoke": True,
        "policy": policy,
        "spots": spots,
    }


def _buy_in(payouts: list[float]) -> float:
    """A representative buy-in scale for a pool: the mean prize per PAID place (``pool / places_paid``).
    For a pool-~100 NORMAL ladder this is ~25-50; for a flat seat ladder it is one seat_value."""
    paid = [p for p in payouts if p > 0.0]
    if not paid:
        return 1.0
    return sum(paid) / len(paid)


# ============================================================================== (1) PHYSICAL SCALE (all cells)
@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("regime", _REGIMES)
def test_canonicalized_per_runout_dollar_delta_is_physical(fmt, regime):
    """Every (format, regime) cell, scored under the canonical ctx, yields a per-game ICM-$ delta bounded
    by the payout scale — NOT the O(20-34) artifact. We use the cheap always-jam vs always-fold anchors
    (no ICM solve) so this is fast AND a tight stress (jamming every hand near the money is the biggest
    legitimate per-game swing). Two bounds:

      * HARD physical ceiling: ``|delta| <= max(payouts)`` (one hand cannot move more $-equity than the
        single largest prize — an absolute ICM truth);
      * O(buy-in) scale: ``|delta| <= 3 * buy_in`` where ``buy_in = pool / places`` (the artifact was
        O(200x) the buy-in for FT/HU/satellite; a physical per-game delta is a small multiple of one
        buy-in)."""
    ca = _cell_args(fmt, regime, policy="jam", spots=8)
    cell = "satellite_bubble" if fmt == "SATELLITE" \
        else rb._CELL_FOR_REGIME.get(regime, "shortstack_pushfold")
    ctx = rb._suite_ctx(fmt, regime, ca)
    assert ctx is not None, "the suite ctx must be built from cell_args.payouts"

    buy_in = _buy_in(ca["payouts"])
    pool_max = max(ca["payouts"])
    for d in range(8):
        spot = rb.sample_root(cell, d, 0, ctx=ctx)
        # the canonical ctx must have set the right FORMAT (not the builder's hardcoded NORMAL)
        if fmt == "SATELLITE":
            assert spot.fmt is TournamentFormat.SATELLITE
        elif fmt == "PKO":
            assert spot.fmt is TournamentFormat.PKO
        else:
            assert spot.fmt is TournamentFormat.NORMAL
        # the payout scale must be the canonical pool (~100 / seat ~10), never the ~1037 / ~800 artifact
        assert sum(p for p in spot.payouts if p > 0) <= 1.5 * sum(ca["payouts"]) + 1e-6

        settle_seed = 4242 + spot.deck_seed
        jam = "call" if spot.role == "facing_jam" else "jam"
        va = "jam" if spot.role == "facing_jam" else "call"
        dol, _chip = rb._score_spot(spot, jam, va, settle_seed, n_runouts=12)
        assert abs(dol) <= pool_max + 1e-6, (
            f"{fmt}/{regime} per-game ICM-$ delta {dol:.3f} exceeds the top prize {pool_max:.3f}")
        assert abs(dol) <= 3.0 * buy_in + 1e-6, (
            f"{fmt}/{regime} per-game ICM-$ delta {dol:.3f} is non-physical vs buy_in {buy_in:.3f}")


def test_artifact_scale_is_gone_final_table():
    """The specific artifact: the FT cell's ΔICM was -23.140. Under the canonical ctx the headline is
    O(0.01-1) (a few % of one buy-in), NOT tens of dollars."""
    ca = _cell_args("NORMAL", "ft", policy="jam", spots=8)
    out = rb.regime_battery_cell("NORMAL", "ft", ca)
    buy_in = _buy_in(ca["payouts"])
    assert abs(out["delta_icm_mean"]) < 5.0, "FT ΔICM must be physical, not the ~23 artifact"
    assert abs(out["delta_icm_mean"]) <= 3.0 * buy_in, "FT ΔICM must be O(buy-in)"


def test_satellite_scale_is_gone():
    """The satellite cell's ΔICM was -33.854 (a seat_value-100 over 8 seats). Under the canonical flat
    seat ladder (seat_value ~10) it is O(0.01-1)."""
    ca = _cell_args("SATELLITE", "bubble", policy="jam", spots=8)
    out = rb.regime_battery_cell("SATELLITE", "bubble", ca)
    assert abs(out["delta_icm_mean"]) < 5.0, "satellite ΔICM must be physical, not the ~34 artifact"


# ============================================================================== (2) FORMAT SENSITIVITY
def test_format_sensitivity_normal_vs_pko_ft():
    """NORMAL and PKO have genuinely different PRIZE ladders (PKO puts half the pool in bounties, so its
    prize vector is flatter / fewer places). On the FT cell — where the canonical ctx applies the
    format-correct ladder — the two formats must NOT collapse to one number (the artifact had them
    identical because the builder hardcoded NORMAL)."""
    normal_pay = _cell_args("NORMAL", "ft", "jam")["payouts"]
    pko_pay = _cell_args("PKO", "ft", "jam")["payouts"]
    assert normal_pay != pko_pay, "the canonical NORMAL and PKO ladders must differ (sanity)"

    out_n = rb.regime_battery_cell("NORMAL", "ft", _cell_args("NORMAL", "ft", "jam"))
    out_p = rb.regime_battery_cell("PKO", "ft", _cell_args("PKO", "ft", "jam"))
    # The two cells run the SAME spots/decks but score under DIFFERENT (format-correct) ladders, so the
    # paired ΔICM must diverge — they are no longer one constant across formats.
    assert out_n["delta_icm_mean"] != pytest.approx(out_p["delta_icm_mean"], abs=1e-9), (
        "NORMAL and PKO FT ΔICM must differ (format-sensitive), not collapse to one number")


def test_format_routing_sets_the_real_format():
    """The canonicalizer must stamp the real format on the spot (NORMAL/SNG->NORMAL, PKO->PKO,
    SATELLITE->SATELLITE) — not the builder's hardcoded NORMAL."""
    assert rb._fmt_from_key("NORMAL") is TournamentFormat.NORMAL
    assert rb._fmt_from_key("SNG") is TournamentFormat.NORMAL
    assert rb._fmt_from_key("PKO") is TournamentFormat.PKO
    assert rb._fmt_from_key("SATELLITE") is TournamentFormat.SATELLITE

    spot = rb.sample_root("final_table_icm", 0, 0, ctx=rb._suite_ctx("PKO", "ft", _cell_args("PKO", "ft", "jam")))
    assert spot.fmt is TournamentFormat.PKO


# ============================================================================== (3) REGIME SENSITIVITY
def test_satellite_deep_differs_from_satellite_bubble():
    """Deep satellite must NOT carry the same seat-bubble pressure as the satellite bubble: the artifact
    reported one constant (-33.854) across all six regimes. The regime-aware geometry places hero FAR from
    the lock boundary when deep (mild seat pressure) and AT the boundary on the bubble — so the locked /
    near-lock fraction (hence the stop-gambling pressure) differs."""
    deep_ctx = rb._suite_ctx("SATELLITE", "deep", _cell_args("SATELLITE", "deep", "jam"))
    bub_ctx = rb._suite_ctx("SATELLITE", "bubble", _cell_args("SATELLITE", "bubble", "jam"))

    def _lock_fraction_mean(ctx) -> float:
        from pokerbot.tournament import satellite as sat
        fracs = []
        for d in range(40):
            spot = rb.sample_root("satellite_bubble", d, 7, ctx=ctx)
            seats_paid = sum(1 for p in spot.payouts if p > 0.0)
            total = sum(s for s in spot.stacks if s > 0.0)
            fracs.append(spot.stacks[spot.hero] / total if total > 0 else 0.0)
        return float(np.mean(fracs))

    deep_frac = _lock_fraction_mean(deep_ctx)
    bubble_frac = _lock_fraction_mean(bub_ctx)
    # On the bubble hero sits AT/above the lock boundary (a larger chip fraction); deep he is well below.
    assert bubble_frac > deep_frac + 0.02, (
        f"deep satellite ({deep_frac:.3f}) must carry LESS lock pressure than the bubble ({bubble_frac:.3f})")


def test_ft_differs_from_hu():
    """The artifact had ft and hu identical (both mapped to ``final_table_icm`` with no regime input). The
    canonical ctx makes hu 2-handed with a top-2 ladder while ft is short-handed — so the cells' payout
    vectors (hence ΔICM) genuinely differ."""
    ft_spot = rb.sample_root("final_table_icm", 0, 0, ctx=rb._suite_ctx("NORMAL", "ft", _cell_args("NORMAL", "ft", "jam")))
    hu_spot = rb.sample_root("final_table_icm", 0, 0, ctx=rb._suite_ctx("NORMAL", "hu", _cell_args("NORMAL", "hu", "jam")))
    # HU collapses to a 2-seat table + top-2 ladder; ft keeps the short-handed final table.
    assert hu_spot.seats == 2, "the HU cell must be 2-handed"
    assert ft_spot.seats > 2, "the FT cell must be short-handed (not HU)"
    # the paid ladders differ (top-2 vs the FT short-handed ladder)
    paid_ft = tuple(round(p, 4) for p in ft_spot.payouts if p > 0)
    paid_hu = tuple(round(p, 4) for p in hu_spot.payouts if p > 0)
    assert paid_ft != paid_hu, "ft and hu must carry different (regime-correct) payout ladders"


# ============================================================================== (4) BACKWARD-COMPAT (no ctx)
def test_no_ctx_is_legacy_scale_unchanged():
    """With NO ctx the builder output is byte-unchanged (the standalone/legacy contract pinned by
    ``tests/test_regime_battery.py``): the FT cell keeps its raw ``buy_in=10`` global ladder."""
    from pokerbot.tournament import payouts as _payouts
    spot = rb.sample_root("final_table_icm", 0, 0)            # no ctx
    gv = _payouts.payout_vector(TournamentFormat.NORMAL, entrants=180, buy_in=10.0)
    assert np.allclose(spot.payouts, gv[: len(spot.payouts)]), "no-ctx must keep the legacy ladder"
    assert spot.fmt is TournamentFormat.NORMAL
