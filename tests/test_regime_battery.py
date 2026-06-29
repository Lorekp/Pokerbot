"""Certification of the E6 CRN twin-policy ΔICM regime battery (``pokerbot.tournament.regime_battery``).

Load-bearing invariants (a broken one silently fabricates or erases an ICM edge):

  * **CRN identity** — when ``hero_policy == chipev_twin`` (same callable) EVERY paired ΔICM is exactly 0:
    same deck, same settle_seed, same action => identical terminal => zero residual. So a same-policy
    battery has mean == se == 0 and ``uplift_zero_frac == 1.0`` (the CRN-health diagnostic at its ceiling).
  * **deck-seed convention** — a spot's cards are a PURE function of (seed, spot index) via the project
    ``deck_seed = 1000 + seed*1_000_000 + d`` (``exploitation_battery.py:1101``): re-sampling the same
    (cell, d, seed) reproduces the identical hole cards (the twin pairing depends on this).
  * **structure-correct payouts** — every cell's payout vector comes from ``payouts.payout_vector`` /
    ``satellite.satellite_payouts`` (NEVER a literal): a NORMAL cell's vector is descending + pool-positive,
    a SATELLITE cell's is the flat ``[seat_value]*seats`` step ladder.
  * **conservation** — a NORMAL/SNG terminal's per-seat $-deltas conserve the prize pool (sum ~= 0); the
    chip leaf is zero-sum.
  * **flip => divergence** — ``icm_delta_saved`` accumulates ONLY over the flips (spots where the ICM-aware
    action reorders the chip-EV action); a no-flip control cell saves ~0.
  * **C1 satellite stop-gambling certificate** — the ICM policy folds EVERY mathematically-locked seat
    (``satellite.seat_locked``); the chip-EV twin does not (it would risk a locked seat).

Net-free, CPU-only, tiny samples — runs in any env with numpy + the AoF HU equity grid (no jax/torch/engine).
"""

from __future__ import annotations

import numpy as np
import pytest

from pokerbot.tournament import payouts as _payouts
from pokerbot.tournament import satellite as _satellite
from pokerbot.tournament import regime_battery as rb

NORMAL = _payouts.TournamentFormat.NORMAL
SATELLITE = _payouts.TournamentFormat.SATELLITE


# =============================================================================== deck-seed / CRN plumbing
def test_deck_seed_convention_matches_project():
    """The spot's deck seed is EXACTLY the project convention 1000 + seed*1_000_000 + d."""
    for seed in (0, 3, 11):
        for d in (0, 1, 57):
            spot = rb.sample_root("shortstack_pushfold", d, seed)
            assert spot.deck_seed == 1000 + seed * 1_000_000 + d


def test_cards_are_a_pure_function_of_seed_and_index():
    """Re-sampling the same (cell, d, seed) reproduces identical hole cards (CRN determinism)."""
    a = rb.sample_root("bubble_icm", 5, seed=2)
    b = rb.sample_root("bubble_icm", 5, seed=2)
    assert a.hero_cards == b.hero_cards
    assert a.villain_cards == b.villain_cards
    assert a.hand_label == b.hand_label
    # A different spot index gives (almost surely) different cards.
    c = rb.sample_root("bubble_icm", 6, seed=2)
    assert (a.hero_cards, a.villain_cards) != (c.hero_cards, c.villain_cards)


def test_hero_and_villain_cards_are_disjoint():
    """Hero and villain are dealt four DISTINCT cards off one shuffled deck."""
    for d in range(20):
        spot = rb.sample_root("payjump", d, seed=1)
        cards = set(spot.hero_cards) | set(spot.villain_cards)
        assert len(cards) == 4, "dealt cards must be distinct"


# =============================================================================== structure-correct payouts
def test_normal_cell_payouts_are_structure_correct_descending():
    """A NORMAL cell's payout vector is descending, pool-positive, and drawn from the structure model.

    The bubble cell models a single live TABLE (ICM is read off the table, not the 180-handed field —
    full-field Malmuth-Harville is intractable), so its ladder is the min-cash TAIL of the structure-
    correct global vector + one unpaid bubble rung. Assert it is a genuine subset of the structure model,
    not a literal."""
    spot = rb.sample_root("bubble_icm", 0, seed=0)
    assert spot.fmt is NORMAL
    pv = spot.payouts
    assert len(pv) >= 2
    assert all(pv[i] >= pv[i + 1] - 1e-9 for i in range(len(pv) - 1)), "non-increasing ladder"
    assert sum(pv) > 0.0
    assert pv[-1] == 0.0, "the bubble table has one unpaid finisher (the bubble bust)"
    # Every paid rung is a rung of the structure-correct global vector (subset, NEVER a literal).
    gv = _payouts.payout_vector(NORMAL, entrants=180, buy_in=10.0)
    paid = [round(x, 6) for x in pv if x > 0.0]
    gv_set = {round(x, 6) for x in gv}
    assert all(x in gv_set for x in paid), "paid rungs must come from the structure model"


def test_other_normal_cells_match_full_structure_vector():
    """The non-bubble NORMAL cells (payjump/shortstack/deep) truncate the FULL structure vector's top."""
    for cell, ent in (("payjump", 180), ("shortstack_pushfold", 90), ("deep_nearmoney", 180)):
        spot = rb.sample_root(cell, 0, seed=0)
        gv = _payouts.payout_vector(NORMAL, entrants=ent, buy_in=10.0)
        # the cell ladder is the leading `len(pv)` rungs of the global vector (table-local truncation).
        assert np.allclose(spot.payouts, gv[: len(spot.payouts)]), f"{cell} ladder must be the structure top"


def test_satellite_cell_payouts_are_flat_step_ladder():
    """A SATELLITE cell's payout vector is the flat [seat_value]*seats step ladder."""
    spot = rb.sample_root("satellite_bubble", 0, seed=0)
    assert spot.fmt is SATELLITE
    pv = spot.payouts
    assert len(set(round(x, 6) for x in pv)) == 1, "satellite seats are identical (flat step)"
    assert pv == _satellite.satellite_payouts(len(pv), pv[0])


def test_shortstack_cell_stays_in_certified_pushfold_zone():
    """The shortstack push/fold cell clamps the effective depth into the certified <=20bb zone."""
    for d in range(30):
        spot = rb.sample_root("shortstack_pushfold", d, seed=4)
        eff = min(spot.stacks[spot.hero], spot.stacks[spot.villain])
        assert eff <= 20.0 + 1e-6


# =============================================================================== CRN identity (the spine)
def test_same_policy_gives_exactly_zero_edge_and_full_crn_health():
    """hero_policy == chipev_twin => every paired ΔICM is exactly 0 (same deck/seed/action)."""
    twin = rb.always_fold_policy
    res = rb.run_battery("bubble_icm", twin, twin, n_spots=24, seed=0, n_runouts=8)
    assert res.n == 24
    assert res.mean == pytest.approx(0.0, abs=1e-12)
    assert res.se == pytest.approx(0.0, abs=1e-12)
    assert res.chipev_mean == pytest.approx(0.0, abs=1e-12)
    assert res.flip_rate == 0.0
    assert res.icm_delta_saved == pytest.approx(0.0, abs=1e-12)
    assert res.uplift_zero_frac == pytest.approx(1.0), "no flips => every pair is an exact-0 (CRN health)"
    assert res.verdict == "INCONCLUSIVE (CI spans 0)"


def test_same_policy_zero_for_always_jam_too():
    """The CRN identity holds for the aggressive anchor as well (contested all-ins cancel under CRN)."""
    jam = rb.always_jam_policy
    res = rb.run_battery("payjump", jam, jam, n_spots=20, seed=1, n_runouts=8)
    assert res.mean == pytest.approx(0.0, abs=1e-12)
    assert res.chipev_mean == pytest.approx(0.0, abs=1e-12)
    assert res.uplift_zero_frac == pytest.approx(1.0)


# =============================================================================== conservation
def test_normal_terminal_conserves_prize_pool():
    """Every NORMAL settle's per-seat $-deltas conserve the fixed prize pool (sum ~= 0)."""
    from pokerbot.tournament.eval_score import score_chip_to_dollar

    spot = rb.sample_root("bubble_icm", 3, seed=7)
    chip = rb._settle_spot(spot, "jam", "call", settle_seed=123, n_runouts=16)
    assert abs(sum(chip)) < 1e-6, "chip leaf is zero-sum"
    dollar = score_chip_to_dollar(spot.stacks, chip, spot.payouts, spot.fmt)
    assert abs(float(np.sum(dollar))) < 1e-6, "NORMAL $-delta conserves the prize pool"


def test_uncontested_jam_scoops_dead_chips_zero_sum():
    """A first-in jam that folds out the villain scoops the dead chips, conserving chips (sum 0)."""
    spot = rb.sample_root("shortstack_pushfold", 2, seed=3)
    chip = rb._settle_spot(spot, "jam", "fold", settle_seed=5, n_runouts=8)
    assert abs(sum(chip)) < 1e-9
    # the jamming hero ends up with a non-negative chip delta (it won the dead blinds/antes).
    assert chip[spot.hero] >= -1e-9


# =============================================================================== flip => divergence
def test_flip_drives_the_edge_and_zero_frac_complements():
    """An ICM-aware policy that diverges from the chip-EV twin produces flips; icm_delta_saved sums them.

    On the bubble cell the ICM policy folds marginal chip-EV jams (bubble factor > 1), so the edge is
    driven by exactly the flip spots. uplift_zero_frac + flip_rate must be consistent (non-flips are the
    exact-0 pairs)."""
    icm = rb.icm_nash_policy
    twin = rb.chip_ev_nash_twin
    res = rb.run_battery("bubble_icm", icm, twin, n_spots=40, seed=2, n_runouts=10)
    assert 0.0 <= res.flip_rate <= 1.0
    # non-flip spots are exactly-0 pairs => zero_frac == 1 - flip_rate (every non-flip pair cancels).
    assert res.uplift_zero_frac == pytest.approx(1.0 - res.flip_rate, abs=1e-9)
    # icm_delta_saved is the SUM of the per-spot edge over the flips only; since every NON-flip pair is
    # exactly 0 (identical terminal under CRN), the total edge lives entirely in the flips => saved == n*mean.
    assert res.icm_delta_saved == pytest.approx(res.mean * res.n, abs=1e-9)


def test_deep_control_cell_has_low_flip_rate():
    """The deep near-money control: chip-EV ~= $-EV, so the ICM policy rarely flips the chip-EV action."""
    res = rb.run_battery("deep_nearmoney", rb.icm_nash_policy, rb.chip_ev_nash_twin,
                         n_spots=30, seed=5, n_runouts=8)
    assert res.flip_rate <= 0.5, "deep near-money should mostly agree with chip-EV (control cell)"


# =============================================================================== chip-EV side-channel
def test_chip_side_channel_is_present_and_paired():
    """The chip-EV side-channel is the SAME pairing scored in chips; its series aligns 1:1 with the $ series."""
    res = rb.run_battery("shortstack_pushfold", rb.icm_nash_policy, rb.chip_ev_nash_twin,
                         n_spots=20, seed=8, n_runouts=8)
    assert len(res.per_spot_chip) == len(res.per_spot_dollar) == res.n
    # a non-flip spot is 0 in BOTH columns (identical terminal).
    for dv, cv in zip(res.per_spot_dollar, res.per_spot_chip):
        if abs(dv) < 1e-12:
            assert abs(cv) < 1e-9, "a non-flip pair must be 0 in the chip column too"


# =============================================================================== C1 certificate
def test_satellite_stop_gambling_certificate_passes():
    """C1: the ICM policy folds EVERY mathematically-locked seat (seat_locked => fold)."""
    cert = rb.satellite_stop_gambling_certificate(n_spots=120, seed=0)
    assert cert["pass"] is True
    if cert["locked_spots"] > 0:
        assert cert["icm_fold_frac_when_locked"] == pytest.approx(1.0)


def test_locked_seat_must_fold_directly():
    """Directly: on any spot where the seat is LOCKED, icm_nash_policy must fold (never risk the seat)."""
    checked = 0
    for d in range(200):
        spot = rb.sample_root("satellite_bubble", d, seed=1)
        seats_paid = len(spot.payouts)
        if _satellite.seat_locked(spot.stacks, seats_paid, spot.hero):
            assert rb.icm_nash_policy(spot, cache={}) == "fold"
            checked += 1
    # we should have hit at least a few locked spots in 200 draws (borderline geometry straddles the lock).
    assert checked >= 1, "no locked satellite spots sampled — widen the borderline band"


# =============================================================================== verdict reducer wiring
def test_verdict_matches_ci_sign():
    """The reported verdict is the standard CI->headline reducer over the battery's own CI."""
    from pokerbot.tournament.eval_score import verdict
    res = rb.run_battery("bubble_icm", rb.icm_nash_policy, rb.chip_ev_nash_twin,
                         n_spots=24, seed=4, n_runouts=8)
    assert res.verdict == verdict(res.ci95[0], res.ci95[1])


# =============================================================================== cv_absolute feed
def test_per_spot_series_feedable_to_cv_absolute():
    """A homogeneous cell's per-spot series is the right shape to feed cv_absolute.decompose_absolute."""
    from pokerbot.eval.cv_absolute import decompose_absolute

    res = rb.run_battery("shortstack_pushfold", rb.icm_nash_policy, rb.chip_ev_nash_twin,
                         n_spots=12, seed=9, n_runouts=6)
    s = np.asarray(res.per_spot_dollar)
    # a degenerate but valid CV call: pair the series with itself + a disjoint half as the cheap arm.
    cv = decompose_absolute(s, np.zeros_like(s), s)
    assert cv.decks_paired == res.n
    assert np.isfinite(cv.mean)
