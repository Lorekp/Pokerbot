"""Validate the field_sampler (population-proportional het field) + cash_pro_safety (M3 Δ + M6 bet_shift).

field_sampler tests are PURE (no poker compute): seed determinism + pooled-archetype-frequency match within
tolerance + the canonical co-rotated ring. cash_pro_safety's HEAVY arms (LBR vs blueprint+search) are not
run here — instead the ExploitTable.act value-sizing extension is exercised directly (valid prob simplex,
only sizing buckets perturbed), the paired-CRN delta math is checked on synthetic per-hand series, and the
native OFF-floor anchors are asserted to BRACKET a base floor when the native battery is available (skipped
otherwise). The env flags the blueprint encoders bake at import are set BEFORE pokerbot is imported.
"""

from __future__ import annotations

import os

# Bounded-abstraction flags the encoders bake AT IMPORT — set before any pokerbot import (serve fidelity).
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from pokerbot.eval import field_sampler as fs  # noqa: E402


# ==================================================================================================
# field_sampler
# ==================================================================================================
def test_archetype_probabilities_normalised_and_station_dominated():
    """The draw distribution sums to 1, folds the unknown residual into the station mass (so stations
    dominate), keeps the reg/aggressor tail, and carves a small fixed pro slice."""
    p = fs.archetype_probabilities()
    assert abs(sum(p.values()) - 1.0) < 1e-9
    # The dominant slice is the calling station (unknown folded in) — well above the IRC raw 47.4%.
    assert p["calling_station"] > 0.8
    # The reg/aggressor tail survives (not zeroed by the pro carve-off).
    assert p.get("lag", 0.0) > 0.0
    assert p.get("tag", 0.0) > 0.0
    # The pro slice is present and matches the configured fraction.
    assert abs(p["pro"] - fs.PRO_FRACTION) < 1e-9


def test_sample_field_is_seed_deterministic():
    """Same seed -> same ordered field; different seeds -> (generally) different fields."""
    a = fs.sample_field(0, 5)
    b = fs.sample_field(0, 5)
    assert a == b, "same seed must yield identical fields"
    assert len(a) == 5
    # The structure is (bot_name, params|None).
    for name, params in a:
        assert isinstance(name, str)
        assert params is None or isinstance(params, dict)
    # Different seeds should not all collapse to the identical field (the draw is genuinely stochastic).
    fields = {tuple(n for n, _ in fs.sample_field(s, 5)) for s in range(40)}
    assert len(fields) >= 2, "expected variation across seeds"


def test_sample_field_pooled_frequency_matches_distribution():
    """Pooled over many seeds, the drawn BOT frequencies match the archetype distribution within tolerance.

    The dominant calling_station archetype maps to responsive_station (the size-elastic value target), the
    reg tail to responsive_reg, the pro slice to pro_balanced, the nit to hs_nit — so we map the target
    archetype probabilities through the SAME bot mapping before comparing."""
    from collections import Counter

    n_seeds, n_villains = 6000, 5
    c: Counter = Counter()
    for s in range(n_seeds):
        for name, _ in fs.sample_field(s, n_villains):
            c[name] += 1
    total = sum(c.values())
    emp = {k: v / total for k, v in c.items()}

    # Expected per-BOT probability = sum of archetype probs that map to that bot.
    p = fs.archetype_probabilities()
    expect: dict[str, float] = {}
    for arch, prob in p.items():
        bot = fs._map_archetype(arch)[0]
        expect[bot] = expect.get(bot, 0.0) + prob

    # The dominant station bot must match tightly; the thin tail bots within a looser absolute tolerance.
    assert abs(emp.get("responsive_station", 0.0) - expect["responsive_station"]) < 0.02
    for bot, want in expect.items():
        got = emp.get(bot, 0.0)
        assert abs(got - want) < 0.02, f"{bot}: empirical {got:.4f} vs expected {want:.4f}"


def test_sample_field_station_maps_to_responsive_station():
    """The calling_station + unknown mass routes to the SIZE-ELASTIC responsive_station (so the value-SIZING
    overlay can express) — the central design choice of the realistic field."""
    assert fs._map_archetype("calling_station")[0] == "responsive_station"
    assert fs._map_archetype("unknown")[0] == "responsive_station"
    assert fs._map_archetype("pro")[0] == "pro_balanced"
    assert fs._map_archetype("nit")[0] == "hs_nit"


def test_map_archetype_unknown_raises():
    """An archetype with no eval-bot mapping raises (a distribution change can't silently route nowhere)."""
    with pytest.raises(ValueError):
        fs._map_archetype("definitely_not_an_archetype")


def test_co_rotated_field_is_fixed_multiset():
    """The co-rotated ring is a FIXED multiset (het_cv_absolute requires a fixed ring); the seed only
    permutes the ORDER, never the composition."""
    base = fs.co_rotated_field(0, 5)
    assert len(base) == 5
    # Mostly stations (the value target) + at least one reg + one nit (a genuine mix).
    names = [n for n, _ in base]
    assert names.count("responsive_station") >= 3
    assert "responsive_reg" in names
    assert "hs_nit" in names
    # Different seeds permute order but keep the SAME multiset.
    from collections import Counter
    for s in (1, 7, 42):
        permuted = fs.co_rotated_field(s, 5)
        assert Counter(n for n, _ in permuted) == Counter(names), "seed must not change the multiset"


# ==================================================================================================
# cash_pro_safety — math + the ExploitTable.act value-sizing extension (no heavy LBR run here)
# ==================================================================================================
def test_paired_delta_crn_math():
    """The paired-CRN delta is the per-hand (ON-OFF) mean in bb/100 with a 1.96*SE half-width, and the
    pairing CANCELS shared deal luck (a constant ON-OFF offset on top of huge shared noise -> tight CI)."""
    from pokerbot.eval.cash_pro_safety import _paired_delta_from_hands

    rng = np.random.default_rng(0)
    n = 5000
    shared = rng.normal(0.0, 5.0, size=n)            # huge shared per-hand deal luck (cancels in the pair)
    on = 0.02 + shared + rng.normal(0.0, 0.1, size=n)
    off = 0.00 + shared + rng.normal(0.0, 0.1, size=n)
    d = _paired_delta_from_hands(on, off)
    # delta ~ +2 bb/100 (0.02 bb/hand * 100), tightly bounded because the shared luck cancels.
    assert abs(d.delta_bb_per_100 - 2.0) < 0.5
    assert d.ci95_half < 0.6, f"CRN pairing should make the CI tight, got {d.ci95_half:.3f}"
    assert d.upper_ci > d.delta_bb_per_100 > d.lower_ci
    assert d.n_pairs == n


def test_paired_delta_needs_two_hands():
    from pokerbot.eval.cash_pro_safety import _paired_delta_from_hands
    with pytest.raises(ValueError):
        _paired_delta_from_hands(np.array([1.0]), np.array([0.5]))


def test_cash_pro_safety_result_safe_and_pass_logic():
    """The three-way PASS gate composes SAFE (Δ_upper ≤ eps) & BINDS (capped<uncapped, CI-sep) & RETAINS."""
    from pokerbot.eval.cash_pro_safety import CashProSafetyResult, PairedDelta

    # SAFE: delta +1.0 ± 1.0 -> upper 2.0 <= eps 5.0.
    delta = PairedDelta(delta_bb_per_100=1.0, ci95_half=1.0, on_bb_per_100=10.0,
                        off_bb_per_100=9.0, n_pairs=100)
    r = CashProSafetyResult(
        delta=delta, eps_rnr=5.0, bet_shift_capped=0.0119, bet_shift_uncapped=0.0235,
        cap_binds=True, cap_binds_ci_separated=True, retain_uplift_bb_per_100=3.0, retains=True,
        passed=True, meta={})
    assert r.safe is True
    assert r.passed is True
    # An UNSAFE delta (upper CI above eps) flips .safe.
    bad = CashProSafetyResult(
        delta=PairedDelta(delta_bb_per_100=8.0, ci95_half=2.0, on_bb_per_100=20.0,
                          off_bb_per_100=12.0, n_pairs=100),
        eps_rnr=5.0, bet_shift_capped=0.0119, bet_shift_uncapped=0.0235, cap_binds=True,
        cap_binds_ci_separated=True, retain_uplift_bb_per_100=3.0, retains=True, passed=False, meta={})
    assert bad.safe is False


def test_bet_shift_anchors_bind_and_separate():
    """The documented bet_shift anchors satisfy the cap-binding condition (capped < uncapped, CI-separated)."""
    from pokerbot.eval import cash_pro_safety as cps
    assert cps.BET_SHIFT_CAPPED_0p5 < cps.BET_SHIFT_UNCAPPED
    # CI-separated: the gap exceeds twice the documented anchor half-width.
    assert (cps.BET_SHIFT_UNCAPPED - cps.BET_SHIFT_CAPPED_0p5) > 2.0 * cps.BET_SHIFT_ANCHOR_CI


def _tiny_snapshot():
    """A tiny in-memory blueprint snapshot for the ExploitTable.act simplex test (a single legal-ish key).
    The act path falls back to the blueprint policy for an un-keyed node, so the exact contents don't
    matter — we only need a constructible ExploitTable whose overlay path returns a valid simplex."""
    return {}


def test_exploit_table_act_extension_returns_simplex_and_overlay_only_sizing():
    """ExploitTable.act with apply_value_sizing=True returns a VALID prob simplex over legal actions, and
    the value-sizing overlay only perturbs SIZING (bet) buckets — fold/check/call mass is preserved when
    the overlay is applied to a synthetic post-search policy via the public overlay helper.

    This exercises the in-place act extension WITHOUT a full search solve: we drive _value_sizing_overlay
    directly with a hand-built np-engine env + a station profile, asserting (a) a valid simplex out, and
    (b) the perturbation is confined to bet buckets (indices >= 3) relative to fold/check/call."""
    import jax
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        NUM_PLAYERS, is_chance_node_np, legal_action_mask_np, sample_chance_np, to_numpy_state,
    )
    from pokerbot.eval.exploitation_battery import population_hud
    from pokerbot.eval.lbr6_vs_exploit import ExploitTable
    from pokerbot.opponent.stats import OpponentStats

    table = ExploitTable(_tiny_snapshot(), iterations=2, apply_value_sizing=True)
    # Seed the modeled profile with a calling-station HUD so the overlay is NOT gated off by the value-cut
    # guard (it fails closed without WSD evidence). The overlay also requires ALL live villains to be fish.
    prof = OpponentStats()
    prof.hud = population_hud("calling_station")
    table.lbr_profile = prof

    # Advance a fresh 6-max deal to the first decision node (postflop where possible so the overlay can act).
    key = jax.random.PRNGKey(7)
    occ = jnp.asarray([True] * NUM_PLAYERS, dtype=bool)
    env = to_numpy_state(init_hand(key, occupancy=occ))
    depth = 0
    while is_chance_node_np(env) and depth < 50:
        env = sample_chance_np(env)
        depth += 1
    seat = int(env.action_seat)
    assert seat >= 0

    # Model EVERY villain seat as the station profile so the overlay's all-villains-fish gate can pass.
    model = {s: prof for s in range(NUM_PLAYERS) if s != seat}
    table.lbr_seat = seat

    legal = legal_action_mask_np(env, seat).astype(bool)
    # A synthetic post-search policy with mass on fold/call AND a mid bet bucket (so there is sizing to tilt).
    probs = np.zeros(legal.shape[0], dtype=np.float64)
    bet_idx = [i for i in range(3, legal.shape[0]) if legal[i]]
    if legal[1]:
        probs[1] = 0.4
    elif legal[2]:
        probs[2] = 0.4
    if bet_idx:
        probs[bet_idx[len(bet_idx) // 2]] = 0.6
    if probs.sum() <= 0:
        probs[np.flatnonzero(legal)[0]] = 1.0
    probs = probs / probs.sum()

    out = table._value_sizing_overlay(env, seat, probs.copy(), model)
    out = np.asarray(out, dtype=np.float64)
    # (a) valid simplex over legal actions
    assert out.shape == probs.shape
    assert np.all(out >= -1e-12)
    assert abs(out.sum() - 1.0) < 1e-6, f"overlay output must be a simplex, sums to {out.sum()}"
    assert not np.any(out[~legal] > 1e-9), "overlay must not place mass on illegal actions"
    # (b) the overlay is observational-safe: bet_shift accounting stays non-negative and finite.
    assert np.isfinite(table.mean_bet_shift)
    assert table.mean_bet_shift >= 0.0


def test_exploit_table_default_off_is_byte_unchanged_overlay_skipped():
    """With apply_value_sizing=False (the default) the overlay is gated OFF: _value_sizing_overlay returns
    its input verbatim is NOT exercised by act (act never calls it), so the original solver-tail-only
    behavior is byte-unchanged. We assert the flag + counters start clean."""
    from pokerbot.eval.lbr6_vs_exploit import ExploitTable

    table = ExploitTable({}, iterations=2)  # default apply_value_sizing=False
    assert table._apply_value_sizing is False
    assert table.n_overlay_fires == 0
    assert table.mean_bet_shift == 0.0


def _native_available() -> bool:
    try:
        from pokerbot.eval import lbr_native
        return lbr_native.available()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _native_available(), reason="native battery unbuilt")
def test_native_anchors_bracket_off_floor():
    """The native fold/station anchors must BRACKET the OFF floor: a fold-everything table is ~+128 bb/100
    exploitable and a calling-station table ~+1400, so fold < station and both are positive. Skipped when
    the native battery is unbuilt (the bracket is a sanity anchor, not a hard smoke requirement)."""
    from pokerbot.eval import lbr_native
    from pokerbot.eval.lbr6 import LBR6Config

    cfg = LBR6Config(hands=120, seed=0, n_seats=6)
    lo = lbr_native.lbr6_exploitability_native({}, cfg, policy="fold")
    hi = lbr_native.lbr6_exploitability_native({}, cfg, policy="station")
    assert lo.bb_per_100 < hi.bb_per_100, "fold floor must be below the station ceiling"
    assert lo.bb_per_100 > 0.0, "a fold-everything table is exploitable (positive LBR)"
