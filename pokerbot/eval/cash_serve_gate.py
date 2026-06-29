"""M7 cash serve-fidelity gate — the mandatory FIRST call of every cash-eval harness.

The #1 correctness risk for the 6-max cash measurement is the BLUEPRINT SERVE SKEW: the
``jax_weva_encode`` / ``weva_encoder`` encoders bake the abstraction flags
(``POKERBOT_BETTING_ABSTRACTION`` / ``POKERBOT_POSTFLOP_BUCKETS`` / ``POKERBOT_POSTFLOP_MODE`` /
``POKERBOT_POSTFLOP_NUM_BUCKETS`` and crucially ``POKERBOT_PREFLOP_LOSSLESS``) into module
constants AT IMPORT. If ANY of them differs from the dict stamped into the trained snapshot
(``snap['abstraction']``), the blueprint fingerprint MISSES on every affected node, the serve
silently falls back to a uniform-legal policy (~0% postflop coverage), and the measured win-rate
is GARBAGE — this is the mechanism that produced the notorious -559 bb/100 "config artifact"
(corrected to +79 only once all three flags matched). Because the flags are baked at import this
CANNOT be fixed post-import; the only safe response is to FAIL CLOSED *before* a single hand is
served.

``assert_cash_serve(snap)`` performs three fail-closed checks and returns the active abstraction
config for logging:

  1. ENV PRE-IMPORT GUARD. ``pokerbot.eval.lbr`` / ``lbr6`` ``setdefault`` only the first four flags
     (betting / postflop_buckets / mode / num_buckets); they do NOT set ``POKERBOT_PREFLOP_LOSSLESS``.
     A snapshot trained with lossless-preflop ON, served without the env var, keys EVERY preflop
     node to a miss. This gate DETECTS-AND-RAISES that case explicitly (the encoder already baked the
     wrong value, so it can't be repaired — only refused).
  2. STAMP CROSS-CHECK. ``lbr.check_blueprint_serve_env(expected=snap['abstraction'], strict=True)``
     raises ``RuntimeError`` on any field mismatch, including the ``depth_bucketed`` cross-check.
  3. COVERAGE PROBE. With ``POKERBOT_BP_COVERAGE=1`` it runs a short blueprint probe over a few
     ``np_engine`` deals and asserts the postflop blueprint hit-rate exceeds a floor. A fingerprint
     mismatch that somehow slipped past (1) and (2) collapses the hit-rate to ~0%, so this is the
     last empirical backstop.

The driver refuses to run if this raises. The accompanying test asserts that a deliberate single-field
fingerprint MISMATCH is DETECTED — making the -559 bb/100 artifact unrepeatable.
"""

from __future__ import annotations

import os

# Build-only dependency note: importing this module does NOT touch the encoders, so it never bakes a
# stale flag itself. The harness is still responsible for setting the abstraction env BEFORE it imports
# pokerbot; this gate's whole job is to verify that was done correctly.

# Postflop coverage floor below which we treat the serve as a fingerprint MISS. A correct serve sees
# near-total coverage on the trained streets; a flag skew collapses it toward 0. 0.5 sits comfortably
# between those two regimes (override via POKERBOT_CASH_COVERAGE_FLOOR for stricter harnesses).
_DEFAULT_COVERAGE_FLOOR = 0.5
# How many np_engine deals to fingerprint-probe. ~20 hands is enough to reach postflop on many of them
# while staying sub-second on CPU; the probe is a fidelity check, not a measurement.
_DEFAULT_PROBE_HANDS = 20

# The four flags the LBR modules setdefault, plus the ONE they do not. POKERBOT_PREFLOP_LOSSLESS is the
# trap: a lossless snapshot served without it baked ON keys every preflop node to a miss, and no amount
# of post-import fixing helps. We check it against the snapshot stamp explicitly so the failure is loud.
_REQUIRED_ENV_BAKED = {
    "POKERBOT_BETTING_ABSTRACTION": "1",
    "POKERBOT_POSTFLOP_BUCKETS": "1",
    "POKERBOT_POSTFLOP_MODE": "fast",
    "POKERBOT_POSTFLOP_NUM_BUCKETS": "200",
}


def _strat_table(snap: dict):
    """The bytes->probs blueprint table from a snapshot dict (``strat_sum``), or the dict itself if it
    is already a raw table (legacy unstamped snapshots are passed through verbatim)."""
    if isinstance(snap, dict) and "strat_sum" in snap:
        return snap["strat_sum"]
    return snap


def _coverage_floor() -> float:
    raw = os.environ.get("POKERBOT_CASH_COVERAGE_FLOOR")
    if raw is None:
        return _DEFAULT_COVERAGE_FLOOR
    try:
        return float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_COVERAGE_FLOOR


def _assert_preflop_lossless_matches_stamp(stamp: dict | None, cfg: dict) -> None:
    """RAISE if the snapshot was trained with preflop-lossless ON but the encoder baked it OFF (or vice
    versa). ``check_blueprint_serve_env`` would also catch this via the field-by-field stamp loop, but we
    surface the lossless-specific reason FIRST and explicitly, because it is the one flag the LBR modules
    do NOT setdefault — the single most common cash serve skew."""
    if not isinstance(stamp, dict) or "preflop_lossless" not in stamp:
        return  # unstamped / legacy snapshot: the env-only checks below still apply
    want = bool(stamp["preflop_lossless"])
    have = bool(cfg.get("preflop_lossless", False))
    if want != have:
        raise RuntimeError(
            "cash serve MISCONFIGURED: POKERBOT_PREFLOP_LOSSLESS mismatch "
            f"(trained={want}, baked={have}). The LBR modules setdefault the betting/postflop flags but "
            "NOT preflop_lossless, and the encoder baked the wrong value at import — every preflop node "
            "would fingerprint-MISS, collapsing coverage to ~0% (the -559 bb/100 artifact). "
            "Set POKERBOT_PREFLOP_LOSSLESS to match the snapshot BEFORE importing pokerbot, then re-run. "
            f"active={cfg}")


def _probe_postflop_coverage(snap, *, hands: int, seed: int = 1234) -> dict:
    """Serve the blueprint over ``hands`` np_engine deals and return the per-street coverage report.

    Runs ONLY when POKERBOT_BP_COVERAGE is enabled (the lookup records hit/miss into a process-global
    counter). Pure CPU, net-free, sub-second; this is a fidelity probe, not a win-rate measurement."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        NUM_PLAYERS,
        apply_action_np,
        is_chance_node_np,
        legal_action_mask_np,
        sample_chance_np,
        to_numpy_state,
    )
    from pokerbot.eval import lbr as _lbr

    table = _strat_table(snap)
    policy = _lbr.make_blueprint_policy(table)

    _lbr.bp_coverage_reset()
    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed + 1)
    occ = jnp.asarray([True] * NUM_PLAYERS, dtype=bool)  # full 6-max table (cash regime)

    for h in range(hands):
        key, k = jax.random.split(key)
        env = to_numpy_state(init_hand(k, occupancy=occ))
        depth = 0
        while (not bool(env.done)) and depth < 400:
            if is_chance_node_np(env):
                env = sample_chance_np(env)
                continue
            seat = int(env.action_seat)
            if seat < 0:
                break
            probs = np.asarray(policy(env, seat), dtype=np.float64)  # records hit/miss for this street
            legal = np.flatnonzero(probs > 0)
            if legal.size == 0:
                mask = legal_action_mask_np(env, seat)
                legal = np.flatnonzero(mask)
                a = int(rng.choice(legal))
            else:
                a = int(rng.choice(legal, p=probs[legal] / probs[legal].sum()))
            env = apply_action_np(env, a)
            depth += 1

    return _lbr.bp_coverage_report()


def assert_cash_serve(snap: dict, *, probe_hands: int | None = None,
                      coverage_floor: float | None = None) -> dict:
    """FAIL-CLOSED M7 serve-fidelity gate. Call this FIRST in every cash-eval harness.

    Verifies the served abstraction matches the trained snapshot's stamp so the blueprint fingerprint
    HITS (else ~0% postflop coverage -> degenerate play -> garbage measurement, the -559 bb/100 artifact)
    and empirically confirms postflop blueprint coverage clears a floor over a short probe. Returns the
    active abstraction config dict for logging.

    Parameters
    ----------
    snap : dict
        The loaded snapshot. May be a full stamped dict (``{"strat_sum": ..., "abstraction": ...}``) or a
        raw bytes->probs table (legacy, unstamped — env-only checks apply, no stamp cross-check).
    probe_hands : int, optional
        Hands for the coverage probe (default 20). Set 0 to skip the probe (env + stamp checks still run).
    coverage_floor : float, optional
        Minimum postflop hit-rate (default 0.5, or POKERBOT_CASH_COVERAGE_FLOOR).

    Raises
    ------
    RuntimeError
        On any of: preflop-lossless env/stamp mismatch; betting/postflop flags off; any stamp field
        mismatch incl. depth_bucketed; postflop coverage below the floor.
    """
    from pokerbot.engine.jax_weva_encode import active_abstraction_config
    from pokerbot.eval.lbr import check_blueprint_serve_env

    probe_hands = _DEFAULT_PROBE_HANDS if probe_hands is None else int(probe_hands)
    floor = _coverage_floor() if coverage_floor is None else float(coverage_floor)
    stamp = snap.get("abstraction") if isinstance(snap, dict) else None

    # (1) Read the flags ACTUALLY baked into this process at import.
    cfg = active_abstraction_config()

    # The four flags the LBR modules setdefault must be the canonical bounded-abstraction values, or the
    # blueprint table the snapshot trained under simply does not exist for these keys.
    if not (cfg["betting_abstraction"] and cfg["postflop_buckets"]):
        raise RuntimeError(
            "cash serve MISCONFIGURED: POKERBOT_BETTING_ABSTRACTION and POKERBOT_POSTFLOP_BUCKETS must "
            "both be set (mode=fast, num_buckets=200) BEFORE importing pokerbot — the encoder bakes them "
            f"at import, else postflop coverage is ~0% and the serve falls back to uniform. active={cfg}")

    # The ONE flag the LBR modules do NOT setdefault — surface its mismatch first and explicitly.
    _assert_preflop_lossless_matches_stamp(stamp, cfg)

    # (2) Full stamp cross-check (field-by-field incl. the depth_bucketed guard, lbr.py:162-176). Strict
    # when the snapshot is stamped; env-only warn for legacy unstamped snapshots (back-compat).
    check_blueprint_serve_env(expected=stamp, strict=stamp is not None)

    # (3) Empirical coverage backstop. A fingerprint mismatch that slipped past (1)/(2) collapses the
    # postflop hit-rate to ~0; a correct serve clears the floor comfortably.
    coverage: dict = {}
    if probe_hands > 0:
        prev = os.environ.get("POKERBOT_BP_COVERAGE")
        os.environ["POKERBOT_BP_COVERAGE"] = "1"
        # make_blueprint_policy reads the coverage flag at MODULE import (_BP_COVERAGE_ON), so flip the
        # module-level switch too — setting only the env var post-import would be ignored.
        from pokerbot.eval import lbr as _lbr

        prev_on = _lbr._BP_COVERAGE_ON
        _lbr._BP_COVERAGE_ON = True
        try:
            coverage = _probe_postflop_coverage(snap, hands=probe_hands)
        finally:
            _lbr._BP_COVERAGE_ON = prev_on
            if prev is None:
                os.environ.pop("POKERBOT_BP_COVERAGE", None)
            else:
                os.environ["POKERBOT_BP_COVERAGE"] = prev

        postflop_streets = ("flop", "turn", "river")
        seen = {s: coverage[s] for s in postflop_streets if s in coverage}
        if not seen:
            raise RuntimeError(
                "cash serve coverage probe reached NO postflop node in "
                f"{probe_hands} hands — cannot certify postflop fidelity; raise probe_hands. "
                f"coverage={coverage}")
        hit = sum(v["hits"] for v in seen.values())
        tot = sum(v["hits"] + v["misses"] for v in seen.values())
        rate = (hit / tot) if tot else 0.0
        if rate < floor:
            raise RuntimeError(
                f"cash serve postflop blueprint coverage {rate:.3f} < floor {floor:.3f} over "
                f"{probe_hands} probe hands — the fingerprint is MISSING (flag skew vs the trained "
                f"snapshot), so the served strategy is degenerate and any measurement is garbage (the "
                f"-559 bb/100 artifact). per-street={seen}. active={cfg}")

    return {
        "abstraction": cfg,
        "stamp": stamp,
        "coverage": coverage,
        "coverage_floor": floor,
        "probe_hands": probe_hands,
    }
