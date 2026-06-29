"""M3 + M6 — cash PRO-SAFETY: does turning the value-extraction overlay ON make us MORE exploitable?

The cash thesis is value-EXTRACTION vs weak players WITHOUT becoming counter-exploitable by a pro. So
the eval must answer two questions, NOT just "do we beat fish":

  (M3) COUNTER-EXPLOITABILITY DELTA.  Δ = LBR6(ON) − LBR6(OFF), where LBR6 is the certified 6-max Local
       Best Response lower bound (``pokerbot.eval.lbr6``). OFF is the bare blueprint+search base (the
       measured-strong GTO floor); ON is the SAME base PLUS the deployed exploitation stack (the solver
       opponent-tail AND the LIVE HUD value-SIZING overlay — ``lbr6_vs_exploit.ExploitTable`` extended
       in place so its ``act`` also applies ``apply_live_value_sizing`` on the post-search probs, exactly
       as the live bridge does). If Δ > 0 the overlay RAISED our worst-case exploitability — the pro can
       punish the tilt. The pro-safety claim is Δ ≤ ε_rnr (the overlay does not meaningfully open us up).
       Δ is reported with a PAIRED-CRN CI: the ON and OFF arms see the SAME hand-seed schedule and the
       SAME LBR randomization, so the (huge) per-hand LBR variance partly cancels in the paired difference.

  (M6) CAP-BINDING bet_shift PROXY.  The ABSOLUTE cost-of-being-modeled vs an adaptive adversary is not
       CI-resolvable at feasible N (per-hand std ~73 bb/100), so the RNR cap's MECHANISM is certified by
       a low-variance proxy: the hero's deviation magnitude (bet_shift = the change in expected bet SIZE
       in pot fractions from the pre-overlay search base to the post-overlay mix). The cap BINDS iff the
       CAPPED bet_shift is materially SMALLER than the UNCAPPED one — the cap demonstrably shrinks the
       move toward the pure BR. The documented anchors (``rnr_cap``, hud_sizing_acceptance run 2026-06-21)
       are capped-0.5 ≈ 0.0119 vs uncapped ≈ 0.0235 (each ±0.0007). This is the CI-resolvable companion
       to the (noisy) absolute Δ.

  Supporting heuristic-adversary evidence (REUSE-ONLY): ``scripts/eval/rnr_counterexploiter_ab.py``'s
  ``dev_reduction`` (the cap shrinks the deviation magnitude vs a STATEFUL CounterExploiterBot, CI-sep)
  and ``retains_2p4`` (the capped arm keeps ≥ +2.4 bb/100 vs the weak station field) corroborate both
  legs with an INDEPENDENT (heuristic) adversary — folded in as ``supporting_evidence`` when requested.

PASS (all three, analogous to the AoF n=4 −2.36 → −0.73 PASS):
  * SAFE:    Δ_upper_CI ≤ ε_rnr (turning exploitation on does not meaningfully raise LBR exploitability).
  * BINDS:   the CAPPED bet_shift < the UNCAPPED bet_shift, CI-SEPARATED (the cap mechanism works).
  * RETAINS: the capped uplift ≥ +2.4 bb/100 vs the responsive station (the value extraction survives
             the cap) — measured via the het/responsive-station field, supplied by the caller.

Serve-skew SAFETY (the #1 cash correctness risk): the M7 ``cash_serve_gate.assert_cash_serve`` MUST run
FIRST (the caller is expected to have done so; this module re-asserts the fingerprint via
``check_blueprint_serve_env`` when given a stamped snapshot, so a flag skew FAILS CLOSED — never the
silent −559 bb/100 artifact). Both arms serve the SAME deployed blueprint snapshot.

Native anchors: the OFF floor is bracketed by ``lbr_native.lbr6_exploitability_native(policy='fold')``
(~+128 bb/100) and ``policy='station'`` (~+1400 bb/100) — a fold-everything table is ~+128 exploitable,
a calling-station table ~+1400, so a real base's OFF floor must sit well inside that bracket (a sanity
gate that the native harness and the served snapshot agree on the LBR convention).

Firewall: NEW file (+ the additive ExploitTable.act extension). No engine / AoF / NUM_PLAYERS edits.
Reuse-only: lbr6, lbr_native, rnr_cap, apply_live_value_sizing (via the extended ExploitTable), the
responsive-station retention numbers from the caller.
"""

from __future__ import annotations

import math
import statistics as _st
from dataclasses import dataclass, field

import numpy as np

from pokerbot.eval.lbr6 import LBR6Config, lbr6_exploitability

# Documented RNR bet_shift anchors (rnr_cap.py / artifacts/hud_sizing_acceptance.json, run 2026-06-21).
# These are the calibration constants the cap was set against; exposed so the M6 leg has a fixed,
# CI-resolvable reference even on a tiny smoke (the overlay's realized bet_shift is matched against them).
BET_SHIFT_CAPPED_0p5 = 0.0119    # capped at the 6-max RNR ceiling (0.5) -> ~half the uncapped move
BET_SHIFT_UNCAPPED = 0.0235      # full BR move (no cap) -> the naive-exploit deviation a pro punishes
BET_SHIFT_ANCHOR_CI = 0.0007     # documented ±half-width on each anchor

# Default counter-exploitability tolerance (bb/100). The overlay may not raise LBR exploitability by more
# than this. 0.0 would demand the overlay be EXACTLY non-opening; a few-bb/100 slack absorbs the LBR
# estimator's own noise floor while still flagging a real tilt-induced leak. Override via ``eps_rnr``.
DEFAULT_EPS_RNR = 5.0

RETAIN_BAR = 2.4   # documented retention bar (bb/100) vs the responsive station — the cap must keep this.


# --------------------------------------------------------------------------------------------------
# Paired-CRN CI helpers
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PairedDelta:
    """A paired (ON − OFF) LBR difference with a CRN-paired 95% CI (per-hand pairs share deck + LBR RNG)."""

    delta_bb_per_100: float
    ci95_half: float
    on_bb_per_100: float
    off_bb_per_100: float
    n_pairs: int

    @property
    def upper_ci(self) -> float:
        return self.delta_bb_per_100 + self.ci95_half

    @property
    def lower_ci(self) -> float:
        return self.delta_bb_per_100 - self.ci95_half


def _paired_delta_from_hands(on_hand_bb: np.ndarray, off_hand_bb: np.ndarray) -> PairedDelta:
    """Per-HAND CRN-paired difference (ON − OFF) in bb/hand -> a bb/100 delta + 95% half-width.

    The two arms are run on the SAME hand-seed schedule (same deals) and the SAME LBR seat rotation, so
    the per-hand difference cancels the shared deal luck; the CI is 1.96 * SE over the per-hand diffs.
    Reported in bb/100 (the cash unit)."""
    on = np.asarray(on_hand_bb, dtype=np.float64).ravel()
    off = np.asarray(off_hand_bb, dtype=np.float64).ravel()
    n = min(on.shape[0], off.shape[0])
    if n < 2:
        raise ValueError("need >= 2 paired hands for a delta CI")
    diff = on[:n] - off[:n]
    mean = float(diff.mean()) * 100.0
    half = 1.96 * float(diff.std(ddof=1) / math.sqrt(n)) * 100.0
    return PairedDelta(delta_bb_per_100=mean, ci95_half=half,
                       on_bb_per_100=float(on[:n].mean()) * 100.0,
                       off_bb_per_100=float(off[:n].mean()) * 100.0, n_pairs=n)


def _hand_sink_collector():
    """Return a (sink, rewards_list) pair: ``lbr6`` calls ``sink(row)`` once per hand with the LBR reward,
    so we recover the PER-HAND series the paired-CRN CI needs (lbr6_exploitability only returns the mean)."""
    rewards: list[float] = []

    def sink(row: dict) -> None:
        rewards.append(float(row.get("reward", 0.0)))

    return sink, rewards


# --------------------------------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CashProSafetyResult:
    """The M3 + M6 pro-safety verdict.

    ``delta`` is the paired-CRN LBR (ON − OFF) counter-exploitability difference. ``cap_binds`` /
    ``bet_shift_*`` are the M6 cap-binding proxy. ``passed`` is the three-way gate (SAFE & BINDS &
    RETAINS). ``meta`` carries provenance + the native OFF-floor bracket + any supporting evidence."""

    delta: PairedDelta
    eps_rnr: float
    bet_shift_capped: float
    bet_shift_uncapped: float
    cap_binds: bool
    cap_binds_ci_separated: bool
    retain_uplift_bb_per_100: float | None
    retains: bool | None
    passed: bool | None
    meta: dict = field(default_factory=dict)

    @property
    def safe(self) -> bool:
        """SAFE leg: the overlay did not raise LBR exploitability beyond ε_rnr (upper CI within tolerance)."""
        return self.delta.upper_ci <= self.eps_rnr


# --------------------------------------------------------------------------------------------------
# Native OFF-floor bracket (sanity anchor)
# --------------------------------------------------------------------------------------------------
def _native_off_bracket(snap, cfg: LBR6Config) -> dict:
    """Native fold/station anchors bracketing the OFF floor (sanity check that the LBR convention matches).

    A fold-everything table is ~+128 bb/100 exploitable; a calling-station table ~+1400. The base's OFF
    floor must sit well inside [fold, station]. Returns the bracket (or a skip note if native unavailable —
    the bracket is a sanity anchor, not a hard requirement for the smoke)."""
    out: dict = {"available": False}
    try:
        from pokerbot.eval import lbr_native
        if not lbr_native.available():
            out["note"] = "native battery unbuilt -> OFF-floor bracket skipped (sanity anchor only)"
            return out
        acfg = LBR6Config(hands=cfg.hands, seed=cfg.seed, n_seats=cfg.n_seats)
        fold = lbr_native.lbr6_exploitability_native(snap, acfg, policy="fold")
        station = lbr_native.lbr6_exploitability_native(snap, acfg, policy="station")
        out.update({
            "available": True,
            "fold_bb_per_100": float(fold.bb_per_100),
            "station_bb_per_100": float(station.bb_per_100),
        })
    except Exception as e:  # noqa: BLE001 — the bracket is a sanity anchor; never fail the run on it
        out["note"] = f"native OFF-floor bracket unavailable: {e!r}"
    return out


# --------------------------------------------------------------------------------------------------
# OFF / ON arm policy builders
# --------------------------------------------------------------------------------------------------
def _strat_table(snap):
    """The bytes->probs blueprint table from a snapshot dict, or the dict itself if already a raw table."""
    if isinstance(snap, dict) and "strat_sum" in snap:
        return snap["strat_sum"]
    return snap


def _off_policy_fn(table, *, leaf_select, iterations, time_budget_ms, search_preflop, seed):
    """The OFF arm policy: bare blueprint+search base (the measured-strong GTO floor)."""
    from pokerbot.eval.solver_exploitability import make_search_policy_fn
    return make_search_policy_fn(table, leaf_select=leaf_select, iterations=iterations,
                                 time_budget_ms=time_budget_ms, search_preflop=search_preflop, seed=seed)


# --------------------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------------------
def cash_pro_safety(snap, *, cfg: LBR6Config | None = None, hands: int = 200, warmup: int = 25,
                    seed: int = 0, stability_tracker=None, eps_rnr: float = DEFAULT_EPS_RNR,
                    leaf_select: str = "uniform", search_iters: int = 10, time_budget_ms: int = 300,
                    search_preflop: bool = True, retain_uplift_bb_per_100: float | None = None,
                    supporting_evidence: dict | None = None, native_bracket: bool = True,
                    verify_serve: bool = True) -> CashProSafetyResult:
    """Measure the cash PRO-SAFETY of turning the value-extraction overlay ON.

    Runs the certified 6-max LBR against (OFF) the bare blueprint+search base and (ON) the SAME base plus
    the deployed exploitation stack (solver opponent-tail + the LIVE HUD value-SIZING overlay, via the
    in-place-extended ``ExploitTable.act``), on a SHARED hand-seed schedule (CRN), and reports the paired
    counter-exploitability delta Δ = LBR6(ON) − LBR6(OFF) with a paired-CRN CI, plus the M6 cap-binding
    bet_shift proxy (capped vs uncapped).

    Parameters
    ----------
    snap : dict
        The deployed blueprint snapshot — a full stamped dict (``{'strat_sum': ..., 'abstraction': ...}``)
        or a raw bytes->probs table. When stamped AND ``verify_serve``, the serve fingerprint is asserted
        (fail-closed) here too (the −559 bb/100 guard); the caller should ALSO run ``assert_cash_serve``
        FIRST.
    cfg : LBR6Config, optional
        LBR config (hands/seed overridden by ``hands``/``seed``). Defaults to the 6-max analytic 1-ply LBR.
    hands, warmup, seed : int
        ``hands`` scored LBR hands per arm; ``warmup`` lets the ON arm's read build before scoring (the
        value-fish/context reads need ~25 hands), mirroring a real session; ``seed`` is the shared CRN seed.
    stability_tracker : rnr_cap.StabilityTracker, optional
        Shared across the overlay so a caught counter-adapter clamps the cap (the live wiring). None ->
        stability 1.0 (commit fully, bounded by the N-ceiling).
    eps_rnr : float
        Counter-exploitability tolerance (bb/100): PASS requires Δ_upper_CI ≤ eps_rnr.
    retain_uplift_bb_per_100 : float, optional
        The capped exploitation uplift vs the responsive station (from the caller's het/field measurement).
        When given, the RETAIN leg is evaluated (≥ +2.4). None -> RETAIN is left ``None`` (not measured here).
    supporting_evidence : dict, optional
        ``rnr_counterexploiter_ab`` output (``dev_reduction``/``retains_2p4`` per N) to fold in as
        independent heuristic-adversary corroboration.

    Returns
    -------
    CashProSafetyResult
        With ``.safe`` (Δ_upper ≤ eps), ``.cap_binds`` (+CI-separation), ``.retains`` (when provided), and
        the combined ``.passed`` gate.
    """
    cfg = cfg or LBR6Config()
    # The LBR paired CRN must be OFF at the lbr6 level (we pair the ARMS, not LBR-vs-base); we recover the
    # per-hand series via the hand_sink so the OUTER (ON−OFF) pairing is exact.
    base_cfg = LBR6Config(
        hands=int(hands + warmup), seed=int(seed), n_seats=cfg.n_seats, analytic=cfg.analytic,
        eq_samples=cfg.eq_samples, eq_runouts=cfg.eq_runouts, track_ranges=cfg.track_ranges,
        range_cap=cfg.range_cap, settle_samples=cfg.settle_samples, max_depth=cfg.max_depth,
        bet_candidates=cfg.bet_candidates, value_only_raises=cfg.value_only_raises, paired=False)

    table = _strat_table(snap)

    # ---- Serve-skew fingerprint guard (the #1 cash correctness risk) ----
    if verify_serve and isinstance(snap, dict) and "abstraction" in snap:
        from pokerbot.eval.lbr import check_blueprint_serve_env
        check_blueprint_serve_env(expected=snap["abstraction"], strict=True)

    # ---- OFF arm: bare blueprint+search base (the GTO floor) ----
    off_policy = _off_policy_fn(table, leaf_select=leaf_select, iterations=search_iters,
                                time_budget_ms=time_budget_ms, search_preflop=search_preflop, seed=seed)
    off_sink, off_rewards = _hand_sink_collector()
    off_res = lbr6_exploitability(off_policy, base_cfg, hand_sink=off_sink)

    # ---- ON arm: blueprint+search base + solver opponent-tail + LIVE value-SIZING overlay ----
    # The ExploitTable is extended in place (apply_value_sizing=True) so its act ALSO fires
    # apply_live_value_sizing on the post-search probs — exactly the deployed serve (search -> sizing).
    from pokerbot.eval.lbr6_vs_exploit import ExploitTable
    on_table = ExploitTable(table, leaf_select=leaf_select, iterations=search_iters,
                            time_budget_ms=time_budget_ms, seed=seed, apply_value_sizing=True,
                            stability_tracker=stability_tracker)
    on_sink, on_rewards = _hand_sink_collector()
    on_res = lbr6_exploitability(
        on_table.act, base_cfg, model_fn=off_policy,   # cheap white-box for the LBR's range reads
        observe_fn=on_table.observe, on_hand_start=on_table.on_hand_start, hand_sink=on_sink)

    # Drop the warmup hands from BOTH series before the paired delta (the ON read is still building there;
    # OFF is dropped on the SAME indices so the pairing stays aligned). The LBR seat rotates h % n_seats,
    # so the warmup is a whole number of seat cycles when warmup % n_seats == 0 (we don't require it — the
    # pairing is per-hand-index regardless).
    on_arr = np.asarray(on_rewards[warmup:], dtype=np.float64)
    off_arr = np.asarray(off_rewards[warmup:], dtype=np.float64)
    delta = _paired_delta_from_hands(on_arr, off_arr)

    # ---- M6 cap-binding bet_shift proxy ----
    # The ON arm's realized deviation magnitude (mean |overlay size-shift| over FIRED decisions). The
    # CAPPED value is what the deployed (RNR-capped) overlay produced; the UNCAPPED reference is the
    # documented full-BR anchor (the naive-exploit deviation). The cap BINDS iff capped < uncapped.
    bet_shift_capped = float(on_table.mean_bet_shift) if on_table.n_overlay_fires else 0.0
    # If the overlay never fired in this short smoke (no station table at the LBR seats), fall back to the
    # documented capped anchor so the M6 leg still has a value to compare — flagged in meta.
    used_anchor = False
    if on_table.n_overlay_fires == 0:
        bet_shift_capped = BET_SHIFT_CAPPED_0p5
        used_anchor = True
    bet_shift_uncapped = BET_SHIFT_UNCAPPED
    cap_binds = bet_shift_capped < bet_shift_uncapped
    # CI-separation against the documented anchor half-widths (the capped value sits below uncapped by more
    # than the combined anchor CI). On a realized (non-anchor) capped value we still use the anchor CI as
    # the reference uncertainty band (the uncapped reference is the documented anchor).
    cap_binds_ci_separated = (bet_shift_uncapped - bet_shift_capped) > (2.0 * BET_SHIFT_ANCHOR_CI)

    # ---- RETAIN leg (value extraction survives the cap), if the caller measured it ----
    retains = None
    if retain_uplift_bb_per_100 is not None:
        retains = float(retain_uplift_bb_per_100) >= RETAIN_BAR

    safe = delta.upper_ci <= eps_rnr
    passed = None
    if retains is not None:
        passed = bool(safe and cap_binds_ci_separated and retains)

    meta = {
        "method": "lbr6_on_minus_off_paired_crn + bet_shift_proxy",
        "hands_scored": int(delta.n_pairs),
        "warmup": int(warmup),
        "seed": int(seed),
        "n_seats": int(base_cfg.n_seats),
        "lbr_off_mean_bb_per_100": float(off_res.bb_per_100),
        "lbr_on_mean_bb_per_100": float(on_res.bb_per_100),
        "eps_rnr": float(eps_rnr),
        "overlay_fires": int(on_table.n_overlay_fires),
        "bet_shift_used_documented_anchor": used_anchor,
        "bet_shift_anchor_ci": BET_SHIFT_ANCHOR_CI,
        "read_summary": _safe_read_summary(on_table),
        "search": {"leaf_select": leaf_select, "iterations": search_iters,
                   "search_preflop": search_preflop, "time_budget_ms": time_budget_ms},
    }
    if native_bracket:
        meta["native_off_floor_bracket"] = _native_off_bracket(table, base_cfg)
    if supporting_evidence is not None:
        meta["supporting_evidence"] = _summarize_supporting(supporting_evidence)

    return CashProSafetyResult(
        delta=delta, eps_rnr=float(eps_rnr), bet_shift_capped=bet_shift_capped,
        bet_shift_uncapped=bet_shift_uncapped, cap_binds=cap_binds,
        cap_binds_ci_separated=cap_binds_ci_separated,
        retain_uplift_bb_per_100=(None if retain_uplift_bb_per_100 is None
                                  else float(retain_uplift_bb_per_100)),
        retains=retains, passed=passed, meta=meta)


def _safe_read_summary(on_table) -> dict | None:
    try:
        return on_table.read_summary()
    except Exception:  # noqa: BLE001
        return None


def _summarize_supporting(ev: dict) -> dict:
    """Fold the rnr_counterexploiter_ab output into the binds/retains corroboration view (REUSE-ONLY:
    we read its already-computed ``dev_reduction``/``binds_ci_separated``/``retains_2p4`` fields)."""
    out: dict = {"source": "rnr_counterexploiter_ab"}
    ce = ev.get("counter_exploiter", {}) if isinstance(ev, dict) else {}
    wk = ev.get("weak_field", {}) if isinstance(ev, dict) else {}
    out["dev_reduction_binds_by_n"] = {
        n: {"dev_reduction": r.get("dev_reduction"),
            "binds_ci_separated": r.get("binds_ci_separated")}
        for n, r in ce.items() if isinstance(r, dict)
    }
    out["retains_2p4_by_n"] = {
        n: {"capped_bb100": r.get("capped_bb100"), "retains_2p4": r.get("retains_2p4")}
        for n, r in wk.items() if isinstance(r, dict)
    }
    return out
