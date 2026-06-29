r"""Unified cash-eval suite driver — the ONE ranked, stake-swept, rake-aware verdict orchestrator.

This is the cash mirror of ``scripts/tournament/eval_tournament_suite.py`` (same orchestration shape:
serve-gate-first, optional-import the sibling pillars, per-cell measurement, ONE JSON, built-in progress
logging) and ``scripts/eval_cv_realistic.py`` (the CV/rotated factory + rake-sweep idiom it extends).

What it ties together (the cash thesis's measurements)
------------------------------------------------------
The deployed cash edge is value-EXTRACTION (the HUD value-SIZING + bet-frequency overlays) vs weak
players, WITHOUT becoming counter-exploitable by a pro. So a trustworthy cash verdict needs FIVE
measurements, per (stake x depth x hero-position) cell, all reported NET of the CoinPoker rake:

  * **M1 — net curve.**  The exploit-ON hero's chip-EV and NET-of-rake bb/100 vs the responsive field,
    swept over ``rake.STAKE_ORDER`` from ONE deck series (swap ``cap_bb`` only — the rake model's central
    simplification). Surfaces the BB-cap "hump" valley.
  * **M2 — exploitation UPLIFT.**  exploit-ON minus exploit-OFF (CRN-paired A/B) vs the SIZE-RESPONSIVE
    station (which actually pays off bigger value bets) — AND the size-BLIND blindness-regression control
    (uplift ~0 there), so we measure the mechanism, not a bookkeeping artifact.
  * **M3/M6 — pro-safety.**  ``Delta = LBR6(ON) - LBR6(OFF)`` exploitability (does the exploit make us
    counter-exploitable?) + the M6 cap-binding ``bet_shift`` proxy. Routed through the sibling
    ``pokerbot.eval.cash_pro_safety`` (C5) when present.
  * **M4 — heterogeneous net.**  The same net measurement vs the realistic POPULATION mix (a couple of
    fish + a weak reg + a solid reg), via the VR-correct ``het_cv_absolute`` (NO AIVAT — it gives
    negative VR on the search path) co-rotated control variate.
  * **M5 — VR provenance.**  ``var_reduction`` / ``corr`` / realized vs target half-width per cell; a
    cell whose CV did not reduce variance (or missed its target CI) is marked INCONCLUSIVE, never a clean
    PASS/FAIL.

Order (the spec's, and the only safe order): ``assert_cash_serve(snap)`` FIRST (the #1 cash correctness
risk — a blueprint served under the wrong import-baked abstraction flags fingerprint-MISSES every node,
collapses postflop coverage to ~0%, and silently mis-measures: the -559 bb/100 "config artifact"). Only
after the gate passes do we load fields and run M1/M4 -> M2 -> M3/M6 and emit ONE JSON.

Serve-skew safety
-----------------
1. ``assert_cash_serve(snap)`` (C6) runs first and FAILS CLOSED on any flag/stamp/coverage mismatch.
2. Every estimator that serves the blueprint is additionally handed the snapshot stamp so it re-asserts
   the fingerprint match (``het_cv_absolute(snap_abstraction=...)``; ``make_cash_hero`` verifies its own).
This is belt-and-suspenders: a mismatch can never silently mis-measure.

Sibling pillars (built in parallel — may be absent)
---------------------------------------------------
``cash_pro_safety`` (C5) and ``field_sampler`` (C2) are owned by OTHER builders. They are imported
OPTIONALLY: if a module (or its entry point) is absent, the corresponding measurement degrades to
``deferred`` and the suite STILL emits a valid verdict from the always-available M1/M2/M4 core (exactly
the ``eval_tournament_suite`` E-pillar discipline). NEVER hard-fail on a missing sibling.

Smoke (WSL Debian) — small N, CPU, against the DEPLOYED snapshot with the CORRECT flags:
  source ~/jaxgpu-venv/bin/activate
  cd .
  PYTHONPATH=. python scripts/eval/cash_eval_smoke.py
  PYTHONPATH=. python -m pytest tests/test_cash_eval_score.py -q

The heavy full stake x depth x position grid is deferred to AWS spot; this driver runs it unchanged
(just more decks / LBR hands). Progress logging (per-unit %, throughput, ETA, heartbeat, stall-detect)
is built IN per the standing rule.
"""

from __future__ import annotations

# ============================================================================================
# env MUST be set BEFORE importing pokerbot — the blueprint encoder bakes the abstraction flags
# AT IMPORT. A mismatch vs the snapshot stamp fingerprint-MISSES every node (-559 bb/100 artifact).
# These mirror scripts/eval/sizing_channel_ab.py + the deployed lossless-preflop serve config.
# setdefault never clobbers an operator's explicit export.
# ============================================================================================
import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")          # GPU held by the 7-max training; cash serve is CPU
_os.environ.setdefault("POKERBOT_INFO_SUITS", "1")
_os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
_os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
_os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
_os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")
_os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")  # the lossless2 ship candidate's stamp
_os.environ.setdefault("POKERBOT_SEARCH_PREFLOP", "1")   # deployed default (de-exploits the preflop hole)

import argparse
import json
import pickle
import sys
import time
from typing import Any, Optional

# Make the repo importable when run as a bare script (mirrors the sibling drivers' bootstrap).
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Atomic JSON write helper (SCRIPTS-EVAL-AUDIT CRIT). The verdict file is
# uploaded to S3 by the babysitter; a torn tail would publish a partial
# verdict. Use ``scripts._atomic_io.write_json_atomic`` (tmp + fsync +
# os.replace + parent-dir fsync; matches label_completion.write_atomic).
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from _atomic_io import write_json_atomic  # noqa: E402

# Default deployed snapshot (the lossless-preflop ship candidate). Overridable via --blueprint-snapshot.
_DEFAULT_SNAPSHOT = _os.environ.get(
    "POKERBOT_BLUEPRINT_SNAPSHOT",
    r"checkpoints/bp6max_lossless2_iter200.pkl")

# The depth grid the spec fixes (effective stack depth in bb). Cash is typically 100bb; 40/200 bracket it.
DEPTHS_BB: list[float] = [40.0, 100.0, 200.0]
N_SEATS = 6
BUTTON_SEAT = 3


# ============================================================================================
# Progress logging (the standing "log non-trivial jobs" rule) — per-unit %, throughput, ETA,
# heartbeat, stall-detect, all to STDERR so the JSON on stdout / --out stays clean.
# ============================================================================================
class _Progress:
    """Per-phase progress + throughput + ETA + heartbeat/stall-detect to STDERR. ``phase_start`` /
    ``tick`` / ``phase_done`` bracket each measurement phase; ``tick`` accounts decks/hands of work and
    emits a heartbeat every ``heartbeat_s`` and a STALL warning if no tick arrives within ``stall_s``."""

    def __init__(self, total_phases: int, *, heartbeat_s: float = 5.0, stall_s: float = 60.0,
                 enabled: bool = True):
        self.total = max(1, total_phases)
        self.done_phases = 0
        self.heartbeat_s = heartbeat_s
        self.stall_s = stall_s
        self.enabled = enabled
        self.t0 = time.time()
        self._phase_t0 = self.t0
        self._last_beat = 0.0
        self._last_tick = self.t0
        self._decks = 0
        self._hands = 0
        self._label = ""

    def _emit(self, msg: str) -> None:
        if self.enabled:
            print(msg, file=sys.stderr, flush=True)

    @staticmethod
    def _eta(eta_s: Optional[float]) -> str:
        if eta_s is None:
            return "--"
        eta_s = int(eta_s)
        if eta_s < 90:
            return f"~{eta_s}s"
        if eta_s < 5400:
            return f"~{eta_s // 60}m"
        return f"~{eta_s / 3600:.1f}h"

    def phase_start(self, idx: int, label: str) -> None:
        self._phase_t0 = time.time()
        self._last_tick = self._phase_t0
        self._label = label
        self._emit(f"[cash] phase {idx + 1}/{self.total} START  {label}")

    def tick(self, *, decks: int = 0, hands: int = 0) -> None:
        now = time.time()
        self._decks += decks
        self._hands += hands
        if (now - self._last_tick) > self.stall_s:
            self._emit(f"[cash] STALL? no progress for {now - self._last_tick:.0f}s "
                       f"on phase '{self._label}'")
        self._last_tick = now
        if (now - self._last_beat) >= self.heartbeat_s:
            self._last_beat = now
            elapsed = max(1e-9, now - self.t0)
            dps = self._decks / elapsed
            hps = self._hands / elapsed
            pct = 100.0 * self.done_phases / self.total
            per_phase = elapsed / max(1, self.done_phases) if self.done_phases else (now - self._phase_t0)
            eta = per_phase * (self.total - self.done_phases)
            self._emit(f"[cash] {self.done_phases}/{self.total} phases ({pct:.0f}%)  "
                       f"{dps:.1f} decks/s  {hps:.0f} hands/s  ETA {self._eta(eta)}  "
                       f"[{self._label}]")

    def phase_done(self, idx: int, label: str, note: str = "") -> None:
        self.done_phases += 1
        dt = time.time() - self._phase_t0
        self._emit(f"[cash] phase {idx + 1}/{self.total} DONE   {label}  ({dt:.1f}s)  {note}")


# ============================================================================================
# Optional import of the sibling pillars (built in parallel — may be absent; degrade to deferred)
# ============================================================================================
def _try_entry(modpath: str, attr: str):
    """Return ``getattr(import(modpath), attr)`` or ``None`` if the module/attr is absent (not yet built).
    A not-yet-built sibling pillar must NEVER crash the suite (the eval_tournament_suite discipline)."""
    try:
        mod = __import__(modpath, fromlist=["_"])
    except Exception:  # noqa: BLE001
        return None
    return getattr(mod, attr, None)


# ============================================================================================
# Snapshot load + the serve-fidelity gate (the #1 cash correctness risk) — MUST run FIRST
# ============================================================================================
def _load_snapshot(path: str) -> dict:
    """Load the stamped blueprint snapshot dict (``{'strat_sum': ..., 'abstraction': {...}}``). Kept as
    the FULL dict (not just strat_sum) so the serve gate + every estimator can re-assert the stamp."""
    with open(path, "rb") as f:
        snap = pickle.load(f)
    if not (isinstance(snap, dict) and "strat_sum" in snap):
        raise SystemExit(
            f"snapshot {path!r} is not a stamped dict (no 'strat_sum'); the cash serve gate needs the "
            "stamped snapshot so it can cross-check the abstraction fingerprint (the -559 bb/100 guard).")
    return snap


def _run_serve_gate(snap: dict, *, probe_hands: int, prog: _Progress) -> dict:
    """C6 ``assert_cash_serve`` — FAIL CLOSED before a single hand if the served abstraction flags differ
    from the snapshot stamp (else ~0% postflop coverage -> degenerate play -> garbage measurement)."""
    from pokerbot.eval.cash_serve_gate import assert_cash_serve

    prog.phase_start(0, "serve-fidelity gate (assert_cash_serve)")
    info = assert_cash_serve(snap, probe_hands=probe_hands)
    prog.phase_done(0, "serve-fidelity gate", note="PASS (fingerprint matches stamp)")
    return info


# ============================================================================================
# Field construction (C1 responsive homogeneous arms + C2 sampled population het, when present)
# ============================================================================================
def _responsive_homog_field(bot_name: str) -> list[tuple[str, dict | None]]:
    """A HOMOGENEOUS responsive field (all five villain seats == ``bot_name``) for the rotated A/B (M2)
    and the het_cv co-rotated control variate (M1/M4 homogeneous arm). ``responsive_station`` (the
    size-elastic payoff fish) is the M1/M2 headline; ``responsive_reg`` is the do-no-harm control."""
    return [(bot_name, None)] * (N_SEATS - 1)


def _sampled_population_field(seed: int) -> Optional[list[tuple[str, dict | None]]]:
    """The C2 ``field_sampler.co_rotated_field(seed)`` realistic population mix for M4, or None when the
    sibling is not yet built (M4-het then degrades to the homogeneous responsive_station het arm)."""
    entry = _try_entry("pokerbot.eval.field_sampler", "co_rotated_field")
    if entry is None:
        return None
    try:
        field = list(entry(seed))
        if len(field) != N_SEATS - 1:
            return None
        return field
    except Exception:  # noqa: BLE001 — a sibling that imports but errors must not crash the suite
        return None


# ============================================================================================
# Hero factories (the deployed cash stack; serves the blueprint with the verified fingerprint)
# ============================================================================================
def _build_hero_factories(snap: dict, *, iterations: int, leaf_select: str, seed: int):
    """Return ``(search_factory, baseline_factory)`` for the het_cv / rotated estimators.

    ``search_factory(exploit=...)`` builds the EXPENSIVE deployed cash hero (Pluribus search on the
    blueprint + the LIVE HUD value-SIZING overlay when ``exploit=True``). ``baseline_factory`` builds the
    CHEAP, highly-correlated blueprint-only baseline (the control variate). Both share the verified raw
    table (the strict stamp check runs ONCE inside ``hero_factory``)."""
    from pokerbot.eval.cash_hero import hero_factory

    # FIXED CFR iterations => deterministic solves => CRN-clean ON/OFF arms (a wall-clock budget makes
    # the iteration count load-dependent and breaks reproducibility). search >> baseline iters so the
    # baseline is near-blueprint (maximal CRN correlation -> maximal variance reduction).
    search_factory = hero_factory(snap, iterations=iterations, pluribus_leaf_select=leaf_select,
                                  seed=seed)
    baseline_factory = hero_factory(snap, iterations=max(1, iterations // 10),
                                    pluribus_leaf_select=leaf_select, seed=seed)
    return search_factory, baseline_factory


# ============================================================================================
# M1 / M4 — the net-of-rake curve over STAKE_ORDER (ONE deck series, swap cap_bb)
# ============================================================================================
def _measure_net_curve(field_specs, *, search_factory, baseline_factory, snap_abstraction,
                       paired_decks: int, cheap_decks: int, depth_bb: float, button_seat: int,
                       seed: int, stakes: list[str], prog: _Progress, phase_idx: int,
                       label: str) -> dict:
    """Run ``het_cv_absolute`` ONCE per stake (the rake cost is a function of the pot distribution at the
    stake; the chip-EV arm is byte-identical across stakes). Returns the per-stake net + chip CV blocks
    and the VR provenance. The het estimator co-rotates the FIXED ring and bans AIVAT (positive VR)."""
    from pokerbot.eval.het_cv_absolute import het_cv_absolute, VarReductionError

    prog.phase_start(phase_idx, label)
    out: dict[str, Any] = {"depth_bb": depth_bb, "by_stake": {}, "field": [s[0] for s in field_specs]}

    n_stakes = max(1, len(stakes))
    for si, stake in enumerate(stakes):
        def _cb(tag, d, total, _si=si, _stake=stake):
            prog.tick(decks=1)

        try:
            res, meta = het_cv_absolute(
                field_specs,
                search_factory=search_factory, baseline_factory=baseline_factory,
                paired_decks=paired_decks, cheap_decks=cheap_decks, seed=seed,
                n_seats=N_SEATS, button_seat=button_seat,
                net_of_rake_stake=stake, rakeback_mode="modeled",
                snap_abstraction=snap_abstraction, progress_cb=_cb)
            chip = res.chip
            net = res.net
            out["by_stake"][stake] = {
                "reducer": res.reducer,
                "chip_bb_per_100": res.chip_bb_per_100,
                "chip_ci95_bb_per_100": [c * 100.0 for c in chip.ci95],
                "chip_var_reduction": chip.var_reduction,
                "chip_corr": chip.corr,
                "net_bb_per_100": res.net_bb_per_100,
                "net_ci95_bb_per_100": ([c * 100.0 for c in net.ci95] if net is not None else None),
                "net_var_reduction": (net.var_reduction if net is not None else None),
                "net_corr": (net.corr if net is not None else None),
                "decks_paired": chip.decks_paired,
                "decks_cheap": chip.decks_cheap,
            }
        except VarReductionError as exc:
            # A non-reducing CV is a mis-wired baseline (the AIVAT-on-search failure class). Record the
            # cell INCONCLUSIVE rather than crashing the whole sweep — M5 provenance captures the why.
            out["by_stake"][stake] = {"reducer": "FAILED", "error": str(exc),
                                      "var_reduction": None}
    prog.phase_done(phase_idx, label, note=f"{n_stakes} stakes")
    return out


# ============================================================================================
# M2 — exploitation UPLIFT (CRN-paired ON/OFF A/B) on the responsive field + blindness control
# ============================================================================================
def _measure_uplift(hero_factory_fn, *, field_name: str, control_field_name: str,
                    decks: int, depth_bb: float, button_seat: int, seed: int,
                    stake: str, prog: _Progress, phase_idx: int) -> dict:
    """A/B the exploit-ON vs exploit-OFF hero (CRN-paired, ON/OFF share the per-hand seed) vs the
    SIZE-RESPONSIVE station, then repeat vs the size-BLIND control (the blindness regression). The
    deployed value-SIZING overlay should print a positive uplift on the responsive station and ~0 on the
    size-blind field (which cannot express the size->fold-vs-pay mechanism)."""
    from pokerbot.eval.exploitation_battery import run_field_absolute_rotated

    prog.phase_start(phase_idx, f"M2 uplift A/B (responsive vs size-blind control) @ {depth_bb}bb")

    def _run(field_name_local: str) -> dict:
        field = _responsive_homog_field(field_name_local)

        def _cb(d, total):
            prog.tick(decks=1)

        # rotated requires a HOMOGENEOUS field; net_of_rake_stake adds the net column; the rotated path
        # consumes the SIZE-RESPONSIVE villains via build_bot (responsive_station/responsive_reg are
        # registered there). One run yields off_base, on_exploit, uplift + net_of_rake.
        return run_field_absolute_rotated(
            field, decks=decks, seed=seed, hero_factory=hero_factory_fn,
            n_seats=N_SEATS, button_seat=button_seat, allin_ev=True,
            net_of_rake_stake=stake, rakeback_mode="modeled", progress_cb=_cb,
            progress_every=max(1, decks // 5))

    responsive = _run(field_name)
    control = _run(control_field_name)

    def _block(res: dict) -> dict:
        up_mean = res["uplift_bb_per_hand"] * 100.0
        up_se = res["uplift_se"] * 100.0
        return {
            "field": res["field"],
            "off_base_bb_per_100": res["off_base"]["bb_per_100"],
            "on_exploit_bb_per_100": res["on_exploit"]["bb_per_100"],
            "uplift_bb_per_100": up_mean,
            "uplift_se_bb_per_100": up_se,
            "uplift_ci95_bb_per_100": [up_mean - 1.96 * up_se, up_mean + 1.96 * up_se],
            "net_of_rake": res.get("net_of_rake"),
            "decks": res["decks"],
        }

    prog.phase_done(phase_idx, "M2 uplift A/B")
    return {"depth_bb": depth_bb, "stake": stake,
            "responsive": _block(responsive), "size_blind_control": _block(control)}


# ============================================================================================
# M3 / M6 — pro-safety (Delta exploitability + bet_shift) via the sibling cash_pro_safety (C5)
# ============================================================================================
def _measure_pro_safety(snap: dict, *, lbr_hands: int, depth_bb: float, seed: int,
                        prog: _Progress, phase_idx: int) -> dict:
    """Route M3/M6 through the sibling ``pokerbot.eval.cash_pro_safety.cash_pro_safety`` (C5). Degrades
    to ``deferred`` when that module is not yet built — the suite still emits a valid M1/M2/M4 verdict
    (the spec's gate-still-passes contract)."""
    entry = _try_entry("pokerbot.eval.cash_pro_safety", "cash_pro_safety")
    if entry is None:
        return {"status": "deferred",
                "reason": "pokerbot.eval.cash_pro_safety (C5) not yet built — M3/M6 deferred"}
    prog.phase_start(phase_idx, f"M3/M6 pro-safety (LBR6 Delta + bet_shift) @ {depth_bb}bb")
    try:
        # cash_pro_safety takes hands/seed as KEYWORDS (NOT a cfg dict — its ``cfg`` param is an
        # LBR6Config object whose .n_seats it reads; passing a dict there breaks). It returns a
        # CashProSafetyResult dataclass; normalize it to the flat dict the verdict assembler reads
        # (delta_bb_per_100 + the bb/100 Delta CI + the M6 bet_shift). native_bracket=False keeps the
        # smoke fast + avoids depending on the unbuilt native battery.
        res = entry(snap, hands=int(lbr_hands), seed=int(seed), native_bracket=False)
        prog.phase_done(phase_idx, "M3/M6 pro-safety")
        return {"status": "built", "result": _normalize_pro_safety(res)}
    except Exception as exc:  # noqa: BLE001 — a sibling that errors must not crash the suite
        prog.phase_done(phase_idx, "M3/M6 pro-safety", note=f"ERROR {exc!r}")
        return {"status": "error", "reason": repr(exc)}


def _normalize_pro_safety(res) -> dict:
    """Flatten a ``cash_pro_safety.CashProSafetyResult`` into the dict the verdict assembler + the smoke
    read (``delta_bb_per_100`` / ``delta_ci95_bb_per_100`` / ``bet_shift`` / ``passed`` / ``safe``). The
    Delta CI is ``[lower_ci, upper_ci]`` from the PairedDelta (already in bb/100)."""
    d = res.delta
    return {
        "delta_bb_per_100": float(d.delta_bb_per_100),
        "delta_ci95_bb_per_100": [float(d.lower_ci), float(d.upper_ci)],
        "delta_upper_ci": float(d.upper_ci),
        "bet_shift": float(res.bet_shift_capped),
        "bet_shift_uncapped": float(res.bet_shift_uncapped),
        "cap_binds": bool(res.cap_binds),
        "cap_binds_ci_separated": bool(res.cap_binds_ci_separated),
        "eps_rnr": float(res.eps_rnr),
        "safe": bool(res.safe),
        "passed": (None if res.passed is None else bool(res.passed)),
        "n_pairs": int(d.n_pairs),
    }


# ============================================================================================
# Verdict assembly — per-(stake x depth x position) cells -> per-stake GO/NO-GO
# ============================================================================================
def _assemble_cells(net_curve: dict, uplift: dict, pro_safety: dict, *, depth_bb: float,
                    position: int) -> tuple[list[dict], dict]:
    """Turn the M1 net curve + M2 uplift + M3/M6 pro-safety into per-stake cell rows + the hump table.

    The M2 uplift (exploit-ON vs OFF) is rake-invariant (the rake cost is policy-agnostic at a fixed
    stake), so the SAME uplift CI is attached to every stake's cell at this (depth, position). The net-ON
    arm is the M1 net curve's per-stake value; the cell verdict is computed on the NET CI with the M5 VR
    provenance override."""
    from pokerbot.eval.cash_eval_score import (
        CellResult, cell_verdict, hump_table, stake_go_no_go,
    )

    # M3 Delta CI (one number for this depth; None when M3 deferred/errored).
    lbr_delta = None
    lbr_delta_ci95 = None
    bet_shift = None
    if pro_safety.get("status") == "built":
        r = pro_safety.get("result", {})
        lbr_delta = r.get("delta_bb_per_100", r.get("lbr_delta"))
        ci = r.get("delta_ci95_bb_per_100", r.get("lbr_delta_ci95"))
        lbr_delta_ci95 = tuple(ci) if ci is not None else None
        bet_shift = r.get("bet_shift", r.get("mean_bet_shift"))

    up = uplift.get("responsive", {})
    up_ci = tuple(up.get("uplift_ci95_bb_per_100", (0.0, 0.0)))

    cells: list[dict] = []
    net_by_stake: dict[str, float] = {}
    for stake, blk in net_curve.get("by_stake", {}).items():
        net_on = blk.get("net_bb_per_100")
        net_ci = blk.get("net_ci95_bb_per_100")
        net_vr = blk.get("net_var_reduction")
        net_corr = blk.get("net_corr")
        if net_on is None or net_ci is None:
            # CV failed (VarReductionError) -> INCONCLUSIVE cell, no curve point.
            cells.append({
                "stake": stake, "depth_bb": depth_bb, "position": position,
                "net_on": None, "verdict": "INCONCLUSIVE (CV failed)",
                "var_reduction": net_vr, "error": blk.get("error"),
            })
            continue
        net_by_stake[stake] = float(net_on)
        # net_off = net_on - uplift (chip uplift == net uplift, rake-invariant). off arm not separately
        # net-measured by het_cv (it serves exploit=True); reconstruct via the rake-invariant uplift.
        net_off = float(net_on) - float(up.get("uplift_bb_per_100", 0.0))
        v = cell_verdict(net_ci[0], net_ci[1], var_reduction=net_vr)
        cell = CellResult(
            stake=stake, depth_bb=depth_bb, position=position,
            net_off=net_off, net_on=float(net_on),
            net_uplift=float(up.get("uplift_bb_per_100", 0.0)), uplift_ci95=up_ci,
            var_reduction=float(net_vr) if net_vr is not None else 0.0,
            corr=float(net_corr) if net_corr is not None else 0.0,
            ci95=tuple(net_ci), verdict=v,
            lbr_delta=lbr_delta, lbr_delta_ci95=lbr_delta_ci95, bet_shift=bet_shift,
        )
        cells.append(cell.as_json())

    ht = hump_table(net_by_stake)
    return cells, ht


def _per_stake_go_no_go(all_cells: list[dict], *, eps_rnr: float) -> dict:
    """Aggregate the per-cell rows into ONE GO/NO-GO per stake (over all depths/positions of that stake).

    Uses the BEST (deepest, most CI-tight) cell's net CI for the net-positive gate, the uplift CI (shared
    across stakes at a depth), and the M3 Delta CI for pro-safety."""
    from pokerbot.eval.cash_eval_score import stake_go_no_go

    by_stake: dict[str, list[dict]] = {}
    for c in all_cells:
        if c.get("net_on") is None:
            continue
        by_stake.setdefault(c["stake"], []).append(c)

    out: dict[str, dict] = {}
    for stake, cells in by_stake.items():
        # representative cell = the one with the tightest net CI (smallest half-width).
        def _hw(c):
            lo, hi = c["ci95"]
            return (hi - lo)
        rep = min(cells, key=_hw)
        verdicts = [c["verdict"] for c in cells]
        out[stake] = stake_go_no_go(
            net_on_ci95=tuple(rep["ci95"]),
            uplift_ci95=tuple(rep["uplift_ci95"]),
            lbr_delta_ci95=(tuple(rep["lbr_delta_ci95"]) if rep.get("lbr_delta_ci95") else None),
            eps_rnr=eps_rnr, cell_verdicts=verdicts)
    return out


# ============================================================================================
# The suite driver
# ============================================================================================
def _emit_gate_failure_report(*, snapshot: str, out: str, gate_exc: Exception,
                              depths: list[float], stakes: list[str], positions: list[int],
                              paired_decks: int, cheap_decks: int, uplift_decks: int,
                              lbr_hands: int, iterations: int, leaf_select: str, seed: int,
                              probe_hands: int, eps_rnr: float, station_field: str,
                              reg_field: str, size_blind_field: str, uplift_stake: str,
                              quiet: bool, t0: float) -> None:
    """SCRIPTS-EVAL-AUDIT HIGH: on serve-gate failure, write a fail-loud partial verdict JSON so the
    babysitter / watcher have a record of the abort (the gate error message is the most useful signal).

    The JSON uses the SAME schema as the happy-path report (``cash_eval_suite/v1``) with:

      * ``status`` = ``"GATE_FAILED"``
      * ``cells`` = ``[]`` (no measurements taken)
      * ``go_no_go_by_stake`` = ``{}`` (no per-stake verdict)
      * ``summary`` populated so the dashboard still has numeric anchors

    The write uses ``write_json_atomic`` so a power-cut during this failure
    write does not corrupt the file the babysitter uploads to S3.
    """
    report = {
        "schema": "cash_eval_suite/v1",
        "status": "GATE_FAILED",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 3),
        "config": {
            "snapshot": snapshot, "depths_bb": depths, "stakes": stakes, "positions": positions,
            "paired_decks": paired_decks, "cheap_decks": cheap_decks, "uplift_decks": uplift_decks,
            "lbr_hands": lbr_hands, "iterations": iterations, "leaf_select": leaf_select, "seed": seed,
            "eps_rnr": eps_rnr, "station_field": station_field, "size_blind_field": size_blind_field,
            "uplift_stake": uplift_stake,
            "abstraction_flags": {k: _os.environ.get(k) for k in (
                "POKERBOT_BETTING_ABSTRACTION", "POKERBOT_POSTFLOP_BUCKETS", "POKERBOT_POSTFLOP_MODE",
                "POKERBOT_POSTFLOP_NUM_BUCKETS", "POKERBOT_PREFLOP_LOSSLESS", "POKERBOT_SEARCH_PREFLOP")},
        },
        "gate_failure": {
            "error_type": type(gate_exc).__name__,
            "error_message": str(gate_exc),
        },
        "cells": [],
        "per_depth": {},
        "go_no_go_by_stake": {},
        "summary": {
            "n_cells": 0, "n_cells_positive": 0, "n_cells_negative": 0,
            "n_stakes_go": 0, "n_stakes_no_go": 0, "pro_safety_built": False,
        },
    }
    if out:
        try:
            write_json_atomic(out, report, indent=2)
            if not quiet:
                print(f"[cash] wrote GATE_FAILED partial -> {out}", file=sys.stderr, flush=True)
        except Exception as write_exc:  # noqa: BLE001
            # The gate already failed; a write failure here would compound
            # the noise. Print to stderr and continue (the raised gate_exc
            # still propagates so the babysitter sees exit != 0).
            print(f"[cash] GATE_FAILED; could not write partial JSON ({write_exc!r})",
                  file=sys.stderr, flush=True)


def run_suite(*, snapshot: str, depths: list[float], stakes: list[str], positions: list[int],
              paired_decks: int, cheap_decks: int, uplift_decks: int, lbr_hands: int,
              iterations: int, leaf_select: str, seed: int, probe_hands: int, eps_rnr: float,
              station_field: str, reg_field: str, size_blind_field: str, uplift_stake: str,
              out: str, quiet: bool) -> dict:
    """Drive serve-gate -> M1/M4 net curve -> M2 uplift A/B -> M3/M6 pro-safety -> per-stake verdict,
    emit ONE JSON. Returns the report dict."""
    t0 = time.time()

    # phases: 1 gate + per depth {M1 station, M4 het, M2 uplift, M3 pro-safety} = 1 + 4*depths.
    n_phases = 1 + len(depths) * 4
    prog = _Progress(n_phases, enabled=not quiet)

    snap = _load_snapshot(snapshot)
    snap_abstraction = snap.get("abstraction")

    # ---- (0) serve-fidelity gate FIRST (FAIL CLOSED) -------------------------------------------
    # SCRIPTS-EVAL-AUDIT HIGH: when the gate raises, the babysitter sees a
    # non-zero exit AND no JSON is published, so the watcher has no
    # record of WHY the run aborted. Catch the gate failure here, write
    # a fail-loud partial verdict JSON with the gate error message, then
    # re-raise so the exit code is still non-zero (the babysitter's
    # "failed gate -> stop the loop" behavior must still trigger).
    try:
        gate_info = _run_serve_gate(snap, probe_hands=probe_hands, prog=prog)
    except Exception as gate_exc:
        _emit_gate_failure_report(
            snapshot=snapshot, out=out, gate_exc=gate_exc,
            depths=depths, stakes=stakes, positions=positions,
            paired_decks=paired_decks, cheap_decks=cheap_decks,
            uplift_decks=uplift_decks, lbr_hands=lbr_hands,
            iterations=iterations, leaf_select=leaf_select, seed=seed,
            probe_hands=probe_hands, eps_rnr=eps_rnr,
            station_field=station_field, reg_field=reg_field,
            size_blind_field=size_blind_field, uplift_stake=uplift_stake,
            quiet=quiet, t0=t0)
        raise

    # ---- hero factories (verified-fingerprint blueprint serve) ---------------------------------
    search_factory, baseline_factory = _build_hero_factories(
        snap, iterations=iterations, leaf_select=leaf_select, seed=seed)

    phase = 1
    per_depth: dict[float, dict] = {}
    all_cells: list[dict] = []
    pro_safety_by_depth: dict[float, dict] = {}

    for depth in depths:
        # M1 — net curve vs the size-RESPONSIVE station (homogeneous), swept over stakes.
        net_curve = _measure_net_curve(
            _responsive_homog_field(station_field),
            search_factory=search_factory, baseline_factory=baseline_factory,
            snap_abstraction=snap_abstraction, paired_decks=paired_decks, cheap_decks=cheap_decks,
            depth_bb=depth, button_seat=BUTTON_SEAT, seed=seed, stakes=stakes,
            prog=prog, phase_idx=phase, label=f"M1 net curve (responsive_station) @ {depth}bb")
        phase += 1

        # M4 — net vs the realistic POPULATION mix (C2 field_sampler), else the responsive het arm.
        het_field = _sampled_population_field(seed)
        m4 = None
        if het_field is not None:
            m4 = _measure_net_curve(
                het_field, search_factory=search_factory, baseline_factory=baseline_factory,
                snap_abstraction=snap_abstraction, paired_decks=paired_decks, cheap_decks=cheap_decks,
                depth_bb=depth, button_seat=BUTTON_SEAT, seed=seed + 13, stakes=stakes,
                prog=prog, phase_idx=phase, label=f"M4 het net curve (population mix) @ {depth}bb")
        else:
            m4 = {"status": "deferred",
                  "reason": "pokerbot.eval.field_sampler (C2) not yet built — M4 het uses M1 station arm"}
            prog.phase_done(phase, f"M4 het net curve @ {depth}bb", note="deferred (no field_sampler)")
        phase += 1

        # M2 — exploitation uplift A/B + the size-blind blindness regression control.
        uplift = _measure_uplift(
            _rotated_hero_factory(snap, iterations=iterations, leaf_select=leaf_select, seed=seed),
            field_name=station_field, control_field_name=size_blind_field,
            decks=uplift_decks, depth_bb=depth, button_seat=BUTTON_SEAT, seed=seed,
            stake=uplift_stake, prog=prog, phase_idx=phase)
        phase += 1

        # M3/M6 — pro-safety (sibling C5; deferred when absent). Runs ONCE per depth. When the sibling is
        # absent it is a no-cost deferred phase (we still advance the phase counter so the ETA is honest).
        pro_safety = _measure_pro_safety(
            snap, lbr_hands=lbr_hands, depth_bb=depth, seed=seed, prog=prog, phase_idx=phase)
        if pro_safety.get("status") == "deferred":
            prog.phase_done(phase, f"M3/M6 pro-safety @ {depth}bb", note="deferred (no cash_pro_safety)")
        phase += 1
        pro_safety_by_depth[depth] = pro_safety

        # ---- assemble cells (one position per the spec; loop positions for the full grid) ----
        for position in positions:
            cells, ht = _assemble_cells(net_curve, uplift, pro_safety,
                                        depth_bb=depth, position=position)
            all_cells.extend(cells)
            per_depth.setdefault(depth, {})[f"pos{position}"] = {"hump_table": ht}

        per_depth[depth]["m1_net_curve"] = net_curve
        per_depth[depth]["m2_uplift"] = uplift
        per_depth[depth]["m3_m6_pro_safety"] = pro_safety
        per_depth[depth]["m4_het"] = m4

    # ---- per-stake GO/NO-GO + report -----------------------------------------------------------
    go_no_go = _per_stake_go_no_go(all_cells, eps_rnr=eps_rnr)

    report = {
        "schema": "cash_eval_suite/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 3),
        "config": {
            "snapshot": snapshot, "depths_bb": depths, "stakes": stakes, "positions": positions,
            "paired_decks": paired_decks, "cheap_decks": cheap_decks, "uplift_decks": uplift_decks,
            "lbr_hands": lbr_hands, "iterations": iterations, "leaf_select": leaf_select, "seed": seed,
            "eps_rnr": eps_rnr, "station_field": station_field, "size_blind_field": size_blind_field,
            "uplift_stake": uplift_stake,
            "abstraction_flags": {k: _os.environ.get(k) for k in (
                "POKERBOT_BETTING_ABSTRACTION", "POKERBOT_POSTFLOP_BUCKETS", "POKERBOT_POSTFLOP_MODE",
                "POKERBOT_POSTFLOP_NUM_BUCKETS", "POKERBOT_PREFLOP_LOSSLESS", "POKERBOT_SEARCH_PREFLOP")},
        },
        "serve_gate": gate_info,
        "cells": all_cells,
        "per_depth": {str(d): v for d, v in per_depth.items()},
        "go_no_go_by_stake": go_no_go,
        "summary": {
            "n_cells": len(all_cells),
            "n_cells_positive": sum(1 for c in all_cells if c.get("verdict") == "POSITIVE"),
            "n_cells_negative": sum(1 for c in all_cells if c.get("verdict") == "NEGATIVE"),
            "n_stakes_go": sum(1 for v in go_no_go.values() if v["verdict"] == "GO"),
            "n_stakes_no_go": sum(1 for v in go_no_go.values() if v["verdict"] == "NO-GO"),
            "pro_safety_built": any(p.get("status") == "built" for p in pro_safety_by_depth.values()),
        },
    }

    if out:
        # Atomic JSON write (SCRIPTS-EVAL-AUDIT CRIT) — torn-tail-safe + parent-dir fsync.
        write_json_atomic(out, report, indent=2)
        if not quiet:
            print(f"[cash] wrote {out}", file=sys.stderr, flush=True)
    return report


def _rotated_hero_factory(snap: dict, *, iterations: int, leaf_select: str, seed: int):
    """The rotated A/B needs a ``hero_factory(exploit: bool) -> CashHero`` (NOT keyword-only) matching the
    ``run_field_absolute_rotated`` contract. ``cash_hero.hero_factory`` returns ``factory(*, exploit)``;
    wrap it so the rotated path's positional/keyword call both work."""
    from pokerbot.eval.cash_hero import hero_factory

    base = hero_factory(snap, iterations=iterations, pluribus_leaf_select=leaf_select, seed=seed)

    def factory(*, exploit: bool):
        return base(exploit=exploit)

    return factory


# ============================================================================================
# CLI
# ============================================================================================
def _parse_args(argv: list[str]) -> argparse.Namespace:
    from pokerbot.economics.rake import STAKE_ORDER

    ap = argparse.ArgumentParser(
        description="Unified cash-eval suite: stake-swept, rake-aware, per-(stake x depth x position) "
                    "verdict (M1 net curve / M2 uplift / M3-M6 pro-safety / M4 het net / M5 VR).")
    ap.add_argument("--blueprint-snapshot", default=_DEFAULT_SNAPSHOT,
                    help="stamped blueprint .pkl (the serve gate cross-checks its abstraction stamp)")
    ap.add_argument("--depths", default="40,100,200",
                    help="comma-list of effective stack depths in bb (default 40,100,200)")
    ap.add_argument("--stakes", default="all",
                    help="comma-list of rake.STAKE_ORDER keys, or 'all' (default all) — the net sweep")
    ap.add_argument("--positions", default="3", help="comma-list of hero positions (button-relative)")
    ap.add_argument("--paired-decks", type=int, default=40, help="EXPENSIVE CRN-paired search decks (M1/M4)")
    ap.add_argument("--cheap-decks", type=int, default=120, help="cheap blueprint-only decks (M1/M4)")
    ap.add_argument("--uplift-decks", type=int, default=40, help="rotated A/B decks (M2)")
    ap.add_argument("--lbr-hands", type=int, default=2000, help="LBR6 hands per arm (M3 pro-safety)")
    ap.add_argument("--iterations", type=int, default=80, help="FIXED CFR iters for the search hero")
    ap.add_argument("--leaf-select", default="uniform", choices=("uniform", "gadget"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-hands", type=int, default=20, help="serve-gate coverage probe hands")
    ap.add_argument("--eps-rnr", type=float, default=5.0,
                    help="pro-safety tolerance: max LBR6 Delta upper-CI (bb/100) for a GO")
    ap.add_argument("--station-field", default="responsive_station")
    ap.add_argument("--reg-field", default="responsive_reg")
    ap.add_argument("--size-blind-field", default="calling_station",
                    help="legacy SIZE-BLIND loose-passive station for the M2 blindness regression — its "
                         "continue/fold reads hand strength only (not bet size), so the value-SIZING "
                         "overlay's uplift should be ~0 there (the mechanism is field-specific). Must be a "
                         "build_bot name (the rotated path uses build_bot, not the calibrated builder).")
    ap.add_argument("--uplift-stake", default="0.50/1",
                    help="single stake for the M2 net column (the rake-invariant uplift is stake-agnostic)")
    ap.add_argument("--out", default="artifacts/cash_eval_suite.json")
    ap.add_argument("--quiet", action="store_true", help="silence the stderr progress log")
    args = ap.parse_args(argv[1:])

    args._stake_order = STAKE_ORDER
    return args


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    stake_order = args._stake_order

    if args.stakes.strip().lower() == "all":
        stakes = list(stake_order)
    else:
        stakes = [s.strip() for s in args.stakes.split(",") if s.strip()]
        bad = [s for s in stakes if s not in stake_order]
        if bad:
            raise SystemExit(f"unknown stake(s) {bad}; allowed = {stake_order}")
    depths = [float(d) for d in args.depths.split(",") if d.strip()]
    positions = [int(p) for p in args.positions.split(",") if p.strip()]

    report = run_suite(
        snapshot=args.blueprint_snapshot, depths=depths, stakes=stakes, positions=positions,
        paired_decks=args.paired_decks, cheap_decks=args.cheap_decks, uplift_decks=args.uplift_decks,
        lbr_hands=args.lbr_hands, iterations=args.iterations, leaf_select=args.leaf_select,
        seed=args.seed, probe_hands=args.probe_hands, eps_rnr=args.eps_rnr,
        station_field=args.station_field, reg_field=args.reg_field,
        size_blind_field=args.size_blind_field, uplift_stake=args.uplift_stake,
        out=args.out, quiet=args.quiet)

    s = report["summary"]
    print(f"[cash] {s['n_cells']} cells | positive={s['n_cells_positive']} negative={s['n_cells_negative']} "
          f"| stakes GO={s['n_stakes_go']} NO-GO={s['n_stakes_no_go']} | "
          f"pro_safety_built={s['pro_safety_built']}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
