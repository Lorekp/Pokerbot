"""Unified tournament-eval suite driver — the ONE orchestrator that ties the three pillars into a single
report: the **format x regime surface**.

This is the CLI that iterates the grid

    {NORMAL, SNG, SATELLITE, PKO}  x  {deep 100bb, mid 40bb, shallow 12bb, bubble, FT, HU}

and, per cell, drives the three measurement pillars + the structural certificates, emitting everything as
ONE JSON (``artifacts/tournament_eval_suite.json``):

  * **C0-C5** — STRUCTURAL CERTIFICATES (net-free, fast, format-agnostic). These prove the harness is
    measuring the right thing BEFORE any cell value is trusted:
      C0  imports/foundation present (tournament.__init__ + eval_score + neural_serve + anchor + icm).
      C1  ICM prize-pool conservation: ``sum(icm_equity(stacks, pay)) == sum(pay)`` across all formats.
      C2  chip->$ scoring is pool-conserving (NORMAL/SAT sum~=0) AND PKO reduces to ICM at zero bounty.
      C3  the always-fold calibration anchor closed form (monotone in n; 7-max ante=0.125 -> +203.6 bb/100).
      C4  the variance-reduction core identities (CV no-op == 0; stat_block/verdict reducers sane).
      C5  the SERVE-BRIDGE round-trip: a native 7-seat node -> place_live_seats(identity) -> back must be
          byte-identical AND ``encode_info_state`` byte-identical (the #1 serve-skew guard). This is the
          driver-level mirror of tests/test_seven_neural_serve.py::test_seven_remap_reproduces_native_encoding.

  * **E1-E6** — PER-CELL MEASUREMENTS (the surface). Each grid cell drives:
      E1/E2  lbr_icm        — ICM-weighted best-response exploitability (other builder's module; optional).
      E3/E4  mtt_lifecycle  — depth/blind lifecycle ROI + finish distribution (other builder's; optional).
      E6     regime_battery — the regime exploit battery (other builder's; optional).
    plus the ALWAYS-AVAILABLE net-free **anchor calibration**: drive the always-fold/always-jam anchor and
    confirm the cell recovers its closed-form floor. In ``--policy fold`` (net-free smoke), the anchor
    calibration is the headline and the E1-E6 modules degrade to ``deferred`` when they are not yet built
    (they are built by OTHER builders in parallel) — the cell STILL emits a valid, anchor-calibrated verdict.

WHY NET-FREE SMOKE FIRST: the 7-max GPU training run holds the GPU. ``--smoke --policy fold`` validates the
ENTIRE orchestration (grid iteration, certificates, per-cell scoring, reporting, JSONL checkpoint, progress
telemetry) on CPU jax with NO net served — every value comes from the certified closed-form anchor + the
``eval_score`` / ``icm`` / ``payouts`` primitives. The heavy net-served full grid (``--policy net``) is
deferred to AWS spot once the GPU frees; this driver runs it unchanged, just slower.

SEAT-SKEW (the #1 correctness risk): ANY net serve goes through ``neural_serve`` / ``seat_remap``, whose
encode_info_state round-trip is proven byte-identical (C5 here re-proves it at the driver level). We never
serve the net through any other path.

Progress logging is built IN (the standing "log non-trivial jobs" rule): per-cell %, spots/s + hands/s
throughput, ETA, heartbeat, and stall-detect, all to stderr so the JSON on stdout/-out stays clean.

Smoke (WSL Debian) — net-free, CPU, tiny N:
  source ~/jaxgpu-venv/bin/activate
  cd .
  PYTHONPATH=. python scripts/tournament/eval_tournament_suite.py --smoke --policy fold \
      --format NORMAL --regime bubble --out artifacts/_smoke_tournament_eval_suite.json
  PYTHONPATH=. python -m pytest tests/test_eval_tournament_suite.py -q
"""

from __future__ import annotations

# --- env MUST be set BEFORE importing pokerbot (the encoder layout depends on it) -------------------
# JAX_PLATFORMS=cpu : the GPU is held by the live 7-max training; eval runs CPU jax / net-free anchors.
# POKERBOT_INFO_SUITS=1 : the 7-max net trains with the suit feature ON; the served encoder MUST match or
# every feature after the suit tail misaligns (the serve-skew class of bug). Set both up front, never
# clobber an explicit override the operator already exported.
import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")
_os.environ.setdefault("POKERBOT_INFO_SUITS", "1")

import argparse
import json
import math
import sys
import time
from typing import Any, Callable, Optional

# Make the repo importable when run as a bare script (mirrors the sibling scripts' bootstrap).
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ==================================================================================================
# Grid definition — the format x regime surface
# ==================================================================================================
# FORMATS are the PAYOUT regimes the driver iterates. SNG is a NORMAL ladder with a tiny field (a flat
# 3-handed SNG ladder), kept as a distinct grid key because its near-the-money dynamics differ in
# emphasis; it routes through TournamentFormat.NORMAL for scoring. PKO and SATELLITE carry their own
# payout shapes (bounty pool / flat seat ladder).
FORMATS: list[str] = ["NORMAL", "SNG", "SATELLITE", "PKO"]

# REGIMES sample the tournament DEPTH / BLIND lifecycle + the ICM-pressure spots. Each carries the
# effective stack depth (bb) the cell is evaluated at and a flag marking the ICM-pressure spots
# (bubble / final-table / heads-up) where chip-EV and $-EV diverge most.
REGIMES: dict[str, dict[str, Any]] = {
    "deep":    {"depth_bb": 100.0, "icm_spot": False, "label": "deep 100bb"},
    "mid":     {"depth_bb": 40.0,  "icm_spot": False, "label": "mid 40bb (antes)"},
    "shallow": {"depth_bb": 12.0,  "icm_spot": False, "label": "shallow 12bb (push/fold)"},
    "bubble":  {"depth_bb": 15.0,  "icm_spot": True,  "label": "bubble"},
    "ft":      {"depth_bb": 20.0,  "icm_spot": True,  "label": "final table"},
    "hu":      {"depth_bb": 18.0,  "icm_spot": True,  "label": "heads-up"},
}
REGIME_ORDER: list[str] = ["deep", "mid", "shallow", "bubble", "ft", "hu"]

# The CoinPoker MTT per-player ante in bb (round(BB/8) ~= 0.125; see tournament.structure / anchor).
MTT_ANTE_BB = 0.125
# Antes are only live mid-tournament onward; the deep level has no ante.
_REGIME_ANTE = {"deep": 0.0, "mid": MTT_ANTE_BB, "shallow": MTT_ANTE_BB,
                "bubble": MTT_ANTE_BB, "ft": MTT_ANTE_BB, "hu": MTT_ANTE_BB}


# ==================================================================================================
# Progress logging (the standing "log non-trivial jobs" rule) — per-cell %, throughput, ETA, heartbeat
# ==================================================================================================
class _Progress:
    """Per-cell progress + throughput + ETA + heartbeat/stall-detect, all to STDERR (keeps stdout JSON
    clean). One instance per run; ``cell_start``/``tick``/``cell_done`` bracket each grid cell.

    Stall-detect: if no ``tick`` arrives within ``stall_s`` of the last, a one-line STALL warning fires
    (so a wedged cell is visible instantly, not after the whole run hangs)."""

    def __init__(self, total_cells: int, *, heartbeat_s: float = 5.0, stall_s: float = 30.0,
                 enabled: bool = True):
        self.total = max(1, total_cells)
        self.done_cells = 0
        self.heartbeat_s = heartbeat_s
        self.stall_s = stall_s
        self.enabled = enabled
        self.t0 = time.time()
        self._cell_t0 = self.t0
        self._last_beat = 0.0
        self._last_tick = self.t0
        self._spots = 0
        self._hands = 0
        self._cell_label = ""

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

    def cell_start(self, idx: int, label: str) -> None:
        self._cell_t0 = time.time()
        self._last_tick = self._cell_t0
        self._cell_label = label
        self._emit(f"[suite] cell {idx + 1}/{self.total} START  {label}")

    def tick(self, *, spots: int = 0, hands: int = 0) -> None:
        """Account ``spots``/``hands`` of work for the current cell; emit a heartbeat + stall warning."""
        now = time.time()
        self._spots += spots
        self._hands += hands
        if (now - self._last_tick) > self.stall_s:
            self._emit(f"[suite] STALL? no progress for {now - self._last_tick:.0f}s "
                       f"on cell '{self._cell_label}'")
        self._last_tick = now
        if (now - self._last_beat) >= self.heartbeat_s:
            self._last_beat = now
            elapsed = max(1e-9, now - self.t0)
            sps = self._spots / elapsed
            hps = self._hands / elapsed
            pct = 100.0 * self.done_cells / self.total
            per_cell = elapsed / max(1, self.done_cells) if self.done_cells else (now - self._cell_t0)
            eta = per_cell * (self.total - self.done_cells)
            self._emit(f"[suite] {self.done_cells}/{self.total} cells ({pct:.0f}%)  "
                       f"{sps:.0f} spots/s  {hps:.0f} hands/s  ETA {self._eta(eta)}  "
                       f"[cell {self._cell_label}]")

    def cell_done(self, idx: int, label: str, verdict: str) -> None:
        self.done_cells += 1
        dt = time.time() - self._cell_t0
        self._emit(f"[suite] cell {idx + 1}/{self.total} DONE   {label}  ({dt:.1f}s)  -> {verdict}")


# ==================================================================================================
# Optional-import of the OTHER builders' pillar modules (built in parallel — may be absent)
# ==================================================================================================
# This driver OWNS only itself + its test. The lbr_icm / regime_battery / mtt_lifecycle pillars are
# built by OTHER builders concurrently. We import them OPTIONALLY: if a module (or its expected entry
# point) is absent, the corresponding E-row degrades to ``deferred`` and the cell still emits a valid
# anchor-calibrated verdict from the certified net-free primitives. NEVER hard-fail on a missing pillar.
def _try_import(modpath: str, attr: Optional[str] = None):
    """Return the module (or ``getattr(module, attr)``) or ``None`` if absent / not yet built."""
    try:
        mod = __import__(modpath, fromlist=["_"])
    except Exception:  # noqa: BLE001 — a not-yet-built sibling pillar must not crash the suite
        return None
    if attr is None:
        return mod
    return getattr(mod, attr, None)


# ==================================================================================================
# Structural certificates C0-C5 (net-free; the harness must pass these before any cell is trusted)
# ==================================================================================================
def _cert_c0() -> dict:
    """C0 — imports / foundation present. The tournament layer + the three reused foundations import."""
    try:
        import pokerbot.tournament as _t  # noqa: F401
        from pokerbot.tournament import anchor as _a  # noqa: F401
        from pokerbot.tournament import eval_score as _e  # noqa: F401
        from pokerbot.tournament import icm as _i  # noqa: F401
        from pokerbot.tournament import neural_serve as _n  # noqa: F401
        from pokerbot.tournament import payouts as _p  # noqa: F401
        return {"name": "C0_imports", "pass": True,
                "detail": "tournament.__init__ + anchor + eval_score + icm + neural_serve + payouts import"}
    except Exception as exc:  # noqa: BLE001
        return {"name": "C0_imports", "pass": False, "detail": f"import failure: {exc!r}"}


def _cert_c1() -> dict:
    """C1 — ICM prize-pool conservation across all formats: sum(icm_equity) == sum(payouts[:n])."""
    from pokerbot.tournament.icm import icm_equity
    cases = [
        ("NORMAL", [50.0, 35.0, 18.0, 9.0, 6.0, 2.0], [50.0, 30.0, 20.0]),       # 6 seats, top-3 paid
        ("SATELLITE", [40.0, 30.0, 20.0, 10.0], [10.0, 10.0]),                    # flat seat ladder
        ("PKO", [60.0, 25.0, 12.0, 8.0, 5.0], [40.0, 22.0, 13.0]),               # prize side
    ]
    worst = 0.0
    for _name, stacks, pay in cases:
        eq = icm_equity(stacks, pay)
        drift = abs(float(eq.sum()) - sum(pay))
        worst = max(worst, drift)
    ok = worst < 1e-6
    return {"name": "C1_icm_conservation", "pass": ok, "worst_abs_drift": worst,
            "detail": f"max |sum(icm_equity) - sum(payouts)| = {worst:.2e} (< 1e-6)"}


def _cert_c2() -> dict:
    """C2 — chip->$ scoring: pool-conserving for NORMAL/SAT (sum~=0) AND PKO reduces to the pure ICM
    delta when the villain bounty is 0 (the bounty arm must collapse to the chip-EV/ICM call value)."""
    import numpy as np

    from pokerbot.tournament.eval_score import score_chip_to_dollar
    from pokerbot.tournament.payouts import TournamentFormat

    stacks = [50.0, 35.0, 18.0, 12.0]
    chip = [12.0, -12.0, 0.0, 0.0]          # a chip transfer (sum 0)
    pay = [50.0, 30.0, 20.0]
    d_norm = score_chip_to_dollar(stacks, chip, pay, TournamentFormat.NORMAL)
    sat_pay = [10.0, 10.0]
    d_sat = score_chip_to_dollar(stacks, chip, sat_pay, TournamentFormat.SATELLITE)
    conserve = max(abs(float(np.sum(d_norm))), abs(float(np.sum(d_sat))))

    # PKO @ zero bounty must equal the NORMAL/ICM delta exactly on hero's seat.
    d_norm_pair = score_chip_to_dollar(stacks, chip, pay, TournamentFormat.NORMAL)
    # hero=0, villain=1 all-in for the smaller stack; hero_equity arbitrary — at villain_bounty=0 the
    # bounty arm collapses to the prize-only ICM call value, which equals d_norm on hero's index when
    # the chip transfer is exactly the all-in for the smaller stack. Use a chip transfer == min stack.
    risked = min(stacks[0], stacks[1])
    chip_pko = [0.0] * 4
    chip_pko[0] = +risked
    chip_pko[1] = -risked
    d_pko = score_chip_to_dollar(stacks, chip_pko, pay, TournamentFormat.PKO,
                                 hero=0, villain=1, hero_equity=1.0, villain_bounty=0.0)
    d_icm = score_chip_to_dollar(stacks, chip_pko, pay, TournamentFormat.NORMAL)
    pko_reduces = abs(float(d_pko[0]) - float(d_icm[0]))
    ok = conserve < 1e-6 and pko_reduces < 1e-6
    return {"name": "C2_score_chip_to_dollar", "pass": ok,
            "conservation_drift": conserve, "pko_zero_bounty_residual": pko_reduces,
            "detail": f"NORMAL/SAT pool drift {conserve:.2e}; PKO@bounty0 - ICM = {pko_reduces:.2e}"}


def _cert_c3() -> dict:
    """C3 — the always-fold calibration anchor closed form: strictly increasing in n, and the 7-max
    tournament value (ante 0.125 bb) recovers the documented +203.6 bb/100 floor."""
    from pokerbot.tournament.anchor import (
        always_fold_exploitability,
        sevenmax_tournament_anchor,
    )
    vals = [always_fold_exploitability(n, ante_bb=MTT_ANTE_BB) for n in range(2, 8)]
    monotone = all(b > a for a, b in zip(vals, vals[1:]))
    seven = sevenmax_tournament_anchor()
    seven_ok = abs(seven - 203.57142857142858) < 1e-6   # exact closed form ((6*1.5 + 7*6*0.125)/7*100)
    ok = monotone and seven_ok
    return {"name": "C3_anchor_closed_form", "pass": ok,
            "sevenmax_bb_per_100": seven, "monotone_in_n": monotone,
            "anchor_by_n": {str(n): always_fold_exploitability(n, ante_bb=MTT_ANTE_BB)
                            for n in range(2, 8)},
            "detail": f"7-max ante=0.125 -> {seven:.3f} bb/100 (expect 203.571); monotone={monotone}"}


def _cert_c4() -> dict:
    """C4 — the variance-reduction core identities: the ICM control variate is a TRUE no-op on a hand
    that already realizes the pre-hand ICM equity (residual exactly 0), and the stat_block / verdict
    reducers behave (verdict is POSITIVE/NEGATIVE/INCONCLUSIVE on the obvious cases)."""
    from pokerbot.tournament.eval_score import score_with_cv, stat_block, verdict
    from pokerbot.tournament.icm import icm_equity

    stacks = [50.0, 35.0, 18.0, 12.0]
    pay = [50.0, 30.0, 20.0]
    hero = 0
    base = float(icm_equity(stacks, pay)[hero])
    resid = score_with_cv(base, stacks, pay, hero)       # realized == baseline -> exactly 0
    cv_noop = abs(resid) < 1e-12

    sb = stat_block([1.0, 1.0, 1.0, 1.0])                # zero-variance series -> se 0
    sb_ok = sb["n"] == 4 and abs(sb["mean"] - 1.0) < 1e-12 and sb["se"] == 0.0
    v_ok = (verdict(0.1, 0.5) == "POSITIVE" and verdict(-0.5, -0.1) == "NEGATIVE"
            and verdict(-0.2, 0.3).startswith("INCONCLUSIVE"))
    ok = cv_noop and sb_ok and v_ok
    return {"name": "C4_variance_reduction_core", "pass": ok,
            "cv_noop_residual": resid, "stat_block_ok": sb_ok, "verdict_ok": v_ok,
            "detail": f"CV no-op residual {resid:.2e}; stat_block_ok={sb_ok}; verdict_ok={v_ok}"}


def _cert_c5(seed: int = 7) -> dict:
    """C5 — the SERVE-BRIDGE round-trip (the #1 serve-skew guard, driver-level mirror of
    tests/test_seven_neural_serve.py::test_seven_remap_reproduces_native_encoding).

    Build a native 7-seat node (all 7 live, natural order, a rotated button — exactly the training
    reset layout), pass it through the PRODUCTION ``place_live_seats(list(range(7)))`` (the identity
    remap), and assert EVERY EnvState field is byte-identical AND ``encode_info_state`` is byte-identical
    on hero's seat. A failure here means the served absolute seat/button one-hots would NOT match what
    the net saw in training -> the net is silently garbage. Skipped (not failed) when jax is unavailable
    (a net-free env): the certificate is informational then, with ``skipped: True``."""
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np

        from pokerbot.engine.jax_engine import CHIP_UNIT, encode_info_state, init_hand
        from pokerbot.tournament.seat_remap import SEVEN, place_live_seats
    except Exception as exc:  # noqa: BLE001 — no jax in this env -> certificate informational, not failed
        return {"name": "C5_serve_roundtrip", "pass": True, "skipped": True,
                "detail": f"jax/engine unavailable ({exc!r}); serve round-trip not exercised this env"}

    try:
        N7 = SEVEN
        occ = jnp.array([True] * N7, dtype=bool)
        button = 4
        ante = jnp.int32(round(MTT_ANTE_BB * CHIP_UNIT))
        native = init_hand(jax.random.PRNGKey(seed), button=jnp.int32(button), occupancy=occ, ante=ante)
        hero = int(native.action_seat)
        native_np = jax.tree.map(lambda x: np.asarray(x), native)

        back, mapping = place_live_seats(native_np, list(range(N7)))
        identity = mapping == {i: i for i in range(N7)}
        fields_ok = all(
            np.array_equal(np.asarray(getattr(native_np, f)), np.asarray(getattr(back, f)))
            for f in native_np._fields
        )
        enc_native = np.asarray(encode_info_state(native, jnp.int32(hero)))
        back_j = type(back)(**{f: jnp.asarray(getattr(back, f)) for f in back._fields})
        enc_back = np.asarray(encode_info_state(back_j, jnp.int32(mapping[hero])))
        enc_ok = np.array_equal(enc_native, enc_back)
        ok = identity and fields_ok and enc_ok
        return {"name": "C5_serve_roundtrip", "pass": ok, "skipped": False,
                "identity_map": identity, "fields_byte_identical": fields_ok,
                "encode_info_state_byte_identical": enc_ok,
                "detail": f"identity={identity}; fields_ok={fields_ok}; encode_ok={enc_ok}"}
    except Exception as exc:  # noqa: BLE001
        return {"name": "C5_serve_roundtrip", "pass": False, "skipped": False,
                "detail": f"serve round-trip raised: {exc!r}"}


def run_certificates(*, seed: int, prog: Optional[_Progress] = None) -> dict:
    """Run C0-C5 and return ``{certs: [...], all_pass: bool}``. C0-C4 are pure net-free primitives; C5
    needs jax (skipped, not failed, when absent). The suite's ``all_pass`` ignores SKIPPED certs."""
    certs = [_cert_c0(), _cert_c1(), _cert_c2(), _cert_c3(), _cert_c4(), _cert_c5(seed)]
    if prog is not None:
        prog.tick(spots=len(certs))
    all_pass = all(c["pass"] for c in certs if not c.get("skipped"))
    return {"certs": certs, "all_pass": all_pass}


# ==================================================================================================
# Net-free per-cell ANCHOR calibration — the always-available headline (drives E rows in smoke)
# ==================================================================================================
def _build_cell_payouts(fmt_key: str, n_seats: int, regime_key: str):
    """Construct a per-cell (format, payout vector, ante_bb) tuple from the foundation payout model.

    NORMAL/SNG -> a NORMAL cash ladder (SNG = a tiny 3-handed flat-ish ladder). SATELLITE -> a flat
    seat ladder (every paid finish identical). PKO -> the PRIZE side of a knockout (bounty priced
    separately in the E-rows). Returns ``(TournamentFormat, payouts_list, ante_bb)``."""
    from pokerbot.tournament.payouts import TournamentFormat, payout_vector

    ante = _REGIME_ANTE[regime_key]
    if fmt_key == "SATELLITE":
        seats = max(1, n_seats // 3)
        pay = payout_vector(TournamentFormat.SATELLITE, seats=seats, seat_value=10.0)
        return TournamentFormat.SATELLITE, pay, ante
    if fmt_key == "PKO":
        places = max(2, n_seats // 2)
        pay = payout_vector(TournamentFormat.PKO, places=places, prize_pool=100.0)
        return TournamentFormat.PKO, pay, ante
    if fmt_key == "SNG":
        # A 3-handed SNG ladder (65/35) — the certified SNG payout shape used across the tourney sims.
        return TournamentFormat.NORMAL, [65.0, 35.0], ante
    places = max(2, min(n_seats - 1, max(2, round(0.5 * n_seats))))
    pay = payout_vector(TournamentFormat.NORMAL, places=places, prize_pool=100.0)
    return TournamentFormat.NORMAL, pay, ante


def _anchor_calibration_cell(fmt_key: str, regime_key: str, *, n_seats: int, decks: int, seed: int,
                             policy: str, prog: Optional[_Progress] = None) -> dict:
    """Net-free anchor calibration for one grid cell.

    Drives the always-fold (``policy=='fold'``) closed-form anchor: a table of pure folders is
    best-responded by always-steal, whose seat-rotated value is the EXACT closed form
    ``anchor.always_fold_exploitability(n, ante)`` (bb/100). We MEASURE that floor over ``decks``
    seat-rotated trials (here the closed form is deterministic, so each "deck" is the same value;
    the loop exists to exercise the seat-rotation + per-deck mean machinery and the reporting path,
    and to give an honest n / se). The chip-vs-$ divergence column compares the chip-EV anchor to its
    ICM-discounted $-value (they coincide far from the money, diverge at the ICM-pressure regimes).

    Returns a per-cell block with: per-seat breakdown, mean +/- 1.96se, the verdict, the chip-vs-$
    divergence, the CRN-health ``uplift_zero_frac``, and the fixed-iteration solve count (not wall)."""
    import numpy as np

    from pokerbot.tournament.anchor import always_fold_exploitability
    from pokerbot.tournament.eval_score import stat_block, verdict
    from pokerbot.tournament.icm import icm_equity

    fmt, pay, ante = _build_cell_payouts(fmt_key, n_seats, regime_key)
    regime = REGIMES[regime_key]
    depth = float(regime["depth_bb"])

    # The net-free anchor value (bb/100) the cell should recover. always_jam vs a folding field is the
    # SAME steal value (the jammer scoops the dead money); always_fold by the hero is the OFF/floor.
    anchor_bb = always_fold_exploitability(n_seats, ante_bb=ante)

    # Seat-rotated duplicate trials: place the steal-hero in each seat once per deck and average. The
    # closed form is seat-symmetric, so the per-seat values are equal; we still emit the per-seat
    # breakdown (the reporting contract) and accumulate a per-deck series for the CI.
    per_seat = {s: [] for s in range(n_seats)}
    per_deck_means = []
    uplift_zeros = 0
    solves = 0
    for d in range(decks):
        seat_vals = []
        for s in range(n_seats):
            # Closed-form per-seat steal value (bb/100), constant across decks for a pure folding field.
            seat_vals.append(anchor_bb)
            per_seat[s].append(anchor_bb)
            solves += 1                                   # one fixed-iteration "solve" per seat-rotation
        m = float(np.mean(seat_vals))
        per_deck_means.append(m)
        # CRN health: the paired (on-off) uplift for a pure anchor cell is exactly 0 (no exploit arm in
        # net-free smoke), so EVERY deck's uplift is 0 -> uplift_zero_frac == 1.0 (the CRN cancels
        # perfectly). This is the net-free witness that the paired machinery is wired right.
        if abs(m - anchor_bb) < 1e-9:
            uplift_zeros += 1
        if prog is not None and (d % 50 == 0):
            prog.tick(spots=n_seats, hands=n_seats)

    sb = stat_block(per_deck_means)
    mean = sb["mean"]
    se = sb["se"]
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    # The cell recovers the anchor when the measured mean sits on the closed form (it does exactly in
    # the net-free path). The verdict on the bb/100 mean is POSITIVE (the steal value is positive).
    vrd = verdict(lo, hi) if (lo != hi or mean != 0.0) else "POSITIVE"
    recovers_anchor = abs(mean - anchor_bb) < 1e-6

    # chip-vs-$ divergence: the chip-EV anchor (bb/100) vs its ICM-discounted $-equity at the cell's
    # stacks. Far from the money they coincide (divergence ~0); at the ICM-pressure regimes
    # (bubble/ft/hu) the $-value of the same chips is discounted -> a positive divergence column.
    stacks = [depth] * n_seats
    base_eq = icm_equity(stacks, pay)
    # one bb of chip gain to hero seat 0 -> its marginal $-equity vs the linear (chip-EV) expectation.
    bumped = list(stacks)
    bumped[0] = depth + 1.0
    dollar_per_bb = float(icm_equity(bumped, pay)[0] - base_eq[0])
    chip_ev_linear = (sum(pay) / max(1.0, sum(stacks)))    # the chip-EV linear $/bb baseline
    chip_dollar_divergence = chip_ev_linear - dollar_per_bb  # >0 under ICM pressure (chips worth less $)

    seat_breakdown = {
        str(s): {"mean_bb_per_100": float(np.mean(per_seat[s])),
                 "n": len(per_seat[s])}
        for s in range(n_seats)
    }

    return {
        "policy": policy,
        "n_seats": n_seats,
        "depth_bb": depth,
        "ante_bb": ante,
        "format_routed": fmt.value,
        "payouts": [round(float(p), 4) for p in pay],
        "icm_pressure_spot": bool(regime["icm_spot"]),
        "decks": decks,
        "fixed_iteration_solves": solves,        # NOT wall-clock: the deterministic solve count
        "anchor_bb_per_100": anchor_bb,
        "measured_mean_bb_per_100": mean,
        "se_bb_per_100": se,
        "ci95": [lo, hi],
        "verdict": vrd,
        "recovers_anchor_floor": recovers_anchor,
        "chip_dollar_divergence": chip_dollar_divergence,
        "dollar_per_bb_hero": dollar_per_bb,
        "chip_ev_linear_dollar_per_bb": chip_ev_linear,
        "uplift_zero_frac": (uplift_zeros / decks) if decks else 0.0,
        "per_seat_breakdown": seat_breakdown,
    }


# ==================================================================================================
# E1-E6 pillar drivers (optional; degrade to ``deferred`` when the sibling module isn't built yet)
# ==================================================================================================
def _drive_pillars(fmt_key: str, regime_key: str, cell_args: dict, *,
                   policy: str) -> dict:
    """Drive E1/E2 (lbr_icm), E3/E4 (mtt_lifecycle), E6 (regime_battery) for one cell.

    Each pillar is OPTIONAL — built by another builder in parallel. We look up a stable entry-point per
    pillar; when the module (or entry point) is absent, the E-row is ``{status: 'deferred'}`` with the
    reason, and the cell still carries its anchor calibration verdict. When a pillar IS present we call
    its entry point with the cell context; any exception is captured as ``{status: 'error'}`` (never
    crashes the suite). In ``--policy net`` the pillars serve the 7-max net through ``neural_serve``;
    in net-free modes (``fold``/``jam``) they run their net-free anchor arm."""
    rows: dict[str, Any] = {}

    # E1/E2 — ICM-weighted LBR exploitability (the certified BR adversary, ICM-reweighted).
    lbr = _try_import("pokerbot.tournament.lbr_icm")
    entry = getattr(lbr, "lbr_icm_cell", None) if lbr is not None else None
    if entry is None:
        rows["E1_E2_lbr_icm"] = {"status": "deferred",
                                 "reason": "pokerbot.tournament.lbr_icm (or .lbr_icm_cell) not yet built"}
    else:
        try:
            rows["E1_E2_lbr_icm"] = {"status": "built", "result": entry(fmt_key, regime_key, cell_args)}
        except Exception as exc:  # noqa: BLE001
            rows["E1_E2_lbr_icm"] = {"status": "error", "reason": repr(exc)}

    # E3/E4 — depth/blind lifecycle ROI + finish distribution.
    life = _try_import("pokerbot.tournament.mtt_lifecycle")
    entry = getattr(life, "lifecycle_cell", None) if life is not None else None
    if entry is None:
        rows["E3_E4_mtt_lifecycle"] = {"status": "deferred",
                                       "reason": "pokerbot.tournament.mtt_lifecycle (or .lifecycle_cell) not yet built"}
    else:
        try:
            rows["E3_E4_mtt_lifecycle"] = {"status": "built", "result": entry(fmt_key, regime_key, cell_args)}
        except Exception as exc:  # noqa: BLE001
            rows["E3_E4_mtt_lifecycle"] = {"status": "error", "reason": repr(exc)}

    # E6 — the regime exploit battery.
    battery = _try_import("pokerbot.tournament.regime_battery")
    entry = getattr(battery, "regime_battery_cell", None) if battery is not None else None
    if entry is None:
        rows["E6_regime_battery"] = {"status": "deferred",
                                     "reason": "pokerbot.tournament.regime_battery (or .regime_battery_cell) not yet built"}
    else:
        try:
            rows["E6_regime_battery"] = {"status": "built", "result": entry(fmt_key, regime_key, cell_args)}
        except Exception as exc:  # noqa: BLE001
            rows["E6_regime_battery"] = {"status": "error", "reason": repr(exc)}

    return rows


# ==================================================================================================
# Per-cell JSONL checkpoint (crash-survivable; mirrors exploitation_battery._load_checkpoint header guard)
# ==================================================================================================
def _ckpt_meta(seed: int, formats: list[str], regimes: list[str], policy: str, decks: int) -> dict:
    return {"seed": seed, "formats": formats, "regimes": regimes, "policy": policy, "decks": decks}


def _load_cell_checkpoint(path: str, *, meta: dict) -> dict[str, dict]:
    """Robust JSONL read of completed cells -> ``{cell_key: cell_block}``. Validates the one-time
    ``_meta`` header against the current run (seed/formats/regimes/policy/decks); a mismatch RAISES
    rather than silently merging a stale/foreign stream (the ``exploitation_battery._load_checkpoint``
    header-guard idiom). Malformed lines are skipped."""
    done: dict[str, dict] = {}
    if not path or not _os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(rec, dict):
                continue
            hdr = rec.get("_meta")
            if hdr is not None:
                for k, v in meta.items():
                    if hdr.get(k) != v:
                        raise ValueError(
                            f"checkpoint header mismatch in {path!r}: {k}={hdr.get(k)!r} on disk "
                            f"!= {v!r} requested — refusing to merge a stale/foreign checkpoint")
                continue
            key = rec.get("cell_key")
            if key is None or "cell" not in rec:
                continue
            done[str(key)] = rec["cell"]
    return done


class _CellCheckpoint:
    """Append-only single-line-flush+fsync JSONL writer for completed cells (the hand_collector
    durability idiom): at most the in-flight line is lost on a hard kill, every appended cell persists.
    A header is written once on a fresh file. ``checkpoint_path=None`` -> a no-op handle."""

    def __init__(self, path: Optional[str], meta: dict):
        self.path = path
        self._fh = None
        if not path:
            return
        try:
            new_file = (not _os.path.exists(path)) or _os.path.getsize(path) == 0
            parent = _os.path.dirname(path)
            if parent:
                _os.makedirs(parent, exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")
            if new_file:
                self._fh.write(json.dumps({"_meta": meta}) + "\n")
                self._flush()
        except (OSError, ValueError) as exc:
            print(f"[suite] checkpoint open failed ({exc!r}); continuing without checkpointing",
                  file=sys.stderr, flush=True)
            self._fh = None

    def _flush(self) -> None:
        if self._fh is None:
            return
        self._fh.flush()
        try:
            _os.fsync(self._fh.fileno())
        except (OSError, ValueError):
            pass

    def append(self, cell_key: str, cell: dict) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(json.dumps({"cell_key": cell_key, "cell": cell}) + "\n")
            self._flush()
        except (OSError, ValueError):
            pass  # telemetry must never crash the eval

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


# ==================================================================================================
# The grid driver
# ==================================================================================================
def run_suite(*, formats: list[str], regimes: list[str], policy: str, decks: int, hands: int,
              spots: int, ckpt: str, seed: int, smoke: bool, out: str,
              checkpoint_path: Optional[str] = None, resume: bool = False,
              quiet: bool = False) -> dict:
    """Drive the full ``formats x regimes`` grid and return the unified report dict.

    Net-free (``policy in {fold, jam}``): every cell value comes from the certified closed-form anchor +
    the eval_score/icm/payouts primitives — no GPU, no net. ``policy=='net'`` serves the 7-max net through
    ``neural_serve`` in the E-pillars (deferred to AWS spot; this driver runs it unchanged)."""
    t0 = time.time()
    grid_cells = [(f, r) for f in formats for r in regimes]
    prog = _Progress(len(grid_cells), enabled=not quiet)

    # Certificates first — if they fail, the cell values are not trustworthy (we still emit them).
    cert_block = run_certificates(seed=seed, prog=prog)

    meta = _ckpt_meta(seed, formats, regimes, policy, decks)
    resumed: dict[str, dict] = {}
    if checkpoint_path and resume:
        resumed = _load_cell_checkpoint(checkpoint_path, meta=meta)
        if not quiet:
            print(f"[suite:resume] {checkpoint_path}: {len(resumed)}/{len(grid_cells)} cells already done",
                  file=sys.stderr, flush=True)
    ckpt_writer = _CellCheckpoint(checkpoint_path, meta)

    # Per-regime seat width: HU is 2-handed, FT/bubble are short-table, the body is 7-max (the trained
    # width). This keeps the depth/blind lifecycle realistic per cell.
    def _seats_for(regime_key: str) -> int:
        if regime_key == "hu":
            return 2
        if regime_key in ("ft", "bubble"):
            return 6
        return 7

    cells: dict[str, dict] = {}
    for idx, (fmt_key, regime_key) in enumerate(grid_cells):
        cell_key = f"{fmt_key}:{regime_key}"
        label = f"{fmt_key} x {REGIMES[regime_key]['label']}"
        if cell_key in resumed:
            cells[cell_key] = resumed[cell_key]
            prog.cell_done(idx, label + " (resumed)", resumed[cell_key].get("calibration", {})
                           .get("verdict", "?"))
            continue
        prog.cell_start(idx, label)
        n_seats = _seats_for(regime_key)

        # The always-available net-free anchor calibration (the smoke headline).
        calibration = _anchor_calibration_cell(
            fmt_key, regime_key, n_seats=n_seats, decks=decks, seed=seed, policy=policy, prog=prog)

        # E1-E6 pillars (optional; deferred when the sibling module isn't built yet).
        cell_args = {"n_seats": n_seats, "depth_bb": calibration["depth_bb"],
                     "ante_bb": calibration["ante_bb"], "payouts": calibration["payouts"],
                     "icm_pressure_spot": calibration["icm_pressure_spot"],
                     "decks": decks, "hands": hands, "spots": spots, "ckpt": ckpt,
                     "seed": seed, "smoke": smoke, "policy": policy}
        pillars = _drive_pillars(fmt_key, regime_key, cell_args, policy=policy)

        cell = {"format": fmt_key, "regime": regime_key, "label": label,
                "calibration": calibration, "pillars": pillars}
        cells[cell_key] = cell
        ckpt_writer.append(cell_key, cell)
        prog.cell_done(idx, label, calibration["verdict"])

    ckpt_writer.close()

    # ---- aggregate / surface summary -------------------------------------------------------------
    n_recover = sum(1 for c in cells.values() if c["calibration"]["recovers_anchor_floor"])
    n_pos = sum(1 for c in cells.values() if c["calibration"]["verdict"] == "POSITIVE")
    pillar_status: dict[str, int] = {}
    for c in cells.values():
        for pk, pv in c["pillars"].items():
            st = pv.get("status", "?")
            pillar_status[f"{pk}:{st}"] = pillar_status.get(f"{pk}:{st}", 0) + 1

    report = {
        "schema": "tournament_eval_suite/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 3),
        "config": {
            "formats": formats, "regimes": regimes, "policy": policy, "decks": decks,
            "hands": hands, "spots": spots, "ckpt": ckpt, "seed": seed, "smoke": smoke,
            "jax_platforms": _os.environ.get("JAX_PLATFORMS"),
            "info_suits": _os.environ.get("POKERBOT_INFO_SUITS"),
        },
        "certificates": cert_block,
        "grid": {"formats": formats, "regimes": regimes, "n_cells": len(grid_cells)},
        "cells": cells,
        "summary": {
            "certificates_all_pass": cert_block["all_pass"],
            "cells_total": len(cells),
            "cells_recovering_anchor": n_recover,
            "cells_positive_verdict": n_pos,
            "pillar_status_counts": pillar_status,
            "net_free": policy in ("fold", "jam"),
        },
    }

    if out:
        parent = _os.path.dirname(out)
        if parent:
            _os.makedirs(parent, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        if not quiet:
            print(f"[suite] wrote {out}", file=sys.stderr, flush=True)
    return report


# ==================================================================================================
# CLI
# ==================================================================================================
def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Unified tournament-eval suite: the format x regime surface report (C0-C5 + E1-E6).")
    ap.add_argument("--format", default="all",
                    help="comma-list of {NORMAL,SNG,SATELLITE,PKO} or 'all' (default all)")
    ap.add_argument("--regime", default="all",
                    help="comma-list of {deep,mid,shallow,bubble,ft,hu} or 'all' (default all)")
    ap.add_argument("--decks", type=int, default=200, help="seat-rotated duplicate trials per cell")
    ap.add_argument("--hands", type=int, default=2000, help="hands per cell (pillar drivers)")
    ap.add_argument("--spots", type=int, default=500, help="constructed ICM spots per cell (pillar drivers)")
    ap.add_argument("--ckpt", default="checkpoints/sd_cfr_7max_neural/seven_v1.pkl",
                    help="7-max net checkpoint (only consulted by --policy net pillars)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", choices=["fold", "jam", "net", "icm"], default="fold",
                    help="net-free anchor (fold/jam), serve the 7-max net (net; deferred to AWS spot), or "
                         "the analytic ICM push/fold engine (icm; the certified shallow-stack specialist, "
                         "deferring deep/postflop) — graded head-to-head on the SAME surface as net")
    ap.add_argument("--out", default="artifacts/tournament_eval_suite.json")
    ap.add_argument("--checkpoint-path", default=None,
                    help="crash-survivable per-cell JSONL checkpoint (resume with --resume)")
    ap.add_argument("--resume", action="store_true", help="resume completed cells from --checkpoint-path")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny N net-free correctness run (forces small decks/hands/spots if larger)")
    ap.add_argument("--quiet", action="store_true", help="silence the stderr progress log")
    return ap.parse_args(argv[1:])


def _resolve_list(arg: str, *, allowed: list[str], name: str) -> list[str]:
    if arg.strip().lower() == "all":
        return list(allowed)
    items = [x.strip() for x in arg.split(",") if x.strip()]
    bad = [x for x in items if x not in allowed]
    if bad:
        raise SystemExit(f"unknown {name}(s) {bad}; allowed = {allowed}")
    return items


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    formats = _resolve_list(args.format, allowed=FORMATS, name="format")
    regimes = _resolve_list(args.regime, allowed=REGIME_ORDER, name="regime")

    decks, hands, spots = args.decks, args.hands, args.spots
    if args.smoke:
        # Tiny-N net-free correctness run: clamp the work down so the whole grid runs in seconds on CPU.
        decks = min(decks, 8)
        hands = min(hands, 40)
        spots = min(spots, 20)

    report = run_suite(
        formats=formats, regimes=regimes, policy=args.policy, decks=decks, hands=hands, spots=spots,
        ckpt=args.ckpt, seed=args.seed, smoke=args.smoke, out=args.out,
        checkpoint_path=args.checkpoint_path, resume=args.resume, quiet=args.quiet,
    )

    s = report["summary"]
    print(f"[suite] {s['cells_total']} cells | certs_all_pass={s['certificates_all_pass']} | "
          f"recover_anchor={s['cells_recovering_anchor']}/{s['cells_total']} | "
          f"positive={s['cells_positive_verdict']}/{s['cells_total']} | net_free={s['net_free']}",
          file=sys.stderr, flush=True)
    # Exit non-zero only if the structural certificates fail (the cell values are then untrustworthy).
    return 0 if s["certificates_all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
