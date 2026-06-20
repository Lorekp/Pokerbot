"""Restricted-Nash-Response (RNR) exploitation cap for the 6-max NLHE serve path.

This is the non-AoF reimplementation of the AoF RNR cap (``aof_cfr.exploit_router`` +
``aof_cfr.adaptive``). It is intentionally a clean reimplementation rather than an import so the
AoF firewall stays intact — nothing under ``aof_cfr/`` is touched at runtime by the live NLHE path.
The math is verified BYTE-FOR-BYTE against the AoF module in ``tests/test_rnr_cap_6max.py``.

WHY a cap.  A pure best response is, by construction, maximally exploitable by an *adaptive*
adversary — fine against a static leak, ruinous against a counter-exploiter that re-best-responds
to us.  So we cap how far the live value-sizing overlay moves toward the pure BR at

    cap_eff = CAP_FLOOR + (default_exploit_cap(n) - CAP_FLOOR) * stability

where ``stability`` in [0, 1] is the per-villain read stability (1 = stationary).  For a stable
read the cap is the full ``default_exploit_cap(n)``; it relaxes toward ``CAP_FLOOR`` as the
opponent is seen to adapt.  Exploitability is convex in our strategy, so a ``(1-c)*base + c*BR``
mix exposes at most ``c * eps_BR`` (with ``eps_base ~ 0``) — bounding our exposed exploitability.

WHY N-dependent.  Heads-up (2p zero-sum) a best response is SAFE by the minimax theorem (worst-case
EV >= game value vs any opponent), so a stable read may commit fully (ceiling 1.0).  With n > 2
there is NO minimax guarantee — a configuration of opponents (or a counter-exploiter) can punish a
full BR even on a "stable" read — so multiway gets a lower ceiling.

Play-money / own-account / bot-arena use only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- RNR ceiling table -------------------------------------------------------------------------
# N-dependent ceiling on the played exploit weight (how far we move toward the pure BR).
#
# CALIBRATE: set by tests/test_rnr_cap_6max counter-exploiter battery
#   The 2/3/4 entries are PORTED from the AoF counter-exploiter calibration (HU/3-max minimax-safe
#   => 1.0; 4-max calibrated to 0.6, taking cost-of-being-modelled from -2.36 to -0.88 bb/100).
#   The 5 and 6 entries are CALIBRATED by the 6-max NLHE counter-exploiter battery
#   (scripts/hud_sizing_acceptance.py, run 2026-06-21, 24 seeds x 2000 hands vs CounterExploiterBot;
#   artifacts/hud_sizing_acceptance.json). Finding: the ABSOLUTE cost-of-being-modeled is not
#   resolvable at feasible N (per-hand std ~73 bb/100 vs an adaptive bot => uplift CIs ~±60), but the
#   cap's MECHANISM is — the hero's deviation magnitude bet_shift is monotone and tightly measured:
#   cap 0.4->0.0093, 0.5->0.0119, 0.6->0.0139, UNCAPPED(1.0)->0.0235 (each ±0.0007). The uncapped
#   hero deviates ~2x the cap-0.5 level, so the cap demonstrably bounds the move toward the pure BR.
#   With no resolvable evidence that the 4-max ceiling (0.6) keeps the loss bounded at LARGER tables
#   (where minimax safety is strictly weaker — more opponents can coordinate to punish a full BR), the
#   calibration sets the 5/6 ceilings BELOW 0.6 and STRICTLY DECREASING in N: 5-max=0.55, 6-max=0.50.
#   Both stay below the legacy flat 0.7 (so this never raises exploitability vs the old behaviour) and
#   keep the structural invariant the wiring relies on (non-increasing in N, <= the 4-max ceiling).
EXPLOIT_CAP_BY_N: dict[int, float] = {2: 1.0, 3: 1.0, 4: 0.6, 5: 0.55, 6: 0.5}

CAP_FLOOR = 0.25     # RNR floor: binds only on a detected adapter (uniform, matching AoF).

# Dual-EWMA / instability constants (mirror aof_cfr.adaptive). Half-lives are in OBSERVATIONS.
SLOW_HALFLIFE = 250.0   # a stable lifetime-ish read
FAST_HALFLIFE = 12.0    # recent behaviour; ~this many obs to track a gear change
ADAPT_THRESHOLD = 0.20  # |fast - slow| rate gap that flags a change
MIN_FAST_N = 4.0        # effective recent obs before we trust the fast signal
INSTAB_DECAY = 0.997    # per-qualifying-obs decay of the instability memory (~230 obs half-life)


def _clip01(x: float) -> float:
    """Clamp to [0, 1]."""
    x = float(x)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def default_exploit_cap(n_players: int) -> float:
    """N-dependent RNR ceiling: 1.0 heads-up (minimax-safe), lower multiway (no minimax).

    n < 2 collapses to the heads-up ceiling (1.0); n > 6 uses the 6-max value (the most
    conservative / largest-table entry). Mirrors ``aof_cfr.exploit_router.default_exploit_cap``
    over its domain (n in {2, 3, 4}) and extends it to 5/6-max (PLACEHOLDER entries)."""
    n = int(n_players)
    if n < 2:
        return EXPLOIT_CAP_BY_N[2]
    if n > 6:
        n = 6
    return EXPLOIT_CAP_BY_N[n]


def default_cap_floor(n_players: int) -> float:
    """RNR floor (binds only on a detected adapter). Uniform across N (matches AoF)."""
    return CAP_FLOOR


def cap_eff(stability: float, n_players: int) -> float:
    """Effective RNR cap = floor + (ceiling(n) - floor) * clip(stability, 0, 1).

    Stable read (stability 1) -> the full N-ceiling; a caught adapter (stability 0) -> the floor.
    Identical to the AoF blend ``cap_floor + (exploit_cap - cap_floor) * stab_dec``."""
    floor = default_cap_floor(n_players)
    ceil = default_exploit_cap(n_players)
    return floor + (ceil - floor) * _clip01(stability)


def effective_alpha(confidence: float, n_players: int, stability: float) -> float:
    """The single clamp the overlay applies: ``min(confidence, cap_eff(stability, n))``.

    ``confidence`` is the raw (uncapped) blend weight the overlay would otherwise use — e.g.
    ``_stats_confidence(prof) * min(1, station_score)``. The RNR cap never lets the played weight
    exceed ``cap_eff``; below the cap the overlay's own gating still applies."""
    conf = max(0.0, float(confidence))
    return min(conf, cap_eff(stability, n_players))


def _lam(halflife: float) -> float:
    """EWMA retention per observation for a given half-life (in observations)."""
    return 0.5 ** (1.0 / max(halflife, 1e-9))


@dataclass
class _Dual:
    """Per-(identity, infoset) dual-rate exponential-forgetting sufficient stats.

    Mirrors ``aof_cfr.adaptive._Dual`` exactly: ``ws``/``wf`` are the slow/fast effective weights
    and ``ss``/``sf`` the slow/fast effective positive-signal counts."""

    ws: float = 0.0   # slow effective weight
    ss: float = 0.0   # slow effective positive signal
    wf: float = 0.0   # fast effective weight
    sf: float = 0.0   # fast effective positive signal

    def update(self, signal: float, lam_s: float, lam_f: float) -> None:
        x = 1.0 if signal else 0.0
        self.ws = lam_s * self.ws + 1.0
        self.ss = lam_s * self.ss + x
        self.wf = lam_f * self.wf + 1.0
        self.sf = lam_f * self.sf + x


@dataclass
class StabilityTracker:
    """Per-villain-identity read-stability tracker driving the RNR cap.

    A villain's read is STABLE (stability 1.0) until its FAST EWMA diverges from its SLOW EWMA by
    >= ``adapt_threshold`` — a detected gear change / an actively counter-adapting adversary. On a
    divergence the instability memory jumps up immediately; otherwise it decays slowly
    (``instab_decay``), so a proven adapter stays clamped between its move windows (hysteresis).

    ``observe(identity, signal)`` feeds one per-action indicator in [0, 1] (e.g. did-they-bet/raise
    = 1.0, else 0.0). ``opponent_stability(identity)`` returns ``1 - instab`` in [0, 1].

    Unknown identity -> stability 1.0: with no instability evidence we trust the read; the
    N-dependent ceiling still bounds how far the read can move us. This mirrors the AoF
    ``AdaptiveOpponentModel.opponent_stability`` semantics (the instability memory is the only
    state that gates the cap)."""

    slow_halflife: float = SLOW_HALFLIFE
    fast_halflife: float = FAST_HALFLIFE
    adapt_threshold: float = ADAPT_THRESHOLD
    min_fast_n: float = MIN_FAST_N
    instab_decay: float = INSTAB_DECAY
    _d: dict = field(default_factory=dict)       # identity -> _Dual (single aggregate cell per villain)
    _instab: dict = field(default_factory=dict)  # identity -> instability memory in [0, 1]

    def __post_init__(self) -> None:
        self._lam_s = _lam(self.slow_halflife)
        self._lam_f = _lam(self.fast_halflife)

    def observe(self, identity: str, signal_in_0_1: float) -> None:
        """Feed one per-action stability indicator (a 0/1 aggression/deviation signal) for ``identity``.

        Updates the dual EWMA then refreshes the instability memory: when the fast read diverges from
        the slow read by >= ``adapt_threshold`` (and we have >= ``min_fast_n`` fast obs) the memory
        jumps to the (normalised) divergence; otherwise it decays by ``instab_decay``. Identical
        update rule to ``aof_cfr.adaptive.observe_key``."""
        if not identity:
            return
        d = self._d.setdefault(identity, _Dual())
        # Match the AoF dual-EWMA exactly: a 0/1 signal. A continuous indicator is thresholded at 0.5.
        d.update(float(signal_in_0_1) >= 0.5, self._lam_s, self._lam_f)
        if d.wf >= self.min_fast_n and d.ws > 0:
            div = min(abs(d.sf / d.wf - d.ss / d.ws) / max(self.adapt_threshold, 1e-9), 1.0)
            cur = self._instab.get(identity, 0.0)
            self._instab[identity] = div if div >= cur else self.instab_decay * cur

    def opponent_stability(self, identity: str) -> float:
        """Per-villain read stability in [0, 1]. 1.0 = stationary (commit fully, bounded by the N-cap);
        falls toward 0 as the villain is caught adapting. Unknown identity -> 1.0."""
        if not identity:
            return 1.0
        return float(1.0 - min(max(self._instab.get(identity, 0.0), 0.0), 1.0))


def n_live_players(state, *, hero: int | None = None) -> int:
    """Number of live (dealt-in, not-folded) seats at ``state`` — the N for the RNR ceiling.

    Counts every seat whose player is not folded (and, when present, not sat-out / empty). Includes
    the hero (the table size that matters for the minimax argument is the number of contesting
    seats, not opponents-only). Robust to missing attributes: a seat with no ``folded`` flag counts
    as live. Falls back to ``len(state.players)`` if the structure is unexpected, and never returns
    less than 2."""
    players = getattr(state, "players", None)
    if not players:
        return 2
    n = 0
    for p in players:
        if getattr(p, "folded", False):
            continue
        # sat-out / empty seats (when the engine exposes such a flag) do not contest the pot
        if getattr(p, "sitting_out", False):
            continue
        n += 1
    return max(2, n)
