"""HUD -> segment classifier + consistency filter + count gate.

Net-new module implementing the router guardrail from
``artifacts/exploit_research/FINAL_exploitation_research.md`` (SS3 exploit
playbook + SS4.1.2 rewire plan).

Design constraints (non-negotiable, per the report):
  * Pure functions, stdlib + ``math`` only. NO engine imports so this stays
    importable standalone (it feeds the arena overlay but must not drag the
    solver / Rust in).
  * Precision-first routing: on any doubt we fall back to ``'gto'`` (the
    two-sided disengage default, Lever 4). We NEVER route an aggressive bot
    into the value branch.
  * ``hud.bluff`` is a RETIRED degenerate stat (uncontested-win share) and
    ``fold_to_cbet`` has no data -- neither is ever read here.

The segments map to the levers:
  * ``'station'``    -> Lever 1 (thin/merged value, count-gated).
  * ``'overfolder'`` -> Lever 2 (steal-widen vs rock/nit).
  * ``'aggressor'``  -> Lever 3 territory (dormant hypothesis; the overlay
                        currently treats it as GTO, but the label is useful
                        loss-avoidance signal / future tail work).
  * ``'gto'``        -> Lever 4 (disengage, two-sided GTO default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["SegmentResult", "classify", "count_gate"]

# --- thresholds (report SS3 GUARDRAIL branch order) ---------------------------
MIN_SAMPLE = 500          # server-truth n below this is untrustworthy
MAX_AF = 20.0             # af above this is a HUD-computation artifact
STATION_VPIP = 0.35       # verified-station: af==0 & vpip > this
OVERFOLD_VPIP = 0.20      # rock/nit: vpip < this ...
OVERFOLD_PFR = 0.08       # ... and pfr < this
AGGRO_AF = 2.0            # aggressor: af >= this ...
AGGRO_VPIP = 0.30         # ... and vpip >= this
COUNT_GATE_MAX = 3        # Lever-1 hard gate: <= this many live villains

# Server ``playingStyle`` archetypes that mean "this bot is aggressive".
# Used ONLY as a cross-check to veto a dangerous value-branch misroute.
AGGRESSIVE_ARCHETYPES = frozenset({"maniac", "shark", "tag"})


@dataclass
class SegmentResult:
    """Outcome of a single HUD classification.

    Attributes:
        segment: one of ``{'station', 'overfolder', 'aggressor', 'gto'}``.
        trusted: ``True`` only when the read is trustworthy AND internally
            consistent. Consistency-filter fallbacks and cross-check demotions
            are ``False`` (data was missing / thin / contradictory). A
            deliberate ``'gto'`` route on good data (the disengage default) is
            ``True``.
        reasons: human-readable trace of why this segment was assigned.
    """

    segment: str
    trusted: bool
    reasons: list[str] = field(default_factory=list)


def _pct(f: Any) -> Optional[float]:
    """Normalise a rate stat to a 0-1 fraction.

    Auto-detects percent (0-100) vs fraction (0-1) encodings: a value > 1 is
    assumed to be a percentage and divided by 100. Returns ``None`` for
    missing / non-numeric input. NOTE: only for *rate* stats (vpip, pfr,
    threeBet, wsd, wtsd) -- never call this on ``af`` (an aggression factor
    that is legitimately > 1).
    """
    if f is None:
        return None
    try:
        v = float(f)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v / 100.0 if v > 1.0 else v


def _num(f: Any) -> Optional[float]:
    """Coerce to float without percent scaling (for af, sample size)."""
    if f is None:
        return None
    try:
        v = float(f)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def _is_aggressive_label(hud: dict) -> bool:
    """Does the server ``playingStyle`` block flag this bot as aggressive?"""
    ps = hud.get("playingStyle")
    if not isinstance(ps, dict):
        return False
    if ps.get("aggression") == "aggressive":
        return True
    return ps.get("archetype") in AGGRESSIVE_ARCHETYPES


def classify(hud: Optional[dict], local: Optional[dict] = None) -> SegmentResult:
    """Classify a bot into an exploit segment from its HUD block.

    Branch order is EXACTLY (report SS3 GUARDRAIL):
      1. Consistency filter -> ``'gto'`` (trusted=False) when the read is
         missing / thin / internally impossible.
      2. Verified station (af == 0 and vpip > 0.35) -> ``'station'``.
      3. Rock / nit (vpip < 0.20 and pfr < 0.08) -> ``'overfolder'``.
      4. Else ``'aggressor'`` if (af >= 2 and vpip >= 0.30) else ``'gto'``.

    Then a cross-check: if we assigned the value branch (``'station'``) but the
    server ``playingStyle`` says the bot is aggressive, demote to ``'gto'``
    (never bet-for-value into an aggressive range).

    Args:
        hud: the server HUD dict (``vpip``, ``pfr``, ``threeBet``, ``af``,
            ``wsd``, ``wtsd``, ``playingStyle``, ``_sampleSize`` ...). May be
            ``None``.
        local: reserved for future local-divergence trust decay; NOT used for
            the segment decision (segmentation is HUD-driven per the report).

    Returns:
        SegmentResult.
    """
    # --- Branch 1: consistency filter ------------------------------------
    if not isinstance(hud, dict) or not hud:
        return SegmentResult("gto", False, ["hud_missing"])

    af = _num(hud.get("af"))
    vpip = _pct(hud.get("vpip"))
    pfr = _pct(hud.get("pfr"))
    three_bet = _pct(hud.get("threeBet"))
    n = _num(hud.get("_sampleSize"))

    reasons: list[str] = []
    if af is None:
        reasons.append("no_af")
    if n is None or n < MIN_SAMPLE:
        reasons.append("sample_lt_%d" % MIN_SAMPLE)
    if vpip is not None and vpip == 0.0:
        reasons.append("idle_vpip_0")
    if three_bet is not None and pfr is not None and three_bet > pfr:
        reasons.append("impossible_3bet_gt_pfr")
    if af is not None and af > MAX_AF:
        reasons.append("af_gt_%g" % MAX_AF)
    if reasons:
        return SegmentResult("gto", False, reasons)

    # Past the filter: af, vpip, pfr are trustworthy numbers.
    # (af/vpip/pfr are guaranteed non-None here; three_bet may be None.)

    # --- Branch 2: verified station --------------------------------------
    if af == 0.0 and vpip is not None and vpip > STATION_VPIP:
        result = SegmentResult("station", True, ["af_0", "vpip_gt_%.2f" % STATION_VPIP])
        # Cross-check: precision-first veto of the value branch.
        if _is_aggressive_label(hud):
            ps = hud.get("playingStyle") or {}
            tag = ps.get("archetype") or ps.get("aggression") or "aggressive"
            return SegmentResult(
                "gto",
                False,
                result.reasons + ["demote_station_label_aggressive:%s" % tag],
            )
        return result

    # --- Branch 3: rock / nit over-folder --------------------------------
    if vpip is not None and vpip < OVERFOLD_VPIP and pfr is not None and pfr < OVERFOLD_PFR:
        return SegmentResult(
            "overfolder",
            True,
            ["vpip_lt_%.2f" % OVERFOLD_VPIP, "pfr_lt_%.2f" % OVERFOLD_PFR],
        )

    # --- Branch 4: aggressor else GTO default ----------------------------
    if af is not None and af >= AGGRO_AF and vpip is not None and vpip >= AGGRO_VPIP:
        return SegmentResult(
            "aggressor",
            True,
            ["af_ge_%g" % AGGRO_AF, "vpip_ge_%.2f" % AGGRO_VPIP],
        )

    return SegmentResult("gto", True, ["disengage_default"])


def count_gate(n_live_villains: int) -> bool:
    """Lever-1 hard count gate.

    Thin/merged value survives only heads-up / short-handed: the identical
    intent measured -22.42 bb/100 five-way (report G2). Returns ``True`` only
    when ``n_live_villains <= 3``.
    """
    try:
        return int(n_live_villains) <= COUNT_GATE_MAX
    except (TypeError, ValueError):
        return False
