"""Tournament payout model -- the structure layer that feeds :mod:`pokerbot.tournament.icm`.

This module turns the three primitives of a CoinPoker NLH tournament

    format      {NORMAL, PKO, SATELLITE}
    entrants    (int)        -- the running entrant field
    buy_in      (currency)   -- per-entry buy-in

into the two things the $-EV layer needs:

    * a per-rank *structure* (length-``places_paid`` fractions, best-first, summing to 1.0), and
    * a per-rank *currency* vector (``prize_pool * structure``) ready to hand to
      :func:`pokerbot.tournament.icm.icm_equity`.

The model is reconstructed from ``tournament_payout.txt`` (the human spec, Sections 1-7) and the
machine-readable curves in ``artifacts/tourney_payout_model.json`` (loaded lazily, cached
module-level -- never the whole 5700-line file into memory at import unless first used).

KEY FACTS (proven in the spec, see ``tournament_payout.txt`` Section 0):
  * The per-place PERCENTAGE structure is a fixed function of ``places_paid`` ONLY -- pool size is
    irrelevant. So a single lookup keyed on ``places_paid`` reproduces every captured ladder.
  * ``places_paid`` is ~a fixed fraction of the field: NORMAL ``0.175``, PKO ``0.133`` (PKO pays
    fewer places because half the money is diverted to bounties).
  * NORMAL: ~92% of each buy-in -> prize pool (8% rake). PKO: the non-rake half is split 50/50
    prize/bounty. SATELLITE: prizes are SEATS, not a cash ladder.

DESIGN CHOICES (deliberate, documented):
  * ``places_paid`` uses the CLEAN fractional model ``clamp(round_half_up(frac*entrants), ...)``.
    We do NOT fit the noisy observed (entrants -> places) points; ``entrants`` in those captures is
    the *running* count, not final, so two of them are outliers we deliberately ignore:
    NORMAL ``97 -> 9`` and ``145 -> 11`` (both fall well below the fractional line). The clean model
    is the spec; it is monotone non-decreasing in entrants by construction.
  * Round-half-up everywhere via ``int(x + 0.5)`` -- NEVER Python's banker's ``round``.
  * Captured ``places_paid`` -> exact json fractions, renormalized to sum 1.0; near-equal adjacent
    ranks (the displayed "clubbed" tiers, e.g. PKO ties 1st/2nd) are snapped to a common value so
    the tie is exact rather than a rounding artifact.
  * Uncaptured sizes -> monotone relative-position interpolation between the two bracketing captured
    curves, then cummin (enforce non-increasing) + renormalize.
"""

from __future__ import annotations

import enum
import json
import os
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------------------------------
# Format
# --------------------------------------------------------------------------------------------------


class TournamentFormat(enum.Enum):
    """The three payout regimes a CoinPoker NLH tournament can take."""

    NORMAL = "normal"      # Freezeout / Rebuy -- a single cash ladder
    PKO = "pko"            # Progressive Knockout -- prize ladder + separate bounty pool
    SATELLITE = "satellite"  # pays SEATS/TICKETS, not a cash ladder


# Lobby ``tournamentType`` string -> format. Note ``MTT_SATTELITE`` is the *real* (misspelled)
# lobby value -- see ``tournament_payout.txt`` Section 6.
_TYPE_TO_FORMAT = {
    "MTT_FREEZEOUT": TournamentFormat.NORMAL,
    "MTT_REBUY": TournamentFormat.NORMAL,
    "MTT_PROGRESSIVE_KNOCK_OUT": TournamentFormat.PKO,
    "MTT_SATTELITE": TournamentFormat.SATELLITE,
}


def classify(
    *,
    tournament_type: Optional[str] = None,
    is_satellite: bool = False,
    bounty: bool = False,
) -> TournamentFormat:
    """Tag a tournament's payout format from its lobby fields.

    Precedence (most specific first):
      1. ``is_satellite`` forces SATELLITE (a satellite is never a cash ladder, even with a bounty).
      2. ``bounty`` forces PKO (unless satellite).
      3. ``tournament_type`` lookup (``MTT_SATTELITE`` -- the real misspelled value -> SATELLITE,
         ``MTT_PROGRESSIVE_KNOCK_OUT`` -> PKO, ``MTT_FREEZEOUT``/``MTT_REBUY`` -> NORMAL).
      4. default NORMAL.

    A satellite type/flag always wins over a bounty flag.
    """
    fmt = _TYPE_TO_FORMAT.get(tournament_type) if tournament_type is not None else None

    if is_satellite or fmt is TournamentFormat.SATELLITE:
        return TournamentFormat.SATELLITE
    if bounty:
        return TournamentFormat.PKO
    if fmt is not None:
        return fmt
    return TournamentFormat.NORMAL


# --------------------------------------------------------------------------------------------------
# places_paid
# --------------------------------------------------------------------------------------------------


def _round_half_up(x: float) -> int:
    """Round-half-up to the nearest int (``int(x + 0.5)`` for ``x >= 0``). NEVER banker's round.

    All our inputs (``frac * entrants``) are non-negative, so the simple ``int(x + 0.5)`` form is
    exact and monotone non-decreasing in ``x``.
    """
    return int(x + 0.5)


# Default ITM field-fractions (``tournament_payout.txt`` Section 2).
_DEFAULT_FRAC = {
    TournamentFormat.NORMAL: 0.175,
    TournamentFormat.PKO: 0.133,
}


def places_paid(
    fmt: TournamentFormat,
    entrants: int,
    *,
    frac: Optional[float] = None,
    min_paid: int = 2,
    seats: Optional[int] = None,
) -> int:
    """How many finishers get paid, as a clean fractional step of the entrant field.

    NORMAL/PKO::

        places = clamp(max(min_paid, round_half_up(frac * entrants)), 1, entrants)

    with ``frac`` defaulting to ``0.175`` (NORMAL) / ``0.133`` (PKO); override via ``frac``. The
    result is monotone non-decreasing in ``entrants`` by construction (``round_half_up`` is monotone,
    ``max``/``clamp`` preserve it) and the noisy observed outliers (NORMAL ``97 -> 9``, ``145 -> 11``)
    are deliberately ignored -- the captures use the *running* entrant count, not the final field.

    SATELLITE: there is no fractional ladder -- ``places_paid`` is the guaranteed ``seats`` count.
    Returns ``min(seats, entrants)`` when ``seats`` is given (you cannot pay more seats than
    entrants); falls back to the NORMAL fractional model if ``seats`` is omitted.
    """
    entrants = int(entrants)
    if entrants <= 0:
        return 0

    if fmt is TournamentFormat.SATELLITE and seats is not None:
        return min(int(seats), entrants)

    f = frac if frac is not None else _DEFAULT_FRAC.get(fmt, 0.175)
    raw = _round_half_up(f * entrants)
    places = max(min_paid, raw)
    # clamp into [1, entrants]
    if places > entrants:
        places = entrants
    if places < 1:
        places = 1
    return int(places)


# --------------------------------------------------------------------------------------------------
# Captured curve loading (lazy, cached module-level)
# --------------------------------------------------------------------------------------------------

# Locate artifacts/tourney_payout_model.json relative to the repo root (this file lives at
# pokerbot/tournament/payouts.py -> repo root is two parents up).
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "artifacts",
    "tourney_payout_model.json",
)

# Snap tolerance (percent units) for consolidating the displayed "clubbed" tiers into exact ties.
# Empirically the smallest gap between two *genuinely distinct* captured ranks is 0.03 pct (a tail
# tier boundary), while the top tied tiers (PKO 1st/2nd) differ by at most 0.017 pct, so 0.02 cleanly
# separates the two. Snapping shifts any rank by at most ~0.0085 pct (8.5e-5 frac) -- far under the
# 1e-3 per-rank reproduction tolerance.
_TIE_SNAP_PCT = 0.02


@lru_cache(maxsize=1)
def _model() -> dict:
    """Load + cache the captured payout-curve JSON once, module-level."""
    with open(_MODEL_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _captured_sizes(fmt: TournamentFormat) -> list[int]:
    """Sorted list of captured ``places_paid`` sizes for ``fmt`` (NORMAL or PKO)."""
    curve = _model()[fmt.value]["payout_curve"]
    return sorted(int(k) for k in curve.keys())


def _raw_pct(fmt: TournamentFormat, size: int) -> list[float]:
    """The captured per-rank ``mean_pct`` list (rank 1..size, best first) for a captured ``size``."""
    curve = _model()[fmt.value]["payout_curve"][str(size)]
    sbr = sorted(curve["share_by_rank"], key=lambda e: e["rank"])
    return [float(e["mean_pct"]) for e in sbr]


def _snap_ties(pct: Sequence[float]) -> list[float]:
    """Consolidate near-equal adjacent ranks (clubbed tiers) into a common value.

    Walks adjacent ranks; ranks within ``_TIE_SNAP_PCT`` of the running tier head are pulled into one
    tier and all set to the tier's mean. This makes the displayed ties (e.g. PKO 1st == 2nd) exact
    instead of off-by-a-rounding-digit, without touching genuinely distinct ranks.
    """
    p = [float(x) for x in pct]
    n = len(p)
    i = 0
    while i < n:
        j = i + 1
        # extend the tier while each next rank is within tolerance of the tier's *first* member
        while j < n and abs(p[j] - p[i]) < _TIE_SNAP_PCT:
            j += 1
        if j - i > 1:
            avg = sum(p[i:j]) / (j - i)
            for k in range(i, j):
                p[k] = avg
        i = j
    return p


def _renorm(vals: Sequence[float]) -> list[float]:
    """Scale to sum exactly 1.0 (defensive against a zero/empty sum)."""
    v = [float(x) for x in vals]
    s = sum(v)
    if s <= 0.0:
        n = len(v)
        return [1.0 / n] * n if n else []
    return [x / s for x in v]


def _cummin(vals: Sequence[float]) -> list[float]:
    """Enforce non-increasing from the top (running minimum)."""
    out = []
    cur = float("inf")
    for x in vals:
        cur = min(cur, float(x))
        out.append(cur)
    return out


def _captured_structure(fmt: TournamentFormat, size: int) -> list[float]:
    """Length-``size`` fractions for a CAPTURED size: snap ties -> renormalize to sum 1.0."""
    pct = _raw_pct(fmt, size)
    snapped = _snap_ties(pct)
    return _renorm(snapped)


def _sample_curve_at(struct: Sequence[float], rel: float) -> float:
    """Linearly sample a fraction-curve at relative position ``rel`` in ``[0, 1]``.

    ``rel = (rank - 1) / (places - 1)`` maps a target rank to a position along the source curve,
    whose ranks span ``[0, 1]`` as ``(r - 1) / (len - 1)``. Linear interpolation between the two
    surrounding source ranks.
    """
    m = len(struct)
    if m == 1:
        return float(struct[0])
    pos = rel * (m - 1)
    lo = int(pos)
    if lo >= m - 1:
        return float(struct[-1])
    frac = pos - lo
    return float(struct[lo]) * (1.0 - frac) + float(struct[lo + 1]) * frac


# --------------------------------------------------------------------------------------------------
# payout_structure
# --------------------------------------------------------------------------------------------------


def payout_structure(fmt: TournamentFormat, places: int) -> list[float]:
    """Per-rank fractions of the prize pool (best-first), length ``places``, sum == 1.0.

    Properties (all enforced): length ``places``, ``sum == 1.0`` (+-1e-9), non-increasing.

    * Captured ``places`` -> the json fractions (``mean_pct / 100``), tie-snapped + renormalized.
    * Uncaptured ``places`` -> MONOTONIC interpolation between the nearest bracketing captured curves
      ``lo < places < hi``: each target rank ``r`` maps to relative position ``(r-1)/(places-1)``,
      both bracketing curves are sampled there (linear across their ranks), blended by
      ``(places-lo)/(hi-lo)``; then cummin (non-increasing) + renormalize to sum 1.0.
    * Out of the captured range -> nearest captured curve, resampled to ``places`` + renormalized.
    """
    places = int(places)
    if places <= 0:
        return []
    if places == 1:
        return [1.0]

    sizes = _captured_sizes(fmt)
    if places in sizes:
        return _captured_structure(fmt, places)

    lo = max((s for s in sizes if s < places), default=None)
    hi = min((s for s in sizes if s > places), default=None)

    if lo is None or hi is None:
        # out of captured range: resample the nearest captured curve to `places`
        nearest = lo if lo is not None else hi
        src = _captured_structure(fmt, nearest)
        vals = [
            _sample_curve_at(src, (r) / (places - 1)) for r in range(places)
        ]
        return _renorm(_cummin(vals))

    lo_struct = _captured_structure(fmt, lo)
    hi_struct = _captured_structure(fmt, hi)
    w = (places - lo) / (hi - lo)  # blend weight toward the high curve

    vals = []
    for r in range(places):
        rel = r / (places - 1)
        a = _sample_curve_at(lo_struct, rel)
        b = _sample_curve_at(hi_struct, rel)
        vals.append(a * (1.0 - w) + b * w)

    return _renorm(_cummin(vals))


# --------------------------------------------------------------------------------------------------
# Buy-in economics
# --------------------------------------------------------------------------------------------------


def buyin_split(fmt: TournamentFormat, buy_in: float, *, rake: float = 0.08) -> dict:
    """Split one entry's buy-in into prize and bounty contributions.

    Returns ``{"prize_contrib": ..., "bounty_contrib": ...}``.

    * NORMAL: the whole non-rake amount -> prize (no bounty).
    * PKO: the non-rake amount is split 50/50 prize/bounty
      (``tournament_payout.txt`` Section 1 -- matches CoinPoker's published 50/50 rule).
    * SATELLITE: prize only (bounty 0). Satellites typically rake less; the caller may pass
      ``rake=0.06`` (the captured median), but we do not override the caller's value here.
    """
    buy_in = float(buy_in)
    net = buy_in * (1.0 - float(rake))
    if fmt is TournamentFormat.PKO:
        half = net * 0.5
        return {"prize_contrib": half, "bounty_contrib": half}
    return {"prize_contrib": net, "bounty_contrib": 0.0}


def bounty_pool(
    fmt: TournamentFormat,
    entrants: int,
    buy_in: float,
    *,
    rake: float = 0.08,
    reentries: int = 0,
) -> float:
    """Total bounty pool = per-entry bounty contribution * (entrants + reentries). 0 unless PKO."""
    if fmt is not TournamentFormat.PKO:
        return 0.0
    n = int(entrants) + int(reentries)
    return buyin_split(fmt, buy_in, rake=rake)["bounty_contrib"] * n


# --------------------------------------------------------------------------------------------------
# Overlay detection (ECON-27) — the parimutuel signal the live coach was missing
# --------------------------------------------------------------------------------------------------


def overlay_fraction(
    collected: float,
    guarantee: float,
    *,
    rake: float = 0.08,
) -> float:
    """Overlay ratio of a tournament's PRIZE pool: ``(guarantee - collected*(1-rake)) / guarantee``.

    A tournament with a guarantee > the collected-after-rake pool is **overlaid** by the operator —
    they are paying out more than the field generated. The signal is the fraction of the pool that
    comes from the operator's pocket, e.g.:

      * ``overlay_fraction(500, 5_000)  = 0.90`` -> a $500-collected $5k-guaranteed satellite is 90%
        overlaid. Effectively free EV at the hero's level (the prize pool is 10x what the field funded).
      * ``overlay_fraction(5_000, 5_000)  = 0.00`` -> exactly covered, no overlay.
      * ``overlay_fraction(10_000, 5_000) = -1.00`` -> UNDERLAY: the field generated 2x the guarantee
        (common in late-registering fields where the operator caps the prize at the guarantee).

    A negative value means the field over-funded the guarantee; a positive value means the operator
    is paying overlay. The function takes the rake-discounted COLLECTED pool, so the caller should
    pass ``collected * (1 - rake)`` if the lobby reports gross collected.

    ECON-27: this was the missing design gap in the live coach's tournament advisory — without this
    helper, a 10x-overlay satellite read identically to a no-overlay satellite, so a hero who would
    enter the no-overlay one was never told about the free-EV overlay one.
    """
    collected = float(collected)
    guarantee = float(guarantee)
    if guarantee <= 0.0:
        # No guarantee posted: there is no overlay by definition (the operator isn't promising a
        # floor). Return 0.0 (neutral) so the caller can treat "no guarantee" as the baseline.
        return 0.0
    if collected < 0.0:
        raise ValueError(
            f"overlay_fraction: collected={collected} is negative (caller bug)"
        )
    net_collected = collected * (1.0 - float(rake))
    return (guarantee - net_collected) / guarantee


def overlay_signal(
    collected: float,
    guarantee: float,
    *,
    rake: float = 0.08,
) -> dict:
    """Operator-overlay signal for the live tournament advisory: returns the overlay fraction and
    qualitative tag (overlay / underlay / neutral) so the live coach can surface a JOIN/SKIP signal
    on tournaments with a large operator overlay.

    Fields:
      * ``overlay_frac``: ``(guarantee - collected*(1-rake)) / guarantee`` (negative = underlay)
      * ``tag``: ``"overlay" | "underlay" | "neutral"``
      * ``join_recommended``: True iff overlay_frac >= 0.50 (operator is paying >= 50% of the prize pool).
    """
    frac = overlay_fraction(collected, guarantee, rake=rake)
    if frac >= 0.50:
        tag = "overlay"
        join = True
    elif frac <= -0.10:
        tag = "underlay"
        join = False
    else:
        tag = "neutral"
        join = False
    return {
        "overlay_frac": frac,
        "tag": tag,
        "join_recommended": join,
    }


def prize_pool(
    fmt: TournamentFormat,
    entrants: int,
    buy_in: float,
    *,
    rake: float = 0.08,
    guarantee: float = 0.0,
    reentries: int = 0,
) -> float:
    """Total PRIZE pool = per-entry prize contribution * (entrants + reentries), floored by guarantee.

    For PKO this is the PRIZE half only (the bounty half is :func:`bounty_pool`). Re-entries/rebuys
    each add another buy-in's worth of contribution.
    """
    n = int(entrants) + int(reentries)
    contrib = buyin_split(fmt, buy_in, rake=rake)["prize_contrib"]
    pool = contrib * n
    return max(pool, float(guarantee))


# --------------------------------------------------------------------------------------------------
# payout vectors (currency, best-first) -- feed icm_equity
# --------------------------------------------------------------------------------------------------


def pool_fraction_vector(
    fmt: TournamentFormat,
    *,
    entrants: Optional[int] = None,
    places: Optional[int] = None,
    seats: Optional[int] = None,
    frac: Optional[float] = None,
    min_paid: int = 2,
) -> list[float]:
    """Unit-agnostic per-rank structure (best-first), the pool-fraction shape of the ladder.

    NORMAL/PKO: ``payout_structure(fmt, places or places_paid(...))``.
    SATELLITE: a flat ``[1/seats]*seats`` (every seat is identical) -- the unit-agnostic seat shape.
    """
    if fmt is TournamentFormat.SATELLITE:
        if seats is None:
            raise ValueError("SATELLITE pool_fraction_vector requires seats")
        s = int(seats)
        if entrants is not None:
            s = min(s, int(entrants))
        return [1.0 / s] * s if s > 0 else []

    p = places if places is not None else places_paid(
        fmt, entrants, frac=frac, min_paid=min_paid
    )
    return payout_structure(fmt, p)


def payout_vector(
    fmt: TournamentFormat,
    *,
    entrants: Optional[int] = None,
    places: Optional[int] = None,
    prize_pool: Optional[float] = None,
    buy_in: Optional[float] = None,
    rake: float = 0.08,
    guarantee: float = 0.0,
    reentries: int = 0,
    seats: Optional[int] = None,
    seat_value: Optional[float] = None,
    frac: Optional[float] = None,
    min_paid: int = 2,
) -> list[float]:
    """Descending best-first CURRENCY payout vector, ready for :func:`icm.icm_equity`.

    Pool-conserving: ``sum(vector) == pool`` (NORMAL/PKO) / ``seats * seat_value`` (SATELLITE).

    * NORMAL/PKO::

          places = places or places_paid(fmt, entrants, ...)
          pool   = prize_pool (if given) else prize_pool(fmt, entrants, buy_in, ...)
          return [pool * f for f in payout_structure(fmt, places)]

    * SATELLITE: ``[seat_value] * seats`` -- every paid finisher wins an identical seat.

    Note the local name ``prize_pool`` shadows the module function; when not supplied we compute the
    pool via the module-level :func:`prize_pool` (referenced through globals).
    """
    if fmt is TournamentFormat.SATELLITE:
        if seats is None or seat_value is None:
            raise ValueError("SATELLITE payout_vector requires seats and seat_value")
        s = int(seats)
        if entrants is not None:
            s = min(s, int(entrants))
        return [float(seat_value)] * s

    p = places if places is not None else places_paid(
        fmt, entrants, frac=frac, min_paid=min_paid
    )

    if prize_pool is not None:
        pool = float(prize_pool)
    else:
        if buy_in is None or entrants is None:
            raise ValueError(
                "NORMAL/PKO payout_vector needs either prize_pool, or (entrants and buy_in)"
            )
        # call the module-level prize_pool function (shadowed by the local arg name)
        pool = globals()["prize_pool"](
            fmt, entrants, buy_in, rake=rake, guarantee=guarantee, reentries=reentries
        )

    structure = payout_structure(fmt, p)
    return [pool * f for f in structure]
