"""
Push/fold range tables for short-stack NLH.

HU (table_size=2): solver-derived from ``hu_preflop_dcfr`` when
``data/aof/hu_pushfold_8bb.json`` is present.

3/4-max: loose chart-pool static profile (ICMIZER-style; not Nash).
Tighter archive for A/B scripts: ``aof_ranges_tight.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Literal

Provenance = Literal[
    "solver_hu_dcfr",
    "hand_tuned",
    "static_profile",
    "conditional_call_chart",
    "fallback_static",
    "unknown",
]
SecondShoverSource = Literal["conditional", "fallback_static"]
ThirdShoverSource = Literal[
    "conditional_two_jams",
    "fallback_call_chart",
    "fallback_static",
]


class RangeIntent(Enum):
    OPEN_SHOVE = "open_shove"
    CALL_OPEN_SHOVE = "call_open_shove"
    SECOND_SHOVER_FACING_JAM = "second_shover_facing_jam"
    THIRD_SHOVER_FACING_TWO_JAMS = "third_shover_facing_two_jams"
    OVERCALL_AFTER_HERO_CALL = "overcall_after_hero_call"
    CALL_AFTER_TWO_ALL_INS = "call_after_two_all_ins"
    CALL_AFTER_THREE_ALL_INS = "call_after_three_all_ins"


@dataclass(frozen=True)
class RangeLookupResult:
    range_str: str
    pct: float | None
    provenance: Provenance
    intent: RangeIntent
    source_key: tuple
    is_fallback: bool
    warning: str | None

# Stack depths (bb) we have tables for; nearest depth is used.
STACK_DEPTHS = (4, 6, 8, 10, 12, 15, 20)

_HU_SOLVER_PATH = Path(__file__).resolve().parent / "data" / "aof" / "hu_pushfold_8bb.json"
_SOLVER_HU: dict | None = None

from aof_ranges_chart import (
    CALL_PCT,
    CALL_RANGE_STR,
    SECOND_JAM_RANGE_STR,
    SHOVE_RANGE_STR,
    THIRD_JAM_RANGE_STR,
)

# Backward-compatible alias (tests / docs may reference chart module name).
CHART_CALL_RANGE_STR = CALL_RANGE_STR
CHART_CALL_PCT = CALL_PCT
CHART_SHOVE_RANGE_STR = SHOVE_RANGE_STR


def _depth_key(stack_bb: float) -> int:
    return min(STACK_DEPTHS, key=lambda d: abs(d - stack_bb))


def _second_jam_map(stack_bb: float) -> dict[tuple[str, str], str]:
    depth = _depth_key(stack_bb)
    return SECOND_JAM_RANGE_STR.get(depth, SECOND_JAM_RANGE_STR[8])


def _third_jam_map(stack_bb: float) -> dict[tuple[str, str, str], str]:
    depth = _depth_key(stack_bb)
    return THIRD_JAM_RANGE_STR.get(depth, THIRD_JAM_RANGE_STR[8])


def has_second_jam_chart(
    first_shover: str,
    second_shover: str,
    stack_bb: float,
    *,
    table_size: int = 4,
) -> bool:
    first_shover = map_pos_for_table(first_shover, table_size)
    second_shover = map_pos_for_table(second_shover, table_size)
    return (second_shover, first_shover) in _second_jam_map(stack_bb)


def has_third_jam_chart(
    first_shover: str,
    second_shover: str,
    third_shover: str,
    stack_bb: float,
    *,
    table_size: int = 4,
) -> bool:
    first_shover = map_pos_for_table(first_shover, table_size)
    second_shover = map_pos_for_table(second_shover, table_size)
    third_shover = map_pos_for_table(third_shover, table_size)
    return (third_shover, first_shover, second_shover) in _third_jam_map(stack_bb)


def _load_solver_hu() -> dict | None:
    global _SOLVER_HU
    if _SOLVER_HU is not None:
        return _SOLVER_HU
    if not _HU_SOLVER_PATH.is_file():
        _SOLVER_HU = {}
        return None
    _SOLVER_HU = json.loads(_HU_SOLVER_PATH.read_text(encoding="utf-8"))
    return _SOLVER_HU


def hu_solve_manifest() -> dict:
    data = _load_solver_hu()
    return dict((data or {}).get("manifest", {}))


def range_provenance(
    caller: str,
    opener: str,
    stack_bb: float,
    *,
    table_size: int = 4,
    shove: bool = False,
) -> Provenance:
    if table_size == 2 and _load_solver_hu():
        if shove and opener == "SB":
            return "solver_hu_dcfr"
        if not shove and caller == "BB" and opener == "SB":
            return "solver_hu_dcfr"
    return "hand_tuned"


def _strict_ranges_enabled() -> bool:
    from aof_equity_config import aof_strict_ranges

    return aof_strict_ranges()


def lookup_range(
    intent: RangeIntent,
    actor: str,
    stack_bb: float,
    *,
    table_size: int = 4,
    first_shover: str | None = None,
    second_shover: str | None = None,
    hero: str | None = None,
    allow_fallback: bool = True,
) -> RangeLookupResult:
    """Unified range lookup with explicit intent and fallback metadata."""
    actor = map_pos_for_table(actor, table_size)
    if intent == RangeIntent.OPEN_SHOVE:
        rng, prov = lookup_shove(actor, stack_bb, table_size=table_size)
        return RangeLookupResult(
            range_str=rng,
            pct=None,
            provenance=prov,
            intent=intent,
            source_key=("shove", actor, _depth_key(stack_bb), table_size),
            is_fallback=False,
            warning=None,
        )
    if intent in (
        RangeIntent.CALL_OPEN_SHOVE,
        RangeIntent.OVERCALL_AFTER_HERO_CALL,
    ):
        opener = map_pos_for_table(first_shover or "", table_size)
        pct, rng, prov = lookup_call(
            opener, actor, stack_bb, table_size=table_size
        )
        is_fb = not has_call_chart(opener, actor, stack_bb, table_size=table_size)
        warn = "missing call chart; generic default" if is_fb and allow_fallback else None
        if is_fb and not allow_fallback and _strict_ranges_enabled():
            raise KeyError(
                f"No call chart for {actor} vs {opener} at {stack_bb}bb "
                f"({table_size}-max)"
            )
        return RangeLookupResult(
            range_str=rng,
            pct=pct,
            provenance=prov if prov != "hand_tuned" else "static_profile",
            intent=intent,
            source_key=("call", actor, opener, _depth_key(stack_bb), table_size),
            is_fallback=is_fb,
            warning=warn,
        )
    if intent == RangeIntent.SECOND_SHOVER_FACING_JAM:
        fs = map_pos_for_table(first_shover or "", table_size)
        ss = map_pos_for_table(second_shover or actor, table_size)
        rng, src, prov = lookup_second_shover_facing_jam(
            fs, ss, stack_bb, table_size=table_size
        )
        is_fb = src == "fallback_static"
        prov_out: Provenance = (
            "conditional_call_chart" if not is_fb else "fallback_static"
        )
        if is_fb and not allow_fallback and _strict_ranges_enabled():
            raise KeyError(f"No conditional chart for 2nd shover {ss} vs {fs}")
        return RangeLookupResult(
            range_str=rng,
            pct=None,
            provenance=prov_out,
            intent=intent,
            source_key=("2nd", fs, ss, _depth_key(stack_bb), table_size),
            is_fallback=is_fb,
            warning="static open-jam fallback" if is_fb else None,
        )
    if intent == RangeIntent.THIRD_SHOVER_FACING_TWO_JAMS:
        fs = map_pos_for_table(first_shover or "", table_size)
        ss = map_pos_for_table(second_shover or "", table_size)
        ts = map_pos_for_table(actor, table_size)
        rng, src, prov = lookup_third_shover_facing_two_jams(
            fs, ss, ts, stack_bb, table_size=table_size
        )
        is_fb = src != "conditional_two_jams"
        prov_out: Provenance = (
            "conditional_call_chart"
            if src == "conditional_two_jams"
            else "fallback_static"
        )
        if is_fb and not allow_fallback and _strict_ranges_enabled():
            raise KeyError(
                f"No THIRD_JAM chart for {ts} vs {fs},{ss} at "
                f"{_depth_key(stack_bb)}bb ({table_size}-max)"
            )
        warn = None
        if src == "fallback_call_chart":
            warn = "fallback_call_chart_not_true_third_shover"
        elif src == "fallback_static":
            warn = "fallback_static open-jam for third shover"
        return RangeLookupResult(
            range_str=rng,
            pct=None,
            provenance=prov_out,
            intent=intent,
            source_key=("3rd", fs, ss, ts, _depth_key(stack_bb), table_size),
            is_fallback=is_fb,
            warning=warn,
        )
    if intent == RangeIntent.CALL_AFTER_TWO_ALL_INS:
        fs = map_pos_for_table(first_shover or "", table_size)
        ss = map_pos_for_table(second_shover or "", table_size)
        r1, _ = lookup_first_shover_range(fs, stack_bb, table_size=table_size)
        r2, src, prov = lookup_second_shover_facing_jam(
            fs, ss, stack_bb, table_size=table_size
        )
        is_fb = src == "fallback_static"
        return RangeLookupResult(
            range_str=r2,
            pct=None,
            provenance="conditional_call_chart"
            if not is_fb
            else "fallback_static",
            intent=intent,
            source_key=("facing2", fs, ss, hero, _depth_key(stack_bb), table_size),
            is_fallback=is_fb,
            warning=f"pair uses 1st={fs} open + 2nd conditional; hero={hero}",
        )
    raise ValueError(f"Unsupported range intent: {intent}")


def lookup_call(
    opener: str,
    caller: str,
    stack_bb: float,
    *,
    table_size: int = 4,
) -> tuple[float, str, Provenance]:
    opener = map_pos_for_table(opener, table_size)
    caller = map_pos_for_table(caller, table_size)
    depth = _depth_key(stack_bb)
    prov = range_provenance(caller, opener, stack_bb, table_size=table_size, shove=False)
    if prov == "solver_hu_dcfr":
        data = _load_solver_hu()
        assert data is not None
        return float(data["call_pct"]), str(data["call_range_str"]), prov
    pct_map = CALL_PCT.get(depth, CALL_PCT[8])
    rng_map = CALL_RANGE_STR.get(depth, CALL_RANGE_STR[8])
    key = (caller, opener)
    pct = pct_map.get(key)
    rng = rng_map.get(key)
    if pct is None or rng is None:
        if _strict_ranges_enabled():
            raise KeyError(
                f"Missing call chart for {caller} vs {opener} at depth {depth} "
                f"({table_size}-max). Set AOF_STRICT_RANGES=0 to allow defaults."
            )
        if caller == "BB":
            pct, rng = 0.12, "88+,A9s+,ATo+,KQs"
        elif caller == "SB":
            pct, rng = 0.28, "22+,A5s+,A8o+,K9s+,KJo+,Q9s+,J9s+"
        else:
            pct, rng = 0.10, "TT+,AJs+,AQo+"
        prov = "unknown"
    return pct, rng, prov


def lookup_shove(
    shover: str,
    stack_bb: float,
    *,
    table_size: int = 4,
) -> tuple[str, Provenance]:
    shover = map_pos_for_table(shover, table_size)
    prov = range_provenance("BB", shover, stack_bb, table_size=table_size, shove=True)
    if prov == "solver_hu_dcfr" and shover == "SB":
        data = _load_solver_hu()
        assert data is not None
        return str(data["shove_range_str"]), prov
    depth = _depth_key(stack_bb)
    rng_map = SHOVE_RANGE_STR.get(depth, SHOVE_RANGE_STR[8])
    rng = rng_map.get(shover)
    if rng is not None:
        return rng, prov
    fallback = SHOVE_RANGE_STR.get(8, {})
    if shover == "SB":
        return fallback.get("SB", "22+,A2s+,A2o+,K2s+,K5o+"), prov
    if shover in ("BTN", "CO"):
        return fallback.get("BTN", "22+,A2s+,A2o+"), prov
    return fallback.get("UTG", "55+,A9s+,ATo+"), prov


def _lookup_shove(shover: str, stack_bb: float, *, table_size: int = 4) -> str:
    rng, _ = lookup_shove(shover, stack_bb, table_size=table_size)
    return rng


def has_call_chart(
    opener: str,
    caller: str,
    stack_bb: float,
    *,
    table_size: int = 4,
) -> bool:
    """True if a hand-tuned continue-vs-open chart exists for this pair."""
    opener = map_pos_for_table(opener, table_size)
    caller = map_pos_for_table(caller, table_size)
    depth = _depth_key(stack_bb)
    rng_map = CALL_RANGE_STR.get(depth, CALL_RANGE_STR[8])
    return (caller, opener) in rng_map


def lookup_first_shover_range(
    position: str,
    stack_bb: float,
    *,
    table_size: int = 4,
) -> tuple[str, Provenance]:
    """Open-jam range for the first shover (before hero acts)."""
    return lookup_shove(position, stack_bb, table_size=table_size)


def lookup_second_shover_facing_jam(
    first_shover: str,
    second_shover: str,
    stack_bb: float,
    *,
    table_size: int = 4,
) -> tuple[str, SecondShoverSource, Provenance]:
    """
    Range for the second jammer facing the first shover's open jam.

    Prefers ``SECOND_JAM_RANGE_STR`` reshove charts; else continue-vs-open
    (``CALL_*``); else static open-jam fallback.
    """
    first_shover = map_pos_for_table(first_shover, table_size)
    second_shover = map_pos_for_table(second_shover, table_size)
    key = (second_shover, first_shover)
    sj = _second_jam_map(stack_bb)
    if key in sj:
        return sj[key], "conditional", "static_profile"
    if has_call_chart(first_shover, second_shover, stack_bb, table_size=table_size):
        _pct, rng, prov = lookup_call(
            first_shover, second_shover, stack_bb, table_size=table_size
        )
        return rng, "conditional", prov
    rng, prov = lookup_shove(second_shover, stack_bb, table_size=table_size)
    return rng, "fallback_static", prov


@dataclass(frozen=True)
class FacingTwoRanges:
    first_shover: str
    second_shover: str
    first_range: str
    second_range: str
    second_source: SecondShoverSource
    provenance_first: Provenance
    provenance_second: Provenance

    @property
    def range_note(self) -> str:
        return (
            f"2nd={self.second_shover}|{self.first_shover} conditional"
            if self.second_source == "conditional"
            else f"2nd={self.second_shover} static fallback"
        )


def lookup_third_shover_facing_two_jams(
    first_shover: str,
    second_shover: str,
    third_shover: str,
    stack_bb: float,
    *,
    table_size: int = 4,
) -> tuple[str, ThirdShoverSource, Provenance]:
    """
    Range for the third jammer facing two prior open jams.

    Prefers ``THIRD_JAM_RANGE_STR``; else legacy call-chart / open-jam fallbacks.
    """
    first_shover = map_pos_for_table(first_shover, table_size)
    second_shover = map_pos_for_table(second_shover, table_size)
    third_shover = map_pos_for_table(third_shover, table_size)
    tkey = (third_shover, first_shover, second_shover)
    tj = _third_jam_map(stack_bb)
    if tkey in tj:
        return tj[tkey], "conditional_two_jams", "static_profile"
    if has_call_chart(first_shover, third_shover, stack_bb, table_size=table_size):
        _pct, rng, prov = lookup_call(
            first_shover, third_shover, stack_bb, table_size=table_size
        )
        return rng, "fallback_call_chart", prov
    r2, src, prov = lookup_second_shover_facing_jam(
        first_shover, third_shover, stack_bb, table_size=table_size
    )
    if src == "conditional":
        return r2, "fallback_call_chart", prov
    rng, prov = lookup_shove(third_shover, stack_bb, table_size=table_size)
    return rng, "fallback_static", prov


@dataclass(frozen=True)
class FacingThreeRangeResult:
    first: RangeLookupResult
    second: RangeLookupResult
    third: RangeLookupResult
    range_note: str
    fallback_used: bool


@dataclass(frozen=True)
class FacingThreeRanges:
    first_shover: str
    second_shover: str
    third_shover: str
    first_range: str
    second_range: str
    third_range: str
    second_source: SecondShoverSource
    third_source: ThirdShoverSource
    provenance_first: Provenance
    provenance_second: Provenance
    provenance_third: Provenance

    @property
    def fallback_used(self) -> bool:
        return (
            self.second_source == "fallback_static"
            or self.third_source != "conditional_two_jams"
        )

    @property
    def range_note(self) -> str:
        s2 = (
            f"2nd={self.second_shover}|{self.first_shover} conditional"
            if self.second_source == "conditional"
            else f"2nd={self.second_shover} static fallback"
        )
        if self.third_source == "fallback_call_chart":
            s3 = (
                f"3rd={self.third_shover}|{self.first_shover},{self.second_shover} "
                "fallback_call_chart"
            )
        elif self.third_source == "fallback_static":
            s3 = f"3rd={self.third_shover} fallback_static"
        elif self.third_source == "conditional_two_jams":
            s3 = (
                f"3rd={self.third_shover}|{self.first_shover},{self.second_shover} "
                "third_jam_chart"
            )
        else:
            s3 = (
                f"3rd={self.third_shover}|{self.first_shover},{self.second_shover} "
                "conditional"
            )
        return f"{s2}; {s3}"


def lookup_facing_three_ranges(
    first_shover: str,
    second_shover: str,
    third_shover: str,
    stack_bb: float,
    *,
    table_size: int = 4,
) -> FacingThreeRangeResult:
    first = lookup_range(
        RangeIntent.OPEN_SHOVE,
        first_shover,
        stack_bb,
        table_size=table_size,
    )
    second = lookup_range(
        RangeIntent.SECOND_SHOVER_FACING_JAM,
        second_shover,
        stack_bb,
        table_size=table_size,
        first_shover=first_shover,
    )
    third = lookup_range(
        RangeIntent.THIRD_SHOVER_FACING_TWO_JAMS,
        third_shover,
        stack_bb,
        table_size=table_size,
        first_shover=first_shover,
        second_shover=second_shover,
    )
    r1, p1 = lookup_first_shover_range(first_shover, stack_bb, table_size=table_size)
    r2, src2, p2 = lookup_second_shover_facing_jam(
        first_shover, second_shover, stack_bb, table_size=table_size
    )
    r3, src3, p3 = lookup_third_shover_facing_two_jams(
        first_shover, second_shover, third_shover, stack_bb, table_size=table_size
    )
    ft = FacingThreeRanges(
        first_shover=first_shover,
        second_shover=second_shover,
        third_shover=third_shover,
        first_range=r1,
        second_range=r2,
        third_range=r3,
        second_source=src2,
        third_source=src3,
        provenance_first=p1,
        provenance_second=p2,
        provenance_third=p3,
    )
    return FacingThreeRangeResult(
        first=first,
        second=second,
        third=third,
        range_note=ft.range_note,
        fallback_used=ft.fallback_used,
    )


def lookup_facing_two_ranges(
    first_shover: str,
    second_shover: str,
    stack_bb: float,
    *,
    table_size: int = 4,
) -> FacingTwoRanges:
    r1, p1 = lookup_first_shover_range(first_shover, stack_bb, table_size=table_size)
    r2, src, p2 = lookup_second_shover_facing_jam(
        first_shover, second_shover, stack_bb, table_size=table_size
    )
    return FacingTwoRanges(
        first_shover=first_shover,
        second_shover=second_shover,
        first_range=r1,
        second_range=r2,
        second_source=src,
        provenance_first=p1,
        provenance_second=p2,
    )


# Uniform-ish range for --equity-only fallback (~top 25% of hands)
EQUITY_ONLY_CALL_RANGE = "88+,A9s+,ATo+,KQs,J9s+,T9s,98s"
EQUITY_ONLY_THRESHOLD = 0.48  # vs tighter range at ~8bb; approximate

# 4-max action order (first to act -> last). Four players: no cutoff seat.
ACTION_ORDER_4 = ("UTG", "BTN", "SB", "BB")

# Preflop action order for short-handed tables (empty seats omitted).
# 3-way: no UTG. HU: button posts SB, BB is the other seat.
ACTIVE_SEATS: dict[int, tuple[str, ...]] = {
    2: ("SB", "BB"),
    3: ("BTN", "SB", "BB"),
    4: ACTION_ORDER_4,
}

# Clockwise button for auto-seat in the picker (within active seats only).
SEAT_CYCLE_CLOCKWISE: dict[int, tuple[str, ...]] = {
    2: ("SB", "BB"),
    3: ("BTN", "SB", "BB"),
    4: ("UTG", "BB", "SB", "BTN"),
}


def active_seats(table_size: int) -> tuple[str, ...]:
    if table_size in ACTIVE_SEATS:
        return ACTIVE_SEATS[table_size]
    if table_size < 2 or table_size > 9:
        raise ValueError(f"table_size must be 2–9, got {table_size}.")
    return ("UTG", "MP", "CO", "BTN", "SB", "BB")[:table_size]


def seat_cycle_clockwise(table_size: int) -> tuple[str, ...]:
    if table_size in SEAT_CYCLE_CLOCKWISE:
        return SEAT_CYCLE_CLOCKWISE[table_size]
    seats = active_seats(table_size)
    return seats

POSITION_ALIASES_COMMON = {
    "EP": "UTG",
    "BUTTON": "BTN",
    "DEALER": "BTN",
    "D": "BTN",
    "SMALLBLIND": "SB",
    "BIGBLIND": "BB",
}

# 6-max+ labels (CO is a real seat).
POSITION_ALIASES_6 = {
    **POSITION_ALIASES_COMMON,
    "UTG+1": "CO",
    "HJ": "CO",
    "CUTOFF": "CO",
    "CUT": "CO",
}

# 4-max: middle seat is the button; accept CO/HJ from 6-max charts as BTN.
POSITION_ALIASES_4 = {
    **POSITION_ALIASES_COMMON,
    "UTG+1": "BTN",
    "HJ": "BTN",
    "CUTOFF": "BTN",
    "CUT": "BTN",
    "CO": "BTN",
}

# Heads-up: button posts SB; BTN/BUTTON aliases map to SB.
POSITION_ALIASES_HU = {
    **POSITION_ALIASES_COMMON,
    "BTN": "SB",
    "BUTTON": "SB",
    "DEALER": "SB",
    "D": "SB",
}

# Backward-compatible name for 6-max parsing paths.
POSITION_ALIASES = POSITION_ALIASES_6


def is_heads_up(table_size: int) -> bool:
    return table_size == 2


def position_aliases(table_size: int) -> dict[str, str]:
    if table_size == 2:
        return POSITION_ALIASES_HU
    if table_size <= 4:
        return POSITION_ALIASES_4
    return POSITION_ALIASES_6


def map_pos_for_table(pos: str, table_size: int) -> str:
    """Map legacy seat names to the table's position set (CO→BTN in 4-max)."""
    key = pos.strip().upper()
    aliases = position_aliases(table_size)
    if key in aliases:
        return aliases[key]
    return key


def normalize_live_position(
    label: str, table_size: int, *, strict: bool = False
) -> str:
    """
    Normalize live / scraper seat labels to canonical advisor tokens.

    4-max: UTG = first to act (live CO-like); CO/CUTOFF/HJ → BTN.
    """
    raw = label.strip().upper().replace(" ", "").replace("-", "")
    aliases = position_aliases(table_size)
    canonical = set(active_seats(table_size))

    if raw in canonical:
        return raw

    mapped = aliases.get(raw, raw)
    if mapped in canonical:
        if strict and raw in ("CO", "CUTOFF", "CUT_OFF", "HJ", "UTG+1") and table_size <= 4:
            if raw not in aliases and raw != mapped:
                raise ValueError(
                    f"Ambiguous live label {label!r} for {table_size}-max; "
                    f"use canonical {list(canonical)}"
                )
        return mapped

    if strict:
        raise ValueError(
            f"Unknown position {label!r} for {table_size}-max table. "
            f"Canonical seats: {list(canonical)}; aliases: {sorted(aliases)}"
        )
    return mapped


def legal_second_jam_pairs(table_size: int) -> tuple[tuple[str, str], ...]:
    """(second, first) shover pairs in preflop order."""
    order = active_seats(table_size)
    out: list[tuple[str, str]] = []
    for i, first in enumerate(order):
        for second in order[i + 1 :]:
            out.append((second, first))
    return tuple(out)


def legal_third_jam_triples(table_size: int) -> tuple[tuple[str, str, str], ...]:
    """(third, first, second) for three sequential jams before a later hero."""
    order = active_seats(table_size)
    out: list[tuple[str, str, str]] = []
    for i, first in enumerate(order):
        for j, second in enumerate(order[i + 1 :], start=i + 1):
            for third in order[j + 1 :]:
                out.append((third, first, second))
    return tuple(out)
