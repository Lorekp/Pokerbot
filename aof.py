#!/usr/bin/env python3
"""
CoinPoker-style 4-max AoF NLH push/fold advisor.

Usage:
  python aof.py "AJo BTN 8bb"           # open-shove
  python aof.py "AA BB vs BTN 8bb"      # call-vs-shove (clean)
  python aof.py --call "AJo SB vs BTN behind 1"
  python aof.py   # REPL

See docs/aof-advisor.md and --help.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

from aof_equity import (
    equity_vs_range,
    equity_vs_two_ranges_adaptive,
    warmup,
)
from aof_equity_config import (
    EquityRuntimeConfig,
    default_equity_config,
    equity_vs_ranges_adaptive,
)
from aof_confidence import AdvisorConfidence, build_confidence
from aof_game_config import default_game_config
from aof_multiway import MultiwayCallTerms, all_in_pot_terms, multiway_call_terms
from aof_parse import CallSpot, OpenSpot, Spot, parse_spot_line
from aof_ranges import (
    EQUITY_ONLY_CALL_RANGE,
    EQUITY_ONLY_THRESHOLD,
    FacingTwoRanges,
    RangeIntent,
    lookup_call,
    FacingThreeRangeResult,
    lookup_facing_three_ranges,
    lookup_facing_two_ranges,
    lookup_range,
    lookup_shove,
    range_provenance,
)

_CFG = default_equity_config()
# Facing 2/3 advisor defaults (CLI/picker may pass higher via evaluate_spot).
FACING2_DEFAULT_SAMPLES = _CFG.facing2_samples
FACING3_DEFAULT_SAMPLES = _CFG.facing3_samples

_EQUITY_BACKEND = "vectorized/phevaluator"

# Flag advice when chosen action beats the alternative by less than this (bb).
MARGINAL_EDGE_BB = 0.30
MARGINAL_EQ_EPS = 0.03  # equity-only mode: distance from threshold

_MARGINAL_RE = re.compile(
    r" \| marginal (win|loss) edge=([+-]?\d+(?:\.\d+)?)bb vs (ALL IN|FOLD)"
)

# Backward-compatible alias
SpotLegacy = OpenSpot


@dataclass(frozen=True)
class AdvisorTeacher:
    """Static-profile chip-EV teacher signal for RL / tooling (not equilibrium)."""

    action: str  # "ALL IN" | "FOLD"
    ev_primary: float  # shove or call EV (bb)
    ev_alt: float  # fold EV (bb)
    edge_bb: float  # |primary - alt|
    p_fold: float | None  # open-shove: P(all fold); facing: None
    facing: bool

    @property
    def action_index(self) -> int:
        """0 = FOLD, 1 = ALL IN."""
        return 1 if self.action == "ALL IN" else 0


# Backward-compatible names (deprecated — not Nash equilibrium).
NashTeacher = AdvisorTeacher
StaticProfileTeacher = AdvisorTeacher


@dataclass(frozen=True)
class AdvisorResult:
    action: str
    detail: str
    confidence: AdvisorConfidence

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "detail": self.detail,
            "confidence": self.confidence.to_dict(),
        }


def _teacher_from_evs(
    ev_primary: float,
    ev_fold: float,
    *,
    p_fold: float | None,
    facing: bool,
) -> AdvisorTeacher:
    action = "ALL IN" if ev_primary > ev_fold else "FOLD"
    ev_chosen = ev_primary if action == "ALL IN" else ev_fold
    ev_alt = ev_fold if action == "ALL IN" else ev_primary
    return AdvisorTeacher(
        action=action,
        ev_primary=ev_primary,
        ev_alt=ev_fold,
        edge_bb=abs(ev_chosen - ev_alt),
        p_fold=p_fold,
        facing=facing,
    )


def static_profile_teacher_open(
    spot: OpenSpot,
    *,
    samples: int,
    seed: int,
) -> AdvisorTeacher:
    ev_shove, ev_fold, detail = shove_ev_static_profile(
        spot, samples=samples, seed=seed
    )
    p_fold = None
    m = re.search(r"P_fold=([\d.]+)", detail)
    if m:
        p_fold = float(m.group(1))
    return _teacher_from_evs(ev_shove, ev_fold, p_fold=p_fold, facing=False)


nash_teacher_open = static_profile_teacher_open


def marginal_suffix(action: str, ev_chosen: float, ev_alt: float) -> str:
    """Append to detail when the recommended line is thin."""
    edge = ev_chosen - ev_alt
    if edge >= MARGINAL_EDGE_BB:
        return ""
    alt = "FOLD" if action == "ALL IN" else "ALL IN"
    kind = "win" if edge > 0 else "loss"
    return f" | marginal {kind} edge={edge:+.2f}bb vs {alt}"


def parse_marginal(detail: str) -> dict | None:
    m = _MARGINAL_RE.search(detail)
    if not m:
        return None
    return {
        "kind": m.group(1),
        "edge_bb": float(m.group(2)),
        "vs": m.group(3),
    }


def detail_without_marginal(detail: str) -> str:
    return _MARGINAL_RE.sub("", detail).strip()


def advice_display_lines(action: str, detail: str) -> list[str]:
    """CLI/picker lines: action, optional marginal warning, core detail."""
    lines = [action]
    m = parse_marginal(detail)
    if m:
        label = "thin win" if m["kind"] == "win" else "thin loss"
        lines.append(
            f"⚠ Marginal {label} ({m['edge_bb']:+.2f}bb vs {m['vs']})"
        )
    lines.append(detail_without_marginal(detail))
    return lines


def _lookup_call(
    opener: str, caller: str, stack_bb: float, *, table_size: int = 4
) -> tuple[float, str]:
    pct, rng, _ = lookup_call(
        opener, caller, stack_bb, table_size=table_size
    )
    return pct, rng


def _open_detail_prefix(spot: OpenSpot) -> str:
    if spot.table_size == 2 and range_provenance(
        "BB", spot.position, spot.stack_bb, table_size=2, shove=True
    ) == "solver_hu_dcfr":
        return "hu-solver:"
    return "static-profile:"


def _facing_three_detail_prefix(
    *,
    fallback_used: bool,
    equity_method: str,
) -> str:
    exact = equity_method == "exact"
    base = "4way-exact" if exact else "4way-mc"
    tag = "fallback" if fallback_used else "conditional"
    return f"{base}+{tag}:"


def _call_detail_prefix(
    spot: CallSpot,
    *,
    facing_two: FacingTwoRanges | None = None,
    facing_three: FacingThreeRangeResult | None = None,
    facing_three_method: str = "mc",
    behind_fallback: bool = False,
) -> str:
    if spot.third_shover and facing_three is not None:
        return _facing_three_detail_prefix(
            fallback_used=facing_three.fallback_used,
            equity_method=facing_three_method,
        )
    if spot.second_shover and facing_two is not None:
        if facing_two.second_source == "conditional":
            return "3way-mc+conditional:"
        return "approx fallback static 2-shover:"
    if spot.players_behind > 0:
        if behind_fallback:
            return "approx fallback: behind static:"
        return "approx: behind conditional:"
    if (
        spot.table_size == 2
        and range_provenance(
            spot.position, spot.shover, spot.stack_bb, table_size=2, shove=False
        )
        == "solver_hu_dcfr"
    ):
        return "hu-solver:"
    return "static-profile:"


def _blind_state(position: str) -> tuple[float, float]:
    """Return (already_posted_bb, dead_pot_if_all_fold_bb) for open shove."""
    if position == "SB":
        return 0.5, 1.0
    if position == "BB":
        return 1.0, 0.5
    return 0.0, 1.5


def _hero_key(hero_cards: tuple[str, str]) -> str:
    return "".join(hero_cards)


def _equity_multiway(
    hero_cards: tuple[str, str],
    villain_ranges: list[str],
    *,
    samples: int,
    seed: int,
    cfg: EquityRuntimeConfig | None = None,
) -> float:
    res = equity_vs_ranges_adaptive(
        _hero_key(hero_cards),
        villain_ranges,
        samples,
        cfg or _CFG,
        seed=seed,
    )
    return res.equity


def _pot_if_called(stack_bb: float, posted: float, callers: int) -> float:
    return 1.5 + callers * stack_bb + (stack_bb - posted)


def _call_pot_terms(hero_pos: str, stack_bb: float) -> tuple[float, float, float, float]:
    posted, _ = _blind_state(hero_pos)
    call_cost = stack_bb - posted
    pot_if_called = 1.5 + 2.0 * stack_bb
    req_equity = call_cost / pot_if_called if pot_if_called > 0 else 1.0
    return posted, call_cost, pot_if_called, req_equity


def _ev_call_fold(
    equity: float,
    pot_if_called: float,
    stack_bb: float,
    hero_posted: float,
) -> tuple[float, float]:
    ev_call = equity * pot_if_called - stack_bb
    ev_fold = -hero_posted
    return ev_call, ev_fold


def _villains_behind(position: str, table_size: int) -> list[str]:
    from aof_ranges import active_seats

    order = active_seats(table_size)
    if position not in order:
        raise ValueError(
            f"Position {position} not in {table_size}-handed order "
            f"({', '.join(order)})."
        )
    return list(order[order.index(position) + 1 :])


def _call_subset_weights(
    call_p: list[float],
    *,
    model: str = "independent",
    rho: float = 0.35,
) -> list[tuple[int, float]]:
    """Return (call_mask, weight) for each caller subset. mask bit i = villain i calls."""
    n = len(call_p)
    out: list[tuple[int, float]] = []
    for mask in range(1 << n):
        w = 1.0
        seen_call = False
        for i in range(n):
            p = min(max(call_p[i], 0.0), 0.99)
            if mask & (1 << i):
                adj = p * (1.0 - rho) if model == "sequential" and seen_call else p
                w *= adj
                seen_call = True
            else:
                w *= 1.0 - p
        out.append((mask, w))
    return out


def shove_ev_open(
    spot: OpenSpot,
    *,
    samples: int,
    seed: int,
    model: str = "independent",
    rho: float = 0.35,
) -> tuple[float, float, str]:
    """Open-shove EV; model is ``independent`` or ``sequential`` caller correlation."""
    posted, fold_win_pot = _blind_state(spot.position)
    invest = spot.stack_bb
    villains = _villains_behind(spot.position, spot.table_size)
    if not villains:
        villains = ["BB"] if spot.position == "SB" else []

    call_p: list[float] = []
    call_rng: list[str] = []
    for v in villains:
        pct, rng = _lookup_call(
            spot.position, v, spot.stack_bb, table_size=spot.table_size
        )
        call_p.append(min(max(pct, 0.0), 0.99))
        call_rng.append(rng)

    ev_fold = -posted
    p_all_fold = 1.0
    for p in call_p:
        p_all_fold *= 1.0 - p

    ev = p_all_fold * fold_win_pot
    weights = _call_subset_weights(call_p, model=model, rho=rho)
    n = len(villains)

    open_samples = samples if samples else _CFG.open_multiway_samples

    for mask, w in weights:
        if w <= 0 or mask == 0:
            continue
        caller_idxs = [i for i in range(n) if mask & (1 << i)]
        k = len(caller_idxs)
        caller_seats = tuple(villains[i] for i in caller_idxs)
        caller_rngs = [call_rng[i] for i in caller_idxs]
        terms = all_in_pot_terms(
            spot.position,
            spot.stack_bb,
            all_in_villains=caller_seats,
            table_size=spot.table_size,
        )
        sub_seed = seed + mask * 9973
        if k == 1:
            eq = _equity_multiway(
                spot.hero_cards,
                caller_rngs,
                samples=open_samples,
                seed=sub_seed,
            )
        else:
            eq = _equity_multiway(
                spot.hero_cards,
                caller_rngs,
                samples=open_samples,
                seed=sub_seed,
            )
        ev += w * (eq * terms.total_pot - terms.hero_total_invested)

    action = "ALL IN" if ev > ev_fold else "FOLD"
    ev_chosen = ev if action == "ALL IN" else ev_fold
    ev_alt = ev_fold if action == "ALL IN" else ev
    tag = model if model != "independent" else "indep"
    prefix = _open_detail_prefix(spot)
    detail = (
        f"{prefix} shoveEV={ev:+.2f}bb foldEV={ev_fold:+.2f}bb "
        f"(P_fold={p_all_fold:.2f}, {len(villains)} behind, {tag}, "
        f"multiway 2+ callers, ~{open_samples} sims, {_EQUITY_BACKEND})"
        f"{marginal_suffix(action, ev_chosen, ev_alt)}"
    )
    return ev, ev_fold, detail


def shove_ev_static_profile(
    spot: OpenSpot,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, str]:
    """Return (ev_shove_bb, ev_fold_bb, detail_line). Independent-call model."""
    return shove_ev_open(spot, samples=samples, seed=seed, model="independent")


shove_ev_nash = shove_ev_static_profile


@dataclass(frozen=True)
class FacingTwoEval:
    ev_call: float
    ev_fold: float
    breakeven_equity: float
    eq_display: float
    terms: MultiwayCallTerms
    ranges: FacingTwoRanges
    samples_used: int


@dataclass(frozen=True)
class FacingThreeEval:
    ev_call: float
    ev_fold: float
    breakeven_equity: float
    eq_display: float
    terms: MultiwayCallTerms
    ranges: FacingThreeRangeResult
    samples_used: int
    equity_method: str


def call_ev_three_shovers(
    spot: CallSpot,
    *,
    samples: int | None = None,
    seed: int,
) -> FacingThreeEval:
    """Call EV vs three prior all-ins (4-way pot; advisor-only)."""
    s2 = spot.second_shover
    s3 = spot.third_shover
    if not s2 or not s3:
        raise ValueError("call_ev_three_shovers requires second_shover and third_shover")
    if spot.table_size < 4:
        raise ValueError("Facing 3 requires 4-max table.")
    n_samples = samples if samples is not None else FACING3_DEFAULT_SAMPLES
    ranges = lookup_facing_three_ranges(
        spot.shover, s2, s3, spot.stack_bb, table_size=spot.table_size
    )
    terms = multiway_call_terms(
        spot.position,
        spot.stack_bb,
        villain_positions=(spot.shover, s2, s3),
        table_size=spot.table_size,
    )
    eq_res = equity_vs_ranges_adaptive(
        _hero_key(spot.hero_cards),
        [
            ranges.first.range_str,
            ranges.second.range_str,
            ranges.third.range_str,
        ],
        n_samples,
        _CFG,
        seed=seed,
    )
    eq = eq_res.equity
    ev_call = eq * terms.total_pot_after_call - terms.hero_total_invested
    ev_fold = terms.ev_fold
    return FacingThreeEval(
        ev_call=ev_call,
        ev_fold=ev_fold,
        breakeven_equity=terms.breakeven_equity,
        eq_display=eq,
        terms=terms,
        ranges=ranges,
        samples_used=eq_res.samples_used,
        equity_method=eq_res.method,
    )


def call_ev_two_shovers(
    spot: CallSpot,
    *,
    samples: int | None = None,
    seed: int,
) -> FacingTwoEval:
    """
    Call EV vs two prior all-ins (advisor-only; RL policy does not model this).

    3-way MC: first shover open-jam range, second shover conditional vs-first-jam range.
    """
    s2 = spot.second_shover
    if not s2:
        raise ValueError("call_ev_two_shovers requires second_shover")
    n_samples = samples if samples is not None else FACING2_DEFAULT_SAMPLES
    ranges = lookup_facing_two_ranges(
        spot.shover, s2, spot.stack_bb, table_size=spot.table_size
    )
    terms = multiway_call_terms(
        spot.position,
        spot.stack_bb,
        villain_positions=(spot.shover, s2),
        table_size=spot.table_size,
    )
    eq_res = equity_vs_ranges_adaptive(
        _hero_key(spot.hero_cards),
        [ranges.first_range, ranges.second_range],
        n_samples,
        _CFG,
        seed=seed,
    )
    eq = eq_res.equity
    ev_call = eq * terms.total_pot_after_call - terms.hero_total_invested
    ev_fold = terms.ev_fold
    return FacingTwoEval(
        ev_call=ev_call,
        ev_fold=ev_fold,
        breakeven_equity=terms.breakeven_equity,
        eq_display=eq,
        terms=terms,
        ranges=ranges,
        samples_used=n_samples,
    )


def call_ev_static_profile(
    spot: CallSpot,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float, float, bool]:
    """Return (ev_call_bb, ev_fold_bb, req_eq, eq_display, messy_behind)."""
    if spot.third_shover:
        f3 = call_ev_three_shovers(spot, samples=samples, seed=seed)
        return f3.ev_call, f3.ev_fold, f3.breakeven_equity, f3.eq_display, True
    if spot.second_shover:
        ft = call_ev_two_shovers(spot, samples=samples, seed=seed)
        return ft.ev_call, ft.ev_fold, ft.breakeven_equity, ft.eq_display, True
    shove_res = lookup_range(
        RangeIntent.OPEN_SHOVE,
        spot.shover,
        spot.stack_bb,
        table_size=spot.table_size,
    )
    shove_rng = shove_res.range_str
    posted, _call_cost, pot_hu, req_eq = _call_pot_terms(spot.position, spot.stack_bb)
    behind_samples = samples if samples else _CFG.behind_samples

    if spot.players_behind == 0:
        eq = _equity_multiway(
            spot.hero_cards, [shove_rng], samples=behind_samples, seed=seed
        )
        ev_call, ev_fold = _ev_call_fold(eq, pot_hu, spot.stack_bb, posted)
        return ev_call, ev_fold, req_eq, eq, False

    behind = spot.behind_seats
    if not behind and spot.players_behind > 0:
        from aof_parse import _infer_players_behind

        _, behind = _infer_players_behind(
            spot.position, spot.shover, spot.table_size
        )

    call_p: list[float] = []
    call_rng: list[str] = []
    behind_fallback = False
    for seat in behind:
        res = lookup_range(
            RangeIntent.CALL_OPEN_SHOVE,
            seat,
            spot.stack_bb,
            table_size=spot.table_size,
            first_shover=spot.shover,
        )
        call_p.append(min(max(res.pct or 0.0, 0.0), 0.99))
        call_rng.append(res.range_str)
        behind_fallback = behind_fallback or res.is_fallback

    eq_hu = _equity_multiway(
        spot.hero_cards, [shove_rng], samples=behind_samples, seed=seed
    )
    ev_hu, ev_fold = _ev_call_fold(eq_hu, pot_hu, spot.stack_bb, posted)

    p_all_fold_behind = 1.0
    for p in call_p:
        p_all_fold_behind *= 1.0 - p

    ev_call = p_all_fold_behind * ev_hu
    n_behind = len(behind)

    for mask in range(1, 1 << n_behind):
        w = 1.0
        active: list[int] = []
        for i in range(n_behind):
            if mask & (1 << i):
                w *= call_p[i]
                active.append(i)
            else:
                w *= 1.0 - call_p[i]
        if w <= 0 or not active:
            continue
        behind_seats = tuple(behind[i] for i in active)
        villain_rngs = [shove_rng] + [call_rng[i] for i in active]
        terms = multiway_call_terms(
            spot.position,
            spot.stack_bb,
            villain_positions=(spot.shover,) + behind_seats,
            table_size=spot.table_size,
        )
        sub = max(behind_samples // 4, 2000)
        eq_mw = _equity_multiway(
            spot.hero_cards,
            villain_rngs,
            samples=sub,
            seed=seed + mask * 7919,
        )
        ev_mw = eq_mw * terms.total_pot_after_call - terms.hero_total_invested
        ev_call += w * ev_mw

    eq_implied = (
        (ev_call + posted) / pot_hu if pot_hu > 0 else 0.0
    )
    return ev_call, ev_fold, req_eq, eq_implied, True


call_ev_nash = call_ev_static_profile


def static_profile_teacher_call(
    spot: CallSpot,
    *,
    samples: int,
    seed: int,
) -> AdvisorTeacher:
    ev_call, ev_fold, _req_eq, _eq, _messy = call_ev_static_profile(
        spot, samples=samples, seed=seed
    )
    return _teacher_from_evs(ev_call, ev_fold, p_fold=None, facing=True)


nash_teacher_call = static_profile_teacher_call


def advise_call_vs_shove(
    spot: CallSpot,
    *,
    samples: int,
    seed: int,
) -> tuple[str, str]:
    if spot.third_shover:
        f3 = call_ev_three_shovers(spot, samples=samples, seed=seed)
        ev_call, ev_fold = f3.ev_call, f3.ev_fold
        req_eq, eq_disp = f3.breakeven_equity, f3.eq_display
        prefix = _call_detail_prefix(
            spot,
            facing_three=f3.ranges,
            facing_three_method=f3.equity_method,
        )
        vs_txt = f"{spot.shover}+{spot.second_shover}+{spot.third_shover}"
        terms = f3.terms
        action = "ALL IN" if ev_call > ev_fold else "FOLD"
        ev_chosen = ev_call if action == "ALL IN" else ev_fold
        ev_alt = ev_fold if action == "ALL IN" else ev_call
        detail = (
            f"{prefix} eq {eq_disp * 100:.1f}% need {req_eq * 100:.1f}% "
            f"pot={terms.gross_pot_after_call:.1f}bb invest={terms.hero_total_invested:.1f}bb "
            f"ranges: {f3.ranges.range_note} samples={f3.samples_used} "
            f"callEV={ev_call:+.2f}bb foldEV={ev_fold:+.2f}bb "
            f"({vs_txt})"
            f"{marginal_suffix(action, ev_chosen, ev_alt)}"
        )
        return action, detail
    if spot.second_shover:
        ft = call_ev_two_shovers(spot, samples=samples, seed=seed)
        ev_call, ev_fold = ft.ev_call, ft.ev_fold
        req_eq, eq_disp = ft.breakeven_equity, ft.eq_display
        prefix = _call_detail_prefix(spot, facing_two=ft.ranges)
        vs_txt = f"{spot.shover}+{spot.second_shover}"
        terms = ft.terms
        action = "ALL IN" if ev_call > ev_fold else "FOLD"
        ev_chosen = ev_call if action == "ALL IN" else ev_fold
        ev_alt = ev_fold if action == "ALL IN" else ev_call
        detail = (
            f"{prefix} eq {eq_disp * 100:.1f}% need {req_eq * 100:.1f}% "
            f"callEV={ev_call:+.2f}bb foldEV={ev_fold:+.2f}bb "
            f"pot={terms.total_pot_after_call:.1f}bb invest={terms.hero_total_invested:.1f}bb "
            f"({vs_txt}; {ft.ranges.range_note}, ~{ft.samples_used} sims)"
            f"{marginal_suffix(action, ev_chosen, ev_alt)}"
        )
        return action, detail

    ev_call, ev_fold, req_eq, eq_disp, messy = call_ev_static_profile(
        spot, samples=samples, seed=seed
    )
    action = "ALL IN" if ev_call > ev_fold else "FOLD"
    ev_chosen = ev_call if action == "ALL IN" else ev_fold
    ev_alt = ev_fold if action == "ALL IN" else ev_call
    behind_fb = False
    if messy and spot.players_behind > 0:
        for seat in spot.behind_seats or ():
            r = lookup_range(
                RangeIntent.CALL_OPEN_SHOVE,
                seat,
                spot.stack_bb,
                table_size=spot.table_size,
                first_shover=spot.shover,
            )
            if r.is_fallback:
                behind_fb = True
                break
    prefix = _call_detail_prefix(spot, behind_fallback=behind_fb)
    vs_txt = spot.shover
    if not messy:
        detail = (
            f"{prefix} eq {eq_disp * 100:.1f}% need {req_eq * 100:.1f}% "
            f"callEV={ev_call:+.2f}bb foldEV={ev_fold:+.2f}bb "
            f"(vs {vs_txt} shove, ~{samples} sims)"
            f"{marginal_suffix(action, ev_chosen, ev_alt)}"
        )
    else:
        detail = (
            f"{prefix} {spot.players_behind} behind | eq {eq_disp * 100:.1f}% "
            f"need {req_eq * 100:.1f}% callEV={ev_call:+.2f}bb foldEV={ev_fold:+.2f}bb "
            f"(vs {vs_txt}, ~{samples} sims)"
            f"{marginal_suffix(action, ev_chosen, ev_alt)}"
        )
    return action, detail


def advise_equity_only(
    spot: OpenSpot,
    *,
    samples: int,
    seed: int,
) -> tuple[str, str]:
    eq = _equity_multiway(
        spot.hero_cards,
        [EQUITY_ONLY_CALL_RANGE],
        samples=samples,
        seed=seed,
    )
    action = "ALL IN" if eq >= EQUITY_ONLY_THRESHOLD else "FOLD"
    detail = (
        f"~equity {eq * 100:.1f}% vs uniform-ish call range "
        f"(threshold {EQUITY_ONLY_THRESHOLD * 100:.0f}%) - APPROX, ignores fold equity"
    )
    return action, detail


def advise_static_profile(
    spot: OpenSpot,
    *,
    samples: int,
    seed: int,
) -> tuple[str, str]:
    ev_shove, ev_fold, detail = shove_ev_static_profile(
        spot, samples=samples, seed=seed
    )
    action = "ALL IN" if ev_shove > ev_fold else "FOLD"
    return action, detail


advise_nash = advise_static_profile


def evaluate_spot_detailed(
    text: str,
    *,
    nash: bool,
    force_call: bool,
    samples: int,
    seed: int,
) -> AdvisorResult:
    """Return action, detail, and structured confidence for a shorthand spot."""
    spot = parse_spot_line(text, force_call=force_call)
    game = default_game_config()
    rake_on = game.rake_model is not None

    if isinstance(spot, CallSpot):
        if spot.third_shover:
            f3 = call_ev_three_shovers(spot, samples=samples, seed=seed)
            action = "ALL IN" if f3.ev_call > f3.ev_fold else "FOLD"
            prefix = _call_detail_prefix(
                spot,
                facing_three=f3.ranges,
                facing_three_method=f3.equity_method,
            )
            prov = (
                "fallback_static"
                if f3.ranges.fallback_used
                else "conditional_call_chart"
            )
            vs_txt = f"{spot.shover}+{spot.second_shover}+{spot.third_shover}"
            ev_chosen = f3.ev_call if action == "ALL IN" else f3.ev_fold
            ev_alt = f3.ev_fold if action == "ALL IN" else f3.ev_call
            detail = (
                f"{prefix} eq {f3.eq_display * 100:.1f}% need {f3.breakeven_equity * 100:.1f}% "
                f"pot={f3.terms.gross_pot_after_call:.1f}bb invest={f3.terms.hero_total_invested:.1f}bb "
                f"ranges: {f3.ranges.range_note} samples={f3.samples_used} "
                f"callEV={f3.ev_call:+.2f}bb foldEV={f3.ev_fold:+.2f}bb "
                f"({vs_txt})"
                f"{marginal_suffix(action, ev_chosen, ev_alt)}"
            )
            conf = build_confidence(
                action=action,
                ev_primary=f3.ev_call,
                ev_alt=f3.ev_fold,
                detail_prefix=prefix,
                equity_method=f3.equity_method,
                samples_used=f3.samples_used,
                range_provenance=prov,
                fallback_used=f3.ranges.fallback_used,
                rake_configured=rake_on,
                side_pots_supported=game.allow_side_pots,
                explicit_warning="Facing 3: static-profile 4-way, not Nash",
            )
            return AdvisorResult(action, detail, conf)

        if spot.second_shover:
            ft = call_ev_two_shovers(spot, samples=samples, seed=seed)
            action = "ALL IN" if ft.ev_call > ft.ev_fold else "FOLD"
            prefix = _call_detail_prefix(spot, facing_two=ft.ranges)
            prov = ft.ranges.second_source
            fallback = ft.ranges.second_source == "fallback_static"
            vs_txt = f"{spot.shover}+{spot.second_shover}"
            ev_chosen = ft.ev_call if action == "ALL IN" else ft.ev_fold
            ev_alt = ft.ev_fold if action == "ALL IN" else ft.ev_call
            detail = (
                f"{prefix} eq {ft.eq_display * 100:.1f}% need {ft.breakeven_equity * 100:.1f}% "
                f"callEV={ft.ev_call:+.2f}bb foldEV={ft.ev_fold:+.2f}bb "
                f"pot={ft.terms.total_pot_after_call:.1f}bb invest={ft.terms.hero_total_invested:.1f}bb "
                f"({vs_txt}; {ft.ranges.range_note}, ~{ft.samples_used} sims)"
                f"{marginal_suffix(action, ev_chosen, ev_alt)}"
            )
            conf = build_confidence(
                action=action,
                ev_primary=ft.ev_call,
                ev_alt=ft.ev_fold,
                detail_prefix=prefix,
                equity_method="mc",
                samples_used=ft.samples_used,
                range_provenance=prov,
                fallback_used=fallback,
                rake_configured=rake_on,
                side_pots_supported=game.allow_side_pots,
            )
            return AdvisorResult(action, detail, conf)

        ev_call, ev_fold, req_eq, eq_disp, messy = call_ev_static_profile(
            spot, samples=samples, seed=seed
        )
        action = "ALL IN" if ev_call > ev_fold else "FOLD"
        behind_fb = False
        if messy and spot.players_behind > 0:
            for seat in spot.behind_seats or ():
                r = lookup_range(
                    RangeIntent.CALL_OPEN_SHOVE,
                    seat,
                    spot.stack_bb,
                    table_size=spot.table_size,
                    first_shover=spot.shover,
                )
                if r.is_fallback:
                    behind_fb = True
                    break
        prefix = _call_detail_prefix(spot, behind_fallback=behind_fb)
        prov = range_provenance(
            spot.position, spot.shover, spot.stack_bb, table_size=spot.table_size, shove=False
        )
        vs_txt = spot.shover
        ev_chosen = ev_call if action == "ALL IN" else ev_fold
        ev_alt = ev_fold if action == "ALL IN" else ev_call
        if not messy:
            detail = (
                f"{prefix} eq {eq_disp * 100:.1f}% need {req_eq * 100:.1f}% "
                f"callEV={ev_call:+.2f}bb foldEV={ev_fold:+.2f}bb "
                f"(vs {vs_txt} shove, ~{samples} sims)"
                f"{marginal_suffix(action, ev_chosen, ev_alt)}"
            )
        else:
            detail = (
                f"{prefix} {spot.players_behind} behind | eq {eq_disp * 100:.1f}% "
                f"need {req_eq * 100:.1f}% callEV={ev_call:+.2f}bb foldEV={ev_fold:+.2f}bb "
                f"(vs {vs_txt}, ~{samples} sims)"
                f"{marginal_suffix(action, ev_chosen, ev_alt)}"
            )
        conf = build_confidence(
            action=action,
            ev_primary=ev_call,
            ev_alt=ev_fold,
            detail_prefix=prefix,
            equity_method="mc",
            samples_used=samples,
            range_provenance=prov,
            fallback_used=behind_fb,
            rake_configured=rake_on,
            side_pots_supported=game.allow_side_pots,
        )
        return AdvisorResult(action, detail, conf)

    if nash:
        ev_shove, ev_fold, detail = shove_ev_static_profile(
            spot, samples=samples, seed=seed
        )
        action = "ALL IN" if ev_shove > ev_fold else "FOLD"
        prefix = _open_detail_prefix(spot)
        if spot.table_size == 2 and prefix.startswith("hu-solver:"):
            prov = "solver_hu_dcfr"
        else:
            prov = "hand_tuned"
        conf = build_confidence(
            action=action,
            ev_primary=ev_shove,
            ev_alt=ev_fold,
            detail_prefix=prefix,
            equity_method="mc",
            samples_used=samples,
            range_provenance=prov,
            rake_configured=rake_on,
            side_pots_supported=game.allow_side_pots,
        )
        return AdvisorResult(action, detail, conf)

    action, detail = advise_equity_only(spot, samples=samples, seed=seed)
    conf = build_confidence(
        action=action,
        ev_primary=0.0,
        ev_alt=0.0,
        detail_prefix="equity-only:",
        equity_method="mc",
        samples_used=samples,
        range_provenance="unknown",
        explicit_warning="equity threshold only — ignores fold equity",
    )
    return AdvisorResult(action, detail, conf)


def evaluate_spot(
    text: str,
    *,
    nash: bool,
    force_call: bool,
    samples: int,
    seed: int,
) -> tuple[str, str]:
    """Return (action, detail) for a shorthand spot."""
    r = evaluate_spot_detailed(
        text, nash=nash, force_call=force_call, samples=samples, seed=seed
    )
    return r.action, r.detail


def run_spot(
    text: str,
    *,
    nash: bool,
    force_call: bool,
    samples: int,
    seed: int,
) -> None:
    action, detail = evaluate_spot(
        text, nash=nash, force_call=force_call, samples=samples, seed=seed
    )
    for line in advice_display_lines(action, detail):
        print(line)


def repl_loop(
    *,
    nash: bool,
    force_call: bool,
    samples: int,
    seed: int,
) -> None:
    print("AoF advisor — enter spots (empty line or 'q' to quit).", file=sys.stderr)
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break
        if not line or line.lower() in {"q", "quit", "exit"}:
            break
        try:
            run_spot(
                line,
                nash=nash,
                force_call=force_call,
                samples=samples,
                seed=seed,
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="4-max AoF NLH push/fold advisor (CoinPoker-style ~8bb).",
        epilog=(
            "Open-shove: 'AJo BTN 8bb'. Call-vs-shove: 'AA BB vs BTN', "
            "'99 SB call BTN', or --call. 4-max seats: UTG → BTN → SB → BB."
        ),
    )
    parser.add_argument(
        "--nash",
        action="store_true",
        default=True,
        help="Chip-EV with fold equity + static-profile ranges (default, open-shove).",
    )
    parser.add_argument(
        "--equity-only",
        action="store_true",
        help="Open-shove: equity threshold only (ignores fold equity).",
    )
    parser.add_argument(
        "--call",
        action="store_true",
        help="Force call-vs-shove mode (facing a shove).",
    )
    parser.add_argument("spot", nargs="?", help='Shorthand spot, e.g. "AJo BTN 8bb".')
    parser.add_argument(
        "--mc",
        type=int,
        default=20000,
        metavar="N",
        help="Monte Carlo samples (default 20000).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42).")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON (action, detail, confidence).",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print confidence tier and warnings after advice.",
    )
    args = parser.parse_args(argv)
    warmup()

    nash = not args.equity_only
    if args.spot:
        try:
            result = evaluate_spot_detailed(
                args.spot,
                nash=nash,
                force_call=args.call,
                samples=args.mc,
                seed=args.seed,
            )
            if args.json:
                import json as _json

                print(_json.dumps(result.to_dict(), indent=2))
            else:
                for line in advice_display_lines(result.action, result.detail):
                    print(line)
                if args.explain:
                    c = result.confidence
                    print(
                        f"[tier={c.approximation_tier} provenance={c.range_provenance} "
                        f"marginal={c.marginal} rake={c.rake_configured}]"
                    )
                    if c.warning:
                        print(f"⚠ {c.warning}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0

    repl_loop(nash=nash, force_call=args.call, samples=args.mc, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
