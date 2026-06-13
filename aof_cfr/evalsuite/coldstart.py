"""Cold-start adaptation curve — uplift as a function of hands observed.

THE number that decides live profitability: the gauntlet's steady-state uplift assumed ~1000
frictionless warmup hands per opponent; live, most opponents are short-sample. This measures
uplift at observation budgets {0, 25, 50, 100, 250, 500, 1000} and reports
``hands_to_half_uplift`` — gate ≤ 300 per the small-game exploitation literature (Hoehn et al.
AAAI'05: meaningful exploitation within 100–300 hands; Southey et al. UAI'05: ~200 hands to
capture a typical opponent with a Nash-prior posterior).

INTERPRETATION: ``budget`` is the number of PRIOR observed hands; the adaptive arm keeps
learning online during the test window (exactly as live). Each point is therefore "session
uplift when sitting down with b prior hands on the field" — the honest live quantity. A
hands_to_half_uplift of ~0 with long test windows means online adaptation inside one session
already captures most of the edge; the curve's rise with budget shows the residual value of
carrying profiles across sessions.
"""

from __future__ import annotations

from dataclasses import replace

from aof_cfr.evalsuite.paired_arena import ArenaConfig, ci95_from_seeds, run_cell

DEFAULT_BUDGETS = (0, 25, 50, 100, 250, 500, 1000)


def coldstart_curve(cfg: ArenaConfig, *, budgets=DEFAULT_BUDGETS, arm: str = "adaptive_online") -> dict:
    """Uplift-vs-observed curve for one arm. Each budget point reruns the cell with
    ``warmup_hands=budget`` (same seeds/deals: paired across budgets too)."""
    points = []
    for b in budgets:
        out = run_cell(replace(cfg, warmup_hands=int(b), arms=("gto", arm)))
        u = out["uplift"][arm]
        points.append({"observed_hands": int(b), "uplift_mean": u["mean"], "uplift_ci95": u["ci95"],
                       "per_seed": u["per_seed"],
                       "exploit_decisions": out["arms"][arm].exploit_decisions})
    return {"arm": arm, "points": points,
            "hands_to_half_uplift": hands_to_half_uplift(points)}


def hands_to_half_uplift(points: list[dict]) -> float | None:
    """First observation budget (linear interpolation) where uplift reaches 50% of the
    largest-budget uplift. None when the steady-state uplift is itself ~0 (nothing to reach)."""
    if not points:
        return None
    steady = points[-1]["uplift_mean"]
    if steady <= 0.25:                       # no real steady-state edge -> metric undefined
        return None
    target = 0.5 * steady
    prev_b, prev_u = points[0]["observed_hands"], points[0]["uplift_mean"]
    if prev_u >= target:
        return float(prev_b)
    for pt in points[1:]:
        b, u = pt["observed_hands"], pt["uplift_mean"]
        if u >= target:
            if u == prev_u:
                return float(b)
            frac = (target - prev_u) / (u - prev_u)
            return float(prev_b + frac * (b - prev_b))
        prev_b, prev_u = b, u
    return float("inf")


def coldstart_cells(cfg: ArenaConfig, *, budgets=DEFAULT_BUDGETS, arm: str = "adaptive_online",
                    gate_hands: float = 300.0):
    """The curve as eval cells (one per budget) + the hands_to_half_uplift gate cell."""
    from aof_cfr.evalsuite.result import AoFEvalCell
    curve = coldstart_curve(cfg, budgets=budgets, arm=arm)
    cells = []
    for pt in curve["points"]:
        m, ci = ci95_from_seeds(pt["per_seed"])
        cells.append(AoFEvalCell(
            metric="coldstart", players=cfg.players, depth=cfg.stack_bb, field_name=str(cfg.field),
            arm=arm, hands=cfg.hands * cfg.players * len(cfg.seeds), seeds=len(cfg.seeds),
            mean_bb100=round(m, 3), se_bb100=round(ci / 2.0, 3), ci95_bb100=round(ci, 3),
            meta={"observed_hands": pt["observed_hands"],
                  "exploit_decisions": pt["exploit_decisions"]}))
    h = curve["hands_to_half_uplift"]
    cells.append(AoFEvalCell(
        metric="coldstart_half_uplift", players=cfg.players, depth=cfg.stack_bb,
        field_name=str(cfg.field), arm=arm, hands=0, seeds=len(cfg.seeds),
        mean_bb100=0.0, se_bb100=0.0, ci95_bb100=0.0,
        gate=f"hands_to_half_uplift <= {gate_hands:g} (Hoehn'05 / Bayes' Bluff)",
        passed=(h is None or h <= gate_hands),
        meta={"hands_to_half_uplift": h}))
    return cells
