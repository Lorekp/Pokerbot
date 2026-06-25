"""Keep-in / withdraw / switch policy as a function of (bankroll, softness), per stake tier.

Model
-----
Per-hand net-EV (USD) at a stake with big-blind BB (USD) and per-hand net bb/100 n:
    per_hand_net_usd = n * BB / 100

The per-hand net bb/100 for a table of mean softness s at stake k is
    n(s, k) = gross_edge(s) - rake_floor(k)
where gross_edge is the linear map between two measured anchors
    (softness=0.0 -> +26.5 bb/100)   (solid reg)
    (softness=0.815 -> +214.9 bb/100) (recreational)
clamped to that range, and rake_floor(k) is the conservative 6-max 5+-dealt floor
(``pokerbot.economics.rake_floor.floor_bb100`` with mode='floor').

Action policy (this is what the live coach surfaces):
  WITHDRAW  if n(s, k) <= 0                                          (rake kills the game)
  SWITCH_UP if next tier k+1 is accessible (bankroll rule) AND the net of
              (k+1) strictly beats (k) by enough to amortize a switch cost C
              over the planning horizon H hands
  KEEP      otherwise.

Switch cost C is the one-time economic loss of moving up: a missed-session cost
plus a redeposit friction floor. We use C = max(2 BB_up, $5).
The amortization condition is:
    n(s, k+1) * BB_up * H/100  -  C   >   n(s, k) * BB * H/100

The "comfortable bankroll" for tier k is the smallest B such that:
  (1) per-hand net at this tier > 0 (the game is +EV at this stake)
  (2) B >= 30 max-buy-ins at this tier (the standard 30-buyin "rolled" line)
This is the bankroll above which the optimal policy is to never withdraw at this tier.

Outputs
-------
* 4 PNG heatmaps of (bankroll, softness) -> {KEEP, WITHDRAW, SWITCH_UP}, one per tier
* a JSON artifact with per-tier comfortable-bankroll numbers
* a markdown summary
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# --- Live-pipeline constants (must mirror pokerbot.economics.rake_floor) -----------------------------
SOLID_SOFTNESS, SOLID_GROSS_EDGE = 0.0, 26.5
REC_SOFTNESS, REC_GROSS_EDGE = 0.815, 214.9


def softness_to_gross_edge(s: float) -> float:
    s = max(0.0, min(1.0, s))
    span = REC_SOFTNESS - SOLID_SOFTNESS
    t = (s - SOLID_SOFTNESS) / span if span > 0 else 0.0
    return SOLID_GROSS_EDGE + t * (REC_GROSS_EDGE - SOLID_GROSS_EDGE)


def per_hand_net_bb100(s: float, floor_bb100: float) -> float:
    return softness_to_gross_edge(s) - floor_bb100


# --- Stakes (canonical CoinPoker 6-max Hold'em, 5+ dealt cap tier) -----------------------------------
@dataclass(frozen=True)
class Tier:
    label: str               # "micro" / "low" / "mid" / "high"
    stake_key: str           # representative mid-tier stake
    bb_usd: float            # big blind USD
    buyin_min_bb: int        # CoinPoker cash buy-in min, in BB
    buyin_max_bb: int        # CoinPoker cash buy-in max, in BB
    floor_bb100: float       # conservative net-rake floor at this stake, 6-max 5+ (USD-bb/100)


# Representative stake per tier, plus the conservative (15% guaranteed) floor computed by
# ``pokerbot.economics.rake_floor.floor_bb100(sk, n_dealt=6, mode='floor')`` -- the exact
# numbers the live table-select advisory subtracts from ``softness_to_gross_edge``.
# Verified live (2026-06-25):
#   0.02/0.05  ->  9.89 bb/100
#   0.10/0.25  -> 11.51 bb/100
#   0.50/1     ->  9.61 bb/100
#   5/10       ->  6.32 bb/100
TIERS: list[Tier] = [
    Tier("micro", "0.02/0.05", 0.05,  20, 200,  9.89),
    Tier("low",   "0.10/0.25", 0.25,  20, 200, 11.51),
    Tier("mid",   "0.50/1",    1.00,  50, 500,  9.61),
    Tier("high",  "5/10",     10.00,  50, 500,  6.32),
]


# --- Bankroll rule for stake access -----------------------------------------------------------------
def min_buyin_usd(t: Tier) -> float:
    return t.buyin_min_bb * t.bb_usd


def max_buyin_usd(t: Tier) -> float:
    return t.buyin_max_bb * t.bb_usd


def can_sit(t: Tier, bankroll_usd: float) -> bool:
    return bankroll_usd >= min_buyin_usd(t)


# --- Switch cost -------------------------------------------------------------------------------------
# One-time economic loss of moving up a stake: missed-session cost + redeposit friction.
# C = max(2 BBs of the NEW stake, $5). Conservative.
def switch_cost_usd(t_cur: Tier, t_next: Tier) -> float:
    return max(2.0 * t_next.bb_usd, 5.0)


# --- Planning horizon --------------------------------------------------------------------------------
# 30 sessions x ~250 hands = 7,500 hands. Long enough to amortize a one-time switch cost
# against fractions of a bb/100; short enough to be a credible "this season" planning unit.
H_HANDS = 7500


# --- Policy ------------------------------------------------------------------------------------------
ACTION_KEEP, ACTION_WITHDRAW, ACTION_SWITCH_UP, ACTION_SWITCH_DOWN = 0, 1, 2, 3
ACTION_LABELS = {
    ACTION_KEEP: "KEEP",
    ACTION_WITHDRAW: "WITHDRAW",
    ACTION_SWITCH_UP: "SWITCH UP",
    ACTION_SWITCH_DOWN: "SWITCH DOWN",
}
ACTION_COLORS = {
    ACTION_KEEP: "#2ca02c",
    ACTION_WITHDRAW: "#d62728",
    ACTION_SWITCH_UP: "#1f77b4",
    ACTION_SWITCH_DOWN: "#ff7f0e",
}


def policy_at(bankroll_usd: float, softness: float, tier_idx: int) -> tuple[int, str]:
    """Return (action, action_label) for (bankroll, softness) at tier TIERS[tier_idx].

    Risk-adjusted policy with three gates:
        (1) WITHDRAW if softness < SOFT_MIN_PLAYABLE OR per-hand net <= 0
            (table is too tough to play, even if technically +EV)
        (2) SWITCH_UP only if:
            (a) next-tier per-hand Sharpe ratio is at least SHARPE_FLOOR_FRACTION of the
                current tier's Sharpe (risk-adjusted), AND
            (b) the per-hand USD sigma at the next tier is at most RISK_BUDGET_FRACTION
                of bankroll (volatility-budget gate -- prevents over-switching to HIGH
                which has 10x the absolute dollar sigma), AND
            (c) the horizon EV gain clears the switch cost
        (3) SWITCH_DOWN if bankroll < min_buyin of current tier (forced)
    """
    import math
    t = TIERS[tier_idx]
    n = per_hand_net_bb100(softness, t.floor_bb100)
    per_hand_usd = n * t.bb_usd / 100.0
    # WITHDRAW: softness floor OR rake kill
    if softness < SOFT_MIN_PLAYABLE or per_hand_usd <= 0.0:
        return ACTION_WITHDRAW, ACTION_LABELS[ACTION_WITHDRAW]
    if tier_idx < len(TIERS) - 1:
        t_next = TIERS[tier_idx + 1]
        if can_sit(t_next, bankroll_usd):
            n_next = per_hand_net_bb100(softness, t_next.floor_bb100)
            per_hand_usd_next = n_next * t_next.bb_usd / 100.0
            sigma_per_hand_usd_cur = 95.0 * t.bb_usd / 100.0
            sigma_per_hand_usd_next = 95.0 * t_next.bb_usd / 100.0
            # Sharpe gate
            sharpe_cur = per_hand_usd / sigma_per_hand_usd_cur if sigma_per_hand_usd_cur > 0 else 0.0
            sharpe_next = per_hand_usd_next / sigma_per_hand_usd_next if sigma_per_hand_usd_next > 0 else 0.0
            sharpe_ok = (sharpe_cur <= 0.0) or (sharpe_next >= SHARPE_FLOOR_FRACTION * sharpe_cur)
            # Volatility-budget gate: per-hand sigma as fraction of bankroll
            vol_ok = (sigma_per_hand_usd_next <= RISK_BUDGET_FRACTION * bankroll_usd)
            if sharpe_ok and vol_ok:
                C = switch_cost_usd(t, t_next)
                horizon_cur = per_hand_usd * H_HANDS
                horizon_next = per_hand_usd_next * H_HANDS - C
                if horizon_next > horizon_cur and horizon_next > 0:
                    return ACTION_SWITCH_UP, ACTION_LABELS[ACTION_SWITCH_UP]
    if bankroll_usd < min_buyin_usd(t):
        return ACTION_SWITCH_DOWN, ACTION_LABELS[ACTION_SWITCH_DOWN]
    return ACTION_KEEP, ACTION_LABELS[ACTION_KEEP]


SHARPE_FLOOR_FRACTION = 0.80   # next-tier Sharpe must be >= 80% of current-tier Sharpe to switch
SOFT_MIN_PLAYABLE = 0.50       # below this softness, withdraw (table too tough to grind)
RISK_BUDGET_FRACTION = 0.0015  # per-hand USD sigma must be <= 0.15% of bankroll to switch up


# --- Comfortable bankroll per tier -------------------------------------------------------------------
def comfortable_bankroll(tier_idx: int, softness: float = 0.5) -> dict:
    """Smallest B such that the optimal policy at TIERS[tier_idx] is KEEP (never withdraw).

    Definition (the bankroll above which optimal policy is to NEVER withdraw at this tier):
      (1) per_hand_net_bb100(s, k) > 0            (this is a +EV game at this stake)
      (2) B >= 30 * max_buyin_usd(k)              (the canonical 30-buyin "rolled" line)
      (3) switching up no longer strictly dominates (this tier is on the optimal ladder
          for this softness given the player's current roll)

    Returns a dict with the bankroll in USD + in BBs and the criteria met.
    """
    t = TIERS[tier_idx]
    n = per_hand_net_bb100(softness, t.floor_bb100)
    per_hand_usd = n * t.bb_usd / 100.0
    if per_hand_usd <= 0:
        return {"tier": t.label, "stake": t.stake_key, "softness": softness,
                "comfortable_bankroll_usd": None, "comfortable_bankroll_bb": None,
                "note": "tier unprofitable at this softness -- comfortable bankroll is undefined"}
    # The canonical "rolled" line: 30 max buy-ins. This is the bankroll above which the
    # standard pro advice is "never withdraw at this stake", so it is our operational definition.
    b_comfort = 30.0 * max_buyin_usd(t)
    # EV-amortizes-switch line: the bankroll where horizon EV at THIS tier clears the cost
    # of switching up. This is the second-order "we shouldn't switch up just because we can"
    # threshold -- a useful sanity check.
    if tier_idx < len(TIERS) - 1:
        t_next = TIERS[tier_idx + 1]
        C = switch_cost_usd(t, t_next)
        n_next = per_hand_net_bb100(softness, t_next.floor_bb100)
        per_hand_usd_next = n_next * t_next.bb_usd / 100.0
        # breakeven softness above which switching up is even +EV
        if per_hand_usd_next > 0:
            horizon_gain_at_breakeven = 0.0
            # bankroll B is implicit through "we can sit at next" (can_sit(t_next, B) >= min_buyin).
            # The amortization-of-switch cost adds a SECOND bankroll threshold independent of
            # can_sit: per_hand_usd * H must exceed C. This is a CONSTANT (independent of B), so
            # it does not produce a "comfortable bankroll" -- it produces a "comfortable per-hand".
            # We instead record the bankroll where can_sit(t_next) flips from FALSE to TRUE
            # (the "min roll for next tier") -- that is the BANKROLL-side threshold for SWITCH_UP.
            b_min_next = min_buyin_usd(t_next)
        else:
            b_min_next = None
    else:
        C = 0.0
        b_min_next = None
    return {
        "tier": t.label,
        "stake": t.stake_key,
        "softness": softness,
        "comfortable_bankroll_usd": b_comfort,
        "comfortable_bankroll_bb": b_comfort / t.bb_usd,
        "max_buyin_usd": max_buyin_usd(t),
        "min_buyin_usd": min_buyin_usd(t),
        "per_hand_net_bb100": n,
        "per_hand_net_usd": per_hand_usd,
        "horizon_net_usd": H_HANDS * per_hand_usd,
        "switch_up_cost_usd": C,
        "min_buyin_for_next_tier_usd": b_min_next,
        "horizon_hands": H_HANDS,
    }


# --- Plotting ----------------------------------------------------------------------------------------
def plot_tier(tier_idx: int, out_path: Path) -> dict:
    t = TIERS[tier_idx]
    b_lo = 0.5 * min_buyin_usd(t)
    b_hi = 60.0 * max_buyin_usd(t)
    n_b = 240
    n_s = 200
    B = np.linspace(b_lo, b_hi, n_b)
    S = np.linspace(0.0, 1.0, n_s)
    grid = np.zeros((n_s, n_b), dtype=np.int8)
    for i, s in enumerate(S):
        for j, b in enumerate(B):
            grid[i, j] = policy_at(float(b), float(s), tier_idx)[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    rgb = np.zeros((n_s, n_b, 3))
    for a, c in ACTION_COLORS.items():
        mask = (grid == a)
        rgb[mask] = matplotlib.colors.to_rgb(c)
    ax.imshow(rgb, aspect="auto", origin="lower",
              extent=[b_lo, b_hi, 0.0, 1.0], interpolation="nearest")

    # comfortable bankroll vertical line
    b_comfort = 30.0 * max_buyin_usd(t)
    if b_comfort < b_hi:
        ax.axvline(b_comfort, color="white", linestyle="--", linewidth=1.6,
                   label=f"comfortable bankroll (30 buy-ins) = ${b_comfort:,.0f}")
    # breakeven softness
    s_break = None
    for ss in np.linspace(0.0, 1.0, 1001):
        if per_hand_net_bb100(float(ss), t.floor_bb100) > 0:
            s_break = float(ss)
            break
    if s_break is not None:
        ax.axhline(s_break, color="white", linestyle=":", linewidth=1.4,
                   label=f"breakeven softness = {s_break:.2f}")
    # min-buyin for next tier
    if tier_idx < len(TIERS) - 1:
        t_next = TIERS[tier_idx + 1]
        b_next = min_buyin_usd(t_next)
        if b_lo <= b_next <= b_hi:
            ax.axvline(b_next, color="white", linestyle="-.", linewidth=1.2, alpha=0.85,
                       label=f"min buy-in for next tier ({t_next.stake_key}) = ${b_next:,.0f}")

    ax.set_title(f"{t.label.upper()} tier -- {t.stake_key}  (BB=${t.bb_usd:.2f}, "
                 f"rake floor {t.floor_bb100:.2f} bb/100)\n"
                 f"per-hand net USD = (softness->edge - {t.floor_bb100:.2f}) x {t.bb_usd:.2f} / 100",
                 fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel("bankroll (USD, log scale)")
    ax.set_ylabel("table mean softness  (0 = tough regs, 1 = calling station)")
    ax.set_xlim(b_lo, b_hi)
    ax.set_ylim(0.0, 1.0)

    handles = [mpatches.Patch(color=ACTION_COLORS[a], label=ACTION_LABELS[a]) for a in
               [ACTION_KEEP, ACTION_WITHDRAW, ACTION_SWITCH_UP, ACTION_SWITCH_DOWN]]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    # return region fractions for the artifact
    counts = {ACTION_LABELS[a]: int((grid == a).sum()) for a in
              [ACTION_KEEP, ACTION_WITHDRAW, ACTION_SWITCH_UP, ACTION_SWITCH_DOWN]}
    total = float(grid.size)
    fracs = {k: v / total for k, v in counts.items()}
    return {"region_counts": counts, "region_fractions": fracs}


# --- Main --------------------------------------------------------------------------------------------
def main() -> dict:
    out_dir = Path("artifacts/keep_withdraw_switch")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"tiers": [], "switch_costs_usd": {}, "horizon_hands": H_HANDS}
    for i, t in enumerate(TIERS):
        png = out_dir / f"tier_{i}_{t.label}.png"
        plot_info = plot_tier(i, png)
        info = comfortable_bankroll(i, softness=0.5)
        if i < len(TIERS) - 1:
            t_next = TIERS[i + 1]
            info["switch_up_cost_to_next_tier_usd"] = switch_cost_usd(t, t_next)
            info["next_tier"] = t_next.label
            info["next_tier_stake"] = t_next.stake_key
        summary["tiers"].append({
            "label": t.label, "stake_key": t.stake_key, "bb_usd": t.bb_usd,
            "min_buyin_usd": min_buyin_usd(t), "max_buyin_usd": max_buyin_usd(t),
            "floor_bb100": t.floor_bb100,
            "png": str(png),
            "regions": plot_info,
            "comfortable": info,
        })
        summary["switch_costs_usd"][
            f"{t.label}->{TIERS[i+1].label}" if i < len(TIERS) - 1 else f"{t.label}->(top)"
        ] = switch_cost_usd(t, TIERS[i+1]) if i < len(TIERS) - 1 else None

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md: list[str] = []
    md.append("# Keep / Withdraw / Switch policy -- per tier")
    md.append("")
    md.append(f"Planning horizon H = {H_HANDS} hands (~30 sessions x 250 hands).")
    md.append("Softness anchors: solid-reg 0.0 -> +26.5 bb/100 gross, recreational 0.815 -> +214.9 bb/100 gross, linear & clamped.")
    md.append("Rake floors are the conservative (15% guaranteed) 6-max 5+-dealt floor from `pokerbot.economics.rake_floor.floor_bb100(sk, n_dealt=6, mode='floor')`.")
    md.append("")
    md.append("## Per-tier summary (at softness=0.5)")
    md.append("")
    md.append("| Tier | Stake | BB (USD) | Floor (bb/100) | per-hand net bb/100 | per-hand net USD | horizon net USD | 30-buyin comfortable BR (USD) | In BBs |")
    md.append("|------|-------|---------:|---------------:|--------------------:|-----------------:|----------------:|------------------------------:|-------:|")
    for row in summary["tiers"]:
        c = row["comfortable"]
        if c.get("comfortable_bankroll_usd") is None:
            br_usd = "n/a"
            br_bb = "n/a"
        else:
            br_usd = f"${c['comfortable_bankroll_usd']:,.0f}"
            br_bb = f"{c['comfortable_bankroll_bb']:,.0f} BB"
        md.append(f"| {row['label']} | {row['stake_key']} | ${row['bb_usd']:.2f} | "
                  f"{row['floor_bb100']:.2f} | {c['per_hand_net_bb100']:+.2f} | "
                  f"${c['per_hand_net_usd']:+.4f} | ${c['horizon_net_usd']:+,.0f} | "
                  f"{br_usd} | {br_bb} |")
    md.append("")
    md.append("## Switch costs (one-time, USD)")
    md.append("")
    md.append("| Move | Cost (USD) |")
    md.append("|------|-----------:|")
    for k, v in summary["switch_costs_usd"].items():
        md.append(f"| {k} | {('$' + format(v, ',.2f')) if v is not None else 'n/a (top tier)'} |")
    md.append("")
    md.append("## Region fractions (bankroll x softness grid)")
    md.append("")
    md.append("| Tier | KEEP | WITHDRAW | SWITCH UP | SWITCH DOWN |")
    md.append("|------|-----:|---------:|----------:|------------:|")
    for row in summary["tiers"]:
        fr = row["regions"]["region_fractions"]
        md.append(f"| {row['label']} | {fr['KEEP']:.3f} | {fr['WITHDRAW']:.3f} | "
                  f"{fr['SWITCH UP']:.3f} | {fr['SWITCH DOWN']:.3f} |")
    md.append("")
    md.append("## Plots")
    md.append("")
    for row in summary["tiers"]:
        md.append(f"### {row['label']} -- {row['stake_key']}")
        md.append(f"![{row['label']}]({os.path.basename(row['png'])})")
        md.append("")
    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return summary


if __name__ == "__main__":
    s = main()
    print(json.dumps(s, indent=2))
