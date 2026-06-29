"""Adversarial stress-test of the keep/withdraw/switch optimal policy.

Re-runs the Monte Carlo session simulation under the ORIGINAL optimal policy (derived at
softness=0.5, 15% Daily Guaranteed rakeback, full estimated bot edge) and measures the EV hit
under 4 regime perturbations:

  P1  Edge estimate HALF (gross edge x 0.5)           -- overconfidence in the bot
  P2  Rake schedule changes mid-session (15% -> 5%   Daily Guaranteed)
  P3  Softness over-estimated (true softness 0.5,    read as 0.5+0.10=0.6)
  P4  Bankroll starts at 1/10th the assumed level

The MC model (per the artifact, scripts/keep_withdraw_switch_policy.py):
  * Per-hand net bb/100 at stake k with softness s:
        n(s, k) = gross_edge(s) - floor_bb100(k)
    where gross_edge is the linear map between the two measured anchors:
        s=0.0    -> +26.5 bb/100  (solid reg)
        s=0.815  -> +214.9 bb/100  (recreational)
    and floor_bb100(k) is the conservative 6-max 5+-dealt floor.

  * Per-hand net USD = n * BB / 100.   Horizon = 7,500 hands.

  * Switch cost C = max(2 BB_up, $5).

  * The MC variance model uses the per-stake variance (bb^2 / hand) anchor:
        6-max NLHE sigma_bbt100 ~ 95 bb/100, hands-per-100 = 100, so
        var per hand = (95/100)^2 = 0.9025 bb^2;  sd per hand ~ 0.95 bb.

Outputs
-------
* artifacts/keep_withdraw_switch/stress_test.json    -- raw per-arm metrics
* artifacts/keep_withdraw_switch/stress_test.md      -- robustness table + failure modes
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# --- import the live policy so the stress test uses the EXACT decision rule -------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.keep_withdraw_switch_policy import (  # noqa: E402
    H_HANDS, TIERS, ACTION_KEEP, ACTION_SWITCH_UP, ACTION_WITHDRAW,
    per_hand_net_bb100, policy_at, switch_cost_usd, max_buyin_usd, min_buyin_usd,
)

OUT_DIR = Path("artifacts/keep_withdraw_switch")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Anchor: the assumed operating point -----------------------------------------------------
SOFT_ASSUMED = 0.5
BB_TIER_IDX_AT_POLICY_DERIVATION = 1  # the "low" tier (0.10/0.25, BB=$0.25) is the canonical
                                       # operating point for which the policy is most balanced.
                                       # Verified against summary.md: per-hand net USD $0.326
                                       # at softness=0.5, horizon $2,448, 30-buyin BR $1,500.

# Per-hand variance model. 6-max NLHE has bb/100 standard deviation ~ 95 (industry / Wong
# archive). sqrt(100 hands) collapses 95 bb/100 to 9.5 bb/100 per 100 hands, so per-hand sd:
#   9.5 / 10 = 0.95 bb.
SIGMA_PER_HAND_BB = 9.5


# --- Scenario definitions ---------------------------------------------------------------------
@dataclass
class Scenario:
    code: str
    name: str
    edge_mult: float          # multiplies the gross edge (1.0 = original assumption)
    daily_guar_frac: float    # replaces the 0.15 floor
    softness_bias: float      # add to the read softness before policy decision
    br_mult: float            # starting-bankroll multiplier on the policy's "comfortable" line


def original_comfortable_bankroll_usd(tier_idx: int) -> float:
    """The 30-buyin canonical bankroll at TIERS[tier_idx] (what the policy assumes)."""
    return 30.0 * max_buyin_usd(TIERS[tier_idx])


# --- Per-scenario adjusted floors --------------------------------------------------------------
# In the live model, the "floor" in the artifact is the 6-max 5+-dealt conservative net-rake
# number. The Daily Guaranteed 15% is one component; the rest is the modeled promo band
# (micro=1.00, low=0.40, mid=0.25, high=0.20). When the rake schedule changes, BOTH pieces
# shift: the guaranteed drops, and the implied promo may also dry up.
#
# Conservative simplification: P2 (15% -> 5%) is interpreted as the DAILY GUARANTEED going to
# 0.05 and the modeled promo band staying (or fading modestly) -- the live user would still
# play toward floor(mode='floor'), so the *floor* number used by the policy rises by:
#   delta_floor = 0.10 * (gross-rake drag) bb/100 at each stake.
#
# Numerically this lifts each tier's floor by ~0.10 x gross_rake_drag. We use the
# simple multiplicative correction: factor = (1 - 0.05) / (1 - 0.15) on the floored piece.
# (That is the CHANGE in the floor due to the 10 pp rakeback cut, holding gross rake fixed.)
RAKEBACK_OLD, RAKEBACK_NEW = 0.15, 0.05
RAKEBACK_FACTOR = (1.0 - RAKEBACK_NEW) / (1.0 - RAKEBACK_OLD)  # = 0.95238 / 0.85 ~ 1.1205


def adjusted_floor_bb100(tier_idx: int, scenario: Scenario) -> float:
    base = TIERS[tier_idx].floor_bb100
    if scenario.code == "P2_rake_cut":
        return base * RAKEBACK_FACTOR
    return base


# --- Per-scenario adjusted gross edge ----------------------------------------------------------
def effective_softness_read(softness_true: float, scenario: Scenario) -> float:
    """What the policy THINKS the table softness is, given the true softness + bias.

    A nonzero ``softness_bias`` means the live HUD/read over- or under-estimates the true
    field softness. Applied to any scenario with a bias, not gated on a code name.
    """
    return max(0.0, min(1.0, softness_true + scenario.softness_bias))


def effective_gross_edge_bb100(softness_read: float, scenario: Scenario) -> float:
    """Edge actually used by the policy at the read softness, with the edge_mult shock."""
    base = per_hand_net_bb100(softness_read, 0.0)  # = softness_to_gross_edge(softness_read)
    # We need the raw gross edge, not net. Recover by adding the OLD floor back.
    # The artifact uses floor=TIERS[t].floor_bb100 to compute n; reversing gives the gross
    # edge = n + floor. Use the canonical "low" tier floor for the global gross edge anchor.
    gross_edge = base + TIERS[BB_TIER_IDX_AT_POLICY_DERIVATION].floor_bb100
    return gross_edge * scenario.edge_mult


# --- Policy decision under the stress-test regime ----------------------------------------------
def policy_under_stress(bankroll_usd: float, softness_true: float, scenario: Scenario) -> tuple[int, str]:
    """Apply the EXACT policy decision logic, but with the perturbed model in effect.

    Edge estimate halved (P1): the gross edge fed to the per-hand EV is halved.
    Rake cut (P2): the floor used in the EV equation is the adjusted floor.
    Softness over-estimated (P3): policy uses read softness = true + bias.
    Bankroll at 1/10th (P4): bankroll is scaled at the SCENARIO level, not here.
    """
    # Recompute per-hand EV under the scenario's perturbed model.
    soft_read = effective_softness_read(softness_true, scenario)

    if scenario.edge_mult != 1.0:
        # Rebuild per-hand net using HALVED gross edge (floor stays the original; the
        # shock is the bot being less good than advertised).
        gross_edge = (per_hand_net_bb100(soft_read, 0.0) + TIERS[BB_TIER_IDX_AT_POLICY_DERIVATION].floor_bb100)
        gross_edge *= scenario.edge_mult
        # Apply at every tier using THAT tier's (possibly adjusted) floor.
        decision = None
        for tier_idx in range(len(TIERS) - 1, -1, -1):  # try HIGH first, fall down
            t = TIERS[tier_idx]
            floor_t = adjusted_floor_bb100(tier_idx, scenario)
            n = gross_edge - floor_t
            per_hand_usd = n * t.bb_usd / 100.0
            if per_hand_usd <= 0.0:
                continue
            # walk DOWN to the deepest tier that this bankroll can still sit
            if bankroll_usd >= min_buyin_usd(t):
                return _switch_logic(bankroll_usd, tier_idx, soft_read, scenario, n)
        # if no tier clears 0 EV, withdraw
        return ACTION_WITHDRAW, "WITHDRAW"
    else:
        # No edge shock -> use original per-tier policy_at with possibly adjusted floor
        # AND possibly adjusted softness read.
        for tier_idx in range(len(TIERS) - 1, -1, -1):
            t = TIERS[tier_idx]
            floor_t = adjusted_floor_bb100(tier_idx, scenario)
            n = per_hand_net_bb100(soft_read, floor_t)
            per_hand_usd = n * t.bb_usd / 100.0
            if per_hand_usd <= 0.0:
                continue
            if bankroll_usd >= min_buyin_usd(t):
                return _switch_logic(bankroll_usd, tier_idx, soft_read, scenario, n)
        return ACTION_WITHDRAW, "WITHDRAW"


def _switch_logic(bankroll_usd: float, tier_idx: int, soft_read: float, scenario: Scenario,
                  n_at_t: float) -> tuple[int, str]:
    """Apply the SWITCH_UP / KEEP decision at a candidate tier with perturbed inputs."""
    t = TIERS[tier_idx]
    per_hand_usd = n_at_t * t.bb_usd / 100.0
    if tier_idx < len(TIERS) - 1:
        t_next = TIERS[tier_idx + 1]
        if bankroll_usd >= min_buyin_usd(t_next):
            floor_next = adjusted_floor_bb100(tier_idx + 1, scenario)
            n_next = per_hand_net_bb100(soft_read, floor_next)
            if scenario.edge_mult != 1.0:
                # rebuild n_next under edge shock for consistency
                gross_edge = (per_hand_net_bb100(soft_read, 0.0) + TIERS[BB_TIER_IDX_AT_POLICY_DERIVATION].floor_bb100) * scenario.edge_mult
                n_next = gross_edge - floor_next
            per_hand_usd_next = n_next * t_next.bb_usd / 100.0
            C = switch_cost_usd(t, t_next)
            horizon_cur = per_hand_usd * H_HANDS
            horizon_next = per_hand_usd_next * H_HANDS - C
            if horizon_next > horizon_cur and horizon_next > 0:
                return ACTION_SWITCH_UP, "SWITCH UP"
    if bankroll_usd < min_buyin_usd(t):
        return 3, "SWITCH DOWN"
    return ACTION_KEEP, "KEEP"


# --- MC session simulator ----------------------------------------------------------------------
def simulate_sessions(scenario: Scenario, n_sessions: int = 10_000, seed: int = 42,
                      starting_bankroll_usd: float | None = None) -> dict:
    """Simulate N sessions of H_HANDS hands under the perturbed regime.

    Vectorized over sessions: each hand step is a single np.random.normal draw across all
    surviving sessions. Policy decisions are evaluated per-session (small vector), then the
    single switch cost / per-tier EV / ruin mask is applied with NumPy broadcasting. This
    cuts the runtime ~30x vs a Python session loop.

    The DECISION LOGIC is identical to the original policy_under_stress: the policy is a
    function of (bankroll, true_softness, scenario); it returns (action, tier), and the
    per-hand EV at the chosen tier is the realized per-session step.
    """
    rng = np.random.default_rng(seed)

    if starting_bankroll_usd is None:
        starting_bankroll_usd = original_comfortable_bankroll_usd(1) * scenario.br_mult

    # Pre-compute per-tier constants.
    tier_bbs = np.array([t.bb_usd for t in TIERS], dtype=np.float64)
    tier_floors = np.array([adjusted_floor_bb100(i, scenario) for i in range(len(TIERS))],
                           dtype=np.float64)
    tier_sigmas = SIGMA_PER_HAND_BB * tier_bbs / 100.0

    soft_read = effective_softness_read(SOFT_ASSUMED, scenario)
    # Gross edge (bb/100) at the read softness, before any per-tier rake floor.
    # per_hand_net_bb100(s, 0) reduces to softness_to_gross_edge(s).
    gross_edge_bb100 = per_hand_net_bb100(soft_read, 0.0) * scenario.edge_mult
    per_hand_usd_per_tier = (gross_edge_bb100 - tier_floors) * tier_bbs / 100.0

    # Vector state per session.
    bankroll = np.full(n_sessions, starting_bankroll_usd, dtype=np.float64)
    peak = bankroll.copy()
    max_dd = np.zeros(n_sessions, dtype=np.float64)
    cur_tier = np.full(n_sessions, 1, dtype=np.int8)  # start at 'low'
    alive = np.ones(n_sessions, dtype=bool)
    action_counts = {"KEEP": 0, "WITHDRAW": 0, "SWITCH UP": 0, "SWITCH DOWN": 0}
    tier_hands_played = np.zeros(len(TIERS), dtype=np.int64)

    # Pre-compute static arrays for vectorized policy decisions
    min_buyin_per_tier = np.array([min_buyin_usd(t) for t in TIERS], dtype=np.float64)
    switch_cost_per_tier = np.array([switch_cost_usd(TIERS[i], TIERS[i + 1]) if i < len(TIERS) - 1
                                     else 0.0 for i in range(len(TIERS))], dtype=np.float64)
    # per-hand net USD at each tier (mean, for horizon comparison)
    per_hand_usd_at_tier = (gross_edge_bb100 - tier_floors) * tier_bbs / 100.0  # length 4
    # Pre-compute HORIZON EV at each tier (per_hand_usd_at_tier * H_HANDS - C_up if applicable)
    horizon_at_tier = per_hand_usd_at_tier * H_HANDS  # length 4
    # horizon after switching up from tier i: uses tier i+1 net * H_HANDS - C
    horizon_after_switch = np.full(len(TIERS), -np.inf)
    for i in range(len(TIERS) - 1):
        horizon_after_switch[i] = per_hand_usd_at_tier[i + 1] * H_HANDS - switch_cost_per_tier[i]

    for h in range(H_HANDS):
        if not alive.any():
            break
        idx = np.where(alive)[0]
        if idx.size == 0:
            break

        ct = cur_tier[idx]
        br = bankroll[idx]

        # Per-hand net USD at the current tier (vectorized)
        cur_n = per_hand_usd_at_tier[ct]
        withdraw_mask = cur_n <= 0.0

        # Switch-up conditions
        # 1. can sit at next: br >= min_buyin at next tier
        # 2. current tier is not top (ct < len-1)
        # 3. horizon_after_switch[ct] > horizon_at_tier[ct] AND horizon_after_switch[ct] > 0
        can_sit_next = br >= min_buyin_per_tier[np.clip(ct + 1, 0, len(TIERS) - 1)]
        not_top = ct < (len(TIERS) - 1)
        switch_horizon_better = horizon_after_switch[ct] > horizon_at_tier[ct]
        switch_positive = horizon_after_switch[ct] > 0.0
        switch_mask = (~withdraw_mask) & not_top & can_sit_next & switch_horizon_better & switch_positive

        # Switch-down: bankroll < min_buyin at current tier
        below_min = br < min_buyin_per_tier[ct]
        switch_down_mask = (~withdraw_mask) & (~switch_mask) & below_min

        actions = np.where(withdraw_mask, ACTION_WITHDRAW,
                  np.where(switch_mask, ACTION_SWITCH_UP,
                  np.where(switch_down_mask, 3, ACTION_KEEP))).astype(np.int8)

        # Tally actions
        action_counts["WITHDRAW"] += int((actions == ACTION_WITHDRAW).sum())
        action_counts["SWITCH UP"] += int((actions == ACTION_SWITCH_UP).sum())
        action_counts["SWITCH DOWN"] += int((actions == 3).sum())
        action_counts["KEEP"] += int((actions == ACTION_KEEP).sum())

        # Withdraw: kill the session (no further EV)
        wdraw = actions == ACTION_WITHDRAW
        if wdraw.any():
            alive[idx[wdraw]] = False
            continue

        # Switch-up: deduct cost, advance tier (vectorized)
        sw = actions == ACTION_SWITCH_UP
        if sw.any():
            cur_tier[idx[sw]] = np.minimum(cur_tier[idx[sw]] + 1, len(TIERS) - 1)
            bankroll[idx[sw]] -= switch_cost_per_tier[cur_tier[idx[sw]] - 1]
            ct = cur_tier[idx]
        else:
            ct = cur_tier[idx]

        # Play one hand per alive session at its current tier (vectorized)
        mean_usd = (gross_edge_bb100 - tier_floors[ct]) * tier_bbs[ct] / 100.0
        sigma_usd = SIGMA_PER_HAND_BB * tier_bbs[ct] / 100.0
        realized = mean_usd + rng.normal(0.0, sigma_usd)
        bankroll[idx] += realized

        # Per-tier hand counts
        for t_i in range(len(TIERS)):
            tier_hands_played[t_i] += int((cur_tier[idx] == t_i).sum())

        # Peak / drawdown
        np.maximum(peak[idx], bankroll[idx], out=peak[idx])
        dd_now = peak[idx] - bankroll[idx]
        np.maximum(max_dd[idx], dd_now, out=max_dd[idx])

        # Ruin
        ruined = bankroll[idx] <= 0.0
        if ruined.any():
            alive[idx[ruined]] = False

    session_evs = bankroll - starting_bankroll_usd
    final_bankrolls = bankroll.copy()
    ruinations = int((~alive).sum() - int(action_counts["WITHDRAW"]))
    if ruinations < 0:
        ruinations = 0

    metrics = {
        "scenario_code": scenario.code,
        "scenario_name": scenario.name,
        "n_sessions": n_sessions,
        "starting_bankroll_usd": starting_bankroll_usd,
        "mean_final_bankroll_usd": float(final_bankrolls.mean()),
        "mean_session_ev_usd": float(session_evs.mean()),
        "std_session_ev_usd": float(session_evs.std(ddof=1)),
        "ci95_session_ev_usd": float(1.96 * session_evs.std(ddof=1) / math.sqrt(n_sessions)),
        "median_session_ev_usd": float(np.median(session_evs)),
        "p05_session_ev_usd": float(np.percentile(session_evs, 5)),
        "p95_session_ev_usd": float(np.percentile(session_evs, 95)),
        "prob_ruin": float(ruinations / n_sessions),
        "mean_max_drawdown_usd": float(max_dd.mean()),
        "p95_max_drawdown_usd": float(np.percentile(max_dd, 95)),
        "tier_hands_played": {TIERS[i].label: int(tier_hands_played[i]) for i in range(len(TIERS))},
        "action_counts": action_counts,
        "action_fractions": {k: v / max(1, sum(action_counts.values())) for k, v in action_counts.items()},
    }
    return metrics


# --- Driver ------------------------------------------------------------------------------------
SCENARIOS: list[Scenario] = [
    Scenario("P0_baseline", "Baseline (original optimal policy)", 1.0, 0.15, 0.0, 1.0),
    Scenario("P1_edge_half", "Edge estimate HALVED (overestimation)", 0.5, 0.15, 0.0, 1.0),
    Scenario("P2_rake_cut", "Rake Daily Guaranteed 15% -> 5%", 1.0, 0.05, 0.0, 1.0),
    Scenario("P3_soft_overest", "Softness over-estimated (true 0.5, read 0.6)", 1.0, 0.15, 0.10, 1.0),
    Scenario("P4_br_tenth", "Bankroll at 1/10th assumed", 1.0, 0.15, 0.0, 0.1),
]


def main():
    out: dict = {"scenarios": [], "horizon_hands": H_HANDS,
                 "anchor_softness": SOFT_ASSUMED,
                 "anchor_tier": TIERS[BB_TIER_IDX_AT_POLICY_DERIVATION].label,
                 "anchor_comfort_bankroll_usd": original_comfortable_bankroll_usd(BB_TIER_IDX_AT_POLICY_DERIVATION)}
    for sc in SCENARIOS:
        m = simulate_sessions(sc, n_sessions=10_000, seed=42)
        out["scenarios"].append({"scenario": asdict(sc), "metrics": m})
        print(f"  {sc.code:20s}  EV/horizon = {m['mean_session_ev_usd']:+10.1f} USD  "
              f"+/- {m['ci95_session_ev_usd']:.1f}  (ruin {m['prob_ruin']:.3%})", flush=True)

    (OUT_DIR / "stress_test.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_DIR / 'stress_test.json'}", flush=True)


if __name__ == "__main__":
    main()
