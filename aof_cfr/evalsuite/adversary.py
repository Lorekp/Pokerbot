"""Safety battery — adapting adversaries + the exploitability budget.

The frozen-profile gauntlet structurally could not lose to an adapting opponent (its profile
never moves). These metrics make that risk measurable:

  * :class:`CounterExploiter` — every K hands recomputes an EXACT legal best response against the
    agent's empirically observed aggregate frequencies (the same hidden-card range-width
    reconstruction the agent itself uses — what a strong observing reg could do). AoF BR is
    analytic (Lisy & Bowling LBR's lesson: keep BR exact where tractable). Gate: the agent loses
    no more than 1.5 bb/100 to being modeled.
  * Gear switch — the field flips archetype mid-run; metrics are per-half uplift and the recovery
    half-life of the dual-EWMA adaptation (first end-to-end exercise of ``aof_cfr.adaptive``).
  * Exploitability budget (Ganzfried & Sandholm TEAC'15 "gifts") — the routed policy's ADDED
    legal NashConv over the base must not exceed the EV the reads actually earn.
"""

from __future__ import annotations

import numpy as np

from aof_cfr.buckets import LABEL_TO_INDEX, LABELS_169
from aof_cfr.evalsuite.fields import GearSwitcher
from aof_cfr.evalsuite.paired_arena import ArenaConfig, _mem_db, ci95_from_seeds, run_cell
from aof_cfr.evalsuite.policy import hists_for, routed_arrays
from aof_cfr.infoset import encode_infoset_key, stack_depth_bucket
from aof_cfr.live_exploit import record_observed_decision


class CounterExploiter:
    """Adversary bot: exact legal BR vs the agent's observed frequencies, refreshed every K hands."""

    name = "counter_exploiter"
    leaky = False

    def __init__(self, n: int, *, refresh_every: int = 250, min_conf: float = 0.05,
                 margin_bb: float = 0.0):
        self.n = int(n)
        self.refresh_every = int(refresh_every)
        self.min_conf, self.margin_bb = float(min_conf), float(margin_bb)
        self.db = _mem_db()
        self.hand_index = 0           # advanced by the arena
        self._agent_seat: str | None = None
        self._routed: dict = {}       # (my_seat, depth_bucket) -> {hist: (p, gated, gain)}
        self._last_refresh = -1

    def observe_agent(self, seat: str, hist: str, depth: float, *, pushed: bool) -> None:
        self._agent_seat = seat
        record_observed_decision(self.db, "agent", seat, hist, depth, pushed)

    def push_fn(self, q, n, pos, hist, hand, stack_bb):
        if self._agent_seat is None or pos == self._agent_seat:
            return q.base_strategy(n, pos, hist, hand, stack_bb)
        dkey = float(stack_depth_bucket(stack_bb))
        due = (self.hand_index - self._last_refresh) >= self.refresh_every
        if due or (pos, dkey) not in self._routed:
            # the adversary plays a near-pure BR: single read, legacy point gate at zero margin
            self._routed[(pos, dkey)] = routed_arrays(
                self.db, n, pos, dkey, {self._agent_seat: "agent"},
                min_conf=self.min_conf, margin=self.margin_bb,
                confidence_mode="min", gate_mode="point")
            self._last_refresh = self.hand_index
        r = self._routed[(pos, dkey)].get(hist)
        if r is None:
            return q.base_strategy(n, pos, hist, hand, stack_bb)
        return float(r[0][LABEL_TO_INDEX[hand]])


def counter_exploiter_cell(cfg: ArenaConfig, *, refresh_every: int = 250,
                           loss_floor_bb100: float = -1.5):
    """Agent (shipping arm) vs a table of counter-exploiters. The gate bounds the cost of being
    modeled: adjusted winrate uplift vs the gto arm must stay above ``loss_floor_bb100``."""
    from aof_cfr.evalsuite.result import AoFEvalCell
    counter = CounterExploiter(cfg.players, refresh_every=refresh_every)
    cc = ArenaConfig(**{**cfg.__dict__, "field": [counter]})
    out = run_cell(cc)
    u = out["uplift"].get("adaptive_online", {"mean": 0.0, "ci95": 0.0, "per_seed": []})
    m, ci = (u["mean"], u["ci95"])
    return AoFEvalCell(
        metric="vs_counter_exploiter", players=cfg.players, depth=cfg.stack_bb,
        field_name="counter_exploiter", arm="adaptive_online",
        hands=out["arms"]["adaptive_online"].hands, seeds=len(cfg.seeds),
        mean_bb100=round(m, 3), se_bb100=round(ci / 2, 3), ci95_bb100=round(ci, 3),
        gate=f"uplift_vs_gto >= {loss_floor_bb100} bb/100 (cost of being modeled bounded)",
        # Conservative: the LOWER CI bound of the (signed) uplift must clear the loss floor. The old
        # `or m >= floor` clause was dead (m >= m-ci always) so it collapsed the gate to the point
        # estimate, letting a noisy cell (e.g. -1.4 ± 5.0) pass the safety bound on luck.
        passed=(m - ci) >= loss_floor_bb100,
        meta={"refresh_every": refresh_every})


def gear_switch_cell(cfg: ArenaConfig, *, before="overfolder", after="calling_station",
                     window: int = 100):
    """Field flips archetype at n/2. Reports per-half uplift + recovery half-life (hands after
    the switch until the rolling uplift regains 50% of the pre-switch steady level)."""
    from aof_cfr.evalsuite.fields import ARCHETYPES
    from aof_cfr.evalsuite.paired_arena import _make_policy, _play_phase, _warmup_profiles
    from aof_cfr.evalsuite.result import AoFEvalCell
    from aof_ranges import active_seats
    import aof_cfr.query as q
    q.load_cfr_base()

    b = ARCHETYPES[before] if isinstance(before, str) else before
    a = ARCHETYPES[after] if isinstance(after, str) else after
    halves, recoveries = {"first": [], "second": []}, []
    for seed in cfg.seeds:
        diff_series: list[float] = []
        for agent_seat in active_seats(cfg.players):
            switch = GearSwitcher(b, a, cfg.hands // 2)
            cc = ArenaConfig(**{**cfg.__dict__, "field": [switch],
                                "arms": ("gto", "adaptive_online")})
            profiles = _warmup_profiles(cc, seed, agent_seat, q)
            s_gto: list[float] = []
            s_adp: list[float] = []
            switch.hand_index = 0
            _play_phase(cc, seed, agent_seat, hands=cc.hands, deal_salt=10_000_000,
                        policy=None, q=q, series=s_gto, bv_cache={})
            switch.hand_index = 0
            pol = _make_policy(cc, "adaptive_online", agent_seat, profiles)
            _play_phase(cc, seed, agent_seat, hands=cc.hands, deal_salt=10_000_000,
                        policy=pol, q=q, series=s_adp, bv_cache={},
                        observe_db=profiles["adaptive_db"], observe_model=profiles["model"])
            if len(diff_series) < len(s_adp):
                diff_series.extend([0.0] * (len(s_adp) - len(diff_series)))
            for i, (x, g) in enumerate(zip(s_adp, s_gto)):
                diff_series[i] += (x - g) / cfg.players      # average over rotations
        half = cfg.hands // 2
        first, second = diff_series[:half], diff_series[half:]
        halves["first"].append(float(np.mean(first)) * 100.0)
        halves["second"].append(float(np.mean(second)) * 100.0)
        recoveries.append(_recovery_halflife(first, second, window))

    m1, ci1 = ci95_from_seeds(halves["first"])
    m2, ci2 = ci95_from_seeds(halves["second"])
    rec = [r for r in recoveries if r is not None]
    post_half = cfg.hands // 2
    # GATE = bounded loss: after a gear flip the agent's stale profile must not make it
    # significantly WORSE than playing pure GTO. (Recovery-to-pre-switch-uplift was the old
    # gate and is wrong: the post-switch archetype is simply worth less — an overfolder's +39
    # bb/100 cannot be "recovered" against a calling station. Recovery half-life stays in meta
    # as the adaptation-speed diagnostic.)
    underpowered = post_half < 4 * window or (m1 - ci1) <= 0.0
    bounded = (m2 + ci2) >= -1.5 and m2 >= -1.5
    return AoFEvalCell(
        metric="gear_switch", players=cfg.players, depth=cfg.stack_bb,
        field_name=f"{b.name}->{a.name}", arm="adaptive_online",
        hands=cfg.hands * cfg.players * len(cfg.seeds), seeds=len(cfg.seeds),
        mean_bb100=round(m2, 3), se_bb100=round(ci2 / 2, 3), ci95_bb100=round(ci2, 3),
        gate="post-switch uplift vs gto >= -1.5 bb/100 (stale profile = bounded loss)",
        passed=(True if bounded else (None if underpowered else False)),
        meta={"pre_switch_uplift": round(m1, 3), "pre_switch_ci95": round(ci1, 3),
              "recovery_halflife_hands": (float(np.median(rec)) if rec else None),
              "window": window, "post_switch_hands": post_half})


def _recovery_halflife(first: list[float], second: list[float], window: int) -> float | None:
    """Hands after the switch until the rolling-mean uplift regains 50% of the pre-switch level."""
    if not first or not second:
        return None
    steady = float(np.mean(first))
    if steady <= 0:
        return None                       # no pre-switch edge to regain
    target = 0.5 * steady
    s = np.asarray(second, dtype=np.float64)
    for i in range(window, len(s) + 1):
        if float(s[i - window:i].mean()) >= target:
            return float(i)
    return float("inf")


def exploitability_budget_cell(cfg: ArenaConfig, *, samples: int = 24000, seed: int = 314):
    """Ganzfried–Sandholm budget: added legal NashConv of the routed table over base must not
    exceed the measured EV gain of the reads that justified the deviation."""
    import aof_cfr.query as q
    from aof_cfr.evalsuite.nash_gates import legal_gap_of_rows
    from aof_cfr.evalsuite.paired_arena import _warmup_profiles
    from aof_cfr.evalsuite.result import AoFEvalCell
    from aof_cfr.profiler import base_policy_rows
    from aof_ranges import active_seats
    q.load_cfr_base()

    n, depth = cfg.players, cfg.stack_bb
    base_rows = base_policy_rows(n, depth)
    nc_base = legal_gap_of_rows(n, base_rows, depth, samples=samples, seed=seed)["gap_bb"]
    added, gains = [], []
    for s_i, seed_w in enumerate(cfg.seeds[:2]):                 # 2 seeds suffice (slow metric)
        for agent_seat in active_seats(n):
            profiles = _warmup_profiles(cfg, seed_w, agent_seat, q)
            opp = _opp_map_static(cfg, agent_seat)
            routed = routed_arrays(profiles["frozen_db"], n, agent_seat, depth, opp,
                                   min_conf=cfg.min_confidence, margin=cfg.margin_bb,
                                   confidence_mode=cfg.confidence_mode, gate_mode=cfg.gate_mode)
            from aof_cfr.profiler import _COMBO_W, _COMBO_W_SUM
            table = dict(base_rows)
            gain_sum = 0.0
            for hist, (p_arr, gated, gain) in routed.items():
                # combo-weighted expected gated gain per hand AT this (seat, hist) — an upper
                # bound on realized gain (ignores P(hist reached) < 1), still far tighter than
                # a raw bucket sum.
                gain_sum += float((_COMBO_W * gain * gated).sum() / _COMBO_W_SUM)
                for bi, lab in enumerate(LABELS_169):
                    key = encode_infoset_key(agent_seat, lab, hist, depth)
                    pv = float(p_arr[bi])
                    table[key] = np.array([1.0 - pv, pv])
            nc_r = legal_gap_of_rows(n, table, depth, samples=samples, seed=seed)["gap_bb"]
            added.append(max(0.0, nc_r - nc_base))
            gains.append(gain_sum)
    added_bb = float(np.mean(added))
    gain_bb = float(np.mean(gains))
    return AoFEvalCell(
        metric="exploitability_budget", players=n, depth=depth, field_name=str(cfg.field),
        arm="shift_frozen", hands=0, seeds=min(2, len(cfg.seeds)),
        mean_bb100=round(added_bb * 100, 3), se_bb100=0.0, ci95_bb100=0.0,
        gate="added_exploitability_bb <= measured_ev_gain_bb (gifts bound)",
        passed=added_bb <= max(gain_bb, 1e-9) or added_bb < 0.005,
        meta={"added_exploitability_bb": round(added_bb, 4), "nc_base_bb": round(nc_base, 4),
              "sum_gated_gain_bb": round(gain_bb, 4), "samples": samples})


def _opp_map_static(cfg: ArenaConfig, agent_seat: str) -> dict:
    from aof_cfr.evalsuite.fields import seat_assignment
    from aof_ranges import active_seats
    non_agent = [s for s in active_seats(cfg.players) if s != agent_seat]
    return {s: ident for s, (_b, ident) in seat_assignment(cfg.bots(), non_agent, 0).items()}
