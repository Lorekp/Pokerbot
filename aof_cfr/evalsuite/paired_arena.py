"""Paired arena — CRN arms x duplicate seat rotation x AIVAT decision term, multi-seed CIs.

The measurement instrument for exploitation uplift. Each cell plays the SAME deals through every
arm (common random numbers: deal + bot-decision RNG streams keyed only on (seed, hand)), replays
each configuration with the agent at every seat (duplicate rotation — positional asymmetry cancels
in arm differences), and removes the agent's own action-randomization variance with an AIVAT
decision-term control variate whose baseline is the ANALYTIC base-table (e_push, e_fold) — the
exact value function for AoF (Burch et al., AAAI'18; unbiased for any baseline). Showdowns settle
by 400-board MC equity shares (``exploit_mc._terminal_returns``), which collapses the chance term.

Arms:
  * ``gto``             — the base policy, no reads.
  * ``shift_frozen``    — legacy gauntlet path: warmup observes, routed policy frozen from the
                          lifetime-DB range-width shift.
  * ``adaptive_online`` — THE SHIPPING PATH (``AdaptiveOpponentModel`` + population prior), kept
                          learning during the test phase, routed arrays refreshed on a schedule.

Statistics: >=8 seeds per cell; the CI is over per-seed means (t distribution), reported as
bb/100 ± ci95. At AoF per-hand sigma ~2.5-4bb, CRN-paired arm differences have sigma_d ~1-2bb, so
40k paired hands/cell give ±1.0-1.5 bb/100 — the single-seed 2k-hand gauntlet had se ~4.5 bb/100,
which is why its small cells were noise.

``depth_mode="drift"`` evolves per-seat stacks hand to hand (re-buy below 2bb), prices each hand
at the live effective stack, and writes observations at each villain's OWN depth — the honest
live number once profiles fragment across depth buckets. Symmetric-stack engine limitation: the
hand settles at the table effective stack (no side pots); documented, applies to every arm
equally so paired differences stay meaningful.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aof_cfr.buckets import LABEL_TO_INDEX, combo_to_bucket
from aof_cfr.combo_deal import FULL_DECK, card_id
from aof_cfr.evalsuite.fields import FIELDS, Bot, seat_assignment
from aof_cfr.evalsuite.policy import base_marginals, routed_arrays
from aof_cfr.exploit_mc import _terminal_returns
from aof_cfr.infoset import stack_depth_bucket
from aof_cfr.live_exploit import record_observed_decision
from aof_ranges import active_seats

ARMS = ("gto", "shift_frozen", "adaptive_online")
DEFAULT_SEEDS = (1, 2, 3, 5, 8, 13, 21, 34)


@dataclass
class ArenaConfig:
    players: int = 3
    stack_bb: float = 8.0
    field: str | list = "all_leaky"
    hands: int = 2000                      # test hands per (seed, rotation)
    seeds: tuple = DEFAULT_SEEDS
    warmup_hands: int = 1000               # observation-only hands before the test phase
    arms: tuple = ARMS
    refresh_every: int = 25                # adaptive arm: hands between routed-policy refreshes
    min_confidence: float = 0.25
    margin_bb: float = 0.30
    confidence_mode: str | None = None     # None -> router default; "min" = legacy veto
    gate_mode: str | None = None           # None -> router default; "point" = legacy gate
    population_confidence: float = 0.0     # adaptive arm cold-start floor (engine default)
    exploit_cap: float | None = None       # RNR ceiling for a stable read; None -> N-dependent default
    cap_floor: float | None = None         # RNR floor for a fully non-stationary read; None -> CAP_FLOOR
    depth_mode: str = "fixed"              # "fixed" | "drift"
    rebuy_below_bb: float = 2.0
    aivat: bool = True

    def bots(self) -> list[Bot]:
        return list(FIELDS[self.field]) if isinstance(self.field, str) else list(self.field)


@dataclass
class ArmResult:
    per_seed_bb100: list = field(default_factory=list)       # AIVAT-adjusted
    per_seed_raw_bb100: list = field(default_factory=list)
    exploit_decisions: int = 0
    hands: int = 0

    def mean(self) -> float:
        return float(np.mean(self.per_seed_bb100)) if self.per_seed_bb100 else 0.0

    def se(self) -> float:
        v = self.per_seed_bb100
        return float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0


_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31,
        9: 2.26, 10: 2.23, 15: 2.13, 20: 2.09, 30: 2.04}


def t95(dof: int) -> float:
    if dof <= 0:
        return float("inf")
    ks = sorted(_T95)
    for k in ks:
        if dof <= k:
            return _T95[k]
    return 1.96


def ci95_from_seeds(values) -> tuple[float, float]:
    """(mean, ci95 half-width) over per-seed means; t distribution, dof = n_seeds - 1."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else 0.0, float("inf"))
    return float(v.mean()), float(t95(len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))


def _deal(n, rng):
    deck = list(FULL_DECK)
    rng.shuffle(deck)
    return [(deck[2 * i], deck[2 * i + 1]) for i in range(n)]


def _snap_depth(stack_bb: float) -> float:
    return float(stack_depth_bucket(stack_bb))


class _AgentPolicy:
    """One arm's agent at a fixed seat: routed-array lookup (refreshable) or pure base."""

    def __init__(self, cfg: ArenaConfig, agent_seat: str, *, db=None, model=None,
                 bots: list | None = None):
        self.cfg, self.seat = cfg, agent_seat
        self.db, self.model = db, model
        self.k = len(bots) if bots else 1
        self._routed: dict = {}      # (phase, depth) -> {hist: (p, gated, gain)}
        self._since_refresh = 0

    def _opp_map(self, phase: int) -> dict:
        seats = active_seats(self.cfg.players)
        non_agent = [s for s in seats if s != self.seat]
        return {s: ident for s, (_b, ident) in
                seat_assignment(self.cfg.bots(), non_agent, phase).items()}

    def refresh(self, depth: float) -> None:
        if self.db is None:
            return
        est = self.model.as_estimator() if self.model is not None else None
        conf = self.model.opponent_confidence if self.model is not None else None
        stab = self.model.opponent_stability if self.model is not None else None
        # adaptive estimator takes (db, ident, n, stack, **kw); routed_arrays' default shim
        # matches the same call shape.
        wrapped = (lambda d, i, n_, s: est(d, i, n_, s)) if est is not None else None
        for phase in range(self.k):
            self._routed[(phase, _snap_depth(depth))] = routed_arrays(
                self.db, self.cfg.players, self.seat, _snap_depth(depth), self._opp_map(phase),
                min_conf=self.cfg.min_confidence, margin=self.cfg.margin_bb,
                estimator=wrapped, confidence_fn=conf, stability_fn=stab,
                confidence_mode=self.cfg.confidence_mode, gate_mode=self.cfg.gate_mode,
                exploit_cap=self.cfg.exploit_cap, cap_floor=self.cfg.cap_floor)
        self._since_refresh = 0

    def maybe_refresh(self, depth: float) -> None:
        key = (0, _snap_depth(depth))
        if key not in self._routed or self._since_refresh >= self.cfg.refresh_every:
            self.refresh(depth)

    def tick(self) -> None:
        self._since_refresh += 1

    def p_push(self, q, hand_index: int, hist: str, hand: str, depth: float):
        """(p, is_exploit). Base policy when no routed arrays exist (gto arm)."""
        if self.db is None:
            return q.base_strategy(self.cfg.players, self.seat, hist, hand, depth), False
        routed = self._routed.get((hand_index % self.k, _snap_depth(depth)))
        if routed is None or hist not in routed:
            return q.base_strategy(self.cfg.players, self.seat, hist, hand, depth), False
        p_arr, ex_arr, _g = routed[hist]
        bi = LABEL_TO_INDEX[hand]
        return float(p_arr[bi]), bool(ex_arr[bi])


def _play_phase(cfg: ArenaConfig, seed: int, agent_seat: str, *, hands: int, deal_salt: int,
                policy: _AgentPolicy | None, observe_db=None, observe_model=None,
                bv_cache: dict | None = None, q=None, series: list | None = None):
    """Play ``hands`` deals; returns (sum_adj_bb, sum_raw_bb, n_exploit, n_played).

    ``policy`` None => pure-base agent with no exploit counting (warmup). Observation targets
    (DB / adaptive model) collect every non-agent decision at the villain's own depth.
    Adversary hooks: a bot with ``observe_agent(seat, hist, depth, pushed)`` sees every agent
    decision (it is sitting at the table); pass ``series`` to collect per-hand adjusted bb.
    """
    n = cfg.players
    seats = active_seats(n)
    agent_idx = seats.index(agent_seat)
    bots = cfg.bots()
    non_agent = [s for s in seats if s != agent_seat]
    drift = cfg.depth_mode == "drift"
    stacks = {s: float(cfg.stack_bb) for s in seats}

    total_adj = total_raw = 0.0
    n_exploit = 0
    for h in range(hands):
        if drift:
            for s in seats:                       # re-buy busted/short seats (post-blind floor)
                if stacks[s] < cfg.rebuy_below_bb:
                    stacks[s] = float(cfg.stack_bb)
            eff = min(stacks.values())
        else:
            eff = float(cfg.stack_bb)

        assign = seat_assignment(bots, non_agent, h)
        dr = random.Random((seed * 1_000_003) ^ (deal_salt + h))
        cr = random.Random((seed * 7_654_321) ^ (deal_salt + h + 1))
        cards = _deal(n, dr)
        folded: set[int] = set()
        all_in: set[int] = set()
        corrections = 0.0

        if policy is not None:
            policy.maybe_refresh(eff)

        for idx, seat in enumerate(seats):
            r = cr.random()
            if len([i for i in range(n) if i not in folded]) == 1:
                continue
            hist = "".join("A" if j in all_in else "F" for j in range(idx)) or "-"
            hand = combo_to_bucket(*cards[idx])
            if seat == agent_seat:
                if policy is not None:
                    p, is_ex = policy.p_push(q, h, hist, hand, eff)
                    n_exploit += int(is_ex)
                else:
                    p = q.base_strategy(n, seat, hist, hand, eff)
                agent_depth = stacks[agent_seat] if drift else eff
                _seen = set()
                for _b, _ident in assign.values():        # adversaries watch the agent act
                    if id(_b) not in _seen and hasattr(_b, "observe_agent"):
                        _seen.add(id(_b))
                        _b.observe_agent(agent_seat, hist, agent_depth, pushed=(r < p))
                if cfg.aivat and bv_cache is not None:
                    bv = _bv(bv_cache, n, seat, eff)
                    if hist in bv:
                        ep, ef = bv[hist]
                        bi = LABEL_TO_INDEX[hand]
                        b_push = float(ep[bi])
                        b_fold = float(ef[bi]) if hasattr(ef, "__len__") else float(ef)
                        taken_push = r < p
                        bv_taken = b_push if taken_push else b_fold
                        corrections += bv_taken - (p * b_push + (1.0 - p) * b_fold)
            else:
                bot, ident = assign[seat]
                if hasattr(bot, "hand_index"):
                    bot.hand_index = h
                own_depth = stacks[seat] if drift else eff
                p = bot.push_fn(q, n, seat, hist, hand, eff)
                action_push = r < p
                if observe_db is not None:
                    record_observed_decision(observe_db, ident, seat, hist, own_depth,
                                             pushed=action_push)
                if observe_model is not None:
                    from aof_cfr.infoset import aggregate_infoset_key
                    observe_model.observe_key(ident, aggregate_infoset_key(seat, hist, own_depth),
                                              pushed=action_push)
            (all_in if r < p else folded).add(idx)

        holes = [(card_id(c0), card_id(c1)) for (c0, c1) in cards]
        payoffs = _terminal_returns(n, seats, holes, folded, eff)
        raw = float(payoffs[agent_idx] - sum(payoffs) / n)
        total_raw += raw
        total_adj += raw - corrections
        if series is not None:
            series.append(raw - corrections)
        if drift:
            for i, s in enumerate(seats):
                stacks[s] = max(0.0, stacks[s] + float(payoffs[i]))
        if policy is not None:
            policy.tick()
    return total_adj, total_raw, n_exploit, hands


def _bv(cache: dict, n: int, seat: str, depth: float) -> dict:
    key = (seat, _snap_depth(depth))
    if key not in cache:
        cache[key] = base_marginals(n, seat, _snap_depth(depth))
    return cache[key]


# Test-phase deal salt: keeps test deals DISJOINT from warmup deals (salt 0..warmup-1) for any
# realistic warmup length, so steady-state uplift is measured on unseen hands.
_TEST_SALT = 10_000_000


def _mem_db():
    """An ephemeral, never-persisted opponent DB (unique non-existent path; save() never called)."""
    import tempfile
    import uuid

    from aof_cfr.opponent_db import AoFOpponentDB
    return AoFOpponentDB(path=Path(tempfile.gettempdir()) / f"aof_eval_{uuid.uuid4().hex}.json")


def run_cell(cfg: ArenaConfig) -> dict:
    """Run one cell: every arm over every (seed, rotation) on CRN-paired deals.

    Returns ``{"arms": {name: ArmResult}, "uplift": {name: {mean, ci95, per_seed}}}`` where
    uplift is each arm's per-seed paired difference vs the ``gto`` arm.
    """
    import aof_cfr.query as q
    q.load_cfr_base()
    n = cfg.players
    seats = active_seats(n)
    arms = {a: ArmResult() for a in cfg.arms}
    bv_cache: dict = {}

    for seed in cfg.seeds:
        per_arm_sum = {a: [0.0, 0.0, 0, 0] for a in cfg.arms}   # adj, raw, exploit, hands
        for agent_seat in seats:                                 # duplicate seat rotation
            profiles = _warmup_profiles(cfg, seed, agent_seat, q) if _needs_profiles(cfg) else {}
            for arm in cfg.arms:
                policy = _make_policy(cfg, arm, agent_seat, profiles)
                adj, raw, nex, nh = _play_phase(
                    cfg, seed, agent_seat, hands=cfg.hands, deal_salt=_TEST_SALT,
                    policy=policy, q=q, bv_cache=bv_cache,
                    observe_db=(profiles.get("adaptive_db") if arm == "adaptive_online" else None),
                    observe_model=(profiles.get("model") if arm == "adaptive_online" else None))
                s = per_arm_sum[arm]
                s[0] += adj; s[1] += raw; s[2] += nex; s[3] += nh
        for arm in cfg.arms:
            adj, raw, nex, nh = per_arm_sum[arm]
            arms[arm].per_seed_bb100.append(adj / nh * 100.0)
            arms[arm].per_seed_raw_bb100.append(raw / nh * 100.0)
            arms[arm].exploit_decisions += nex
            arms[arm].hands += nh

    uplift = {}
    if "gto" in arms:
        g = arms["gto"].per_seed_bb100
        for name, res in arms.items():
            if name == "gto":
                continue
            diffs = [a - b for a, b in zip(res.per_seed_bb100, g)]
            m, ci = ci95_from_seeds(diffs)
            uplift[name] = {"mean": m, "ci95": ci, "per_seed": diffs}
    return {"arms": arms, "uplift": uplift}


def _needs_profiles(cfg: ArenaConfig) -> bool:
    return any(a in ("shift_frozen", "adaptive_online") for a in cfg.arms)


def _warmup_profiles(cfg: ArenaConfig, seed: int, agent_seat: str, q) -> dict:
    """Observation-only warmup. Two passes over the SAME warmup deals: one fills the frozen
    arm's DB, one fills the adaptive arm's DB + EWMA model together — separate objects so the
    adaptive arm's online test-phase writes never leak into the frozen profile."""
    from aof_cfr.adaptive import AdaptiveOpponentModel, PopulationPrior

    frozen_db, adaptive_db = _mem_db(), _mem_db()
    model = AdaptiveOpponentModel(prior=PopulationPrior(),
                                  population_confidence=cfg.population_confidence)
    if cfg.warmup_hands > 0:
        _play_phase(cfg, seed, agent_seat, hands=cfg.warmup_hands, deal_salt=0,
                    policy=None, q=q, observe_db=frozen_db, bv_cache=None)
        _play_phase(cfg, seed, agent_seat, hands=cfg.warmup_hands, deal_salt=0,
                    policy=None, q=q, observe_db=adaptive_db, observe_model=model, bv_cache=None)
    return {"frozen_db": frozen_db, "adaptive_db": adaptive_db, "model": model}


def _make_policy(cfg: ArenaConfig, arm: str, agent_seat: str, profiles: dict):
    if arm == "gto":
        return None
    if arm == "shift_frozen":
        pol = _AgentPolicy(_frozen(cfg), agent_seat, db=profiles["frozen_db"], bots=cfg.bots())
        pol.refresh(cfg.stack_bb)       # one BR pass from the frozen profile; never refreshed
        return pol
    if arm == "adaptive_online":
        return _AgentPolicy(cfg, agent_seat, db=profiles["adaptive_db"],
                            model=profiles["model"], bots=cfg.bots())
    raise ValueError(f"unknown arm {arm!r}")


def _frozen(cfg: ArenaConfig) -> ArenaConfig:
    from dataclasses import replace
    return replace(cfg, refresh_every=10 ** 9)
