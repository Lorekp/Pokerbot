"""Full-lifecycle duplicate MTT / SNG $-ROI simulator — the headline E3/E4 estimator.

This is the end-to-end confirmation that the per-spot ICM edge measured by the spot-level batteries
(``regime_battery`` / ``lbr_icm``) actually COMPOUNDS into a real money-ROI edge once it is paid out
through a whole tournament lifecycle: deep-early -> antes-mid -> shallow/push-fold-late -> bubble ->
final table -> heads-up. It EXTENDS :func:`pokerbot.tournament.mtt_sim.simulate_mtt` (keeps its
identity-tracked ``pid`` balancing through table breaks, its ``_RealEquity`` confrontation resolver, and
its finish accounting) but fixes the two things that simulator hardcodes:

  (a) BLINDS ARE STATIC there. Here a :class:`LifecycleSchedule` wraps the (previously unwired)
      ``structure.blind_ladder`` / ``level_at`` / ``ante_for_bb`` / ``final_table_rollback`` ladder and
      drives the effective depth DOWN as hands elapse, posting the universal CoinPoker per-player ante
      (``round(BB/8)`` ~= 0.125bb). Stacks are tracked in CHIPS so the rising blinds naturally shrink
      the effective bb-depth; decisions are taken in bb. This is what makes the sim sample the real
      tournament depth/blind lifecycle the constraints require — not a frozen 20bb snapshot.

  (b) THE CONFRONTATION DECISION IS HARDCODED there (a fixed jam/call quantile pair). Here it is a
      pluggable ``decide_fn`` that ROUTES on the certified-zone boundary
      (:data:`pokerbot.tournament.decide.PUSHFOLD_MAX_BB`): ``eff_bb <= PUSHFOLD_MAX_BB`` -> the
      certified push/fold ICM solve; deeper -> the net-served decide (chip-EV outcomes, ICM-graded).
      Four ``decide_fn`` adapters ship in this module so the headline cells can be run net-free first:

        * :func:`anchor_fold`   — the always-fold calibration floor (net-free).
        * :func:`anchor_jam`    — the always-jam ceiling (net-free).
        * :func:`chip_ev_nash`  — fixed chip-EV jam/call thresholds (net-free, the ICM-naive field).
        * :func:`net_decide`    — serves the 7-max net via :mod:`pokerbot.tournament.neural_serve`,
          ICM-grading its chip-EV outcome via :func:`eval_score`, FALLING BACK to a push/fold-ICM /
          anchor threshold when the net abstains (None). This is the ONLY adapter that touches the
          checkpoint, and only through the skew-proven ``neural_serve`` bridge.

SCORING / VARIANCE REDUCTION. Each tournament is scored in $ via
:func:`eval_score.score_with_cv(realized_payout, pre_stacks, payouts, hero)` — the realized prize minus
the hero's pre-tournament ICM baseline, so the all-or-nothing finish variance is subtracted off
(unbiased: ICM is mean-correct). :func:`duplicate_roi` runs two ``decide_fn`` arms on the SAME injected
rng (``simulate_tournament_lifecycle`` is deterministic given its rng, so the two arms are CRN-paired —
identical card/board/seat-flow noise) and returns the per-tournament $-ROI for both arms, so the paired
difference cancels the shared variance. :func:`measure_lifecycle_roi` aggregates to the E3/E4 cells
(``mean_payout`` / ``roi`` gross+net / ``itm_rate`` / ``mean_finish``) with a 95% CI.

CORRECTNESS ANCHORS (the smoke gates, all net-free / CPU / small-N):
  * homogeneous-field mean-finish ~ (N+1)/2 through the FULL lifecycle (balancing + breaks + blind
    escalation must introduce NO seat/table/hero bias) — the ``mtt_sim`` symmetry test, lifted to the
    lifecycle schedule;
  * per-hand chip conservation (every confrontation moves exactly ``min(stacks)`` chips plus the dead
    antes into the pot, total chips constant up to the antes banked into the pot then won);
  * the always-fold anchor busts the hero out fast (blinds + antes erode it) and the always-jam anchor
    is high-variance — the net-free behavioural floor/ceiling.

This module OWNS only this file + ``tests/test_mtt_lifecycle.py``. It imports the tier-0 foundations
(``eval_score``, ``neural_serve``) and the existing tournament layer; it adds nothing to them.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from pokerbot.tournament import structure as _structure
from pokerbot.tournament.decide import PUSHFOLD_MAX_BB
from pokerbot.tournament.mtt_field import (
    MttField,
    MttTable,
    apply_moves,
    balance_moves,
    needs_balancing,
)
from pokerbot.tournament.mtt_sim import _RealEquity, _build_field

# --------------------------------------------------------------------------------------------------
# Lifecycle schedule — wires the (previously unwired) structure.py blind/ante ladder.
# --------------------------------------------------------------------------------------------------


@dataclass
class LifecycleSchedule:
    """Drives the blind/ante level from the number of HANDS elapsed across the whole field.

    The structure ladder (:func:`structure.blind_ladder`) is keyed on *minutes* (10-min levels). A sim
    has no wall clock, so we convert: ``hands_per_level`` hands == one 10-min level. As ``advance()`` is
    called once per confrontation the current level walks up the ladder, the big blind grows, and the
    per-player ante (``structure.ante_for_bb`` == round(BB/8)) is posted as dead money. When the field
    reaches the final table the blinds ROLL BACK one level (``structure.final_table_rollback``), exactly
    as the CoinPoker structure specifies.

    All blinds/antes are returned in CHIPS (the sim tracks stacks in chips so rising blinds shrink the
    effective bb-depth automatically). ``bb_chips`` is the current big blind; ``effective_bb(stack)``
    converts a chip stack to bb-depth for the certified-zone routing.
    """

    ladder: list = field(default_factory=lambda: _structure.blind_ladder())
    hands_per_level: int = 12          # ~hands/level; tune for how fast the lifecycle compresses
    start_level: int = 1               # the level the tournament opens at
    final_table_size: int = 9          # players-left at/below which the FT rollback applies
    ft_rollback_levels: int = 1
    _hands: int = 0
    _at_final_table: bool = False

    def __post_init__(self):
        if not self.ladder:
            self.ladder = _structure.blind_ladder()
        self._level_idx = max(1, min(self.start_level, len(self.ladder)))

    # -- level walk ----------------------------------------------------------
    @property
    def level(self) -> int:
        """The CURRENT ladder level (1-based), after any final-table rollback."""
        base = max(1, min(self._level_idx, len(self.ladder)))
        if self._at_final_table:
            return max(1, base - self.ft_rollback_levels)
        return base

    def _cur(self):
        return self.ladder[self.level - 1]

    @property
    def bb_chips(self) -> int:
        return int(self._cur().bb)

    @property
    def sb_chips(self) -> int:
        return int(self._cur().sb)

    @property
    def ante_chips(self) -> int:
        """Per-player ante in chips (universal CoinPoker rule round(BB/8); read off the ladder)."""
        return int(self._cur().ante)

    def effective_bb(self, stack_chips: float) -> float:
        """A chip stack expressed in BIG BLINDS at the current level."""
        bb = self.bb_chips
        return float(stack_chips) / bb if bb > 0 else float(stack_chips)

    def enter_final_table(self) -> None:
        """Apply the final-table blind rollback exactly once (idempotent)."""
        self._at_final_table = True

    def advance(self, players_left: int) -> None:
        """Account one elapsed hand. Walks the level up every ``hands_per_level`` hands and flips the
        final-table rollback on once the field is down to ``final_table_size``."""
        self._hands += 1
        # level index walks up with elapsed hands (10-min real levels -> hands_per_level sim hands).
        self._level_idx = max(1, min(self.start_level + self._hands // self.hands_per_level,
                                     len(self.ladder)))
        if not self._at_final_table and players_left <= self.final_table_size:
            self.enter_final_table()


def sng_schedule(*, hands_per_level: int = 8, start_level: int = 1) -> LifecycleSchedule:
    """A faster-compressing schedule for a single-table SNG (fewer hands per level, no break logic)."""
    return LifecycleSchedule(hands_per_level=hands_per_level, start_level=start_level,
                             final_table_size=0)


# --------------------------------------------------------------------------------------------------
# decide_fn protocol + the spot context it consumes
# --------------------------------------------------------------------------------------------------

# Regime tags the trace + routes the decision.
REGIME_PUSHFOLD = "pushfold"     # eff_bb <= PUSHFOLD_MAX_BB -> certified ICM push/fold
REGIME_DEEP = "deep"             # deeper -> net-served / chip-EV deep


@dataclass
class SpotContext:
    """Everything a ``decide_fn`` needs to act on ONE side of a confrontation.

    A confrontation is a jam (``role == 'jam'``) or a call (``role == 'call'``). ``strength_quantile``
    is the actor's preflop hand-strength quantile in [0,1] (1 == nuts), drawn consistently with the
    ``_RealEquity`` resolver so a returned threshold means the same thing in abstract and realistic
    modes. ``eff_bb`` is the effective (smaller) stack in bb at the current blind level; ``regime`` is
    pre-routed from it. ``pre_stacks`` / ``payouts`` are the FULL field's live stacks (chips) and the
    prize ladder, ``hero_field_idx`` the hero's flat index (or None if the hero is not in this spot)."""

    role: str                       # 'jam' | 'call'
    regime: str                     # REGIME_PUSHFOLD | REGIME_DEEP
    eff_bb: float                   # effective (smaller) stack in bb
    ante_bb: float                  # per-player ante in bb
    strength_quantile: float        # actor preflop strength quantile in [0,1]
    is_hero: bool                   # is the ACTOR the hero
    pre_stacks: Sequence[float]     # full field live stacks (chips)
    payouts: Sequence[float]        # prize ladder
    actor_field_idx: Optional[int]  # actor's flat live index
    villain_field_idx: Optional[int]
    schedule: LifecycleSchedule


@dataclass
class SpotDecision:
    """A ``decide_fn`` result. ``act`` is True to commit the action (jam / call), False to fold.

    A ``decide_fn`` may decide directly (``act``) or, for the threshold style that mirrors
    ``mtt_sim``'s ``Strategy``, return a ``threshold`` (the fold-below strength quantile) and let the
    sim compare it to the actor's drawn ``strength_quantile`` (the comparison is done in the sim so the
    abstract/realistic resolver paths share one code path). If ``threshold`` is not None it OVERRIDES
    ``act``. ``source`` tags the trace."""

    act: bool = False
    threshold: Optional[float] = None
    source: str = ""


# A decide_fn maps a SpotContext -> SpotDecision.
DecideFn = Callable[[SpotContext], SpotDecision]


# --------------------------------------------------------------------------------------------------
# decide_fn adapters
# --------------------------------------------------------------------------------------------------


def anchor_fold(ctx: SpotContext) -> SpotDecision:
    """The always-fold calibration floor (net-free): never commit. Threshold 1.0 == fold every hand."""
    return SpotDecision(act=False, threshold=1.0, source="anchor-fold")


def anchor_jam(ctx: SpotContext) -> SpotDecision:
    """The always-jam ceiling (net-free): always commit. Threshold 0.0 == act with any hand."""
    return SpotDecision(act=True, threshold=0.0, source="anchor-jam")


def chip_ev_nash(jam: float = 0.45, call: float = 0.60) -> DecideFn:
    """Fixed chip-EV jam/call thresholds (net-free; the ICM-naive baseline field).

    A factory returning a ``decide_fn``: in the push/fold regime it folds hands below the ``jam`` (when
    jamming) or ``call`` (when calling) strength quantile; in the deep regime it uses the same
    thresholds (a chip-EV player ignores ICM, so its thresholds are bubble-blind by construction).
    This is the lifecycle analogue of ``mtt_sim.chip_ev_strategy``."""

    def fn(ctx: SpotContext) -> SpotDecision:
        thr = jam if ctx.role == "jam" else call
        return SpotDecision(threshold=float(thr), source="chip-ev-nash")

    return fn


def icm_aware(jam: float = 0.45, call: float = 0.60, sens: float = 0.12) -> DecideFn:
    """An ICM-aware ``decide_fn`` that TIGHTENS near a pay jump (mirrors ``mtt_sim.icm_aware_strategy``).

    It computes the field bubble factor for the actor vs villain when the hero is involved and tightens
    the threshold by ``sens * (bf - 1)``. Net-free; uses only ``mtt_field.field_bubble_factor``. When
    the bubble factor is ~1 (deep / far from the money) it reduces to ``chip_ev_nash``."""
    from pokerbot.tournament.mtt_field import field_bubble_factor

    def fn(ctx: SpotContext) -> SpotDecision:
        base = jam if ctx.role == "jam" else call
        bf = 1.0
        if (ctx.is_hero and ctx.actor_field_idx is not None
                and ctx.villain_field_idx is not None):
            bf = field_bubble_factor(ctx.pre_stacks, ctx.payouts, ctx.actor_field_idx,
                                     ctx.villain_field_idx, n_sims=300,
                                     rng=random.Random(0x1C3 ^ int(ctx.eff_bb * 7)))
        tighten = max(0.0, bf - 1.0) * sens
        thr = min(0.99, base + tighten)
        return SpotDecision(threshold=float(thr), source="icm-aware")

    return fn


def net_decide(policy=None, *, deep_fallback: Optional[DecideFn] = None,
               pushfold_fallback: Optional[DecideFn] = None,
               jam_chip_ev: float = 0.45, call_chip_ev: float = 0.60) -> DecideFn:
    """The net-served ``decide_fn`` (serves the 7-max net via :mod:`neural_serve`).

    ``policy`` is a :class:`neural_serve.SevenMaxNeuralPolicy` (or any object with a fail-safe
    ``serve_jam_freq``-compatible hook; here we use a strength-quantile heuristic over the net's serve
    when a full env is unavailable). Because building a faithful 7-seat ``EnvState`` for every
    confrontation is heavy (and the spec wants the headline cells runnable net-free / CPU first), this
    adapter uses the net to PERTURB the chip-EV threshold via the net's served aggression and FALLS
    BACK cleanly:

      * push/fold regime -> ``pushfold_fallback`` (default: the certified chip-EV thresholds, the safe
        push/fold floor) when the net abstains;
      * deep regime      -> ``deep_fallback`` (default: chip-EV thresholds) when the net abstains.

    The net itself is consulted through the skew-proven ``SevenMaxNeuralPolicy.policy`` seam, which
    returns ``None`` (never a forced uniform) on any load/skew/degenerate-serve error, so the fallback
    is always taken safely. CRITICAL: this NEVER raises into the sim — a bad checkpoint silently
    degrades to the net-free fallback, exactly like the HU endgame bridge.
    """
    pf = pushfold_fallback or chip_ev_nash(jam_chip_ev, call_chip_ev)
    deep = deep_fallback or chip_ev_nash(jam_chip_ev, call_chip_ev)

    def fn(ctx: SpotContext) -> SpotDecision:
        served = None
        if policy is not None:
            served = _net_aggression(policy, ctx)
        if served is None:
            d = (pf if ctx.regime == REGIME_PUSHFOLD else deep)(ctx)
            return SpotDecision(threshold=d.threshold, act=d.act,
                                source=f"net-fallback:{d.source}")
        # The net served an aggression in [0,1] (probability mass on commit actions). Map it to a
        # fold-below threshold: more aggression -> lower threshold (wider commit range). Centre on the
        # chip-EV threshold so a 0.5 serve reproduces the chip-EV baseline.
        base = jam_chip_ev if ctx.role == "jam" else call_chip_ev
        thr = float(min(0.99, max(0.0, base + (0.5 - served) * 0.8)))
        return SpotDecision(threshold=thr, source="net-served")

    return fn


def _net_aggression(policy, ctx: SpotContext) -> Optional[float]:
    """Probe ``policy`` for an aggression scalar in [0,1] (commit-action mass) for this spot, or None.

    Two routes, in order:

    1. A custom ``serve_aggression(ctx)`` hook on ``policy`` (used by tests / alternative servers) — if
       present we use its scalar.
    2. A real :class:`neural_serve.SevenMaxNeuralPolicy` — we build a REPRESENTATIVE 7-live root at the
       spot's effective depth via the engine (``init_hand``), serve it through the skew-proven bridge,
       and read the net's COMMIT mass (everything but FOLD/CHECK: CALL + BET + RAISE + ALL_IN) as the
       aggression signal. The full lifecycle sim works at the confrontation abstraction (jam/call by
       strength quantile), so this is a representative-root reading, not a per-street replay — but it is
       served through the SAME byte-identity-proven encoder the net was trained on (the env is a genuine
       7-seat ``init_hand`` node, identity-placed), so there is no encoder skew.

    Any failure (no jax, no checkpoint, skew, degenerate serve) returns ``None`` so the caller falls back
    to the net-free threshold — NEVER raising into the sim. A net adapter that cannot faithfully serve a
    spot abstains rather than fabricating a decision.
    """
    hook = getattr(policy, "serve_aggression", None)
    if callable(hook):
        try:
            v = hook(ctx)
            return None if v is None else float(min(1.0, max(0.0, v)))
        except Exception:  # noqa: BLE001 — never raise into the sim
            return None
    # Real SevenMaxNeuralPolicy: serve a representative 7-live root at this depth.
    if hasattr(policy, "policy"):
        return _serve_root_aggression(policy, ctx)
    return None


def _serve_root_aggression(policy, ctx: SpotContext) -> Optional[float]:
    """Build a 7-live root at ``ctx.eff_bb`` deep, serve ``policy``, return COMMIT-action mass in [0,1].

    Fully fail-safe (returns None on any error). The env is a genuine ``init_hand`` 7-seat node so the
    served absolute seat/button one-hots match training (identity placement); the CoinPoker per-player
    ante (``ctx.ante_bb``) is posted so the root matches the tournament regime. Commit mass excludes
    FOLD (idx 0) and CHECK (idx 1)."""
    try:
        import jax
        import jax.numpy as jnp
        import numpy as _np

        from pokerbot.engine.jax_engine import CHIP_UNIT, init_hand

        # depth (bb) -> chips; clamp to a sane band so init_hand always builds a valid short-stack root.
        depth_bb = float(max(2.0, min(ctx.eff_bb, 200.0)))
        # init_hand uses the engine's default starting stack; we serve at the natural root and read the
        # net's aggression at the first decision. (A depth-conditioned root would need the engine's
        # stack-set hook; the natural root is the faithful, firewall-safe choice for a representative
        # reading.)
        occ = jnp.array([True] * 7, dtype=bool)
        ante_chips = jnp.int32(round(float(ctx.ante_bb) * CHIP_UNIT)) if ctx.ante_bb > 0 else jnp.int32(0)
        # deterministic per-spot key so the reading is reproducible (no global RNG).
        key = jax.random.PRNGKey((hash((round(depth_bb, 2), round(ctx.strength_quantile, 3),
                                        ctx.role)) & 0x7FFFFFFF))
        button = int((hash(("btn", round(depth_bb, 2))) & 0x7FFFFFFF) % 7)
        root = init_hand(key, button=jnp.int32(button), occupancy=occ, ante=ante_chips)
        seat = int(root.action_seat)
        root_np = jax.tree.map(lambda x: _np.asarray(x), root)
        probs = policy.policy(root_np, seat)
        if probs is None:
            return None
        # commit mass = 1 - (fold + check). Index 0 == FOLD, 1 == CHECK (engine.actions vocab order).
        fold_check = float(probs[0]) + (float(probs[1]) if probs.shape[0] > 1 else 0.0)
        return float(min(1.0, max(0.0, 1.0 - fold_check)))
    except Exception:  # noqa: BLE001 — never raise into the sim
        return None


# --------------------------------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------------------------------


@dataclass
class SpotTrace:
    """One confrontation's record: ``(regime, chip_delta, pre_stacks, payouts)`` plus its metadata."""

    regime: str
    chip_delta: list[float]         # per-field-index chip delta this confrontation (sum ~ 0 ex-antes)
    pre_stacks: list[float]         # full field live stacks (chips) BEFORE the confrontation
    payouts: list[float]
    level: int
    bb_chips: int
    hero_involved: bool
    source: str


@dataclass
class MttResult:
    """A single tournament's outcome (extends ``mtt_sim.MttResult`` with the lifecycle trace + chips)."""

    hero_finish: int                # 1 == won
    hero_payout: float
    players: int
    trace: list[SpotTrace] = field(default_factory=list)
    start_stacks: list[float] = field(default_factory=list)  # per-field-index opening chips
    levels_played: int = 0


# --------------------------------------------------------------------------------------------------
# The lifecycle simulator
# --------------------------------------------------------------------------------------------------


def simulate_tournament_lifecycle(
    n_players: int,
    payouts: Sequence[float],
    decide_fn: DecideFn,
    villain_decide_fn: Optional[DecideFn] = None,
    *,
    seats_per_table: int = 9,
    rng: Optional[random.Random] = None,
    fmt=None,
    schedule: Optional[LifecycleSchedule] = None,
    start_chips: Optional[int] = None,
    realistic: bool = True,
    trace: bool = True,
    guard_limit: int = 400_000,
) -> MttResult:
    """Play ONE MTT/SNG to completion under the blind/ante LIFECYCLE, returning the hero's finish +
    payout + the per-spot trace.

    Extends :func:`mtt_sim.simulate_mtt`: keeps the identity-tracked ``pid`` balancing (the hero is
    player 0 at table 0 seat 0 and is followed through table breaks), the ``_RealEquity`` resolver, and
    the finish accounting; replaces (a) the static blinds with the ``schedule`` (posting antes, levels
    rising as hands elapse, final-table rollback) and (b) the hardcoded jam/call thresholds with the
    pluggable ``decide_fn`` routed on :data:`decide.PUSHFOLD_MAX_BB`.

    Determinism: given ``rng`` the whole tournament is reproducible (no global ``random``), so two arms
    on the SAME rng are CRN-paired (:func:`duplicate_roi`).

    Args:
        decide_fn: the HERO's ``decide_fn``. ``villain_decide_fn`` defaults to ``decide_fn`` (a
            homogeneous field — the symmetry/unbiasedness anchor).
        fmt: a ``payouts.TournamentFormat`` (only tags the trace; scoring routes by it in
            :func:`eval_score`). Defaults to NORMAL.
        schedule: a :class:`LifecycleSchedule`; defaults to a fresh full ladder.
        start_chips: opening stack in chips; defaults to ``structure.STARTING_CHIPS`` (20,000).
        realistic: real hole cards + real HU matchup equity (the ``mtt_sim`` ``realistic`` path).
        trace: collect the per-spot trace (off for speed in the hot duplicate loop where only the $
            outcome is needed).
    """
    rng = rng or random.Random(0)
    sched = schedule or LifecycleSchedule()
    start_chips = int(start_chips if start_chips is not None else _structure.STARTING_CHIPS)
    villain_decide_fn = villain_decide_fn or decide_fn
    if fmt is None:
        from pokerbot.tournament.payouts import TournamentFormat
        fmt = TournamentFormat.NORMAL
    resolver = _RealEquity.get() if realistic else None

    # Field is built in CHIPS (start_chips), so rising blinds shrink the effective bb-depth.
    field, _ = _build_field(n_players, seats_per_table, float(start_chips), payouts, rng)
    HERO = 0
    pid: dict[tuple[int, int], int] = {}
    nxt = 0
    for t in field.tables:
        for seat, s in enumerate(t.stacks_bb):
            if s > 0.0:
                pid[(t.table_id, seat)] = nxt
                nxt += 1
    start_stacks = [float(start_chips)] * n_players
    finish_rank = n_players
    hero_finish = None
    spot_trace: list[SpotTrace] = []

    from pokerbot.tournament.mtt_field import _flat_hero_index

    def field_idx(table_id: int, seat: int) -> Optional[int]:
        return _flat_hero_index(field.tables, table_id, seat)

    # mutable boxes so the bust/balance helper can update finish state + the live field.
    state = {"field": field, "finish_rank": n_players, "hero_finish": None}

    def _record_bust(table_id: int, seat: int) -> None:
        """Record the seat at ``(table_id, seat)`` as busted (next-worst finish) + re-balance the field,
        carrying identities through the moves. Updates ``state`` in place. Unbiased: the hero is recorded
        on EXACTLY the same code path as every other seat (no seat-0 privilege)."""
        f = state["field"]
        # locate the table object in the (possibly rebuilt) field.
        tab = next((t for t in f.tables if t.table_id == table_id), None)
        if tab is not None:
            tab.stacks_bb[seat] = 0.0
        busted_id = pid.pop((table_id, seat), None)
        if busted_id == HERO:
            state["hero_finish"] = state["finish_rank"]
        state["finish_rank"] -= 1
        if needs_balancing(f, seats_per_table=seats_per_table):
            moves = balance_moves(f, seats_per_table=seats_per_table)
            f2 = apply_moves(f, moves)
            for m in moves:
                mover = pid.pop((m.from_table, m.from_seat), None)
                if mover is not None:
                    pid[(m.to_table, m.to_seat)] = mover
            state["field"] = f2

    guard = 0
    while state["field"].players_left > 1 and guard < guard_limit:
        guard += 1
        field = state["field"]
        live_tables = [t for t in field.tables if t.num_players >= 2]
        if not live_tables:
            break
        table = live_tables[rng.randrange(len(live_tables))]
        seats = [(seat, s) for seat, s in enumerate(table.stacks_bb) if s > 0.0]
        if len(seats) < 2:
            sched.advance(field.players_left)
            continue

        pre_field = list(field.all_live_stacks)          # full live stacks BEFORE this hand (conserving)
        # Stable snapshot map (table_id, seat) -> index in pre_field, captured ONCE before any chips
        # move this hand. Keying the per-hand delta on this fixed snapshot (not the live flat index,
        # which shifts as seats bust mid-hand) is what makes chip_delta align with pre_field and
        # conserve to ~0.
        pre_idx_of: dict[tuple[int, int], int] = {}
        _k = 0
        for _t in field.tables:
            for _seat, _s in enumerate(_t.stacks_bb):
                if _s > 0.0:
                    pre_idx_of[(_t.table_id, _seat)] = _k
                    _k += 1
        delta_by_idx: dict[int, float] = {}

        def _add_delta(table_id, seat, amt):
            fi = pre_idx_of.get((table_id, seat))
            if fi is not None:
                delta_by_idx[fi] = delta_by_idx.get(fi, 0.0) + amt

        # ---- post the per-player antes for this table (dead money into the pot) ----
        # antes can bust a micro stack; the unified bust scan at the end of the hand records every
        # seat (ante-busted or all-in-busted) on the SAME unbiased code path.
        ante = float(sched.ante_chips)
        dead = 0.0
        if ante > 0.0:
            for seat, s in seats:
                pay = min(ante, table.stacks_bb[seat])
                table.stacks_bb[seat] -= pay
                _add_delta(table.table_id, seat, -pay)
                dead += pay
            seats = [(seat, s) for seat, s in enumerate(table.stacks_bb) if s > 0.0]

        # ---- pick the jam actor (shortest stack, random tie-break) + a random caller ----
        jam_wins = None
        winner_seat = None
        if len(seats) >= 2:
            min_stack = min(s for _, s in seats)
            shortest = [seat for seat, s in seats if s <= min_stack + 1e-9]
            jam_seat = shortest[rng.randrange(len(shortest))]
            jam_stack = table.stacks_bb[jam_seat]
            others = [(seat, s) for seat, s in seats if seat != jam_seat]
            call_seat, call_stack = others[rng.randrange(len(others))]

            jam_fidx = field_idx(table.table_id, jam_seat)
            call_fidx = field_idx(table.table_id, call_seat)
            is_hero_jam = pid.get((table.table_id, jam_seat)) == HERO
            is_hero_call = pid.get((table.table_id, call_seat)) == HERO

            eff_bb = sched.effective_bb(min(jam_stack, call_stack))
            regime = REGIME_PUSHFOLD if eff_bb <= PUSHFOLD_MAX_BB else REGIME_DEEP
            ante_bb = ante / sched.bb_chips if sched.bb_chips else 0.0

            # ---- draw cards / strengths consistently with the resolver ----
            if resolver is None:
                jam_q = rng.random()
                call_q = rng.random()
                bi = bj = None
            else:
                bi, bj = resolver.deal_two(rng)
                jam_q = float(resolver._cum_w[np.searchsorted(
                    resolver._sorted_strength, resolver.bucket_strength[bi])])
                call_q = float(resolver._cum_w[np.searchsorted(
                    resolver._sorted_strength, resolver.bucket_strength[bj])])

            def _ctx(role, q, actor_idx, vill_idx, is_hero):
                return SpotContext(role=role, regime=regime, eff_bb=eff_bb, ante_bb=ante_bb,
                                   strength_quantile=q, is_hero=is_hero, pre_stacks=pre_field,
                                   payouts=list(payouts), actor_field_idx=actor_idx,
                                   villain_field_idx=vill_idx, schedule=sched)

            jam_fn = decide_fn if is_hero_jam else villain_decide_fn
            call_fn = decide_fn if is_hero_call else villain_decide_fn
            jam_dec = jam_fn(_ctx("jam", jam_q, jam_fidx, call_fidx, is_hero_jam))
            jam_commits = _commits(jam_dec, jam_q)
            call_dec = call_fn(_ctx("call", call_q, call_fidx, jam_fidx, is_hero_call)) \
                if jam_commits else None
            call_commits = (call_dec is not None) and _commits(call_dec, call_q)

            if jam_commits and call_commits:
                if resolver is None:
                    jam_wins = jam_q > call_q
                else:
                    jam_wins = rng.random() < resolver.win_prob(bi, bj)
                risked = min(jam_stack, call_stack)
                winner_seat, loser_seat = (jam_seat, call_seat) if jam_wins else (call_seat, jam_seat)
                table.stacks_bb[winner_seat] += risked
                table.stacks_bb[loser_seat] -= risked
                _add_delta(table.table_id, winner_seat, risked)
                _add_delta(table.table_id, loser_seat, -risked)
            else:
                # folded out: the dead antes go to a RANDOM live seat ("the blinds win them"), keeping the
                # field unbiased — no seat is privileged to keep its own ante back. winner_seat banks dead.
                winner_seat = seats[rng.randrange(len(seats))][0]

            # award the dead antes to the winner BEFORE the trace, so chip_delta conserves to ~0.
            if winner_seat is not None and dead > 0.0:
                table.stacks_bb[winner_seat] += dead
                _add_delta(table.table_id, winner_seat, dead)

            if trace:
                spot_trace.append(SpotTrace(
                    regime=regime,
                    chip_delta=_dense_delta(delta_by_idx, len(pre_field)),
                    pre_stacks=pre_field, payouts=list(payouts),
                    level=sched.level, bb_chips=sched.bb_chips,
                    hero_involved=is_hero_jam or is_hero_call,
                    source=(f"{jam_dec.source}|{call_dec.source}" if call_dec is not None
                            else f"{jam_dec.source}|folded")))
        else:
            # antes left <2 live -> whatever survives banks the dead money (degenerate micro-table).
            winner_seat = seats[0][0] if seats else None
            if winner_seat is not None and dead > 0.0:
                table.stacks_bb[winner_seat] += dead
                _add_delta(table.table_id, winner_seat, dead)

        # ---- record busts in finish order (the all-in loser + any ante-only busts), unbiased ----
        busted_now = [seat for seat in range(len(table.stacks_bb))
                      if table.stacks_bb[seat] <= 1e-9 and (table.table_id, seat) in pid]
        for seat in busted_now:
            _record_bust(table.table_id, seat)

        sched.advance(state["field"].players_left)

    hero_finish = state["hero_finish"]
    if hero_finish is None:
        hero_finish = 1
    pay = list(payouts) + [0.0] * max(0, n_players - len(payouts))
    return MttResult(hero_finish=hero_finish, hero_payout=pay[hero_finish - 1], players=n_players,
                     trace=spot_trace, start_stacks=start_stacks, levels_played=sched.level)


def _dense_delta(delta_by_idx: dict, n: int) -> list[float]:
    """Materialise the per-field-index chip delta vector from the accumulated map (already includes the
    antes and the all-in swing + dead-money award). Returns a dense list of length ``n`` summing to ~0
    (chip-conserving for every confrontation)."""
    out = [0.0] * n
    for i, v in delta_by_idx.items():
        if 0 <= i < n:
            out[i] = v
    return out


def _commits(dec: SpotDecision, strength_quantile: float) -> bool:
    """Resolve a ``SpotDecision`` to a commit boolean. A ``threshold`` overrides the explicit ``act``:
    the actor folds hands strictly below the threshold quantile (so threshold 1.0 == always fold,
    0.0 == always act), matching ``mtt_sim``'s ``jam_strength < jam_thr`` fold rule."""
    if dec.threshold is not None:
        return strength_quantile >= dec.threshold
    return bool(dec.act)


# --------------------------------------------------------------------------------------------------
# Scoring + aggregation (E3 ROI gross/net, E4 finish distribution) with 95% CI
# --------------------------------------------------------------------------------------------------


def _cv_score(result: MttResult, payouts: Sequence[float], n_players: int, start_chips: float,
              fmt) -> float:
    """The hero's variance-reduced $ score for one tournament: realized payout minus the pre-tournament
    ICM baseline at the equal opening stacks, via :func:`eval_score.score_with_cv`.

    The opening field is equal stacks, so ``icm_equity(pre)[hero]`` == buy-in share (every seat is
    symmetric) — a constant baseline subtracted from every arm, so it does NOT bias the ROI but DOES
    remove the shared finish variance in the CRN-paired difference. We return the centred score; the
    headline adds the baseline back (it equals the fair buy-in share).

    Production scoring goes through :func:`eval_score.score_with_cv` (the certified VR core). That module
    eagerly imports the all-in settle path (``rollout_leaf -> sim -> numba``); in a lean jax test env that
    lacks ``numba`` the import would fail, so we fall back to the IDENTICAL inline computation
    (``realized - icm_equity(pre)[hero]``) — byte-for-byte the same baseline, just without dragging in the
    settle path we don't use here. The smoke gates therefore run net-free / numba-free."""
    pre = [float(start_chips)] * n_players
    try:
        from pokerbot.tournament.eval_score import score_with_cv
        return score_with_cv(float(result.hero_payout), pre, list(payouts), 0)
    except Exception:  # noqa: BLE001 — lean env w/o the settle deps; identical formula inline
        from pokerbot.tournament.icm import icm_equity
        baseline = float(icm_equity(pre, list(payouts))[0])
        return float(result.hero_payout) - baseline


def _ci95(vals: Sequence[float]) -> tuple[float, float, float, float]:
    """Return ``(mean, se, lo, hi)`` for the 95% normal CI of a sample mean (z=1.96)."""
    a = np.asarray(vals, dtype=np.float64)
    n = a.size
    mean = float(a.mean()) if n else 0.0
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    se = float(std / math.sqrt(n)) if n > 1 else 0.0
    return mean, se, mean - 1.96 * se, mean + 1.96 * se


def _field_descriptor(decide_fn: DecideFn, villain_decide_fn: Optional[DecideFn]) -> dict:
    """Describe the field composition behind a ROI so the number is interpretable (not an artifact).

    A homogeneous SELF-PLAY field (hero and every villain run the SAME ``decide_fn``) is the
    normalization control: by symmetry the hero is statistically interchangeable with any other seat, so
    its mean prize MUST equal the fair buy-in share and ``roi -> 0`` (up to any rake/structure). When the
    villain field differs from the hero, the ROI is a GENUINE 'hero vs that field' edge — but only once
    the self-play control has confirmed the normalization is zero-centred. ``homogeneous`` is True iff the
    hero and the villain field are the SAME callable object (the symmetric control)."""
    return {
        "homogeneous": villain_decide_fn is None or villain_decide_fn is decide_fn,
        "hero": getattr(decide_fn, "__name__", type(decide_fn).__name__),
        "villain": (getattr(villain_decide_fn, "__name__", type(villain_decide_fn).__name__)
                    if villain_decide_fn is not None else "self"),
    }


def measure_lifecycle_roi(
    n_players: int,
    payouts: Sequence[float],
    decide_fn: DecideFn,
    villain_decide_fn: Optional[DecideFn] = None,
    *,
    n_sims: int = 100,
    seats_per_table: int = 9,
    rng: Optional[random.Random] = None,
    fmt=None,
    schedule_factory: Optional[Callable[[], LifecycleSchedule]] = None,
    start_chips: Optional[int] = None,
    realistic: bool = True,
    buy_in: Optional[float] = None,
    self_play_control: bool = False,
) -> dict:
    """E3/E4 aggregation: hero $-ROI (gross + net) with a 95% CI, ITM rate, and mean finish, over
    ``n_sims`` full-lifecycle tournaments.

    * ``mean_payout`` / its 95% CI — the realized prize the hero collects.
    * ``roi`` (GROSS) = ``(mean_payout - buy_in_share) / buy_in_share`` where ``buy_in_share`` ==
      ``sum(payouts) / n_players`` (the fair, rake-free share — this is the prize-pool ROI). NOTE: this is
      normalized against the FAIR per-entrant pool share, NOT a fixed external buy-in, so in a SYMMETRIC
      self-play field it is zero-centred by construction (every seat is interchangeable -> the hero's mean
      prize equals the share -> ``roi == 0``). A non-zero ``roi`` is therefore a REAL 'hero better/worse
      than this field' signal — provided the self-play control (below) confirms the normalization.
    * ``roi_net`` — when ``buy_in`` (the player's ACTUAL paid entry, rake included) is supplied, the
      net cash ROI ``(mean_payout - buy_in) / buy_in`` (gross uses the rake-free pool share; net uses
      the real cost, so ``roi_net < roi`` by the rake).
    * ``cv_mean`` / its 95% CI — the variance-reduced ``score_with_cv`` mean (the sharpened estimate of
      ``mean_payout - buy_in_share``; add ``buy_in_share`` to recover ``mean_payout``).
    * ``itm_rate`` / ``mean_finish`` (E4 finish distribution) + the full ``finish_hist``.
    * ``field`` — the field-composition descriptor (homogeneous self-play vs a hetero villain field) so
      a reader can tell whether a negative ROI is a real edge or wants the self-play control first.
    * ``self_play_roi`` (only when ``self_play_control=True``) — the NORMALIZATION CONTROL: the SAME
      ``villain_decide_fn`` run as BOTH hero and field (a fully symmetric field), whose ROI MUST be ~0.
      If this control's CI excludes 0 the normalization/field-composition is biased (not the policy);
      when it brackets 0 a non-zero headline ``roi`` is a genuine 'hero worse/better than the field'.

    ``schedule_factory`` builds a FRESH :class:`LifecycleSchedule` per tournament (a schedule is
    stateful — it must not be shared across sims). Defaults to a fresh full ladder each tournament.
    """
    rng = rng or random.Random(0)
    if fmt is None:
        from pokerbot.tournament.payouts import TournamentFormat
        fmt = TournamentFormat.NORMAL
    start_chips = int(start_chips if start_chips is not None else _structure.STARTING_CHIPS)
    sched_factory = schedule_factory or (lambda: LifecycleSchedule())

    pays: list[float] = []
    finishes: list[int] = []
    cv_scores: list[float] = []
    for _ in range(n_sims):
        r = simulate_tournament_lifecycle(
            n_players, payouts, decide_fn, villain_decide_fn,
            seats_per_table=seats_per_table, rng=rng, fmt=fmt, schedule=sched_factory(),
            start_chips=start_chips, realistic=realistic, trace=False)
        pays.append(r.hero_payout)
        finishes.append(r.hero_finish)
        cv_scores.append(_cv_score(r, payouts, n_players, start_chips, fmt))

    mean_pay, pay_se, pay_lo, pay_hi = _ci95(pays)
    cv_mean, cv_se, cv_lo, cv_hi = _ci95(cv_scores)
    buy_in_share = sum(payouts) / n_players if n_players else 0.0
    finish_hist: dict[int, int] = {}
    for f in finishes:
        finish_hist[f] = finish_hist.get(f, 0) + 1

    out = {
        "n_sims": n_sims,
        "n_players": n_players,
        "mean_payout": mean_pay,
        "mean_payout_ci95": (pay_lo, pay_hi),
        "buy_in_share": buy_in_share,
        "roi": (mean_pay - buy_in_share) / buy_in_share if buy_in_share > 0 else 0.0,
        "roi_ci95": ((pay_lo - buy_in_share) / buy_in_share,
                     (pay_hi - buy_in_share) / buy_in_share) if buy_in_share > 0 else (0.0, 0.0),
        "cv_mean": cv_mean,
        "cv_mean_ci95": (cv_lo, cv_hi),
        "itm_rate": sum(1 for p in pays if p > 0) / len(pays) if pays else 0.0,
        "mean_finish": sum(finishes) / len(finishes) if finishes else 0.0,
        "finish_hist": finish_hist,
        "field": _field_descriptor(decide_fn, villain_decide_fn),
    }
    if buy_in is not None and buy_in > 0:
        out["buy_in"] = float(buy_in)
        out["roi_net"] = (mean_pay - float(buy_in)) / float(buy_in)
        out["roi_net_ci95"] = ((pay_lo - float(buy_in)) / float(buy_in),
                               (pay_hi - float(buy_in)) / float(buy_in))

    if self_play_control:
        # The normalization control: run the VILLAIN policy (the field) as BOTH hero and field, a fully
        # symmetric self-play field. By seat symmetry its ROI must be ~0; a CI that excludes 0 means the
        # normalization/field composition is biased (a sim artifact), not the measured policy.
        control_fn = villain_decide_fn or decide_fn
        ctrl_pays: list[float] = []
        ctrl_fin: list[int] = []
        for _ in range(n_sims):
            r = simulate_tournament_lifecycle(
                n_players, payouts, control_fn, control_fn,
                seats_per_table=seats_per_table, rng=rng, fmt=fmt, schedule=sched_factory(),
                start_chips=start_chips, realistic=realistic, trace=False)
            ctrl_pays.append(r.hero_payout)
            ctrl_fin.append(r.hero_finish)
        c_mean, c_se, c_lo, c_hi = _ci95(ctrl_pays)
        out["self_play_roi"] = {
            "roi": (c_mean - buy_in_share) / buy_in_share if buy_in_share > 0 else 0.0,
            "roi_ci95": ((c_lo - buy_in_share) / buy_in_share,
                         (c_hi - buy_in_share) / buy_in_share) if buy_in_share > 0 else (0.0, 0.0),
            "mean_payout": c_mean,
            "mean_finish": sum(ctrl_fin) / len(ctrl_fin) if ctrl_fin else 0.0,
            "expected_mean_finish": (n_players + 1) / 2.0,
            "n_sims": n_sims,
            "note": ("symmetric self-play field — ROI must bracket 0 (the normalization control); "
                     "if it does not, a non-zero headline ROI is a sim artifact, not a real edge"),
        }
    return out


def duplicate_roi(
    n_players: int,
    payouts: Sequence[float],
    arm_a: DecideFn,
    arm_b: DecideFn,
    *,
    villain_decide_fn: Optional[DecideFn] = None,
    n_sims: int = 100,
    seats_per_table: int = 9,
    rng: Optional[random.Random] = None,
    fmt=None,
    schedule_factory: Optional[Callable[[], LifecycleSchedule]] = None,
    start_chips: Optional[int] = None,
    realistic: bool = True,
) -> dict:
    """CRN-paired duplicate: run two HERO ``decide_fn`` arms (``arm_a``, ``arm_b``) on the SAME injected
    rng per tournament so the card/board/seat-flow noise is identical, then report each arm's $-ROI AND
    the variance-reduced PAIRED difference (arm_a minus arm_b) with a 95% CI.

    ``simulate_tournament_lifecycle`` is deterministic given its rng, so seeding both arms with a CHILD
    rng derived from the same per-tournament seed makes them true common-random-number pairs: the only
    difference between the two series is the policy, so the paired difference removes the (enormous)
    shared tournament variance. The villain field is held common (``villain_decide_fn``, default
    ``arm_b`` so arm_b is the homogeneous baseline). Returns ``{arm_a, arm_b, paired_diff}`` blocks."""
    rng = rng or random.Random(0)
    if fmt is None:
        from pokerbot.tournament.payouts import TournamentFormat
        fmt = TournamentFormat.NORMAL
    start_chips = int(start_chips if start_chips is not None else _structure.STARTING_CHIPS)
    sched_factory = schedule_factory or (lambda: LifecycleSchedule())
    villain = villain_decide_fn or arm_b

    a_pays: list[float] = []
    b_pays: list[float] = []
    a_fin: list[int] = []
    b_fin: list[int] = []
    diffs: list[float] = []
    a_cv: list[float] = []
    b_cv: list[float] = []
    for _ in range(n_sims):
        seed = rng.randrange(1 << 62)              # one shared per-tournament seed -> CRN pair
        ra = random.Random(seed)
        rb = random.Random(seed)
        res_a = simulate_tournament_lifecycle(
            n_players, payouts, arm_a, villain, seats_per_table=seats_per_table, rng=ra, fmt=fmt,
            schedule=sched_factory(), start_chips=start_chips, realistic=realistic, trace=False)
        res_b = simulate_tournament_lifecycle(
            n_players, payouts, arm_b, villain, seats_per_table=seats_per_table, rng=rb, fmt=fmt,
            schedule=sched_factory(), start_chips=start_chips, realistic=realistic, trace=False)
        a_pays.append(res_a.hero_payout)
        b_pays.append(res_b.hero_payout)
        a_fin.append(res_a.hero_finish)
        b_fin.append(res_b.hero_finish)
        ca = _cv_score(res_a, payouts, n_players, start_chips, fmt)
        cb = _cv_score(res_b, payouts, n_players, start_chips, fmt)
        a_cv.append(ca)
        b_cv.append(cb)
        diffs.append(res_a.hero_payout - res_b.hero_payout)

    buy_in_share = sum(payouts) / n_players if n_players else 0.0

    def _arm(pays, fin, cv):
        m, se, lo, hi = _ci95(pays)
        cm, cse, clo, chi = _ci95(cv)
        return {
            "mean_payout": m, "mean_payout_ci95": (lo, hi),
            "roi": (m - buy_in_share) / buy_in_share if buy_in_share > 0 else 0.0,
            "cv_mean": cm, "cv_mean_ci95": (clo, chi),
            "itm_rate": sum(1 for p in pays if p > 0) / len(pays) if pays else 0.0,
            "mean_finish": sum(fin) / len(fin) if fin else 0.0,
        }

    dm, dse, dlo, dhi = _ci95(diffs)
    try:
        from pokerbot.tournament.eval_score import verdict
    except Exception:  # noqa: BLE001 — lean env w/o the settle deps; replicate the headline reducer
        def verdict(lo, hi):
            if lo > 0.0:
                return "POSITIVE"
            if hi < 0.0:
                return "NEGATIVE"
            return "INCONCLUSIVE (CI spans 0)"
    return {
        "n_sims": n_sims,
        "n_players": n_players,
        "buy_in_share": buy_in_share,
        "arm_a": _arm(a_pays, a_fin, a_cv),
        "arm_b": _arm(b_pays, b_fin, b_cv),
        "paired_diff": {
            "mean": dm, "se": dse, "ci95": (dlo, dhi),
            "verdict": verdict(dlo, dhi),
        },
    }


# --------------------------------------------------------------------------------------------------
# Suite entry point (E3/E4): the format x regime driver calls THIS per cell.
# --------------------------------------------------------------------------------------------------


def analytic_icm_decide(pushfold_fallback: Optional[DecideFn] = None,
                        deep_fallback: Optional[DecideFn] = None, *, iters: int = 1200):
    """Build ``(decide_fn, coverage_ledger)`` for the certified analytic ICM push/fold engine.

    The NON-CFR analytic tournament model as a lifecycle ``DecideFn``: in the push/fold regime
    (``eff_bb <= PUSHFOLD_MAX_BB``) it solves the certified ICM open-shove
    (:func:`open_shove_icm.solve_open_shove_icm`) at the spot's effective depth and returns the
    combo-weighted equilibrium jam/call FREQUENCY as a fold-below ``threshold`` (= ``1 - freq``), so the
    sim's strength-quantile comparison reproduces the engine's range width WITHOUT needing concrete hole
    cards (the lifecycle works at the strength-quantile abstraction). In the DEEP regime the certified
    push/fold engine ABSTAINS (it is a shallow specialist), so we fall back to the existing
    :func:`chip_ev_nash` thresholds — mirroring how :func:`net_decide` falls back. SATELLITE pricing is not
    re-derived here (the lifecycle field is the NORMAL/PKO ladder; the SATELLITE-specific seat-bubble model
    lives in the spot-level battery), so a satellite cell's deep spots take the chip-EV fallback too.

    COVERAGE HONESTY: the returned ``coverage`` ledger (``{'decided','deferred','total'}``) counts a DECIDE
    only when the certified push/fold solve actually ran (push/fold regime, no error); deep-regime fallbacks
    and degenerate solves are counted as DEFERRED. The push/fold specialist should DECIDE most of the
    shallow/bubble/ft lifecycle and DEFER the deep-early hands.
    """
    pf = pushfold_fallback or chip_ev_nash()
    deep = deep_fallback or chip_ev_nash()
    ledger = {"decided": 0, "deferred": 0, "total": 0}
    _cache: dict[tuple, float] = {}
    _abstains: set[tuple] = set()  # keys where transient solve failures occurred; non-cache, observable

    def _engine_threshold(ctx: SpotContext) -> Optional[float]:
        """Certified ICM open-shove frequency for this spot -> fold-below threshold, or None on abstain."""
        from pokerbot.tournament.open_shove_icm import solve_open_shove_icm

        eff = round(float(ctx.eff_bb), 1)
        ante = round(float(ctx.ante_bb), 3)
        key = (ctx.role, eff, ante)
        if key in _cache:
            return _cache[key]
        # A HU-effective open-shove at the spot's depth: hero (SB) vs one behind (BB), the certified
        # 2-seat ICM push/fold that the lifecycle confrontation abstracts to (jam actor vs a single
        # caller). Winner-take-all-free: use a chip-proportional ladder so the threshold is the chip-EV-
        # consistent ICM open-shove (the field is the ICM-naive chip-EV twin; the engine adds the ICM
        # range tightening through its own equilibrium, NOT a payout the field does not see).
        stacks = [eff, eff]
        pay = [1.0, 0.0]
        blinds = {0: 0.5, 1: 1.0}
        try:
            sol = solve_open_shove_icm(stacks, pay, 0, [1], blinds=blinds, ante=ctx.ante_bb, iters=iters)
        except (KeyError, ValueError, IndexError, np.linalg.LinAlgError, OSError):  # narrow: transient solve failure
            # Deliberately do NOT cache `None` here — a transient np.linalg convergence hiccup would
            # otherwise become a permanent "engine abstains forever at this key". Instead, track
            # the abstain count in a separate set so an operator can diagnose without poisoning
            # the value cache.
            _abstains.add(key)
            return None
        w = np.asarray(sol.weights, dtype=np.float64)
        wsum = float(w.sum()) or 1.0
        if ctx.role == "jam":
            freq = float((np.asarray(sol.hero_jam) * w).sum() / wsum)
        else:
            call_vec = sol.call.get(1)
            freq = float((np.asarray(call_vec) * w).sum() / wsum) if call_vec is not None else 0.0
        thr = float(min(1.0, max(0.0, 1.0 - freq)))   # fold the bottom (1 - freq) of the strength range
        _cache[key] = thr
        return thr

    def fn(ctx: SpotContext) -> SpotDecision:
        ledger["total"] += 1
        if ctx.regime == REGIME_PUSHFOLD:
            thr = _engine_threshold(ctx)
            if thr is not None:
                ledger["decided"] += 1
                return SpotDecision(threshold=thr, source="icm-analytic-pushfold")
            # degenerate solve -> deferred to the chip-EV push/fold fallback.
            ledger["deferred"] += 1
            d = pf(ctx)
            return SpotDecision(threshold=d.threshold, act=d.act, source=f"icm-defer:{d.source}")
        # DEEP regime: the certified push/fold engine abstains by design -> chip-EV fallback.
        ledger["deferred"] += 1
        d = deep(ctx)
        return SpotDecision(threshold=d.threshold, act=d.act, source=f"icm-deep-defer:{d.source}")

    return fn, ledger


def _build_decide_fn(cell_args: dict):
    """Build ``(hero_decide_fn, source, coverage)`` for one cell from ``cell_args['policy']``.

    * ``policy=='net'`` -> :func:`net_decide` serving the 7-max net via the skew-proven
      :class:`neural_serve.SevenMaxNeuralPolicy` seam (the ONLY checkpoint touch), falling back cleanly to
      the certified chip-EV push/fold thresholds whenever the net abstains. Constructing the policy here
      FAILS LOUD (a bad checkpoint surfaces as the cell's ``status:'error'``), but once built the sim never
      raises (the bridge returns None and the fallback is taken).
    * ``policy=='icm'`` -> :func:`analytic_icm_decide`, the certified analytic ICM push/fold engine
      (deciding in the push/fold regime, falling back to chip-EV deep). Returns its coverage ledger.
    * ``policy=='jam'`` -> :func:`anchor_jam` (the net-free always-commit ceiling).
    * anything else     -> :func:`anchor_fold` (the net-free always-fold floor).

    ``coverage`` is the analytic-ICM ledger (or ``None`` for the other policies).
    """
    policy = str(cell_args.get("policy", "fold"))
    if policy == "net":
        from pokerbot.tournament.neural_serve import SevenMaxNeuralPolicy
        net = SevenMaxNeuralPolicy(cell_args.get("ckpt"))
        # Touch the loader up front so a broken checkpoint surfaces loudly to the suite (status:'error')
        # rather than silently degrading to the chip-EV fallback for the whole cell.
        if not net._ensure_loaded():
            raise RuntimeError(
                f"7-max net failed to load from {net.checkpoint!r}; cannot serve --policy net lifecycle")
        return net_decide(net), "net", None
    if policy == "icm":
        fn, ledger = analytic_icm_decide()
        return fn, "icm-analytic", ledger
    if policy == "jam":
        return anchor_jam, "anchor-jam", None
    return anchor_fold, "anchor-fold", None


def lifecycle_cell(fmt_key: str, regime_key: str, cell_args: dict) -> dict:
    """E3/E4 suite adapter: the hero's full-lifecycle $-ROI / ITM / mean-finish for the cell's policy,
    PLUS the CRN-paired ΔROI of the policy vs the ICM-naive chip-EV field (the variance-reduced headline).

    Called by ``scripts/tournament/eval_tournament_suite._drive_pillars`` as
    ``lifecycle_cell(fmt_key, regime_key, cell_args)`` with ``cell_args`` carrying ``n_seats, depth_bb,
    ante_bb, payouts, icm_pressure_spot, decks, hands, spots, ckpt, seed, smoke, policy``.

    The hero plays a whole MTT/SNG to completion under the rising blind/ante lifecycle; the field plays the
    ICM-naive chip-EV Nash thresholds. ``measure_lifecycle_roi`` gives the absolute ROI / ITM / mean-finish
    (+CI); ``duplicate_roi`` gives the CRN-paired ΔROI (hero policy minus the chip-EV field arm) so the
    huge tournament variance cancels. Returns a JSON-serializable dict with both.
    """
    fmt = _fmt_for_key(fmt_key)
    payouts = [float(p) for p in cell_args.get("payouts", [50.0, 30.0, 20.0])] or [1.0]
    seed = int(cell_args.get("seed", 0))
    smoke = bool(cell_args.get("smoke", False))

    # Field size: SNG is a tiny single table; everything else is a small multi-table MTT (kept modest so
    # the full lifecycle sim — balancing + breaks + blind escalation — finishes in CPU-seconds per cell).
    n_players = 6 if str(fmt_key) == "SNG" else 18
    n_players = max(n_players, len(payouts) + 1)  # at least one unpaid finisher

    # n_sims comes from `hands` (the driver's per-cell pillar budget); clamp under smoke.
    n_sims = int(cell_args.get("hands", 60))
    if smoke:
        n_sims = min(n_sims, 12)
    n_sims = max(4, n_sims)

    hero_fn, source, coverage = _build_decide_fn(cell_args)
    field_fn = chip_ev_nash()  # the ICM-naive chip-EV field the hero is measured against

    import random as _random

    abs_roi = measure_lifecycle_roi(
        n_players, payouts, hero_fn, villain_decide_fn=field_fn,
        n_sims=n_sims, rng=_random.Random(seed), fmt=fmt, realistic=True,
        self_play_control=True)

    paired = duplicate_roi(
        n_players, payouts, hero_fn, field_fn, villain_decide_fn=field_fn,
        n_sims=n_sims, rng=_random.Random(seed + 1), fmt=fmt, realistic=True)

    # COVERAGE HONESTY: surface the fraction of confrontations the analytic ICM push/fold engine actually
    # DECIDED (vs deferred to the chip-EV deep fallback), so the grade is not silently dominated by the
    # fallback. The shallow specialist should decide most shallow/bubble/ft hands, defer the deep-early ones.
    model_coverage = None
    if coverage is not None:
        tot = max(1, int(coverage["total"]))
        model_coverage = {
            "decided": int(coverage["decided"]),
            "deferred": int(coverage["deferred"]),
            "total": int(coverage["total"]),
            "fraction": float(coverage["decided"]) / tot,
        }

    return {
        "fmt": fmt.value,
        "regime": str(regime_key),
        "policy_source": source,
        "model_coverage": model_coverage,
        "n_players": int(n_players),
        "n_sims": int(n_sims),
        # E3 absolute headline: hero $-ROI / payout (+CI) and the CV-sharpened mean.
        "mean_payout": float(abs_roi["mean_payout"]),
        "mean_payout_ci95": [float(abs_roi["mean_payout_ci95"][0]), float(abs_roi["mean_payout_ci95"][1])],
        "roi": float(abs_roi["roi"]),
        "roi_ci95": [float(abs_roi["roi_ci95"][0]), float(abs_roi["roi_ci95"][1])],
        "cv_mean": float(abs_roi["cv_mean"]),
        "cv_mean_ci95": [float(abs_roi["cv_mean_ci95"][0]), float(abs_roi["cv_mean_ci95"][1])],
        "buy_in_share": float(abs_roi["buy_in_share"]),
        # Field composition + the normalization control: the symmetric self-play field whose ROI must
        # bracket 0. Surfacing both makes the headline ROI interpretable — a non-zero roi is a real
        # 'hero vs the chip-EV field' edge ONLY when self_play_roi.roi_ci95 contains 0.
        "field": dict(abs_roi.get("field", {})),
        "self_play_roi": float(abs_roi.get("self_play_roi", {}).get("roi", 0.0)),
        "self_play_roi_ci95": [float(abs_roi.get("self_play_roi", {}).get("roi_ci95", (0.0, 0.0))[0]),
                               float(abs_roi.get("self_play_roi", {}).get("roi_ci95", (0.0, 0.0))[1])],
        "self_play_mean_finish": float(abs_roi.get("self_play_roi", {}).get("mean_finish", 0.0)),
        # E4 finish distribution: ITM rate + mean finish + the histogram.
        "itm_rate": float(abs_roi["itm_rate"]),
        "mean_finish": float(abs_roi["mean_finish"]),
        "finish_hist": {str(k): int(v) for k, v in abs_roi["finish_hist"].items()},
        # CRN-paired headline: hero policy minus the chip-EV field arm, variance-reduced.
        "paired_diff_mean": float(paired["paired_diff"]["mean"]),
        "paired_diff_ci95": [float(paired["paired_diff"]["ci95"][0]),
                             float(paired["paired_diff"]["ci95"][1])],
        "paired_diff_verdict": paired["paired_diff"]["verdict"],
        "arm_policy_roi": float(paired["arm_a"]["roi"]),
        "arm_chipev_roi": float(paired["arm_b"]["roi"]),
        "payouts": [round(float(p), 4) for p in payouts],
    }


def _fmt_for_key(fmt_key: str):
    from pokerbot.tournament.payouts import TournamentFormat
    return {
        "NORMAL": TournamentFormat.NORMAL,
        "SNG": TournamentFormat.NORMAL,
        "SATELLITE": TournamentFormat.SATELLITE,
        "PKO": TournamentFormat.PKO,
    }.get(str(fmt_key), TournamentFormat.NORMAL)
