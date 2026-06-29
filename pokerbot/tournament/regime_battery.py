"""E6 CRN twin-policy ΔICM batteries — the regime cells where chip-EV and $-EV diverge most.

THE QUESTION this answers
-------------------------
A tournament is a **$-EV** game, but the strong all-in machinery (the cash net, a chip-EV Nash
push/fold chart) optimizes **chip-EV**. The two coincide deep / far from the money and DIVERGE near
pay jumps — the bubble, the final-table ladder, the satellite seat bubble. This module measures, per
regime cell, how much $-equity (ICM) a genuinely ICM-aware ``hero_policy`` SAVES over the SAME
machinery played ICM-naively (``chipev_twin``): the **twin-policy ΔICM edge**. It is the E6 centerpiece.

WHY A TWIN-POLICY CRN PAIRING (the variance-reduction spine)
------------------------------------------------------------
The brutal variance in any tournament measurement is *card luck* (did hero's jam hold up). We kill it
the way ``cv_absolute`` kills seat luck: a **common-random-numbers (CRN) paired difference**. For each
spot we draw a root, **deal the cards ONCE** (the project deck-seed convention
``deck_seed = 1000 + seed*1_000_000 + d`` — ``exploitation_battery.py:1101``), then run BOTH policies on
the **same deck + same settle_seed** to terminal and map each chip result through
:func:`eval_score.score_chip_to_dollar`. The paired ΔICM ``= score(hero) − score(twin)`` cancels the
shared card luck: when both policies pick the same action the pair is identically 0 (it never enters the
variance), and the residual is only the $-impact of the spots where ICM REORDERS the chip-EV-optimal
action. That residual is what ``uplift_zero_frac`` reports (the CRN-health diagnostic: a healthy cell has
most spots agreeing, a few high-leverage flips driving a tight, significant edge).

THE CHIP-EV SIDE-CHANNEL (proving divergence, not just asserting it)
-------------------------------------------------------------------
Alongside the $-EV ΔICM we accumulate the **chip-EV side-channel**: the same paired difference scored in
CHIPS instead of $. By construction the chip-EV-optimal twin can only ever WIN (weakly) on the chip
column — so a cell where the $ ΔICM is POSITIVE while the chip Δchip is ≤0 is a *proof* that the edge is
ICM-structural, not a chip-EV artifact (hero gives up chip-EV to buy $-equity). The ``flip_rate`` (fraction
of spots where ICM reorders the chip-EV-optimal action) and the summed ``icm_delta_saved`` quantify it.

CELLS (the regimes, each a co-located spot-builder returning (root, pre_stacks, payouts, hero_seat, fmt))
---------------------------------------------------------------------------------------------------------
  * ``bubble_icm``        N just above the money, short hero (the classic ICM bubble).
  * ``payjump`` / ``final_table_icm``  FT ladder at a pay-jump, FT-rollback blinds, premium pressure.
  * ``shortstack_pushfold``  ≤20bb first-in jam/call — the CERTIFIED push/fold zone (anchor-calibratable).
  * ``satellite_bubble``  ``satellite.seat_locked`` borderline, flat seat ladder — the C1 stop-gambling cell.
  * ``deep_nearmoney``    35–50bb near money (chip-EV and $-EV barely diverge — a low-flip control cell).

Each cell draws (depth, ante, level) from :func:`structure.sample_spot` and the STRUCTURE-CORRECT payout
vector from :func:`payouts.payout_vector` / :func:`satellite.satellite_payouts` (NEVER a literal). The
lifecycle root sampler (``sample_root``) lives inline here to keep the core bounded.

THE POLICY PROTOCOL (net-free anchors first, net later)
-------------------------------------------------------
A policy is a callable ``policy(spot) -> str`` returning one of ``{"jam","fold","call"}`` for the spot's
all-in decision (the spot carries ``role`` = "first_in" or "facing_jam"). This is deliberately the SAME
shape as the net-free anchors (always-fold / always-jam) and the chip-EV Nash twin, so the battery is
smoke-tested on CPU with anchors + the chip-EV twin FIRST (calibrate the cells against the anchor floor),
then the CPU net policy is plugged in unchanged. The 7-max net is consulted ONLY through
``neural_serve.make_seven_neural_policy`` (the skew-safe bridge) wrapped to this protocol — this module
NEVER touches the checkpoint directly, so there is no encoder-skew surface here.

Dependency-light: numpy + the certified tournament primitives (``icm``, ``payouts``, ``satellite``,
``structure``, ``eval_score``, ``pushfold_icm`` for the chip-EV Nash twin + the 169-bucket equity grid,
``aof_equity_grids._eval7`` via ``rollout_leaf``/``sim`` for the side-pot settle). NO jax, NO engine
search, NO firewall surface.

Smoke (WSL Debian) — CPU, tiny N, net-free anchors:
  source ~/jaxgpu-venv/bin/activate
  cd .
  PYTHONPATH=. POKERBOT_INFO_SUITS=1 JAX_PLATFORMS=cpu \
    python -m pytest tests/test_regime_battery.py -q
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

_log = logging.getLogger(__name__)

from pokerbot.tournament import payouts as _payouts
from pokerbot.tournament import satellite as _satellite
from pokerbot.tournament import structure as _structure
from pokerbot.tournament.eval_score import score_chip_to_dollar, verdict
from pokerbot.tournament.pushfold_icm import hand_to_label

TournamentFormat = _payouts.TournamentFormat

# Deck-seed convention — IDENTICAL to exploitation_battery.py:1101 / cv_absolute, so a regime cell shares
# the CRN deck-seed namespace with the rest of the eval suite (a card-luck deck is a pure function of
# (seed, spot_index), making the twin pairing automatic and reproducible).
_DECK_SEED_BASE = 1000
_DECK_SEED_SEED_STRIDE = 1_000_000

# Mean-preserving runout count for the equity settle of an unfinished board (the rollout_leaf default).
_DEFAULT_RUNOUTS = 120

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"

# ICM is computed over the LIVE TABLE, not the whole remaining field: Malmuth-Harville is O(n!/(n-k)!),
# so a 30-seat field with 30 paid ranks is intractable (and not how ICM is applied in practice — ICM
# pressure is read off the table hero is sitting at, with the payout ladder carrying the global
# structure). Every cell therefore models a single-table-or-smaller live field, capped here. A
# bubble/pay-jump is the table-local question; the structure-correct GLOBAL ladder (truncated to the live
# seats + the next pay rungs below them) is what makes the ICM premium correct.
_MAX_TABLE_SEATS = 9


def _ladder_for_table(payouts: list[float], seats: int) -> list[float]:
    """The payout ladder a table of ``seats`` live players competes for: the top ``seats`` rungs of the
    STRUCTURE-CORRECT global vector (best-first). Truncating to the live count keeps ICM tractable while
    preserving the real prize shape (the steep top of the ladder is exactly where the ICM premium lives).
    Pads with 0 if the global ladder is shorter (more live seats than paid places => some bust unpaid)."""
    pv = list(payouts)
    if len(pv) >= seats:
        return pv[:seats]
    return pv + [0.0] * (seats - len(pv))


# ======================================================================================== card helpers
def _card_str(cid: int) -> str:
    """52-card integer index -> 'As' / 'Td' string (rank-major, suit = index % 4)."""
    return _RANKS[cid // 4] + _SUITS[cid % 4]


def _label_of(c0: int, c1: int) -> str:
    """169-bucket label ('AKs') for a two-card integer hand — the strength key the push/fold grids use."""
    return hand_to_label(_card_str(c0) + _card_str(c1))


# ======================================================================================== state stub
# The side-pot-correct equity settle (rollout_leaf.equity_allin_settle) reads ONLY these attributes:
#   state.players[i].{folded, hole, committed_total_bb}, state.board (each card has an integer .index).
# A faithful, engine-FREE stub (identical contract to tests/test_eval_score.py) — keeps us off the
# firewalled engine entirely while reusing the certified side-pot settle.
@dataclass
class _Card:
    index: int


@dataclass
class _Player:
    committed_total_bb: float
    folded: bool = False
    hole: tuple | None = None


@dataclass
class _AllInState:
    players: list
    board: list = field(default_factory=list)


# ======================================================================================== Spot
@dataclass
class Spot:
    """One CRN twin-policy spot: a root + the dealt all-in confrontation + the structure context.

    A policy is asked ``policy(spot) -> {"jam","fold","call"}``. The spot exposes everything a policy
    needs to decide WITHOUT re-dealing: the pre-hand ``stacks``, the ``payouts`` ladder, ``fmt``, the
    hero/villain seats, the hero's 169 ``hand_label``, and ``role`` ("first_in" => jam/fold, "facing_jam"
    => call/fold). ``seats`` is the live count (for the multiway context). ``cell`` tags the regime.
    """

    cell: str
    fmt: "TournamentFormat"
    stacks: list[float]                 # full live field (bb), pre-hand
    payouts: list[float]                # best-first currency ladder (structure-correct, NEVER a literal)
    hero: int
    villain: int
    role: str                           # "first_in" | "facing_jam"
    hand_label: str                     # hero's 169 bucket ('AKs')
    hero_cards: tuple[int, int]         # hero's two 52-card indices (dealt ONCE)
    villain_cards: tuple[int, int]      # villain's two 52-card indices (dealt ONCE)
    blinds: dict                        # {sb_seat: 0.5, bb_seat: 1.0}
    ante: float
    level: int
    depth_bb: float
    deck_seed: int                      # the CRN deck seed (1000 + seed*1e6 + d)
    seats: int                          # live players remaining (= number of positive stacks)


# ======================================================================================== BatteryResult
@dataclass
class BatteryResult:
    """The ΔICM verdict for one regime cell + the chip-EV side-channel + CRN health.

    ``mean`` ± ``half_width`` is the per-spot $-equity ΔICM edge (mean ± 1.96·se). ``verdict`` reduces
    the 95% CI to POSITIVE / NEGATIVE / INCONCLUSIVE. ``chipev_*`` is the SAME paired difference scored
    in chips (the side-channel proving the divergence is ICM-structural, not a chip-EV artifact).
    ``flip_rate`` is the fraction of spots where the ICM-aware action ≠ the chip-EV-optimal action;
    ``icm_delta_saved`` is the summed $ ΔICM over exactly those flips (the dollars ICM buys).
    ``uplift_zero_frac`` is the CRN-health diagnostic: the fraction of spots whose paired ΔICM is exactly
    0 (both policies agreed) — high is healthy (the residual variance lives only in the flips).
    """

    cell: str
    n: int
    mean: float
    se: float
    half_width: float
    ci95: tuple
    verdict: str
    # chip-EV side-channel (same pairing, chip units)
    chipev_mean: float
    chipev_se: float
    chipev_ci95: tuple
    # divergence quantifiers
    flip_rate: float
    icm_delta_saved: float
    uplift_zero_frac: float
    # provenance / homogeneity (per-spot series feedable to cv_absolute.decompose_absolute)
    per_spot_dollar: list = field(default_factory=list)
    per_spot_chip: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ======================================================================================== net-free policies
def always_fold_policy(spot: Spot) -> str:
    """Net-free anchor: always fold (the calibration floor — every confrontation is a fold)."""
    return "fold"


def always_jam_policy(spot: Spot) -> str:
    """Net-free anchor: always commit (jam first-in, call facing a jam) — the aggressive floor."""
    return "call" if spot.role == "facing_jam" else "jam"


def _chip_ev_payouts(stacks: Sequence[float]) -> list[float]:
    """A CHIP-PROPORTIONAL payout ladder: ``[1, 0, 0, ...]`` (winner-take-all). Over winner-take-all the
    ICM solver's terminals are chip-proportional (``pushfold_icm`` docstring), so a solve under this ladder
    is the chip-EV Nash push/fold — the ICM-NAIVE reference the twin plays. The list length matches the
    live field so the multiway open-shove solver has a payout for every seat."""
    return [1.0] + [0.0] * (len(stacks) - 1)


# The open-shove solver enumerates 2^|behind| caller subsets per iteration, so the behind set MUST be
# bounded or a big table hangs. We keep the most RELEVANT callers — the largest (covering) stacks plus
# hero — since a tiny stack behind contributes negligible fold-equity loss; this is the standard
# multiway-ICM approximation and keeps every solve at 2^MAX_BEHIND subsets (cheap).
_MAX_BEHIND = 4


def _open_shove_action(spot: Spot, payouts: Sequence[float], iters: int, cache: dict | None) -> str:
    """Hero's jam/fold (first_in) or call/fold (facing_jam) under ``payouts`` via the multiway ICM solver.

    Uses :func:`open_shove_icm.solve_open_shove_icm` so the chip-EV twin (``payouts`` = winner-take-all)
    and the ICM policy (``payouts`` = the real ladder) are the SAME solve differing ONLY in the payout
    structure — the clean isolation of the ICM effect. For ``facing_jam`` hero reads its GTO call
    frequency off the jammer's open-shove solve (the ``decide.decide_pushfold`` idiom). The behind set is
    capped to the ``_MAX_BEHIND`` largest covering stacks (+ hero) to keep the 2^|behind| subset
    enumeration cheap. Cached on a coarse (role, rounded-stacks, payout-fingerprint) key."""
    from pokerbot.tournament.open_shove_icm import solve_open_shove_icm

    n = len(spot.stacks)
    hero, villain = spot.hero, spot.villain
    blinds = {int(k): float(v) for k, v in spot.blinds.items()}
    pkey = tuple(round(float(p), 3) for p in payouts)

    if spot.role == "first_in":
        live_behind = [i for i in range(n) if i != hero and spot.stacks[i] > 0.0]
        behind = sorted(sorted(live_behind, key=lambda i: -spot.stacks[i])[:_MAX_BEHIND])
        seats = sorted({hero} | set(behind))
        sub_stacks, sub_pay, hero_l, behind_l, blinds_l = _subgame(spot, seats, payouts, blinds, hero)
        key = ("jam", tuple(round(s, 1) for s in sub_stacks), pkey, tuple(behind_l))
        if cache is not None and key in cache:
            sol = cache[key]
        else:
            sol = solve_open_shove_icm(sub_stacks, sub_pay, hero_l, behind_l,
                                       blinds=blinds_l, ante=spot.ante, iters=iters)
            if cache is not None:
                cache[key] = sol
        return "jam" if sol.jam_freq(spot.hand_label) >= 0.5 else "fold"

    # facing_jam: solve the JAMMER's (villain's) open-shove vs the behind set incl hero, read hero's call.
    live_behind = [i for i in range(n) if i not in (villain,) and spot.stacks[i] > 0.0 and i != hero]
    behind_seats = sorted({hero} | set(sorted(live_behind, key=lambda i: -spot.stacks[i])[:_MAX_BEHIND - 1]))
    seats = sorted({villain} | set(behind_seats))
    sub_stacks, sub_pay, villain_l, behind_l, blinds_l = _subgame(spot, seats, payouts, blinds, villain,
                                                                  behind_orig=behind_seats)
    hero_l = seats.index(hero)
    key = ("call", tuple(round(s, 1) for s in sub_stacks), pkey, villain_l, tuple(behind_l))
    if cache is not None and key in cache:
        sol = cache[key]
    else:
        sol = solve_open_shove_icm(sub_stacks, sub_pay, villain_l, behind_l,
                                   blinds=blinds_l, ante=spot.ante, iters=iters)
        if cache is not None:
            cache[key] = sol
    idx = list(sol.labels).index(spot.hand_label)
    call_freq = float(sol.call[hero_l][idx]) if hero_l in sol.call else 0.0
    return "call" if call_freq >= 0.5 else "fold"


def _subgame(spot: Spot, seats: list[int], payouts: Sequence[float], blinds: dict, actor: int,
             behind_orig: list[int] | None = None):
    """Restrict the ICM open-shove to ``seats`` (the actor + the capped behind set), with the BYSTANDERS'
    chips folded into the ladder via the payout TAIL.

    The seats not in the subgame are out of THIS hand but still in the tournament; their ICM presence is
    preserved by keeping the FULL payout vector but truncating to the subgame size with the bystanders'
    finish positions baked into the tail rungs (the subgame competes for the top ``len(seats)`` rungs).
    Returns ``(sub_stacks, sub_payouts, actor_local, behind_local, blinds_local)`` all reindexed to
    0..len(seats)-1 in ``seats`` order."""
    sub_stacks = [float(spot.stacks[s]) for s in seats]
    k = len(seats)
    sub_pay = list(payouts)[:k] + [0.0] * max(0, k - len(payouts))
    actor_local = seats.index(actor)
    blinds_local = {seats.index(s): float(b) for s, b in blinds.items() if s in seats}
    src_behind = behind_orig if behind_orig is not None else [s for s in seats if s != actor]
    behind_local = [seats.index(s) for s in src_behind if s in seats]
    return sub_stacks, sub_pay, actor_local, behind_local, blinds_local


def chip_ev_nash_twin(spot: Spot, *, iters: int = 1200, cache: dict | None = None) -> str:
    """The ICM-NAIVE twin: the chip-EV Nash push/fold action (multiway open-shove under WINNER-TAKE-ALL).

    Solves the same multiway open-shove the ICM policy does, but under a chip-proportional
    (winner-take-all) payout — so ICM has NO bite and the action is the pure chip-EV push/fold. Isolates
    the ICM structure as the ONLY difference between the two arms. Cached (the chart is stack-only here)."""
    return _open_shove_action(spot, _chip_ev_payouts(spot.stacks), iters, cache)


def icm_nash_policy(spot: Spot, *, iters: int = 1200, cache: dict | None = None) -> str:
    """A net-free ICM-AWARE reference policy: the multiway open-shove solved UNDER THE REAL PAYOUT LADDER.

    Two layers of ICM correctness:

    * **Satellite seat lock (the C1 certificate, an EXACT theorem)** — on a SATELLITE spot where hero's
      seat is mathematically locked (:func:`satellite.seat_locked`), the policy FOLDS any chip-risking
      action unconditionally. The locked seat is max $-equity; there is no ladder upside, only downside —
      the rigorous, play-independent stop-gambling rule (not a heuristic tilt).
    * **Real-ladder ICM open-shove** — otherwise hero plays the multiway open-shove
      (:func:`open_shove_icm.solve_open_shove_icm`) solved against the SPOT'S OWN payout vector, so the
      range tightens exactly where the ladder makes busting cost more than doubling gains. This is the
      SAME solve as the chip-EV twin except for the payout, so a difference in action is a genuine
      ICM-driven flip (not a post-hoc tilt).

    This is the cheap, net-free stand-in for the ICM-aware net the battery ultimately serves — enough to
    exercise the full pairing + flip-rate machinery + the C1 certificate on CPU at smoke scale."""
    if spot.fmt is TournamentFormat.SATELLITE:
        return _satellite_action(spot)
    return _open_shove_action(spot, spot.payouts, iters, cache)


# A satellite all-in must clear this risk:reward (seat bubble factor) ceiling to be worth committing. The
# step ladder makes near-seat chips ~worthless, so the seat bubble factor is huge (39-1300+, measured) for
# essentially every borderline spot -> the ICM-correct action is to FOLD almost always (survive to a seat),
# which is the satellite "stop gambling" lever the chip-EV twin completely misses.
_SATELLITE_BF_CEILING = 2.0


def _satellite_action(spot: Spot) -> str:
    """The ICM-correct SATELLITE action: fold any chip-risking all-in whose seat bubble factor exceeds the

    ceiling (the flat seat ladder makes marginal chips worthless near a seat). A LOCKED seat folds
    unconditionally (the exact C1 stop-gambling certificate); otherwise the pairwise
    :func:`satellite.seat_bubble_factor` (risk-$ per gain-$ under the step payoff) must be under
    ``_SATELLITE_BF_CEILING`` to commit. This is the satellite-specific ICM model — the open-shove range
    solve mis-prices a FLAT ladder (it reads equal-paying ranks as no penalty for finishing lower and so
    jams too wide), so the satellite path uses the seat bubble factor directly."""
    seats_paid = len(spot.payouts)
    # C1 exact certificate first: a mathematically-locked seat folds everything that risks chips.
    if _satellite.should_fold_for_seat(spot.stacks, seats_paid, spot.hero, action_risks_chips=True):
        return "fold"
    seat_value = spot.payouts[0] if spot.payouts else 1.0
    bf = _satellite.seat_bubble_factor(spot.stacks, seats_paid, spot.hero, spot.villain,
                                       seat_value=seat_value)
    risky = "jam" if spot.role == "first_in" else "call"
    if not math.isfinite(bf) or bf > _SATELLITE_BF_CEILING:
        return "fold"
    return risky


# ======================================================================================== spot-builders
# Each returns (root_state, pre_stacks, payouts, hero_seat, fmt). The cards are dealt by run_battery (CRN);
# the builder only fixes the STACK GEOMETRY + the structure-correct payout vector for its regime.

def _eff_depth(rng: "np.random.Generator", regime: str) -> tuple[float, float, int]:
    """Draw (depth_bb, ante_bb, level) from the certified structure sampler for a regime."""
    return _structure.sample_spot(rng, regime=regime)


def bubble_icm_spot(rng, *, entrants: int = 180, buy_in: float = 10.0, ctx: dict | None = None):
    """N just above the money, short hero (the classic ICM bubble).

    The live TABLE sits right at the bubble: ``seats`` players competing for the BOTTOM money rungs (the
    last paid place is one bust away), hero the SHORT stack (max ICM pressure). The ladder is the
    structure-correct GLOBAL vector's tail rungs that are actually in contention at the table (the
    min-cash rungs), so the steep "one bust from the money" premium is exact. The full field is collapsed
    to a single table because ICM is read off the table, not the 180-handed field (which is intractable).

    REGIME SENSITIVITY: with a suite ``ctx`` the live seat count + depth track the regime (so the bubble
    table size matches the suite cell), and hero is kept to a MIDDLING short stack (not so short that it
    calls any-two and erases the ICM signal). The canonical pool-normalized ladder is applied afterwards
    by :func:`_canonical_payouts`.
    """
    depth, ante_bb, level = _eff_depth(rng, "mid")
    seats = int(rng.integers(4, _MAX_TABLE_SEATS + 1))     # a realistic bubble table
    if ctx and ctx.get("n_seats"):
        seats = max(3, min(int(ctx["n_seats"]), _MAX_TABLE_SEATS))
    if ctx and ctx.get("depth_bb"):
        depth = float(ctx["depth_bb"])
    gv = _payouts.payout_vector(TournamentFormat.NORMAL, entrants=entrants, buy_in=buy_in)
    # The bubble table competes for the MIN-CASH rungs: take the bottom `seats-1` paid rungs (so one seat
    # is the unpaid bubble bust) — the flattest, most ICM-pressured part of the ladder.
    tail = gv[-(seats - 1):] if len(gv) >= seats - 1 else gv
    payouts = list(tail) + [0.0]                            # +1 unpaid finisher => the bubble
    # hero is the short stack; the rest are a spread of comfortable stacks (covered). Keep hero a MIDDLING
    # short (>= ~6bb under a suite ctx) so the call/fold is a genuine ICM decision, not an any-two shove.
    hero_stack = max(6.0, depth * 0.7) if ctx else max(2.0, depth)
    big = hero_stack * float(rng.uniform(2.0, 3.5))
    stacks = [hero_stack] + [big * float(rng.uniform(0.7, 1.3)) for _ in range(seats - 1)]
    hero = 0
    return _root(stacks, _ladder_for_table(payouts, seats), hero, ante_bb, level, depth,
                 TournamentFormat.NORMAL)


def payjump_spot(rng, *, entrants: int = 180, buy_in: float = 10.0, ft_size: int = 7,
                 ctx: dict | None = None):
    """Final-table ladder AT a pay jump: ``ft_size`` left, hero a middling stack facing a covering jam.

    Blinds use the FT-rollback level (``structure.final_table_rollback``); the payout vector is the
    structure-correct NORMAL ladder. The pay-jump premium is implicit in the steep FT ladder shape.

    REGIME SENSITIVITY: with a suite ``ctx`` the table SIZE tracks the regime — ``hu`` collapses the final
    table to 2-handed (a genuinely different, top-2-only ladder + heads-up geometry) while ``ft`` keeps the
    short-handed final table. Combined with the format-correct canonical payouts, this makes the ``hu`` and
    ``ft`` cells genuinely distinct (the artifact had them identical because both mapped to this builder
    with no regime input).
    """
    depth, ante_bb, level = _eff_depth(rng, "late")
    ladder = _structure.final_table_rollback(max(level, 2))
    ante_bb = ladder.ante / ladder.bb
    payouts = _payouts.payout_vector(TournamentFormat.NORMAL, entrants=entrants, buy_in=buy_in)
    seats = min(ft_size, _MAX_TABLE_SEATS)
    if ctx and ctx.get("n_seats"):
        seats = max(2, min(int(ctx["n_seats"]), _MAX_TABLE_SEATS))
    if ctx and ctx.get("depth_bb"):
        depth = float(ctx["depth_bb"])
    # Hero holds a MIDDLING stack (a genuine call/fold decision — at 3bb you call any-two, which erases the
    # ICM signal). A COVERING opponent jams (the largest stack); short stacks below hero make busting a
    # real pay jump for the rest (the ICM premium on hero's call). Scale the stacks to the regime depth.
    base = float(depth) if (ctx and ctx.get("depth_bb")) else float(rng.uniform(8.0, 16.0))
    hero_stack = max(2.0, base * float(rng.uniform(0.6, 1.1)))
    jammer_stack = hero_stack * float(rng.uniform(1.5, 3.0))            # covers hero -> hero is at risk
    stacks = [hero_stack, jammer_stack]
    for _ in range(seats - 2):
        stacks.append(hero_stack * float(rng.uniform(0.3, 0.9)))       # shorter than hero (below in the ladder)
    hero = 0
    return _root(stacks, _ladder_for_table(payouts, seats), hero, ante_bb, level, hero_stack,
                 TournamentFormat.NORMAL)


# final_table_icm_spot is an alias regime (FT ladder at a pay jump) — same builder, premium-tagged.
final_table_icm_spot = payjump_spot


def shortstack_pushfold_spot(rng, *, entrants: int = 90, buy_in: float = 10.0, seats: int = 5):
    """≤20bb first-in jam/fold — the CERTIFIED push/fold zone (anchor-calibratable against the floor).

    Hero is ≤20bb (the certified zone — ``decide.PUSHFOLD_MAX_BB``); the structure-correct NORMAL ladder.
    This is the cell the net-free anchors calibrate cleanly (every action is a certified push/fold).
    """
    depth, ante_bb, level = _eff_depth(rng, "late")
    depth = min(depth, 20.0)                     # clamp into the certified push/fold zone
    seats = min(seats, _MAX_TABLE_SEATS)
    payouts = _payouts.payout_vector(TournamentFormat.NORMAL, entrants=entrants, buy_in=buy_in)
    hero_stack = max(2.0, depth)
    stacks = [hero_stack] + [hero_stack * float(rng.uniform(0.8, 2.5)) for _ in range(seats - 1)]
    hero = 0
    return _root(stacks, _ladder_for_table(payouts, seats), hero, ante_bb, level, depth,
                 TournamentFormat.NORMAL)


def satellite_bubble_spot(rng, *, seats_paid: int = 8, seat_value: float = 100.0, live: int = 9,
                          ctx: dict | None = None):
    """The C1 satellite cell: ``live`` players, ``seats_paid`` identical seats, a BORDERLINE hero.

    The flat ``satellite.satellite_payouts`` step ladder drives EXTREME ICM near the seat bubble: a
    covered-but-near-locked hero should FOLD everything that risks the seat. Hero is placed at the
    borderline (just under / around the ``1/(seats+1)`` lock fraction) so ``seat_locked`` is the live
    question. Returns a SATELLITE root.

    REGIME SENSITIVITY: with a suite ``ctx`` the lock-PROXIMITY tracks the regime — a ``deep`` satellite
    spreads hero FAR from the lock boundary (deep stacks, the seat bubble is many hands away, so the seat
    pressure is MILD), whereas the on-bubble regimes place hero right at the boundary (the classic
    stop-gambling pressure). So deep-satellite no longer carries the same seat-bubble pressure as the
    satellite bubble (the artifact's single-constant-across-regimes was exactly this geometry collapse).
    """
    depth, ante_bb, level = _eff_depth(rng, "late")
    regime = str(ctx.get("regime", "")) if ctx else ""
    if ctx and ctx.get("depth_bb"):
        depth = float(ctx["depth_bb"])
    live = min(live, _MAX_TABLE_SEATS)
    if ctx and ctx.get("n_seats"):
        live = max(3, min(int(ctx["n_seats"]) + 1, _MAX_TABLE_SEATS))
    seats_paid = min(seats_paid, live - 1)          # at least one seat must be unpaid (a real bubble)
    payouts = _satellite.satellite_payouts(seats_paid, seat_value)   # flat [seat_value]*seats
    n = live
    hero_stack = max(3.0, depth)
    # Lock-proximity multiplier: near the boundary on the bubble (the seat is one bust away -> max ICM);
    # FAR below the boundary when DEEP (the seat bubble is distant -> the step ladder barely bites). This
    # is what makes the deep-satellite ΔICM differ from the satellite-bubble ΔICM.
    if regime == "deep":
        prox_lo, prox_hi = 0.35, 0.6        # well under the lock fraction: deep, mild seat pressure
    elif regime in ("mid", "shallow"):
        prox_lo, prox_hi = 0.6, 0.95
    else:                                   # bubble / ft / hu (or no ctx): right at the lock boundary
        prox_lo, prox_hi = 0.85, 1.25
    total = hero_stack * (seats_paid + 1) / float(rng.uniform(prox_lo, prox_hi))
    rest = max(1e-6, total - hero_stack)
    rivals = list(rng.dirichlet(np.ones(n - 1)) * rest)
    stacks = [hero_stack] + [float(x) for x in rivals]
    hero = 0
    return _root(stacks, payouts, hero, ante_bb, level, depth, TournamentFormat.SATELLITE)


def deep_nearmoney_spot(rng, *, entrants: int = 180, buy_in: float = 10.0, seats: int = 6):
    """35–50bb near money — a low-divergence CONTROL cell (chip-EV ≈ $-EV; few flips expected).

    Deep stacks near (but not on) the money: ICM pressure is mild, so the chip-EV twin and the ICM-aware
    policy should AGREE on almost every spot (low flip-rate, ΔICM ~0). A healthy control that proves the
    battery does not manufacture an edge where none exists.
    """
    ante_bb = _structure.ante_for_bb(100) / 100.0       # representative ~0.125bb ante
    depth = float(rng.uniform(35.0, 50.0))
    level = 8
    places = _payouts.places_paid(TournamentFormat.NORMAL, entrants)
    payouts = _payouts.payout_vector(TournamentFormat.NORMAL, entrants=entrants, buy_in=buy_in)
    n = min(seats, _MAX_TABLE_SEATS, max(2, places + 4))   # several off the money -> deep, mild ICM
    hero_stack = depth
    stacks = [hero_stack] + [hero_stack * float(rng.uniform(0.85, 1.2)) for _ in range(n - 1)]
    hero = 0
    return _root(stacks, _ladder_for_table(payouts, n), hero, ante_bb, level, depth,
                 TournamentFormat.NORMAL)


def _root(stacks, payouts, hero, ante_bb, level, depth, fmt):
    """Package a builder's stack geometry into the (root_state, pre_stacks, payouts, hero, fmt) tuple.

    The ``root_state`` is a lightweight marker dict (the concrete all-in confrontation is materialized by
    ``sample_root`` once the CRN cards are dealt) carrying the chosen villain + blind posting. We pick the
    villain as the LARGEST OTHER stack (the covering opponent — the one whose jam/call is the real ICM
    question) and post SB/BB on the two acting seats by convention (hero SB, villain BB)."""
    n = len(stacks)
    others = [i for i in range(n) if i != hero and stacks[i] > 0.0]
    villain = max(others, key=lambda i: stacks[i]) if others else (hero + 1) % n
    blinds = {hero: 0.5, villain: 1.0}
    root = {
        "ante_bb": float(ante_bb), "level": int(level), "depth_bb": float(depth),
        "villain": int(villain), "blinds": blinds,
    }
    return root, [float(s) for s in stacks], [float(p) for p in payouts], int(hero), fmt


# Registry: cell name -> (builder, role, premium_tag). role decides first_in (jam/fold) vs facing_jam.
CELLS: dict[str, tuple] = {
    "bubble_icm": (bubble_icm_spot, "first_in"),
    "payjump": (payjump_spot, "facing_jam"),
    "final_table_icm": (final_table_icm_spot, "facing_jam"),
    "shortstack_pushfold": (shortstack_pushfold_spot, "first_in"),
    "satellite_bubble": (satellite_bubble_spot, "first_in"),
    "deep_nearmoney": (deep_nearmoney_spot, "first_in"),
}


# ============================================================================== suite-context canonicalize
# THE METRIC-PHYSICALITY FIX (root cause of the artifactual ΔICM).
# --------------------------------------------------------------------------------------------------------
# A builder fixes its own STACK GEOMETRY + an INTRINSIC payout ladder. The intrinsic ladders, however, are
# self-built at INCONSISTENT scales and from a single hardcoded format:
#   * ``bubble_icm`` takes the min-cash *tail* of a ``buy_in=10`` global vector (rungs ~17, pool ~89);
#   * ``final_table_icm`` takes the *full top* of that SAME vector (rungs ~298, pool ~1037);
#   * ``satellite_bubble`` uses ``seat_value=100`` over 8 seats (pool ~800);
#   * every non-satellite builder hardcodes ``TournamentFormat.NORMAL``.
# Consequences (the observed artifact): a per-game ΔICM-$ scored against the FT/satellite ladders is
# O(20-34) — NON-PHYSICAL for a per-game $ delta (a sane one is O(0.01-1), like the analytic-ICM −0.094
# reference) — and, because the format/payouts are builder-hardcoded, NORMAL/SNG/PKO collapse to ONE number
# and the satellite seat-bubble pressure is identical across every regime.
#
# THE FIX: when the suite passes its canonical per-cell context (``ctx``: the FORMAT-correct,
# REGIME-sized, pool-NORMALIZED payout vector + depth/ante/seats that ``eval_tournament_suite`` already
# computes via ``_build_cell_payouts``), re-express the spot at that canonical scale + format. This
# (a) NORMALIZES every cell to one pool scale so the $ delta is physical, (b) makes the format real so
# NORMAL/SNG/PKO/SATELLITE diverge where their ladders differ, and (c) makes the satellite/FT pressure
# track the regime (hu is 2-handed + shallower, deep-satellite carries less seat-bubble pressure than the
# satellite bubble). With NO ``ctx`` the builders are byte-unchanged (the standalone/legacy contract that
# ``tests/test_regime_battery.py`` pins), so this is purely additive.


def _builder_params(builder) -> frozenset:
    """The parameter names a spot-builder accepts (so ``sample_root`` only passes ``ctx`` to the
    regime-aware builders). Cached per callable via ``inspect``."""
    import inspect
    try:
        return frozenset(inspect.signature(builder).parameters)
    except (TypeError, ValueError):
        return frozenset()


def _resize_ladder(ladder: list[float], k: int, *, unpaid_tail: int = 0) -> list[float]:
    """Best-first ladder resized to ``k`` live seats, optionally reserving ``unpaid_tail`` zero rungs.

    Keeps the steep TOP of the structure-correct ladder (where the ICM premium lives) and pads with 0 for
    any seat beyond the paid places (an unpaid finisher). ``unpaid_tail`` forces at least that many trailing
    zero rungs (the bubble's one-bust-from-the-money structure)."""
    pv = [float(x) for x in ladder if True]
    paid = max(0, k - max(0, unpaid_tail))
    head = pv[:paid]
    if len(head) < paid:
        head = head + [0.0] * (paid - len(head))
    out = head + [0.0] * (k - len(head))
    return out[:k]


def _canonical_payouts(cell: str, spot_payouts: list[float], fmt, ctx: dict) -> tuple[list[float], object]:
    """Re-express the spot's payout ladder at the suite's canonical (format-correct, normalized) scale.

    ``ctx['payouts']`` is the canonical per-cell vector the suite already built (``_build_cell_payouts`` —
    NORMAL/SNG/PKO pooled to ~100, SATELLITE a flat seat ladder of ``seat_value`` ~10). We resize it to the
    spot's live seat count, preserving the cell's structural shape:

      * ``bubble_icm`` keeps its ONE unpaid bubble rung (last seat busts unpaid) — the min-cash tail premium;
      * SATELLITE keeps the flat ``[seat_value]*paid`` step (so ``satellite.seat_locked`` math is intact);
      * every other NORMAL/SNG/PKO cell takes the steep top of the canonical ladder.

    Returns ``(payouts, fmt)`` with ``fmt`` taken from ``ctx['fmt']`` (so NORMAL/SNG/PKO/SATELLITE are real,
    not the builder's hardcoded NORMAL). This is the single point that makes the ΔICM PHYSICAL, FORMAT-
    sensitive, and REGIME-sensitive."""
    cpv = [float(x) for x in ctx.get("payouts", []) if x is not None]
    cfmt = ctx.get("fmt", fmt)
    k = len(spot_payouts)
    if not cpv or k <= 0:
        return spot_payouts, fmt
    if cfmt is TournamentFormat.SATELLITE:
        # Flat seat ladder at the canonical seat_value, resized to the live seats with one unpaid bubble
        # rung (a real seat bubble), so seat_locked / seat_bubble_factor keep their step-payoff meaning.
        seat_value = float(max(cpv))
        paid = max(1, min(len(cpv), k - 1))
        return [seat_value] * paid + [0.0] * (k - paid), cfmt
    if cell == "bubble_icm":
        # The bubble keeps exactly ONE unpaid finisher (the bubble bust); the paid rungs are the canonical
        # ladder's MIN-CASH tail (the flattest, most ICM-pressured part — where the bubble premium lives).
        paid = max(1, k - 1)
        tail = cpv[-paid:] if len(cpv) >= paid else cpv
        return list(tail) + [0.0] * (k - len(tail)), cfmt
    # Every other NORMAL/SNG/PKO cell: the steep TOP of the canonical ladder (table-local truncation).
    return _resize_ladder(cpv, k), cfmt


# ======================================================================================== sample_root
def sample_root(cell: str, d: int, seed: int, ctx: dict | None = None) -> Spot:
    """Materialize ONE fully-dealt :class:`Spot` for ``cell`` at spot index ``d`` under ``seed``.

    The lifecycle sampler (inline here, per the spec, to keep the core bounded): draw the cell's stack
    geometry from its builder under a deterministic numpy RNG, then DEAL the cards ONCE from the project
    deck seed ``deck_seed = 1000 + seed*1_000_000 + d`` (so a spot's card luck is a pure function of
    ``(seed, d)`` — the CRN hook every arm shares). Hero + villain get two cards each off a shuffled deck.

    ``ctx`` (the suite's canonical per-cell context — FORMAT-correct, REGIME-sized, pool-NORMALIZED
    payout vector + depth/ante/seats from ``eval_tournament_suite._build_cell_payouts``) re-expresses the
    spot's payout ladder + format at the canonical scale (see :func:`_canonical_payouts`) so the ΔICM is a
    PHYSICAL per-game $ delta that VARIES across formats and regimes. With ``ctx is None`` the builder
    output is byte-unchanged (the standalone/legacy contract)."""
    if cell not in CELLS:
        raise KeyError(f"unknown cell {cell!r}; known: {sorted(CELLS)}")
    builder, role = CELLS[cell]
    # Geometry RNG: deterministic from (seed, d, cell) but DISTINCT from the deck RNG so stacks and cards
    # are independent draws (a different stack spread does not perturb the card luck).
    geom_seed = (hash((cell, seed, d)) & 0x7FFFFFFF)
    grng = np.random.default_rng(geom_seed)
    # Builders that accept a ``ctx`` kwarg (the regime-aware geometry: satellite/payjump) get it; the rest
    # keep their fixed geometry (the canonical payout/format override below still makes them physical).
    if ctx and "ctx" in _builder_params(builder):
        root, stacks, payouts, hero, fmt = builder(grng, ctx=ctx)
    else:
        root, stacks, payouts, hero, fmt = builder(grng)
    villain = root["villain"]

    # Canonicalize the payouts + format to the suite scale when a cell context is supplied. This is THE
    # metric-physicality fix: the raw builder ladder (FT full-top / seat_value-100 / NORMAL-only) is
    # replaced by the format-correct, regime-sized, pool-normalized canonical vector so the per-game ΔICM
    # is O(0.01-1), format-sensitive, and regime-sensitive.
    ante_bb = root["ante_bb"]
    depth_bb = root["depth_bb"]
    if ctx:
        payouts, fmt = _canonical_payouts(cell, payouts, fmt, ctx)
        # The regime ante/depth from the suite (so the dead-money settle + reported depth are regime-correct).
        if ctx.get("ante_bb") is not None:
            ante_bb = float(ctx["ante_bb"])
        if ctx.get("depth_bb"):
            depth_bb = float(ctx["depth_bb"])

    deck_seed = _DECK_SEED_BASE + seed * _DECK_SEED_SEED_STRIDE + d
    drng = random.Random(deck_seed)
    deck = list(range(52))
    drng.shuffle(deck)
    hero_cards = (deck[0], deck[1])
    villain_cards = (deck[2], deck[3])

    return Spot(
        cell=cell, fmt=fmt, stacks=stacks, payouts=payouts, hero=hero, villain=villain,
        role=role, hand_label=_label_of(*hero_cards),
        hero_cards=hero_cards, villain_cards=villain_cards,
        blinds=root["blinds"], ante=ante_bb, level=root["level"], depth_bb=depth_bb,
        deck_seed=deck_seed, seats=sum(1 for s in stacks if s > 0.0),
    )


# ======================================================================================== terminal settle
def _settle_runouts(spot: Spot, hero_action: str, villain_action: str, settle_seed: int,
                    n_runouts: int) -> list[list[float]]:
    """Resolve the all-in to a LIST of per-runout terminal CHIP-delta vectors under CRN (``settle_seed``).

    The two seats commit the effective stack on a jam+call; otherwise the aggressor takes the dead chips
    (ante + blinds) uncontested. A contested all-in is settled side-pot-correctly (``sim._settle_allin``)
    on EACH of ``n_runouts`` board completions seeded by ``settle_seed`` — and we return ALL of them (NOT
    the chip-mean) so the $-EV mapping can ICM-score EACH runout terminal and average in $-SPACE.

    WHY PER-RUNOUT (the Jensen correctness point): ICM is non-linear in chips, so ``ICM(mean_chips) !=
    mean(ICM(chips))``. A locked-or-near-money seat that busts realizes ~0 $ (loses the seat), an
    enormous downside that the chip-MEAN averages away into a positive number. Mapping EACH runout through
    ICM then averaging prices that bust risk correctly — which is exactly what makes a near-money jam look
    bad in $ even when it is chip-EV-positive. The shared ``settle_seed`` keeps it CRN: the SAME runout
    cards are dealt for both arms of the pairing, so the per-runout difference still cancels card luck.

    Returns ``[chip_vec_runout_0, chip_vec_runout_1, ...]`` (each dense over seats, sum ~= 0). An
    UNCONTESTED terminal is deterministic -> a single-element list.
    """
    from pokerbot.tournament.sim import _settle_allin

    n = len(spot.stacks)
    hero, villain = spot.hero, spot.villain
    ante = spot.ante
    blinds = spot.blinds

    # committed dead money: every live seat posts the ante; the two blind seats post their blind.
    committed = {i: (min(ante, spot.stacks[i]) if spot.stacks[i] > 0 else 0.0) for i in range(n)}
    for s, b in blinds.items():
        committed[s] = committed.get(s, 0.0) + min(b, spot.stacks[s])

    contested = False
    if spot.role == "first_in":
        # hero jams or folds; villain is the lone caller (the covering opponent).
        if hero_action == "jam":
            risk = min(spot.stacks[hero], spot.stacks[villain])
            committed[hero] = committed.get(hero, 0.0) + risk
            if villain_action == "call":
                committed[villain] = committed.get(villain, 0.0) + risk
                contested = True
        # hero folds -> nothing more committed (loses only its posted ante/blind)
    else:  # facing_jam: villain has jammed; hero calls or folds.
        risk = min(spot.stacks[hero], spot.stacks[villain])
        committed[villain] = committed.get(villain, 0.0) + risk
        if hero_action == "call":
            committed[hero] = committed.get(hero, 0.0) + risk
            contested = True

    if contested:
        contenders = [hero, villain]
        holes = {hero: spot.hero_cards, villain: spot.villain_cards}
        c_committed = {s: committed[s] for s in contenders}
        dead = sum(committed[i] for i in range(n) if i not in contenders)   # folders' chips -> main pot
        used = set(spot.hero_cards) | set(spot.villain_cards)
        deck = [c for c in range(52) if c not in used]
        rng = random.Random(settle_seed)
        runs: list[list[float]] = []
        for _ in range(max(1, n_runouts)):
            board = rng.sample(deck, 5)
            won = _settle_allin(contenders, c_committed, holes, board, dead)
            vec = [0.0] * n
            for i in range(n):
                vec[i] = (won.get(i, 0.0) - committed[i]) if i in contenders else -committed[i]
            runs.append(vec)
        return runs

    # Uncontested: the aggressor scoops the dead chips (all committed) uncontested. Deterministic.
    pot = sum(committed.values())
    if spot.role == "first_in" and hero_action == "jam":
        winner = hero
    elif spot.role == "facing_jam" and hero_action == "fold":
        winner = villain                     # villain's jam took it down
    else:
        # hero folded a first-in spot, or a degenerate no-commit case -> villain (BB) keeps the dead pot.
        winner = villain
    out = [-committed[i] for i in range(n)]
    out[winner] += pot
    return [out]


def _settle_spot(spot: Spot, hero_action: str, villain_action: str, settle_seed: int,
                 n_runouts: int) -> list[float]:
    """The CHIP-mean terminal (mean-preserving equity leaf): the per-runout average chip-delta vector.

    This is the chip side-channel's settle (chip-EV is LINEAR, so the chip-mean is the right summary).
    The $-EV path uses :func:`_settle_runouts` + per-runout ICM averaging instead (Jensen). Kept as the
    public, test-facing chip settle (zero-sum, CRN-stable)."""
    runs = _settle_runouts(spot, hero_action, villain_action, settle_seed, n_runouts)
    n = len(spot.stacks)
    return [float(np.mean([r[i] for r in runs])) for i in range(n)]


def _score_spot(spot: Spot, hero_action: str, villain_action: str, settle_seed: int,
                n_runouts: int) -> tuple[float, float]:
    """Return ``(dollar_hero, chip_hero)`` for hero's action — the $-EV (ICM-per-runout) + chip side-channel.

    The $-EV is ``mean_runouts( score_chip_to_dollar(per-runout terminal)[hero] )`` — ICM applied to EACH
    runout then averaged (correct under Jensen, prices bust risk). The chip side-channel is the chip-mean
    on hero's seat. Both share the SAME ``settle_seed`` so the runout cards are identical across arms (CRN).
    """
    runs = _settle_runouts(spot, hero_action, villain_action, settle_seed, n_runouts)
    hero = spot.hero
    sfmt = _prize_score_fmt(spot.fmt)
    dollars = [float(score_chip_to_dollar(spot.stacks, r, spot.payouts, sfmt)[hero]) for r in runs]
    chips = [r[hero] for r in runs]
    return float(np.mean(dollars)), float(np.mean(chips))


def _prize_score_fmt(fmt):
    """The format the E6 battery SCORES a chip terminal under.

    NORMAL/SATELLITE route through :func:`icm.icm_delta` directly (the battery's spot carries no bounty
    model). PKO is scored on its **prize** ladder via the SAME ICM wrap (i.e. as NORMAL) — the bounty side
    is the separate E1 ``lbr_icm`` / ``bounty`` concern, which needs the all-in pair + showdown equity the
    E6 spot does not carry. This keeps the E6 ΔICM well-defined while preserving PKO's FORMAT sensitivity:
    the PKO PRIZE ladder has a genuinely different shape than the NORMAL ladder (half the pool is in
    bounties), so a PKO cell still diverges from its NORMAL twin through the ladder shape, not the bounty."""
    return TournamentFormat.NORMAL if fmt is TournamentFormat.PKO else fmt


# ======================================================================================== run_battery
def run_battery(cell: str, hero_policy: Callable[[Spot], str],
                chipev_twin: Callable[[Spot], str], *, n_spots: int = 200, seed: int = 0,
                n_runouts: int = _DEFAULT_RUNOUTS, settle_offset: int = 7777,
                ctx: dict | None = None) -> BatteryResult:
    """The E6 CRN twin-policy ΔICM battery for one regime ``cell``.

    For each of ``n_spots`` spots: draw a fully-dealt root via :func:`sample_root` (cards dealt ONCE from
    the CRN deck seed), run BOTH ``hero_policy`` (ICM-aware) and ``chipev_twin`` (ICM-naive) to terminal on
    the SAME deck + the SAME ``settle_seed``, map each chip result through
    :func:`eval_score.score_chip_to_dollar`, and accumulate the paired ΔICM (= hero $ − twin $) on the
    hero seat plus the chip-EV side-channel (= hero chip − twin chip).

    ``ctx`` (the suite's canonical per-cell context) re-expresses each spot's payouts + format at the
    canonical, pool-normalized, format-/regime-correct scale (see :func:`sample_root`), so the reported
    ΔICM is a PHYSICAL per-game $ delta. ``ctx is None`` => legacy standalone behaviour.

    Returns a :class:`BatteryResult`: mean ± 1.96·se ΔICM + verdict, the chip-EV-vs-$ side-channel, the
    flip-rate + summed ``icm_delta_saved``, and ``uplift_zero_frac`` (CRN health). The per-spot series are
    retained so a HOMOGENEOUS cell can be handed to ``cv_absolute.decompose_absolute``.
    """
    if cell not in CELLS:
        raise KeyError(f"unknown cell {cell!r}; known: {sorted(CELLS)}")

    dollar_diffs: list[float] = []
    chip_diffs: list[float] = []
    flips = 0
    icm_saved = 0.0
    zero_pairs = 0

    for d in range(n_spots):
        spot = sample_root(cell, d, seed, ctx=ctx)
        # The villain's action: against the chip-EV Nash *caller/jammer* — the covering opponent plays the
        # chip-EV push/fold line (its own decision is ICM-naive too, so the ONLY axis we vary is hero's
        # policy). This keeps the pairing clean: both arms share the identical villain action.
        villain_action = _villain_action(spot, n_runouts)

        # Same settle_seed for BOTH arms (CRN: a contested all-in gets identical runout samples).
        settle_seed = settle_offset + spot.deck_seed

        a_hero = hero_policy(spot)
        a_twin = chipev_twin(spot)

        # $-EV is ICM-applied PER RUNOUT then averaged (Jensen: prices bust risk); chip is the chip-mean.
        dol_hero, chip_hero = _score_spot(spot, a_hero, villain_action, settle_seed, n_runouts)
        dol_twin, chip_twin = _score_spot(spot, a_twin, villain_action, settle_seed, n_runouts)

        ddol = dol_hero - dol_twin
        dchip = chip_hero - chip_twin
        dollar_diffs.append(ddol)
        chip_diffs.append(dchip)

        if a_hero != a_twin:
            flips += 1
            icm_saved += ddol
        else:
            # Identical action => identical terminal (same deck, same settle_seed) => exactly-0 pair.
            if abs(ddol) < 1e-12:
                zero_pairs += 1

    n = len(dollar_diffs)
    arr = np.asarray(dollar_diffs, dtype=np.float64)
    carr = np.asarray(chip_diffs, dtype=np.float64)
    mean = float(arr.mean()) if n else 0.0
    se = float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    hw = 1.96 * se
    ci = (mean - hw, mean + hw)

    cmean = float(carr.mean()) if n else 0.0
    cse = float(carr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    chw = 1.96 * cse
    cci = (cmean - chw, cmean + chw)

    return BatteryResult(
        cell=cell, n=n, mean=mean, se=se, half_width=hw, ci95=ci, verdict=verdict(ci[0], ci[1]),
        chipev_mean=cmean, chipev_se=cse, chipev_ci95=cci,
        flip_rate=(flips / n) if n else 0.0, icm_delta_saved=icm_saved,
        uplift_zero_frac=(zero_pairs / n) if n else 0.0,
        per_spot_dollar=dollar_diffs, per_spot_chip=chip_diffs,
        meta={"seed": seed, "n_spots": n_spots, "n_runouts": n_runouts,
              "fmt": CELLS[cell][0].__name__},
    )


def _villain_action(spot: Spot, n_runouts: int) -> str:
    """The covering opponent's ICM-naive (chip-EV Nash) action — shared by BOTH arms of the pairing.

    In a "first_in" spot the villain is the lone CALLER (call/fold vs hero's jam); in a "facing_jam" spot
    the villain is the JAMMER (always jam — the spot is defined as facing its jam). Keeping the villain
    fixed across arms is what makes the difference attributable solely to hero's policy."""
    if spot.role == "facing_jam":
        return "jam"
    # first_in: villain decides call/fold vs a representative jam using the chip-EV Nash caller range.
    from pokerbot.tournament.pushfold_icm import solve_hu_icm_pushfold
    eff = round(float(min(spot.stacks[spot.hero], spot.stacks[spot.villain])), 1)
    sol = solve_hu_icm_pushfold([eff, eff], [1.0, 0.0], 0, 1, ante=spot.ante, iters=800)
    vlabel = _label_of(*spot.villain_cards)
    return "call" if sol.call_freq(vlabel) >= 0.5 else "fold"


# ======================================================================================== certificate
def satellite_stop_gambling_certificate(n_spots: int = 200, seed: int = 0) -> dict:
    """C1 certificate: on the satellite seat bubble, the ICM-aware policy NEVER risks a LOCKED seat.

    Draws ``satellite_bubble`` spots and checks the rigorous, play-independent
    :func:`satellite.seat_locked` certificate: in every spot where hero's seat is mathematically LOCKED,
    an ICM-aware policy MUST fold any chip-risking all-in (the locked seat is max equity; only downside
    exists). Reports the fraction of locked spots where :func:`icm_nash_policy` correctly folds (must be
    1.0 for the certificate to PASS) — a structural guard the chip-EV twin would FAIL.
    """
    locked = 0
    locked_folds = 0
    chipev_locked_commits = 0
    cache_icm: dict = {}
    cache_chip: dict = {}
    for d in range(n_spots):
        spot = sample_root("satellite_bubble", d, seed)
        # seats_paid recovered from the flat payout vector (all entries equal => len == seats).
        seats_paid = len(spot.payouts)
        if not _satellite.seat_locked(spot.stacks, seats_paid, spot.hero):
            continue
        locked += 1
        a_icm = icm_nash_policy(spot, cache=cache_icm)
        a_chip = chip_ev_nash_twin(spot, cache=cache_chip)
        if a_icm == "fold":
            locked_folds += 1
        if a_chip in ("jam", "call"):
            chipev_locked_commits += 1
    frac = (locked_folds / locked) if locked else 1.0
    return {
        "cell": "satellite_bubble",
        "n_spots": n_spots,
        "locked_spots": locked,
        "icm_fold_frac_when_locked": frac,
        "chipev_commits_when_locked": chipev_locked_commits,
        "pass": locked == 0 or frac >= 1.0 - 1e-9,
        "note": ("no locked spots sampled (raise n_spots / live)" if locked == 0
                 else "ICM folds every locked seat (stop-gambling certificate)"),
    }


# ======================================================================================== suite entry point
# Map the suite's (format, regime) onto this module's regime CELLS. The format chooses the satellite cell
# when SATELLITE; otherwise the regime depth/ICM-pressure picks the bubble / final-table / pushfold /
# deep-control cell. This is the E6 surface the format x regime driver iterates.
_CELL_FOR_REGIME = {
    "deep": "deep_nearmoney",
    "mid": "shortstack_pushfold",
    "shallow": "shortstack_pushfold",
    "bubble": "bubble_icm",
    "ft": "final_table_icm",
    "hu": "final_table_icm",
}


def _fmt_from_key(fmt_key: str):
    """The suite ``fmt_key`` -> :class:`payouts.TournamentFormat`. SNG is a NORMAL ladder variant (scored
    NORMAL); PKO / SATELLITE carry their own format (so the chip->$ routing + ladder shape are correct)."""
    k = str(fmt_key).upper()
    if k == "SATELLITE":
        return TournamentFormat.SATELLITE
    if k == "PKO":
        return TournamentFormat.PKO
    return TournamentFormat.NORMAL          # NORMAL and SNG both score through the NORMAL ICM ladder


def _suite_ctx(fmt_key: str, regime_key: str, cell_args: dict) -> dict | None:
    """Build the canonical per-cell context the spot canonicalizer consumes from the suite ``cell_args``.

    Carries the FORMAT-correct, REGIME-sized, pool-NORMALIZED ``payouts`` (the suite's ``_build_cell_payouts``
    output — NORMAL/SNG/PKO pooled to ~100, SATELLITE a flat seat ladder ~10) plus the regime ``depth_bb`` /
    ``ante_bb`` / ``n_seats`` and the regime tag. Returns ``None`` (legacy scale) only if no canonical
    payouts were supplied — a bare standalone call — so the standalone/legacy contract is untouched."""
    payouts = cell_args.get("payouts")
    if not payouts:
        return None
    return {
        "fmt": _fmt_from_key(fmt_key),
        "regime": str(regime_key),
        "payouts": [float(p) for p in payouts],
        "depth_bb": float(cell_args["depth_bb"]) if cell_args.get("depth_bb") else None,
        "ante_bb": float(cell_args["ante_bb"]) if cell_args.get("ante_bb") is not None else None,
        "n_seats": int(cell_args["n_seats"]) if cell_args.get("n_seats") else None,
    }


def _net_spot_policy(cell_args: dict) -> Callable[["Spot"], str]:
    """A spot-level ``policy(spot) -> {'jam','fold','call'}`` that serves the 7-max net.

    The 7-max net is consulted ONLY through ``neural_serve.make_seven_neural_policy`` (the skew-safe
    bridge) — this module never touches the checkpoint directly, so there is no encoder-skew surface here.
    Building a faithful per-street ``EnvState`` for every spot is heavy, so (mirroring
    ``mtt_lifecycle._serve_root_aggression``) we serve the net on a REPRESENTATIVE 7-live root dealt at the
    spot's effective depth + ante via the engine (a genuine ``init_hand`` node, identity-placed, so the
    served absolute seat/button one-hots match training) and read the net's COMMIT mass (everything but
    FOLD/CHECK). The commit mass thresholds the spot's jam/fold or call/fold action: commit-mass >= 0.5 =>
    act, else fold. On ANY net failure (no jax, skew, degenerate serve) the policy falls back to the
    net-free ICM-aware reference (``icm_nash_policy``) — never raising, never fabricating a decision.

    Loading the net FAILS LOUD up front (constructed once) so a bad checkpoint surfaces as the cell's
    ``status:'error'`` rather than a silent anchor measurement under a ``policy=='net'`` claim.
    """
    from pokerbot.tournament.neural_serve import make_seven_neural_policy

    policy_fn, _meta = make_seven_neural_policy(cell_args.get("ckpt"))
    _icm_cache: dict = {}

    def _commit_mass(spot: "Spot") -> float | None:
        try:
            import jax
            import jax.numpy as jnp
            import numpy as _np

            from pokerbot.engine.jax_engine import CHIP_UNIT, init_hand

            depth_bb = float(max(2.0, min(spot.depth_bb, 200.0)))
            occ = jnp.array([True] * 7, dtype=bool)
            ante_chips = jnp.int32(round(float(spot.ante) * CHIP_UNIT)) if spot.ante > 0 else jnp.int32(0)
            stacks = jnp.full((7,), int(round(depth_bb * CHIP_UNIT)), dtype=jnp.int32)
            key = jax.random.PRNGKey(spot.deck_seed & 0x7FFFFFFF)
            button = int(spot.deck_seed % 7)
            root = init_hand(key, button=jnp.int32(button), stacks=stacks, occupancy=occ, ante=ante_chips)
            seat = int(root.action_seat)
            root_np = jax.tree.map(lambda x: _np.asarray(x), root)
            probs = policy_fn(root_np, seat)
            if probs is None:
                return None
            fold_check = float(probs[0]) + (float(probs[1]) if probs.shape[0] > 1 else 0.0)
            return float(min(1.0, max(0.0, 1.0 - fold_check)))
        except (KeyError, ValueError, IndexError, np.linalg.LinAlgError, OSError):  # narrow: net serve failure
            _log.exception("[regime_battery] _commit_mass failed; falling back to ICM reference")
            return None

    def policy(spot: "Spot") -> str:
        mass = _commit_mass(spot)
        if mass is None:
            return icm_nash_policy(spot, cache=_icm_cache)  # net-free ICM-aware fallback
        risky = "call" if spot.role == "facing_jam" else "jam"
        return risky if mass >= 0.5 else "fold"

    return policy


def analytic_icm_policy(*, pushfold_max_bb: float = 20.0, iters: int = 1500):
    """The certified analytic ICM push/fold engine as a battery ``policy(spot) -> {'jam','fold','call'}``,
    PLUS a coverage ledger (the fraction of spots the engine actually DECIDED vs deferred).

    This is the NON-CFR analytic tournament model graded head-to-head with the net on this surface. It is
    the certified shallow-stack push/fold specialist: it routes a ``Spot`` through
    :func:`pokerbot.tournament.decide.decide_pushfold` (open-shove for ``role=='first_in'``, call/fold for
    ``role=='facing_jam'``), which speaks the SAME jam/fold/call vocabulary the battery wants. The engine
    DEFERS (returns ``action='defer'``) outside its certified zone — effective stack > ``pushfold_max_bb``
    (deep play) — and on SATELLITE the flat seat ladder is mispriced by the open-shove range solve, so we
    route SATELLITE through the certified seat-bubble model (:func:`_satellite_action`) which IS the
    analytic ICM answer there.

    COVERAGE HONESTY: the returned object carries a mutable ``coverage`` ledger
    (``{'decided', 'deferred', 'total'}``) so the cell can surface ``model_coverage`` and the grade is not
    silently dominated by the fallback. On a DEFER the policy falls back to the net-free ICM-aware reference
    (:func:`icm_nash_policy`) — the closest analytic stand-in for the deep spot — so the battery always
    gets a concrete action, but the deferral is COUNTED. Expect high coverage in short/bubble/ft/hu
    regimes (the push/fold zone) and low coverage deep (the engine abstains by design).
    """
    from pokerbot.tournament.decide import decide_pushfold

    ledger = {"decided": 0, "deferred": 0, "total": 0}
    _icm_cache: dict = {}
    _shove_cache: dict = {}

    def _eff_stack(spot: "Spot") -> float:
        return float(min(spot.stacks[spot.hero], spot.stacks[spot.villain]))

    def policy(spot: "Spot") -> str:
        ledger["total"] += 1
        # SATELLITE: the flat seat ladder is mispriced by the open-shove range solve (jams too wide); the
        # certified seat-bubble model IS the analytic ICM answer, and it always decides — full coverage.
        if spot.fmt is TournamentFormat.SATELLITE:
            ledger["decided"] += 1
            return _satellite_action(spot)
        # Deep/postflop: above the certified push/fold zone the engine DEFERS by design. Count it and fall
        # back to the net-free ICM-aware reference so the battery still gets a concrete action.
        if _eff_stack(spot) > pushfold_max_bb:
            ledger["deferred"] += 1
            return icm_nash_policy(spot, cache=_icm_cache)
        try:
            if spot.role == "first_in":
                behind = [i for i in range(len(spot.stacks))
                          if i != spot.hero and spot.stacks[i] > 0.0]
                dec = decide_pushfold(spot.stacks, spot.payouts, spot.hero, spot.hand_label,
                                      blinds={int(k): float(v) for k, v in spot.blinds.items()},
                                      behind=behind, ante=spot.ante,
                                      pushfold_max_bb=pushfold_max_bb, iters=iters)
            else:  # facing_jam -> call/fold vs the jammer
                dec = decide_pushfold(spot.stacks, spot.payouts, spot.hero, spot.hand_label,
                                      blinds={int(k): float(v) for k, v in spot.blinds.items()},
                                      jammer=spot.villain, behind=[], ante=spot.ante,
                                      pushfold_max_bb=pushfold_max_bb, iters=iters)
        except (KeyError, ValueError, IndexError, np.linalg.LinAlgError, OSError):  # narrow: degenerate solve
            _log.exception("[regime_battery] policy solve failed; deferring to net-free ICM reference")
            ledger["deferred"] += 1
            return icm_nash_policy(spot, cache=_icm_cache)
        if dec.action == "defer":
            ledger["deferred"] += 1
            return icm_nash_policy(spot, cache=_icm_cache)
        ledger["decided"] += 1
        # decide_pushfold returns the role-correct action ('jam'/'fold' first-in, 'call'/'fold' facing).
        return dec.action

    return policy, ledger


def regime_battery_cell(fmt_key: str, regime_key: str, cell_args: dict) -> dict:
    """E6 suite adapter: the regime ΔICM uplift of the cell's policy vs the ICM-naive chip-EV field.

    Called by ``scripts/tournament/eval_tournament_suite._drive_pillars`` as
    ``regime_battery_cell(fmt_key, regime_key, cell_args)`` with ``cell_args`` carrying ``n_seats,
    depth_bb, ante_bb, payouts, icm_pressure_spot, decks, hands, spots, ckpt, seed, smoke, policy``.

    It selects the regime cell for ``(fmt_key, regime_key)`` (SATELLITE -> the satellite seat-bubble cell;
    otherwise the regime depth/ICM-pressure picks bubble / final-table / pushfold / deep-control), builds
    the hero policy (the net via the skew-safe bridge when ``policy=='net'``, else the net-free ICM-aware
    ``icm_nash_policy`` reference), runs the CRN twin-policy battery against the ICM-naive chip-EV Nash
    twin, and returns the ΔICM uplift ($ saved by ICM-awareness) + its 95% CI + the chip-EV side-channel +
    flip-rate.
    """
    cell = "satellite_bubble" if str(fmt_key) == "SATELLITE" \
        else _CELL_FOR_REGIME.get(str(regime_key), "shortstack_pushfold")
    seed = int(cell_args.get("seed", 0))
    smoke = bool(cell_args.get("smoke", False))

    # The suite's canonical per-cell context: the FORMAT-correct, REGIME-sized, pool-NORMALIZED payout
    # vector (``_build_cell_payouts``) + the regime depth/ante/seats. Threading it into the battery is THE
    # fix that makes ``delta_icm_mean`` a physical per-game $ delta that VARIES by format & regime (see
    # ``_canonical_payouts`` / the regime-aware builders). Absent (a bare standalone call) -> legacy scale.
    ctx = _suite_ctx(fmt_key, regime_key, cell_args)

    n_spots = int(cell_args.get("spots", 200))
    if smoke:
        n_spots = min(n_spots, 12)
    n_spots = max(4, n_spots)
    # Runout samples for the equity settle: keep modest under smoke / first read (variance vs wall trade).
    n_runouts = 24 if smoke else 60

    policy_str = str(cell_args.get("policy", "fold"))
    icm_ledger: dict | None = None
    if policy_str == "net":
        hero_policy = _net_spot_policy(cell_args)
        source = "net"
    elif policy_str == "icm":
        # The certified analytic ICM push/fold engine as the hero (graded head-to-head with the net).
        hero_policy, icm_ledger = analytic_icm_policy()
        source = "icm-analytic"
    elif policy_str == "jam":
        hero_policy = always_jam_policy
        source = "anchor-jam"
    else:
        # The net-free ICM-aware reference IS the hero for fold/baseline runs (the ΔICM-vs-chip-EV headline
        # only makes sense against an ICM-aware arm; always-fold would just measure the cost of folding).
        _icm_cache: dict = {}
        hero_policy = lambda s: icm_nash_policy(s, cache=_icm_cache)  # noqa: E731
        source = "icm-nash-reference"

    _chip_cache: dict = {}
    twin = lambda s: chip_ev_nash_twin(s, cache=_chip_cache)  # noqa: E731

    res = run_battery(cell, hero_policy, twin, n_spots=n_spots, seed=seed, n_runouts=n_runouts, ctx=ctx)

    # COVERAGE HONESTY: for the analytic ICM hero, surface the fraction of spots the engine actually
    # DECIDED (vs deferred/fell back), so the grade is not silently dominated by the fallback. The push/fold
    # specialist should have high coverage in short/bubble/ft/hu regimes, low coverage deep.
    model_coverage = None
    if icm_ledger is not None:
        tot = max(1, int(icm_ledger["total"]))
        model_coverage = {
            "decided": int(icm_ledger["decided"]),
            "deferred": int(icm_ledger["deferred"]),
            "total": int(icm_ledger["total"]),
            "fraction": float(icm_ledger["decided"]) / tot,
        }

    out = {
        "fmt": str(fmt_key),
        "regime": str(regime_key),
        "battery_cell": cell,
        "policy_source": source,
        "model_coverage": model_coverage,
        "n": int(res.n),
        # E6 headline: the ΔICM uplift ($-equity ICM-awareness SAVES over the chip-EV field) + CI.
        "delta_icm_mean": float(res.mean),
        "delta_icm_se": float(res.se),
        "delta_icm_ci95": [float(res.ci95[0]), float(res.ci95[1])],
        "verdict": res.verdict,
        # chip-EV side-channel: the SAME paired difference in chips (proves the edge is ICM-structural).
        "chipev_mean": float(res.chipev_mean),
        "chipev_ci95": [float(res.chipev_ci95[0]), float(res.chipev_ci95[1])],
        # divergence quantifiers.
        "flip_rate": float(res.flip_rate),
        "icm_delta_saved": float(res.icm_delta_saved),
        "uplift_zero_frac": float(res.uplift_zero_frac),
        "meta": res.meta,
    }
    return out
