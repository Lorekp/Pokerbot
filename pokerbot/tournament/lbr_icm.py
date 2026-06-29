"""ICM-aware 7-seat Local Best Response -- the **$-exploitability** surface for the tournament net.

WHAT THIS IS (and why it is NOT just ``lbr6`` at 7 seats)
--------------------------------------------------------
``pokerbot.eval.lbr6`` measures the **chip-EV** exploitability of a 6-max strategy: a greedy
best-responder sits at a table of N-1 copies and its win rate is reported in chip ``bb/100``. A
tournament is a **$-EV** game (a chip result only matters through the payout structure -- ICM, satellite
step-payouts, PKO prize+bounty), so a chip-LBR can be badly miscalibrated about what is exploitable:

  * an adversary that wins CHIPS on average can still LOSE $-equity if it busts itself near the money
    (ICM risk premium), and
  * a chip-spewing adversary at a SATELLITE-locked seat (already holding a seat-equivalent stack) cannot
    gain $ at all -- every paid finish is worth one identical seat, so risking chips is pure downside.

This module is the **ICM-LBR**: it REUSES ``lbr6``'s entire decision machinery (``_play_hand``, the
``_Belief`` range tracker, the analytic 1-ply value, the all-in equity settle) -- which is seat-count
agnostic because ``np_engine`` derives the seat count from array shapes (np_engine.py:44-48), so a 7-wide
numpy state flows through ``_play_hand`` unchanged -- and swaps only TWO things:

  (a) its OWN 7-seat root-dealing loop (``init_hand(occupancy=[True]*7, stacks=<cell>, ante=<cell>)``,
      exactly as ``cfr_env.nlhe_7max_env_spec`` deals, button + lbr_seat rotating over all 7 -- do NOT use
      the ``lbr6`` default loop whose ``occ=[s<n_seats for s in range(NUM_PLAYERS=6)]`` caps at 6); and

  (b) the LEAF: instead of ``reward = env.rewards[lbr_seat]/CHIP_UNIT`` (a chip value), it reads the
      per-seat CHIP deltas of the hand and maps them to per-seat **$-equity** deltas under the payout
      structure via ``eval_score.score_chip_to_dollar(pre_stacks, chip_deltas, payouts, fmt)`` -- the
      single certified chip->$ bridge. The all-in branch routes through the SAME mean-preserving runout
      average ``lbr6`` uses (``_board_ev_settle``-style), generalised to ALL seats and seeded with a
      shared ``settle_seed`` so paired arms cancel.

The result carries BOTH valuations side by side (E2): the ``$``-LBR exploitability AND the chip-LBR
exploitability run on the same hands, so where ICM binds you can SEE ``chip_LBR >= $_LBR`` (the ICM risk
premium suppresses the $-exploitability below the chip number).

THE MULTIPLAYER CAVEAT (read this -- it is a guardrail, NOT a certified bound)
-----------------------------------------------------------------------------
``lbr6``'s caveat applies verbatim and is COMPOUNDED by ICM. In a >=3-player game there is no
Nash-exploitability guarantee (Brown & Sandholm 2019); and at a tournament leaf the $-mapping uses a
*pairwise/independent* ICM (Malmuth-Harville at the pre-hand stacks) that is itself a model, plus the
LBR's belief tracker and analytic fold reads are approximations. So a POSITIVE $-exploitability is a
genuine "an adversary CAN extract this many $ from a table of us" certificate; a value ``<= 0`` (within
CI) is the strongest empirical evidence of $-unbeatability multiplayer game theory permits -- a
BEHAVIORAL guardrail, not a proof. The result ``note`` states this.

SEAT-SKEW SAFETY
----------------
When ``policy_fn`` is the trained 7-max net (``neural_serve.make_seven_neural_policy``) the served
``EnvState`` MUST be in the trained 7-seat absolute layout. Our root deals a genuinely-7-live table in
NATURAL absolute order (occupancy ``[True]*7``, button rotating) -- exactly the training layout -- so the
served absolute seat/button one-hots coincide with the trained ones (IDENTITY remap, no permutation).
``tests/test_lbr_icm.py`` carries the round-trip ``encode_info_state`` byte-identity proof (mirroring
``tests/test_seven_neural_serve.py`` / ``tests/test_hu_neural_endgame.py``) so a silent encoder skew
cannot turn the served net into garbage.

Smoke (WSL Debian) -- net-free anchors run on CPU jax, no GPU (the 7-max training holds the GPU):
  source ~/jaxgpu-venv/bin/activate
  cd .
  PYTHONPATH=. POKERBOT_INFO_SUITS=1 JAX_PLATFORMS=cpu \
    python -m pytest tests/test_lbr_icm.py -q
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

_log = logging.getLogger(__name__)

# Bounded abstraction must match the trained table; set BEFORE the encoders import (mirrors lbr6).
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from pokerbot.engine.jax_engine import BB_CHIPS, CHIP_UNIT, SB_CHIPS, init_hand  # noqa: E402
from pokerbot.engine.np_engine import (  # noqa: E402
    SEAT_ALLIN,
    SEAT_FOLDED,
    SEAT_LIVE,
    is_chance_node_np,
    sample_chance_np,
    to_numpy_state,
)

# REUSE the lbr6 decision machinery AS-IS (seat-count agnostic via np_engine array shapes). We import the
# private helpers deliberately: this module is the ICM-aware *driver* around the same per-hand play, not a
# re-implementation of LBR (which would duplicate the belief tracker + analytic value).
from pokerbot.eval.lbr6 import LBR6Config  # noqa: E402

# anchor + payouts are dependency-light (numpy/stdlib); safe to import eagerly.
from pokerbot.tournament.anchor import sevenmax_tournament_anchor  # noqa: E402
from pokerbot.tournament.payouts import TournamentFormat  # noqa: E402

# NOTE: the certified chip->$ scoring core (``tournament.eval_score``) is imported LAZILY inside the driver
# (``_score_dollar`` / ``_verdict``), NOT at module top. ``eval_score`` transitively pulls
# ``rollout_leaf -> sim -> aof_equity_grids`` (a ``numba`` dependency), which the lean WSL jax test venv
# lacks. Deferring it keeps ``import pokerbot.tournament.lbr_icm`` numba-free, so the load-bearing encoder
# round-trip fidelity test (which needs only jax) runs in that env even though the full $-LBR run needs the
# numba-equipped env. We only ever call ``score_chip_to_dollar`` + ``verdict`` from it (NOT ``settle_leaf``
# -- this module does its own np-native per-seat all-in settle, mirroring ``lbr6._board_ev_settle``).

NORMAL = TournamentFormat.NORMAL
SATELLITE = TournamentFormat.SATELLITE
PKO = TournamentFormat.PKO

SEVEN: int = 7

PolicyFn = Callable[[object, int], np.ndarray]


def _eval_score():
    """Lazy handle to the chip->$ scoring core (deferred so ``import lbr_icm`` stays numba-free)."""
    from pokerbot.tournament import eval_score
    return eval_score


# --------------------------------------------------------------------------- #
# Configuration + result
# --------------------------------------------------------------------------- #
@dataclass
class ICMCell:
    """One point on the ICM-LBR $-exploitability surface: the tournament *spot* the hands are dealt at.

    A cell fixes the payout regime (``fmt`` + ``payouts``), the table's pre-hand chip stacks (``pre_stacks``,
    in bb -- the SAME unit the per-hand chip deltas are reported in, so ICM ratios and the $-scale line up),
    and the engine deal parameters (``stack_bb`` each seat is dealt the hand with, ``ante_bb``). The cell is
    what makes the surface PAYOUT-STRUCTURE-aware and lifecycle-aware: vary ``pre_stacks`` from deep+even
    (chip-EV-like, ICM ~ inert) to a bubble/pay-jump spread (ICM binds hard) to probe where the $-LBR
    diverges from the chip-LBR.

    ``pre_stacks`` defaults to a flat ``[stack_bb]*7`` (the always-fold anchor's symmetric table); set it to
    an explicit spread to probe ICM pressure. ``seat_locked`` (SATELLITE) marks a seat already holding a
    seat-equivalent stack -- the C1 probe asserts a chip-spewing adversary there cannot profit in $.
    """

    fmt: TournamentFormat = NORMAL
    payouts: tuple[float, ...] = (50.0, 30.0, 20.0)  # best-first $ ladder (NORMAL/SNG) or flat seats (SAT)
    stack_bb: float = 25.0                            # chips each seat is dealt the hand with (shallow MTT)
    ante_bb: float = 0.125                            # CoinPoker MTT per-player ante (round(BB/8))
    pre_stacks: tuple[float, ...] | None = None       # pre-hand table stacks (bb); None -> flat [stack_bb]*7
    # PKO-only bounty accounting (ignored unless fmt is PKO).
    villain_bounty: float = 0.0
    progressive_frac: float = 0.5
    growth_value: float = 0.5

    def resolved_pre_stacks(self) -> list[float]:
        if self.pre_stacks is not None:
            return [float(s) for s in self.pre_stacks]
        return [float(self.stack_bb)] * SEVEN


@dataclass
class ICMLBRResult:
    """The ICM-LBR $-exploitability surface result for one cell.

    Headline (E1): ``dollar_exploit_per_game`` is the LBR seat's mean $-equity gain per hand (seat-rotated
    over all 7), with its CI. ``mbb_of_dollar_per_game`` scales it x1000 (the tournament analogue of
    mbb/g, in $-equity units), and ``pct_of_buyin`` expresses it as a fraction of one buy-in (``buy_in``
    given) for a structure-agnostic headline.

    E2: ``chip_exploit_bb_per_100`` is the SAME LBR run scored in chip ``bb/100`` (no payout structure), so
    you can read ``chip_exploit >= $_exploit`` exactly where ICM binds (the divergence is the ICM risk
    premium). ``leaf_divergence`` reports the per-seat mean |chip_$ - $| gap.

    C0: ``leaf_conservation`` is the max per-hand |sum of the $-leaf over all seats| -- the $-mapping is
    pool-conserving for NORMAL/SATELLITE, so this is ~0 (a non-zero value flags a leaf that minted/burned
    prize-pool equity).
    """

    dollar_exploit_per_game: float
    dollar_stderr_per_game: float
    ci95: tuple[float, float]
    mbb_of_dollar_per_game: float
    pct_of_buyin: float | None
    chip_exploit_bb_per_100: float
    chip_stderr_bb_per_100: float
    leaf_divergence: float
    leaf_conservation: float
    hands: int
    per_seat: dict
    verdict: str
    note: str = (
        "ICM-aware 7-seat Local Best Response: greedy range-tracked best-responder vs a table of 6 copies "
        "of the target, scored in $-equity under the payout structure (E1). Positive => an adversary can "
        "extract this many $ from a table of us. Multiplayer + ICM has NO Nash-exploitability guarantee "
        "(Brown & Sandholm 2019; ICM is itself a model), so this is a BEHAVIORAL guardrail / lower bound, "
        "NOT a certified bound. chip_exploit_bb_per_100 is the same run in chips: chip >= $ where ICM binds."
    )
    detail: dict = field(default_factory=dict)


from phevaluator import evaluate_cards as _phe  # noqa: E402  (fast C eval; lower rank == stronger)

_CARD_STR = ["23456789TJQKA"[c // 4] + "cdhs"[c % 4] for c in range(52)]


# --------------------------------------------------------------------------- #
# Per-seat chip-delta leaf (mean-preserving all-in settle, generalised to ALL seats)
# --------------------------------------------------------------------------- #
def _terminal_allin_equity_settle(env, locked_board_n: int, settle_rng: np.random.Generator,
                                  settle_samples: int) -> np.ndarray | None:
    """Mean-preserving re-settle of a TERMINAL all-in showdown -> per-seat chip deltas (bb), or ``None``.

    The np_engine resolves an all-in showdown ATOMICALLY: once betting closes it deals every remaining
    board card and emits a terminal with ``rewards`` from a SINGLE internal runout (high variance) -- so
    ``lbr6._board_ev_settle``'s chance-node branch never fires for an all-in (there is no incomplete-board
    chance node to intercept). For the tournament $-EV estimator that single-runout variance is exactly
    what we must remove. This re-settles the showdown by EQUITY: hold the first ``locked_board_n`` board
    cards (the ones public at betting-close) fixed and average each contender's pot share over
    ``settle_samples`` completions of the remaining ``5 - locked_board_n`` cards. Each runout is equally
    likely, so the mean equals the engine's single-runout expectation -- a mean-preserving variance cut --
    and the shared ``settle_rng`` (seeded by ``settle_seed``) lets paired arms cancel under common randoms.

    SCOPE (returns ``None`` to fall back to ``rewards`` when not applicable):
      * only for a genuine multiway all-in SHOWDOWN (>= 2 contenders not folded, with holes);
      * only for the EQUAL-contribution main-pot case (every contender committed the same chips => one
        pot, no side pots). Unequal contributions (side pots from differing pre-stacks) are left to the
        engine's exact (single-runout) settle -- correct, just higher variance. In this harness the deal is
        equal-stack, so the all-in is single-pot and this path applies to the common case.
    """
    ss = np.asarray(env.seat_status)
    n = ss.shape[0]
    committed = np.asarray(env.committed_total, dtype=np.int64)
    folded = (ss == int(SEAT_FOLDED))
    contenders = [i for i in range(n) if (not folded[i]) and int(np.asarray(env.holes)[i, 0]) >= 0
                  and int(np.asarray(env.holes)[i, 1]) >= 0]
    if len(contenders) < 2:
        return None
    # EQUAL-contribution main-pot only (no side pots): every contender committed the same, and folders'
    # dead chips all go to the single pot.
    c_commit = committed[contenders]
    if not (c_commit == c_commit[0]).all():
        return None
    board_all = np.asarray(env.board)
    locked = [int(c) for c in board_all[:locked_board_n].tolist() if int(c) >= 0]
    need = 5 - len(locked)
    if need < 0:
        return None
    holes = {i: (int(np.asarray(env.holes)[i, 0]), int(np.asarray(env.holes)[i, 1])) for i in contenders}
    pot_chips = int(committed.sum())                      # main pot = everyone's committed + dead
    dead_used = set(locked) | {c for i in contenders for c in holes[i]}
    deck = [c for c in range(52) if c not in dead_used]

    acc = {i: 0.0 for i in contenders}                    # expected chips WON (gross) per contender
    samples = max(1, int(settle_samples)) if need > 0 else 1
    locked_s = [_CARD_STR[c] for c in locked]
    hole_s = {i: (_CARD_STR[holes[i][0]], _CARD_STR[holes[i][1]]) for i in contenders}
    for _ in range(samples):
        if need > 0:
            pick = settle_rng.choice(len(deck), size=need, replace=False)
            full = locked_s + [_CARD_STR[deck[int(d)]] for d in pick]
        else:
            full = locked_s
        ranks = {i: _phe(hole_s[i][0], hole_s[i][1], *full) for i in contenders}
        best = min(ranks.values())
        winners = [i for i in contenders if ranks[i] == best]
        share = pot_chips / len(winners)
        for w in winners:
            acc[w] += share
    won = {i: acc[i] / samples for i in contenders}        # expected gross chips won
    deltas = np.zeros(n, dtype=np.float64)
    for i in range(n):
        gross = won.get(i, 0.0)
        deltas[i] = (gross - float(committed[i])) / CHIP_UNIT   # net = won - committed, in bb
    return deltas


def _per_seat_chip_deltas_bb(env, settle_rng: np.random.Generator, settle_samples: int,
                             locked_board_n: int | None = None) -> np.ndarray:
    """Per-seat chip delta (in **bb**) of a finished / all-in-locked hand.

    Three cases, generalising ``lbr6._board_ev_settle`` to ALL seats (the $-leaf needs the whole vector to
    price the payout structure, not just the LBR seat):

      * TERMINAL all-in SHOWDOWN (``env.done`` + ``locked_board_n`` given + >=2 contenders, equal main pot):
        mean-preserving EQUITY re-settle over board completions (``_terminal_allin_equity_settle``) -- the
        variance cut the engine's atomic single-runout settle does not provide. Falls back to ``rewards`` if
        the showdown has side pots / <2 contenders.
      * TERMINAL otherwise (a fold-out, or a side-pot all-in): read ``env.rewards`` directly (exact).
      * NON-terminal all-in-locked chance node (the ``lbr6`` settle branch, when it DOES fire on a later
        street): average ``env.rewards`` over ``settle_samples`` runouts of the undealt deck tail.

    Returns a length-``n_seats`` float array of bb deltas summing to ~0 (the chip game is zero-sum).
    """
    rewards_bb = lambda e: np.asarray(e.rewards, dtype=np.float64) / CHIP_UNIT  # noqa: E731

    if bool(env.done):
        if locked_board_n is not None and settle_samples > 1:
            re = _terminal_allin_equity_settle(env, locked_board_n, settle_rng, settle_samples)
            if re is not None:
                return re
        return rewards_bb(env)

    ptr = int(env.deck_ptr)
    base_deck = np.asarray(env.deck).copy()
    tail = base_deck[ptr:].copy()
    if tail.shape[0] == 0:
        return rewards_bb(env)

    n = np.asarray(env.seat_status).shape[0]
    total = np.zeros(n, dtype=np.float64)
    samples = max(1, int(settle_samples))
    for _ in range(samples):
        settle_rng.shuffle(tail)
        d = base_deck.copy()
        d[ptr:] = tail
        e = env._replace(deck=d)
        guard = 0
        while not bool(e.done) and guard < 8:
            if is_chance_node_np(e):
                e = sample_chance_np(e)
            else:
                break
            guard += 1
        total += rewards_bb(e)
    return total / samples


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def icm_lbr_exploitability(
    policy_fn: PolicyFn,
    cfg: LBR6Config,
    payouts: Sequence[float],
    pre_stacks: Sequence[float],
    *,
    fmt: TournamentFormat = NORMAL,
    cell: ICMCell | None = None,
    model_fn: PolicyFn | None = None,
    buy_in: float | None = None,
    seat_locked: int | None = None,
) -> ICMLBRResult:
    """E1 centerpiece: the ICM-aware 7-seat Local Best Response $-exploitability of ``policy_fn``.

    ``policy_fn(env_np, seat) -> probs`` (len ``NUM_ACTIONS``, sums to 1 over legal) is played by the 6
    non-LBR seats at a genuinely-7-live table. The LBR seat rotates over all 7 hands so every position is
    best-responded from. The per-hand result is scored TWICE: in $-equity (the headline, via
    ``eval_score.score_chip_to_dollar`` under ``fmt`` + ``payouts``) AND in chips (the E2 comparison).

    Two-policy split (the feasibility key when the net is expensive to model): ``policy_fn`` is what the
    seats ACT with (the strategy whose exploitability we measure); ``model_fn`` (default = ``policy_fn``,
    but pass the cheap blueprint when serving the net) is the cheap white-box the LBR reads internally for
    range tracking + fold-equity. ``cfg.value_only_raises`` should be True when ``policy_fn`` is the net so
    the LBR never suicide-bluffs into an opponent it cannot model cheaply (lbr6 docstring).

    Args:
        cfg: a ``LBR6Config``. Its ``n_seats`` is FORCED to 7 (the tournament table); ``stack_bb``/``ante_bb``
            are read from the ``cell`` (or the explicit args) -- this driver deals its OWN 7-seat root.
        payouts: best-first $ ladder (NORMAL/SNG/PKO) or flat ``[seat_value]*k`` (SATELLITE step-payout).
        pre_stacks: per-seat pre-hand table stacks (bb) -- the ICM context the chip deltas are priced in.
        fmt: the payout regime (``payouts`` is interpreted under it).
        cell: optional ``ICMCell`` overriding ``payouts``/``pre_stacks``/``fmt``/deal params in one object.
        model_fn: cheap white-box for the LBR's internal reads (defaults to ``policy_fn``).
        buy_in: if given, the headline is also expressed as ``pct_of_buyin``.
        seat_locked: if given (the SATELLITE C1 probe), the per-seat breakdown flags that seat so the
            caller / test can assert a chip-spewing adversary there has $-EV <= 0.

    Returns:
        An ``ICMLBRResult`` carrying the $-exploitability (E1), the chip-vs-$ leaf divergence (E2), the
        always-fold anchor recovery target, and the leaf conservation (C0).
    """
    # The cell is the single source of the tournament SPOT (payout regime, deal depth, ante). When the
    # caller passes an explicit ``cell`` it wins outright; otherwise we synthesise the cell from the
    # explicit ``fmt``/``payouts``/``pre_stacks`` args + the deal params, defaulting the deal depth/ante to
    # the CoinPoker MTT tournament defaults (ICMCell's defaults: stack_bb=25 shallow, ante_bb=0.125). This
    # is the load-bearing fix vs a naive ``cfg.ante_bb`` read: LBR6Config's ante defaults to the CASH 0.0,
    # which would deal an ANTE-LESS table and silently undershoot the always-fold anchor (no antes to
    # steal). cfg.stack_bb / cfg.ante_bb, if set, override the cell deal depth/ante (the tournament-shallow
    # knob LBR6Config already documents).
    if cell is None:
        cell = ICMCell(
            fmt=fmt,
            payouts=tuple(float(p) for p in payouts),
            pre_stacks=tuple(float(s) for s in pre_stacks),
            stack_bb=(cfg.stack_bb if cfg.stack_bb is not None else ICMCell.stack_bb),
            ante_bb=(cfg.ante_bb if cfg.ante_bb else ICMCell.ante_bb),
        )
    fmt = cell.fmt
    payouts = [float(p) for p in cell.payouts]
    pre = cell.resolved_pre_stacks()
    stack_bb = cell.stack_bb
    ante_bb = cell.ante_bb

    if len(pre) != SEVEN:
        raise ValueError(f"pre_stacks must have {SEVEN} entries (one per seat), got {len(pre)}")

    model_fn = model_fn or policy_fn
    # Force the 7-seat table; honour the rest of cfg (belief cap, analytic flag, value_only_raises, ...).
    cfg.n_seats = SEVEN

    es = _eval_score()  # lazy chip->$ scoring core (one import; numba-tainted, deferred off module top)
    key = jax.random.PRNGKey(cfg.seed)
    rng = np.random.default_rng(cfg.seed + 1)
    occ = jnp.array([True] * SEVEN, dtype=bool)               # ALL 7 seats live (training layout)
    chip_stacks = jnp.full((SEVEN,), int(round(stack_bb * CHIP_UNIT)), dtype=jnp.int32)
    ante_chips = jnp.int32(int(round(ante_bb * CHIP_UNIT)))

    dollar = np.zeros(cfg.hands, dtype=np.float64)             # LBR seat $-equity gain per hand (headline)
    chip = np.zeros(cfg.hands, dtype=np.float64)               # LBR seat chip bb per hand (E2)
    leaf_div = np.zeros(cfg.hands, dtype=np.float64)           # per-hand |chip_$leaf - $leaf| on LBR seat
    leaf_cons = np.zeros(cfg.hands, dtype=np.float64)          # per-hand |sum of $-leaf over all seats| (C0)
    lbr_seats = np.zeros(cfg.hands, dtype=np.int64)
    lbr_decisions = 0

    for h in range(cfg.hands):
        key, k = jax.random.split(key)
        # Rotate the BUTTON over all 7 (the training deal) AND the LBR seat over all 7 INDEPENDENTLY, so the
        # adversary is best-responded from every position relative to every button (covers BTN-steal through
        # blind-defense -- the full positional lifecycle, not just on-the-button). button advances each hand;
        # lbr_seat sweeps every offset from it via the (h//7) outer index, so over 49 hands every
        # (button, lbr_seat) pair is visited once.
        button = h % SEVEN
        lbr_seat = (h % SEVEN + h // SEVEN) % SEVEN
        env0 = to_numpy_state(
            init_hand(k, button=jnp.int32(button), stacks=chip_stacks, occupancy=occ, ante=ante_chips)
        )
        lbr_seats[h] = lbr_seat
        settle_seed = (cfg.seed * 1_000_003 + h) & 0x7FFFFFFF  # shared runout draws => paired arms cancel

        # ONE leaf-capturing replay drives the whole hand: it REUSES the lbr6 decision helpers (analytic
        # LBR action + policy sampler + belief tracker) and threads the terminal / all-in-locked env out so
        # we read the WHOLE-TABLE per-seat chip-delta vector (which _play_hand only returns for the LBR
        # seat). chip + $ are scored on the SAME replayed hand, so E2 and C0 are exact on one outcome.
        leaf_bb, lbr_from_leaf, nd, _info = _play_hand_capturing_leaf(
            env0, lbr_seat, policy_fn, cfg, rng=rng, model_fn=model_fn, settle_seed=settle_seed,
        )
        lbr_decisions += nd
        chip_deltas_bb = leaf_bb

        # Score the actual hand's per-seat chip delta as an ICM-PRIZE delta (NORMAL/SATELLITE/PKO-prize).
        # NOTE on PKO: ``score_chip_to_dollar``'s PKO branch prices a CLEAN hero-vs-villain all-in CALL
        # (``pko_call_dollar_ev`` over the smaller-stack win/lose post-stacks), NOT an arbitrary multiway
        # chip delta -- feeding it the LBR's general per-seat deltas mis-scores even at ``villain_bounty=0``
        # (it would NOT reduce to the ICM delta of the realised outcome). So the LBR $-leaf scores the PRIZE
        # side via the ICM delta of the realised chips (correct for the prize ladder) and leaves the BOUNTY
        # side UNCREDITED for now (the LBR does not yet model which KO it scored). That makes PKO's prize
        # exploitability exact and PKO(bounty=0) identical to NORMAL; the bounty uplift is a documented
        # PARTIAL (``detail['pko_bounty_credited']=False``). NORMAL/SATELLITE are fully scored here.
        d_vec = es.score_chip_to_dollar(pre, chip_deltas_bb.tolist(), payouts, NORMAL
                                        if fmt is PKO else fmt)

        dollar[h] = float(d_vec[lbr_seat])
        chip[h] = float(lbr_from_leaf)
        # Chip-$ leaf: the LBR seat's chip delta valued as a raw chip number (bb), vs its $-leaf value.
        leaf_div[h] = abs(float(lbr_from_leaf) - float(d_vec[lbr_seat]))
        leaf_cons[h] = abs(float(np.sum(d_vec)))

    return _assemble_result(
        dollar, chip, leaf_div, leaf_cons, lbr_seats, cfg, fmt, payouts, pre,
        ante_bb, stack_bb, buy_in, seat_locked, lbr_decisions, model_fn is not policy_fn,
    )


# --------------------------------------------------------------------------- #
# Leaf-capturing replay (per-seat chip deltas under a shared settle seed)
# --------------------------------------------------------------------------- #
def _play_hand_capturing_leaf(env0, lbr_seat: int, policy_fn, cfg: LBR6Config, *,
                              rng: np.random.Generator, model_fn, settle_seed: int):
    """Play one LBR-vs-6-copies hand; return ``(per_seat_chip_delta_bb, lbr_seat_bb, n_decisions, info)``.

    This is a thin driver that REUSES the lbr6 decision helpers (the analytic LBR action + the policy
    sampler + the belief tracker) but threads the terminal / all-in-locked ENV out so we can read the
    WHOLE-TABLE chip-delta vector (``_per_seat_chip_deltas_bb``) -- which ``lbr6._play_hand`` does not
    expose (it returns only the LBR seat's reward). The decision logic is byte-for-byte the same as
    ``lbr6._play_hand`` (same helpers, same belief pruning, same all-in settle branch); the ONLY additions
    are (1) returning the per-seat vector and (2) the ``info`` dict (``folded`` / signed ``reward``) the
    PKO showdown-equity proxy reads. The all-in settle uses the SAME mean-preserving runout average seeded
    by ``settle_seed`` so a paired base arm cancels under common random numbers.

    ``rng`` is the SHARED driver RNG (advanced across hands, exactly as ``lbr6.lbr6_exploitability`` threads
    it) so the hand sequence is a single faithful Monte-Carlo stream, not a per-hand reseed.
    """
    from pokerbot.engine.np_engine import FOLD, apply_action_np
    from pokerbot.eval.lbr6 import (
        _init_beliefs,
        _lbr_action,
        _sample_policy_action,
    )

    def _board_n(e) -> int:
        b = np.asarray(e.board)
        return int((b >= 0).sum())

    settle_rng = np.random.default_rng(settle_seed)
    beliefs = _init_beliefs(env0, lbr_seat, cfg, rng) if cfg.track_ranges else None
    env = env0
    depth = 0
    nd = 0
    folded = False
    # Board card count public at the LAST decision before terminal. When a closing action sends the hand
    # straight to a terminal all-in showdown (the np_engine deals the runout atomically -- no incomplete-
    # board chance node to intercept), this is the betting-CLOSED board, the cards we must hold fixed in the
    # mean-preserving equity re-settle. Updated to the pre-action board count on every action.
    locked_board_n = _board_n(env)
    while depth < cfg.max_depth and not bool(env.done):
        if is_chance_node_np(env):
            # All live seats all-in (betting over, board incomplete) -> mean-preserving settle leaf (this
            # branch fires only when the engine DOES pause at an incomplete-board chance node; for a typical
            # all-in the engine resolves atomically and we re-settle at the terminal below instead).
            if int((np.asarray(env.seat_status) == SEAT_LIVE).sum()) == 0 and cfg.settle_samples > 1:
                leaf = _per_seat_chip_deltas_bb(env, settle_rng, cfg.settle_samples, _board_n(env))
                info = {"folded": folded, "reward": float(leaf[lbr_seat]), "showdown": not folded}
                return leaf, float(leaf[lbr_seat]), nd, info
            env = sample_chance_np(env)
            if beliefs is not None:
                board = np.asarray(env.board)
                blocked = set(int(c) for c in board[board >= 0].tolist())
                for b in beliefs.values():
                    b.prune(blocked)
            continue
        seat = int(env.action_seat)
        if seat < 0:
            break
        if seat == lbr_seat:
            a = _lbr_action(env, lbr_seat, beliefs, model_fn, cfg, rng)
            nd += 1
            if a == FOLD:
                folded = True
        else:
            a = _sample_policy_action(env, seat, policy_fn, rng)
            if beliefs is not None and seat in beliefs:
                beliefs[seat].bayes_update(env, seat, a, model_fn, cfg.range_cap, rng)
        locked_board_n = _board_n(env)   # board public BEFORE this (possibly betting-closing) action
        env = apply_action_np(env, a)
        depth += 1
    # Terminal: re-settle an all-in showdown by equity over completions of the betting-closed board.
    leaf = _per_seat_chip_deltas_bb(env, settle_rng, cfg.settle_samples, locked_board_n)
    info = {"folded": folded, "reward": float(leaf[lbr_seat]), "showdown": not folded}
    return leaf, float(leaf[lbr_seat]), nd, info


# --------------------------------------------------------------------------- #
# Result assembly
# --------------------------------------------------------------------------- #
def _assemble_result(dollar, chip, leaf_div, leaf_cons, lbr_seats, cfg, fmt, payouts, pre,
                     ante_bb, stack_bb, buy_in, seat_locked, lbr_decisions, two_policy):
    n = cfg.hands
    d_mean = float(dollar.mean()) if n else 0.0
    d_se = float(dollar.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    ci = (d_mean - 1.96 * d_se, d_mean + 1.96 * d_se)
    c_mean = float(chip.mean()) if n else 0.0
    c_se = float(chip.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    per_seat = {}
    for s in range(SEVEN):
        mask = lbr_seats == s
        ds = dollar[mask]
        cs = chip[mask]
        if ds.size:
            per_seat[s] = {
                "n": int(ds.size),
                "dollar_per_game": float(ds.mean()),
                "dollar_se": float(ds.std(ddof=1) / np.sqrt(ds.size)) if ds.size > 1 else 0.0,
                "chip_bb_per_100": float(cs.mean()) * 100.0,
                "seat_locked": (seat_locked is not None and s == seat_locked),
            }

    return ICMLBRResult(
        dollar_exploit_per_game=d_mean,
        dollar_stderr_per_game=d_se,
        ci95=ci,
        mbb_of_dollar_per_game=d_mean * 1000.0,
        pct_of_buyin=(100.0 * d_mean / float(buy_in)) if (buy_in and buy_in > 0) else None,
        chip_exploit_bb_per_100=c_mean * 100.0,
        chip_stderr_bb_per_100=c_se * 100.0,
        leaf_divergence=float(leaf_div.mean()) if n else 0.0,
        leaf_conservation=float(leaf_cons.max()) if n else 0.0,
        hands=n,
        per_seat=per_seat,
        verdict=_eval_score().verdict(ci[0], ci[1]),
        detail={
            "fmt": fmt.value,
            "payouts": list(payouts),
            "pre_stacks": list(pre),
            "stack_bb": stack_bb,
            "ante_bb": ante_bb,
            "n_seats": SEVEN,
            "two_policy_split": bool(two_policy),
            "value_only_raises": cfg.value_only_raises,
            "track_ranges": cfg.track_ranges,
            "settle_samples": cfg.settle_samples,
            "lbr_decisions": lbr_decisions,
            "seat_locked": seat_locked,
            # PKO is scored on the PRIZE side only (ICM delta of the realised chips); the progressive-bounty
            # uplift is NOT yet credited in the LBR leaf (the LBR does not model which KO it scored), so a
            # PKO surface here measures prize-side $-exploitability and equals NORMAL at bounty=0. PARTIAL.
            "pko_bounty_credited": False,
        },
    )


# --------------------------------------------------------------------------- #
# Net-free anchor policies (calibration floors -- no GPU, no checkpoint)
# --------------------------------------------------------------------------- #
def always_fold_policy(env_np, seat: int) -> np.ndarray:
    """Pure folder: all mass on FOLD when legal, else CHECK (a folder cannot fold a free check).

    The always-fold anchor's $-exploitability MUST recover ``sevenmax_tournament_anchor()`` converted to
    $ at the cell's payout scale (the always-steal best response scoops the dead money every hand)."""
    from pokerbot.engine.np_engine import CHECK, FOLD, legal_action_mask_np
    from pokerbot.engine.actions import NUM_ACTIONS

    mask = np.asarray(legal_action_mask_np(env_np, seat), dtype=bool)
    out = np.zeros(NUM_ACTIONS, dtype=np.float64)
    if mask[FOLD]:
        out[FOLD] = 1.0
    elif mask[CHECK]:
        out[CHECK] = 1.0
    else:
        out[np.argmax(mask)] = 1.0
    return out


def always_jam_policy(env_np, seat: int) -> np.ndarray:
    """Pure jammer: all mass on ALL_IN when legal, else CALL, else CHECK -- the chip-spew adversary.

    Used by the C1 satellite probe: at a seat-locked SATELLITE the jammer cannot profit in $ (every paid
    finish is one identical seat, so risking chips is pure downside)."""
    from pokerbot.engine.np_engine import ALL_IN, CALL, CHECK, legal_action_mask_np
    from pokerbot.engine.actions import NUM_ACTIONS

    mask = np.asarray(legal_action_mask_np(env_np, seat), dtype=bool)
    out = np.zeros(NUM_ACTIONS, dtype=np.float64)
    for a in (ALL_IN, CALL, CHECK):
        if mask[a]:
            out[a] = 1.0
            return out
    out[np.argmax(mask)] = 1.0
    return out


# --------------------------------------------------------------------------- #
# Anchor recovery target (the calibration floor, in $)
# --------------------------------------------------------------------------- #
def anchor_dollar_target(payouts: Sequence[float], pre_stacks: Sequence[float], *,
                         ante_bb: float = 0.125) -> float:
    """The always-fold anchor expressed in **$-equity per hand** at this cell.

    The chip anchor ``sevenmax_tournament_anchor(ante_bb)`` is +203.6 bb/100 = +2.036 bb/hand of dead money
    the always-steal BR scoops. To express it in $ we map that average chip win (and the symmetric losses
    of the 6 folders) through ICM at the cell's ``pre_stacks``. Because the steal is small relative to the
    stacks and ICM is ~linear there, the $ target is ``chip_per_hand_bb * (local ICM $/bb slope at the LBR
    seat)``. We compute the slope by a tiny finite difference of ``icm_equity`` at the (flat) pre-stacks --
    the exact local conversion the leaf uses -- so the test can assert the always-fold run lands on this
    value within CI rather than hard-coding a number that drifts with the cell.
    """
    from pokerbot.tournament.icm import icm_equity

    chip_per_hand_bb = sevenmax_tournament_anchor(ante_bb) / 100.0  # bb/hand of dead money stolen
    pre = [float(s) for s in pre_stacks]
    # Local $/bb slope at seat 0 (flat table -> same at every seat): how many $ one extra bb is worth.
    eps = max(1e-3, 0.01 * (sum(pre) / len(pre)))
    up = list(pre); up[0] += eps
    dn = list(pre); dn[0] -= eps
    slope = (float(icm_equity(up, payouts)[0]) - float(icm_equity(dn, payouts)[0])) / (2.0 * eps)
    return chip_per_hand_bb * slope


# --------------------------------------------------------------------------- #
# Suite entry point (E1/E2): the format x regime driver calls THIS per cell.
# --------------------------------------------------------------------------- #
_FMT_BY_KEY = {"NORMAL": NORMAL, "SNG": NORMAL, "SATELLITE": SATELLITE, "PKO": PKO}


def make_analytic_icm_policy(payouts: Sequence[float], pre_stacks: Sequence[float], *,
                             ante_bb: float = 0.125, pushfold_max_bb: float = 20.0, iters: int = 1500):
    """Build ``(policy_fn, coverage_ledger)``: the certified analytic ICM push/fold engine wrapped as an
    ``policy_fn(env_np, seat) -> NUM_ACTIONS-vec`` for the ICM-LBR's two-policy split.

    This is the env-wrapper the spec flags as the fiddliest — and it is DELIBERATELY a SAFE SUBSET. The
    certified engine (:func:`decide.decide_pushfold`) speaks a preflop push/fold abstraction over the FULL
    field's bb stacks + a 169-bucket hand. From a raw ``EnvState`` we can faithfully reconstruct that spot
    ONLY for the clean preflop push/fold cases:

      * PREFLOP ONLY (``street == 0``): postflop nodes are outside the engine's certified zone -> defer.
      * FIRST-IN open-shove: folded to hero, no seat has committed beyond the big blind (only blinds/antes
        posted). ``behind`` = the live seats yet to act after hero.
      * FACING A SINGLE JAM: exactly one seat is all-in for more than the big blind, every other live seat
        is still at its blind (a clean single jam the hero faces). Multiway / limped / min-raised / re-jam
        pots are NOT push/fold and are deferred.

    For a DECIDED spot it maps the engine's action to the engine's vocabulary: ``fold -> FOLD`` (idx 0),
    ``jam`` / ``call`` -> ``ALL_IN`` (the push/fold abstraction commits the stack). For ANY deferral
    (postflop, multiway, ambiguous betting, > ``pushfold_max_bb`` deep, a degenerate solve, or an
    unreadable hand) it returns :func:`always_fold_policy` — the SAFE white-box fallback (the LBR's
    two-policy split already uses always-fold as the cheap internal model), so the wrapper NEVER wraps a
    spot it cannot faithfully reconstruct.

    COVERAGE HONESTY: the returned ``coverage`` ledger (``{'decided','deferred','total'}``) counts a DECIDE
    only when the engine actually solved a clean preflop push/fold spot; everything else is DEFERRED. The
    push/fold specialist should DECIDE most short/bubble/ft preflop nodes and DEFER deep / postflop ones.
    """
    from pokerbot.engine.actions import NUM_ACTIONS
    from pokerbot.engine.np_engine import ALL_IN, FOLD
    from pokerbot.tournament.decide import decide_pushfold
    from pokerbot.tournament.pushfold_icm import hand_to_label

    pay = [float(p) for p in payouts]
    ledger = {"decided": 0, "deferred": 0, "total": 0}

    def _hand_label(holes_seat) -> str | None:
        c0, c1 = int(holes_seat[0]), int(holes_seat[1])
        if c0 < 0 or c1 < 0:
            return None
        try:
            return hand_to_label(_CARD_STR[c0] + _CARD_STR[c1])
        except (IndexError, ValueError):  # narrow: encoder-skew must surface, not be swallowed
            return None

    def _detect(env_np, seat: int):
        """Return ('open_shove'|'call', behind, jammer, stacks_bb, hand_label) or None (defer)."""
        if int(np.asarray(env_np.street)) != 0:
            return None  # postflop -> outside the certified push/fold zone
        board = np.asarray(env_np.board)
        if int((board >= 0).sum()) != 0:
            return None  # any community card -> not a preflop spot
        ss = np.asarray(env_np.seat_status)
        n = ss.shape[0]
        if seat < 0 or seat >= n:
            return None
        holes = np.asarray(env_np.holes)
        hand = _hand_label(holes[seat])
        if hand is None:
            return None
        committed = np.asarray(env_np.committed_total, dtype=np.float64) / CHIP_UNIT
        stacks_bb = (np.asarray(env_np.stacks, dtype=np.float64) / CHIP_UNIT) + committed  # hand-start bb
        bb_unit = float(BB_CHIPS) / CHIP_UNIT  # 1.0 bb
        # The largest FORCED post is BB + ante (every dealt seat posts the ante, so the BB's committed
        # exceeds 1.0 bb by the ante). A seat is a voluntary RAISER only if it committed beyond that
        # forced post — comparing against bb_unit alone would mis-flag the BB seat (BB+ante > 1.0). The
        # ante is read off the env as the smallest positive commit (every live seat posted it).
        live_committed = [committed[i] for i in range(n) if committed[i] > 1e-9]
        ante_bb_seen = min(live_committed) if live_committed else 0.0
        forced_post = bb_unit + ante_bb_seen + 1e-6  # BB + ante, the max non-voluntary commit
        live = [i for i in range(n) if int(ss[i]) in (int(SEAT_LIVE), int(SEAT_ALLIN))]
        if seat not in live or len(live) < 2:
            return None
        allin_seats = [i for i in range(n) if int(ss[i]) == int(SEAT_ALLIN)
                       and committed[i] > forced_post]
        # Seats that have voluntarily committed beyond a blind (a raise/jam), excluding the all-in jam.
        raisers = [i for i in range(n) if committed[i] > forced_post and i not in allin_seats]
        if raisers:
            return None  # a limp-raise / min-raise / re-open -> not the push/fold abstraction
        # behind = live seats acting AFTER hero that have not folded (use seat order from action_seat on).
        behind = [i for i in live if i != seat and int(ss[i]) == int(SEAT_LIVE)]
        if not allin_seats:
            # FIRST-IN open shove: only blinds posted ahead, hero first to voluntarily act.
            return ("open_shove", behind, None, stacks_bb.tolist(), hand)
        if len(allin_seats) == 1 and allin_seats[0] != seat:
            # FACING A SINGLE JAM: one jammer all-in beyond the blind, hero to act.
            jammer = allin_seats[0]
            behind2 = [i for i in behind if i != jammer]
            return ("call", behind2, jammer, stacks_bb.tolist(), hand)
        return None  # multiway all-in / re-jam -> defer

    def _blinds_from_env(env_np, n: int) -> dict[int, float]:
        """Recover {sb_seat: 0.5, bb_seat: 1.0} from committed blinds (the two smallest positive non-ante
        commits). Falls back to a button-relative convention if the read is ambiguous."""
        committed = np.asarray(env_np.committed_total, dtype=np.float64) / CHIP_UNIT
        bb_unit = float(BB_CHIPS) / CHIP_UNIT
        sb_unit = float(SB_CHIPS) / CHIP_UNIT
        out: dict[int, float] = {}
        for i in range(n):
            # match against the posted-blind amounts (ante is smaller; a jam is larger).
            if abs(committed[i] - bb_unit) < 0.26:
                out[i] = 1.0
            elif abs(committed[i] - sb_unit) < 0.26 and committed[i] < bb_unit - 1e-9:
                out[i] = 0.5
        return out

    def policy_fn(env_np, seat: int) -> np.ndarray:
        ledger["total"] += 1
        det = _detect(env_np, seat)
        if det is None:
            ledger["deferred"] += 1
            return always_fold_policy(env_np, seat)
        kind, behind, jammer, stacks_bb, hand = det
        eff_pool = [stacks_bb[seat]] + [stacks_bb[o] for o in
                                        (behind if kind == "open_shove" else ([jammer] + list(behind)))]
        if float(min(eff_pool)) > pushfold_max_bb:
            ledger["deferred"] += 1  # deep -> the engine abstains by design
            return always_fold_policy(env_np, seat)
        n = np.asarray(env_np.seat_status).shape[0]
        blinds = _blinds_from_env(env_np, n) or {seat: 0.5}
        try:
            if kind == "open_shove":
                dec = decide_pushfold(stacks_bb, pay, seat, hand, blinds=blinds, behind=behind,
                                      ante=ante_bb, pushfold_max_bb=pushfold_max_bb, iters=iters)
            else:
                dec = decide_pushfold(stacks_bb, pay, seat, hand, blinds=blinds, jammer=jammer,
                                      behind=behind, ante=ante_bb, pushfold_max_bb=pushfold_max_bb,
                                      iters=iters)
        except (KeyError, ValueError, IndexError, np.linalg.LinAlgError, OSError):  # narrow: solver/API failure -> safe fallback
            _log.exception("[lbr_icm] degenerate pushfold solve (seat=%d kind=%s); returning fold anchor", seat, kind)
            ledger["deferred"] += 1
            ledger.setdefault("degraded_cells", []).append({"seat": int(seat), "kind": str(kind)})
            return always_fold_policy(env_np, seat)
        if dec.action == "defer":
            ledger["deferred"] += 1
            return always_fold_policy(env_np, seat)
        ledger["decided"] += 1
        out = np.zeros(NUM_ACTIONS, dtype=np.float64)
        if dec.action == "fold":
            out[FOLD] = 1.0
        else:  # 'jam' / 'call' -> commit the stack (the push/fold abstraction is all-in)
            out[ALL_IN] = 1.0
        return out

    return policy_fn, ledger


def _build_cell_policy(cell_args: dict, *, payouts=None, pre_stacks=None, ante_bb: float = 0.125):
    """Build ``(policy_fn, model_fn, two_policy, source, coverage)`` for one cell from
    ``cell_args['policy']``.

    * ``policy=='net'``  -> serve the 7-max net via the skew-proven ``neural_serve`` bridge (the ONLY
      checkpoint seam). The expensive net is what the seats ACT with; the LBR's internal white-box reads
      (range tracking + fold equity) use the cheap ``always_fold`` model so the LBR never suicide-bluffs
      into a net it cannot model cheaply (the two-policy split lbr6 documents). ``value_only_raises`` is
      forced on for the same reason.
    * ``policy=='icm'``  -> the certified analytic ICM push/fold engine wrapped as a ``policy_fn``
      (:func:`make_analytic_icm_policy`). The white-box model reads use the cheap always-fold (the same
      two-policy split as the net). Returns the engine's coverage ledger so the cell can surface
      ``model_coverage`` (the fraction of LBR decision nodes the engine actually decided vs deferred to the
      always-fold fallback). ``payouts``/``pre_stacks``/``ante_bb`` carry the cell's tournament context.
    * ``policy=='jam'``  -> the net-free always-jam chip-spew adversary policy (calibration ceiling).
    * anything else      -> the net-free always-fold anchor policy (calibration floor).

    Loading the net FAILS LOUD here (raises) so a bad checkpoint surfaces as the cell's ``status:'error'``
    in the suite rather than silently measuring an anchor while claiming ``policy=='net'``.

    Returns a 5-tuple; ``coverage`` is the analytic-ICM ledger (or ``None`` for the other policies).
    """
    policy = str(cell_args.get("policy", "fold"))
    if policy == "net":
        from pokerbot.tournament.neural_serve import make_seven_neural_policy
        policy_fn, _meta = make_seven_neural_policy(cell_args.get("ckpt"))
        # White-box model for the LBR's internal reads: the cheap always-fold (lbr6 two-policy split).
        return policy_fn, always_fold_policy, True, "net", None
    if policy == "icm":
        policy_fn, ledger = make_analytic_icm_policy(
            payouts if payouts is not None else [50.0, 30.0, 20.0],
            pre_stacks if pre_stacks is not None else [25.0] * SEVEN,
            ante_bb=ante_bb)
        # Two-policy split: the analytic engine ACTS; the LBR reads the cheap always-fold white-box.
        return policy_fn, always_fold_policy, True, "icm-analytic", ledger
    if policy == "jam":
        return always_jam_policy, always_jam_policy, False, "anchor-jam", None
    return always_fold_policy, always_fold_policy, False, "anchor-fold", None


def lbr_icm_cell(fmt_key: str, regime_key: str, cell_args: dict) -> dict:
    """E1/E2 suite adapter: the ICM-aware 7-seat $-exploitability of the cell's policy.

    Called by ``scripts/tournament/eval_tournament_suite._drive_pillars`` as
    ``lbr_icm_cell(fmt_key, regime_key, cell_args)``. ``cell_args`` carries (from the driver):
    ``n_seats, depth_bb, ante_bb, payouts, icm_pressure_spot, decks, hands, spots, ckpt, seed, smoke,
    policy``.

    It builds the cell policy (net via ``neural_serve.make_seven_neural_policy`` when
    ``policy=='net'``, else the fold/jam net-free anchor), deals its OWN genuinely-7-live root at the
    cell depth/ante (the ICM-LBR is always a 7-seat table internally), runs ``icm_lbr_exploitability``,
    and returns a JSON-serializable dict with the headline $-exploitability + chip-EV side-channel + 95%
    CI + n + the always-fold $-anchor floor for comparison.

    The work scale comes from ``spots`` (mapped to LBR ``hands``); ``smoke`` clamps the per-cell hands so
    the whole grid finishes in seconds. Never silently fakes a net serve — the net loads loud.
    """
    fmt = _FMT_BY_KEY.get(str(fmt_key), NORMAL)
    depth_bb = float(cell_args.get("depth_bb", 25.0))
    ante_bb = float(cell_args.get("ante_bb", 0.125))
    payouts = [float(p) for p in cell_args.get("payouts", [50.0, 30.0, 20.0])] or [1.0]
    seed = int(cell_args.get("seed", 0))
    smoke = bool(cell_args.get("smoke", False))

    # The ICM-LBR ALWAYS deals a 7-seat table (it forces cfg.n_seats=7 and needs a 7-wide pre_stacks).
    # The cell's n_seats / icm_pressure_spot shape the PRE-HAND table stacks: a flat 7-wide table far from
    # the money (ICM ~ chip-linear), OR an ICM-pressure spread (a few short stacks one bust from the money)
    # at the ICM-pressure regimes so the $-LBR diverges from the chip-LBR exactly where ICM binds.
    icm_spot = bool(cell_args.get("icm_pressure_spot", False))
    if icm_spot and fmt is not SATELLITE:
        # bubble/ft/hu: big covering stacks + short stacks at risk near the money.
        big = depth_bb * 2.5
        pre = [depth_bb] + [big] * 3 + [max(1.0, depth_bb * 0.35)] * 3
    elif fmt is SATELLITE:
        # satellite: seat-locked giants + dust (the C1 stop-gambling geometry).
        pre = [depth_bb * 6.0] * max(1, len(payouts)) + [max(1.0, depth_bb * 0.2)] * 7
        pre = pre[:7] + [max(1.0, depth_bb * 0.2)] * max(0, 7 - len(pre))
        pre = pre[:7]
    else:
        pre = [depth_bb] * SEVEN

    policy_fn, model_fn, two_policy, source, coverage = _build_cell_policy(
        cell_args, payouts=payouts, pre_stacks=pre, ante_bb=ante_bb)

    # hands per cell: drive off `spots` (constructed ICM spots), clamped under smoke. Net serves fast but
    # is still ~10x a net-free anchor, so keep a sane cap.
    hands = int(cell_args.get("spots", 200))
    if smoke:
        hands = min(hands, 21)
    hands = max(7, hands)  # at least one full seat rotation

    cfg = LBR6Config(
        hands=hands, seed=seed, stack_bb=depth_bb, ante_bb=ante_bb,
        # CPU-friendly variance-reduction knobs (mirror tests/_fast_cfg); cheap but honest.
        settle_samples=8, eq_samples=10, eq_runouts=3, range_cap=24,
        value_only_raises=(source == "net"),  # never suicide-bluff into a net we can't cheaply model
    )

    res = icm_lbr_exploitability(
        policy_fn, cfg, payouts, pre, fmt=fmt, model_fn=model_fn, buy_in=sum(payouts),
    )

    anchor_dollar = anchor_dollar_target(payouts, pre, ante_bb=ante_bb)
    sevenmax_chip_anchor = sevenmax_tournament_anchor(ante_bb)  # +203.6 bb/100 floor (the always-fold $-anchor)

    # For a NET run, also measure the always-fold floor at THIS EXACT cell geometry (same stacks/payouts/
    # depth/ante) so the net's exploitability is compared apples-to-apples against a pure folder, not just
    # the symmetric-table closed form. (Cheap: a folder's hand is a fast fold-out.)
    anchor_at_cell = None
    if source == "net":
        anchor_cfg = LBR6Config(
            hands=cfg.hands, seed=seed, stack_bb=depth_bb, ante_bb=ante_bb,
            settle_samples=8, eq_samples=10, eq_runouts=3, range_cap=24)
        anchor_res = icm_lbr_exploitability(
            always_fold_policy, anchor_cfg, payouts, pre, fmt=fmt, buy_in=sum(payouts))
        anchor_at_cell = float(anchor_res.chip_exploit_bb_per_100)

    # COVERAGE HONESTY: for the analytic ICM policy, surface the fraction of LBR decision nodes the engine
    # actually DECIDED (a clean preflop push/fold spot) vs DEFERRED to the always-fold fallback (postflop /
    # multiway / deep / ambiguous), so the grade is not silently dominated by the fallback. The push/fold
    # specialist should decide most preflop short/bubble/ft nodes and defer deep / postflop ones.
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
        "two_policy_split": bool(two_policy),
        "model_coverage": model_coverage,
        "hands": int(res.hands),
        # E1 headline: ICM-$ exploitability of the policy (per game) + CI + mbb-of-$.
        "dollar_exploit_per_game": float(res.dollar_exploit_per_game),
        "dollar_stderr_per_game": float(res.dollar_stderr_per_game),
        "ci95": [float(res.ci95[0]), float(res.ci95[1])],
        "mbb_of_dollar_per_game": float(res.mbb_of_dollar_per_game),
        "pct_of_buyin": (float(res.pct_of_buyin) if res.pct_of_buyin is not None else None),
        "verdict": res.verdict,
        # E2 side-channel: the SAME run scored in chips (chip >= $ exactly where ICM binds).
        "chip_exploit_bb_per_100": float(res.chip_exploit_bb_per_100),
        "chip_stderr_bb_per_100": float(res.chip_stderr_bb_per_100),
        "leaf_divergence": float(res.leaf_divergence),
        "leaf_conservation": float(res.leaf_conservation),
        # Calibration anchors: the always-fold floor in $ AND the chip anchor (+203.6 bb/100) it derives from.
        "anchor_dollar_target_per_game": float(anchor_dollar),
        "sevenmax_chip_anchor_bb_per_100": float(sevenmax_chip_anchor),
        # The always-fold $-anchor exploitability at THIS cell's geometry (the honest same-cell floor to
        # compare the net against; the closed-form +203.6 bb/100 chip anchor is the symmetric-flat-table
        # reference, but ICM-pressure / satellite geometries shift the floor, so we report both).
        "anchor_chip_exploit_at_cell_bb_per_100": anchor_at_cell,  # same-geometry always-fold floor (net runs)
        # True when the policy is LESS exploitable (lower chip-LBR) than a pure folder — i.e. the net is
        # harder for the best-responder to beat than always-fold. Compared at the SAME cell geometry for a
        # net run; against the closed-form floor otherwise. False => the net is MORE exploitable than
        # folding (the LBR extracts more from it), the candid read.
        "less_exploitable_than_always_fold": bool(
            res.chip_exploit_bb_per_100 < (anchor_at_cell if anchor_at_cell is not None
                                           else sevenmax_chip_anchor)),
        "icm_pressure_spot": icm_spot,
        "pre_stacks": [round(float(s), 3) for s in pre],
        "payouts": [round(float(p), 4) for p in payouts],
        "note": res.note,
    }
