"""6-max Local Best Response (LBR6) -- a tractable LOWER BOUND on the exploitability of a
*6-player* NLHE strategy.

Why this file exists
--------------------
Every certified exploitability tool in the repo is HEADS-UP only: ``pokerbot/eval/lbr.py``
(HU LBR), ``pokerbot/eval/solver_exploitability.py`` (HU postflop node-locking),
``pokerbot/eval/exploitability.py`` (HU heuristic ensemble). There is no certified
adversary for the 6-max foundation -- only heuristic battery opponents (CounterExploiter,
AdaptiveBR) whose EV is suggestive, not a worst-case bound. This module fills that gap.

The multiplayer caveat (read this)
----------------------------------
In two-player zero-sum poker a Nash strategy is provably unexploitable, and LBR's measured
EV is a lower bound on a well-defined exploitability. In games with THREE OR MORE players
there is NO such guarantee: CFR's average strategy is not guaranteed to converge to a Nash
equilibrium, equilibria need not share a value, and a strategy that is "in equilibrium"
can still lose to a particular opponent profile (Brown & Sandholm, *Superhuman AI for
multiplayer poker*, Science 2019 -- Pluribus itself carries no exploitability guarantee).
So "exploitability" here means the concrete, operational thing LBR actually measures:

    Seat a single greedy best-responder at a table of N-1 COPIES of the target strategy and
    measure how much it wins. A value <= 0 (within CI) means *this* adversary cannot beat a
    table of us; a positive value is a certified amount by which the target IS beatable.

A positive LBR6 value is therefore a genuine "someone CAN beat it" certificate. A value at
or below zero is the strongest empirical evidence of unbeatability that multiplayer game
theory permits -- it is *not* a proof, and cannot be (see above).

How LBR6 works (range-tracked MC-rollout best response)
-------------------------------------------------------
The N-1 non-LBR seats all play the target ``policy_fn``. LBR maintains an independent Bayesian
belief over EACH in-hand opponent's hole cards, updated from that opponent's own ``policy_fn``
at every one of its decisions (range tracking, exactly like HU LBR but per-opponent -- the
standard product-of-marginals multiway approximation). At each LBR decision it estimates the
EV of {fold, check/call, pot-bet, all-in} by Monte-Carlo rollout: it samples each live
opponent's hole cards FROM ITS TRACKED BELIEF, re-randomises the undealt board consistently,
applies the candidate action, then rolls the hand to showdown with every opponent drawing
from its REAL ``policy_fn`` and LBR playing a fixed check/call-down continuation. LBR plays
the argmax-EV action.

Range tracking is what makes the bound TIGHT against a strong target: without it, a raiser's
range looks uniformly weak, LBR over-calls and bleeds, and the bound (though still valid)
collapses to a useless negative number. With it, the sampled opponents reflect their actual
post-action ranges, so LBR folds correctly and only the REAL leaks (over-folding to bets,
paying off value, mis-defending blinds) translate into a positive bound. Set
``track_ranges=False`` for the fast uniform-belief path (fine for calibration vs trivially
exploitable targets, where fold equity alone dominates and the rollout already queries the
real opponent strategy).

Because LBR follows a fully-specified legal strategy, its realised EV (estimated over many
hands, seats rotated) is a guaranteed LOWER BOUND on how exploitable the target is.

Units: bb/hand, mbb/g (= bb/hand * 1000), bb/100 (= bb/hand * 100). Everything runs on the
pure-NumPy engine and keys the blueprint with the same ``fingerprint_fast`` used at training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# Bounded abstraction must match the trained table; set BEFORE the encoders import (mirrors lbr.py).
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from pokerbot.engine.jax_engine import CHIP_UNIT, init_hand  # noqa: E402

from phevaluator import evaluate_cards as _phe  # noqa: E402  (fast C eval; lower rank == stronger)

_CARD_STR = ["23456789TJQKA"[c // 4] + "cdhs"[c % 4] for c in range(52)]
from pokerbot.engine.np_engine import (  # noqa: E402
    ALL_IN,
    BET_BUCKET_START,
    CALL,
    CHECK,
    FOLD,
    NUM_PLAYERS,
    SEAT_ALLIN,
    SEAT_LIVE,
    apply_action_np,
    is_chance_node_np,
    legal_action_mask_np,
    sample_chance_np,
    to_numpy_state,
)

_HALF_POT_INDEX = BET_BUCKET_START + 1  # frac 0.5 (BET_SIZE_BUCKETS[1])
_POT_BET_INDEX = BET_BUCKET_START + 4   # frac 1.0 == full pot (BET_SIZE_BUCKETS[4]); same as HU LBR


# --------------------------------------------------------------------------- #
# Configuration + result
# --------------------------------------------------------------------------- #
@dataclass
class LBR6Config:
    hands: int = 400
    seed: int = 0
    n_seats: int = 6                 # table size (occupancy = first n_seats live)
    analytic: bool = True            # ANALYTIC 1-ply LBR (Lisy & Bowling 2017): price each action with the
                                     # closed-form fold-equity + showdown-equity value (no deep rollout), so
                                     # V(call)<0 when pot odds fail => folds correctly, CANNOT bleed below
                                     # fold. The canonical, tight method. False = legacy MC-rollout (bleeds).
    eq_samples: int = 40             # belief-sampled opponent-hand sets averaged for the analytic wp (equity)
    eq_runouts: int = 6              # board runouts per equity sample (higher = less wp noise => fewer bad calls)
    rollouts: int = 20               # (legacy rollout mode) MC rollouts per candidate action per LBR node
    max_depth: int = 400             # hard cap on engine steps (a hand terminates well under this)
    track_ranges: bool = True        # Bayes-track each opponent's range (tight); False = uniform belief
    range_cap: int = 120             # max belief combos carried per opponent (weighted subsample beyond)
    paired: bool = False             # CRN-pair vs a base copy on same deck. NOTE: empirically LBR and base
                                     # diverge too much to correlate, so pairing does NOT cut variance here
                                     # (unlike the battery's ON/OFF arms). Left available; default off.
    smart_continuation: bool = True  # VALID competent continuation: fold-when-behind by equity vs holes
                                     # freshly SAMPLED FROM LBR'S BELIEF (its legitimate range estimate),
                                     # NOT the rollout's actual hidden holes. A real deployable decision
                                     # rule => the bound stays valid, but LBR stops the passive-call-down
                                     # over-aggression. False = passive call-down (valid but weaker/looser).
    cont_rollouts: int = 2           # runout samples for the continuation equity estimate
    settle_samples: int = 64         # all-in EV: runouts averaged when every contender is all-in (var-cut)
    bet_candidates: tuple[int, ...] = (_HALF_POT_INDEX, _POT_BET_INDEX, ALL_IN)  # LBR bet menu (+fold/chk/call)
    # Stack depth + per-player ante (BB) the hand is dealt at. None => engine default 100bb / ante 0
    # (unchanged). Set to measure a TOURNAMENT-shallow target (e.g. stack_bb=20, ante_bb=0.125) -- the
    # regime a tournament net actually plays. Honoured by the NATIVE harness (lbr_native.py); the pure-
    # Python driver below still deals at the engine default (these fields parameterise the fast path).
    stack_bb: float | None = None
    ante_bb: float = 0.0
    value_only_raises: bool = False  # VALUE-ONLY raise valuation: price each raise assuming it is ALWAYS
                                     # called (fp=0, no fold-equity credit), so the LBR bets/jams only when it
                                     # profits on pure showdown value vs the calling range. Needed when the
                                     # opponents ACT via a policy the LBR cannot model cheaply (e.g. SEARCH):
                                     # a blueprint-read fp over-credits fold equity, so the LBR suicide-bluffs
                                     # into search and bleeds (-600 bb/100). Dropping the fold-equity term
                                     # makes the bound robust to that model error (it can no longer lose on a
                                     # failed bluff) at the cost of not exploiting genuine over-folding -- the
                                     # right trade vs a strong, expensive-to-model opponent. Also skips the
                                     # white-box fold reads entirely (no policy_fn calls for raises).


@dataclass
class LBR6Result:
    ev_bb_per_hand: float
    mbb_per_g: float
    bb_per_100: float
    hands: int
    stderr_bb_per_hand: float
    note: str = (
        "6-max Local Best Response: greedy MC-rollout best-responder vs a table of N-1 copies of the "
        "target. Lower bound on exploitability. Positive => beatable by at least this much. Multiplayer "
        "has no Nash exploitability guarantee (Brown & Sandholm 2019); <=0 is strong evidence, not proof."
    )
    detail: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Belief over each opponent's hole cards
# --------------------------------------------------------------------------- #
def _board_cards(env) -> np.ndarray:
    b = np.asarray(env.board)
    return b[b >= 0]


def _in_hand_opponents(env, lbr_seat: int) -> list[int]:
    """Seats other than LBR still contesting the pot (LIVE or ALL-IN) -- their holes matter for showdown."""
    ss = env.seat_status.tolist()
    return [s for s in range(NUM_PLAYERS)
            if s != lbr_seat and ss[s] in (int(SEAT_LIVE), int(SEAT_ALLIN))]


def _all_combos(dead: set[int]) -> np.ndarray:
    live = [c for c in range(52) if c not in dead]
    return np.array([(a, b) for i, a in enumerate(live) for b in live[i + 1:]], dtype=np.int32)


def _set_holes(env, seat: int, c0: int, c1: int):
    holes = env.holes.copy()
    holes[seat] = (c0, c1)
    return env._replace(holes=holes)


class _Belief:
    """Per-opponent range belief: capped set of (combo, weight). Independent across opponents
    (product-of-marginals approximation). Card-removal handled by pruning on every board reveal."""

    __slots__ = ("combos", "weights")

    def __init__(self, combos: np.ndarray, weights: np.ndarray):
        self.combos = combos
        self.weights = weights

    @classmethod
    def uniform(cls, dead: set[int], cap: int, rng: np.random.Generator) -> "_Belief":
        combos = _all_combos(dead)
        if combos.shape[0] > cap:
            combos = combos[rng.choice(combos.shape[0], size=cap, replace=False)]
        w = np.full(combos.shape[0], 1.0 / max(combos.shape[0], 1), dtype=np.float64)
        return cls(combos, w)

    def prune(self, blocked: set[int]) -> None:
        if self.combos.shape[0] == 0:
            return
        keep = np.array([int(c0) not in blocked and int(c1) not in blocked
                         for c0, c1 in self.combos], dtype=bool)
        if not keep.all():
            self.combos, self.weights = self.combos[keep], self.weights[keep]
            self._renorm()

    def _renorm(self) -> None:
        s = self.weights.sum()
        self.weights = (self.weights / s if s > 1e-12
                        else np.full(self.combos.shape[0], 1.0 / max(self.combos.shape[0], 1)))

    def bayes_update(self, env, seat: int, action: int, policy_fn, cap: int,
                     rng: np.random.Generator) -> None:
        n = self.combos.shape[0]
        if n == 0:
            return
        upd = np.empty(n, dtype=np.float64)
        for i in range(n):
            env_c = _set_holes(env, seat, int(self.combos[i, 0]), int(self.combos[i, 1]))
            upd[i] = float(policy_fn(env_c, seat)[action])
        self.weights = self.weights * upd
        self._renorm()
        if self.combos.shape[0] > cap:
            idx = rng.choice(self.combos.shape[0], size=cap, replace=False, p=self.weights)
            self.combos, self.weights = self.combos[idx], self.weights[idx]
            self._renorm()

    def sample(self, blocked: set[int], rng: np.random.Generator) -> tuple[int, int] | None:
        """Draw one combo by weight that avoids ``blocked``; a few rejection tries, else None."""
        n = self.combos.shape[0]
        if n == 0:
            return None
        for _ in range(8):
            i = int(rng.choice(n, p=self.weights))
            c0, c1 = int(self.combos[i, 0]), int(self.combos[i, 1])
            if c0 not in blocked and c1 not in blocked:
                return c0, c1
        return None


def _init_beliefs(env, lbr_seat: int, cfg: LBR6Config, rng: np.random.Generator) -> dict[int, _Belief]:
    dead = {int(env.holes[lbr_seat][0]), int(env.holes[lbr_seat][1])}
    dead |= set(int(c) for c in _board_cards(env).tolist())
    return {s: _Belief.uniform(dead, cfg.range_cap, rng) for s in _in_hand_opponents(env, lbr_seat)}


# --------------------------------------------------------------------------- #
# Rollout sampling + value
# --------------------------------------------------------------------------- #
def _sample_rollout_env(env, lbr_seat: int, beliefs: dict[int, _Belief] | None,
                        rng: np.random.Generator):
    """Return a COPY of ``env`` with in-hand opponents' holes resampled (from their tracked beliefs when
    available, else uniform over the unseen deck) and the undealt board re-randomised consistently. LBR's
    own holes and the already-dealt board are preserved; the engine deals future streets deterministically
    from ``deck[deck_ptr:]`` so rebuilding that tail keeps the deal valid and collision-free."""
    holes = env.holes.copy()
    board = _board_cards(env)
    lbr_h = (int(holes[lbr_seat][0]), int(holes[lbr_seat][1]))
    seen = set(int(c) for c in board.tolist()) | set(lbr_h)

    opps = _in_hand_opponents(env, lbr_seat)
    pool = [c for c in range(52) if c not in seen]
    rng.shuffle(pool)
    pi = 0
    for s in opps:
        drawn = None
        if beliefs is not None and s in beliefs:
            drawn = beliefs[s].sample(seen, rng)
        if drawn is None:  # uniform fallback / belief exhausted
            while pi < len(pool) - 1 and (pool[pi] in seen or pool[pi + 1] in seen):
                pi += 1
            drawn = (pool[pi], pool[pi + 1])
            pi += 2
        holes[s] = drawn
        seen.add(drawn[0])
        seen.add(drawn[1])

    ptr = int(env.deck_ptr)
    remaining = [c for c in range(52) if c not in seen]
    rng.shuffle(remaining)
    deck = env.deck.copy()
    n_slots = deck.shape[0] - ptr
    deck[ptr:] = np.asarray(remaining[:n_slots], dtype=deck.dtype)
    return env._replace(holes=holes, deck=deck)


def _board_ev_settle(env, lbr_seat: int, rng: np.random.Generator, samples: int) -> float:
    """All-in EV settlement: every remaining contender is all-in (no more betting), so the only randomness
    left is the board runout. Average LBR's terminal reward over ``samples`` independent runouts of the
    UNDEALT cards (deck[deck_ptr:]) instead of dealing one. Removes the dominant multiway stack-off
    variance while preserving the mean exactly (every runout is equally likely)."""
    ptr = int(env.deck_ptr)
    base_deck = env.deck.copy()
    tail = base_deck[ptr:].copy()
    if tail.shape[0] == 0:
        return float(env.rewards[lbr_seat]) / CHIP_UNIT
    total = 0.0
    for _ in range(samples):
        rng.shuffle(tail)
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
        total += float(e.rewards[lbr_seat]) / CHIP_UNIT
    return total / samples


def _sample_policy_action(env, seat: int, policy_fn, rng: np.random.Generator) -> int:
    probs = np.asarray(policy_fn(env, seat), dtype=np.float64)
    legal = np.flatnonzero(probs > 0)
    if legal.size == 0:
        mask = legal_action_mask_np(env, seat)
        legal = np.flatnonzero(mask)
        return int(rng.choice(legal))
    return int(rng.choice(legal, p=probs[legal] / probs[legal].sum()))


def _multiway_equity(lbr_hole, opp_holes: list, board: np.ndarray, rng: np.random.Generator,
                     rollouts: int) -> float:
    """P(LBR's hand is best, ties shared) vs the GIVEN opponent holes over sampled runouts. Used only by
    the smart continuation, where the rolled-out opponents' holes are known. Cheap (few runouts)."""
    if not opp_holes:
        return 1.0
    need = 5 - int(board.shape[0])
    lbr_s = [_CARD_STR[int(lbr_hole[0])], _CARD_STR[int(lbr_hole[1])]]
    board_s = [_CARD_STR[int(c)] for c in board]
    dead = set(int(c) for c in board.tolist()) | {int(lbr_hole[0]), int(lbr_hole[1])}
    for oh in opp_holes:
        dead |= {int(oh[0]), int(oh[1])}
    if need == 0:
        lh = _phe(*(lbr_s + board_s))
        ohs = [_phe(_CARD_STR[int(o[0])], _CARD_STR[int(o[1])], *board_s) for o in opp_holes]
        best = min(ohs)
        return 1.0 if lh < best else (1.0 / (1 + sum(o == lh for o in ohs)) if lh == best else 0.0)
    deck = [c for c in range(52) if c not in dead]
    tot = 0.0
    for _ in range(max(1, rollouts)):
        pick = rng.choice(len(deck), size=need, replace=False)
        full = board_s + [_CARD_STR[deck[d]] for d in pick]
        lh = _phe(*(lbr_s + full))
        ohs = [_phe(_CARD_STR[int(o[0])], _CARD_STR[int(o[1])], *full) for o in opp_holes]
        best = min(ohs)
        tot += 1.0 if lh < best else (1.0 / (1 + sum(o == lh for o in ohs)) if lh == best else 0.0)
    return tot / max(1, rollouts)


def _lbr_continuation(env, lbr_seat: int, cfg: LBR6Config, rng: np.random.Generator,
                      beliefs: dict | None) -> int:
    """LBR's continuation inside a rollout. VALID in both modes (no peeking at hidden cards).

    Passive (``smart_continuation=False``): check when free, else call, else fold. Deliberately weak --
    it can only UNDER-state LBR's strength, so the realised EV stays a true lower bound; it biases LBR
    toward premature aggression, loosening (never invalidating) the bound.

    Competent (``smart_continuation=True``, default): check when free; facing a bet, estimate LBR's equity
    vs opponent holes freshly SAMPLED FROM LBR'S BELIEF (the range it has legitimately inferred -- NOT the
    rollout's actual hidden holes), and CALL only when that equity clears the immediate pot odds, else
    FOLD. Because the decision depends solely on LBR's own cards, the board, and its belief, it is a real
    deployable strategy and the lower bound stays valid -- while LBR stops the passive-call-down bleed that
    made it over-aggress, tightening the bound."""
    mask = legal_action_mask_np(env, lbr_seat).astype(bool)
    if mask[CHECK]:
        return CHECK
    if not cfg.smart_continuation:
        return CALL if mask[CALL] else (FOLD if mask[FOLD] else int(np.argmax(mask)))
    if not mask[CALL]:
        return FOLD if mask[FOLD] else int(np.argmax(mask))
    committed_street = float(env.committed_street[lbr_seat])
    to_call = max(0.0, float(env.current_bet) - committed_street)
    pot = float(np.asarray(env.committed_total).sum())
    if to_call <= 0.0:
        return CALL
    # Sample a representative opponent holding per in-hand opponent FROM BELIEF (valid -- LBR's own range
    # estimate), excluding LBR's holes, the board, and each other. Uniform fallback if belief is exhausted.
    board = _board_cards(env)
    blocked = set(int(c) for c in board.tolist()) | {int(env.holes[lbr_seat][0]),
                                                      int(env.holes[lbr_seat][1])}
    opp_holes = []
    for s in _in_hand_opponents(env, lbr_seat):
        drawn = beliefs[s].sample(blocked, rng) if (beliefs and s in beliefs) else None
        if drawn is None:
            pool = [c for c in range(52) if c not in blocked]
            if len(pool) < 2:
                continue
            idx = rng.choice(len(pool), size=2, replace=False)
            drawn = (pool[int(idx[0])], pool[int(idx[1])])
        opp_holes.append(drawn)
        blocked |= {int(drawn[0]), int(drawn[1])}
    eq = _multiway_equity(np.asarray(env.holes[lbr_seat]), opp_holes, board, rng, cfg.cont_rollouts)
    # call EV >= 0  <=>  eq*(pot+to_call) - to_call >= 0  <=>  eq >= to_call/(pot+to_call)
    return CALL if eq >= to_call / (pot + to_call + 1e-9) else (FOLD if mask[FOLD] else CALL)


def _rollout_value(env, lbr_seat: int, policy_fn, cfg: LBR6Config, rng: np.random.Generator,
                   beliefs: dict | None) -> float:
    """Roll ``env`` to terminal; opponents draw from ``policy_fn`` (holes already set by caller), LBR plays
    its continuation. ``beliefs`` (the decision-node range estimate) feeds the competent continuation's
    fold/call rule -- never the rollout's actual holes. Return LBR's reward in bb."""
    depth = 0
    while depth < cfg.max_depth and not bool(env.done):
        if is_chance_node_np(env):
            env = sample_chance_np(env)
            continue
        seat = int(env.action_seat)
        if seat < 0:
            break
        a = _lbr_continuation(env, lbr_seat, cfg, rng, beliefs) if seat == lbr_seat \
            else _sample_policy_action(env, seat, policy_fn, rng)
        env = apply_action_np(env, a)
        depth += 1
    return float(env.rewards[lbr_seat]) / CHIP_UNIT


# --------------------------------------------------------------------------- #
# Analytic 1-ply LBR (Lisy & Bowling 2017) -- the canonical, tight method
# --------------------------------------------------------------------------- #
def _wp_vs_beliefs(env, lbr_seat: int, beliefs: dict[int, _Belief] | None,
                   rng: np.random.Generator, samples: int, runouts: int,
                   opps: list | None = None) -> float:
    """LBR's multiway showdown win-probability vs ``opps`` sampled FROM BELIEF (valid -- LBR's own range
    estimate, never the actual hidden holes), averaged over ``samples`` belief draws. Used by the analytic
    value as the 'everyone checks down' equity. ``opps`` defaults to all in-hand opponents; pass an explicit
    list (e.g. only the CALLING opponents) to value a raise's called-branch. Returns P(LBR best, ties shared)."""
    if opps is None:
        opps = _in_hand_opponents(env, lbr_seat)
    if not opps:
        return 1.0
    board = _board_cards(env)
    lbr_hole = np.asarray(env.holes[lbr_seat])
    base_block = set(int(c) for c in board.tolist()) | {int(lbr_hole[0]), int(lbr_hole[1])}
    tot = 0.0
    for _ in range(max(1, samples)):
        blocked = set(base_block)
        opp_holes = []
        for s in opps:
            drawn = beliefs[s].sample(blocked, rng) if (beliefs and s in beliefs) else None
            if drawn is None:
                pool = [c for c in range(52) if c not in blocked]
                if len(pool) < 2:
                    continue
                idx = rng.choice(len(pool), size=2, replace=False)
                drawn = (pool[int(idx[0])], pool[int(idx[1])])
            opp_holes.append(drawn)
            blocked |= {int(drawn[0]), int(drawn[1])}
        tot += _multiway_equity(lbr_hole, opp_holes, board, rng, runouts)
    return tot / max(1, samples)


def _fold_and_calling(env_raised, lbr_seat: int, beliefs: dict[int, _Belief] | None, policy_fn):
    """Return (fp, calling) for LBR's raise. ``fp`` = P(every opponent folds) = product of per-opponent
    fold probabilities, each marginalised over that opponent's belief via the WHITE-BOX strategy (the LBR
    fold-equity read). ``calling`` = {seat -> _Belief} the opponent's CALLING range (belief re-weighted by
    1 - fold-prob-per-combo). Valuing the raise's called-branch against the calling range (not the full
    range) is the fix that stops LBR over-valuing bets and bleeding -- the callers are stronger than the
    folders, so LBR's equity-when-called is lower than its equity vs the full range."""
    fp = 1.0
    calling: dict[int, _Belief] = {}
    for s in _in_hand_opponents(env_raised, lbr_seat):
        b = beliefs.get(s) if beliefs else None
        if b is None or b.combos.shape[0] == 0:
            pf = float(policy_fn(env_raised, s)[FOLD])
            fp *= pf
            continue  # no belief -> omit from the calling set (uniform sampled later only if needed)
        folds = np.empty(b.combos.shape[0], dtype=np.float64)
        for i in range(b.combos.shape[0]):
            env_c = _set_holes(env_raised, s, int(b.combos[i, 0]), int(b.combos[i, 1]))
            folds[i] = float(policy_fn(env_c, s)[FOLD])
        fp *= float((b.weights * folds).sum())
        cw = b.weights * (1.0 - folds)
        tot = cw.sum()
        if tot > 1e-9:
            calling[s] = _Belief(b.combos.copy(), cw / tot)  # this opponent's not-fold (calling) range
    return fp, calling


def _lbr_action_analytic(env, lbr_seat: int, beliefs: dict[int, _Belief] | None, policy_fn,
                         cfg: LBR6Config, rng: np.random.Generator) -> int:
    """Analytic 1-ply LBR action (Lisy & Bowling 2017), multiway-generalised. Price each candidate by a
    closed form that assumes the hand checks down to showdown after this action, using:
      * wp  = LBR's showdown equity vs the opponents' tracked ranges (belief-sampled), and
      * fp  = P(all opponents fold to a raise), read from the white-box strategy (fold equity).
    Take the argmax. Because V(call) = wp*(pot+asked) - asked goes NEGATIVE exactly when pot odds fail,
    LBR folds correctly and can never bleed below the fold value -- the key property a deep rollout lacks.
    The chosen action is then PLAYED in the real engine (correct multiway settlement), so the realised EV
    is a valid lower bound regardless of the formula's multiway approximations."""
    mask = legal_action_mask_np(env, lbr_seat).astype(bool)
    ct = np.asarray(env.committed_total, dtype=np.float64)
    pot_others = float(ct.sum() - ct[lbr_seat])                       # chips LBR wins at showdown
    committed_street_lbr = float(env.committed_street[lbr_seat])
    asked = max(0.0, float(env.current_bet) - committed_street_lbr)   # to-call
    wp = _wp_vs_beliefs(env, lbr_seat, beliefs, rng, cfg.eq_samples, cfg.eq_runouts)

    best_a, best_v = None, -1e18
    if mask[FOLD]:
        best_a, best_v = FOLD, 0.0                                    # V(fold) = 0 (baseline)
    if mask[CHECK]:                                                   # free showdown
        v = wp * pot_others
        if v > best_v:
            best_v, best_a = v, CHECK
    if mask[CALL]:
        v = wp * (pot_others + asked) - asked                        # Lisy&Bowling V(call)
        if v > best_v:
            best_v, best_a = v, CALL
    for a in cfg.bet_candidates:
        if not mask[a]:
            continue
        env_r = apply_action_np(env, a)
        invest = float(np.asarray(env_r.committed_total)[lbr_seat]) - float(ct[lbr_seat])  # asked + raise
        # fold equity only if the raise actually leaves an opponent a decision (not all-in-closed / chance)
        bp_to_act = ((not bool(env_r.done)) and (not is_chance_node_np(env_r))
                     and int(env_r.action_seat) != lbr_seat)
        if cfg.value_only_raises:
            # No fold-equity credit (fp=0, assume always called). Equity vs the FULL belief range is the
            # conservative "always called by the whole range" read -- no white-box fold query, robust to a
            # mis-modelled opponent. The LBR raises only when it profits on showdown value alone.
            fp, wp_called = 0.0, wp
        elif bp_to_act:
            fp, calling = _fold_and_calling(env_r, lbr_seat, beliefs, policy_fn)
            # equity-when-called: vs the CALLING ranges only (stronger than the full range -> no over-bet)
            wp_called = (_wp_vs_beliefs(env, lbr_seat, calling, rng, cfg.eq_samples, cfg.eq_runouts,
                                        opps=list(calling.keys())) if calling else wp)
        else:
            fp, wp_called = 0.0, wp                                   # all-in closes action -> forced showdown
        pot_if_called = pot_others + 2.0 * invest                    # approx single matching caller
        v = fp * pot_others + (1.0 - fp) * (wp_called * pot_if_called - invest)
        if v > best_v:
            best_v, best_a = v, a
    if best_a is None:
        return int(np.argmax(mask))
    return best_a


def _lbr_action(env, lbr_seat: int, beliefs: dict[int, _Belief] | None, policy_fn,
                cfg: LBR6Config, rng: np.random.Generator) -> int:
    """Dispatch to the analytic 1-ply LBR (default, tight) or the legacy MC-rollout LBR."""
    if cfg.analytic:
        return _lbr_action_analytic(env, lbr_seat, beliefs, policy_fn, cfg, rng)
    return _lbr_action_rollout(env, lbr_seat, beliefs, policy_fn, cfg, rng)


def _lbr_action_rollout(env, lbr_seat: int, beliefs: dict[int, _Belief] | None, policy_fn,
                        cfg: LBR6Config, rng: np.random.Generator) -> int:
    """Legacy greedy best-response action: argmax over {fold, check/call, pot-bet, all-in} of MC-rollout EV.

    COMMON RANDOM NUMBERS across candidates: every rollout draws ONE shared (opponent holes, board)
    sample and a shared continuation RNG seed, then evaluates ALL candidates on it. The candidates'
    futures are then identical except for LBR's own action, so the *comparison* is near-zero-variance.
    Without this, the all-in candidate's +/-200bb runout noise wins the argmax by chance (optimizer's
    curse) and LBR spews -- the dominant failure mode of a naive rollout best-responder. NOTE: even with
    CRN this rollout responder can play worse than folding vs a strong opponent (mis-valued aggression);
    the analytic 1-ply value above is the principled fix."""
    mask = legal_action_mask_np(env, lbr_seat).astype(bool)
    candidates: list[int] = []
    for a in (FOLD, CHECK, CALL, *cfg.bet_candidates):
        if mask[a] and a not in candidates:
            candidates.append(a)
    if not candidates:
        return int(np.argmax(mask))
    if len(candidates) == 1:
        return candidates[0]

    totals = {a: 0.0 for a in candidates}
    for _ in range(cfg.rollouts):
        env_s = _sample_rollout_env(env, lbr_seat, beliefs, rng)  # shared opponent holes + board
        seed_r = int(rng.integers(1 << 62))                       # shared continuation randomness
        for a in candidates:
            env_a = apply_action_np(env_s, a)
            totals[a] += _rollout_value(env_a, lbr_seat, policy_fn, cfg,
                                        np.random.default_rng(seed_r), beliefs)
    best_a, best_v = candidates[0], -1e18
    for a in candidates:
        v = totals[a] / cfg.rollouts
        if v > best_v:
            best_v, best_a = v, a
    return best_a


# --------------------------------------------------------------------------- #
# Single-hand play (one arm)
# --------------------------------------------------------------------------- #
def _play_hand(env, lbr_seat: int, br_active: bool, policy_fn, cfg: LBR6Config,
               rng: np.random.Generator, settle_rng: np.random.Generator,
               model_fn=None, observe_fn=None, hand_sink=None) -> tuple[float, int]:
    """Play one hand from ``env`` and return (lbr_seat reward in bb, n_lbr_decisions).

    ``br_active`` True  -> ``lbr_seat`` plays the greedy LBR best response (range-tracked).
    ``br_active`` False -> ``lbr_seat`` plays ``policy_fn`` like everyone else (the BASE arm: a 6th copy
                           of the target, whose EV is ~0 by table symmetry -- the CRN pairing baseline).

    TWO-POLICY split (the feasibility key when ``policy_fn`` is EXPENSIVE, e.g. blueprint+search):
      * ``policy_fn`` (ACT) -- played by the non-LBR seats for their REAL actions (called once per
        opponent decision). This is the strategy whose exploitability we measure.
      * ``model_fn``  (MODEL, defaults to ``policy_fn``) -- the cheap WHITE-BOX the LBR uses internally
        for range tracking (``bayes_update``, O(range_cap) calls/decision) and the analytic fold-equity
        reads (``_fold_and_calling``, O(range_cap) calls/candidate). Using the cheap blueprint here while
        the seats ACT via search keeps the run tractable, and the bound stays VALID: the LBR's chosen
        action is a legal function of its cards + public state + belief (never the hidden holes), so its
        realised EV vs the real (search) seats is <= the search base's true exploitability, whatever model
        it used to pick the action. (Tightness, not validity, is what an imperfect model costs.)"""
    model_fn = model_fn or policy_fn
    beliefs = _init_beliefs(env, lbr_seat, cfg, rng) if (br_active and cfg.track_ranges) else None
    lbr_decisions = 0
    depth = 0
    last_call_street = -1   # street (0=preflop..3=river) of the LBR's LAST voluntary CALL (-1 = none)
    lbr_folded = False
    while depth < cfg.max_depth and not bool(env.done):
        if is_chance_node_np(env):
            if int((env.seat_status == SEAT_LIVE).sum()) == 0 and cfg.settle_samples > 1:
                r = _board_ev_settle(env, lbr_seat, settle_rng, cfg.settle_samples)
                if hand_sink is not None:
                    hand_sink({"reward": r, "last_call_street": last_call_street,
                               "end_street": int(np.asarray(env.street)), "folded": lbr_folded,
                               "showdown": not lbr_folded})  # all-in pre-showdown -> settles at showdown
                return r, lbr_decisions
            env = sample_chance_np(env)
            if beliefs is not None:
                blocked = set(int(c) for c in _board_cards(env).tolist())
                for b in beliefs.values():
                    b.prune(blocked)
            continue
        seat = int(env.action_seat)
        if seat < 0:
            break
        if seat == lbr_seat and br_active:
            a = _lbr_action(env, lbr_seat, beliefs, model_fn, cfg, rng)  # LBR lookahead uses its MODEL
            lbr_decisions += 1
            if hand_sink is not None:                    # attribute: where does the LBR commit chips?
                if a == CALL:
                    last_call_street = int(np.asarray(env.street))
                elif a == FOLD:
                    lbr_folded = True
        else:
            a = _sample_policy_action(env, seat, policy_fn, rng)         # seats ACT via the real policy
            if beliefs is not None and seat in beliefs:
                beliefs[seat].bayes_update(env, seat, a, model_fn, cfg.range_cap, rng)  # belief from MODEL
        if observe_fn is not None:
            observe_fn(env, seat, a)   # PRE-action env -> the adapting opponents ingest this decision
        env = apply_action_np(env, a)
        depth += 1
    reward = float(env.rewards[lbr_seat]) / CHIP_UNIT if bool(env.done) else 0.0
    if hand_sink is not None:
        hand_sink({"reward": reward, "last_call_street": last_call_street,
                   "end_street": int(np.asarray(env.street)), "folded": lbr_folded,
                   "saw_through": not lbr_folded,
                   "lbr_decisions": int(lbr_decisions)})
    return reward, lbr_decisions


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def lbr6_exploitability(policy_fn, cfg: LBR6Config | None = None, model_fn=None,
                        observe_fn=None, on_hand_start=None, hand_sink=None,
                        progress_fn=None, **_unused) -> LBR6Result:
    """Monte-Carlo 6-max LBR lower bound on the exploitability of ``policy_fn``.

    ``policy_fn(env, seat) -> probs`` (shape NUM_ACTIONS, sums to 1 over legal) is played by all non-LBR
    seats. The LBR seat rotates across hands so every position is best-responded from.

    ``model_fn`` (optional, defaults to ``policy_fn``): a CHEAP white-box model the LBR uses internally for
    range tracking + analytic fold-equity reads, decoupled from the (possibly expensive) ``policy_fn`` the
    seats actually ACT with. This makes "blueprint+SEARCH" exploitability tractable: the seats act via
    search (~1 solve per opponent decision) while the LBR reads fold probs from the cheap blueprint
    (O(range_cap) reads/decision would be thousands of search solves otherwise). The realised EV is a valid
    lower bound on ``policy_fn``'s exploitability regardless of the model -- see ``_play_hand``.

    ``cfg.paired`` (default True): also play a BASE arm (the LBR seat plays the target instead) on the
    SAME deck, and report the per-hand difference LBR - base. The base arm earns ~0 by symmetry, so the
    difference is an unbiased estimator of LBR's exploitability whose shared-deck card luck CANCELS --
    cutting per-hand variance ~5-10x (the only way to a tight CI without ~10^5 hands)."""
    cfg = cfg or LBR6Config()
    model_fn = model_fn or policy_fn
    key = jax.random.PRNGKey(cfg.seed)
    rng = np.random.default_rng(cfg.seed + 1)
    occ = jnp.asarray([s < cfg.n_seats for s in range(NUM_PLAYERS)], dtype=bool)

    results = np.zeros(cfg.hands, dtype=np.float64)      # the headline estimator (paired diff or raw BR)
    br_raw = np.zeros(cfg.hands, dtype=np.float64)       # LBR absolute reward (for reference)
    lbr_decisions = 0

    for h in range(cfg.hands):
        key, k = jax.random.split(key)
        env0 = to_numpy_state(init_hand(k, occupancy=occ))
        lbr_seat = h % cfg.n_seats
        settle_seed = (cfg.seed * 1_000_003 + h) & 0x7FFFFFFF  # shared runout draws => arms cancel
        if on_hand_start is not None:
            on_hand_start(lbr_seat)   # let stateful adapting opponents key this hand to the LBR's seat

        r_br, nd = _play_hand(env0, lbr_seat, True, policy_fn, cfg, rng,
                              np.random.default_rng(settle_seed), model_fn=model_fn,
                              observe_fn=observe_fn, hand_sink=hand_sink)
        lbr_decisions += nd
        br_raw[h] = r_br
        if cfg.paired:
            r_base, _ = _play_hand(env0, lbr_seat, False, policy_fn, cfg, rng,
                                   np.random.default_rng(settle_seed), model_fn=model_fn)
            results[h] = r_br - r_base
        else:
            results[h] = r_br

        if progress_fn is not None:
            # progress_fn(done, total, lbr_seat, reward, lbr_decisions) — match eval driver's _make_hand_progress signature.
            try:
                progress_fn(h + 1, cfg.hands, lbr_seat, float(results[h]), int(lbr_decisions))
            except Exception:  # noqa: BLE001 - never let progress callback break the LBR loop
                pass

    ev = float(results.mean())
    stderr = float(results.std(ddof=1) / np.sqrt(cfg.hands)) if cfg.hands > 1 else 0.0
    # Per-seat deviation gain: results[h] for lbr_seat = h % n_seats, so seat s == results[s::n_seats].
    # button_seat=0 each hand, so seat index == offset from the button (0=BTN .. 5=CO going around).
    per_seat = {}
    for s in range(cfg.n_seats):
        rs = results[s::cfg.n_seats]
        if rs.size:
            per_seat[s] = {"n": int(rs.size), "bb_per_100": float(rs.mean()) * 100.0,
                           "std_bb_per_hand": float(rs.std(ddof=1)) if rs.size > 1 else 0.0}
    return LBR6Result(
        ev_bb_per_hand=ev,
        mbb_per_g=ev * 1000.0,
        bb_per_100=ev * 100.0,
        hands=cfg.hands,
        stderr_bb_per_hand=stderr,
        detail={
            "n_seats": cfg.n_seats,
            "method": "analytic_1ply_lbr" if cfg.analytic else "mc_rollout",
            "eq_samples": cfg.eq_samples,
            "rollouts_per_candidate": cfg.rollouts,
            "track_ranges": cfg.track_ranges,
            "range_cap": cfg.range_cap,
            "paired": cfg.paired,
            "per_seat": per_seat,
            "br_raw_bb_per_100": float(br_raw.mean()) * 100.0,
            "lbr_decisions": lbr_decisions,
            "bet_candidates": list(cfg.bet_candidates),
        },
    )
