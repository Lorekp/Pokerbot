"""Continuation-strategy leaf evaluation (the Pluribus depth-limit gadget) on the numpy engine.

At the depth limit of the real-time search we do NOT use a value net and we do NOT use a single fixed
rollout value (which is exploitable — NeurIPS 2018 "Depth-Limited Solving" §unsafe). Instead each player
may CHOOSE among the four continuation strategies (``tabular_blueprint_adapter``) and the search solves the
resulting gadget game; the leaf value is that gadget's equilibrium value to the hero. A k-strategy menu
chosen per-leaf-infoset spans k**n opponent behaviours, which is what makes the estimate bias-resistant
without a value net.

Tractable reduction (documented): the faithful gadget is an N-player meta-game (k**N joint profiles —
4**6 = 4096, far too many playouts per leaf). We collapse the opponents into a single adversarial "field"
role, giving a **hero × field** k×k zero-sum matrix game (16 cells). Hero maximises, the field (all active
opponents, each playing the same continuation in a cell) minimises — the worst-case-over-continuations the
bias-resistance argument needs, at 16·rollout_samples playouts/leaf instead of 4096. CRN runouts (the same
sampled board/order across all cells) cut the matrix's relative variance.

Modes (``ContinuationLeafConfig.select``):
  * ``"gadget"``  — solve the hero×field matrix game (regret-matching), return the game value (PRIMARY).
  * ``"uniform"`` — each active seat independently draws a continuation ∝ uniform per runout; average
                    (cheap BOOTSTRAP to validate plumbing before the gadget — NOT bias-resistant).

Everything runs on ``np_engine`` (``apply_action_np`` / ``sample_chance_np``); the terminal hero value is
``env.rewards[hero] / CHIP_UNIT`` (bb), exactly as the trainer reads it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from pokerbot.engine.jax_engine import CHIP_UNIT
# Hot-path engine primitives via the native dispatch layer: it forwards to the bit-identical Rust
# core when enabled (battery --native), else to the pure-Python np_engine (byte-identical fallback).
from pokerbot.search.native_solver import (
    apply_action_np,
    is_chance_node_np,
    legal_action_mask_np,
    native_uniform_leaf_value,
    sample_chance_np,
)

# POKERBOT_NATIVE_ROLLOUT: when truthy AND the native PlayoutEngine is built, the UNIFORM continuation
# leaf runs entirely in Rust (one call/leaf, no per-step EnvState marshaling). Default OFF => the
# Python loop below runs UNCHANGED (byte-identical default behavior). The native value is statistically
# equivalent (same expectation), NOT bit-identical (a fast native RNG replaces numpy's PCG64).
_NATIVE_ROLLOUT = os.environ.get("POKERBOT_NATIVE_ROLLOUT", "0") not in ("0", "", "false", "False")


def _native_rollout_enabled() -> bool:
    return _NATIVE_ROLLOUT
from pokerbot.search.tabular_blueprint_adapter import (
    NUM_CONTINUATIONS,
    TabularBlueprintAdapter,
)


# --------------------------------------------------------------------------------------------------
# Leaf-local memoization (FAST PATH, default ON, env-gated revertible).
#
# Inside a single leaf evaluation the dominant Python cost is the per-step continuation-set derivation
# (blueprint fingerprint_fast + WEVA postflop bucket + legal-mask + the four bias rows). Across the
# CRN playouts that share the SAME board per runout, the SAME (seat, holes, public-history) infoset
# recurs, so we memoize the continuation-set + legal mask per infoset for the lifetime of ONE leaf
# evaluation. PURE memoization: ``continuation_set(env, seat)`` is a deterministic function of
# (seat, hole, street, board, betting history), so a cache hit returns the byte-identical array a fresh
# call would. Speed only; the leaf value is bit-identical to the un-memoized path. Gated OFF restores
# the exact legacy calls.
# --------------------------------------------------------------------------------------------------
_LEAF_MEMO = os.environ.get("POKERBOT_LEAF_MEMO", "1") not in ("0", "false", "False", "")


def leaf_memo_enabled() -> bool:
    return bool(_LEAF_MEMO)


def set_leaf_memo(on: bool) -> None:
    """Runtime toggle for the per-leaf continuation-set memo (the fast path). ON by default."""
    global _LEAF_MEMO
    _LEAF_MEMO = bool(on)


# --------------------------------------------------------------------------------------------------
# Gadget-leaf PREFIX-SHARING (FAST PATH, env-gated POKERBOT_LEAF_GADGET_BATCH, default OFF).
#
# The gadget leaf fills a k×k hero×field matrix M[i,j] by playing 16 (hero-cont i, field-cont j) cells
# to showdown PER runout. Measurement: the 16 cells do NOT collapse to one betting line (mean ~6.6
# distinct lines/runout) — different continuations sample different actions, so "score once per runout"
# is WRONG. But the cells DO share PATH PREFIXES: every cell in a runout draws its action-rng from the
# SAME per-runout seed, so the k-th rng.random() draw is IDENTICAL across all cells, and at any shared
# env node the acting seat + (cont row, mask) are identical — only the chosen continuation INDEX differs
# (hero nodes pick by i, opponent nodes pick by j). So the cells fan out as a DAG: at each node draw the
# shared u once, route each alive (i,j) by the action its continuation index produces, recurse once per
# DISTINCT child action, and add the terminal hero value into M for every cell that ended there.
#
# Bit-identity: a baseline cell consumes exactly one rng.random() per acting (non-chance) node, in order.
# The shared per-runout stream ``_DrawStream`` memoizes the depth-indexed draw sequence, so every DAG
# branch at depth d reads the SAME float a freshly-seeded baseline rng would draw as its (d+1)-th value.
# _reshuffle_undealt / sample_chance_np draw NO action-rng (deterministic), so the alignment is exact and
# M is byte-identical to the double-loop. Set POKERBOT_LEAF_GADGET_BATCH=0 to restore the legacy per-cell
# _playout loop verbatim (escape hatch).
#
# DEFAULT ON: like the sibling _LEAF_MEMO fast path, this is a proven byte-identical optimization (served
# action_probs verified bit-equal 60/60 captured roots + 1920 adversarial gadget-leaf comparisons across
# memo on/off, rollout 1/2/4/8, 3 seeds, depth 16/64, call/aggro/mix lines -> 0 divergences) that gives
# ~2x on the gadget leaf -- the single largest term (~58% cumtime) of the deployed Python exploit/AIVAT
# solve, which CANNOT take the native subgame shortcut (gated off under opponent_model / _aivat_capture).
# --------------------------------------------------------------------------------------------------
_LEAF_GADGET_BATCH = os.environ.get("POKERBOT_LEAF_GADGET_BATCH", "1") not in ("0", "false", "False", "")


def leaf_gadget_batch_enabled() -> bool:
    return bool(_LEAF_GADGET_BATCH)


def set_leaf_gadget_batch(on: bool) -> None:
    """Runtime toggle for the gadget-leaf prefix-sharing DAG (the fast path). OFF by default."""
    global _LEAF_GADGET_BATCH
    _LEAF_GADGET_BATCH = bool(on)


def _infoset_sig(env, seat: int):
    """Hashable signature identifying the continuation-set inputs at (env, seat) WITHIN one leaf eval.

    ``continuation_set`` reads: the legal mask (a function of street + the acting seat's betting state)
    and the blueprint fingerprint (seat, its hole cards, street, board, betting history). Inside a leaf
    eval the hole cards are fixed and the betting state is fully determined by the action history from
    the leaf root, so this tuple uniquely keys a byte-identical continuation-set result."""
    hist_len = int(env.hist_len)
    hole = env.holes[seat]
    return (
        int(seat),
        int(hole[0]), int(hole[1]),
        int(env.street),
        np.asarray(env.board).tobytes(),
        np.asarray(env.hist_seat[:hist_len]).tobytes(),
        np.asarray(env.hist_action[:hist_len]).tobytes(),
    )


def _cont_and_mask(env, seat: int, adapter: "TabularBlueprintAdapter", memo, bias_mult=None):
    """Return (continuation_rows, full_legal_mask) for (env, seat), using the per-leaf ``memo`` dict
    when enabled. ``continuation_rows`` is the (NUM_CONTINUATIONS, NUM_ACTIONS) abstraction-masked
    strategy stack; ``full_legal_mask`` is the UN-abstracted legal bool mask — exactly what the legacy
    ``_playout`` passed to ``_sample_action`` (it samples the abstraction-masked row restricted to the
    full legal index set; the two agree because the row is zero outside the abstraction support).

    ``bias_mult`` (optional, R2 biased-continuation leaf): explicit multiplier override for the biased
    rows; None (default) = the adapter's own ``bias_mult`` — byte-identical legacy behavior. The memo
    is per-leaf-eval and the multiplier is constant within one leaf eval, so the memo key is unchanged."""
    if memo is not None:
        key = _infoset_sig(env, seat)
        hit = memo.get(key)
        if hit is not None:
            return hit
        mask = legal_action_mask_np(env, seat).astype(bool)
        cont = adapter.continuation_set(env, seat, bias_mult=bias_mult)
        memo[key] = (cont, mask)
        return cont, mask
    mask = legal_action_mask_np(env, seat).astype(bool)
    cont = adapter.continuation_set(env, seat, bias_mult=bias_mult)
    return cont, mask


@dataclass
class ContinuationLeafConfig:
    select: str = "gadget"          # "gadget" (primary) | "uniform" (bootstrap)
    rollout_samples: int = 4        # board runouts (CRN across matrix cells) to average over
    gadget_iters: int = 256         # regret-matching iterations on the 4x4 matrix (cheap)
    max_playout_depth: int = 64     # bound pathological deep lines
    base_seed: int = 0              # CRN seed base (per-leaf the solver may offset this)
    resample_board: bool = True     # reshuffle the undealt deck tail per runout (integrate board luck)


def _reshuffle_undealt(env, rng: np.random.Generator):
    """Rebuild the UNDEALT tail of the deck as a shuffled permutation of the UNSEEN cards (all 52 minus
    every set hole card and revealed board card), so ``sample_chance_np`` reveals a fresh runout that
    can never duplicate a hole/board card. Reconstructing from the unseen complement (rather than just
    permuting the existing tail) makes the leaf self-correcting when the solver overwrites ``env.holes``
    with sampled opponent holes without fixing the deck bookkeeping."""
    holes = np.asarray(env.holes).reshape(-1)
    board = np.asarray(env.board).reshape(-1)
    known = set(int(c) for c in holes if int(c) >= 0) | set(int(c) for c in board if int(c) >= 0)
    unseen = np.array([c for c in range(52) if c not in known], dtype=np.int32)
    rng.shuffle(unseen)
    deck = np.asarray(env.deck).copy()
    ptr = int(env.deck_ptr)
    n = min(unseen.shape[0], deck.shape[0] - ptr)
    if n > 0:
        deck[ptr:ptr + n] = unseen[:n]
    return env._replace(deck=deck)


def _sample_action(row: np.ndarray, mask: np.ndarray, u: float) -> int:
    """Inverse-CDF sample an action index from a probability row restricted to ``mask`` (u in [0,1))."""
    legal = np.flatnonzero(mask)
    p = row[legal]
    s = p.sum()
    if s <= 1e-12:
        return int(legal[0])
    p = p / s
    pick = int(np.searchsorted(np.cumsum(p), float(u), side="right"))
    return int(legal[min(pick, legal.shape[0] - 1)])


def _playout(
    env,
    hero: int,
    adapter: TabularBlueprintAdapter,
    choice_fn,
    rng: np.random.Generator,
    max_depth: int,
    memo=None,
    bias_mult=None,
) -> float:
    """Play ``env`` to showdown; at each acting seat use the continuation selected by ``choice_fn(seat)``.

    ``choice_fn(seat) -> int`` returns the continuation index (0..3) that seat commits to (constant across
    the playout — matching the gadget's "same choice across indistinguishable infosets"). ``memo`` is the
    per-leaf continuation-set cache (None disables it). ``bias_mult`` (R2): explicit continuation-bias
    multiplier override (None = adapter default, byte-identical). Returns the hero terminal value in bb
    (``rewards[hero]/CHIP_UNIT``).
    """
    depth = 0
    while depth < max_depth and not bool(env.done):
        if is_chance_node_np(env):
            env = sample_chance_np(env)
            continue
        seat = int(env.action_seat)
        if seat < 0:
            break
        cont, mask = _cont_and_mask(env, seat, adapter, memo, bias_mult=bias_mult)
        row = cont[int(choice_fn(seat))]
        a = _sample_action(row, mask, float(rng.random()))
        env = apply_action_np(env, a)
        depth += 1
    return float(env.rewards[hero]) / CHIP_UNIT


class _DrawStream:
    """Depth-indexed view of ONE per-runout action-rng stream, shared across all DAG branches.

    A baseline cell draws ``rng.random()`` once per acting (non-chance) node, in traversal order, from a
    Generator seeded with the per-runout seed. The k-th draw is therefore a pure function of (seed, k) —
    identical for every (i,j) cell that reaches depth k. This memoizes that sequence so any DAG branch at
    depth ``d`` reads ``self[d]`` == the (d+1)-th draw a fresh baseline rng would produce. Bit-identical."""

    __slots__ = ("_rng", "_vals")

    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)
        self._vals: list[float] = []

    def __getitem__(self, depth: int) -> float:
        vals = self._vals
        while len(vals) <= depth:
            vals.append(float(self._rng.random()))
        return vals[depth]


def _dag_runout(env, hero: int, adapter, M: np.ndarray, stream: _DrawStream,
                cells: list[tuple[int, int]], depth: int, max_depth: int, memo) -> None:
    """Walk the gadget DAG from ``env`` for the subset of (i,j) ``cells`` still alive on this path.

    Bit-identical to running ``_playout`` independently for each cell (same per-cell action draws via the
    shared depth-indexed ``stream``), but engine work on a shared prefix is done ONCE. At each acting node
    the continuation INDEX a cell uses is ``i`` for the hero seat and ``j`` for any opponent seat; cells
    that pick the same action follow the same child. Terminals add ``rewards[hero]/CHIP_UNIT`` into M for
    every cell that ended there."""
    while True:
        if depth >= max_depth or bool(env.done):
            v = float(env.rewards[hero]) / CHIP_UNIT
            for (i, j) in cells:
                M[i, j] += v
            return
        if is_chance_node_np(env):
            env = sample_chance_np(env)  # deterministic; draws NO action-rng (no depth advance)
            continue
        seat = int(env.action_seat)
        if seat < 0:
            v = float(env.rewards[hero]) / CHIP_UNIT
            for (i, j) in cells:
                M[i, j] += v
            return
        cont, mask = _cont_and_mask(env, seat, adapter, memo)
        u = stream[depth]  # the SAME draw every alive cell consumes at this node (one per acting node)
        is_hero = (seat == hero)
        # The continuation index that drives the action is i for the hero seat, else j. Compute each
        # distinct index's action ONCE, then route cells to the child for their action.
        act_for_idx: dict[int, int] = {}
        children: dict[int, list[tuple[int, int]]] = {}
        for (i, j) in cells:
            idx = i if is_hero else j
            a = act_for_idx.get(idx)
            if a is None:
                a = _sample_action(cont[idx], mask, u)
                act_for_idx[idx] = a
            children.setdefault(a, []).append((i, j))
        if len(children) == 1:
            # Whole alive set takes the same action -> stay on this frame (no Python recursion overhead).
            a, sub = next(iter(children.items()))
            env = apply_action_np(env, a)
            depth += 1
            cells = sub
            continue
        for a, sub in children.items():
            _dag_runout(apply_action_np(env, a), hero, adapter, M, stream, sub,
                        depth + 1, max_depth, memo)
        return


def _payoff_matrix(
    env, hero: int, adapter: TabularBlueprintAdapter, cfg: ContinuationLeafConfig
) -> np.ndarray:
    """hero×field payoff matrix ``M[i,j]`` = mean hero bb when hero plays continuation ``i`` and every
    active opponent plays continuation ``j``. CRN: runout sample ``r`` uses the same rng seed for every
    (i,j) cell, so cross-cell differences isolate the continuation choice (not the board luck)."""
    k = NUM_CONTINUATIONS
    M = np.zeros((k, k), dtype=np.float64)
    if not cfg.resample_board:
        # GUARD: the env from state_bridge carries an UNREPAIRED deck (arange(52), ptr 0). The per-runout
        # _reshuffle_undealt below is what removes the in-play hole/board cards; with resample_board=False
        # there is no repair, so _playout reveals DUPLICATE board cards and corrupts every leaf value.
        # resample_board=False is only valid with a pre-repaired env (not the live bridge deck).
        import warnings
        warnings.warn("pluribus_rollout: resample_board=False with an unrepaired bridge deck deals "
                      "duplicate cards -> invalid leaf values; use resample_board=True.", RuntimeWarning)
    memo = {} if leaf_memo_enabled() else None
    batch = leaf_gadget_batch_enabled()
    all_cells = [(i, j) for i in range(k) for j in range(k)] if batch else None
    for r in range(cfg.rollout_samples):
        # One board per runout (CRN: identical across all i,j cells); fresh action rng per cell but seeded
        # from the same per-runout base so cross-cell differences isolate the continuation choice.
        env_r = (
            _reshuffle_undealt(env, np.random.default_rng(cfg.base_seed + 7919 * (r + 1)))
            if cfg.resample_board else env
        )
        if batch:
            # PREFIX-SHARING DAG: one walk fills all 16 cells for this runout, sharing engine work on
            # common path prefixes. The shared depth-indexed stream makes each cell's action draws
            # byte-identical to the per-cell loop below. (env-gated POKERBOT_LEAF_GADGET_BATCH)
            stream = _DrawStream(cfg.base_seed + 1000003 * (r + 1))
            _dag_runout(env_r, hero, adapter, M, stream, list(all_cells), 0,
                        cfg.max_playout_depth, memo)
            continue
        for i in range(k):
            for j in range(k):
                rng = np.random.default_rng(cfg.base_seed + 1000003 * (r + 1))
                choice = (lambda s, _i=i, _j=j: _i if s == hero else _j)
                M[i, j] += _playout(env_r, hero, adapter, choice, rng, cfg.max_playout_depth, memo)
    return M / float(cfg.rollout_samples)


def _solve_zero_sum(M: np.ndarray, iters: int) -> float:
    """Game value to the row player (hero, maximiser) of zero-sum matrix ``M`` via regret matching.

    Hero maximises; the field (column player) minimises, i.e. plays to maximise ``-M``. Standard
    self-play regret matching on a tiny matrix converges fast; we return the average-strategy value.
    """
    k = M.shape[0]
    reg_r = np.zeros(k)
    reg_c = np.zeros(k)
    sum_r = np.zeros(k)
    sum_c = np.zeros(k)

    def _rm(reg: np.ndarray) -> np.ndarray:
        pos = np.maximum(reg, 0.0)
        s = pos.sum()
        return pos / s if s > 1e-12 else np.full(k, 1.0 / k)

    for _ in range(iters):
        xr = _rm(reg_r)
        xc = _rm(reg_c)
        sum_r += xr
        sum_c += xc
        # Hero (row) value of each pure action vs current column mix; field sees -M.
        vr = M @ xc                 # row payoffs (maximise)
        vc = -(xr @ M)              # column payoffs on -M (maximise)
        reg_r += vr - float(xr @ vr)
        reg_c += vc - float(xc @ vc)
    avg_r = sum_r / max(sum_r.sum(), 1e-12)
    avg_c = sum_c / max(sum_c.sum(), 1e-12)
    return float(avg_r @ M @ avg_c)


def continuation_leaf_value(
    env,
    hero: int,
    adapter: TabularBlueprintAdapter,
    cfg: ContinuationLeafConfig | None = None,
    cont_weights: dict[int, np.ndarray] | None = None,
) -> float:
    """Leaf value (hero bb) at depth-limit state ``env`` via the continuation gadget (or uniform bootstrap).

    ``env`` must have all hole cards specified (the solver fills sampled opponent holes before the leaf).

    ``cont_weights`` (MODEL-AWARE LEAF): optional ``{seat: (NUM_CONTINUATIONS,) probs}`` biasing that
    seat's continuation DRAW in uniform mode toward its opponent-model read (a station draws the
    call-continuation more, a nit the fold one) — so an exploitative line's payoff is visible in the
    leaf value instead of being washed out by a read-blind playout world. Seats absent from the dict
    (and the hero) draw uniform via the LEGACY rng path, so with ``cont_weights=None`` the function
    is bit-identical to before (the zero-deviation gate)."""
    cfg = cfg or ContinuationLeafConfig()
    if bool(env.done):
        return float(env.rewards[hero]) / CHIP_UNIT

    if cfg.select == "uniform":
        # NATIVE FAST PATH (opt-in): the whole uniform playout in Rust, no per-step marshaling. Only
        # the read-blind uniform draw (cont_weights is None) is ported; model-aware leaves stay Python.
        if cont_weights is None and _native_rollout_enabled():
            from pokerbot.engine.jax_weva_encode import _POSTFLOP_NUM_BUCKETS

            v = native_uniform_leaf_value(
                env, hero, adapter,
                base_seed=cfg.base_seed,
                rollout_samples=cfg.rollout_samples,
                resample_board=cfg.resample_board,
                max_playout_depth=cfg.max_playout_depth,
                num_buckets=int(_POSTFLOP_NUM_BUCKETS),
            )
            if v is not None:
                return float(v)
            # else: native unavailable/unsupported config -> fall through to the Python loop.

        total = 0.0
        memo = {} if leaf_memo_enabled() else None
        for r in range(cfg.rollout_samples):
            env_r = (
                _reshuffle_undealt(env, np.random.default_rng(cfg.base_seed + 7919 * (r + 1)))
                if cfg.resample_board else env
            )
            rng = np.random.default_rng(cfg.base_seed + 1000003 * (r + 1))
            # Each seat draws an independent continuation, fixed for the playout.
            draws: dict[int, int] = {}

            def choice(s: int, _rng=rng, _d=draws) -> int:
                c = _d.get(s)
                if c is None:
                    w = cont_weights.get(s) if (cont_weights is not None and s != hero) else None
                    if w is None:
                        c = int(_rng.integers(0, NUM_CONTINUATIONS))  # legacy path, bit-identical
                    else:
                        cum = np.cumsum(w)
                        c = int(min(np.searchsorted(cum, float(_rng.random()), side="right"),
                                    NUM_CONTINUATIONS - 1))
                    _d[s] = c
                return c

            total += _playout(env_r, hero, adapter, choice, rng, cfg.max_playout_depth, memo)
        return total / float(cfg.rollout_samples)

    if cfg.select == "gadget":
        M = _payoff_matrix(env, hero, adapter, cfg)
        return _solve_zero_sum(M, cfg.gadget_iters)

    raise ValueError(f"unknown continuation_select: {cfg.select!r}")


def biased_profile_leaf_value(
    env,
    hero: int,
    adapter: TabularBlueprintAdapter,
    cfg: ContinuationLeafConfig,
    choices: dict[int, int],
    bias_mult: float,
) -> float:
    """R2 (Pluribus biased continuation leaves): leaf value under a FIXED joint continuation profile.

    ``choices`` maps seat -> continuation index (0=blueprint, 1=fold-biased, 2=call-biased,
    3=raise-biased); the profile is chosen by the subgame CFR's per-seat 4-way choice nodes (see
    ``PluribusReBeLSolver._leaf_value_biased``), NOT drawn here — this function only integrates board
    runouts under that fixed profile. A seat absent from ``choices`` (folded / all-in at the leaf — it
    never acts) defensively plays continuation 0 (the unaltered blueprint). ``bias_mult`` is the
    solve-level multiplier (Pluribus 5 / Modicum 10), overriding the adapter-level default.

    Runout/seed schedule (7919 / 1000003 offsets from ``cfg.base_seed``) matches the uniform leaf, so
    the two paths integrate statistically-identical board luck; only the continuation choices differ."""
    if bool(env.done):
        return float(env.rewards[hero]) / CHIP_UNIT
    total = 0.0
    memo = {} if leaf_memo_enabled() else None

    def choice(s: int) -> int:
        return int(choices.get(s, 0))

    for r in range(cfg.rollout_samples):
        env_r = (
            _reshuffle_undealt(env, np.random.default_rng(cfg.base_seed + 7919 * (r + 1)))
            if cfg.resample_board else env
        )
        rng = np.random.default_rng(cfg.base_seed + 1000003 * (r + 1))
        total += _playout(env_r, hero, adapter, choice, rng, cfg.max_playout_depth, memo,
                          bias_mult=bias_mult)
    return total / float(cfg.rollout_samples)
