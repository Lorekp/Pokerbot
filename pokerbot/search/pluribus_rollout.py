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


def _cont_and_mask(env, seat: int, adapter: "TabularBlueprintAdapter", memo):
    """Return (continuation_rows, full_legal_mask) for (env, seat), using the per-leaf ``memo`` dict
    when enabled. ``continuation_rows`` is the (NUM_CONTINUATIONS, NUM_ACTIONS) abstraction-masked
    strategy stack; ``full_legal_mask`` is the UN-abstracted legal bool mask — exactly what the legacy
    ``_playout`` passed to ``_sample_action`` (it samples the abstraction-masked row restricted to the
    full legal index set; the two agree because the row is zero outside the abstraction support)."""
    if memo is not None:
        key = _infoset_sig(env, seat)
        hit = memo.get(key)
        if hit is not None:
            return hit
        mask = legal_action_mask_np(env, seat).astype(bool)
        cont = adapter.continuation_set(env, seat)
        memo[key] = (cont, mask)
        return cont, mask
    mask = legal_action_mask_np(env, seat).astype(bool)
    cont = adapter.continuation_set(env, seat)
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
) -> float:
    """Play ``env`` to showdown; at each acting seat use the continuation selected by ``choice_fn(seat)``.

    ``choice_fn(seat) -> int`` returns the continuation index (0..3) that seat commits to (constant across
    the playout — matching the gadget's "same choice across indistinguishable infosets"). ``memo`` is the
    per-leaf continuation-set cache (None disables it). Returns the hero terminal value in bb
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
        cont, mask = _cont_and_mask(env, seat, adapter, memo)
        row = cont[int(choice_fn(seat))]
        a = _sample_action(row, mask, float(rng.random()))
        env = apply_action_np(env, a)
        depth += 1
    return float(env.rewards[hero]) / CHIP_UNIT


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
    for r in range(cfg.rollout_samples):
        # One board per runout (CRN: identical across all i,j cells); fresh action rng per cell but seeded
        # from the same per-runout base so cross-cell differences isolate the continuation choice.
        env_r = (
            _reshuffle_undealt(env, np.random.default_rng(cfg.base_seed + 7919 * (r + 1)))
            if cfg.resample_board else env
        )
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
