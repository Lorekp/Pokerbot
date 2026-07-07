"""Tabular blueprint policy + the FOUR Pluribus continuation strategies (numpy / np_engine).

This is the P1 bedrock for the Pluribus-faithful depth-limited search. It wraps the converged
``{fingerprint -> strat row}`` blueprint snapshot as a numpy-native policy ``(env, seat) -> probs(11)``
(reusing ``eval/lbr.make_blueprint_policy`` verbatim so lookups are byte-identical to the trained table),
and derives the **four continuation strategies** used at the depth-limit leaf gadget:

  1. ``blueprint``    — the blueprint itself.
  2. ``fold``         — fold-biased: fold-prob × ``bias_mult`` (~10), renormalized over the legal mask.
  3. ``call``         — call-biased: check/call mass × ``bias_mult``.
  4. ``raise``        — raise-biased: aggressive-bin (bet + all-in) mass × ``bias_mult``.

Why these four (Brown & Sandholm, Science 2019; NeurIPS 2018 "Depth-Limited Solving"): at the depth limit
every player simultaneously CHOOSES one of these continuations (the gadget solved in ``pluribus_rollout``),
so a k-strategy menu spans k**(#leaf infosets) opponent behaviours — that exponential coverage is what makes
the leaf estimate bias-resistant WITHOUT a value net. Fewer than ten suffice empirically; four is Pluribus's
choice and spans the relevant fold/call/raise axes.

FIREWALL NOTE: these continuation tilts are **leaf-robustness** constructs and are deliberately INDEPENDENT
of the opponent-EXPLOITATION tilts (``apply_opponent_bias`` / ``apply_context_bias`` in ``turbo_rebel_search``).
They never touch ``opponent/context.py`` keys, the 11->4 ``action_bin`` map, or the gift/bounded-eps math.
The exploitation deviation is applied separately at opponent nodes inside the solver and remains byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pokerbot.engine.jax_engine import CHIP_UNIT as _CHIP_UNIT  # BB_CHIPS == CHIP_UNIT (chip units)
from pokerbot.engine.np_engine import (
    ALL_IN,
    BET_BUCKET_START,
    CALL,
    CHECK,
    FOLD,
    NUM_ACTIONS,
)
# Native dispatch (Rust when enabled, else np_engine) for the per-leaf-seat legal mask.
from pokerbot.search.native_solver import legal_action_mask_np

# Action groups (np_engine layout: 0 fold, 1 check, 2 call, 3..9 pot-fraction bets, 10 all-in).
_FOLD_IDX = np.array([FOLD], dtype=np.intp)
_PASSIVE_IDX = np.array([CHECK, CALL], dtype=np.intp)
_AGGRO_IDX = np.array([a for a in range(BET_BUCKET_START, NUM_ACTIONS)], dtype=np.intp)  # bets + all-in

# Canonical continuation order. Index 0 is ALWAYS the unbiased blueprint (the solver's opponent-node mix
# and the gadget both rely on this ordering).
CONTINUATION_NAMES: tuple[str, ...] = ("blueprint", "fold", "call", "raise")
NUM_CONTINUATIONS = len(CONTINUATION_NAMES)
_BIAS_GROUPS = (None, _FOLD_IDX, _PASSIVE_IDX, _AGGRO_IDX)

DEFAULT_BIAS_MULT = 10.0  # Pluribus: fold-prob multiplied by ~10 then renormalized.


def _build_allowed_action_mask(bet_bucket_keep: tuple[int, ...] | None) -> np.ndarray | None:
    """Lean copy of the training action-abstraction mask."""
    if bet_bucket_keep is None:
        return None
    allowed = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in (FOLD, CHECK, CALL, ALL_IN):
        allowed[a] = True
    n_buckets = ALL_IN - BET_BUCKET_START
    for i in bet_bucket_keep:
        if not 0 <= int(i) < n_buckets:
            raise ValueError(f"bet_bucket_keep index {i} out of range [0,{n_buckets})")
        allowed[BET_BUCKET_START + int(i)] = True
    return allowed


def _apply_abstraction_mask(
    mask: np.ndarray, street: int, allowed_pre: np.ndarray | None, allowed_post: np.ndarray | None
) -> np.ndarray:
    """Lean copy of ``sd_cfr_nlhe_hogwild._apply_abstraction``."""
    if allowed_pre is None and allowed_post is None:
        return mask
    allowed = allowed_pre if street == 0 else allowed_post
    if allowed is None:
        return mask
    return mask & allowed


def _renorm_legal(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero illegal actions and renormalize; uniform-legal fallback if all mass vanished."""
    q = np.asarray(p, dtype=np.float64) * mask
    s = q.sum()
    if s > 1e-12:
        return q / s
    m = mask.astype(np.float64)
    return m / max(m.sum(), 1.0)


def bias_continuation(
    blueprint_probs: np.ndarray,
    group_idx: np.ndarray | None,
    mask: np.ndarray,
    *,
    bias_mult: float = DEFAULT_BIAS_MULT,
    bias_floor: float = 0.0,
) -> np.ndarray:
    """Pluribus continuation tilt: multiply ``group_idx`` mass by ``bias_mult``, renormalize over legal.

    ``group_idx is None`` returns the (legal-renormalized) blueprint unchanged.

    ``bias_floor`` (default 0 = faithful) optionally injects a tiny uniform mass onto the *legal* target
    actions BEFORE the multiply, so a continuation can still bias toward an action the blueprint zeroed at
    that infoset (otherwise e.g. a never-fold node yields fold==blueprint). Keep 0 for byte-faithfulness;
    a small value (e.g. 1e-3) strengthens leaf robustness at nodes where the blueprint is near-pure.
    """
    p = np.asarray(blueprint_probs, dtype=np.float64) * mask
    if group_idx is None:
        return _renorm_legal(p, mask)
    legal = mask.astype(bool)
    if bias_floor > 0.0:
        tgt_legal = [int(i) for i in group_idx if legal[i]]
        if tgt_legal:
            p = p.copy()
            p[tgt_legal] += bias_floor
    p = p.copy()
    sel = [int(i) for i in group_idx if legal[i]]
    if sel:
        p[sel] *= float(bias_mult)
    return _renorm_legal(p, mask)


@dataclass
class TabularBlueprintAdapter:
    """Blueprint lookup + continuation-strategy derivation over the numpy engine.

    ``snapshot``: the trained ``{fingerprint(bytes) -> strat row (11,)}`` table.
    ``client_bet_sizing``: apply the SAME street action abstraction as training (so lookups hit).
    ``bias_mult`` / ``bias_floor``: continuation tilt knobs (see ``bias_continuation``).
    """

    snapshot: dict[bytes, np.ndarray]
    client_bet_sizing: bool = True
    bias_mult: float = DEFAULT_BIAS_MULT
    bias_floor: float = 0.0

    def __post_init__(self) -> None:
        # Reuse the LBR blueprint policy verbatim: numpy engine + fingerprint_fast == trained key space.
        # _PRE_KEEP/_PRE_RR_KEEP/_POST_KEEP are the trainer keep-list twins (POKERBOT_PRE_KEEP /
        # POKERBOT_PRE_RR_KEEP / POKERBOT_POST_KEEP, blueprint-v4 arm-3); all-unset defaults
        # ((4,5,6) everywhere preflop / (0,1,3,4,5)) => byte-identical.
        from pokerbot.eval.lbr import _POST_KEEP, _PRE_KEEP, _PRE_RR_KEEP, make_blueprint_policy

        self._policy = make_blueprint_policy(self.snapshot, client_bet_sizing=self.client_bet_sizing)
        # Precompute the street action-abstraction masks ONCE (rebuilding per-leaf is wasteful).
        if self.client_bet_sizing:
            self._allowed_pre = _build_allowed_action_mask(_PRE_KEEP)
            # Preflop RE-raise (3-bet+) menu; None when the env is unset => _allowed_pre everywhere.
            self._allowed_pre_rr = (_build_allowed_action_mask(_PRE_RR_KEEP)
                                    if _PRE_RR_KEEP is not None else None)
            self._allowed_post = _build_allowed_action_mask(_POST_KEEP)
        else:
            self._allowed_pre = None
            self._allowed_pre_rr = None
            self._allowed_post = None

    def blueprint(self, env, seat: int) -> np.ndarray:
        """The unbiased blueprint policy at ``(env, seat)`` (already legal-masked + renormalized)."""
        return np.asarray(self._policy(env, seat), dtype=np.float64)

    # Backwards/clarity alias — the solver's opponent-node mix uses the plain blueprint.
    __call__ = blueprint

    def continuation_set(self, env, seat: int, bias_mult: float | None = None) -> np.ndarray:
        """All four continuation strategies stacked as ``(NUM_CONTINUATIONS, NUM_ACTIONS)``.

        Row 0 is the blueprint; rows 1-3 are fold/call/raise-biased. Each row sums to 1 over the legal
        (abstracted) mask. The blueprint policy is evaluated ONCE and reweighted, so this is one snapshot
        lookup + cheap arithmetic per leaf seat.

        ``bias_mult`` (optional): explicit multiplier override for the biased rows — used by the R2
        biased-continuation leaf, whose solve-level ``leaf_bias_mult`` knob (Pluribus's 5) replaces the
        adapter-level default. ``None`` (default) uses ``self.bias_mult``, byte-identical to before.
        """
        mult = self.bias_mult if bias_mult is None else float(bias_mult)
        mask = legal_action_mask_np(env, seat).astype(bool)
        if self.client_bet_sizing:
            # arm-3 preflop split, same predicate as lbr.make_blueprint_policy: a preflop node with a
            # raise already in (current_bet > BB_CHIPS == CHIP_UNIT; limps don't move current_bet)
            # uses the RE-raise menu when POKERBOT_PRE_RR_KEEP is set. Unset => byte-identical.
            if (self._allowed_pre_rr is not None and int(env.street) == 0
                    and float(env.current_bet) > float(_CHIP_UNIT)):
                mask = (mask & self._allowed_pre_rr).astype(bool)
            else:
                mask = _apply_abstraction_mask(
                    mask, int(env.street), self._allowed_pre, self._allowed_post
                ).astype(bool)
        bp = self.blueprint(env, seat)
        out = np.empty((NUM_CONTINUATIONS, NUM_ACTIONS), dtype=np.float64)
        for r, group in enumerate(_BIAS_GROUPS):
            out[r] = bias_continuation(
                bp, group, mask, bias_mult=mult, bias_floor=self.bias_floor
            )
        return out
