"""Pluggable environment adapter for the VR-DeepPDCFR+ trainer.

The trainer (``neural_cfr.py``) was hardwired to the 6-max NLHE ``jax_engine``. To VALIDATE the algorithm
against a game with computable exact exploitability (the pre-cloud gate), the trainer must run on Leduc
too. ``EnvSpec`` is the seam: a hashable bundle of the env's pure functions + dimensions, threaded
through the trainer as a static argument. Two adapters: :func:`nlhe_env_spec` (the production game,
reproduces prior behavior exactly) and :func:`leduc_env_spec` (the exploitability gate).

An adapter must expose, over the env's own state type:
  reset(key)->S, step(S,a)->S, is_chance(S)->bool, sample_chance(S)->S, legal_mask(S)->bool[A],
  encode_info(S,seat)->f32[info_dim]  (the acting player's INFOSET; never leaks opponent cards),
  encode_history(S)->f32[history_dim] (the PUBLIC history; no private cards -- the VR baseline input),
  action_seat(S)->int, done(S)->bool, rewards(S)->f32[num_players],
plus the static ints num_players, num_actions, info_dim, history_dim.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class EnvSpec(NamedTuple):
    """Hashable env adapter (callables + ints) -- usable as a jax.jit static argument."""

    num_players: int
    num_actions: int
    info_dim: int
    history_dim: int
    reset: Callable[[Array], Any]
    step: Callable[[Any, Array], Any]
    is_chance: Callable[[Any], Array]
    sample_chance: Callable[[Any], Any]
    legal_mask: Callable[[Any], Array]
    encode_info: Callable[[Any, Array], Array]
    encode_history: Callable[[Any], Array]
    action_seat: Callable[[Any], Array]
    done: Callable[[Any], Array]
    rewards: Callable[[Any], Array]
    # Fixed scan horizon for one hand. The traversal scan runs exactly this many steps and masks
    # everything past terminal, so this MUST be >= the true maximum number of decision/transition
    # steps any realizable hand can take (truncating a real hand changes the estimator and is a
    # correctness bug). Measured worst case (never-fold + random raise sizes, 600k hands): NLHE=28,
    # Leduc=8; bounds below carry >2x margin. Default 128 preserves the legacy global bound for any
    # adapter that does not set it. See scripts/stress_hand_length2.py for the empirical ceiling.
    max_steps: int = 128


# ---------------------------------------------------------------------------
# NLHE adapter (production game) -- wraps jax_engine, reproduces prior behavior
# ---------------------------------------------------------------------------


def nlhe_env_spec() -> EnvSpec:
    from pokerbot.engine import jax_engine as je

    return EnvSpec(
        num_players=je.NUM_PLAYERS,
        num_actions=je.NUM_ACTIONS,
        info_dim=je.INFO_FEATURE_DIM,
        history_dim=je.HISTORY_FEATURE_DIM,
        reset=je.reset_state,
        step=je.step_state,
        is_chance=je.is_chance_node,
        sample_chance=je.sample_chance,
        legal_mask=je.legal_action_mask,
        encode_info=je.encode_info_state,
        encode_history=je.encode_history_state,
        action_seat=lambda s: s.action_seat,
        done=lambda s: s.done,
        rewards=lambda s: s.rewards.astype(jnp.float32),
        # Worst-case NLHE hand depth measured at 28 (never-fold + random sizes, 600k hands); 64
        # carries >2x margin and never truncates a real hand. Halves the scan vs the legacy 128.
        max_steps=64,
    )


# ---------------------------------------------------------------------------
# Heads-up (2-player) NLHE adapter -- the CORRECT target for VR-DeepPDCFR+
# ---------------------------------------------------------------------------


def _read_hu_depth_band() -> tuple[float, float] | None:
    """POKERBOT_HU_DEPTH_BB='lo,hi' -> per-deal log-uniform effective stack (in BB). Unset -> None
    (fixed 100bb starts, byte-identical to the legacy reset). Tournament HU is late & high-blind, so the
    intended band is SHALLOW (e.g. 5,60); a deeper band covers cash too."""
    raw = os.environ.get("POKERBOT_HU_DEPTH_BB", "").strip()
    if not raw:
        return None
    try:
        lo_s, hi_s = raw.split(",")
        lo, hi = float(lo_s), float(hi_s)
    except ValueError as e:
        raise ValueError(f"POKERBOT_HU_DEPTH_BB must be 'lo,hi' BB, got {raw!r}") from e
    if not (0 < lo <= hi):
        raise ValueError(f"POKERBOT_HU_DEPTH_BB needs 0 < lo <= hi, got {lo},{hi}")
    return lo, hi


def _read_hu_ante() -> tuple[float, float]:
    """POKERBOT_HU_ANTE_BB (per-player ante in BB, default 0 = no ante = cash/legacy) and
    POKERBOT_HU_ANTE_ZERO_PROB (default 0 = always ante; >0 zeroes that fraction of deals so ONE net
    also covers no-ante/cash HU, conditioning on the encoded pot)."""
    ante_bb = float(os.environ.get("POKERBOT_HU_ANTE_BB", "0") or "0")
    zero_p = float(os.environ.get("POKERBOT_HU_ANTE_ZERO_PROB", "0") or "0")
    if ante_bb < 0 or not (0.0 <= zero_p <= 1.0):
        raise ValueError("POKERBOT_HU_ANTE_BB must be >= 0 and POKERBOT_HU_ANTE_ZERO_PROB in [0,1]")
    return ante_bb, zero_p


def nlhe_hu_env_spec(seat_a: int = 3, seat_b: int = 5) -> EnvSpec:
    """Heads-up (2-player) NLHE adapter for neural CFR.

    The production ``jax_engine`` is hardwired to 6 seats, but the VR baseline correction in
    ``neural_cfr`` (``persp_sign = +1`` for the traverser, ``-1`` for the next/opponent seat) is
    EXACT only in a 2-player zero-sum game. This adapter pins EXACTLY two live seats via occupancy;
    the four dead seats never act (``action_seat`` never points to them) and contribute zero reward,
    so the engine plays a genuine heads-up game and the estimator's zero-sum perspective flip is
    valid between the two live seats. The button rotates between the two live seats each hand so
    both blind positions are learned. ``num_players`` stays 6 (dead seats are inert -> a little
    wasted traversal, never a correctness issue) and ``reward_scale`` stays the full stack. Crucially
    the env DIMENSIONS (num_players/num_actions/info_dim/history_dim) are identical to
    :func:`nlhe_env_spec`, so a trainer/checkpoint built for one is buffer- and net-compatible with
    the other: only ``reset`` differs (occupancy), which is a jit static arg.
    """
    from pokerbot.engine import jax_engine as je

    base = nlhe_env_spec()
    occ = jnp.array([(i == seat_a or i == seat_b) for i in range(je.NUM_PLAYERS)], dtype=bool)
    live = jnp.array([seat_a, seat_b], dtype=jnp.int32)
    # Tournament/cash training distribution, read once at spec build (env-gated; defaults reproduce the
    # legacy fixed-100bb no-ante reset exactly). Captured as Python constants -> static under jit; only
    # the per-deal draws are traced.
    band = _read_hu_depth_band()
    ante_bb, ante_zero_p = _read_hu_ante()
    unit = je.CHIP_UNIT

    def reset(key: Array):
        k_btn, k_hand, k_depth, k_ante = jax.random.split(key, 4)
        button = live[jax.random.randint(k_btn, (), 0, 2)]  # rotate SB/BB across the 2 live seats
        # Per-deal effective depth: both live seats share it. Chip-EV only needs the COVERING (shorter)
        # stack -- you can't win more than that heads-up -- so symmetric stacks suffice; asymmetric-stack
        # ($/ICM) effects are handled by the serve-time ICM overlay, not the chip-EV net.
        if band is None:
            stacks = None
        else:
            lo, hi = band
            u = jax.random.uniform(k_depth, (), minval=math.log(lo), maxval=math.log(hi))
            chips = jnp.round(jnp.exp(u) * unit).astype(jnp.int32)
            stacks = jnp.full((je.NUM_PLAYERS,), chips, dtype=jnp.int32)
        # Per-player ante (tournament). Optionally zero a fraction of deals so one net also covers
        # no-ante cash HU (it conditions on the encoded pot, which carries the antes).
        if ante_bb <= 0.0:
            ante = jnp.int32(0)
        else:
            base_ante = jnp.int32(round(ante_bb * unit))
            ante = (jnp.where(jax.random.uniform(k_ante, ()) < ante_zero_p, jnp.int32(0), base_ante)
                    if ante_zero_p > 0.0 else base_ante)
        return je.init_hand(k_hand, button=button, stacks=stacks, occupancy=occ, ante=ante)

    return base._replace(reset=reset)


# ---------------------------------------------------------------------------
# Leduc adapter (exploitability gate) -- wraps leduc_jax + injective encodings
# ---------------------------------------------------------------------------

# Leduc has 3 card ranks, public in {-1(none),0,1,2}, up to MAX_HISTORY betting slots, 3 actions.
LEDUC_INFO_DIM: int = 96
LEDUC_HISTORY_DIM: int = 96


def _leduc_hist_onehot(leduc, state) -> Array:
    """One-hot of the betting sequence: MAX_HISTORY slots x 4 (none + 3 actions). Injective over
    Leduc betting histories (short trees), so distinct infosets get distinct encodings."""
    h = state.hist_action  # (MAX_HISTORY,), -1 = empty, else 0..2
    # map -1->0 (none), action a -> a+1 ; one-hot over 4 then flatten
    idx = jnp.where(h >= 0, h + 1, 0)
    return jax.nn.one_hot(idx, 4).reshape(-1)  # (MAX_HISTORY*4,)


def _leduc_encode_info(leduc):
    def encode_info(state, seat: Array) -> Array:
        own = jax.nn.one_hot(state.cards[seat], leduc.NUM_CARD_RANKS)                 # private (3)
        public = jax.nn.one_hot(jnp.where(state.public >= 0, state.public + 1, 0), leduc.NUM_CARD_RANKS + 1)  # (4)
        rnd = jax.nn.one_hot(state.round, 2)                                          # (2)
        acting = jax.nn.one_hot(state.acting, leduc.NUM_PLAYERS)                      # (2)
        scal = jnp.array([
            state.current_bet.astype(jnp.float32) / 10.0,
            state.raises.astype(jnp.float32) / float(leduc.MAX_RAISES),
            state.committed[0].astype(jnp.float32) / 10.0,
            state.committed[1].astype(jnp.float32) / 10.0,
            state.pot.astype(jnp.float32) / 20.0,
        ], dtype=jnp.float32)
        feat = jnp.concatenate([own, public, rnd, acting, scal, _leduc_hist_onehot(leduc, state)])
        pad = jnp.zeros(LEDUC_INFO_DIM, dtype=jnp.float32)
        return jax.lax.dynamic_update_slice(pad, feat, (0,))

    return encode_info


def _leduc_encode_history(leduc):
    def encode_history(state) -> Array:
        """PUBLIC history only -- no private cards (this is the VR baseline input)."""
        public = jax.nn.one_hot(jnp.where(state.public >= 0, state.public + 1, 0), leduc.NUM_CARD_RANKS + 1)
        rnd = jax.nn.one_hot(state.round, 2)
        acting = jax.nn.one_hot(state.acting, leduc.NUM_PLAYERS)
        scal = jnp.array([
            state.current_bet.astype(jnp.float32) / 10.0,
            state.raises.astype(jnp.float32) / float(leduc.MAX_RAISES),
            state.committed[0].astype(jnp.float32) / 10.0,
            state.committed[1].astype(jnp.float32) / 10.0,
            state.pot.astype(jnp.float32) / 20.0,
        ], dtype=jnp.float32)
        feat = jnp.concatenate([public, rnd, acting, scal, _leduc_hist_onehot(leduc, state)])
        pad = jnp.zeros(LEDUC_HISTORY_DIM, dtype=jnp.float32)
        return jax.lax.dynamic_update_slice(pad, feat, (0,))

    return encode_history


def leduc_env_spec() -> EnvSpec:
    """Leduc Hold'em adapter for the exploitability gate. Leduc has no explicit chance NODES -- the
    public-card deal happens inside ``apply_action`` (a stochastic transition, sampled once per hand
    via the state rng), so ``is_chance`` is always False and ``sample_chance`` is identity."""
    from pokerbot.engine import leduc_jax as leduc

    return EnvSpec(
        num_players=leduc.NUM_PLAYERS,
        num_actions=leduc.NUM_ACTIONS,
        info_dim=LEDUC_INFO_DIM,
        history_dim=LEDUC_HISTORY_DIM,
        reset=leduc.init_hand,
        step=lambda s, a: leduc.apply_action(s, a),
        is_chance=lambda s: jnp.bool_(False),
        sample_chance=lambda s: s,
        legal_mask=leduc.legal_actions,
        encode_info=_leduc_encode_info(leduc),
        encode_history=_leduc_encode_history(leduc),
        action_seat=lambda s: s.acting,
        done=lambda s: s.done,
        rewards=lambda s: s.rewards.astype(jnp.float32),
        # Worst-case Leduc hand depth measured at 8; 24 is 3x margin and never truncates.
        max_steps=24,
    )
