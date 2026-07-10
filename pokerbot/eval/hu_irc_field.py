"""Seat-rotated duplicate HU win-rate vs an IRC-calibrated heads-up field (Python eval, NO Rust crate).

WHY this exists (Workstream-E harness gap)
------------------------------------------
The E acceptance bundle wants a SEAT-ROTATED HU win-rate vs a realistic IRC HU field -- the one E
deliverable that was never built. The 6-max rotated harness
(``exploitation_battery.run_field_absolute_rotated``) runs on the Python ``GameState`` engine and a
6-villain homogeneous field; the HU neural net, by contrast, serves natively on the ``np_engine``
``EnvState`` (the SAME engine ``pokerbot.eval.lbr`` runs HU on, seats 3 & 5). Rather than bridge the
net onto the 6-seat ``GameState`` (a translation-skew minefield), this builds the duplicate cell
DIRECTLY on the ``np_engine`` HU game -- byte-compatible with the trained encoder + the
``make_neural_policy`` serve path -- so ANY ``policy_fn(env_np, seat)->probs`` (the dev checkpoint, the
blueprint fallback, or the future converged net) plugs straight in.

The IRC field is the same loose-passive population the value-extraction gauntlet calibrates to (see
``pokerbot.eval.irc_villains`` + ``data/player_profiles/irc_population_prior.json``: VPIP~48, PFR~10,
AF~0.65), here RESTRICTED to heads-up and re-expressed as ``np_engine`` policy_fns. Each archetype's
own-initiative aggression is gated on a Monte-Carlo equity-vs-random-range proxy (the SAME
phevaluator rollout ``lbr._equity_vs_combos`` uses), so the archetype thresholds (call_hs/raise_hs/
bet_hs) carry their IRC meaning unchanged.

SEAT-ROTATED DUPLICATE (the variance reduction)
-----------------------------------------------
For each distinct deck seed we play the SAME deck TWICE with the button FIXED on seat 3: once with the
hero as the button seat (3) and once as the big-blind seat (5). The hero therefore holds BOTH dealt
hole-hands once and sits in BOTH blind positions once; the dominant hole-card-allocation luck cancels
in the per-deck 2-seat MEAN (duplicate-bridge / AoF ``absolute_winrate`` seat rotation, ported to HU).
We report the per-deck seat-mean's mean as bb/100 with a paired-across-decks CI. ``allin_ev`` settles
any all-in confrontation by EXACT enumeration-free Monte-Carlo runout average (chance is sampled by the
engine here; for HU shove/fold the variance is already tiny, so we keep the engine's sampled runout and
let the duplicate rotation + deck count drive precision -- mirroring ``lbr.py`` which also samples).

This is a Python-only harness: no ``tools/battery_native``, no engine-triad goldens, no AoF. It closes
the measurement gap regardless of which HU policy is plugged in. The number it returns for TODAY's
policy (dev checkpoint or blueprint fallback) is a CURRENT-POLICY baseline -- NOT the converged target.
The converged HU-vs-IRC number is gated on the ~24h converged training run (see docs/hu_engine_status).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from pokerbot.engine.jax_engine import CHIP_UNIT, init_hand  # noqa: E402
from pokerbot.engine.np_engine import (  # noqa: E402
    ALL_IN,
    BET_BUCKET_START,
    CALL,
    CHECK,
    FOLD,
    NUM_ACTIONS,
    apply_action_np,
    is_chance_node_np,
    leaf_terminal_reward_np,
    legal_action_mask_np,
    sample_chance_np,
)

from phevaluator import evaluate_cards as _phe  # noqa: E402

_CARD_STR = ["23456789TJQKA"[c // 4] + "cdhs"[c % 4] for c in range(52)]

_HU_OCC = np.array([False, False, False, True, False, True], dtype=bool)
_HU_LIVE = (3, 5)  # the two live seats (match lbr.py + nlhe_hu_env_spec)

# Pot-fraction -> bet-bucket index. BET_SIZE_BUCKETS = (.3333,.5,.66,.75,1.0,1.5,2.0); index = +START.
_BET_FRACS = (1.0 / 3.0, 0.5, 0.66, 0.75, 1.0, 1.5, 2.0)


# --------------------------------------------------------------------------- #
# Hand-strength proxy on the numpy EnvState (the IRC archetype gate)
# --------------------------------------------------------------------------- #
def _board_cards(env) -> np.ndarray:
    b = np.asarray(env.board)
    return b[b >= 0]


def _equity_vs_random(env, seat: int, rng: np.random.Generator, rollouts: int = 80) -> float:
    """Monte-Carlo equity of ``seat``'s hand vs ONE uniform-random opponent hand, averaged over random
    runouts. The same phevaluator rollout the LBR equity uses (lower rank == stronger). Returns [0,1] --
    directly comparable to the IRC archetype thresholds (which are equity-based)."""
    hole = np.asarray(env.holes[seat], dtype=np.int32)
    board = _board_cards(env)
    dead = {int(hole[0]), int(hole[1])} | {int(c) for c in board.tolist()}
    deck = [c for c in range(52) if c not in dead]
    need = 5 - board.shape[0]
    me = [_CARD_STR[int(hole[0])], _CARD_STR[int(hole[1])]]
    board_str = [_CARD_STR[int(c)] for c in board]
    wins = 0.0
    for _ in range(rollouts):
        pick = rng.choice(len(deck), size=2 + max(need, 0), replace=False)
        opp = [_CARD_STR[deck[pick[0]]], _CARD_STR[deck[pick[1]]]]
        run = [_CARD_STR[deck[pick[2 + j]]] for j in range(need)] if need > 0 else []
        full = board_str + run
        mh = _phe(*(me + full))
        oh = _phe(*(opp + full))
        wins += 1.0 if mh < oh else (0.5 if mh == oh else 0.0)
    return wins / rollouts


# --------------------------------------------------------------------------- #
# IRC archetypes as numpy-engine policy_fns
# --------------------------------------------------------------------------- #
@dataclass
class IRCArchetype:
    """An IRC-population HU villain expressed on the np_engine. Mirrors ``irc_villains.IRCVillain``
    thresholds; aggression is gated on the equity proxy + expressed at a realistic pot fraction. All
    behavior is the SAME archetype meaning (loose-passive station, lag, nit, tag), restricted to HU."""

    name: str
    pre_fold_hs: float = 0.0
    call_hs: float = 0.30
    raise_hs: float = 0.80
    raise_freq: float = 0.0
    bluff_raise_freq: float = 0.0
    bet_hs: float = 0.75
    bet_freq: float = 0.5
    bluff_bet_freq: float = 0.0
    value_size: float = 0.55
    bluff_size: float = 0.66
    rollouts: int = 80

    def _bucket_for_frac(self, mask: np.ndarray, frac: float) -> int | None:
        """Closest LEGAL bet bucket to ``frac`` of pot; None if no sized bet legal (jam handled by
        caller). Mirrors irc_villains._pick_sized_bet's 'closest legal sizing' choice."""
        best_i, best_d = None, 1e18
        for i, f in enumerate(_BET_FRACS):
            idx = BET_BUCKET_START + i
            if idx < NUM_ACTIONS and mask[idx]:
                d = abs(f - frac)
                if d < best_d:
                    best_d, best_i = d, idx
        return best_i

    def policy_fn(self, rng: np.random.Generator):
        """Return a STOCHASTIC policy_fn(env_np, seat)->probs. The randomness (raise/bet frequencies)
        is drawn from ``rng`` and BAKED into a one-hot probs vector per call (the duplicate harness wants
        a concrete action per decision; CRN is supplied by reseeding ``rng`` per deck)."""
        def policy(env, seat: int) -> np.ndarray:
            mask = legal_action_mask_np(env, seat).astype(bool)
            p = np.zeros(NUM_ACTIONS, dtype=np.float64)
            if not mask.any():
                return p
            facing = bool(mask[CALL])  # CALL legal => there is a bet to call (else CHECK legal)
            preflop = int(env.street) == 0
            hs = _equity_vs_random(env, seat, rng, self.rollouts)

            def one_hot(a: int) -> np.ndarray:
                q = np.zeros(NUM_ACTIONS, dtype=np.float64)
                q[a] = 1.0
                return q

            def jam_or(a_fallback: int) -> int:
                return ALL_IN if mask[ALL_IN] else a_fallback

            if facing:
                if preflop and hs < self.pre_fold_hs and mask[FOLD]:
                    return one_hot(FOLD)
                if hs >= self.raise_hs and rng.random() < self.raise_freq:
                    b = self._bucket_for_frac(mask, self.value_size)
                    if b is not None:
                        return one_hot(b)
                    if mask[ALL_IN]:
                        return one_hot(ALL_IN)
                if hs >= self.call_hs and mask[CALL]:
                    return one_hot(CALL)
                if self.bluff_raise_freq and rng.random() < self.bluff_raise_freq:
                    b = self._bucket_for_frac(mask, self.bluff_size)
                    if b is not None:
                        return one_hot(b)
                return one_hot(FOLD if mask[FOLD] else (CHECK if mask[CHECK] else int(np.argmax(mask))))

            # Unopened (can bet/check).
            if hs >= self.bet_hs and rng.random() < self.bet_freq:
                b = self._bucket_for_frac(mask, self.value_size)
                if b is not None:
                    return one_hot(b)
            if self.bluff_bet_freq and rng.random() < self.bluff_bet_freq:
                b = self._bucket_for_frac(mask, self.bluff_size)
                if b is not None:
                    return one_hot(b)
            return one_hot(CHECK if mask[CHECK] else (FOLD if mask[FOLD] else int(np.argmax(mask))))

        return policy


# IRC HU archetypes -- thresholds mirror irc_villains.IRC{Station,Lag,Nit,Tag}, restricted to HU.
_ARCHETYPES = {
    "irc_station": IRCArchetype("irc_station", pre_fold_hs=0.0, call_hs=0.18, raise_hs=0.88,
                                raise_freq=0.15, bet_hs=0.72, bet_freq=0.45, bluff_bet_freq=0.02,
                                value_size=0.5, bluff_size=0.5),
    "irc_lag": IRCArchetype("irc_lag", pre_fold_hs=0.18, call_hs=0.34, raise_hs=0.70, raise_freq=0.5,
                            bluff_raise_freq=0.18, bet_hs=0.42, bet_freq=0.7, bluff_bet_freq=0.30,
                            value_size=0.75, bluff_size=0.9),
    "irc_nit": IRCArchetype("irc_nit", pre_fold_hs=0.52, call_hs=0.6, raise_hs=0.85, raise_freq=0.35,
                            bet_hs=0.78, bet_freq=0.5, bluff_bet_freq=0.0, value_size=0.5,
                            bluff_size=0.5),
    "irc_tag": IRCArchetype("irc_tag", pre_fold_hs=0.45, call_hs=0.5, raise_hs=0.74, raise_freq=0.5,
                            bluff_raise_freq=0.08, bet_hs=0.58, bet_freq=0.65, bluff_bet_freq=0.12,
                            value_size=0.66, bluff_size=0.75),
}


def list_archetypes() -> list[str]:
    return sorted(_ARCHETYPES)


# --------------------------------------------------------------------------- #
# Seat-rotated duplicate HU cell
# --------------------------------------------------------------------------- #
@dataclass
class HUFieldConfig:
    decks: int = 600
    seed: int = 0
    villain: str = "irc_station"
    stack_bb: float | None = None      # None => engine default (100bb); set e.g. 20 for shallow MTT HU
    ante_bb: float = 0.0
    max_depth: int = 120
    villain_rollouts: int = 80


@dataclass
class HUFieldResult:
    villain: str
    decks: int
    bb_per_100: float
    bb_per_100_ci95: float
    bb_per_hand: float
    stderr_bb_per_hand: float
    hero_hands: int
    note: str = ("Seat-rotated duplicate HU win-rate vs an IRC HU villain. Positive => hero beats the "
                 "field. CURRENT-policy baseline unless the plugged net is the converged blueprint.")
    detail: dict = field(default_factory=dict)


def _play_hu_deck(hero_policy, villain_policy, *, deck_key, hero_seat: int, button: int,
                  stack_bb, ante_bb, max_depth: int, rng: np.random.Generator) -> float:
    """Play ONE HU hand on the np_engine with the given button + hero seat assignment; return the
    hero's reward in BB. The deck (hole cards + board) is a pure function of ``deck_key`` + ``button``,
    so replaying with the hero in the OTHER seat reuses the SAME cards (duplicate)."""
    from pokerbot.engine.np_engine import to_numpy_state

    stacks = (jnp.full(len(_HU_OCC), int(round(stack_bb * CHIP_UNIT)), dtype=jnp.int32)
              if stack_bb is not None else None)
    ante = jnp.int32(int(round(ante_bb * CHIP_UNIT)))
    env = to_numpy_state(init_hand(deck_key, button=jnp.int32(button), stacks=stacks,
                                   occupancy=jnp.asarray(_HU_OCC), ante=ante))
    villain_seat = _HU_LIVE[1] if hero_seat == _HU_LIVE[0] else _HU_LIVE[0]

    depth = 0
    while depth < max_depth and not bool(env.done):
        if is_chance_node_np(env):
            env = sample_chance_np(env)
            continue
        seat = int(env.action_seat)
        if seat < 0:
            # street=STREET_TERMINAL, done=False (checked-down / river-all-in showdown): the engine
            # never settles this class (showdown-leaf-zero bug). The legacy loop queried
            # policy(env, -1) here (warning + CHECK fallback) until the depth cap and returned 0.0;
            # break instead and let the gate-aware terminal read below score it (value-identical 0.0
            # with the gate OFF; the real settle with POKERBOT_SETTLE_SHOWDOWN_LEAF=1).
            break
        policy = hero_policy if seat == hero_seat else villain_policy
        probs = policy(env, seat)
        legal = np.flatnonzero(probs > 0)
        if legal.size == 0:
            legal = np.flatnonzero(legal_action_mask_np(env, seat))
            a = int(rng.choice(legal)) if legal.size else FOLD
        elif legal.size == 1:
            a = int(legal[0])
        else:
            a = int(rng.choice(legal, p=probs[legal] / probs[legal].sum()))
        env = apply_action_np(env, a)
        depth += 1

    if bool(env.done):
        return float(env.rewards[hero_seat]) / CHIP_UNIT
    # Gate-aware terminal read (SETTLE_SHOWDOWN_LEAF, default OFF = 0.0 exactly as before): settles
    # the unsettled-showdown class with trainer terminal_return semantics when the gate is ON.
    return leaf_terminal_reward_np(env, hero_seat) / CHIP_UNIT


def hu_vs_irc_rotated(hero_policy, cfg: HUFieldConfig | None = None) -> HUFieldResult:
    """Seat-rotated duplicate HU win-rate of ``hero_policy`` vs one IRC HU villain.

    ``hero_policy`` is any ``policy_fn(env_np, seat)->probs`` (e.g. ``hu_lbr_neural.make_neural_policy``
    or ``lbr.make_blueprint_policy``). Returns bb/100 + a 95% CI computed over the per-deck seat-means.
    """
    cfg = cfg or HUFieldConfig()
    arch = _ARCHETYPES.get(cfg.villain)
    if arch is None:
        raise ValueError(f"unknown IRC HU villain {cfg.villain!r}; known: {list_archetypes()}")
    arch.rollouts = cfg.villain_rollouts

    key = jax.random.PRNGKey(cfg.seed)
    button = _HU_LIVE[0]  # button fixed on seat 3; hero rotates between seat 3 (button) and seat 5 (BB)
    per_deck = np.zeros(cfg.decks, dtype=np.float64)
    hero_hands = 0

    for d in range(cfg.decks):
        key, k = jax.random.split(key)
        seat_vals = []
        for hero_seat in _HU_LIVE:
            # CRN: villain + tie-break rng reseeded identically for BOTH rotations of this deck, so the
            # only difference between the two plays is WHICH seat the hero occupies (true duplicate).
            v_rng = np.random.default_rng(900_000 + cfg.seed * 1_000_003 + d)
            t_rng = np.random.default_rng(700_000 + cfg.seed * 1_000_003 + d)
            villain_policy = arch.policy_fn(v_rng)
            val = _play_hu_deck(hero_policy, villain_policy, deck_key=k, hero_seat=hero_seat,
                                button=button, stack_bb=cfg.stack_bb, ante_bb=cfg.ante_bb,
                                max_depth=cfg.max_depth, rng=t_rng)
            seat_vals.append(val)
            hero_hands += 1
        per_deck[d] = float(np.mean(seat_vals))

    ev = float(per_deck.mean())
    stderr = float(per_deck.std(ddof=1) / np.sqrt(cfg.decks)) if cfg.decks > 1 else 0.0
    return HUFieldResult(
        villain=cfg.villain,
        decks=cfg.decks,
        bb_per_100=ev * 100.0,
        bb_per_100_ci95=1.96 * stderr * 100.0,
        bb_per_hand=ev,
        stderr_bb_per_hand=stderr,
        hero_hands=hero_hands,
        detail={"stack_bb": cfg.stack_bb, "ante_bb": cfg.ante_bb,
                "villain_rollouts": cfg.villain_rollouts, "button_seat": int(button)},
    )
