"""SD-CFR on NLHE with WEVA-bucket belief-weighted regrets (Phase 2b)."""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from jax import Array

from pokerbot.engine.jax_engine import (
    ALL_IN,
    BET_BUCKET_START,
    BET_SIZE_BUCKETS,
    CALL,
    CHECK,
    CHIP_UNIT,
    FOLD,
    NUM_ACTIONS as NLHE_NUM_ACTIONS,
    STARTING_STACK_BB,
    EnvState,
    init_hand,
    is_chance_node,
    legal_action_mask,
    sample_chance,
    step_state,
)
from pokerbot.engine.jax_weva_encode import (
    NLHEEncodeBundle,
    active_abstraction_config as _active_abstraction_config,
    encode_nlhe_weva_bundle,
)
from pokerbot.engine.jax_weva_fast import (
    FastEncodeCache,
    default_weva_bucket_table,
    encode_nlhe_bundle_fast,
    structural_infoset_key,
)
from pokerbot.engine.nlhe_bridge import live_seat_indices
from pokerbot.encoding.weva_encoder import WEVAMapper
from pokerbot.training.nlhe_policy_net import NLHEWEVAPolicyNet

# Max |opp_reach * (u[a]-u_node)| with opp_reach<=1 over NLHE tree (conservative).
NLHE_MAX_INSTANT_ADV: float = float(STARTING_STACK_BB) * 6.0

# Domain-separation fold for the traversal RNG root ("NLHE" in ASCII). Single source of truth:
# both __init__ and the checkpoint save-side fallback derive the root key with this constant so
# they can never diverge (a divergent fallback would silently persist a different deal stream).
_TRAVERSAL_ROOT_KEY_FOLD: int = 0x4E4C4845

# Pre-built device int32 action scalars: ``step_state`` is jitted on an int32 action arg, so passing
# a fresh ``jnp.int32(a)`` per child re-runs the eager convert-element-type primitive every node.
# Building them once avoids that per-edge dispatch (the action set is tiny and fixed).
_ACTION_CONST: tuple[Array, ...] = tuple(jnp.int32(a) for a in range(NLHE_NUM_ACTIONS))


# --- Runtime collision meter -------------------------------------------------------------------
# Tripwire that confirms the tabular KEY stays injective over infosets at convergence scale: if two
# nodes share a ``fingerprint`` but have a different INDEPENDENT ``collision_witness``, that is a
# genuine representational collision (two distinct infosets merged into one key). With the lossless
# history key it should stay 0; this surfaces any regression loudly instead of silently corrupting
# the blueprint. Process-global (works inside each spawn worker); per-process counts are aggregated
# through ``TraversalShardResult.collisions``. Bounded dict caps memory; gated by env.
_COLLISION_METER_ON = os.environ.get("POKERBOT_COLLISION_METER", "1").lower() not in ("0", "false", "no", "")
_COLLISION_METER_CAP = int(os.environ.get("POKERBOT_COLLISION_METER_CAP", "1500000"))
_collision_witness: dict[bytes, bytes] = {}
_collision_count = 0
_collision_samples: list[str] = []


def _meter_collision(info_key: bytes, bundle: NLHEEncodeBundle) -> None:
    global _collision_count
    prev = _collision_witness.get(info_key)
    wit = bundle.collision_witness()
    if prev is None:
        if len(_collision_witness) < _COLLISION_METER_CAP:
            _collision_witness[info_key] = wit
        return
    if prev != wit:
        _collision_count += 1
        if len(_collision_samples) < 32:
            _collision_samples.append(info_key.hex())
        if _collision_count <= 5:
            print(
                f"!!! TABULAR KEY COLLISION #{_collision_count} fp={info_key.hex()} — distinct "
                "infosets share one key (regrets would silently merge)",
                file=sys.stderr, flush=True,
            )


def collision_meter_report() -> dict[str, Any]:
    """Snapshot of this process's collision meter (count, metered keys, sample fps)."""
    return {
        "enabled": _COLLISION_METER_ON,
        "collisions": _collision_count,
        "metered_keys": len(_collision_witness),
        "cap": _COLLISION_METER_CAP,
        "samples": list(_collision_samples),
    }


def _default_traversal_root_key(seed: int) -> Array:
    return jax.random.fold_in(jax.random.PRNGKey(seed), _TRAVERSAL_ROOT_KEY_FOLD)


def _seed_from_jax_key(key: Array) -> int:
    """Stable, process-independent uint64 seed from a JAX PRNG key.

    Opponent-action sampling in ``traverse`` is a single categorical draw; ``jax.random.choice``
    spends ~1ms of eager-primitive dispatch on it. We seed a host-side ``np.random.Generator`` once
    per traverse root from the JAX unit key instead, so the draws are microsecond-cheap *and* still
    deterministic per (iteration, sweep, trav) — keeping the parallel/single split bit-identical.
    """
    try:
        data = np.asarray(jax.random.key_data(key))
    except Exception:
        data = np.asarray(key)
    raw = data.astype(np.uint32, copy=False).tobytes()[:8].ljust(8, b"\x00")
    return int.from_bytes(raw, "little")


def build_allowed_action_mask(bet_bucket_keep: tuple[int, ...] | None) -> np.ndarray | None:
    """Boolean (NUM_ACTIONS,) of actions the traversal may branch on under action abstraction.

    fold/check/call/all-in are always allowed; only the configured subset of the 7 bet-size buckets
    is. ANDed into the per-node legal mask, this restricts branching to the abstracted bet sizes
    (engine untouched, deterministic ⇒ pool/single split stays bit-identical). None ⇒ no abstraction.
    """
    if bet_bucket_keep is None:
        return None
    allowed = np.zeros(NLHE_NUM_ACTIONS, dtype=bool)
    for a in (FOLD, CHECK, CALL, ALL_IN):
        allowed[a] = True
    for i in bet_bucket_keep:
        if not 0 <= int(i) < len(BET_SIZE_BUCKETS):
            raise ValueError(f"bet_bucket_keep index {i} out of range [0,{len(BET_SIZE_BUCKETS)})")
        allowed[BET_BUCKET_START + int(i)] = True
    return allowed


@dataclass
class _NLHERegretVisit:
    opp_reach: float
    util: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray
    bundle: NLHEEncodeBundle
    opponent_weva_bucket: int


@dataclass
class SDCFRNLHEConfig:
    iterations: int = 10
    traversals_per_iter: int = 8
    lr: float = 1e-4
    seed: int = 0
    # Solver-label HU uses BTN/BB at seats 3/5 (6-max layout), not JAX 2-seat 0/1.
    occupancy: tuple[bool, ...] = (False, False, False, True, False, True)
    button_seat: int = 3
    buffer_capacity: int = 50_000
    train_batch_size: int = 2048
    deal_sweeps: int = 1
    max_traverse_depth: int = 64
    sft_checkpoint: str | None = "checkpoints/policy_sft_prod6000.pt"
    policy_hidden: int = 128
    # Action (bet-size) ABSTRACTION — Pluribus-style. Tuple of indices into BET_SIZE_BUCKETS
    # (0.33,0.50,0.66,0.75,1.0,1.5,2.0) that the traversal is allowed to branch on; fold/check/call/
    # all-in are always kept. None = keep ALL 7 (exact, current behaviour). Restricting to a few
    # sizes collapses the betting tree (~9^d -> ~5^d sequences), bounding the tabular regret tables so
    # a full-depth single-machine convergence is feasible. e.g. (1, 4) = half-pot + pot.
    bet_bucket_keep: tuple[int, ...] | None = None
    # Client-matched (CoinPoker) bet sizing: STREET-AWARE action abstraction. Preflop the engine's
    # pot-fraction buckets 4/5/6 land on ~2x/2.75x/3.5x the BB (small fractions are sub-min-raise →
    # illegal preflop); postflop buckets 0/1/3 are exactly 1/3, 1/2, 3/4 pot — the client's
    # {2,2.5,3,4}X preflop + {33,50,75,Max}% postflop button presets. True ⇒ overrides bet_bucket_keep
    # with preflop={4,5,6}, postflop={0,1,3} (+fold/check/call/all-in). See coinpoker-bet-sizing memory.
    client_bet_sizing: bool = False


class SDCFRNLHETrainer:
    """External-sampling SD-CFR on JAX NLHE with WEVA infoset encoding."""

    def __init__(self, config: SDCFRNLHEConfig | None = None) -> None:
        self.config = config or SDCFRNLHEConfig()
        self.key = jax.random.PRNGKey(self.config.seed)
        self.occupancy = np.asarray(self.config.occupancy, dtype=bool)
        self._live_seats = live_seat_indices(self.occupancy)
        self._n_live = len(self._live_seats)
        self._mapper = WEVAMapper.default()
        self._weva_table = default_weva_bucket_table()
        self._encode_cache = FastEncodeCache(self._weva_table)
        self._use_fast_encode = True
        # Precomputed action-abstraction mask (None ⇒ exact, all bet sizes).
        self._allowed_actions = build_allowed_action_mask(self.config.bet_bucket_keep)
        # Client (CoinPoker) STREET-AWARE bet sizing: preflop = X-mult-like buckets {4,5,6}=2x/2.75x/
        # 3.5x; postflop = %pot buckets {0,1,3}=1/3,1/2,3/4. Overrides bet_bucket_keep when on.
        self._client_sizing = bool(getattr(self.config, "client_bet_sizing", False))
        # Preflop = X-mult-like {4,5,6}=2x/2.75x/3.5x (+all-in for bigger). Postflop spans small→overbet:
        # {0,1,3,4,5}=1/3, 1/2, 3/4, POT(100%), 1.5x-OVERBET (+all-in) — crucial >75% sizes (pot,
        # overbet, jam) for polarized value/bluffs, not just a 75%→all-in cliff.
        self._allowed_pre = build_allowed_action_mask((4, 5, 6)) if self._client_sizing else None
        self._allowed_post = build_allowed_action_mask((0, 1, 3, 4, 5)) if self._client_sizing else None
        # Iteration-frozen regret snapshot for sigma (set in _begin_iteration_regret); None ⇒ live.
        self._frozen_regret: dict[tuple, np.ndarray] | None = None
        sft_path = Path(self.config.sft_checkpoint) if self.config.sft_checkpoint else None
        if sft_path is not None and sft_path.exists():
            from pokerbot.training.nlhe_sft_init import build_sft_aligned_params

            self._net, variables = build_sft_aligned_params(sft_path, seed=self.config.seed)
            print(f"Initialized NLHE policy from SFT: {sft_path}", flush=True)
        else:
            self._net = NLHEWEVAPolicyNet(hidden=self.config.policy_hidden)
            self.key, k = jax.random.split(self.key)
            dummy = {
                "weva_ids": jnp.zeros((1,), dtype=jnp.int32),
                "cards": jnp.zeros((1, 9, 4, 13), dtype=jnp.float32),
                "actions": jnp.zeros((1, 6, 4, 12), dtype=jnp.float32),
                "position": jnp.zeros((1, self._net.position_dim), dtype=jnp.float32),
                "legal_mask": jnp.ones((1, NLHE_NUM_ACTIONS), dtype=bool),
            }
            variables = self._net.init(k, **dummy)
        self._batch_stats = variables["batch_stats"]

        def _apply(variables_dict: dict, **inputs: Array) -> Array:
            return self._net.apply(variables_dict, **inputs)

        self.state = train_state.TrainState.create(
            apply_fn=_apply,
            params=variables["params"],
            tx=optax.adam(self.config.lr),
        )
        self._cum_regret: dict[tuple, np.ndarray] = {}
        self._strat_sum: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64)
        )
        self._reservoir: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        self._reservoir_insert_count = 0
        self._iter_weight: float = 1.0
        self._avg_policy_cache: dict[tuple, np.ndarray] = {}
        self._completed_iterations = 0
        self._traversal_root_key = _default_traversal_root_key(self.config.seed)
        self._bundle_cache: dict[tuple, NLHEEncodeBundle] = {}
        self._encode_stats: dict[str, int] = {
            "encode_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def clear_policy_cache(self) -> None:
        self._avg_policy_cache.clear()

    def reset_encode_stats(self) -> None:
        self._encode_stats = {"encode_calls": 0, "cache_hits": 0, "cache_misses": 0}

    def _bundle(self, state: EnvState, seat: int, **kwargs: Any) -> NLHEEncodeBundle:
        if self._use_fast_encode and not kwargs:
            return encode_nlhe_bundle_fast(state, seat, weva_table=self._weva_table)
        return encode_nlhe_weva_bundle(state, seat, self._mapper, **kwargs)

    def _bundle_cached(self, state: EnvState, seat: int, **kwargs: Any) -> NLHEEncodeBundle:
        """Per-iteration memo on structural infoset key; skips duplicate encodes on same node."""
        if kwargs:
            self._encode_stats["encode_calls"] += 1
            return self._bundle(state, seat, **kwargs)
        key = structural_infoset_key(state, seat)
        hit = self._bundle_cache.get(key)
        if hit is not None:
            self._encode_stats["cache_hits"] += 1
            return hit
        self._encode_stats["cache_misses"] += 1
        self._encode_stats["encode_calls"] += 1
        bundle = self._bundle(state, seat)
        self._bundle_cache[key] = bundle
        return bundle

    def _opponent_weva_bucket_fast(self, state: EnvState, opponent_seat: int) -> int:
        holes = np.asarray(state.holes)
        button = int(np.asarray(state.button))
        h0, h1 = int(holes[opponent_seat, 0]), int(holes[opponent_seat, 1])
        from pokerbot.engine.jax_weva_fast import _POS_BY_OFFSET

        pos_enum = int(_POS_BY_OFFSET[(opponent_seat - button) % 6])
        return self._encode_cache.weva_bucket(h0, h1, pos_enum)

    def _encode_key(self, state: EnvState, seat: int) -> tuple:
        return self._bundle_cached(state, seat).fingerprint()

    def _policy_variables(self) -> dict:
        return {"params": self.state.params, "batch_stats": self._batch_stats}

    def _bundle_policy_inputs(self, bundle: NLHEEncodeBundle, mask: np.ndarray) -> dict[str, Array]:
        # NCHW layout to match Phase 1 PseudoSiamesePolicy (net transposes to NHWC).
        return {
            "weva_ids": jnp.asarray([bundle.weva_id], dtype=jnp.int32),
            "cards": jnp.asarray(bundle.cards, dtype=jnp.float32).reshape(1, 9, 4, 13),
            "actions": jnp.asarray(bundle.actions, dtype=jnp.float32).reshape(1, 6, 4, 12),
            "position": jnp.asarray(bundle.policy_position, dtype=jnp.float32)[None, :],
            "legal_mask": jnp.asarray(mask, dtype=bool)[None, :],
        }

    def network_strategy_policy(self, state: EnvState, seat: int) -> np.ndarray:
        """Iterate policy: softmax logits from the advantage (policy) net."""
        mask = np.asarray(legal_action_mask(state, jnp.int32(seat)), dtype=bool)
        bundle = self._bundle(state, seat)
        logits = self.state.apply_fn(self._policy_variables(), **self._bundle_policy_inputs(bundle, mask))
        probs = np.asarray(jax.nn.softmax(logits[0]))
        probs = probs * mask
        tot = probs.sum()
        if tot > 0:
            return probs / tot
        return mask.astype(np.float64) / max(mask.sum(), 1)

    @staticmethod
    def _sigma_from_regrets(reg: np.ndarray, mask: np.ndarray) -> np.ndarray:
        positive = np.maximum(reg, 0.0) * mask
        total = positive.sum()
        if total > 0:
            return positive / total
        return mask.astype(np.float64) / max(mask.sum(), 1)

    def _regret_matching_sigma(
        self, state: EnvState, seat: int, *, info_key: tuple | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if mask is None:
            mask = np.asarray(legal_action_mask(state, jnp.int32(seat)), dtype=bool)
        key = info_key if info_key is not None else self._encode_key(state, seat)
        # Read the strategy from the iteration-START regret snapshot, not the live table. Regret
        # deltas accumulate into ``_cum_regret`` AS we traverse, so reading it live would make sigma
        # depend on how many units already ran — order-dependent, and NOT bit-identical between the
        # single-threaded path and the sharded pool (each worker sees only its own units). Freezing
        # the strategy per iteration is the standard parallelizable CFR formulation and makes the
        # shard merge exactly order-independent. (Latent before; action abstraction's smaller tree
        # revisits nodes within an iteration and exposed it.)
        table = self._frozen_regret if self._frozen_regret is not None else self._cum_regret
        reg = table.get(key)
        if reg is None:
            return mask.astype(np.float64) / max(mask.sum(), 1)
        return self._sigma_from_regrets(reg, mask)

    def _reach_for_seat(self, seat: int, p_by_seat: dict[int, float]) -> float:
        return float(p_by_seat.get(seat, 1.0))

    def _opponent_reach(self, traverser: int, p_by_seat: dict[int, float]) -> float:
        opp = 1.0
        for s in self._live_seats:
            if s != traverser:
                opp *= self._reach_for_seat(s, p_by_seat)
        return opp

    def _begin_iteration_regret(self) -> None:
        self._pending_regret: dict[tuple, list[_NLHERegretVisit]] = defaultdict(list)
        self._bundle_cache.clear()
        self.reset_encode_stats()
        # Freeze the regret-matching strategy for this whole iteration (read by sigma). Copy so the
        # live ``_cum_regret`` accumulation below doesn't mutate it. Cheap under action abstraction
        # (small tables); for the pool each worker freezes its loaded base ⇒ all units, all workers,
        # and the single-threaded reference compute identical sigma ⇒ bit-identical merge.
        self._frozen_regret = {k: v.copy() for k, v in self._cum_regret.items()}

    def _flush_pending_regret(self) -> None:
        for info_key, visits in self._pending_regret.items():
            if not visits:
                continue
            reg_inc = np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64)
            total_opp_reach = 0.0
            for visit in visits:
                mask = visit.mask
                node_util = float(
                    sum(visit.sigma[a] * visit.util[a] for a in range(NLHE_NUM_ACTIONS) if mask[a])
                )
                for a in range(NLHE_NUM_ACTIONS):
                    if mask[a]:
                        reg_inc[a] += visit.opp_reach * (visit.util[a] - node_util)
                total_opp_reach += visit.opp_reach
            last = visits[-1]
            if not hasattr(self, "_iter_samples"):
                self._iter_samples = []
            scale = max(total_opp_reach, 1e-12)
            inst_net = (reg_inc / scale).astype(np.float32)
            self._iter_samples.append((last.bundle, inst_net, last.mask.copy()))
        self._pending_regret.clear()

    def traverse(
        self,
        state: EnvState,
        traverser: int,
        key: Array,
        p_by_seat: dict[int, float] | None = None,
        depth: int = 0,
    ) -> float:
        if depth >= self.config.max_traverse_depth or bool(state.done):
            return float(np.asarray(state.rewards[traverser], dtype=np.float64) / CHIP_UNIT)
        # Opponent-action sampling uses a host-side numpy Generator (microsecond categorical draws)
        # instead of jax.random.choice (~1ms eager dispatch/node). Coerce once at the traverse root;
        # recursive calls already receive the Generator, so the isinstance check short-circuits.
        if not isinstance(key, np.random.Generator):
            key = np.random.default_rng(_seed_from_jax_key(key))
        if bool(is_chance_node(state)):
            state = sample_chance(state)
        seat = int(state.action_seat)
        if seat < 0:
            return 0.0
        if p_by_seat is None:
            p_by_seat = {s: 1.0 for s in self._live_seats}
        mask = np.asarray(legal_action_mask(state), dtype=bool)
        if self._client_sizing:
            # Client STREET-AWARE sizing: preflop X-mult buckets, postflop %pot buckets.
            pre = int(np.asarray(state.street)) == 0
            mask = mask & (self._allowed_pre if pre else self._allowed_post)
        elif self._allowed_actions is not None:
            # Action abstraction: branch only on the configured bet sizes (fold/check/call/all-in
            # always kept). ``& allowed`` can't empty a decision node — at least one of check/call or
            # fold is always legal, and all-in is kept as the aggressive option.
            mask = mask & self._allowed_actions
        opp_reach = self._opponent_reach(traverser, p_by_seat)
        bundle = self._bundle_cached(state, seat)
        info_key = bundle.fingerprint()
        if _COLLISION_METER_ON:
            _meter_collision(info_key, bundle)
        sigma = self._regret_matching_sigma(state, seat, info_key=info_key, mask=mask)
        acting_reach = self._reach_for_seat(seat, p_by_seat)
        self._strat_sum[info_key] += self._iter_weight * acting_reach * sigma

        if seat == traverser:
            util = np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64)
            for a in range(NLHE_NUM_ACTIONS):
                if mask[a]:
                    util[a] = self.traverse(
                        step_state(state, _ACTION_CONST[a]),
                        traverser,
                        key,
                        p_by_seat,
                        depth + 1,
                    )
            node_util = float(sum(sigma[a] * util[a] for a in range(NLHE_NUM_ACTIONS) if mask[a]))
            inst = np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64)
            for a in range(NLHE_NUM_ACTIONS):
                if mask[a]:
                    inst[a] = opp_reach * (util[a] - node_util)
            if info_key not in self._cum_regret:
                self._cum_regret[info_key] = np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64)
            self._cum_regret[info_key] += inst
            opp_bucket = -1
            opps = [s for s in self._live_seats if s != seat]
            if len(opps) == 1:
                opp_bucket = self._opponent_weva_bucket_fast(state, opps[0])
            self._pending_regret[info_key].append(
                _NLHERegretVisit(
                    opp_reach=float(opp_reach),
                    util=util.copy(),
                    sigma=sigma.copy(),
                    mask=mask.copy(),
                    bundle=bundle,
                    opponent_weva_bucket=opp_bucket,
                )
            )
            return node_util

        legal_idx = np.flatnonzero(mask)
        probs = sigma[legal_idx]
        probs = probs / max(probs.sum(), 1e-12)
        cdf = np.cumsum(probs)
        r = float(key.random()) * float(cdf[-1])
        pick = int(np.searchsorted(cdf, r, side="right"))
        if pick >= legal_idx.shape[0]:
            pick = legal_idx.shape[0] - 1
        action = int(legal_idx[pick])
        np_by = dict(p_by_seat)
        np_by[seat] = np_by.get(seat, 1.0) * sigma[action]
        return self.traverse(
            step_state(state, _ACTION_CONST[action]),
            traverser,
            key,
            np_by,
            depth + 1,
        )

    def _reservoir_add(self, bundle: NLHEEncodeBundle, inst: np.ndarray, mask: np.ndarray, action: int) -> None:
        sample = (bundle, inst, mask, action)
        cap = self.config.buffer_capacity
        self._reservoir_insert_count += 1
        if len(self._reservoir) < cap:
            self._reservoir.append(sample)
            return
        j = int(np.random.randint(0, self._reservoir_insert_count))
        if j < cap:
            self._reservoir[j] = sample

    def _log_target_scale(self, samples: list) -> None:
        if not samples:
            return
        abs_vals = [
            abs(float(inst[a]))
            for _bundle, inst, mask in samples
            for a in range(NLHE_NUM_ACTIONS)
            if mask[a]
        ]
        mx = max(abs_vals) if abs_vals else 0.0
        print(f"  target_scale max_abs={mx:.4f} (NLHE bound {NLHE_MAX_INSTANT_ADV:.1f})", flush=True)
        if mx > NLHE_MAX_INSTANT_ADV:
            raise AssertionError(
                f"instantaneous advantage {mx:.4f} exceeds NLHE bound {NLHE_MAX_INSTANT_ADV}"
            )

    def train_iteration(
        self,
        iteration_weight: float = 1.0,
        *,
        traversal_logger: Any = None,
    ) -> float:
        iteration_num = int(iteration_weight)
        from pokerbot.training.sd_cfr_nlhe_distributed import run_traversal_phase_single_threaded

        # 1.1/1.4 fix (CFR-AUDIT): wire ``self._iter_weight = iteration_weight`` DEFENSIVELY in the
        # parent train_iteration. The distributed single-thread path (``run_traversal_phase_single_
        # threaded``) ALSO sets it on line 310 of sd_cfr_nlhe_distributed.py, so the production
        # path is already correct; this assignment here is the backstop for any caller that uses
        # ``train_iteration`` directly without going through the distributed path (a unit test,
        # a future experimental harness, etc.). The strategy-sum update on line 432 multiplies
        # by ``self._iter_weight``; without this assignment it would silently stay at the
        # ``__init__`` value 1.0 (legacy uniform averaging) instead of the CFR+ linear weight
        # ``t``. Cheap to do; closes a silent regression.
        self._iter_weight = float(iteration_weight)
        self._iter_samples = []
        run_traversal_phase_single_threaded(
            self,
            iteration_num,
            iteration_weight,
            logger=traversal_logger,
        )
        if traversal_logger is not None:
            traversal_logger.emit("net_train_start", iteration=iteration_num)
        loss = self._finish_iteration_after_traversal()
        if traversal_logger is not None:
            traversal_logger.emit("net_train_done", iteration=iteration_num, loss=float(loss))
        return loss

    def _finish_iteration_after_traversal(self) -> float:
        self._flush_pending_regret()
        if not self._iter_samples:
            return 0.0
        self._log_target_scale(self._iter_samples)
        for bundle, inst, mask in self._iter_samples:
            legal = np.flatnonzero(mask)
            a = int(legal[0]) if legal.size else 0
            self._reservoir_add(bundle, inst, mask, a)
        batch_n = min(self.config.train_batch_size, len(self._iter_samples))
        idx = np.random.randint(0, len(self._iter_samples), size=batch_n)
        batch = [self._iter_samples[i] for i in idx]
        weva = jnp.stack([jnp.int32(b[0].weva_id) for b in batch])
        cards = jnp.stack(
            [jnp.asarray(b[0].cards, dtype=jnp.float32).reshape(9, 4, 13) for b in batch]
        )
        actions = jnp.stack(
            [jnp.asarray(b[0].actions, dtype=jnp.float32).reshape(6, 4, 12) for b in batch]
        )
        positions = jnp.stack([jnp.asarray(b[0].policy_position, dtype=jnp.float32) for b in batch])
        masks = jnp.stack([jnp.asarray(b[2]) for b in batch])
        targets = jnp.stack([jnp.asarray(b[1], dtype=jnp.float32) for b in batch])

        def loss_fn(params):
            variables = {"params": params, "batch_stats": self._batch_stats}
            logits = self.state.apply_fn(
                variables,
                weva_ids=weva,
                cards=cards,
                actions=actions,
                position=positions,
                legal_mask=masks,
            )
            diff = (logits - targets) * masks
            per_sample = jnp.sum(jnp.square(diff), axis=-1) / jnp.maximum(
                jnp.sum(masks, axis=-1), 1.0
            )
            return jnp.mean(per_sample)

        loss, grads = jax.value_and_grad(loss_fn)(self.state.params)
        self.state = self.state.apply_gradients(grads=grads)
        return float(loss)

    def train(
        self,
        log_every: int = 1,
        checkpoint_dir: Path | str | None = None,
        checkpoint_every: int = 0,
        start_iter: int = 0,
    ) -> list[float]:
        losses: list[float] = []
        # Guard: never ship a new blueprint on the made-hand-only postflop abstraction (it ignores
        # draw equity). FAIL-CLOSED — raises unless POKERBOT_ALLOW_MADE_HAND_BUCKETS=1 (smoke runs).
        from pokerbot.engine.jax_weva_encode import require_potential_aware_buckets
        require_potential_aware_buckets(context="SD-CFR NLHE blueprint training")
        t0 = time.perf_counter()
        for i in range(start_iter, self.config.iterations):
            iter_num = i + 1
            losses.append(self.train_iteration(iteration_weight=float(iter_num)))
            if log_every > 0 and iter_num % log_every == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"SD-CFR NLHE iter {iter_num}/{self.config.iterations} "
                    f"loss={losses[-1]:.6f} elapsed={elapsed:.1f}s",
                    flush=True,
                )
            if (
                checkpoint_dir is not None
                and checkpoint_every > 0
                and iter_num % checkpoint_every == 0
            ):
                self._completed_iterations = iter_num
                ckpt = Path(checkpoint_dir)
                from pokerbot.training.sd_cfr import save_nlhe_checkpoint

                save_nlhe_checkpoint(self, ckpt / f"sd_cfr_nlhe_iter_{iter_num:04d}.pkl")
                save_nlhe_checkpoint(self, ckpt / "sd_cfr_nlhe_latest.pkl")
        return losses

    def average_strategy_policy(self, state: EnvState, seat: int) -> np.ndarray:
        """CFR average strategy when ``_strat_sum`` has mass; else SFT/network iterate."""
        cache_key = (self._encode_key(state, seat), int(seat))
        cached = self._avg_policy_cache.get(cache_key)
        if cached is not None:
            return cached
        mask = np.asarray(legal_action_mask(state, jnp.int32(seat)), dtype=bool)
        key = cache_key[0]
        s = self._strat_sum.get(key)
        if s is not None:
            sigma = np.asarray(s, dtype=np.float64) * mask
            tot = sigma.sum()
            if tot > 1e-12:
                sigma = sigma / tot
                self._avg_policy_cache[cache_key] = sigma
                return sigma
        return self.network_strategy_policy(state, seat)

    def eval_strategy_policy(self, state: EnvState, seat: int) -> np.ndarray:
        """Gate A / diagnostics: average strategy if trained, else network (SFT init at iter 0)."""
        return self.average_strategy_policy(state, seat)

    def network_strategy_policy_from_bundle(
        self, bundle: NLHEEncodeBundle, mask: np.ndarray
    ) -> np.ndarray:
        logits = self.state.apply_fn(
            self._policy_variables(),
            **self._bundle_policy_inputs(bundle, mask),
        )
        probs = np.asarray(jax.nn.softmax(logits[0]))
        probs = probs * mask
        tot = probs.sum()
        if tot > 0:
            return probs / tot
        return mask.astype(np.float64) / max(mask.sum(), 1)

    def eval_strategy_policy_from_parquet(self, rec: dict[str, Any]) -> np.ndarray:
        """Gate A on held-out rows: JAX replay + fast encode (same path as training)."""
        from pokerbot.engine.jax_weva_fast import encode_nlhe_bundle_fast
        from pokerbot.training.nlhe_spot_replay import (
            _spot_kwargs_from_record,
            jax_state_from_parquet_record,
        )

        state, hero_seat = jax_state_from_parquet_record(rec)
        bundle = encode_nlhe_bundle_fast(
            state,
            hero_seat,
            weva_table=self._weva_table,
            **_spot_kwargs_from_record(rec),
        )
        mask = np.asarray(rec["legal_mask"], dtype=bool)
        key = bundle.fingerprint()
        s = self._strat_sum.get(key)
        if s is not None:
            sigma = np.asarray(s, dtype=np.float64) * mask
            tot = sigma.sum()
            if tot > 1e-12:
                return sigma / tot
        return self.network_strategy_policy_from_bundle(bundle, mask)

    def assert_profile_ev_in_range(self, samples: int = 8) -> tuple[float, float]:
        """Zero-sum sanity: sum of seat EVs ~ 0."""
        occ = jnp.asarray(self.occupancy, dtype=bool)
        evs = []
        for i in range(samples):
            self.key, k = jax.random.split(self.key)
            state = init_hand(k, occupancy=occ)
            for _ in range(120):
                if bool(state.done):
                    break
                while bool(is_chance_node(state)):
                    state = sample_chance(state)
                if bool(state.done):
                    break
                mask = np.asarray(legal_action_mask(state), dtype=bool)
                legal = np.flatnonzero(mask)
                if legal.size == 0:
                    break
                self.key, sub = jax.random.split(self.key)
                a = int(legal[int(jax.random.randint(sub, (), 0, legal.size))])
                state = step_state(state, jnp.int32(a))
            if bool(state.done):
                r = np.asarray(state.rewards, dtype=np.float64) / CHIP_UNIT
                evs.append(float(r[self._live_seats].sum()))
        mx = max(abs(e) for e in evs) if evs else 0.0
        if mx > 0.5:
            raise AssertionError(f"NLHE profile EV not zero-sum: max_abs_sum={mx}")
        return (min(evs) if evs else 0.0, max(evs) if evs else 0.0)


def _iter_num_from_path(p: Path) -> int:
    """Extract the iteration number from a checkpoint path like ``sd_cfr_nlhe_iter_0042.pkl``.
    Returns -1 for files that don't match (e.g. ``latest.pkl``) so they sort last."""
    import re
    m = re.search(r"iter_(\d+)", p.stem)
    return int(m.group(1)) if m else -1


def save_nlhe_checkpoint(trainer: SDCFRNLHETrainer, path: Path, *, worker_base: bool = False) -> None:
    """Pickle the trainer state. ``worker_base=True`` writes a SLIM base for pool workers.

    During traversal a worker only READS ``cum_regret`` (the regret-matching strategy source); it
    only WRITES ``strat_sum`` (the average-strategy accumulator) and the reservoir, and the merge sums
    each worker's delta additively. So a worker that starts from an EMPTY ``strat_sum``/reservoir
    produces the identical merge delta as one started from the full base (bit-identical), while the
    per-iteration base it must deserialize shrinks by ~the strat_sum fraction (~75% at depth 16). The
    durable S3 resume checkpoint always uses the FULL save (worker_base=False).
    """
    import pickle

    # Universal ship-time safety net: refuse to persist ANY NLHE blueprint artifact (full base OR slim
    # worker base — both are made-hand-only tables) keyed with the draw-equity-blind 'fast' bucketer,
    # unless POKERBOT_ALLOW_MADE_HAND_BUCKETS=1 acknowledges a deliberate made-hand run.
    from pokerbot.engine.jax_weva_encode import require_potential_aware_buckets
    require_potential_aware_buckets(context="saving NLHE blueprint checkpoint")

    path.parent.mkdir(parents=True, exist_ok=True)
    # CRASH-RESUME-AUDIT #4: wrap the pickle in tmp + fsync + os.replace + dir-fsync. The 2026-06-23
    # power-cut that zeroed seven_v1_full.pkl is the documented precedent — without atomicity, a kill -9
    # mid-pickle truncates the file and resume picks up a torn pickle that fails to load → forced to
    # re-start from iter 0. The fix follows the reference pattern in pokerbot/training/neural_cfr.py
    # (_atomic_save_trainer_checkpoint) and player_profiles.save. Same code path for all callers; the
    # `worker_base` flag only affects the payload (above), not the durability wrapper.
    import os
    tmp = path.with_name(path.name + ".atomic.tmp")
    with tmp.open("wb") as f:
        pickle.dump(
            {
                "trainer_type": "nlhe",
                "params": trainer.state.params,
                "opt_state": trainer.state.opt_state,
                "batch_stats": getattr(trainer, "_batch_stats", {}),
                "config": trainer.config,
                # Provenance: the abstraction flags this table was KEYED with. Serve cross-checks this
                # via lbr.check_blueprint_serve_env(expected=...) so a flag skew (e.g. PREFLOP_LOSSLESS
                # off at serve) is caught loudly instead of silently missing every key.
                "abstraction": _active_abstraction_config(),
                "completed_iterations": getattr(trainer, "_completed_iterations", 0),
                "cum_regret": getattr(trainer, "_cum_regret", {}),
                "strat_sum": {} if worker_base else dict(getattr(trainer, "_strat_sum", {})),
                "reservoir": [] if worker_base else getattr(trainer, "_reservoir", []),
                "reservoir_insert_count": 0 if worker_base else getattr(trainer, "_reservoir_insert_count", 0),
                "traversal_root_key": np.asarray(
                    getattr(
                        trainer,
                        "_traversal_root_key",
                        _default_traversal_root_key(trainer.config.seed),
                    )
                ),
            },
            f,
        )
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # Some filesystems (rare) refuse fsync — best-effort, since the tmp→replace still
            # gives atomicity on platforms that DON'T journal the rename away.
            pass
    os.replace(tmp, path)
    # CRASH-RESUME-AUDIT cross-cutting A: parent-dir fsync after the rename so the directory
    # entry change survives a power cut. NTFS is journaled so this is mostly belt-and-braces;
    # POSIX without this loses the rename from the inode table on power cut.
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def load_nlhe_checkpoint(trainer: SDCFRNLHETrainer, path: Path) -> int:
    """Restore trainer state from a checkpoint pickle; returns completed iteration count.

    CRASH-RESUME-AUDIT cross-cutting E: torn-pickle recovery. The original code raised on a
    torn file and forced the caller to re-run from iter 0. Now: if ``path`` is unreadable as
    a pickle, scan the parent directory for sibling ``iter_NNNN.pkl`` files (written every
    ``checkpoint_every`` iters during training) and try them in descending order. This means
    a torn ``latest.pkl`` falls back to the previous good iter — losing at most
    ``checkpoint_every`` iterations of compute instead of the whole run.

    The atomic-write side (save_nlhe_checkpoint, CRASH-RESUME-AUDIT #4) means a torn file
    is now rare, but this wrapper is the safety net for the older checkpoints already on disk.
    """
    import logging
    import pickle

    target = Path(path)
    try:
        with target.open("rb") as f:
            data = pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError, ValueError, ImportError) as primary_exc:
        # Try the sibling ``iter_NNNN.pkl`` files in the same dir, highest first.
        parent = target.parent
        try:
            siblings = sorted(
                parent.glob("sd_cfr_nlhe_iter_*.pkl"),
                key=lambda p: _iter_num_from_path(p),
                reverse=True,
            )
        except OSError:
            siblings = []
        for sib in siblings:
            try:
                with sib.open("rb") as f:
                    data = pickle.load(f)
                logging.getLogger(__name__).warning(
                    "load_nlhe_checkpoint: primary %s torn (%s), recovered from %s",
                    target, primary_exc, sib,
                )
                break
            except (OSError, pickle.UnpicklingError, EOFError, ValueError, ImportError):
                continue
        else:
            # Nothing recoverable — re-raise the original failure so the caller sees the real cause.
            raise
    replace_kw: dict = {"params": data["params"]}
    if "opt_state" in data and data["opt_state"] is not None:
        replace_kw["opt_state"] = data["opt_state"]
    trainer.state = trainer.state.replace(**replace_kw)
    if "batch_stats" in data:
        trainer._batch_stats = data["batch_stats"]
    if "config" in data:
        trainer.config = data["config"]
    trainer._cum_regret = dict(data.get("cum_regret", {}))
    if "strat_sum" in data:
        trainer._strat_sum = defaultdict(
            lambda: np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64),
            {k: np.asarray(v) for k, v in data["strat_sum"].items()},
        )
    if "reservoir" in data:
        trainer._reservoir = list(data["reservoir"])
    if "reservoir_insert_count" in data:
        trainer._reservoir_insert_count = int(data["reservoir_insert_count"])
    if "traversal_root_key" in data:
        trainer._traversal_root_key = jnp.asarray(data["traversal_root_key"])
    # Set on the trainer (not just returned) so a resume loop cannot silently forget to assign it
    # and re-run already-completed iterations. The return value is kept for back-compat callers.
    completed = int(data.get("completed_iterations", 0))
    trainer._completed_iterations = completed
    return completed
