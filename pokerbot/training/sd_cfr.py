"""Single Deep CFR (SD-CFR) — advantage network + external sampling."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from jax import Array

from pokerbot.engine.leduc_bridge import (
    PyLeducState,
    apply_action_py,
    init_py_from_key,
    jax_from_py,
    py_legal,
    py_state_from_deal,
)
from pokerbot.engine.leduc_jax import (
    ANTE,
    MAX_HISTORY,
    MAX_RAISES,
    NUM_ACTIONS,
    NUM_PLAYERS,
    RAISE_SIZE,
    LeducState,
    legal_actions,
)

LEDUC_ENCODE_DIM = 7 + MAX_HISTORY  # card, pub, pot, seat, round, to_call, raises + history


def _leduc_max_instant_adv() -> float:
    from pokerbot.training.leduc_tabular_cfr import derive_leduc_max_instant_cf_bound

    return derive_leduc_max_instant_cf_bound()


# opp_reach <= 1; |util[a]-node_util| <= max action-value spread on tree (see derive_*).
LEDUC_MAX_INSTANT_ADV = _leduc_max_instant_adv()


class LeducAdvantageNet(nn.Module):
    hidden: int = 64

    @nn.compact
    def __call__(self, x: Array, legal_mask: Array) -> Array:
        h = nn.relu(nn.Dense(self.hidden)(x))
        h = nn.relu(nn.Dense(self.hidden)(h))
        logits = nn.Dense(NUM_ACTIONS)(h)
        return jnp.where(legal_mask, logits, -1e9)


def encode_leduc(state: LeducState, seat: int) -> Array:
    """History-aware Leduc infoset encoding — betting history is part of the information set."""
    card = state.cards[seat].astype(jnp.float32) / 3.0
    pub = jnp.maximum(state.public, 0).astype(jnp.float32) / 3.0
    pot = state.pot.astype(jnp.float32) / 100.0
    seat_f = jnp.float32(seat)
    rnd = state.round.astype(jnp.float32)
    to_call = jnp.maximum(0, state.current_bet - state.committed[seat]).astype(jnp.float32)
    to_call = to_call / float(RAISE_SIZE * MAX_RAISES + ANTE)
    raises_n = state.raises.astype(jnp.float32) / float(MAX_RAISES)
    base = jnp.array([card, pub, pot, seat_f, rnd, to_call, raises_n], dtype=jnp.float32)
    slots = []
    for i in range(MAX_HISTORY):
        a = jnp.where(i < state.hist_len, state.hist_action[i], jnp.int32(-1))
        slots.append(jnp.where(a >= 0, a.astype(jnp.float32) / 2.0, jnp.float32(0.0)))
    return jnp.concatenate([base, jnp.stack(slots)])


def encode_leduc_py(state: PyLeducState, seat: int) -> Array:
    card = float(state.cards[seat]) / 3.0
    pub = float(max(state.public, 0)) / 3.0
    pot = float(state.pot) / 100.0
    seat_f = float(seat)
    rnd = float(state.round)
    to_call = float(max(0, state.current_bet - state.committed[seat]))
    to_call /= float(RAISE_SIZE * MAX_RAISES + ANTE)
    raises_n = float(state.raises) / float(MAX_RAISES)
    hist = list(state.hist) + [0.0] * (MAX_HISTORY - len(state.hist))
    for i, a in enumerate(state.hist):
        hist[i] = float(a) / 2.0
    return jnp.array(
        [card, pub, pot, seat_f, rnd, to_call, raises_n, *hist[:MAX_HISTORY]],
        dtype=jnp.float32,
    )


@dataclass
class SDCFRConfig:
    iterations: int = 200
    traversals_per_iter: int = 32
    lr: float = 1e-3
    seed: int = 0
    buffer_capacity: int = 50_000
    train_batch_size: int = 2048
    # Full Leduc deal enumeration (90 deals × 2 seats) per sweep; repeat via deal_sweeps.
    enumerate_all_deals: bool = False
    deal_sweeps: int = 1


@dataclass
class _RegretVisit:
    """One traverser-node sample before infoset-level reach aggregation."""

    opp_reach: float
    util: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray
    enc: Array
    opponent_card: int


class SDCFRTrainer:
    """External-sampling SD-CFR on Leduc (validation ruler)."""

    def __init__(self, config: SDCFRConfig | None = None) -> None:
        self.config = config or SDCFRConfig()
        self.key = jax.random.PRNGKey(self.config.seed)
        net = LeducAdvantageNet()
        self.key, k = jax.random.split(self.key)
        dummy_x = jnp.zeros(LEDUC_ENCODE_DIM)
        dummy_m = jnp.ones(NUM_ACTIONS, dtype=bool)
        params = net.init(k, dummy_x, dummy_m)
        self.state = train_state.TrainState.create(
            apply_fn=net.apply,
            params=params,
            tx=optax.adam(self.config.lr),
        )
        # Tabular cumulative regrets for policy; strat_sum for average (reach-weighted, all acting nodes).
        self._cum_regret: dict[tuple, np.ndarray] = {}
        self._strat_sum: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_ACTIONS, dtype=np.float64)
        )
        self._strategy_snapshots: list[tuple[int, dict[tuple, np.ndarray]]] = []
        self._iter_weight: float = 1.0
        self._avg_policy_cache: dict[tuple, np.ndarray] = {}
        self._reservoir: list[tuple[Array, np.ndarray, np.ndarray]] = []
        self._reservoir_insert_count = 0
        self._target_max_history: list[float] = []

    def clear_policy_cache(self) -> None:
        self._avg_policy_cache.clear()

    def _reservoir_add(self, enc: Array, inst_adv: np.ndarray, mask: np.ndarray) -> None:
        """Reservoir sample: (info_state, instantaneous advantage vector, legal_mask)."""
        sample = (enc, inst_adv.astype(np.float32), mask)
        cap = self.config.buffer_capacity
        self._reservoir_insert_count += 1
        if len(self._reservoir) < cap:
            self._reservoir.append(sample)
            return
        j = int(np.random.randint(0, self._reservoir_insert_count))
        if j < cap:
            self._reservoir[j] = sample

    def _log_and_check_target_scale(self, iter_samples: list[tuple[Array, np.ndarray, np.ndarray]]) -> None:
        if not iter_samples:
            return
        abs_vals = [
            abs(float(inst[a]))
            for _, inst, mask in iter_samples
            for a in range(NUM_ACTIONS)
            if mask[a]
        ]
        mx = float(max(abs_vals))
        self._target_max_history.append(mx)
        print(
            f"  target_scale max_abs={mx:.4f} (Leduc bound {LEDUC_MAX_INSTANT_ADV:.1f})",
            flush=True,
        )
        if mx > LEDUC_MAX_INSTANT_ADV:
            raise AssertionError(
                f"instantaneous advantage target {mx:.4f} exceeds bound {LEDUC_MAX_INSTANT_ADV}"
            )
        # Catch runaway targets (cumulative-regret bug), not normal CF variance.
        if len(self._target_max_history) >= 5:
            baseline = float(np.median(self._target_max_history[-5:-1]))
            if baseline > 1.0 and mx > baseline * 10.0:
                raise AssertionError(
                    f"advantage target scale runaway: median_recent={baseline:.4f}, "
                    f"now={mx:.4f} (likely cumulative-regret regression bug)"
                )

    @staticmethod
    def _sigma_from_regrets(reg: np.ndarray, mask: np.ndarray) -> np.ndarray:
        positive = np.maximum(reg, 0.0) * mask
        total = positive.sum()
        if total > 0:
            return positive / total
        return mask.astype(np.float64) / max(mask.sum(), 1)

    def _encode_key(self, state: PyLeducState, seat: int) -> tuple:
        return tuple(np.asarray(encode_leduc_py(state, seat)).tolist())

    @staticmethod
    def _opponent_hole_card(state: PyLeducState, acting_seat: int) -> int:
        """Opponent hole card for the acting player's hidden world (not in py_info_key)."""
        return int(state.cards[1 - acting_seat])

    def _regret_matching_sigma(self, state: PyLeducState, seat: int) -> np.ndarray:
        """Strategy during traversal: regret-match tabular cumulative regrets (same as vanilla CFR)."""
        mask = py_legal(state)
        key = self._encode_key(state, seat)
        reg = self._cum_regret.get(key)
        if reg is None:
            return mask.astype(np.float64) / max(mask.sum(), 1)
        return self._sigma_from_regrets(reg, mask)

    def _begin_iteration_regret(self) -> None:
        """Collect traverser visits; flush reach-weighted targets once per infoset per iteration."""
        self._pending_regret: dict[tuple, list[_RegretVisit]] = defaultdict(list)

    def _flush_pending_regret(self) -> None:
        """Belief-weighted instantaneous targets for the net (reach sum/count, not count avg over worlds)."""
        for info_key, visits in self._pending_regret.items():
            if not visits:
                continue
            reg_inc = np.zeros(NUM_ACTIONS, dtype=np.float64)
            total_opp_reach = 0.0
            reach_by_world: dict[int, float] = defaultdict(float)
            for visit in visits:
                mask = visit.mask
                node_util = float(
                    sum(visit.sigma[a] * visit.util[a] for a in range(NUM_ACTIONS) if mask[a])
                )
                for a in range(NUM_ACTIONS):
                    if mask[a]:
                        reg_inc[a] += visit.opp_reach * (visit.util[a] - node_util)
                total_opp_reach += visit.opp_reach
                reach_by_world[visit.opponent_card] += visit.opp_reach
            last = visits[-1]
            if not hasattr(self, "_iter_samples"):
                self._iter_samples = []
            scale = max(total_opp_reach, 1e-12)
            inst_net = (reg_inc / scale).astype(np.float32)
            self._iter_samples.append((last.enc, inst_net, last.mask.copy()))
            if getattr(self, "_log_regret_aggregation", False):
                print(
                    f"  regret_flush key={info_key[:4]}... "
                    f"visits={len(visits)} opp_reach_sum={total_opp_reach:.4f} "
                    f"worlds={dict(reach_by_world)} inc={reg_inc}",
                    flush=True,
                )
        self._pending_regret.clear()

    def traverse(
        self,
        state: PyLeducState,
        traverser: int,
        key: Array,
        p0: float = 1.0,
        p1: float = 1.0,
    ) -> float:
        """External-sampling traversal; collect bounded instantaneous CF advantages."""
        if state.done:
            return float(state.rewards[traverser])

        seat = state.acting
        mask = py_legal(state)
        opp_reach = p1 if traverser == 0 else p0
        sigma = self._regret_matching_sigma(state, seat)
        info_key = self._encode_key(state, seat)
        acting_reach = p0 if seat == 0 else p1
        self._strat_sum[info_key] += self._iter_weight * acting_reach * sigma

        if seat == traverser:
            util = np.zeros(NUM_ACTIONS, dtype=np.float64)
            for a in range(NUM_ACTIONS):
                if mask[a]:
                    util[a] = self.traverse(
                        apply_action_py(state, a), traverser, key, p0, p1
                    )
            node_util = float(sum(sigma[a] * util[a] for a in range(NUM_ACTIONS) if mask[a]))
            inst = np.zeros(NUM_ACTIONS, dtype=np.float64)
            for a in range(NUM_ACTIONS):
                if mask[a]:
                    inst[a] = opp_reach * (util[a] - node_util)
            if info_key not in self._cum_regret:
                self._cum_regret[info_key] = np.zeros(NUM_ACTIONS, dtype=np.float64)
            self._cum_regret[info_key] += inst
            pending = getattr(self, "_pending_regret", None)
            if pending is not None:
                pending[info_key].append(
                    _RegretVisit(
                        opp_reach=float(opp_reach),
                        util=util.copy(),
                        sigma=sigma.copy(),
                        mask=mask.copy(),
                        enc=encode_leduc_py(state, seat),
                        opponent_card=self._opponent_hole_card(state, seat),
                    )
                )
            return node_util

        legal_idx = np.flatnonzero(mask)
        if self.config.enumerate_all_deals:
            util = 0.0
            for action in legal_idx:
                action = int(action)
                np0 = p0 * sigma[action] if seat == 0 else p0
                np1 = p1 * sigma[action] if seat == 1 else p1
                util += sigma[action] * self.traverse(
                    apply_action_py(state, action), traverser, key, np0, np1
                )
            return float(util)

        probs = sigma[legal_idx]
        probs = probs / probs.sum()
        key, sub = jax.random.split(key)
        pick = int(
            jax.random.choice(sub, legal_idx.shape[0], p=jnp.asarray(probs, dtype=jnp.float32))
        )
        action = int(legal_idx[pick])
        np0 = p0 * sigma[action] if seat == 0 else p0
        np1 = p1 * sigma[action] if seat == 1 else p1
        return self.traverse(apply_action_py(state, action), traverser, key, np0, np1)

    def train_iteration(self, iteration_weight: float = 1.0) -> float:
        from pokerbot.training.leduc_tabular_cfr import all_leduc_deals

        self._iter_weight = iteration_weight
        self._iter_samples = []
        self._begin_iteration_regret()
        if self.config.enumerate_all_deals:
            deals = all_leduc_deals()
            for _ in range(max(1, self.config.deal_sweeps)):
                for cards, pub in deals:
                    state = py_state_from_deal(cards, pub)
                    for seat in range(NUM_PLAYERS):
                        self.key, k2 = jax.random.split(self.key)
                        self.traverse(state, seat, k2)
        else:
            for _ in range(self.config.traversals_per_iter):
                self.key, k1, k2 = jax.random.split(self.key, 3)
                state = init_py_from_key(k1)
                for seat in range(NUM_PLAYERS):
                    self.traverse(state, seat, k2)

        self._flush_pending_regret()

        if not self._iter_samples:
            return 0.0

        self._log_and_check_target_scale(self._iter_samples)
        for enc, inst, mask in self._iter_samples:
            self._reservoir_add(enc, inst, mask)

        batch_n = min(self.config.train_batch_size, len(self._reservoir))
        idx = np.random.randint(0, len(self._reservoir), size=batch_n)
        batch = [self._reservoir[i] for i in idx]

        xs = jnp.stack([b[0] for b in batch])
        masks = jnp.stack([jnp.asarray(b[2]) for b in batch])
        targets = jnp.stack([jnp.asarray(b[1], dtype=jnp.float32) for b in batch])

        def loss_fn(params):
            pred = jax.vmap(lambda x, m: self.state.apply_fn(params, x, m))(xs, masks)
            diff = (pred - targets) * masks
            denom = jnp.maximum(jnp.sum(masks), 1.0)
            return jnp.sum(jnp.square(diff)) / denom

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(self.state.params)
        self.state = self.state.apply_gradients(grads=grads)
        return float(loss)

    def train(
        self,
        log_every: int = 1,
        checkpoint_dir: Path | None = None,
        checkpoint_every: int = 0,
        start_iter: int = 0,
    ) -> list[float]:
        losses: list[float] = []
        t0 = time.perf_counter()
        for i in range(start_iter, self.config.iterations):
            iter_num = i + 1
            losses.append(self.train_iteration(iteration_weight=float(iter_num)))
            if log_every > 0 and iter_num % log_every == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"SD-CFR iter {iter_num}/{self.config.iterations} "
                    f"loss={losses[-1]:.6f} elapsed={elapsed:.1f}s",
                    flush=True,
                )
            self._record_strategy_snapshot(iter_num)
            if (
                checkpoint_dir is not None
                and checkpoint_every > 0
                and iter_num % checkpoint_every == 0
            ):
                self._completed_iterations = iter_num
                ckpt_dir = Path(checkpoint_dir)
                save_checkpoint(self, ckpt_dir / f"sd_cfr_leduc_iter_{iter_num:04d}.pkl")
                save_checkpoint(self, ckpt_dir / "sd_cfr_leduc_latest.pkl")
                print(f"Checkpoint saved at iter {iter_num}", flush=True)
                self._log_checkpoint_eval(iter_num)
        return losses

    def _log_checkpoint_eval(self, iter_num: int) -> None:
        from pokerbot.training.leduc_tabular_cfr import (
            infoset_exploitability_mbb_from_policy,
            mean_l1_recovered_avg_vs_tabular,
        )

        self.clear_policy_cache()
        expl, u0, u1, br0, br1 = infoset_exploitability_mbb_from_policy(
            self.average_strategy_policy, samples=90
        )
        l1_mean = float("nan")
        tabular_avg = getattr(self, "_tabular_avg_ref", None)
        if tabular_avg is not None:
            l1_mean, _ = mean_l1_recovered_avg_vs_tabular(
                self.average_strategy_policy, tabular_avg
            )
        from pokerbot.training.leduc_tabular_cfr import LEDUC_SD_CFR_EXPLOITABILITY_MBB

        gate = "PASS" if expl <= LEDUC_SD_CFR_EXPLOITABILITY_MBB else "above_gate"
        print(
            f"Checkpoint eval iter {iter_num}: infoset BR={expl:.2f} mbb "
            f"(gate {LEDUC_SD_CFR_EXPLOITABILITY_MBB:.0f} {gate}) "
            f"L1_vs_tabular_mean={l1_mean:.4f} "
            f"u0={u0:.4f} u1={u1:.4f} br0={br0:.4f} br1={br1:.4f}",
            flush=True,
        )

    def _record_strategy_snapshot(self, iteration: int) -> None:
        """Linear-CFR weighted snapshot of regret-matched policies from cumulative regrets."""
        snap = {k: v.copy() for k, v in self._cum_regret.items()}
        self._strategy_snapshots.append((iteration, snap))

    def regret_matching_policy(self, state: LeducState, seat: int) -> np.ndarray:
        """Regret-matching strategy from accumulated CFR regrets (not the advantage net)."""
        py = PyLeducState.from_jax(state)
        mask = np.asarray(legal_actions(state), dtype=bool)
        key = self._encode_key(py, seat)
        reg = self._cum_regret.get(key, np.zeros(NUM_ACTIONS, dtype=np.float64))
        return self._sigma_from_regrets(reg, mask)

    def average_strategy_policy(self, state: LeducState, seat: int) -> np.ndarray:
        """Reach-weighted linear-CFR average from strat_sum (same accumulation as tabular CFR)."""
        cache_key = (tuple(np.asarray(encode_leduc(state, seat)).tolist()), int(seat))
        cached = self._avg_policy_cache.get(cache_key)
        if cached is not None:
            return cached
        mask = np.asarray(legal_actions(state), dtype=bool)
        key = cache_key[0]
        s = self._strat_sum.get(key)
        if s is not None:
            sigma = np.asarray(s, dtype=np.float64) * mask
            tot = sigma.sum()
            if tot > 0:
                sigma = sigma / tot
                self._avg_policy_cache[cache_key] = sigma
                return sigma
        sigma = self.regret_matching_policy(state, seat)
        self._avg_policy_cache[cache_key] = sigma
        return sigma


def save_checkpoint(trainer: SDCFRTrainer, path: Path) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {
                "params": trainer.state.params,
                "config": trainer.config,
                "completed_iterations": getattr(trainer, "_completed_iterations", 0),
                "strategy_snapshots": getattr(trainer, "_strategy_snapshots", []),
                "cum_regret": getattr(trainer, "_cum_regret", {}),
                "strat_sum": dict(getattr(trainer, "_strat_sum", {})),
            },
            f,
        )


def load_checkpoint(trainer: SDCFRTrainer, path: Path) -> int:
    """Restore params from checkpoint; returns completed iteration count."""
    import pickle

    with Path(path).open("rb") as f:
        data = pickle.load(f)
    trainer.state = trainer.state.replace(params=data["params"])
    if "config" in data:
        trainer.config = data["config"]
    trainer._strategy_snapshots = list(data.get("strategy_snapshots", []))
    trainer._cum_regret = dict(data.get("cum_regret", {}))
    if "strat_sum" in data:
        trainer._strat_sum = defaultdict(
            lambda: np.zeros(NUM_ACTIONS, dtype=np.float64),
            {k: np.asarray(v) for k, v in data["strat_sum"].items()},
        )
    return int(data.get("completed_iterations", 0))


# ---------------------------------------------------------------------------
# HU NLHE SD-CFR (Phase 2b) — see sd_cfr_nlhe.py
# ---------------------------------------------------------------------------

from pokerbot.training.sd_cfr_nlhe import (  # noqa: E402
    NLHE_MAX_INSTANT_ADV,
    SDCFRNLHEConfig,
    SDCFRNLHETrainer,
    load_nlhe_checkpoint,
    save_nlhe_checkpoint,
)
from pokerbot.training.nlhe_policy_net import NLHEWEVAPolicyNet  # noqa: E402

NLHEAdvantageNet = NLHEWEVAPolicyNet  # backwards-compatible alias

