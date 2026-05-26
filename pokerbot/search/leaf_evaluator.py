"""JAX-batched HistoryValueNet leaf evaluation for TurboReBeL search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from pokerbot.engine.jax_engine import HISTORY_FEATURE_DIM, NUM_ACTIONS
from pokerbot.training.neural_cfr import HistoryValueNet, load_network_params


DEFAULT_CHECKPOINT = Path("checkpoints/history_value_net.pkl")


class HistoryValueEvaluator:
    """Batched Q(h, a) inference using trained HistoryValueNet params."""

    def __init__(
        self,
        checkpoint: str | Path | None = DEFAULT_CHECKPOINT,
        hidden_dim: int = 256,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.module = HistoryValueNet(hidden_dim)
        self.params: Any = None
        self._blueprint_params: Any = None
        self._total_iterations = 100

        ckpt_path = Path(checkpoint) if checkpoint else None
        if ckpt_path and ckpt_path.exists():
            payload = load_network_params(ckpt_path)
            self.params = payload["history_value_params"]
            self._blueprint_params = payload.get("average_strategy_params")
            self.hidden_dim = int(payload.get("hidden_dim", hidden_dim))
            self._total_iterations = int(payload.get("total_iterations", 100))
            self.module = HistoryValueNet(self.hidden_dim)
        else:
            key = jax.random.PRNGKey(0)
            dummy = jnp.zeros((1, HISTORY_FEATURE_DIM), dtype=jnp.float32)
            self.params = self.module.init(key, dummy)

        @jax.jit
        def _eval_batch(params: Any, histories: Array) -> Array:
            return jax.vmap(lambda h: self.module.apply(params, h))(histories)

        self._eval_batch_jit = _eval_batch

    @property
    def has_trained_checkpoint(self) -> bool:
        return self._blueprint_params is not None

    @property
    def blueprint_params(self) -> Any:
        return self._blueprint_params

    def evaluate_batch(self, history_features: np.ndarray) -> np.ndarray:
        """Evaluate Q(h,a) for batch of history vectors. Returns (B, 11)."""
        arr = jnp.asarray(history_features, dtype=jnp.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        q = self._eval_batch_jit(self.params, arr)
        return np.asarray(q)

    def evaluate_batch_jit(self, history_features: Array) -> Array:
        """JIT path for pre-JAX history arrays."""
        if history_features.ndim == 1:
            history_features = history_features[None, :]
        return self._eval_batch_jit(self.params, history_features)

    def leaf_value(self, history_features: np.ndarray, legal_mask: np.ndarray | None = None) -> float:
        """Scalar leaf value: max_a Q(h,a) over legal actions."""
        q = self.evaluate_batch(history_features)[0]
        if legal_mask is not None:
            q = np.where(legal_mask, q, -1e9)
        return float(np.max(q))


class MockHistoryValueEvaluator(HistoryValueEvaluator):
    """Deterministic mock leaf evaluator for tests without a trained checkpoint."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__(checkpoint=None, hidden_dim=hidden_dim)
        self._blueprint_params = None

    def evaluate_batch(self, history_features: np.ndarray) -> np.ndarray:
        if history_features.ndim == 1:
            history_features = history_features[None, :]
        b = history_features.shape[0]
        base = history_features[:, :NUM_ACTIONS].astype(np.float32)
        if base.shape[1] < NUM_ACTIONS:
            pad = np.zeros((b, NUM_ACTIONS - base.shape[1]), dtype=np.float32)
            base = np.concatenate([base, pad], axis=1)
        return base[:, :NUM_ACTIONS]
