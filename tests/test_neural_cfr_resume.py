"""Resume-correctness regression for VR-DeepPDCFR+ checkpoints (the 24h-run durability fixes).

A checkpoint is "always good / run until killed" only if --resume restores ALL mutable state. Pre-fix,
load rebuilt fresh TrainStates (Adam moments re-zeroed -> R1), empty reservoirs (the served avg net
trains only on the strategy reservoir -> R2), and prev==current cum params (PDCFR+ bootstrap target ->
R3). These tests train a few iterations, round-trip the checkpoint, and assert every piece is restored.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokerbot.training.cfr_env import nlhe_hu_env_spec
from pokerbot.training.neural_cfr import (
    VRDeepPDCFRConfig,
    VRDeepPDCFRPlusTrainer,
    _atomic_save_trainer_checkpoint,
    load_trainer_checkpoint,
)


def _tree_allclose(a, b) -> bool:
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    if len(la) != len(lb):
        return False
    return all(np.allclose(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def _any_nonzero(tree) -> bool:
    return any(np.any(np.asarray(x) != 0) for x in jax.tree_util.tree_leaves(tree))


def _train_small(tmp_path, iters=2):
    cfg = VRDeepPDCFRConfig(
        total_iterations=iters, traversals_per_iter=8, batch_size=32, hidden_dim=16,
        buffer_capacity=500, train_steps_per_iter=2, seed=0,
    )
    trainer = VRDeepPDCFRPlusTrainer(cfg, env=nlhe_hu_env_spec())
    for t in range(1, iters + 1):
        trainer.train_iteration(t)
    trainer._iteration = iters
    return trainer


def test_resume_restores_optimizer_buffers_and_prev_cum(tmp_path):
    trainer = _train_small(tmp_path)
    ckpt = tmp_path / "hu.pkl"
    full = _atomic_save_trainer_checkpoint(trainer, ckpt)
    assert ckpt.exists() and full.exists()                      # A1: both files promoted
    assert (tmp_path / "hu_buffers.pkl").exists()               # reservoirs in the sidecar (default)

    loaded = load_trainer_checkpoint(ckpt)

    # iteration counter (the PDCFR+ discount schedule depends on it)
    assert loaded._start_iteration == trainer._iteration

    # R1: Adam moments restored (and actually non-trivial after real training), not re-zeroed.
    assert _any_nonzero(trainer.avg_state.opt_state), "test setup: optimizer never moved"
    assert _tree_allclose(loaded.avg_state.opt_state, trainer.avg_state.opt_state)
    assert _tree_allclose(loaded.cum_states[3].opt_state, trainer.cum_states[3].opt_state)

    # R2: reservoirs restored (size AND reservoir count -> replacement probability stays correct).
    assert int(trainer.strat_buffer.buffer.size) > 0, "test setup: strat reservoir empty"
    assert int(loaded.strat_buffer.buffer.size) == int(trainer.strat_buffer.buffer.size)
    assert int(loaded.strat_buffer.buffer.count) == int(trainer.strat_buffer.buffer.count)
    assert int(loaded.hist_buffer.buffer.size) == int(trainer.hist_buffer.buffer.size)
    assert _tree_allclose(loaded.strat_buffer.buffer.data, trainer.strat_buffer.buffer.data)

    # R3: previous-iteration cumulative params restored (not reset to current).
    assert _tree_allclose(loaded._prev_cum_params, trainer._prev_cum_params)

    # params themselves restored
    assert _tree_allclose(loaded.avg_state.params, trainer.avg_state.params)


def test_resume_backward_compatible_with_legacy_payload(tmp_path):
    """A pre-R1/R2/R3 checkpoint (no opt_states/buffers/prev-cum keys, no sidecar) must still load --
    falling back to a fresh optimizer + empty reservoirs rather than raising KeyError."""
    import pickle

    trainer = _train_small(tmp_path, iters=1)
    full = _atomic_save_trainer_checkpoint(trainer, tmp_path / "hu.pkl")
    (tmp_path / "hu_buffers.pkl").unlink()                       # remove the reservoir sidecar (legacy)
    with full.open("rb") as f:
        payload = pickle.load(f)
    for k in ("history_value_opt_state", "average_strategy_opt_state",
              "cumulative_advantage_opt_states", "instantaneous_advantage_opt_states",
              "prev_cumulative_advantage_params", "strategy_buffer", "history_value_buffer"):
        payload.pop(k, None)
    with full.open("wb") as f:
        pickle.dump(payload, f)

    loaded = load_trainer_checkpoint(tmp_path / "hu.pkl")
    assert loaded._start_iteration == 1
    assert int(loaded.strat_buffer.buffer.size) == 0           # fell back to empty reservoir
    assert _tree_allclose(loaded.avg_state.params, trainer.avg_state.params)   # params still restored


def test_buffer_sidecar_split_cheap_then_restore(tmp_path):
    """A cheap checkpoint (include_buffers=False) must NOT embed the reservoirs in _full, yet resume
    still restores them from a RETAINED sidecar from an earlier heavy save. This is the long-run path:
    frequent cheap core checkpoints + an occasional heavy reservoir snapshot."""
    import pickle

    trainer = _train_small(tmp_path, iters=2)
    ckpt = tmp_path / "hu.pkl"
    _atomic_save_trainer_checkpoint(trainer, ckpt, include_buffers=True)    # heavy: writes sidecar
    sidecar = tmp_path / "hu_buffers.pkl"
    assert sidecar.exists()
    heavy_size = sidecar.stat().st_mtime

    # a later cheap save (no reservoir snapshot) -- _full must carry NO embedded buffer keys, and the
    # sidecar must be left untouched.
    trainer.train_iteration(3); trainer._iteration = 3
    full = _atomic_save_trainer_checkpoint(trainer, ckpt, include_buffers=False)
    with full.open("rb") as f:
        core = pickle.load(f)
    assert "strategy_buffer" not in core and "history_value_buffer" not in core   # core stays light
    assert sidecar.stat().st_mtime == heavy_size                                  # sidecar untouched

    loaded = load_trainer_checkpoint(ckpt)
    assert loaded._start_iteration == 3                                           # core is current
    assert int(loaded.strat_buffer.buffer.size) > 0                               # reservoir from sidecar
    assert _tree_allclose(loaded.avg_state.opt_state, trainer.avg_state.opt_state)


def test_legacy_embedded_buffers_still_restore(tmp_path):
    """An old single-file _full that EMBEDS the reservoirs (pre-split format), with no sidecar, must
    still restore them -- this is the current on-disk hu_vrpdcfr_v2 checkpoint's shape."""
    import pickle

    trainer = _train_small(tmp_path, iters=2)
    full = _atomic_save_trainer_checkpoint(trainer, tmp_path / "hu.pkl", include_buffers=True)
    # fold the sidecar back INTO _full and delete it -> the legacy embedded shape.
    sidecar = tmp_path / "hu_buffers.pkl"
    with full.open("rb") as f:
        core = pickle.load(f)
    with sidecar.open("rb") as f:
        bufs = pickle.load(f)
    core["strategy_buffer"] = bufs["strategy_buffer"]
    core["history_value_buffer"] = bufs["history_value_buffer"]
    with full.open("wb") as f:
        pickle.dump(core, f)
    sidecar.unlink()

    loaded = load_trainer_checkpoint(tmp_path / "hu.pkl")
    assert int(loaded.strat_buffer.buffer.size) == int(trainer.strat_buffer.buffer.size) > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
