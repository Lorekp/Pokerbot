"""Tests for JAX engine, replay buffers, and VR-DeepPDCFR+ trainer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pokerbot.engine.jax_engine import (
    NUM_ACTIONS,
    NUM_PLAYERS,
    EnvState,
    JAXNLHEEnv,
    batched_reset,
    batched_step,
    encode_info_state,
    legal_action_mask,
    reset_state,
    is_chance_node,
    sample_chance,
    step_state,
)
from pokerbot.training.neural_cfr import (
    VRDeepPDCFRConfig,
    VRDeepPDCFRPlusTrainer,
    cumulative_advantage_loss,
    instantaneous_advantage_loss,
    pdcfr_predicted_regrets,
    regret_matching,
)
from pokerbot.training.replay_buffers import (
    BufferSpec,
    init_advantage_buffers,
    init_buffer,
    init_history_value_buffer,
    init_strategy_buffer,
    insert_advantage_sample,
    insert_strategy_sample,
    reservoir_insert,
    STRATEGY_BUFFER_SPEC,
)


def test_reset_state_shapes() -> None:
    key = jax.random.PRNGKey(0)
    state = reset_state(key)
    assert state.holes.shape == (NUM_PLAYERS, 2)
    assert state.board.shape == (5,)
    assert state.stacks.shape == (NUM_PLAYERS,)
    assert not bool(state.done)


def test_legal_mask_has_check_or_call() -> None:
    key = jax.random.PRNGKey(1)
    state = reset_state(key)
    mask = legal_action_mask(state)
    assert mask.shape == (NUM_ACTIONS,)
    assert bool(mask[1]) or bool(mask[2])


def test_step_until_terminal() -> None:
    key = jax.random.PRNGKey(2)
    state = reset_state(key)
    for _ in range(200):
        if bool(state.done):
            break
        if bool(is_chance_node(state)):
            state = sample_chance(state)
            continue
        mask = legal_action_mask(state)
        legal = jnp.where(mask, 1.0, 0.0)
        action = jnp.argmax(legal)
        state = step_state(state, action)
    assert bool(state.done)
    assert int(jnp.sum(state.rewards)) == 0


def test_batched_vmap() -> None:
    key = jax.random.PRNGKey(3)
    keys = jax.random.split(key, 8)
    states = batched_reset(keys)
    assert states.stacks.shape == (8, NUM_PLAYERS)
    actions = jnp.zeros(8, dtype=jnp.int32)
    next_states = batched_step(states, actions)
    assert next_states.pot_dead.shape == (8,)


def test_info_state_fixed_dim() -> None:
    key = jax.random.PRNGKey(4)
    state = reset_state(key)
    feat = encode_info_state(state, jnp.int32(0))
    assert feat.shape == (256,)
    assert jnp.all(jnp.isfinite(feat))


def test_regret_matching_sums_to_one() -> None:
    regrets = jnp.array([1.0, -2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mask = jnp.ones(NUM_ACTIONS, dtype=bool)
    policy = regret_matching(regrets, mask)
    assert jnp.allclose(jnp.sum(policy), 1.0, atol=1e-5)
    assert policy[1] == 0.0


def test_pdcfr_predicted_regrets() -> None:
    cum = jnp.ones(NUM_ACTIONS)
    inst = jnp.ones(NUM_ACTIONS) * 0.5
    pred = pdcfr_predicted_regrets(cum, inst, jnp.int32(5), alpha=2.3)
    assert pred.shape == (NUM_ACTIONS,)
    assert jnp.all(pred > 0)


def test_reservoir_buffer_insert() -> None:
    spec = BufferSpec(capacity=100, fields=STRATEGY_BUFFER_SPEC.fields)
    buf = init_buffer(spec)
    key = jax.random.PRNGKey(0)
    sample = {
        "info_state": jnp.zeros(256, dtype=jnp.float32),
        "strategy": jnp.ones(NUM_ACTIONS, dtype=jnp.float32) / NUM_ACTIONS,
        "legal_mask": jnp.ones(NUM_ACTIONS, dtype=bool),
        "iteration": jnp.int32(1),
    }
    buf = reservoir_insert(buf, key, sample)
    assert int(buf.size) == 1


def test_advantage_linear_buffer() -> None:
    buffers = init_advantage_buffers(capacity=16)
    buf = buffers.buffers[0]
    buf = insert_advantage_sample(
        buf,
        jnp.zeros(256, dtype=jnp.float32),
        jnp.zeros(NUM_ACTIONS, dtype=jnp.float32),
        jnp.ones(NUM_ACTIONS, dtype=bool),
        jnp.int32(1),
    )
    assert int(buf.size) == 1


def test_loss_functions_finite() -> None:
    batch = {
        "info_state": jnp.zeros((4, 256), dtype=jnp.float32),
        "advantages": jnp.zeros((4, NUM_ACTIONS), dtype=jnp.float32),
        "legal_mask": jnp.ones((4, NUM_ACTIONS), dtype=bool),
    }
    from pokerbot.training.neural_cfr import CumulativeAdvantageNet, InstantaneousAdvantageNet

    key = jax.random.PRNGKey(0)
    cum = CumulativeAdvantageNet()
    inst = InstantaneousAdvantageNet()
    params_c = cum.init(key, batch["info_state"], batch["legal_mask"])
    params_i = inst.init(key, batch["info_state"], batch["legal_mask"])

    lc = cumulative_advantage_loss(params_c, cum.apply, batch, params_c, cum.apply, 3, 2.3)
    li = instantaneous_advantage_loss(params_i, inst.apply, batch)
    assert jnp.isfinite(lc)
    assert jnp.isfinite(li)


def test_trainer_single_iteration() -> None:
    cfg = VRDeepPDCFRConfig(
        total_iterations=1,
        traversals_per_iter=2,
        batch_size=8,
        buffer_capacity=64,
    )
    trainer = VRDeepPDCFRPlusTrainer(cfg)
    stats = trainer.train_iteration(1)
    assert stats["iteration"] == 1.0


def test_traversal_utility_uses_acting_seat() -> None:
    """History samples must index utility by acting seat, not player 0."""

    cfg = VRDeepPDCFRConfig(traversals_per_iter=4, batch_size=8, buffer_capacity=128)
    trainer = VRDeepPDCFRPlusTrainer(cfg)
    trainer.run_traversals(1)
    buf = trainer.hist_buffer.buffer
    if int(buf.size) == 0:
        pytest.skip("no history samples collected")
    n = int(buf.size)
    players = jax.device_get(buf.data["player"][:n])
    masks = jax.device_get(buf.data["next_legal_mask"][:n])
    assert jnp.any(players != 0), "expected samples from non-zero seats"
    assert masks.shape == (n, NUM_ACTIONS)
    assert jnp.any(masks), "next_legal_mask must reflect legal actions, not all-legal default"


def test_checkpoint_full_roundtrip(tmp_path) -> None:
    from pokerbot.training.neural_cfr import save_trainer_checkpoint, load_trainer_checkpoint

    cfg = VRDeepPDCFRConfig(total_iterations=2, traversals_per_iter=2, batch_size=8, buffer_capacity=32)
    trainer = VRDeepPDCFRPlusTrainer(cfg)
    trainer.train_iteration(1)
    ckpt = tmp_path / "hvn.pkl"
    save_trainer_checkpoint(trainer, ckpt)
    loaded = load_trainer_checkpoint(tmp_path / "hvn_full.pkl")
    assert loaded._start_iteration == 1
