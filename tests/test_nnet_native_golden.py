"""Golden cross-check for the native neural serve (`tools/battery_native/src/nnet.rs`).

The native LBR serves the HU neural net by re-implementing `encode_info_state` + the AverageStrategyNet
MLP forward in Rust. The one real skew risk is the encoder feature LAYOUT, so -- exactly like the other
five native tools -- we gate it on a golden cross-check: serve N random states through BOTH the JAX net
and the native port and assert the action distributions match to f32 tolerance.

Run with the SAME POKERBOT_INFO_SUITS the checkpoint was trained with (the served checkpoint stamps it):
  POKERBOT_INFO_SUITS=1 POKERBOT_PREFLOP_LOSSLESS=1 python -m pytest tests/test_nnet_native_golden.py -q
"""

from __future__ import annotations

import os

import numpy as np
import pytest

CKPT = os.environ.get("POKERBOT_HU_CKPT", "checkpoints/sd_cfr_hu_neural/hu_vrpdcfr_v2.pkl")


@pytest.mark.skipif(not os.path.exists(CKPT), reason="HU neural checkpoint not present")
def test_native_neural_serve_matches_jax():
    import jax
    import jax.numpy as jnp

    import pokerbot.engine.jax_engine as _je
    from pokerbot.engine.jax_engine import (
        EnvState, NUM_PLAYERS, encode_info_state, init_hand, legal_action_mask,
    )
    from pokerbot.engine.np_engine import (
        apply_action_np, is_chance_node_np, legal_action_mask_np, sample_chance_np, to_numpy_state,
    )
    from pokerbot.training.neural_cfr import AverageStrategyNet, load_network_params
    from pokerbot.eval import lbr_native as L

    if not L.available():
        pytest.skip("native battery_native (Lbr*Harness) not built")

    payload = load_network_params(CKPT)
    if bool(payload.get("info_suits")) != bool(_je._INFO_SUITS):
        pytest.skip(
            f"POKERBOT_INFO_SUITS ({int(bool(_je._INFO_SUITS))}) != checkpoint "
            f"({int(bool(payload.get('info_suits')))}); set it to match before importing pokerbot")

    # JAX serve (the make_neural_policy path).
    params = payload["average_strategy_params"]
    net = AverageStrategyNet(int(payload["hidden_dim"]))

    @jax.jit
    def _jserve(state, seat):
        return net.apply(params, encode_info_state(state, seat), legal_action_mask(state, seat)) \
            * legal_action_mask(state, seat)

    fields = EnvState._fields

    def jax_probs(env_np, seat):
        st = EnvState(**{f: jnp.asarray(getattr(env_np, f)) for f in fields})
        pr = np.asarray(_jserve(st, jnp.int32(seat)), np.float64)
        m = np.asarray(legal_action_mask_np(env_np, seat), bool)
        pr = pr * m
        s = pr.sum()
        return pr / s if s > 1e-12 else m / max(m.sum(), 1)

    harness, _ = L.build_neural_harness(CKPT)
    from pokerbot.search import native_solver

    HU_OCC = jnp.asarray(np.array([0, 0, 0, 1, 0, 1], bool))
    HU = (3, 5)
    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(0)
    max_diff, decisions = 0.0, 0
    for h in range(120):
        key, k = jax.random.split(key)
        depth = float(rng.choice([10, 20, 40, 80]))
        stacks = jnp.full(NUM_PLAYERS, int(depth * 100), jnp.int32)
        env = to_numpy_state(init_hand(k, button=jnp.int32(HU[h % 2]), stacks=stacks, occupancy=HU_OCC))
        d = 0
        while not bool(env.done) and d < 24:
            if is_chance_node_np(env):
                env = sample_chance_np(env)
                continue
            seat = int(env.action_seat)
            if seat < 0:
                break
            pj = jax_probs(env, seat)
            pn = np.asarray(harness.serve(native_solver._to_flat(env).tobytes(), seat), np.float64)
            max_diff = max(max_diff, float(np.abs(pj - pn).max()))
            decisions += 1
            m = legal_action_mask_np(env, seat)
            env = apply_action_np(env, int(rng.choice(np.flatnonzero(m))))
            d += 1
    assert decisions > 200, f"too few decisions sampled: {decisions}"
    assert max_diff < 1e-4, f"native serve diverges from JAX: max |diff|={max_diff:.2e}"
