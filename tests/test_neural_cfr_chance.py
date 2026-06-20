"""Regression: the VR-DeepPDCFR+ traversal must NEVER treat a standalone chance (deal) node as a
decision. ``jax_engine`` emits a chance node between streets (``pending_deal=True`` -> ``action_seat=-1``);
the pre-fix scan dealt the CURRENT node and then applied a sampled action to the dealt state, which both
recorded the chance node as a phantom ``seat=-1`` sample (polluting the avg-strategy + history buffers)
AND overwrote the genuine first decision of every new street -- corrupting essentially all postflop
learning. The Leduc gate cannot catch this (Leduc has no standalone chance node), so it lived latent.

The fix folds the deal INTO the transition (apply action, then deal-if-chance) so the scan only ever
lands on real decision nodes. These tests fail on the old behavior:
  * test_engine_deal_after_step_invariant -- the acting seat is never -1 along a real hand to showdown.
  * test_scan_emits_no_chance_decisions -- no ingested traversal row carries seat<0 or next_seat<0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.training.cfr_env import nlhe_hu_env_spec


def _deal_after_step(env, state, action):
    """The fixed transition (mirrors neural_cfr._traversal_step_partial): apply the action, deal any
    street it closed, then finalize a river showdown (the deal advances to TERMINAL but the pot is
    computed on the next apply_action; an inert CHECK resolves it)."""
    from pokerbot.engine.jax_engine import CHECK

    state = env.step(state, action)
    state = jax.lax.cond(env.is_chance(state), env.sample_chance, lambda s: s, state)
    showdown_pending = jnp.logical_and(jnp.logical_not(env.done(state)), env.action_seat(state) < 0)
    return jax.lax.cond(showdown_pending, lambda s: env.step(s, jnp.int32(CHECK)), lambda s: s, state)


def test_engine_deal_after_step_invariant():
    """Drive a HU hand check/call-down to showdown. With deal-after-step the scan never sees a chance
    node as a decision: the acting seat is >=0 at every decision, the flop/turn/river get dealt, and the
    hand ends terminal -- all without a standalone chance node ever being the acting state."""
    env = nlhe_hu_env_spec()
    state = env.reset(jax.random.PRNGKey(0))

    max_board_seen = 0
    saw_postflop_decision = False
    for _ in range(64):
        if bool(env.done(state)):
            break
        # Invariant under test: after deal-after-step, a decision node is NEVER a chance node and the
        # acting seat is a real seat (the bug produced action_seat == -1 here).
        assert not bool(env.is_chance(state)), "scan landed on a standalone chance node"
        seat = int(env.action_seat(state))
        assert seat >= 0, f"acting seat is {seat} (phantom chance-node decision)"

        n_board = int(jnp.sum(state.board >= 0))
        max_board_seen = max(max_board_seen, n_board)
        if n_board >= 3:
            saw_postflop_decision = True

        mask = np.asarray(env.legal_mask(state))
        # Passive line to reach showdown: prefer CALL(2), else CHECK(1), else any non-fold legal.
        if mask[2]:
            action = 2
        elif mask[1]:
            action = 1
        else:
            nonfold = [i for i in range(1, mask.shape[0]) if mask[i]]
            action = nonfold[0] if nonfold else 0
        state = _deal_after_step(env, state, jnp.int32(action))

    assert bool(env.done(state)), "check/call-down did not terminate"
    assert saw_postflop_decision, "never reached a postflop decision (test would not exercise the fix)"
    assert max_board_seen >= 3, f"flop never dealt (max board {max_board_seen})"


def test_scan_emits_no_chance_decisions():
    """The real traversal scan must emit no active sample whose acting seat (or next seat) is a chance
    node. Pre-fix, every street-closing action recorded next_seat=-1 and each deal recorded seat=-1."""
    from pokerbot.training.neural_cfr import VRDeepPDCFRConfig, VRDeepPDCFRPlusTrainer

    cfg = VRDeepPDCFRConfig(hidden_dim=16, traversals_per_iter=48, buffer_capacity=1000, seed=0)
    trainer = VRDeepPDCFRPlusTrainer(cfg, env=nlhe_hu_env_spec())

    cum_params = tuple(s.params for s in trainer.cum_states)
    inst_params = tuple(s.params for s in trainer.inst_states)
    hv_params = trainer.hv_state.params

    k = cfg.traversals_per_iter
    keys = jax.random.split(jax.random.PRNGKey(7), k * 2)
    adv, strat, hist, is_trav, active = trainer._batched_scans(
        keys[:k], keys[k:], jnp.int32(0), jnp.int32(1),
        cum_params, inst_params, hv_params,
        cfg.alpha, cfg.epsilon, cfg.reward_scale, cfg.hidden_dim, trainer.env,
    )

    active = np.asarray(active)                      # (k, steps) bool
    seat = np.asarray(hist["player"])                # (k, steps) int -- == record seat
    next_seat = np.asarray(hist["next_seat"])        # (k, steps) int
    done = np.asarray(hist["done"]) > 0.5            # (k, steps) bool -- next_state is terminal

    # A real terminal transition legitimately has next_seat == -1 (terminal has no actor); only
    # NON-terminal next nodes must be real decisions. The bug instead produced next_seat == -1 on
    # NON-terminal street-closing actions (their next node was a standalone chance node).
    nonterminal_next = active & ~done
    assert active.sum() > k, "traversals were degenerate (<=1 decision/hand); fix not exercised"
    assert (seat[active] >= 0).all(), "an ingested sample has a chance-node (seat<0) decision"
    assert (next_seat[nonterminal_next] >= 0).all(), \
        "an ingested sample's non-terminal next node is a chance node (next_seat<0)"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
