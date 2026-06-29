"""Sharpening tests for the highest-leverage CFR-AUDIT fixes (1.1/1.4, 2.2, 4.1).

Defends against the silent-uniform-averaging, runaway-discount, and
unbounded-advantage bugs identified in the CFR-AUDIT.

Audit IDs covered
-----------------
1.1/1.4: ``self._iter_weight`` is wired in the parent train_iteration so any caller
         (not just the distributed single-thread path) gets CFR+ linear averaging.
2.2:     ``apply_dcfr_discount`` clamps post-discount regret to a sentinel ceiling
         to bound the explosion risk and noise-induced sign flips.
4.1:     ``insert_advantage_sample`` clamps advantages to [-1, 1] at insert so a
         runaway regret value cannot dominate the MSE distillation loss.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# 1.1/1.4: self._iter_weight is wired in the parent train_iteration
# --------------------------------------------------------------------------- #
class TestIterWeightWired:
    """The audit found that ``self._iter_weight`` was stuck at 1.0 in any caller
    that did not go through ``run_traversal_phase_single_threaded`` (the
    distributed path). The parent ``train_iteration`` now sets it defensively."""

    def test_iter_weight_assigned_at_init_to_1(self) -> None:
        """At construction, ``self._iter_weight`` is the documented default 1.0."""
        from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer
        cfg = SDCFRNLHEConfig(seed=0, max_traverse_depth=4)
        trainer = SDCFRNLHETrainer(cfg)
        assert trainer._iter_weight == 1.0

    def test_train_iteration_assigns_iter_weight(self, monkeypatch) -> None:
        """Construct a trainer, then patch ``run_traversal_phase_single_threaded``
        to a no-op so we can observe the assignment on ``self``. The parent MUST
        set ``self._iter_weight = float(iteration_weight)`` before calling the
        distributed path; without the audit fix, the field would stay at 1.0."""
        from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer
        cfg = SDCFRNLHEConfig(seed=0, max_traverse_depth=4)
        trainer = SDCFRNLHETrainer(cfg)

        # Patch the distributed path to a no-op so the test does not need the
        # full training state (and does not take seconds to run).
        from pokerbot.training import sd_cfr_nlhe_distributed
        monkeypatch.setattr(
            sd_cfr_nlhe_distributed, "run_traversal_phase_single_threaded",
            lambda *args, **kwargs: None,
        )
        # And patch the finish step to a no-op (it needs a fitted net).
        monkeypatch.setattr(
            trainer, "_finish_iteration_after_traversal", lambda: 0.0,
        )
        # The audit-failing path: call train_iteration with a non-1 weight.
        trainer.train_iteration(iteration_weight=7.0)
        assert trainer._iter_weight == 7.0, (
            f"train_iteration failed to assign self._iter_weight=7.0; "
            f"got {trainer._iter_weight} (audit finding 1.1/1.4 still latent)"
        )

    def test_train_iteration_assigns_iter_weight_42(self, monkeypatch) -> None:
        """The 42 is a memorable value (Hitchhiker's Guide) to confirm the
        assignment is the actual iteration_weight arg and not a hard-coded constant."""
        from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer
        from pokerbot.training import sd_cfr_nlhe_distributed
        cfg = SDCFRNLHEConfig(seed=0, max_traverse_depth=4)
        trainer = SDCFRNLHETrainer(cfg)
        monkeypatch.setattr(
            sd_cfr_nlhe_distributed, "run_traversal_phase_single_threaded",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(trainer, "_finish_iteration_after_traversal", lambda: 0.0)
        trainer.train_iteration(iteration_weight=42.0)
        assert trainer._iter_weight == 42.0


# --------------------------------------------------------------------------- #
# 2.2: apply_dcfr_discount clamps post-discount regret
# --------------------------------------------------------------------------- #
class TestDCFRClamp:
    """``apply_dcfr_discount`` now clamps the post-discount regret to a sentinel
    ceiling (default ±1e9) and the strategy sum similarly. The clamp prevents
    a noise-induced sign flip on the next discount round from applying the
    wrong factor forever."""

    def _make_table(self, capacity: int = 16) -> "SharedHogwildTable":
        from pokerbot.training.shared_hogwild_table import SharedHogwildTable
        return SharedHogwildTable.create(capacity=capacity, num_actions=4)

    def test_post_discount_regret_clipped(self) -> None:
        """A regret row with magnitude > clip value gets clipped after discount."""
        from pokerbot.training.shared_hogwild_table import SharedHogwildTable
        table = SharedHogwildTable.create(capacity=4, num_actions=2)
        # Claim slot 0 and write a row with magnitude 1e15.
        slot = table.slot_for(b"\x00" * 16)
        row = np.array([1.0e15, 5.0e14], dtype=np.float64)
        table.regret[slot] = row
        # Discount at t=1 (pos=neg=0.5 with default alpha=1.5,beta=0).
        table.apply_dcfr_discount(1.0, reg_clip=1.0e9, strat_clip=1.0e9)
        # Without the clamp the post-discount value would be ~5e14; with the clamp it is 1e9.
        out = table.regret[slot]
        assert out[0] <= 1.0e9 + 1e-3 and out[0] >= -1.0e9 - 1e-3, (
            f"regret[0]={out[0]} not clipped to ±1e9"
        )
        assert out[1] <= 1.0e9 + 1e-3 and out[1] >= -1.0e9 - 1e-3

    def test_clip_can_be_disabled(self) -> None:
        """``reg_clip=0`` disables the regret clip (preserves the legacy behavior
        for callers that explicitly want it)."""
        from pokerbot.training.shared_hogwild_table import SharedHogwildTable
        table = SharedHogwildTable.create(capacity=4, num_actions=2)
        slot = table.slot_for(b"\x00" * 16)
        # A small regret row — no clipping needed either way; the test just
        # confirms the legacy path still computes a finite discount.
        table.regret[slot] = np.array([2.0, -1.0], dtype=np.float64)
        table.strat[slot] = np.array([3.0, 4.0], dtype=np.float64)
        # reg_clip=0 = no clip; strat_clip=0 = no clip.
        table.apply_dcfr_discount(1.0, reg_clip=0.0, strat_clip=0.0)
        # With alpha=1.5, beta=0, gamma=2, t=1:
        #   ta = 1^1.5 = 1, pos = 1/(1+1) = 0.5
        #   tb = 1^0 = 1, neg = 1/(1+1) = 0.5
        #   strat = (1/2)^2 = 0.25
        # pos_reg * 0.5 = 2.0 * 0.5 = 1.0
        # neg_reg * 0.5 = -1.0 * 0.5 = -0.5
        # strat * 0.25 = [0.75, 1.0]
        out_reg = table.regret[slot]
        out_strat = table.strat[slot]
        assert abs(out_reg[0] - 1.0) < 1e-9, f"pos reg not discounted to 1.0; got {out_reg[0]}"
        assert abs(out_reg[1] - (-0.5)) < 1e-9, f"neg reg not discounted to -0.5; got {out_reg[1]}"
        assert abs(out_strat[0] - 0.75) < 1e-9
        assert abs(out_strat[1] - 1.0) < 1e-9

    def test_strat_clip_also_applied(self) -> None:
        """The strategy sum is also clipped to ±strat_clip."""
        from pokerbot.training.shared_hogwild_table import SharedHogwildTable
        table = SharedHogwildTable.create(capacity=4, num_actions=2)
        slot = table.slot_for(b"\x00" * 16)
        table.regret[slot] = np.array([1.0, 1.0], dtype=np.float64)
        table.strat[slot] = np.array([1.0e15, 1.0e15], dtype=np.float64)
        # With small discount factors the strat is only slightly down-weighted;
        # clamp kicks in only on the post-discount value. We use t=1 (smallest
        # discount = 0.5) to leave a 5e14 value post-discount, which the clamp
        # brings to 1e9.
        table.apply_dcfr_discount(1.0, reg_clip=1.0e9, strat_clip=1.0e9)
        out = table.strat[slot]
        assert out[0] <= 1.0e9 + 1e-3 and out[0] >= -1.0e9 - 1e-3


# --------------------------------------------------------------------------- #
# 4.1: insert_advantage_sample clamps advantages to [-1, 1]
# --------------------------------------------------------------------------- #
class TestAdvantageClamp:
    """``insert_advantage_sample`` clamps the inserted advantages to [-1, 1] so a
    single outlier infoset (visited O(1e4) times) cannot dominate the MSE loss."""

    def test_advantages_above_1_get_clamped(self) -> None:
        """An advantage of +5.0 must be stored as +1.0 (the saturated ceiling)."""
        from pokerbot.training.replay_buffers import (
            init_advantage_buffers,
            insert_advantage_sample,
            BufferSpec,
        )
        spec = BufferSpec(
            capacity=8,
            fields={
                "info_state": ((1,), jnp.float32),
                "advantages": ((3,), jnp.float32),
                "legal_mask": ((3,), jnp.bool_),
                "iteration": ((), jnp.int32),
            },
        )
        bufs = init_advantage_buffers(spec=spec, num_players=1)
        out = insert_advantage_sample(
            bufs.buffers[0],
            info_state=jnp.array([1.0]),
            advantages=jnp.array([5.0, -10.0, 0.5]),
            legal_mask=jnp.array([True, True, True]),
            iteration=jnp.int32(0),
        )
        assert float(out.data["advantages"][0, 0]) == 1.0
        assert float(out.data["advantages"][0, 1]) == -1.0
        assert abs(float(out.data["advantages"][0, 2]) - 0.5) < 1e-5

    def test_in_range_advantages_pass_through(self) -> None:
        """An advantage of -0.3 stays -0.3 (no false saturation)."""
        from pokerbot.training.replay_buffers import (
            init_advantage_buffers,
            insert_advantage_sample,
            BufferSpec,
        )
        spec = BufferSpec(
            capacity=4,
            fields={
                "info_state": ((1,), jnp.float32),
                "advantages": ((3,), jnp.float32),
                "legal_mask": ((3,), jnp.bool_),
                "iteration": ((), jnp.int32),
            },
        )
        bufs = init_advantage_buffers(spec=spec, num_players=1)
        out = insert_advantage_sample(
            bufs.buffers[0],
            info_state=jnp.array([0.0]),
            advantages=jnp.array([-0.3, 0.0, 0.999]),
            legal_mask=jnp.array([True, True, True]),
            iteration=jnp.int32(0),
        )
        assert abs(float(out.data["advantages"][0, 0]) - (-0.3)) < 1e-5
        assert abs(float(out.data["advantages"][0, 1]) - 0.0) < 1e-5
        assert abs(float(out.data["advantages"][0, 2]) - 0.999) < 1e-5
