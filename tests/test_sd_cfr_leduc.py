"""SD-CFR on Leduc — infoset-BR gate on canonical game tree."""

from __future__ import annotations

import jax
import pytest

from pokerbot.engine.leduc_jax import apply_action, init_hand, legal_actions
from pokerbot.training.leduc_tabular_cfr import (
    LEDUC_SD_CFR_EXPLOITABILITY_MBB,
    LEDUC_TABULAR_EXPLOITABILITY_MBB,
    infoset_exploitability_mbb,
    infoset_exploitability_mbb_from_policy,
    tabular_cfr,
)
from pokerbot.training.sd_cfr import SDCFRConfig, SDCFRTrainer


def test_leduc_init_legal() -> None:
    key = jax.random.PRNGKey(0)
    s = init_hand(key)
    mask = legal_actions(s)
    assert mask.shape == (3,)
    assert bool(mask.any())


def test_leduc_apply_purity() -> None:
    key = jax.random.PRNGKey(1)
    s = init_hand(key)
    s1 = apply_action(s, 1)
    _s2 = apply_action(s, 0)
    assert int(s.hist_len) == 0
    assert int(s1.hist_len) == 1


def test_tabular_cfr_converges_to_equilibrium() -> None:
    avg, _ = tabular_cfr(iterations=1000, seed=0)
    expl, _, _, _, _ = infoset_exploitability_mbb(avg, samples=90)
    assert expl <= LEDUC_TABULAR_EXPLOITABILITY_MBB, (
        f"tabular infoset BR {expl:.2f} mbb exceeds gate {LEDUC_TABULAR_EXPLOITABILITY_MBB}"
    )


@pytest.mark.slow
def test_sd_cfr_leduc_converges_to_equilibrium() -> None:
    avg, _ = tabular_cfr(iterations=2000, seed=0)
    ref_expl, _, _, _, _ = infoset_exploitability_mbb(avg, samples=90)
    cfg = SDCFRConfig(iterations=200, traversals_per_iter=32, seed=2)
    trainer = SDCFRTrainer(cfg)
    trainer.train()
    sd_expl, _, _, _, _ = infoset_exploitability_mbb_from_policy(
        trainer.regret_matching_policy, samples=90
    )
    assert ref_expl <= LEDUC_TABULAR_EXPLOITABILITY_MBB
    assert sd_expl <= LEDUC_SD_CFR_EXPLOITABILITY_MBB
    assert sd_expl >= ref_expl - 1.0
