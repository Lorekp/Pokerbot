"""Leduc game identity: JAX canonical path vs Python bridge must agree on all reachable states."""

from __future__ import annotations

from itertools import permutations

import pytest

from pokerbot.engine.leduc_bridge import (
    apply_action_py,
    collect_reachable_infosets,
    jax_state_fingerprint,
    py_fingerprint,
    py_legal,
    py_state_from_deal,
)
from pokerbot.engine.leduc_jax import CALL, NUM_ACTIONS, apply_action, legal_actions
from pokerbot.training.leduc_collision_check import require_no_leduc_encode_collisions
from pokerbot.training.leduc_tabular_cfr import (
    LEDUC_SD_CFR_EXPLOITABILITY_MBB,
    LEDUC_TABULAR_EXPLOITABILITY_MBB,
    infoset_exploitability_mbb,
    infoset_exploitability_mbb_from_policy,
    openspiel_nash_conv_uniform_mbb,
    tabular_cfr,
)
from pokerbot.training.sd_cfr import SDCFRConfig, SDCFRTrainer


def _all_deals() -> list[tuple[tuple[int, int], int]]:
    deck = (0, 0, 1, 1, 2, 2)
    return [((p[0], p[1]), p[2]) for p in sorted(set(permutations(deck)))]


def _walk_py_jax_parallel(root) -> None:
    """Every reachable node: bridge transition must match direct JAX apply_action."""
    stack = [root]
    seen: set[tuple] = set()
    while stack:
        state = stack.pop()
        fp = py_fingerprint(state)
        if fp in seen:
            continue
        seen.add(fp)
        assert jax_state_fingerprint(state.jax)[:13] == fp
        if state.done:
            continue
        mask = py_legal(state)
        for a in range(NUM_ACTIONS):
            if mask[a]:
                py_child = apply_action_py(state, a)
                jax_child = apply_action(state.jax, a)
                assert py_fingerprint(py_child) == jax_state_fingerprint(jax_child)[:13]
                stack.append(py_child)


def test_deal_public_preserved_after_private_round() -> None:
    """Distinct deals must deal distinct public cards — cache must not collapse them."""
    s0 = apply_action_py(apply_action_py(py_state_from_deal((0, 1), 0), CALL), CALL)
    s1 = apply_action_py(apply_action_py(py_state_from_deal((0, 1), 1), CALL), CALL)
    s2 = apply_action_py(apply_action_py(py_state_from_deal((0, 1), 2), CALL), CALL)
    assert (s0.public, s1.public, s2.public) == (0, 1, 2)


def test_hand_traced_payoff_matches_code() -> None:
    """Deal (0,1)/pub=2, private CC, public CC — P1 wins at showdown (reward -1 for P0)."""
    root = py_state_from_deal((0, 1), 2)
    s = apply_action_py(apply_action_py(root, CALL), CALL)
    assert s.public == 2
    s = apply_action_py(s, CALL)
    s = apply_action_py(s, CALL)
    assert s.done
    assert s.rewards == (-1.0, 1.0)


def test_py_jax_reachable_fingerprints_match() -> None:
    for cards, pub in _all_deals()[:10]:
        _walk_py_jax_parallel(py_state_from_deal(cards, pub))


def test_leduc_infoset_count_sane() -> None:
    """Canonical Leduc tree has hundreds of infosets, not the collapsed ~18 from the old bug."""
    deals = _all_deals()
    roots = [py_state_from_deal(c, p) for c, p in deals]
    infosets = collect_reachable_infosets(roots)
    assert len(infosets) >= 100, f"expected full public-betting tree (got {len(infosets)} infosets)"


def test_leduc_encode_no_collisions() -> None:
    result = require_no_leduc_encode_collisions(_all_deals())
    assert result["n_infosets"] == result["n_unique_encodes"]
    assert result["n_infosets"] >= 100


def test_tabular_cfr_converges_to_near_zero_exploitability() -> None:
    """Converged tabular CFR on correct tree must have infoset BR ≈ 0."""
    avg, _ = tabular_cfr(iterations=1000, seed=0)
    expl, u0, u1, br0, br1 = infoset_exploitability_mbb(avg, samples=90)
    os_anchor = openspiel_nash_conv_uniform_mbb()
    print(
        f"\nTabular CFR: infoset BR={expl:.2f} mbb (gate {LEDUC_TABULAR_EXPLOITABILITY_MBB}), "
        f"u0={u0:.4f} u1={u1:.4f} br0={br0:.4f} br1={br1:.4f}, "
        f"OpenSpiel uniform nash_conv={os_anchor:.1f} mbb (reference anchor)"
    )
    assert expl <= LEDUC_TABULAR_EXPLOITABILITY_MBB, (
        f"tabular infoset BR {expl:.2f} mbb exceeds gate {LEDUC_TABULAR_EXPLOITABILITY_MBB}"
    )


@pytest.mark.slow
def test_sd_cfr_leduc_infoset_br_gate() -> None:
    """SD-CFR with history encoding: tabular ≈ 0, SD-CFR < 100 mbb, SD-CFR >= tabular."""
    avg, _ = tabular_cfr(iterations=1000, seed=0)
    tab_expl, _, _, _, _ = infoset_exploitability_mbb(avg, samples=90)

    cfg = SDCFRConfig(iterations=200, traversals_per_iter=32, seed=2)
    trainer = SDCFRTrainer(cfg)
    trainer.train(log_every=50)
    sd_expl, u0, u1, br0, br1 = infoset_exploitability_mbb_from_policy(
        trainer.regret_matching_policy, samples=90
    )
    print(
        f"\nLeduc gate: tabular infoset BR={tab_expl:.2f} mbb, "
        f"SD-CFR infoset BR={sd_expl:.2f} mbb (gate {LEDUC_SD_CFR_EXPLOITABILITY_MBB}), "
        f"SD u0={u0:.4f} u1={u1:.4f}"
    )
    assert tab_expl <= LEDUC_TABULAR_EXPLOITABILITY_MBB
    assert sd_expl <= LEDUC_SD_CFR_EXPLOITABILITY_MBB
    assert sd_expl >= tab_expl - 1.0, "neural policy should not beat tabular on infoset BR"
