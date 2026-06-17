"""Hogwild external-sampling MCCFR on the production NLHE abstraction (H4).

Validates the NLHE sibling of the proven Leduc hogwild path: the shared-table traversal runs and
accumulates a valid average strategy, the street-aware action abstraction is honored, concurrency is
integrity-safe (no overflow / corruption), the tabular blueprint bot is no MORE exploitable than
random, and the net-distillation checkpoint round-trips through the SD-CFR loader.

LOCAL budgets are deliberately tiny: NLHE traversal is heavy eagerly (~2s/deal WITH action
abstraction; ~20s/deal WITHOUT — so these always run abstracted, which is also the intended training
mode). Absolute Nash convergence is the cloud job (H5); the heuristic-exploiter metric here is
relative, used only as a sanity floor.
"""

from __future__ import annotations

import os

# Bounded (Pluribus-style) fingerprint — must be set BEFORE the encoder is imported. This is the mode
# that saturates the infoset count and the one the convergence run uses.
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

import numpy as np
import pytest

from pokerbot.engine.jax_engine import BET_BUCKET_START
from pokerbot.training.sd_cfr_nlhe_hogwild import (
    TabularBlueprintBot,
    run_nlhe_hogwild,
    save_hogwild_nlhe_checkpoint,
)

# NLHE hogwild always runs abstracted (full-action branching is ~10x slower and not the training mode).
_CBS = dict(client_bet_sizing=True)


def _valid_rows(snapshot) -> bool:
    return all(np.all(np.isfinite(r)) and np.all(r >= 0.0) for r in snapshot.values())


def test_traversal_runs_and_accumulates():
    """One small single-worker run occupies infosets, every strat row is finite & non-negative, and
    the table never overflows."""
    res = run_nlhe_hogwild(traversals=12, num_workers=1, capacity=1 << 15, seed=1, **_CBS)
    assert res["n_infosets"] > 0
    assert res["load_factor"] < 1.0
    assert _valid_rows(res["strat"])


def test_action_abstraction_honored():
    """With client bet sizing on, the excluded bucket (0.66x pot = action index 5) must carry ZERO
    average-strategy mass anywhere — proof the street-aware mask is applied identically in traversal."""
    res = run_nlhe_hogwild(traversals=12, num_workers=1, capacity=1 << 15, seed=2, **_CBS)
    excluded = BET_BUCKET_START + 2  # bucket index 2 (0.66x) is in neither pre {4,5,6} nor post {0,1,3,4,5}
    for row in res["strat"].values():
        assert row[excluded] == 0.0


def test_single_worker_deterministic():
    """Same seed + 1 worker ⇒ bit-identical snapshot (deterministic deals + numpy/JAX seeding)."""
    a = run_nlhe_hogwild(traversals=10, num_workers=1, capacity=1 << 15, seed=7, **_CBS)
    b = run_nlhe_hogwild(traversals=10, num_workers=1, capacity=1 << 15, seed=7, **_CBS)
    assert set(a["strat"].keys()) == set(b["strat"].keys())
    for k in a["strat"]:
        assert np.allclose(a["strat"][k], b["strat"][k])


def test_concurrency_integrity():
    """4 workers sharing one table complete error-free, occupy infosets, never overflow, and produce
    only valid rows — the lock-free claims (no lost updates / no duplicate slots) are proven
    rigorously on Leduc in test_hogwild_cfr_leduc.py; here we assert NLHE-level integrity."""
    res = run_nlhe_hogwild(traversals=8, num_workers=4, capacity=1 << 16, seed=11, **_CBS)
    assert res["n_infosets"] > 0
    assert res["load_factor"] < 1.0
    assert _valid_rows(res["strat"])


def test_tabular_bot_not_worse_than_random():
    """The converged tabular policy must be NO MORE exploitable (heuristic ensemble) than uniform
    random — the independent, net-free Nash sanity check (generous slack for the tiny local budget;
    the real convergence proof is the H5 cloud run)."""
    import random

    from pokerbot.eval.bots import RandomBot
    from pokerbot.eval.exploitability import estimate_exploitability

    res = run_nlhe_hogwild(traversals=40, num_workers=2, capacity=1 << 16, seed=3, **_CBS)
    bot = TabularBlueprintBot(res["strat"], client_bet_sizing=True)
    tab_ex = estimate_exploitability(bot, hands=40, seed=42).exploiter_ev_bb_per_hand
    rnd_ex = estimate_exploitability(RandomBot(random.Random(0)), hands=40, seed=42).exploiter_ev_bb_per_hand
    assert tab_ex <= rnd_ex + 0.5, f"tabular {tab_ex:.3f} >> random {rnd_ex:.3f} bb/hand"


def test_persistent_session_runs_and_stops():
    """The persistent-worker session (production cloud path) spawns workers ONCE that traverse
    continuously; the parent snapshots the LIVE table without respawning, then stops cleanly."""
    import time

    from pokerbot.training.sd_cfr_nlhe_hogwild import start_hogwild_session

    s = start_hogwild_session(1 << 15, 2, client_bet_sizing=True, seed=1)
    try:
        deadline = time.time() + 30
        while s.total_traversals() == 0 and time.time() < deadline:
            time.sleep(1)
        assert s.total_traversals() > 0, "persistent workers made no progress"
        assert s.workers_alive() == 2
        snap = s.snapshot()
        assert snap["n_infosets"] > 0 and _valid_rows(snap["strat"])
    finally:
        s.stop()
    assert s.workers_alive() == 0, "workers did not stop cleanly"


def test_reseed_continuation_accumulates():
    """H5 cross-round continuation: seeding round 2 with round 1's snapshot must CARRY round 1's
    infosets and ADD on top (regret continues) — not restart. Strategy-sum is monotone (Σσ, σ≥0), so a
    seeded round's strat ≥ the seed everywhere; and no seeded infoset is dropped."""
    r1 = run_nlhe_hogwild(traversals=10, num_workers=1, capacity=1 << 15, seed=1, **_CBS)
    r2 = run_nlhe_hogwild(
        traversals=10, num_workers=1, capacity=1 << 15, seed=2,
        init_strat=r1["strat"], init_regret=r1["regret"], **_CBS,
    )
    assert set(r2["strat"].keys()) >= set(r1["strat"].keys()), "seeded round dropped infosets"
    for k, row1 in r1["strat"].items():
        assert r2["strat"][k].sum() + 1e-9 >= row1.sum(), "seeded strat-sum decreased (no continuation)"


def test_distill_checkpoint_round_trips(tmp_path):
    """Distillation produces a finite loss and a checkpoint that loads back through the SD-CFR loader
    with the strat_sum keys and completed-iterations intact (machinery built + fidelity-tested here;
    the real distill runs on the converged cloud table at H6)."""
    from pathlib import Path

    from pokerbot.training.sd_cfr_nlhe import (
        SDCFRNLHEConfig,
        SDCFRNLHETrainer,
        load_nlhe_checkpoint,
    )

    res = run_nlhe_hogwild(
        traversals=20, num_workers=1, capacity=1 << 15, reservoir_cap=2000, seed=5, **_CBS
    )
    assert len(res["reservoir"]) > 0

    cfg = SDCFRNLHEConfig(sft_checkpoint=None, policy_hidden=32, client_bet_sizing=True)
    ckpt = Path(tmp_path) / "hogwild_nlhe.pkl"
    save_hogwild_nlhe_checkpoint(res, ckpt, cfg, distill_steps=15, seed=0)
    assert ckpt.exists()

    trainer = SDCFRNLHETrainer(cfg)
    completed = load_nlhe_checkpoint(trainer, ckpt)
    assert completed == 1
    assert set(trainer._strat_sum.keys()) == set(res["strat"].keys())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
