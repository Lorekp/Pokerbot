"""Foundation smoke + unit tests for the human-realistic tournament self-play package.

Covers the four deliverable pieces:
  1. BC clone model + data pipeline (engine-smoke shard -> dataset -> forward -> train step).
  2. piKL operator: the BINDING limits (λ→∞ == τ, λ=0 == argmax(Q)) + the monotone KL leash.
  3. League/exploiter manager: add / prune-to-cap / PFSP / frozen-anchor assert / serialization.
  4. Tournament-self-play env interface: a round over the REFERENCE (mtt_lifecycle) backend with
     $-ROI CV reward, and the Rust-sim protocol contract (StubLeagueMatch implements it).

Run (WSL jaxgpu-venv):
    pytest -p no:cacheprovider tests/test_tourney_selfplay_foundation.py -q
"""

from __future__ import annotations

import random
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

warnings.filterwarnings("ignore")  # silence the inherited torch mask-deprecation warning


# -- 2. piKL operator (the strength dial) ----------------------------------------------------------


def test_pikl_lambda_inf_reproduces_anchor():
    from pokerbot.training.tourney_selfplay import pikl

    rng = np.random.default_rng(0)
    q = rng.standard_normal(6)
    tau = rng.standard_normal(6)
    legal = np.array([True, True, False, True, True, False])

    pi_inf = pikl.pikl_policy(q, tau, legal, pikl.LAMBDA_FAITHFUL)
    tau_only = pikl.pikl_policy(np.zeros(6), tau, legal, pikl.LAMBDA_FAITHFUL)
    assert np.allclose(pi_inf, tau_only, atol=1e-9)
    assert abs(pi_inf.sum() - 1.0) < 1e-9
    assert pi_inf[2] == 0.0 and pi_inf[5] == 0.0  # illegal exactly zero


def test_pikl_lambda_zero_reproduces_argmax_q():
    from pokerbot.training.tourney_selfplay import pikl

    rng = np.random.default_rng(1)
    q = rng.standard_normal(6)
    tau = rng.standard_normal(6)
    legal = np.array([True, False, True, True, True, True])

    pi0 = pikl.pikl_policy(q, tau, legal, 0.0, floor_lambda=False)
    legal_idx = np.flatnonzero(legal)
    best = legal_idx[np.argmax(q[legal_idx])]
    assert pi0[best] == 1.0 and pi0.sum() == 1.0


def test_pikl_kl_leash_monotone_in_lambda():
    """Smaller λ ⇒ larger D_KL(π‖τ) (the strength dial pulls further from the human anchor)."""
    from pokerbot.training.tourney_selfplay import pikl

    rng = np.random.default_rng(2)
    q = rng.standard_normal(5) * 2.0
    tau = rng.standard_normal(5)
    legal = np.ones(5, dtype=bool)
    lambdas = [1.0, 1e-1, 1e-2, 1e-3]
    kls = [
        pikl.kl_to_anchor(pikl.pikl_policy(q, tau, legal, L), tau, legal) for L in lambdas
    ]
    assert all(kls[i] <= kls[i + 1] + 1e-9 for i in range(len(kls) - 1)), kls


def test_pikl_mixture_is_valid_distribution():
    from pokerbot.training.tourney_selfplay import pikl

    q = np.array([0.0, 1.0, 2.0, 0.5])
    tau = np.array([0.2, 0.8, 0.1, 0.3])
    legal = np.array([True, True, True, False])
    p = pikl.pikl_mixture(q, tau, legal, pikl.TIER_LAMBDA_SETS["modern_pro"])
    assert abs(p.sum() - 1.0) < 1e-9 and p[3] == 0.0


def test_pikl_index_mismatch_raises():
    from pokerbot.training.tourney_selfplay import pikl

    with pytest.raises(ValueError):
        pikl.pikl_policy([0.0, 1.0], [0.0, 1.0, 2.0], [True, True], 0.1)


# -- 1. BC clone model + data pipeline -------------------------------------------------------------


def _small_clone():
    from pokerbot.training.tourney_selfplay.human_clone import HumanCloneConfig, HumanCloneNet

    return HumanCloneNet(HumanCloneConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128))


def test_clone_forward_and_skill_conditioning():
    import torch

    from pokerbot.training.tourney_selfplay import clone_data as cd
    from pokerbot.training.tourney_selfplay.human_clone import NUM_SKILL_BINS

    net = _small_clone()
    tmp = Path(tempfile.mkdtemp())
    cd.build_engine_smoke_shard(tmp, n_hands=16, seed=1)
    ds = cd.SkillTrajectoryDataset(str(tmp / "*.parquet"))
    assert len(ds) > 0
    batch = ds.collate([ds[i] for i in range(min(8, len(ds)))])
    out = net(
        batch["token_types"], batch["token_features"], batch["attn_mask"], batch["hero_seat"],
        skill_bins=batch["skill_bins"], table_skill=batch["table_skill"],
    )
    assert out["policy_logits"].shape[-1] == 11
    # skill conditioning changes the policy logits (the new mechanism is wired)
    lo = net(batch["token_types"], batch["token_features"], batch["attn_mask"], batch["hero_seat"],
             skill_bins=torch.zeros_like(batch["skill_bins"]), table_skill=torch.zeros_like(batch["table_skill"]))
    hi = net(batch["token_types"], batch["token_features"], batch["attn_mask"], batch["hero_seat"],
             skill_bins=torch.full_like(batch["skill_bins"], NUM_SKILL_BINS - 1),
             table_skill=torch.full_like(batch["table_skill"], 9))
    assert (lo["policy_logits"] - hi["policy_logits"]).abs().mean().item() > 1e-4


def test_clone_train_step_and_match_rate():
    from pokerbot.training.tourney_selfplay import clone_data as cd
    from pokerbot.training.tourney_selfplay.human_anchor import (
        CloneTrainConfig,
        top1_match_rate,
        train_clone,
    )

    tmp = Path(tempfile.mkdtemp())
    cd.build_engine_smoke_shard(tmp, n_hands=48, seed=2)
    ds = cd.SkillTrajectoryDataset(str(tmp / "*.parquet"))
    model, state = train_clone(
        ds, CloneTrainConfig(epochs=2, batch_size=16, log_every=1, max_steps=5), model=_small_clone()
    )
    assert state.step == 5 and state.total_loss > 0
    mr = top1_match_rate(model, ds, max_batches=2)
    assert 0.0 <= mr["overall"] <= 1.0 and mr["n"] > 0


def test_skill_score_ranks_tag_above_station():
    from pokerbot.training.tourney_selfplay.clone_data import composite_skill_score, skill_bin

    tag = composite_skill_score(0.22, 0.18, 2.8)
    station = composite_skill_score(0.55, 0.06, 0.5)
    assert tag > station and skill_bin(tag) > skill_bin(station)
    # showdown signal separates a winning from a losing same-style player
    assert composite_skill_score(0.3, 0.2, 2.0, showdown_bb100=15.0) > composite_skill_score(
        0.3, 0.2, 2.0, showdown_bb100=-15.0
    )


# -- 3. frozen-anchor invariant + league -----------------------------------------------------------


def test_frozen_anchor_assert():
    import torch

    from pokerbot.training.tourney_selfplay.human_anchor import (
        assert_anchor_frozen,
        load_frozen_anchor,
    )

    anchor = load_frozen_anchor(model=_small_clone())
    assert_anchor_frozen(anchor)  # frozen, no optimizer
    other = _small_clone()
    opt = torch.optim.AdamW(other.parameters(), lr=1e-3)
    assert_anchor_frozen(anchor, opt)  # anchor absent from opt -> OK
    # an unfrozen model fails the requires_grad check
    with pytest.raises(AssertionError):
        assert_anchor_frozen(other)


def test_league_add_prune_pfsp_and_roundtrip():
    from pokerbot.training.tourney_selfplay.league import League, Role
    from pokerbot.training.tourney_selfplay.league_match import MemberSpec, StubLeagueMatch

    lg = League(cap=6, rng=random.Random(0))
    hero = lg.add_member(Role.MAIN, state_id="hero")
    lg.seed_human_floor({"fish": 1e9, "weak_reg": 0.2, "strong_reg": 0.05})
    backend = StubLeagueMatch()
    hero_spec = MemberSpec(hero.member_id, decide_fn=None, is_hero=True, metadata={"strength": 0.9})
    for m in lg.by_role(Role.FROZEN_HUMAN):
        spec = MemberSpec(m.member_id, decide_fn=None, metadata={"strength": 0.4})
        res = backend.play_match({0: hero_spec, 1: spec}, 10, 1, payouts=[100, 60, 40])
        roi = res.mean_cv(spec.member_id)
        lg.record_result(hero.member_id, m.member_id, roi, win=roi > 0)
    w = lg.pfsp_weights(hero.member_id, lg.by_role(Role.FROZEN_HUMAN), mode="f_hard")
    assert abs(sum(w) - 1.0) < 1e-9
    for i in range(8):
        lg.snapshot_hero(hero.member_id, state_id=f"snap{i}")
    pruned = lg.prune_to_cap()
    assert len(lg.members) <= 6 and len(pruned) > 0
    assert any(m.role == Role.MAIN for m in lg.members)
    assert any(m.role == Role.FROZEN_HUMAN for m in lg.members)
    lg2 = League.from_dict(lg.to_dict())
    assert len(lg2.members) == len(lg.members)


def test_lifted_agent_fish_is_tau_pro_is_strengthened():
    from pokerbot.training.tourney_selfplay.pikl_lift import LiftedAgent

    A = 4
    tau_fn = lambda obs: np.array([0.1, 2.0, 0.5, 0.3])  # noqa: E731
    q_fn = lambda obs: np.array([0.0, 0.0, 3.0, 0.0])  # noqa: E731
    legal = [True, True, True, True]
    fish = LiftedAgent.from_tier("fish", tau_fn, A, q_fn=q_fn)
    pro = LiftedAgent.from_tier("modern_pro", tau_fn, A, q_fn=q_fn)
    pf = fish.policy(None, legal)
    pp = pro.policy(None, legal)
    assert int(pf.argmax()) == 1            # fish == τ (favors action 1)
    assert pp[2] > pf[2]                     # pro pulled toward the high-Q action 2


# -- 4. tournament-self-play env interface ---------------------------------------------------------


def test_reference_backend_implements_protocol():
    from pokerbot.training.tourney_selfplay.league_match import (
        LeagueMatchBackend,
        ReferenceLeagueMatch,
        StubLeagueMatch,
    )

    assert isinstance(ReferenceLeagueMatch(), LeagueMatchBackend)
    assert isinstance(StubLeagueMatch(), LeagueMatchBackend)


def test_env_round_over_reference_lifecycle_backend():
    from pokerbot.tournament import mtt_lifecycle as life
    from pokerbot.training.tourney_selfplay.league import League, Role
    from pokerbot.training.tourney_selfplay.league_match import MemberSpec, ReferenceLeagueMatch
    from pokerbot.training.tourney_selfplay.tournament_reward import returns_and_advantages
    from pokerbot.training.tourney_selfplay.tourney_env import TourneySelfPlayEnv

    lg = League(cap=20, rng=random.Random(0))
    hero = lg.add_member(Role.MAIN, state_id="hero")
    lg.seed_human_floor({"fish": 1e9, "weak_reg": 0.2})
    hero_fn = life.icm_aware(jam=0.42, call=0.55)
    field_fn = life.chip_ev_nash(jam=0.50, call=0.62)
    hero_spec = MemberSpec(hero.member_id, decide_fn=hero_fn, is_hero=True)
    builder = lambda mem: MemberSpec(mem.member_id, decide_fn=field_fn)  # noqa: E731

    env = TourneySelfPlayEnv(
        lg, backend=ReferenceLeagueMatch(seats_per_table=6), payouts=[100, 60, 40], n_players=9
    )
    roll = env.collect_round(hero_spec, builder, n_matches=3, seed=7, n_opponents=5)
    assert roll.n_matches == 3 and len(roll.cv_rewards) > 0
    vals = [0.0] * len(roll.cv_rewards)
    dones = [False] * (len(roll.cv_rewards) - 1) + [True]
    returns, adv = returns_and_advantages(roll.cv_rewards, vals, dones)
    assert len(returns) == len(roll.cv_rewards) == len(adv)
    # the anti-drift metric runs (hero vs the FROZEN_HUMAN floor)
    floor_roll = env.eval_vs_human_floor(hero_spec, builder, n_matches=2, seed=11)
    assert floor_roll.n_matches == 2


def test_tournament_reward_cv_identity():
    """The CV identity (design risk #6): realized_$ == cv_reward + ICM(pre)[hero].

    ``cv_reward`` subtracts the pre-hand ICM baseline from the realized $-EV (the variance-reduction
    levers in ``eval_score.score_with_cv``); adding the baseline back recovers the realized $-EV
    exactly, so the CV is mean-preserving (never biasing). On a no-op hand the realized $-EV is 0, so
    cv_reward == -ICM(pre)[hero] (the residual carries no spurious signal beyond the baseline).
    """
    from pokerbot.training.tourney_selfplay.tournament_reward import (
        HandOutcome,
        cv_baseline,
        cv_reward,
        hand_dollar_reward,
    )

    pre = [1200.0, 1000.0, 800.0]
    out = HandOutcome(pre_stacks=pre, chip_deltas=[100.0, -60.0, -40.0], payouts=[100, 60, 40], hero=0)
    realized = hand_dollar_reward(out)
    base = cv_baseline(pre, [100, 60, 40], 0)
    assert abs((cv_reward(out) + base) - realized) < 1e-9
    # no-op hand: realized 0 -> cv_reward == -baseline
    noop = HandOutcome(pre_stacks=pre, chip_deltas=[0.0, 0.0, 0.0], payouts=[100, 60, 40], hero=0)
    assert abs(hand_dollar_reward(noop)) < 1e-9
    assert abs(cv_reward(noop) + cv_baseline(pre, [100, 60, 40], 0)) < 1e-9
