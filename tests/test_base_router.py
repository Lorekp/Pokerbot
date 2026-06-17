"""Pluggable GTO base + table-toughness router (exploitation task (b)).

Covers: (1) the mixer's ``base_probs_override`` seam (None == bit-identical SFT path; an override
replaces the GTO base the read->mixer blends against); (2) ``villain_toughness`` (fish ~0, confirmed
reg high, unknown 0); (3) ``BaseRouter.decide`` gating + monotone exploit damping; (4) HybridAgent
wiring (a reg villain routes to the blueprint base with exploit driven to ~0; default == unchanged).
"""

from __future__ import annotations

import os

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

import numpy as np
import torch

from pokerbot.opponent.base_router import (
    BaseRouter,
    BlueprintBaseProvider,  # noqa: F401  (import smoke — heavy adapter built only in cloud/live)
    villain_toughness,
)
from pokerbot.opponent.mixer import HybridPolicy
from pokerbot.opponent.nash_baseline import nash_reference
from pokerbot.opponent.population_prior import context_prior
from pokerbot.opponent.stats import OpponentStats, OpponentTracker

_CTXS = [f"{st}|{tex}|{fac}"
         for st in ("flop", "turn", "river")
         for tex in ("dry", "wet", "paired")
         for fac in ("unopened", "facing_bet")]


class _StubModel:
    """Returns a fixed GTO policy (a slight check-lean) regardless of input."""

    def __init__(self):
        g = np.array([0.05, 0.40, 0.30, 0.10, 0.05, 0.03, 0.03, 0.02, 0.01, 0.005, 0.005])
        self._g = g / g.sum()

    def __call__(self, *a, **k):
        o = type("_Out", (), {})()
        o.probs = torch.tensor(self._g, dtype=torch.float64).unsqueeze(0)
        return o

    # HybridAgent.__init__ calls model.to(...).eval()
    def to(self, *a, **k):
        return self

    def eval(self):
        return self


def _inputs():
    weva = torch.zeros(1, dtype=torch.long)
    cards = torch.zeros((1, 1, 1), dtype=torch.long)
    actions = torch.zeros((1, 1), dtype=torch.long)
    position = torch.zeros((1, 1), dtype=torch.long)
    legal = torch.ones((1, 11), dtype=torch.bool)
    return weva, cards, actions, position, legal


def _station_profile() -> OpponentStats:
    """Confident over-caller (station): all-passive bins ⇒ far from near-Nash ⇒ exploitable ⇒ low
    toughness."""
    s = OpponentStats()
    s.hands = 200
    for ck in _CTXS:
        for h in range(1, 60):
            s.decay_and_add(ck, [0, 3, 0, 0], context_prior(ck), h)
        s.add_ctx(ck, [0, 60, 0, 0])
    return s


def _reg_profile() -> OpponentStats:
    """Confident near-equilibrium reg: per-context bin counts proportional to the near-Nash reference
    ⇒ tv(read, nash) ~ 0 with high confidence ⇒ high toughness."""
    s = OpponentStats()
    s.hands = 200
    for ck in _CTXS:
        nash = np.asarray(nash_reference(ck), dtype=np.float64)
        bins = [int(round(x * 8)) for x in nash]  # ~8 obs/hand in nash proportions
        for h in range(1, 60):
            s.decay_and_add(ck, bins, context_prior(ck), h)
        s.add_ctx(ck, [b * 8 for b in bins])
    return s


# ---- (1) mixer base_probs_override seam -------------------------------------------------------

def test_override_replaces_gto_base_pure_arm():
    """exploit=False returns the (masked) override base, not the SFT GTO — the protection base path."""
    hp = HybridPolicy(_StubModel(), OpponentTracker(), device="cpu")
    weva, cards, actions, position, legal = _inputs()
    bp = np.zeros(11)
    bp[2] = 1.0  # a blueprint that pure-calls here
    out = hp.get_probs(weva, cards, actions, position, legal, opponent_id=1,
                       exploit=False, base_probs_override=bp)
    assert np.allclose(out, bp, atol=1e-9), "override must drive the pure (exploit-off) base"


def test_override_none_is_identical_sft_path():
    """base_probs_override=None ⇒ the exact pre-existing SFT path (masked GTO)."""
    hp = HybridPolicy(_StubModel(), OpponentTracker(), device="cpu")
    weva, cards, actions, position, legal = _inputs()
    gto = _StubModel()._g
    out = hp.get_probs(weva, cards, actions, position, legal, opponent_id=1,
                       exploit=False, base_probs_override=None)
    assert np.allclose(out, gto, atol=1e-9)


def test_override_is_blended_base_when_exploiting_a_station():
    """With a confident station read, exploiting off the blueprint base shifts aggressive mass UP vs
    the blueprint alone — i.e. the read->mixer blends against the override, not the SFT net."""
    hp = HybridPolicy(_StubModel(), OpponentTracker(), device="cpu")
    weva, cards, actions, position, legal = _inputs()
    bp = np.array([0.10, 0.45, 0.25, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002])
    bp = bp / bp.sum()
    out = hp.get_probs(weva, cards, actions, position, legal, opponent_id=1,
                       opponent_profile=_station_profile(), hand_strength=0.85,
                       board=["Kh", "7d", "2c"], n_prior_aggr=1, exploit=True,
                       base_probs_override=bp)
    assert abs(out.sum() - 1.0) < 1e-6
    assert out[3:].sum() > bp[3:].sum() - 1e-9, "value-bet a station off the blueprint base"


# ---- (2) villain_toughness --------------------------------------------------------------------

def test_toughness_unknown_and_none_are_zero():
    assert villain_toughness(None) == 0.0
    assert villain_toughness(OpponentStats()) == 0.0


def test_toughness_fish_is_low():
    assert villain_toughness(_station_profile()) < 0.2


def test_toughness_reg_is_high():
    t = villain_toughness(_reg_profile())
    assert t > 0.4, f"a confident near-Nash reg should read tough, got {t}"


def test_toughness_reg_exceeds_fish():
    assert villain_toughness(_reg_profile()) > villain_toughness(_station_profile())


# ---- (3) BaseRouter.decide --------------------------------------------------------------------

class _FakeBP:
    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype=np.float64)

    def probs(self, state, seat):
        return self.vec


def test_router_disabled_is_noop():
    r = BaseRouter(_FakeBP(np.ones(11) / 11), enabled=False)
    d = r.decide(_reg_profile())
    assert (d.use_blueprint, d.exploit_scale) == (False, 1.0)


def test_router_fish_keeps_sft_and_full_exploit():
    r = BaseRouter(_FakeBP(np.ones(11) / 11))
    d = r.decide(_station_profile())
    assert d.use_blueprint is False
    assert d.exploit_scale > 0.8


def test_router_reg_routes_to_blueprint_and_kills_exploit():
    r = BaseRouter(_FakeBP(np.ones(11) / 11))
    d = r.decide(_reg_profile())
    assert d.use_blueprint is True
    assert d.exploit_scale == 0.0  # toughness >= tough_high ⇒ no deviation (max protection)


def test_router_without_provider_only_damps():
    r = BaseRouter(blueprint_provider=None)
    d = r.decide(_reg_profile())
    assert d.use_blueprint is False
    assert d.exploit_scale < 0.5  # still protects by damping exploit


def test_router_base_probs_passthrough():
    vec = np.zeros(11)
    vec[2] = 1.0
    r = BaseRouter(_FakeBP(vec))
    assert np.allclose(r.base_probs(None, 0), vec)
    assert BaseRouter(blueprint_provider=None).base_probs(None, 0) is None


# ---- (4) HybridAgent wiring -------------------------------------------------------------------

def _agent():
    from pokerbot.agent import HybridAgent

    return HybridAgent(_StubModel(), device="cpu", use_transformer=False)


def _new_state():
    from pokerbot.engine.game import GameState
    import random

    return GameState.new_hand(button_seat=3, rng=random.Random(123))


def test_agent_default_has_no_router():
    ag = _agent()
    assert ag._base_router is None  # validated SFT vehicle, unchanged


def test_agent_routes_reg_to_blueprint_base():
    """A reg villain ⇒ the agent samples the blueprint base (exploit damped to 0). The blueprint here
    pure-folds; with exploit off and the base swapped, the agent's captured policy must concentrate on
    fold among legal actions — distinct from the SFT stub's check-lean."""
    ag = _agent()
    bp = np.zeros(11)
    bp[0] = 1.0  # blueprint pure-folds
    ag.set_base_router(BaseRouter(_FakeBP(bp)))
    state = _new_state()
    seat = state.action_seat
    opp_id = (seat + 1) % 6
    ag.set_opponent_profiles({opp_id: _reg_profile()})
    ag.reset_hand(seat)
    ag.enable_decision_capture(rng=None)
    ag(state)
    decs = ag.pop_decisions_raw()
    assert decs, "expected a captured decision"
    probs = np.asarray(decs[-1][0], dtype=float)
    # fold may be illegal in the opening spot; if legal, the blueprint base should dominate it.
    if probs[0] > 0:
        assert probs.argmax() == 0, f"blueprint base (pure-fold) should drive the policy, got {probs}"
