"""Table-aggression kill-switch: disable the whole exploit when >= AGG_KILL_COUNT seated villains are
confirmed LOOSE-aggressive (lags). Must fire on a lag-dominated table (the -0.48 leak) but NOT on a tag
("pro") table (the +0.99 gain we keep) -- the classifier keys on loose aggression (raises AND folds
little), which separates a lag from a tag.
"""

from __future__ import annotations

from pokerbot.opponent.population_prior import context_prior
from pokerbot.opponent.root_mixer import (AGG_KILL_COUNT, _is_dangerous_aggressor,
                                          _table_too_aggressive)
from pokerbot.opponent.stats import OpponentStats

_ST = ("flop", "turn", "river")
_TEX = ("dry", "wet", "paired")


def _profile(facing_bins, unopened_bins, hands=200):
    prof = OpponentStats()
    prof.hands = hands
    for st in _ST:
        for tex in _TEX:
            for fac, bins in (("unopened", unopened_bins), ("facing_bet", facing_bins)):
                ck = f"{st}|{tex}|{fac}"
                for h in range(1, 50):
                    prof.decay_and_add(ck, bins, context_prior(ck), h)
                prof.add_ctx(ck, [b * 24 for b in bins])
    return prof


def _lag():      # raises facing bets, never folds (loose) -> dangerous
    return _profile([0, 0, 1, 1], [0, 0, 1, 1])


def _tag():      # raises for value BUT folds tight -> aggressive yet NOT loose -> not dangerous
    return _profile([3, 1, 1, 1], [1, 1, 1, 0])


def _station():  # passive caller -> not aggressive
    return _profile([0, 1, 0, 0], [0, 1, 0, 0])


def _nit():      # folds a lot -> not loose, not aggressive
    return _profile([1, 0, 0, 0], [0, 1, 0, 0])


def _unknown():
    return _profile([0, 0, 1, 1], [0, 0, 1, 1], hands=10)


# ---- classifier: lag vs tag is the crux ----

def test_lag_is_dangerous_tag_is_not():
    assert _is_dangerous_aggressor(_lag()) is True
    assert _is_dangerous_aggressor(_tag()) is False   # keeps the +0.99 vs pros
    assert _is_dangerous_aggressor(_station()) is False
    assert _is_dangerous_aggressor(_nit()) is False
    assert _is_dangerous_aggressor(_unknown()) is False
    assert _is_dangerous_aggressor(None) is False


# ---- the table kill-switch ----

def test_all_lag_table_triggers_kill():
    profs = {s: _lag() for s in range(1, 6)}          # 5 lags
    assert _table_too_aggressive(profs) is True


def test_all_tag_pro_table_does_not_kill():
    profs = {s: _tag() for s in range(1, 6)}          # 5 tags -> exploit stays ON (+0.99 kept)
    assert _table_too_aggressive(profs) is False


def test_two_lags_below_threshold_keeps_exploit():
    profs = {1: _lag(), 2: _lag(), 3: _tag(), 4: _tag(), 5: _nit()}  # 2 lags < 3 (irc_tough -> +0.71)
    assert _table_too_aggressive(profs) is False


def test_three_lags_triggers_kill():
    profs = {1: _lag(), 2: _lag(), 3: _lag(), 4: _station(), 5: _station()}
    assert _table_too_aggressive(profs) is True


def test_cold_start_and_none_do_not_kill():
    assert _table_too_aggressive({s: _unknown() for s in range(1, 6)}) is False
    assert _table_too_aggressive(None) is False
    assert _table_too_aggressive({}) is False


def test_threshold_constant_is_three():
    assert AGG_KILL_COUNT == 3


# ---- AUDIT wiup89smb fixes: maniac-FISH vs skilled-LAG (showdown discriminator) ----

def _maniac_fish():
    # count-aggressive (raises facing bets, never folds — like a lag) BUT a weak-showdown server HUD
    # (WSD 0.38 < break-even) -> a loose-aggressive MANIAC FISH that PAYS OFF, not a skilled LAG.
    prof = _lag()
    prof.hud = {"vpip": 62, "pfr": 45, "wsd": 38, "foldToCbet": 22, "wtsd": 42}
    return prof


def _skilled_lag():
    # count-aggressive AND wins at showdown (WSD 0.54 >= break-even) -> a real LAG, must STILL be dangerous.
    prof = _lag()
    prof.hud = {"vpip": 30, "pfr": 26, "wsd": 54, "foldToCbet": 48, "wtsd": 28}
    return prof


def test_maniac_fish_excused_skilled_lag_still_dangerous():
    # finding #1 (HIGH): a confident weak-showdown maniac is a FISH, NOT a dangerous LAG -> exploit it.
    assert _is_dangerous_aggressor(_maniac_fish()) is False
    # a real skilled LAG (wins at showdown) is STILL dangerous (kill-switch preserved vs regs/pros).
    assert _is_dangerous_aggressor(_skilled_lag()) is True
    # a HUD-less loose-aggressor (no showdown signal) is still dangerous (veto fails CLOSED).
    assert _is_dangerous_aggressor(_lag()) is True


def test_maniac_fish_table_not_killed():
    # 3 maniac fish must NOT trip the kill-switch — we WANT to crush them (the headline value fix).
    assert _table_too_aggressive({s: _maniac_fish() for s in range(1, 4)}) is False
    # 3 real skilled LAGs still trip it (protect-over-extract preserved).
    assert _table_too_aggressive({s: _skilled_lag() for s in range(1, 4)}) is True


def test_partial_hud_loose_passive_admitted_as_fish():
    # finding #5: a loose-passive station on a PARTIAL play-money HUD (vpip/pfr only, no wsd/foldToCbet)
    # is admitted as a value-fish (whole-history vpip/pfr are high-confidence), so the value lever fires.
    from pokerbot.opponent.root_mixer import _is_value_fish
    from pokerbot.opponent.stats import OpponentStats

    assert _is_value_fish(OpponentStats(hands=0, hud={"vpip": 60, "pfr": 10})) is True   # wide gap = station
    # do-no-harm: a loose-AGGRESSIVE maniac (small gap) on a partial HUD is NOT admitted here.
    assert _is_value_fish(OpponentStats(hands=0, hud={"vpip": 62, "pfr": 48})) is False


def test_fish_relax_vetoes_hud_fish_that_is_count_aggressor(monkeypatch):
    # finding #8: a HUD-loose-passive villain who is ALSO locally count-aggressive must NOT collapse the
    # RNR N to the HU ceiling (a real punisher is present). fish_relaxed_n now requires not-dangerous too.
    from pokerbot.opponent.rnr_cap import fish_relaxed_n

    monkeypatch.setenv("POKERBOT_RNR_FISH_RELAX", "1")

    def _hud_fish_but_count_aggressor():
        prof = _lag()                                          # count-aggressive (raises, never folds)
        prof.hud = {"vpip": 55, "pfr": 10, "wsd": 54}          # wide gap -> _is_value_fish; wsd 0.54 not weak
        return prof

    model = {s: _hud_fish_but_count_aggressor() for s in range(1, 6)}
    assert fish_relaxed_n(6, model) == 6                       # a punisher present -> NOT relaxed to HU(2)


# ---- end-to-end: the kill-switch actually disables the exploit in SearchAgent (deterministic, no EV) ----

def _facing_probs(agent, seed):
    """Drive a facing-a-raise decision and return the hero's action probs. button=0 => seat 3 acts first
    preflop; it raises, 4/5 fold => hero (seat 0) faces a raise (aggressor = seat 3)."""
    import numpy as np
    import random as _r

    from pokerbot.engine.actions import Action, ActionType
    from pokerbot.engine.game import GameState

    st = GameState.new_hand(button_seat=0, rng=_r.Random(seed))
    st.deal_hole_cards()
    for _ in range(6):
        if st.action_seat == 0:
            break
        legal = st.legal_actions()
        raise_a = next((a for a in legal if a.action_type == ActionType.RAISE), None)
        st.apply_action(raise_a if (st.action_seat == 3 and raise_a) else Action(ActionType.FOLD))
        st.advance_to_next_actor()
    if st.action_seat != 0:
        return None
    agent.enable_decision_capture(_r.Random(1))
    agent.reset_hand(0)
    agent(st)
    return np.asarray(agent.pop_decisions_raw()[-1][0], dtype=float)


def _make_agent():
    from pokerbot.eval.search_agent import SearchAgent

    ag = SearchAgent(use_pluribus_search=True, blueprint_snapshot={}, hero_seat=0,
                     time_budget_ms=10 ** 9, transformer_checkpoint=None, root_exploit_mixer=True,
                     root_mixer_unopened=True)
    ag.solver.config.max_iterations = 4
    ag.solver.config.leaf = type(ag.solver.config.leaf)(select="uniform", rollout_samples=1)
    return ag


def test_searchagent_kill_switch_zeroes_exploit_on_lag_table():
    """A 5-lag table trips the kill-switch -> the hero plays the PURE base (== no-profile agent) on every
    decision, so the exploit's harm vs a lag-dominated table is zero by construction (no EV run needed).
    A 2-lag table is below threshold -> the exploit still fires (output shifts on >=1 seed), proving the
    equality above isn't trivial."""
    import numpy as np

    base = _make_agent()
    base.set_opponent_profiles(None)
    killed = _make_agent()
    killed.set_opponent_profiles({s: _lag() for s in range(1, 6)})   # 5 lags -> kill
    for seed in range(8):
        pb, pk = _facing_probs(base, seed), _facing_probs(killed, seed)
        if pb is None or pk is None:
            continue
        assert np.allclose(pb, pk), f"kill-switch did not fully disable the exploit (seed {seed})"

    active = _make_agent()
    active.set_opponent_profiles({1: _lag(), 3: _lag()})             # 2 lags -> below threshold
    shifted = False
    for seed in range(12):
        pb, pa = _facing_probs(base, seed), _facing_probs(active, seed)
        if pb is None or pa is None:
            continue
        if not np.allclose(pb, pa):
            shifted = True
            break
    assert shifted, "below-threshold exploit never fired -> the kill-equality test would be vacuous"
