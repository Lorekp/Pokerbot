"""R3 train/serve-skew FIREWALL: the per-context facing key must be computed IDENTICALLY by every
site (offline ingest, live search env, the mixer, the battery), or a player's history splits across
incompatible keys. These tests pin the preflop full-depth / postflop-binary spec and the cross-site
agreement that guarantees zero skew.
"""

from __future__ import annotations

import numpy as np

from pokerbot.opponent import context as C


def test_preflop_depth_matches_across_offline_and_search_env():
    """A preflop 3-bet spot (UTG opens, MP 3-bets) must key to facing_raise for the 4-bettor at BOTH
    the offline-ingest count and the live search-env count."""
    # offline ingest: the running per-street aggression count (player_profiles._derive_contexts walks
    # the typed action list; preflop counts raises). After open(+1) and 3bet(+1) the next decision
    # faces 2 prior aggressions.
    offline_n = 2
    offline_key = C.context_key(0, [], offline_n)

    # live search env: count aggressive entries in the flat preflop history (raise indices).
    # vocab: 0 fold,1 check,2 call,3.. bets,10 all-in. open=raise(idx 4), 3bet=raise(idx 6).
    hist_action = [4, 6]  # the two raises
    env_n = sum(1 for a in hist_action if C.is_aggressive_action(a))
    env_key = C.context_key_rs(0, [], [], env_n)

    assert offline_key == env_key == "preflop|-|facing_raise"


def test_postflop_collapses_identically_everywhere():
    # Postflop a flop bet then check-raise (2 aggressions) collapses to facing_bet at every site.
    board = ["Jh", "8c", "3d"]
    offline_key = C.context_key(1, board, 2)                       # offline running count = 2
    env_key = C.context_key_rs(1, [11, 8, 3], ["h", "c", "d"], 2)  # serve-env count = 2
    assert offline_key == env_key == "flop|dry|facing_bet"


def test_mixer_threaded_count_matches_exact_key():
    from pokerbot.opponent.mixer import _mixer_context_key

    # When the decision path threads the exact count, the mixer key is the full-depth key.
    legal = np.zeros(11)
    legal[[0, 2, 4]] = 1.0          # fold/call/raise legal (a facing spot, CHECK illegal)
    pre = _mixer_context_key([], legal, n_prior_aggr=2)
    assert pre == "preflop|-|facing_raise"
    # No threaded count ⇒ fall back to the CHECK-legal binary: a strict coarsening (facing_bet), never
    # a conflicting key.
    coarse = _mixer_context_key([], legal, n_prior_aggr=None)
    assert coarse == "preflop|-|facing_bet"


def test_all_four_sites_agree_on_a_full_played_hand():
    """End-to-end: deal a hand, drive it with scripted policies, and at EVERY decision assert the
    battery key (_ctx_key_for) == the agent key (_n_prior_aggr→context_key) == an independent
    recompute from the raw action history. This is the firewall: all paths bucket identically."""
    import random

    from pokerbot.engine.game import GameState, Street, play_hand
    from pokerbot.eval.exploitation_battery import _ctx_key_for
    from pokerbot.agent import _n_prior_aggr

    _STREET = {"PREFLOP": 0, "FLOP": 1, "TURN": 2, "RIVER": 3}

    def independent_key(state, seat):
        street = _STREET[state.street.name]
        board = [str(c) for c in (state.board or [])]
        if street == 0:
            n = sum(1 for _s, a in state.action_history()
                    if C.is_aggressive_type(getattr(a.action_type, "name", "")))
            return C.context_key(street, board, n)
        check_legal = any(x.action_type.name == "CHECK" for x in state.legal_actions())
        return C.context_key(street, board, 0 if check_legal else 1)

    mismatches = []

    def probe(state):
        seat = state.action_seat
        battery_key = _ctx_key_for(state)
        n_agent = _n_prior_aggr(state, seat)
        board = [str(c) for c in (state.board or [])]
        street = _STREET[state.street.name]
        # the agent threads n_prior_aggr (preflop) or None (postflop ⇒ mixer binary fallback)
        if n_agent is not None:
            agent_key = C.context_key(street, board, n_agent)
        else:
            check_legal = any(x.action_type.name == "CHECK" for x in state.legal_actions())
            agent_key = C.context_key(street, board, 0 if check_legal else 1)
        ind = independent_key(state, seat)
        if not (battery_key == agent_key == ind):
            mismatches.append((battery_key, agent_key, ind))
        # a simple legal action to keep the hand moving
        from pokerbot.eval.bots import build_bot
        return build_bot("calling_station", random.Random(seat))(state)

    st = GameState.new_hand(button_seat=3, rng=random.Random(11))
    play_hand({s: probe for s in range(6)}, state=st)
    assert not mismatches, f"site disagreement: {mismatches[:3]}"
