"""Opponent & population analytics (aof_cfr/opponent_analytics.py) — stats 1-6.

Builds synthetic AoF hand records (the same schema the live recorder writes) and asserts the field
reads the exploit engine relies on: fold-to-jam, open/call widths, range shape, timing tells,
population drift, repeat-encounter rate.
"""

from __future__ import annotations

from aof_cfr.opponent_analytics import analyze, render


def _hand(hid, *, ts, dealer, actions, players, hero_seat=0, shown=None, sb=0.5, bb=1.0):
    return {
        "schema_version": 2, "hand_id": hid, "ts": ts,
        "table": {"sb": sb, "bb": bb, "dealer_seat_id": dealer, "game_mode": "aof",
                  "n_players": len(players), "fee_bb": 0.12},
        "hero": {"seat": hero_seat, "name": "subge"},
        "players": players, "actions": actions, "shown": shown or [], "winners": [],
    }


def _p(seat, uid, name, stack=8.0, playing=True):
    return {"seat": seat, "user_id": uid, "name": name, "stack": stack, "playing": playing}


def test_fold_to_jam_and_widths():
    # 4-max, dealer=0 → order BTN(0)=UTG? positions derived by the parser; we drive villain VILLA
    # who, across two hands, OPEN-JAMS once and FOLDS-TO-JAM once.
    players = [_p(0, 100, "subge"), _p(1, 201, "VILLA"), _p(2, 202, "VILLB"), _p(3, 203, "VILLC")]
    # Hand 1: a villain open-jams first-in (everyone else folds behind).
    h1 = _hand("h1", ts="2026-06-10T12:00:00+00:00", dealer=0, players=players, actions=[
        {"street": "preflop", "seat": 3, "user_id": 203, "name": "VILLC", "type": "fold", "dt": 0.8},
        {"street": "preflop", "seat": 0, "user_id": 100, "name": "subge", "type": "fold", "dt": 1.0},
        {"street": "preflop", "seat": 1, "user_id": 201, "name": "VILLA", "type": "all_in", "dt": 0.5},
        {"street": "preflop", "seat": 2, "user_id": 202, "name": "VILLB", "type": "fold", "dt": 6.0},
    ])
    # Hand 2: someone jams ahead and VILLA folds to the jam.
    h2 = _hand("h2", ts="2026-06-11T12:00:00+00:00", dealer=0, players=players, actions=[
        {"street": "preflop", "seat": 3, "user_id": 203, "name": "VILLC", "type": "all_in", "dt": 0.4},
        {"street": "preflop", "seat": 0, "user_id": 100, "name": "subge", "type": "fold", "dt": 1.0},
        {"street": "preflop", "seat": 1, "user_id": 201, "name": "VILLA", "type": "fold", "dt": 5.0},
        {"street": "preflop", "seat": 2, "user_id": 202, "name": "VILLB", "type": "fold", "dt": 0.9},
    ])
    an = analyze([h1, h2])
    a = an.opps["201"]
    assert a.name == "VILLA"
    # VILLA: open-jammed once (first-in), folded to a jam once
    assert a.open == [1, 1] and a.open_jam_freq() == 1.0
    assert a.facing_jam == [1, 0] and a.fold_to_jam() == 1.0 and a.call_vs_jam_freq() == 0.0
    # population sees both villains' decisions
    assert an.pop.decisions >= 4
    # repeat-encounter: VILLA seen on 2 distinct days
    assert len(a.days) == 2


def test_range_shape_from_shown_and_timing():
    players = [_p(0, 100, "subge"), _p(1, 201, "VILLA"), _p(2, 202, "VILLB")]
    # 3-max; VILLA open-jams and shows a hand → a range-shape data point.
    h = _hand("h3", ts="2026-06-12T12:00:00+00:00", dealer=0, players=players, hero_seat=0, actions=[
        {"street": "preflop", "seat": 0, "user_id": 100, "name": "subge", "type": "fold", "dt": 1.0},
        {"street": "preflop", "seat": 1, "user_id": 201, "name": "VILLA", "type": "all_in", "dt": 0.3},
        {"street": "preflop", "seat": 2, "user_id": 202, "name": "VILLB", "type": "fold", "dt": 7.0},
    ], shown=[{"seat": 1, "user_id": 201, "name": "VILLA", "cards": ["Ah", "Kd"]}])
    an = analyze([h])
    a = an.opps["201"]
    # AKo entered the open-jam shape bucket as a jam
    assert any(cnt[1] >= 1 for cnt in a.shape_open.values())
    # timing captured: a fast jam + a slow fold somewhere in the population
    assert an.pop.t_jam and an.pop.t_fold
    assert min(an.pop.t_jam) < 1.5     # snap jam present
    assert max(an.pop.t_fold) > 4.0    # tank fold present


def test_render_smoke_and_drift():
    players = [_p(0, 100, "subge"), _p(1, 201, "VILLA"), _p(2, 202, "VILLB")]
    recs = []
    for i, day in enumerate(("2026-06-10", "2026-06-17")):  # two ISO weeks → drift table
        recs.append(_hand(f"d{i}", ts=f"{day}T12:00:00+00:00", dealer=0, players=players, actions=[
            {"street": "preflop", "seat": 0, "user_id": 100, "name": "subge", "type": "fold"},
            {"street": "preflop", "seat": 1, "user_id": 201, "name": "VILLA", "type": "all_in"},
            {"street": "preflop", "seat": 2, "user_id": 202, "name": "VILLB", "type": "fold"},
        ]))
    an = analyze(recs)
    assert len(an.drift) == 2
    text = render(an, min_dec=1)
    assert "POPULATION" in text and "DRIFT" in text and "REPEAT ENCOUNTERS" in text
