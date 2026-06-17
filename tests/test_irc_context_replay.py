"""Phase 3 (G3): IRC per-context replay — golden-hand decode against the locked context spec."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "analyze_irc_population", REPO / "scripts" / "analyze_irc_population.py")
irc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(irc)


def test_parse_pdb_line_full():
    # nick ts n_players pos <preflop> <flop> <turn> <river> bankroll wagered won [cards]
    line = "Marzon 766303976 8 1 Bc bc kc kf 12653 300 0"
    nick, ts, pos, streets = irc.parse_pdb_line_full(line)
    assert (nick, ts, pos) == ("Marzon", "766303976", 1)
    assert streets == ["Bc", "bc", "kc", "kf"]
    # preflop-fold row: later streets absent
    nick2, _, _, streets2 = irc.parse_pdb_line_full("jb 766303976 8 4 f 1000 0 0")
    assert nick2 == "jb" and streets2 == ["f"]
    assert irc.parse_pdb_line_full("garbage") is None


def test_parse_hdb_board():
    ts, board = irc.parse_hdb_board("766303976 1 1 8 4/800 4/1600 3/2400 3/3600 3h Kh 5d 8d 5s")
    assert ts == "766303976"
    assert board == ["3h", "Kh", "5d", "8d", "5s"]
    ts2, board2 = irc.parse_hdb_board("766303999 1 2 5 2/200 0 0 0")  # no flop
    assert board2 == []


def test_replay_golden_hand_contexts():
    """3-handed golden hand, hand-verified against the locked spec:

    Preflop (pos order; blinds B are leading chars): p1 'Bc', p2 'Bk' would be no-raise — use a
    raised pot: p1 'Bc' (SB blind, then call), p2 'Bc' (BB blind, call the raise), p3 'r' (raise).
    Sequence round-robin: k=0: p1 B(skip), p2 B(skip), p3 r @aggr0 -> preflop|-|unopened, AGG_SMALL.
    k=1: p1 c @aggr1 -> preflop|-|facing_bet PASSIVE; p2 c @aggr1 -> facing_bet PASSIVE.
    Flop (board Kh7d2c, dry): p1 'k', p2 'b', p3 'f' ... k=0: p1 k @0 -> flop|dry|unopened PASSIVE,
    p2 b @0 -> flop|dry|unopened AGG_SMALL, p3 f @1 -> flop|dry|facing_bet FOLD.
    k=1: p1 'c' @1 -> flop|dry|facing_bet PASSIVE.
    """
    rows = [
        (1, "p1", ["Bc", "kc"]),
        (2, "p2", ["Bc", "b"]),
        (3, "p3", ["r", "f"]),
    ]
    board = ["Kh", "7d", "2c"]
    got = list(irc.replay_hand_contexts(rows, board))
    assert got == [
        ("p3", "preflop|-|unopened", 2),
        ("p1", "preflop|-|facing_bet", 1),
        ("p2", "preflop|-|facing_bet", 1),
        ("p1", "flop|dry|unopened", 1),
        ("p2", "flop|dry|unopened", 2),
        ("p3", "flop|dry|facing_bet", 0),
        ("p1", "flop|dry|facing_bet", 1),
    ]


def test_replay_allin_and_fold_drops_player():
    """All-in maps to AGG_LARGE; a folded player consumes no later-street actions; quit drops."""
    rows = [
        (1, "a", ["Bc", "A"]),
        (2, "b", ["Bf"]),          # folds preflop -> no flop action
        (3, "c", ["c", "c"]),
    ]
    board = ["9h", "9d", "2s"]     # paired texture
    got = list(irc.replay_hand_contexts(rows, board))
    assert ("b", "preflop|-|unopened", 0) in got
    assert ("a", "flop|paired|unopened", 3) in got            # the all-in
    assert ("c", "flop|paired|facing_bet", 1) in got          # calls the jam
    assert not any(n == "b" and ck.startswith("flop") for n, ck, _ in got)


def test_fit_prior_shapes_and_fallback():
    from pokerbot.opponent.context import all_contexts
    from pokerbot.opponent.population_prior import PRIOR_STRENGTH

    ctx_player = {"flop|dry|facing_bet": {f"p{i}": [30, 50, 10, 10] for i in range(40)}}
    ctx_global = {"flop|dry|facing_bet": [300, 500, 100, 100],
                  "river|wet|unopened": [5, 80, 10, 5]}
    prior, pct = irc.fit_prior(ctx_player, ctx_global, PRIOR_STRENGTH)
    assert set(prior.keys()) == set(all_contexts())            # every locked key present
    for ck, row in prior.items():
        assert len(row) == 4 and abs(sum(row) - PRIOR_STRENGTH) < 0.05
    # median fit where players qualify; aggregate fallback elsewhere
    fb = prior["flop|dry|facing_bet"]
    assert abs(fb[1] / PRIOR_STRENGTH - 0.5) < 0.02            # passive 50% from the medians
    assert pct["flop|dry|facing_bet"]["n_players"] == 40
