"""Deferred pre-smoke audit items (zero smoke impact, eval/hygiene fidelity):
  O4  — name<->uid reconciliation in PlayerProfileStore (no split-brain profile per player).
  SE5 — game.py odd-chip side-pot split routes the indivisible chip left-of-button (matches JAX).
  SE8 — native_solver's PROCESS-GLOBAL enable flag warns once when an explicit policy is clobbered.
"""

from __future__ import annotations

import warnings

import pytest

from pokerbot.engine.cards import Card
from pokerbot.engine.game import GameState, PlayerState, Position, Street


# --------------------------------------------------------------------------- SE5
_STRAIGHT = ["2c", "3d", "4h", "5s", "6c"]  # 6-high straight ON the board -> both blanks tie


def _c(s):
    return Card.from_str(s)


def _mk(seat, pos, ct, hole=None, folded=False):
    p = PlayerState(seat=seat, position=pos, stack_bb=0.0,
                    hole=(_c(hole[0]), _c(hole[1])) if hole else None, folded=folded)
    p.committed_total_bb = ct
    return p


def _odd_chip_state(button_seat):
    """Two tied winners (0,1) + a folded dead contributor (2) make the main layer an odd cent count
    (0.15bb = 15 chips over 2 winners -> 7,7 + 1 remainder)."""
    st = GameState()
    st.board = [_c(x) for x in _STRAIGHT]
    st.street = Street.SHOWDOWN
    st.button_seat = button_seat
    st.players = [
        _mk(0, Position.BTN, 0.10, ("Kc", "Qd")),   # plays the board straight (tie)
        _mk(1, Position.SB, 0.10, ("Kh", "Qs")),    # plays the board straight (tie)
        _mk(2, Position.BB, 0.05, folded=True),     # dead money -> odd main layer
        _mk(3, Position.UTG, 0.0, folded=True),
        _mk(4, Position.HJ, 0.0, folded=True),
        _mk(5, Position.CO, 0.0, folded=True),
    ]
    return st


def test_odd_chip_goes_to_first_winner_left_of_button():
    # button=0 -> first seat clockwise is 1 (a winner) -> seat 1 gets the odd chip.
    d = _odd_chip_state(button_seat=0).resolve_showdown()
    assert abs(d[1] - 0.03) < 1e-9       # 7c + 1c remainder + 5c side layer - 10c committed
    assert abs(d[0] - 0.02) < 1e-9       # 7c + 5c - 10c
    assert abs(d[2] + 0.05) < 1e-9
    assert abs(sum(d.values())) < 1e-9    # zero-sum preserved


def test_odd_chip_follows_the_button():
    # button=5 -> first seat clockwise is 0 (a winner) -> the odd chip moves to seat 0.
    d = _odd_chip_state(button_seat=5).resolve_showdown()
    assert abs(d[0] - 0.03) < 1e-9
    assert abs(d[1] - 0.02) < 1e-9
    assert abs(sum(d.values())) < 1e-9


def test_even_split_unaffected():
    # No dead money: each winner committed 0.10 -> main layer 0.20 = 20 chips / 2 = exact 10,10.
    st = _odd_chip_state(button_seat=0)
    st.players[2] = _mk(2, Position.BB, 0.0, folded=True)
    d = st.resolve_showdown()
    assert abs(d[0] - d[1]) < 1e-9       # exact even split, no odd-chip skew
    assert abs(sum(d.values())) < 1e-9


# --------------------------------------------------------------------------- O4
def _hand(hid, name, uid=None, atype="raise"):
    a = {"street": "preflop", "type": atype, "name": name}
    if uid is not None:
        a["user_id"] = uid
    return {"hand_id": hid, "board": [], "actions": [a]}


def test_name_then_uid_reconciles_into_one_profile(tmp_path):
    from pokerbot.opponent.player_profiles import PlayerProfileStore
    store = PlayerProfileStore(tmp_path / "p.json", wal=False)

    store.update_from_hand(_hand("h1", "Bob"))                 # uid dropped -> keyed name:Bob
    assert store.profiles["name:Bob"].hands == 1

    store.update_from_hand(_hand("h2", "Bob", uid=999))        # uid seen -> learn alias + merge orphan
    assert "name:Bob" not in store.profiles                     # split-brain twin reconciled away
    assert store.get("999").hands == 2                          # both hands now under the uid
    assert store.alias["name:Bob"] == "999"

    store.update_from_hand(_hand("h3", "Bob"))                 # name-only again -> resolves to the uid
    assert store.get("999").hands == 3
    assert "name:Bob" not in store.profiles


def test_alias_survives_reload(tmp_path):
    from pokerbot.opponent.player_profiles import PlayerProfileStore
    p = tmp_path / "p.json"
    store = PlayerProfileStore(p, wal=False)
    store.update_from_hand(_hand("h1", "Bob", uid=999))
    store.save()

    reopened = PlayerProfileStore(p, wal=False)
    assert reopened.alias.get("name:Bob") == "999"
    reopened.update_from_hand(_hand("h2", "Bob"))              # name-only post-restart -> uid profile
    assert reopened.get("999").hands == 2
    assert "name:Bob" not in reopened.profiles


# --------------------------------------------------------------------------- SE8
def test_native_policy_clobber_warns_once():
    from pokerbot.search import native_solver as ns
    saved = (ns._LAST_EXPLICIT_POLICY, ns._POLICY_CLASH_WARNED, ns._ENABLED)
    try:
        ns._LAST_EXPLICIT_POLICY = None
        ns._POLICY_CLASH_WARNED = False
        ns.disable()                       # first explicit policy (OFF) -> no warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ns.enable()                    # OFF -> ON conflict -> warns once
            ns.disable()                   # further flips stay silent
            ns.enable()
        msgs = [str(x.message) for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(msgs) == 1
        assert "process-global" in msgs[0].lower()
    finally:
        ns._LAST_EXPLICIT_POLICY, ns._POLICY_CLASH_WARNED, ns._ENABLED = saved


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
