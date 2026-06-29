"""Tests for the satellite seat-lock fold gate (live_adapter.satellite_fold_override).

The gate fires ONLY when (a) it is a satellite, (b) the caller certifies the passed stacks are the WHOLE
remaining field (a table-local lock does not certify a field lock), and (c) hero is mathematically locked
and the action risks chips. Everything else returns False (play normally) -- folding a not-actually-locked
seat would burn EV, so the gate is conservative by construction.
"""

from __future__ import annotations

from pokerbot.tournament import resolve_payouts, satellite_fold_override


def _sat(seats):
    return resolve_payouts(is_satellite=True, seats=seats, seat_value=100.0, entrants=50)


def _normal():
    return resolve_payouts(tournament_type="MTT_FREEZEOUT", entrants=100, buy_in=10.0)


def test_non_satellite_never_fires():
    assert satellite_fold_override(_normal(), [50, 30, 20], 0, field_is_complete=True) is False
    assert satellite_fold_override(None, [50, 30, 20], 0, field_is_complete=True) is False


def test_locked_full_field_forces_fold():
    # seats=2, stacks [50,30,20]: hero 50 >= total/(seats+1)=33.3 -> locked -> fold a chip-risking action
    assert satellite_fold_override(_sat(2), [50, 30, 20], 0, field_is_complete=True) is True


def test_partial_field_refuses_to_fire():
    # Same locked-looking table stacks, but the field is NOT certified complete -> never fold (other
    # tables may hold bigger stacks; a table-local lock is not a field lock).
    assert satellite_fold_override(_sat(2), [50, 30, 20], 0, field_is_complete=False) is False


def test_unlocked_does_not_fold():
    # hero 20 < 33.3 -> not locked
    assert satellite_fold_override(_sat(2), [20, 30, 50], 0, field_is_complete=True) is False


def test_free_check_never_forced_to_fold():
    assert satellite_fold_override(_sat(2), [50, 30, 20], 0, field_is_complete=True,
                                   action_risks_chips=False) is False


def test_trivial_lock_when_field_at_or_below_seats():
    # 2 players left, 3 seats paid -> everyone is in -> locked
    assert satellite_fold_override(_sat(3), [10, 5], 0, field_is_complete=True) is True
