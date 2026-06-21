"""state_bridge LOUD pot-conservation telemetry (ground-rule 5: no silent clamps that hide bugs).

``pokerbot.search.state_bridge.game_state_to_env`` used to do ``pot_dead = max(pot_dead, 0)`` and SILENTLY
shrink the dead-money pot when the live reconstruction produced ``pot_bb < sum(committed_street)`` --
masking a reconstruction bug on the LIVE serve path. The clamp is now preceded by a LOUD, structured
signal (a ``PotConservationBreach`` -> logging.warning + a process counter + an optional observer sink).

This test asserts:
  * a CONSISTENT state is byte-identical to before -- no warning, no breach record, no spurious shrink
    (pot_dead reconstructed faithfully); and
  * an INCONSISTENT state (pot_bb < sum(committed_street)) emits the loud signal (warning + counter + sink
    + structured fields) INSTEAD of silently clamping, while the returned state stays usable (pot_dead
    floored at 0, no crash).
"""

from __future__ import annotations

import logging

import numpy as np

from pokerbot.engine.game import GameState, Street
from pokerbot.engine.jax_engine import CHIP_UNIT
from pokerbot.search import state_bridge as sb


def _clean_state() -> GameState:
    """A consistent 6-handed preflop frame: SB+BB posted, pot == sum(committed_street), no dead money."""
    gs = GameState.new_hand(button_seat=3)
    gs._deck = None
    # new_hand already posted blinds: SB=0.5, BB=1.0 on the by-position seats, pot_bb=1.5, all on-street.
    # That is exactly consistent (pot_bb == sum committed_street, pot_dead == 0).
    return gs


def test_consistent_state_no_breach_no_shrink():
    sb.set_pot_breach_sink(None)
    before = sb.pot_breach_count()
    gs = _clean_state()

    captured: list = []
    sb.set_pot_breach_sink(captured.append)
    try:
        env = sb.game_state_to_env(gs, hero_seat=0)
    finally:
        sb.set_pot_breach_sink(None)

    # No breach recorded, no sink call.
    assert sb.pot_breach_count() == before, "consistent state spuriously recorded a pot-conservation breach"
    assert captured == [], "consistent state spuriously fired the breach sink"

    # pot_dead reconstructed FAITHFULLY (== pot_bb - sum(committed_street), in chips), not shrunk.
    committed_street_bb = sum(p.committed_street_bb for p in gs.players)
    expected_pot_dead = int(round((gs.pot_bb - committed_street_bb) * CHIP_UNIT))
    assert int(np.asarray(env.pot_dead)) == expected_pot_dead, (
        f"pot_dead shrunk on a clean state: got {int(np.asarray(env.pot_dead))} want {expected_pot_dead}"
    )
    assert expected_pot_dead == 0, "fixture sanity: clean preflop frame has zero dead money"


def test_dead_money_state_is_faithful_and_loud_free():
    """A state with LEGITIMATE dead money (pot_bb > sum(committed_street), e.g. a prior street's chips)
    must reconstruct pot_dead > 0 and NOT trip the breach (it is the opposite, consistent direction)."""
    sb.set_pot_breach_sink(None)
    before = sb.pot_breach_count()
    gs = _clean_state()
    gs.street = Street.FLOP
    # Carry 5bb of settled dead money from preflop; nobody has bet yet this street.
    for p in gs.players:
        p.committed_street_bb = 0.0
    gs.pot_bb = 5.0  # all dead, zero on-street -> pot_dead = 5.0, fully consistent (pot_bb >= street sum)

    env = sb.game_state_to_env(gs, hero_seat=0)
    assert sb.pot_breach_count() == before, "dead-money (consistent) state wrongly flagged a breach"
    assert int(np.asarray(env.pot_dead)) == int(round(5.0 * CHIP_UNIT)), "legit dead money not preserved"


def test_inconsistent_state_emits_loud_signal_not_silent_clamp(caplog):
    """pot_bb < sum(committed_street): the OLD code silently shrank pot_dead to 0. The NEW code must emit
    the LOUD signal (warning log + counter increment + structured sink record) and STILL return a usable
    state (pot_dead floored at 0, no crash)."""
    sb.set_pot_breach_sink(None)
    before = sb.pot_breach_count()
    gs = _clean_state()
    # Force an inconsistency: two seats each show 3bb on-street (sum=6bb) but the feed pot is only 4bb.
    for p in gs.players:
        p.committed_street_bb = 0.0
    gs.players[0].committed_street_bb = 3.0
    gs.players[1].committed_street_bb = 3.0
    gs.pot_bb = 4.0  # < 6bb on-street -> 2bb deficit -> reconstruction inconsistency

    captured: list = []
    sb.set_pot_breach_sink(captured.append)
    try:
        with caplog.at_level(logging.WARNING, logger="pokerbot.search.state_bridge"):
            env = sb.game_state_to_env(gs, hero_seat=2)
    finally:
        sb.set_pot_breach_sink(None)

    # LOUD: counter incremented, sink fired with a structured record, warning logged.
    assert sb.pot_breach_count() == before + 1, "breach was not counted (silently absorbed?)"
    assert len(captured) == 1, "breach sink not fired exactly once"
    breach = captured[0]
    assert isinstance(breach, sb.PotConservationBreach)
    assert breach.pot_bb == 4.0
    assert breach.committed_street_bb == 6.0
    assert abs(breach.deficit_bb - 2.0) < 1e-9
    assert abs(breach.pot_dead_bb - (-2.0)) < 1e-9
    assert breach.hero_seat == 2
    assert any("POT-CONSERVATION BREACH" in r.message for r in caplog.records), "loud warning not logged"

    # USABLE: state still produced, pot_dead floored at 0 (not negative), no crash.
    assert int(np.asarray(env.pot_dead)) == 0, "pot_dead must be floored at 0 to keep the engine usable"


def test_below_tolerance_float_noise_does_not_trip():
    """Sub-chip float noise (< 0.5/CHIP_UNIT bb) is benign rounding, not a reconstruction bug -> NO breach."""
    sb.set_pot_breach_sink(None)
    before = sb.pot_breach_count()
    gs = _clean_state()
    for p in gs.players:
        p.committed_street_bb = 0.0
    gs.players[0].committed_street_bb = 1.0
    # pot_bb a hair below the street sum, but within half-a-chip -> benign.
    gs.pot_bb = 1.0 - (0.4 / CHIP_UNIT)

    captured: list = []
    sb.set_pot_breach_sink(captured.append)
    try:
        sb.game_state_to_env(gs, hero_seat=0)
    finally:
        sb.set_pot_breach_sink(None)
    assert sb.pot_breach_count() == before, "sub-chip float noise wrongly flagged as a breach"
    assert captured == [], "sub-chip float noise wrongly fired the sink"
