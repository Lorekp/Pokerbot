"""The sim battery must be able to MEASURE the value-sizing overlay: seeding the villain's server HUD
(population_hud) lets the HUD-gated overlay fire in sim exactly as it does live. Without the HUD the WSD
value-cut guard fails closed (no showdown-weakness evidence) and the overlay is a no-op."""

from __future__ import annotations

from types import SimpleNamespace

from pokerbot.engine.cards import Card
from pokerbot.engine.game import GameState, PlayerState, Position, Street
from pokerbot.eval.exploitation_battery import population_hud, run_archetype
from pokerbot.opponent.value_sizing import overlay_hand_strength


def test_population_hud_present_for_station_absent_for_unknown():
    assert population_hud("calling_station") is not None
    assert "wsd" in population_hud("calling_station")        # the showdown-weakness stat the gate needs
    assert population_hud("random") is None
    assert population_hud("gear_changer") is None


def test_overlay_hand_strength_is_board_aware():
    def _state(hole, board):
        st = GameState()
        st.players = [PlayerState(seat=0, position=Position.BTN, stack_bb=100.0)]
        st.board = [Card.from_str(c) for c in board]
        st.players[0].hole = (Card.from_str(hole[0]), Card.from_str(hole[1]))
        return st
    made = overlay_hand_strength(_state(("As", "Ah"), ("2c", "7d", "Qh")), 0)   # overpair
    air = overlay_hand_strength(_state(("3c", "2h"), ("Ks", "9d", "7h")), 0)     # bottom air
    assert made > 0.5 > air                                  # a made hand reads value; air reads air


def test_seed_hud_lets_overlay_fire_vs_station_not_nit(monkeypatch):
    def arm(arch, sizing):
        monkeypatch.setenv("POKERBOT_VALUE_SIZING", "1" if sizing else "0")
        return run_archetype(arch, hands=150, seed=0, progress_every=10**9,
                             seed_hud=True)["uplift_bb_per_hand"]
    # vs a HUD-seeded calling station the overlay changes the policy -> the ON/OFF arms diverge.
    assert arm("calling_station", True) != arm("calling_station", False)
    # vs a nit (folds too much -> not a value-fish, weak-showdown guard off) the overlay stays a no-op,
    # so the toggle is byte-identical under CRN.
    assert arm("nit", True) == arm("nit", False)
