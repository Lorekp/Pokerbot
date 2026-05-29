"""AoF game config and rake plumbing (Priority 12)."""

from __future__ import annotations

import pytest

from aof_game_config import AoFGameConfig, RakeModel, default_game_config, net_pot_bb
from aof_multiway import multiway_call_terms


def test_default_matches_legacy_pot() -> None:
    terms = multiway_call_terms("BB", 8.0, villain_positions=("UTG",), table_size=4)
    legacy = multiway_call_terms(
        "BB", 8.0, villain_positions=("UTG",), table_size=4, game=default_game_config()
    )
    assert terms.total_pot_after_call == pytest.approx(legacy.total_pot_after_call)
    assert terms.breakeven_equity == pytest.approx(legacy.breakeven_equity)


def test_rake_reduces_net_pot() -> None:
    game = AoFGameConfig(
        rake_model=RakeModel(percent=0.05, cap_bb=1.0, applies_to_all_in_preflop=True)
    )
    gross = 24.5
    net = net_pot_bb(gross, game)
    assert net < gross
    terms = multiway_call_terms(
        "BB",
        8.0,
        villain_positions=("UTG", "BTN"),
        table_size=4,
        game=game,
    )
    assert terms.total_pot_after_call == pytest.approx(net)
    assert terms.rake_bb > 0


def test_rake_lowers_breakeven_need() -> None:
    no_rake = multiway_call_terms(
        "BB", 8.0, villain_positions=("UTG", "BTN"), table_size=4
    )
    with_rake = multiway_call_terms(
        "BB",
        8.0,
        villain_positions=("UTG", "BTN"),
        table_size=4,
        game=AoFGameConfig(rake_model=RakeModel(0.05, 2.0)),
    )
    assert with_rake.breakeven_equity > no_rake.breakeven_equity


def test_rake_detail_prefix() -> None:
    from aof_game_config import rake_detail_prefix

    assert rake_detail_prefix(None).startswith("pre-rake")
    assert "rake-adj" in rake_detail_prefix(
        AoFGameConfig(rake_model=RakeModel(0.05, 1.0))
    )
