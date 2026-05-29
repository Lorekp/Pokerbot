"""AoF table / rake parameters (defaults: pre-rake chip EV, no rake)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RakeModel:
    percent: float
    cap_bb: float
    no_flop_no_drop: bool = False
    applies_to_all_in_preflop: bool = True


@dataclass(frozen=True)
class AoFGameConfig:
    game: str = "coinpoker_nlh_aof"
    table_size: int = 4
    stack_bb: float = 8.0
    sb: float = 0.5
    bb: float = 1.0
    ante_bb: float = 0.0
    rake_model: RakeModel | None = None
    jackpot_drop_bb: float = 0.0
    equal_stacks_only: bool = True
    allow_side_pots: bool = False


def default_game_config() -> AoFGameConfig:
    return AoFGameConfig()


def compute_rake_bb(gross_pot_bb: float, game: AoFGameConfig) -> float:
    """Rake in bb taken from gross pot (0 when rake_model is None)."""
    if game.rake_model is None:
        return 0.0
    rm = game.rake_model
    if rm.no_flop_no_drop and not rm.applies_to_all_in_preflop:
        return 0.0
    if not rm.applies_to_all_in_preflop:
        return 0.0
    return min(rm.cap_bb, rm.percent * gross_pot_bb)


def net_pot_bb(gross_pot_bb: float, game: AoFGameConfig) -> float:
    rake = compute_rake_bb(gross_pot_bb, game)
    return max(0.0, gross_pot_bb - rake - game.jackpot_drop_bb)


def rake_detail_prefix(game: AoFGameConfig | None) -> str:
    if game is None or game.rake_model is None:
        return "pre-rake:"
    return "rake-adj:"
