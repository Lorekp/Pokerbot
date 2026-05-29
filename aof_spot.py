"""Build AoF advisor shorthand from structured spot fields."""

from __future__ import annotations

from aof_parse import _infer_players_behind


def card_label(rank: str, suit: str) -> str:
    return f"{rank.upper()}{suit.lower()}"


def hand_token(card_a: str, card_b: str) -> str:
    """Two cards as one token, e.g. AsKh (parser accepts explicit cards)."""
    return f"{card_a}{card_b}"


def build_spot_line(
    *,
    card_a: str,
    card_b: str,
    hero_pos: str,
    stack_bb: float,
    facing_shove: bool,
    shover_pos: str | None = None,
    second_shover_pos: str | None = None,
    third_shover_pos: str | None = None,
    players_behind: int | None = None,
    table_size: int = 4,
) -> str:
    if not card_a or not card_b:
        raise ValueError("Select two hole cards.")

    parts = [hand_token(card_a, card_b), hero_pos]

    if facing_shove:
        if not shover_pos:
            raise ValueError("Select shover position for call-vs-shove.")
        if second_shover_pos or third_shover_pos:
            from aof_parse import _validate_shovers_before_hero

            _validate_shovers_before_hero(
                hero_pos,
                shover_pos,
                table_size,
                second_shover=second_shover_pos,
                third_shover=third_shover_pos,
            )
            shovers = [shover_pos]
            if second_shover_pos:
                shovers.append(second_shover_pos)
            if third_shover_pos:
                shovers.append(third_shover_pos)
            parts.extend(["vs", ",".join(shovers)])
            players_behind = 0
        else:
            parts.extend(["vs", shover_pos])
        if players_behind is None:
            players_behind, _ = _infer_players_behind(
                hero_pos, shover_pos, table_size
            )
        if players_behind > 0:
            parts.extend(["behind", str(players_behind)])

    if table_size == 2:
        parts.append("hu")
    elif table_size != 4:
        parts.append(str(table_size))
    stack_s = str(int(stack_bb)) if stack_bb == int(stack_bb) else str(stack_bb)
    parts.append(f"{stack_s}bb")
    return " ".join(parts)
