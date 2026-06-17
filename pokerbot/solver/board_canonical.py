"""Suit-isomorphism canonical keys for flop boards and solver spots."""

from __future__ import annotations

from pokerbot.engine.cards import Card, Rank, Suit
from pokerbot.search.isomorphism import canonicalize_cards


def canonical_flop_key(board: list[Card] | str) -> tuple[tuple[int, int], ...]:
    """Canonical suit-isomorphism key for a 3-card flop (ranks + relabeled suits)."""
    if isinstance(board, str):
        cards = [Card.from_str(board[i : i + 2]) for i in range(0, len(board), 2)]
    else:
        cards = list(board)
    _, canon_board, _ = canonicalize_cards(None, cards)
    parts = sorted(
        ((idx // 4, idx % 4) for idx in canon_board),
        key=lambda rs: (-rs[0], rs[1]),
    )
    return tuple(parts)


def canonical_flop_str(board: list[Card] | str | tuple[tuple[int, int], ...]) -> str:
    """Human-readable canonical flop string (rank chars + canonical suit ids)."""
    if isinstance(board, tuple):
        key = board
    else:
        key = canonical_flop_key(board)
    rank_chars = "23456789TJQKA"
    return "".join(f"{rank_chars[r - 2]}{s}" for r, s in key)


def _card_from_index(idx: int) -> Card:
    return Card(Rank(idx // 4 + 2), Suit(idx % 4))


def canonical_spot_key(spot_key: str, *, hero_hole: tuple[str, str] | None = None) -> str:
    """Canonical key for spot_key + hero hole under suit relabeling.

    ``spot_key`` format: ``{spot_id}|{board}|hero={seat}``. When ``hero_hole`` is
    omitted, only the board is canonicalized (weaker duplicate detection).
    """
    parts = spot_key.split("|")
    if len(parts) < 2:
        return spot_key
    board_str = parts[1]
    board = [Card.from_str(board_str[i : i + 2]) for i in range(0, len(board_str), 2)]
    hole_cards: tuple[Card, Card] | None = None
    if hero_hole is not None:
        hole_cards = (Card.from_str(hero_hole[0]), Card.from_str(hero_hole[1]))
    canon_hole, canon_board, _ = canonicalize_cards(hole_cards, board)
    board_key = "".join(
        _card_from_index(idx).to_str()
        for idx in sorted(canon_board, key=lambda i: (-i // 4, i % 4))
    )
    spot_id = parts[0]
    hero_part = parts[2] if len(parts) > 2 else ""
    if canon_hole is not None:
        h0 = _card_from_index(canon_hole[0]).to_str()
        h1 = _card_from_index(canon_hole[1]).to_str()
        hero_part = f"hero={h0}{h1}"
    return f"{spot_id}|{board_key}|{hero_part}"


def dedupe_boards_by_isomorphism(boards: list[list[Card]]) -> tuple[list[list[Card]], int]:
    """Return boards with suit-isomorphic duplicates removed (first occurrence kept)."""
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[list[Card]] = []
    skipped = 0
    for board in boards:
        key = canonical_flop_key(board)
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        out.append(board)
    return out, skipped
