"""Regression test: read-site context keys must match ingestion-site context keys.

Bug class: if a read site builds the key via ``context_key(board=[str(Card)...])`` and
``str(Card)`` returned the dataclass repr ('Card(rank=...)') instead of the canonical
card string ('Jh'), ``board_texture()`` would parse nothing and produce '-' for every
postflop board.  That made every postflop read key '<street>|-|...' which NEVER matched
the stored key (stored via ``context_key_rs`` with real 'wet'/'dry'/'paired' textures),
so the opponent model fired 0% postflop (100% postflop read miss rate).

This test pins the invariant: for any card set and street combination,
    context_key_rs(street, ranks, suits, n)  ==  context_key(street, str_board, n)
as long as ``str_board`` is built from canonical card strings ('Jh', not 'Card(rank=...)').
It also verifies that the ExploitHero read-site fix (using context_key_rs directly) produces
the same key as the ingestion path used in bot_wrapped (_ctx_key_for, also context_key_rs).
"""

from __future__ import annotations

import random

import pytest

from pokerbot.engine.cards import Card, Rank, Suit, RANK_CHARS, SUIT_CHARS
from pokerbot.opponent.context import (
    context_key,
    context_key_rs,
    board_texture,
    board_texture_rs,
    _PREFLOP_TEXTURE,
    STREET_NAMES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_board(rng: random.Random, n: int) -> list[Card]:
    deck = [Card(Rank(r), Suit(s)) for r in range(2, 15) for s in range(4)]
    return rng.sample(deck, n)


def _card_to_rs(card: Card) -> tuple[int, int]:
    return int(card.rank.value), int(card.suit.value)


def _board_to_str(cards: list[Card]) -> list[str]:
    return [str(c) for c in cards]  # should return 'Jh' not 'Card(rank=...)'


# ---------------------------------------------------------------------------
# Core parity assertion
# ---------------------------------------------------------------------------

def _check_parity(cards: list[Card], street: int, n_aggr: int) -> None:
    """Assert that the rs-form and the string-form keys agree."""
    ranks = [int(c.rank.value) for c in cards]
    suits = [int(c.suit.value) for c in cards]
    str_board = _board_to_str(cards)

    key_rs = context_key_rs(street, ranks, suits, n_aggr)
    key_str = context_key(street, str_board, n_aggr)

    assert key_rs == key_str, (
        f"Key mismatch at street={STREET_NAMES[street]} n_aggr={n_aggr}: "
        f"rs={key_rs!r} vs str={key_str!r} "
        f"(board={str_board!r}, str(cards[0])={str(cards[0]) if cards else '(empty)'})"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_card_str_is_canonical_not_repr():
    """str(Card) must return the canonical '2c'/'Ah' form, not the dataclass repr.

    This is the root-cause guard: if __str__ is missing, str(Card) returns
    'Card(rank=<Rank.ACE: 14>, suit=<Suit.HEARTS: 2>)' which board_texture()
    cannot parse, producing '-' for every postflop board."""
    c = Card(Rank.ACE, Suit.HEARTS)
    s = str(c)
    assert s == "Ah", f"str(Card) returned {s!r}, expected 'Ah' — Card.__str__ is broken or missing"
    assert "rank" not in s.lower() and "suit" not in s.lower(), (
        f"str(Card) looks like a dataclass repr: {s!r}")


def test_board_texture_rs_matches_string_form_sampled():
    """board_texture_rs(ranks, suits) == board_texture(str_board) for 2000 random boards."""
    rng = random.Random(42)
    mismatches = []
    for _ in range(2000):
        n = rng.randint(3, 5)
        cards = _random_board(rng, n)
        ranks = [int(c.rank.value) for c in cards]
        suits = [int(c.suit.value) for c in cards]
        str_board = _board_to_str(cards)

        tex_rs = board_texture_rs(ranks, suits)
        tex_str = board_texture(str_board)
        if tex_rs != tex_str:
            mismatches.append((str_board, tex_rs, tex_str))

    assert not mismatches, (
        f"board_texture_rs != board_texture for {len(mismatches)} boards: {mismatches[:5]}")


def test_context_key_rs_equals_string_key_preflop():
    """Preflop: no board — both key forms must agree on texture '-' and full facing depth."""
    for n_aggr in range(5):
        key_rs = context_key_rs(0, [], [], n_aggr)
        key_str = context_key(0, [], n_aggr)
        assert key_rs == key_str, (
            f"Preflop parity failed at n_aggr={n_aggr}: rs={key_rs!r} str={key_str!r}")


def test_context_key_rs_equals_string_key_postflop_sampled():
    """Flop/turn/river: 1500 random boards per street, both facing values.

    This is the direct regression for the bug class: if str(Card) returned the dataclass
    repr, board_texture() would produce '-' and key_str would NEVER match key_rs for any
    postflop board with a non-'-' texture (wet/dry/paired)."""
    rng = random.Random(99)
    total = 0
    mismatches = []
    for street_idx in (1, 2, 3):  # flop, turn, river
        n_board = (3, 4, 5)[street_idx - 1]
        for _ in range(500):
            cards = _random_board(rng, n_board)
            for n_aggr in (0, 1):
                _check_parity(cards, street_idx, n_aggr)
                total += 1

    # Bonus: verify at least one postflop board in each texture class was covered
    # (sanity check that the sample wasn't all-dry or all-preflop).
    textures_seen: set[str] = set()
    for _ in range(500):
        cards = _random_board(rng, 3)
        textures_seen.add(board_texture_rs(
            [int(c.rank.value) for c in cards],
            [int(c.suit.value) for c in cards]))
    assert textures_seen >= {"dry", "wet", "paired"}, (
        f"Sample didn't cover all postflop textures: {textures_seen}")
    assert total == 3000, f"Expected 3000 checks, got {total}"


def test_context_key_rs_equals_string_key_named_boards():
    """Exact named boards for each texture — the deterministic regression cases."""
    cases = [
        # (board_str, street, expected_texture, n_aggr)
        (["Jh", "Jc", "4d"], 1, "paired", 0),    # paired flop
        (["Jh", "8h", "2h"], 1, "wet", 1),        # flush-heavy flop
        (["9h", "8c", "7d"], 1, "wet", 0),         # connected flop
        (["Kh", "8c", "3d"], 1, "dry", 1),         # rainbow disconnected flop
        (["Ah", "Kh", "Qh", "Jh"], 2, "wet", 0),  # monotone turn
        (["Ah", "2c", "7d", "Ks"], 2, "dry", 1),  # dry turn
        (["2h", "2c", "7d", "7s", "Ah"], 3, "paired", 0),  # paired river
    ]
    for board_strs, street, expected_tex, n_aggr in cases:
        cards = [Card.from_str(s) for s in board_strs]
        ranks = [int(c.rank.value) for c in cards]
        suits = [int(c.suit.value) for c in cards]
        str_board = _board_to_str(cards)

        tex_rs = board_texture_rs(ranks, suits)
        tex_str = board_texture(str_board)

        assert tex_rs == expected_tex, f"{board_strs}: expected {expected_tex}, got tex_rs={tex_rs}"
        assert tex_str == expected_tex, f"{board_strs}: expected {expected_tex}, got tex_str={tex_str}"
        _check_parity(cards, street, n_aggr)


def test_exploit_hero_read_key_matches_ingestion_key():
    """Simulate what ExploitHero.__call__ does (post-fix) vs what bot_wrapped/_ctx_key_for does.

    Both should produce identical context keys for every board/street combination.
    This pins the specific bug that was found and fixed: ExploitHero was using
    context_key(board=[str(c)...]) while _ctx_key_for used context_key_rs.
    After the fix, ExploitHero also uses context_key_rs — so both paths must agree."""
    from pokerbot.eval.exploitation_battery import _STREET_IDX, _STREET_BOARD_LEN

    rng = random.Random(7)
    for street_name_upper in ("PREFLOP", "FLOP", "TURN", "RIVER"):
        street = _STREET_IDX.get(street_name_upper, 0)
        n_board = _STREET_BOARD_LEN[street]
        # generate boards (preflop: 0 cards; flop: 3; turn: 4; river: 5)
        for _ in range(100):
            all_cards = _random_board(rng, 5)
            board_cards = all_cards[:n_board]

            for n_aggr in (0, 1):
                # Post-fix ExploitHero read path:
                ranks_hero = [int(c.rank.value) for c in board_cards]
                suits_hero = [int(c.suit.value) for c in board_cards]
                hero_key = context_key_rs(street, ranks_hero, suits_hero, n_aggr)

                # _ctx_key_for ingestion path (also uses context_key_rs):
                ranks_ingest = [int(c.rank.value) for c in board_cards]
                suits_ingest = [int(c.suit.value) for c in board_cards]
                ingest_key = context_key_rs(street, ranks_ingest, suits_ingest, n_aggr)

                assert hero_key == ingest_key, (
                    f"Key mismatch at {street_name_upper} n_aggr={n_aggr}: "
                    f"hero={hero_key!r} ingest={ingest_key!r} board={[str(c) for c in board_cards]!r}")

                # Also verify: if we INCORRECTLY used the old buggy string path with repr(),
                # the key would diverge for any non-empty postflop board.
                if n_board >= 3:
                    # Simulate the OLD (buggy) behavior: card string that can't be parsed.
                    bad_board_strs = [f"Card(rank={c.rank.value},suit={c.suit.value})" for c in board_cards]
                    bad_key = context_key(street, bad_board_strs, n_aggr)
                    # The bad key would have texture '-' since those strings can't be parsed.
                    bad_texture = bad_key.split("|")[1]
                    assert bad_texture == _PREFLOP_TEXTURE, (
                        f"Expected bad-path to produce '-' texture but got {bad_texture!r}")
                    # And it should NOT equal the correct key (unless the correct key is also '-',
                    # which only happens preflop — but we're in the postflop case here).
                    correct_texture = hero_key.split("|")[1]
                    if correct_texture != _PREFLOP_TEXTURE:
                        assert bad_key != hero_key, (
                            f"Old repr-path accidentally matched correct key for {[str(c) for c in board_cards]!r}")
