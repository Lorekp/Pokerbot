"""The per-context keying spec (the train/serve-skew firewall). Pins the spec hash so any silent
change to the context/bin partition fails loudly and forces an intentional migration."""

from __future__ import annotations

from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.opponent import context as C


def test_spec_hash_is_pinned_and_deterministic() -> None:
    # If this fails after an intentional spec change, bump the literal AND migrate stored DB keys.
    # c960... = 4-level preflop facing (open/3bet/4bet) + binary postflop (R3 finer-facing spec).
    assert C.context_spec_hash() == "c960cfceab695388"
    assert C.context_spec_hash() == C.context_spec_hash()


def test_action_bins_partition_the_whole_action_space() -> None:
    bins = [C.action_bin(a) for a in range(NUM_ACTIONS)]
    assert set(bins) <= set(range(C.NUM_BINS))
    assert bins[0] == C.FOLD_BIN
    assert bins[1] == C.PASSIVE_BIN and bins[2] == C.PASSIVE_BIN  # check, call
    assert bins[3] == C.AGG_SMALL_BIN                              # smallest bet
    assert bins[NUM_ACTIONS - 1] == C.AGG_LARGE_BIN               # all-in
    assert C.AGG_SMALL_BIN in bins and C.AGG_LARGE_BIN in bins    # both aggression bins reachable


def test_board_texture() -> None:
    assert C.board_texture([]) == "-"                       # preflop
    assert C.board_texture(["Jh", "4c"]) == "-"             # <3 cards
    assert C.board_texture(["Jh", "Jc", "4d"]) == "paired"
    assert C.board_texture(["Jh", "8h", "2h"]) == "wet"     # flush-heavy
    assert C.board_texture(["9h", "8c", "7d"]) == "wet"     # connected
    assert C.board_texture(["Kh", "8c", "3d"]) == "dry"     # rainbow disconnected


def test_facing_buckets() -> None:
    # Full-depth bucket (used preflop): 0=>unopened, 1=>facing_bet, 2=>facing_raise, >=3=>facing_3bet.
    assert [C.facing_bucket(n) for n in (0, 1, 2, 3, 9)] == [
        "unopened", "facing_bet", "facing_raise", "facing_3bet", "facing_3bet",
    ]


def test_facing_token_caps_postflop_but_not_preflop() -> None:
    # The skew firewall: preflop resolves full depth; postflop collapses to the reliable binary.
    assert C.facing_token("preflop", 2) == "facing_raise"
    assert C.facing_token("preflop", 3) == "facing_3bet"
    assert C.facing_token("flop", 2) == "facing_bet"      # postflop capped at 1
    assert C.facing_token("river", 5) == "facing_bet"


def test_aggression_predicates_agree_across_index_and_type() -> None:
    # The bet/raise/all-in indices are aggressive; fold/check/call are not.
    assert C.is_aggressive_action(NUM_ACTIONS - 1)        # all-in
    assert C.is_aggressive_action(3)                       # smallest bet
    assert not C.is_aggressive_action(0) and not C.is_aggressive_action(1) and not C.is_aggressive_action(2)
    assert C.is_aggressive_type("bet") and C.is_aggressive_type("raise") and C.is_aggressive_type("all_in")
    assert not C.is_aggressive_type("call") and not C.is_aggressive_type("check")
    assert not C.is_aggressive_type("bb") and not C.is_aggressive_type("fold")


def test_context_key_composition_and_enumeration() -> None:
    assert C.context_key(0, [], 0) == "preflop|-|unopened"
    assert C.context_key(0, [], 2) == "preflop|-|facing_raise"     # preflop 3bet spot, full depth
    assert C.context_key(1, ["Jh", "Jc", "4d"], 1) == "flop|paired|facing_bet"
    assert C.context_key(1, ["Jh", "Jc", "4d"], 3) == "flop|paired|facing_bet"  # postflop capped
    # rank/suit serve-side entry agrees with the string entry.
    assert C.context_key_rs(1, [11, 11, 4], ["h", "c", "d"], 1) == "flop|paired|facing_bet"
    ctx = C.all_contexts()
    assert len(ctx) == 22 and len(set(ctx)) == 22        # 4 preflop + 3x3x2 postflop, all distinct
    assert "preflop|-|facing_3bet" in ctx and "river|wet|facing_bet" in ctx
    assert "flop|dry|facing_raise" not in ctx            # postflop never realizes facing_raise
    assert all(k.split("|")[1] == "-" for k in ctx if k.startswith("preflop"))  # preflop texture only '-'
