"""Per-player profile derivation + persistence + idempotency."""

from __future__ import annotations

from pokerbot.opponent.player_profiles import PlayerProfileStore, derive_hand_stats


def _hand(hand_id="H1"):
    # fish 3-bets preflop and barrels; hero calls pre then folds flop; villain folds pre.
    return {
        "hand_id": hand_id,
        "board": ["2c", "7d", "Qh"],
        "winners": [{"seat": 3, "user_id": 333}],
        "actions": [
            {"street": "preflop", "seat": 1, "user_id": 111, "name": "hero", "type": "sb", "amount": 50},
            {"street": "preflop", "seat": 2, "user_id": 222, "name": "villain", "type": "bb", "amount": 100},
            {"street": "preflop", "seat": 4, "user_id": 444, "name": "reg", "type": "raise", "amount": 250},
            {"street": "preflop", "seat": 3, "user_id": 333, "name": "fish", "type": "raise", "amount": 800},
            {"street": "preflop", "seat": 1, "user_id": 111, "name": "hero", "type": "call", "amount": 800},
            {"street": "preflop", "seat": 2, "user_id": 222, "name": "villain", "type": "fold", "amount": 0},
            {"street": "preflop", "seat": 4, "user_id": 444, "name": "reg", "type": "fold", "amount": 0},
            {"street": "flop", "seat": 3, "user_id": 333, "name": "fish", "type": "bet", "amount": 600},
            {"street": "flop", "seat": 1, "user_id": 111, "name": "hero", "type": "fold", "amount": 0},
        ],
    }


def test_derive_hand_stats():
    s = derive_hand_stats(_hand())
    # fish: voluntarily in, raised pre, and 3-bet (raise after reg's raise), bet flop
    assert s["333"]["vpip"] == 1 and s["333"]["pfr"] == 1 and s["333"]["three_bet"] == 1
    assert s["333"]["aggression_bets_raises"] == 2  # pre raise + flop bet
    # reg: opened (first raise) -> not a 3-bet
    assert s["444"]["pfr"] == 1 and s["444"]["three_bet"] == 0
    # villain only posted the BB -> not voluntary
    assert s["222"]["vpip"] == 0
    # hero called pre then folded the flop facing the cbet
    assert s["111"]["vpip"] == 1 and s["111"]["fold_to_cbet"] == 1


def test_persistence_and_idempotency(tmp_path):
    path = tmp_path / "profiles.json"
    store = PlayerProfileStore(path)
    assert store.update_from_hand(_hand("H1")) > 0
    store.save()

    # reload: stats survive
    store2 = PlayerProfileStore(path)
    assert store2.get("333").hands == 1
    assert store2.get("333").three_bet == 1

    # re-folding the SAME hand id is a no-op (no double counting)
    assert store2.update_from_hand(_hand("H1")) == 0
    assert store2.get("333").hands == 1

    # a new hand id accumulates
    assert store2.update_from_hand(_hand("H2")) > 0
    assert store2.get("333").hands == 2


def test_derive_contexts():
    s = derive_hand_stats(_hand())
    # fish 3-bets (faces the open) with a large size, then makes a small flop cbet on a dry board.
    assert s["333"]["_ctx"]["preflop|-|facing_bet"] == [0, 0, 0, 1]   # agg-large bin
    assert s["333"]["_ctx"]["flop|dry|unopened"] == [0, 0, 1, 0]      # agg-small bin
    # hero calls facing the 3-bet preflop (facing_raise — full preflop depth) then folds the cbet.
    assert s["111"]["_ctx"]["preflop|-|facing_raise"] == [0, 1, 0, 0]  # passive bin, vs a 3bet
    assert s["111"]["_ctx"]["flop|dry|facing_bet"] == [1, 0, 0, 0]     # fold bin


def test_ctx_persistence_and_idempotency(tmp_path):
    path = tmp_path / "profiles.json"
    store = PlayerProfileStore(path)
    store.update_from_hand(_hand("H1"))
    store.save()

    store2 = PlayerProfileStore(path)  # ctx counts survive reload
    assert store2.get("333").ctx_counts["preflop|-|facing_bet"] == [0, 0, 0, 1]
    assert store2.get("333").ctx_total("preflop|-|facing_bet") == 1

    store2.update_from_hand(_hand("H1"))  # idempotent — no double count
    assert store2.get("333").ctx_counts["preflop|-|facing_bet"] == [0, 0, 0, 1]

    store2.update_from_hand(_hand("H2"))  # new hand accumulates the context
    assert store2.get("333").ctx_counts["preflop|-|facing_bet"] == [0, 0, 0, 2]


def test_v1_legacy_file_loads_without_ctx(tmp_path):
    """Back-compat: a pre-migration (scalar-only, no schema_version) file loads, with empty ctx."""
    path = tmp_path / "profiles.json"
    path.write_text(
        '{"profiles": {"999": {"hands": 5, "vpip": 3, "name": "old"}}, "seen_hands": ["X"]}',
        encoding="utf-8",
    )
    store = PlayerProfileStore(path)
    assert store.get("999").hands == 5 and store.get("999").vpip == 3
    assert store.get("999").ctx_counts == {}
    # and the X hand stays deduped
    assert "X" in store.seen_hands
