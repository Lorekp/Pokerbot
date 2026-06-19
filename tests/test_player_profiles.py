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


def _hand_at_table(hand_id, room_key):
    rec = _hand(hand_id)
    rec["table"] = {"room_key": room_key, "game_mode": "cash"}
    return rec


def test_per_table_hand_dedup_namespacing(tmp_path):
    # Two tables can carry the same server hand id; namespacing by room_key keeps both.
    store = PlayerProfileStore(tmp_path / "p.json", wal=False)
    assert store.update_from_hand(_hand_at_table("H1", "tableA")) > 0
    assert store.update_from_hand(_hand_at_table("H1", "tableB")) > 0   # same id, other table -> counted
    assert store.update_from_hand(_hand_at_table("H1", "tableA")) == 0  # same id+table -> deduped
    # legacy record with no room_key keeps the raw id
    assert store.update_from_hand(_hand("H1")) > 0                      # raw "H1" not yet seen
    assert store.update_from_hand(_hand("H1")) == 0


def test_concurrent_access_does_not_corrupt(tmp_path):
    # N threads hammering get/update/seed/save on ONE shared store must not raise (the multitable race:
    # a dict mutated while save() iterates it -> "dict changed size during iteration").
    import threading
    store = PlayerProfileStore(tmp_path / "p.json", wal=False)
    errors = []

    def worker(t):
        try:
            for i in range(150):
                store.update_from_hand(_hand_at_table(f"H{t}_{i}", f"table{t}"))
                store.get(f"u{(t * 7 + i) % 50}")
                store.seed_from_hud(f"u{i % 20}", {"vpip": 40 + i % 10, "pfr": 12})
                if i % 10 == 0:
                    store.save()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(t,)) for t in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors, f"concurrent store access raised: {errors[:3]}"
    store.save()  # final save must also be clean


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


def _hud_hand(hand_id="H1"):
    """A recorded hand carrying the server HUD block (keyed by str(user_id), as hand_recorder emits)."""
    rec = _hand(hand_id)
    rec["hud"] = {
        "333": {"vpip": 62, "pfr": 9, "wtsd": 41, "wsd": 44, "foldToCbet": 28, "fta": 30},
        "444": {"vpip": 21, "pfr": 18, "wtsd": 28, "wsd": 52},
    }
    return rec


def test_update_from_hand_persists_hud_snapshot(tmp_path):
    """The server HUD block on a recorded hand is folded into OpponentStats.hud via seed_from_hud —
    so the value-sizing/exploit path keys on it for opponents seen only in history, not just live."""
    store = PlayerProfileStore(tmp_path / "p.json")
    store.update_from_hand(_hud_hand("H1"))
    assert store.get("333").hud.get("wsd") == 44 and store.get("333").hud.get("foldToCbet") == 28
    assert store.get("444").hud.get("vpip") == 21
    # survives reload
    store.save()
    assert PlayerProfileStore(tmp_path / "p.json").get("333").hud.get("wsd") == 44


def test_hud_persist_is_idempotent(tmp_path):
    """Re-folding the same hand (dedup) or an unchanged HUD payload does not thrash the dirty flag."""
    store = PlayerProfileStore(tmp_path / "p.json")
    store.update_from_hand(_hud_hand("H1"))
    store._dirty = False
    # same hand_id -> dedup short-circuits before HUD seeding
    assert store.update_from_hand(_hud_hand("H1")) == 0
    assert store.dirty is False
    # a NEW hand id with the IDENTICAL hud payload -> seed_from_hud no-ops on the hud (idempotent)
    store.update_from_hand(_hud_hand("H2"))
    before = dict(store.get("333").hud)
    assert before == _hud_hand()["hud"]["333"]


def test_persisted_hud_drives_value_sizing(tmp_path):
    """End-to-end: a history-only station (no live seed_from_hud RPC) now triggers the value-sizing
    overlay purely from its recorded HUD snapshot — the gap this lever closes."""
    from pokerbot.opponent.value_sizing import value_sizing_logits

    store = PlayerProfileStore(tmp_path / "p.json")
    store.update_from_hand(_hud_hand("H1"))
    adj = value_sizing_logits(store.get("333"), hand_strength=0.85)  # value hand vs a weak-WSD station
    assert adj[-2] > 0.0   # over-bet bucket boosted (NUM_ACTIONS-2 is the top non-all-in size)
    assert adj[1] < 0.0    # checking value discouraged
