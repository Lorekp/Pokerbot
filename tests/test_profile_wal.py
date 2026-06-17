"""Phase 6 (G6): PlayerProfileStore crash journal (WAL) + throttled snapshots + ledger cap."""

from __future__ import annotations

import json
import os

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

from pokerbot.opponent.player_profiles import SEEN_HANDS_CAP, PlayerProfileStore


def _rec(hid: str, name: str = "v1") -> dict:
    return {
        "hand_id": hid,
        "board": [],
        "actions": [{"street": "preflop", "type": "call", "user_id": 1, "name": name},
                    {"street": "preflop", "type": "raise", "user_id": 2, "name": "v2"}],
    }


def test_wal_recovers_unsaved_hands(tmp_path):
    """Updates after the last save() survive a hard kill via WAL replay on the next load."""
    p = tmp_path / "profiles.json"
    store = PlayerProfileStore(p)
    store.update_from_hand(_rec("h1"))
    store.save()                                   # snapshot has h1; WAL truncated
    assert not store.wal_path.exists()
    store.update_from_hand(_rec("h2"))
    store.update_from_hand(_rec("h3"))
    assert store.wal_path.exists()                 # journaled, NOT yet snapshotted
    # hard kill: no save(). A fresh store must see all three hands.
    store2 = PlayerProfileStore(p)
    assert store2.get("1").hands == 3
    assert {"h1", "h2", "h3"} <= store2.seen_hands
    # replay did not re-append the WAL
    lines = [l for l in store2.wal_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2


def test_wal_torn_tail_line_ignored(tmp_path):
    p = tmp_path / "profiles.json"
    store = PlayerProfileStore(p)
    store.update_from_hand(_rec("h1"))
    with open(store.wal_path, "a", encoding="utf-8") as f:
        f.write('{"hand_id": "h2", "actions": [{"street"')  # kill mid-write
    store2 = PlayerProfileStore(p)
    assert store2.get("1").hands == 1              # torn line skipped, good line replayed


def test_save_truncates_wal_and_rotates_bak(tmp_path):
    p = tmp_path / "profiles.json"
    store = PlayerProfileStore(p)
    store.update_from_hand(_rec("h1"))
    store.save()
    store.update_from_hand(_rec("h2"))
    store.save()
    assert not store.wal_path.exists()
    assert p.with_suffix(p.suffix + ".bak").exists()
    # corrupt the snapshot -> load falls back to .bak (h1 state) + nothing crashes
    p.write_text("{corrupt", encoding="utf-8")
    store3 = PlayerProfileStore(p)
    assert store3.get("1").hands >= 1


def test_save_throttled_cadence(tmp_path):
    p = tmp_path / "profiles.json"
    store = PlayerProfileStore(p)
    saved = 0
    for i in range(45):
        store.update_from_hand(_rec(f"h{i}"))
        saved += int(store.save_throttled(min_hands=20, min_seconds=10 ** 9))
    assert saved == 2                              # at 20 and 40 updates, not 45 times
    # the unsnapshotted tail is still safe in the WAL
    store2 = PlayerProfileStore(p)
    assert store2.get("1").hands == 45


def test_seen_hands_cap(tmp_path):
    p = tmp_path / "profiles.json"
    store = PlayerProfileStore(p, wal=False)
    store.seen_hands = {f"hand_{i:07d}" for i in range(SEEN_HANDS_CAP + 500)}
    store.profiles  # noqa: B018
    store._dirty = True
    store.save()
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert len(raw["seen_hands"]) == SEEN_HANDS_CAP
    assert "hand_0000000" not in raw["seen_hands"]  # oldest dropped
    assert f"hand_{SEEN_HANDS_CAP + 499:07d}" in raw["seen_hands"]


def test_battery_store_path_unaffected(tmp_path):
    """The battery's profile-store usage (direct profiles[] assignment + save) is WAL-neutral."""
    from pokerbot.opponent.stats import OpponentStats

    p = tmp_path / "profiles.json"
    store = PlayerProfileStore(p)
    store.profiles["archetype:hs_lag"] = OpponentStats(hands=7)
    store.save()
    store2 = PlayerProfileStore(p)
    assert store2.get("archetype:hs_lag").hands == 7
