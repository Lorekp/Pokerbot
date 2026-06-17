"""AoFOpponentDB write-ahead crash journal (WAL): lossless recovery between snapshots.

Mirrors tests/test_profile_wal.py for the AoF opponent DB. Unlike PlayerProfileStore the AoF DB
saves on a background thread, so the WAL is rotated under the snapshot lock (save() captures the
journal consistent with counts_snap, then discards it once durable)."""

from __future__ import annotations

from aof_cfr.opponent_db import AoFOpponentDB


def _obs(ident="42"):
    # (identity, infoset_key, pushed)
    return [(ident, "BTN|*|-|d8", True), ("99", "SB|*|A|d8", False)]


def _tot(db):
    return sum(sum(c[1] for c in rows.values()) for rows in db.counts.values())


def test_wal_recovers_unsaved_hands(tmp_path):
    """record_hand deltas after the last save() survive a hard kill via WAL replay on next load."""
    p = tmp_path / "profiles.json"
    db = AoFOpponentDB(path=p)
    db.record_hand("h1", _obs(), names={"42": "Alice"})
    db.save(merge=True)                                  # snapshot has h1; WAL discarded
    assert not db.wal_path.exists()
    db.record_hand("h2", _obs(), names={"42": "Alice"})
    db.record_hand("h3", _obs(), names={"99": "Bob"})
    assert db.wal_path.exists()                          # journaled, not yet snapshotted
    total = _tot(db)
    # hard kill: no save/close. A fresh DB on the same path replays the WAL.
    db2 = AoFOpponentDB(path=p)
    assert _tot(db2) == total
    assert {"h1", "h2", "h3"} <= set(db2.seen_hands)
    assert db2.names == {"42": "Alice", "99": "Bob"}     # names recovered too


def test_save_discards_wal_no_double_count(tmp_path):
    p = tmp_path / "profiles.json"
    db = AoFOpponentDB(path=p)
    for h in ("h1", "h2", "h3"):
        db.record_hand(h, _obs())
    total = _tot(db)
    db.save(merge=True)
    assert not db.wal_path.exists()
    assert not list(p.parent.glob(p.name + ".wal.flushing.*"))
    db2 = AoFOpponentDB(path=p)                          # reload from snapshot only
    assert _tot(db2) == total                            # not doubled


def test_replay_is_idempotent_against_snapshot(tmp_path):
    """A WAL entry already folded into the snapshot is skipped on replay (seen_hands dedup)."""
    p = tmp_path / "profiles.json"
    db = AoFOpponentDB(path=p)
    db.record_hand("h1", _obs())
    db.save(merge=True)
    # forge a stale WAL line for the already-snapshotted hand, then reload.
    db.wal_path.write_text('{"id": "h1", "o": [["42", "BTN|*|-|d8", true]], "n": {}}\n',
                           encoding="utf-8")
    db2 = AoFOpponentDB(path=p)
    assert _tot(db2) == 2                                 # h1 counted once, not twice


def test_wal_torn_tail_line_ignored(tmp_path):
    p = tmp_path / "profiles.json"
    db = AoFOpponentDB(path=p)
    db.record_hand("h1", _obs())
    with open(db.wal_path, "a", encoding="utf-8") as f:
        f.write('{"id": "h2", "o": [["42"')             # killed mid-write
    db2 = AoFOpponentDB(path=p)
    assert "h1" in db2.seen_hands and "h2" not in db2.seen_hands


def test_flushing_file_recovered(tmp_path):
    """A crash AFTER rotate but BEFORE discard leaves a .wal.flushing.* file -> folded on load."""
    p = tmp_path / "profiles.json"
    db = AoFOpponentDB(path=p)
    db.record_hand("h1", _obs())
    flushing = p.with_name(p.name + ".wal.flushing.123-456-789")
    db.wal_path.rename(flushing)                          # simulate rotate-then-crash
    assert not db.wal_path.exists()
    db2 = AoFOpponentDB(path=p)
    assert "h1" in db2.seen_hands and _tot(db2) == 2
    assert not flushing.exists()                          # folded back and cleaned


def test_wal_disabled_flag(tmp_path):
    p = tmp_path / "profiles.json"
    db = AoFOpponentDB(path=p, wal=False)
    db.record_hand("h1", _obs())
    assert not db.wal_path.exists()                       # no journal when disabled
