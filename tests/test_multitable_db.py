"""Multi-table robustness: the shared opponent DB must not lose updates, crash, or corrupt under
concurrent tables, and must survive a corrupt file and co-running processes."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from aof_cfr.opponent_db import AoFOpponentDB, _merge_counts


def test_no_lost_updates_under_concurrency(tmp_path) -> None:
    db = AoFOpponentDB(path=tmp_path / "p.json")
    key = "BTN|AA|-|d8"
    threads_n, per = 8, 500
    barrier = threading.Barrier(threads_n)

    def worker():
        barrier.wait()
        for _ in range(per):
            db.record_action("villain", key, pushed=True)

    ts = [threading.Thread(target=worker) for _ in range(threads_n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    push, total = db.cell("villain", key)
    assert total == threads_n * per, f"lost updates: {total} != {threads_n*per}"
    assert push == threads_n * per


def test_save_during_concurrent_writes_no_crash(tmp_path) -> None:
    db = AoFOpponentDB(path=tmp_path / "p.json")
    stop = threading.Event()
    errors = []

    def writer():
        i = 0
        while not stop.is_set():
            try:
                db.record_action(f"v{i % 20}", f"BTN|AA|-|d{i % 9}", pushed=bool(i % 2))
                i += 1
            except Exception as e:  # noqa: BLE001
                errors.append(("write", e)); return

    def saver():
        for _ in range(40):
            try:
                db.save()
            except Exception as e:  # noqa: BLE001
                errors.append(("save", e)); return

    w = threading.Thread(target=writer); w.start()
    s = threading.Thread(target=saver); s.start()
    s.join()
    stop.set()
    w.join()
    assert not errors, errors  # no "dict changed size during iteration", etc.
    # the persisted file is valid JSON
    json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))


def test_corrupt_file_recovers(tmp_path) -> None:
    p = tmp_path / "p.json"
    p.write_text("{ this is not valid json", encoding="utf-8")
    db = AoFOpponentDB(path=p)  # must NOT raise
    assert db.counts == {}
    assert (tmp_path / "p.json.corrupt").exists(), "corrupt file should be backed up"
    # still usable afterwards
    db.record_action("v", "BTN|AA|-|d8", True)
    db.save()
    assert AoFOpponentDB(path=p).cell("v", "BTN|AA|-|d8") == (1, 1)


def test_merge_counts_is_additive_delta() -> None:
    disk = {"v": {"k": [3, 5]}}
    baseline = {"v": {"k": [1, 2]}}
    current = {"v": {"k": [4, 6]}, "w": {"k2": [1, 1]}}  # our delta: +3 push/+4 tot on k, new w
    merged = _merge_counts(disk, baseline, current)
    assert merged["v"]["k"] == [3 + 3, 5 + 4]  # disk + (current-baseline)
    assert merged["w"]["k2"] == [1, 1]


def test_cross_process_merge_no_clobber(tmp_path) -> None:
    """Two DBs on one path (simulating two engine processes) must not overwrite each other."""
    p = tmp_path / "shared.json"
    a = AoFOpponentDB(path=p)
    b = AoFOpponentDB(path=p)  # both start empty
    a.record_action("p1", "BTN|AA|-|d8", True)
    b.record_action("p2", "SB|KK|F|d8", True)
    a.save(merge=True)
    b.save(merge=True)            # must re-read A's write and merge, not clobber
    c = AoFOpponentDB(path=p)
    assert c.cell("p1", "BTN|AA|-|d8") == (1, 1)
    assert c.cell("p2", "SB|KK|F|d8") == (1, 1)

    # same cell from two processes is additive
    a2 = AoFOpponentDB(path=p)
    b2 = AoFOpponentDB(path=p)
    a2.record_action("p1", "BTN|AA|-|d8", True)
    b2.record_action("p1", "BTN|AA|-|d8", True)
    a2.save(merge=True)
    b2.save(merge=True)
    assert AoFOpponentDB(path=p).cell("p1", "BTN|AA|-|d8") == (3, 3)  # 1 + 1 + 1


def test_record_hand_idempotent_and_threadsafe(tmp_path) -> None:
    db = AoFOpponentDB(path=tmp_path / "p.json")
    obs = [("v", "BTN|AA|-|d8", True), ("w", "SB|KK|A|d8", False)]
    results = []

    def worker():
        results.append(db.record_hand("hand-1", obs))  # same hand_id from many threads

    ts = [threading.Thread(target=worker) for _ in range(10)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sum(1 for r in results if r > 0) == 1, "exactly one thread records the hand"
    assert db.cell("v", "BTN|AA|-|d8") == (1, 1)  # recorded exactly once despite 10 threads


def test_backward_compatible_api(tmp_path) -> None:
    db = AoFOpponentDB(path=tmp_path / "p.json")
    db.record_action("v", "BTN|AA|-|d8", True)
    assert db.cell("v", "BTN|AA|-|d8") == (1, 1)
    assert db.n_obs("v") == 1
    db.counts.clear(); db.seen_hands.clear()  # gauntlet/practice reset pattern still works
    assert db.n_obs("v") == 0
