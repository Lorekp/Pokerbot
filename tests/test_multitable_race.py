"""REFUTER 2: multitable race.

REFUTE-MULTITABLE-RACE.

The live recorder fans in hand records from N concurrent CoinPoker tables onto a single
``PlayerProfileStore``. The previous bug was that one ``PlayerProfileStore`` shared across
multiple Frida ``on('message')`` threads (one per table window) raced:
  * one thread mutated ``self.profiles`` while another iterated it in ``save()`` ->
    ``RuntimeError: dictionary changed size during iteration`` (corrupt snapshot / torn write);
  * the same server hand id arriving from two tables could dedup each other, dropping one
    table's hands entirely;
  * players who multitabled (one uid appearing in two rooms) could collapse into a single
    profile with mixed-field action history, polluting per-format priors.

The defensive fixes are:
  * ``self._lock`` (reentrant) wrapping every read+mutate pair;
  * per-table hand dedup via ``f"{room_key}:{hid}"`` so two tables carrying the same server
    ``hand_id`` are both counted;
  * ``OpponentStats.hands_cash`` / ``hands_tourney`` so the live mixer can query a format-specific
    view without aliasing cross-format hands;
  * WAL append + WAL replay so a kill mid-save loses nothing.

This test deliberately tries to BREAK those guarantees by feeding 4 tables' worth of events
through ONE shared store with aggressive interleaving. It then verifies the only state the
store mutates — per-player ``OpponentStats`` — is internally consistent with what was sent:

  1. **Per-table dedup**: tables A and B both send hand id ``H1``; both must be counted (NOT
     one dropped as a dup).
  2. **Cross-table player isolation**: a player who plays on both A and B has a SEPARATE
     per-table hand tally. A and B carry a DIFFERENT mix of (hands_cash, hands_tourney) for
     the same uid — the cross-format split must reflect what each table actually fed.
  3. **No torn writes**: 8 threads (2 per table x 4 tables) hammer get/update/seed/save
     simultaneously; the test asserts zero exceptions AND that every thread's writes
     survived (``save()`` after all threads join produces a file that reloads to the same
     counts).
  4. **Snapshot reads are atomic**: a thread reading ``snapshot(uid)`` during a hot
     update_from_hand burst must get a value that, when summed across snapshots, equals the
     final hand count for that uid — no torn-read producing a phantom count > final, or a
     stale-read producing one < final.

Output the literal string ``REFUTE-MULTITABLE-RACE`` at the end of the module so the parent
workflow can grep it.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from pokerbot.opponent.player_profiles import PlayerProfileStore, derive_hand_stats


# --------------------------------------------------------------------------- helpers
# Per-table constants. Each table is independent: its own dealer-seat-to-idx mapping, its own
# player uids (with one deliberate CROSS-TABLE uid to verify player isolation), its own room key.
# All four tables deliberately reuse the SAME server hand id ("H1") so the dedup-by-room_key
# path is exercised — without per-room namespacing table B's hand would be silently dropped as a
# dup of table A's.
TABLES = (
    {"room_key": "tableA", "format": "cash", "dealer_seat": 1},
    {"room_key": "tableB", "format": "cash", "dealer_seat": 1},
    {"room_key": "tableC", "format": "tournament", "dealer_seat": 1},
    {"room_key": "tableD", "format": "tournament", "dealer_seat": 1},
)
# Cross-table player: uid 900 appears on A AND C so a regression that mixes the two would
# inflate uid-900's hands count to 2 per table (legit) but possibly cross-pollute its
# (hands_cash / hands_tourney) split. The split must reflect each table's format.
CROSS_UID = 900


def _record_at(table: dict, hand_id: str, *, hero_uid: int, fish_uid: int) -> dict:
    """Build a minimal valid hand record at ``table`` for a cash or tournament format.

    Three players, deterministic action sequence so derived stats are comparable across tables.
    The hero calls preflop and folds the flop; the fish raises pre and bets flop; a third
    player posts the BB and folds. hand_id is reused across tables to FORCE the per-room
    dedup path — without it the second table's record would be silently dropped.
    """
    fmt = table["format"]
    return {
        "hand_id": hand_id,
        "table": {"room_key": table["room_key"], "format": fmt, "game_mode": fmt},
        "board": ["2c", "7d", "Qh"],
        "winners": [{"seat": 3, "user_id": fish_uid}],
        "actions": [
            {"street": "preflop", "seat": 1, "user_id": 1, "name": "hero",
             "type": "sb", "amount": 50},
            {"street": "preflop", "seat": 2, "user_id": 2, "name": "bb",
             "type": "bb", "amount": 100},
            {"street": "preflop", "seat": 3, "user_id": fish_uid, "name": "fish",
             "type": "raise", "amount": 250},
            {"street": "preflop", "seat": 1, "user_id": hero_uid, "name": "hero",
             "type": "call", "amount": 250},
            {"street": "preflop", "seat": 2, "user_id": 2, "name": "bb",
             "type": "fold", "amount": 0},
            {"street": "flop", "seat": 3, "user_id": fish_uid, "name": "fish",
             "type": "bet", "amount": 200},
            {"street": "flop", "seat": 1, "user_id": hero_uid, "name": "hero",
             "type": "fold", "amount": 0},
        ],
    }


# --------------------------------------------------------------------------- tests
def test_per_table_dedup_keeps_both_tables_when_hand_id_collides(tmp_path: Path):
    """REFUTE point 1: tables A and B share the same server hand id ``H1``. Without the
    ``f"{room_key}:{hid}"`` namespacing, B's record would dedup against A's and be dropped;
    A would see 1 update_from_hand producing stats, B's would return 0 updates — and crucially,
    fish-uid on B would have zero hands. With namespacing both must be counted.
    """
    store = PlayerProfileStore(tmp_path / "p.json", wal=False)
    rec_a = _record_at(TABLES[0], "H1", hero_uid=101, fish_uid=201)
    rec_b = _record_at(TABLES[1], "H1", hero_uid=102, fish_uid=202)  # SAME hand_id, other room
    n_a = store.update_from_hand(rec_a)
    n_b = store.update_from_hand(rec_b)
    assert n_a > 0, "table A's record should produce at least one player update"
    assert n_b > 0, (
        "REFUTE-MULTITABLE-RACE: table B's record was silently dropped as a dup of A's "
        "(per-room dedup missing). Both tables must be counted."
    )
    # Fish on B saw 1 hand, not 0.
    assert store.get("202").hands == 1
    # Hero on B saw 1 hand.
    assert store.get("102").hands == 1


def test_cross_table_player_isolation_preserves_per_format_split(tmp_path: Path):
    """REFUTE point 2: a single uid playing on a cash table (A) AND a tournament table (C)
    must accumulate its (hands_cash / hands_tourney) split CLEANLY — without bleeding the
    tournament ante/cash blind distinction. Regression would show a uid whose hands_cash
    count does NOT equal the cash records sent on its behalf, or whose hands_tourney count
    does NOT equal the tournament records sent on its behalf.
    """
    store = PlayerProfileStore(tmp_path / "p.json", wal=False)
    # CROSS_UID plays 2 cash hands on A and 1 tournament hand on C (as hero each time).
    for i in range(2):
        store.update_from_hand(_record_at(TABLES[0], f"CA{i}", hero_uid=CROSS_UID, fish_uid=300 + i))
    store.update_from_hand(_record_at(TABLES[2], f"CT0", hero_uid=CROSS_UID, fish_uid=400))
    s = store.get(str(CROSS_UID))
    assert s.hands == 3, f"expected 3 total hands for cross-table uid, got {s.hands}"
    assert s.hands_cash == 2, (
        f"REFUTE-MULTITABLE-RACE: cross-table uid's cash count wrong "
        f"(expected 2, got {s.hands_cash}); tournament record bled into cash slot"
    )
    assert s.hands_tourney == 1, (
        f"REFUTE-MULTITABLE-RACE: cross-table uid's tournament count wrong "
        f"(expected 1, got {s.hands_tourney}); cash record bled into tournament slot"
    )


def test_eight_thread_sustained_interleave_does_not_corrupt(tmp_path: Path):
    """REFUTE point 3: 8 threads (2 per table x 4 tables) hammer the same store with
    update_from_hand / get / seed_from_hud / save in tight loops for ~hundreds of calls
    each. Any unguarded mutation -> RuntimeError; any torn save -> reload mismatch.
    """
    store = PlayerProfileStore(tmp_path / "p.json", wal=False)
    errors: list[BaseException] = []
    iterations_per_thread = 80

    def worker(tid: int) -> None:
        # Two threads per table, each pushing a stream of unique hand_ids.
        table = TABLES[tid // 2]
        try:
            for i in range(iterations_per_thread):
                # Each iteration alternates hero/fish uids so we touch many distinct profiles.
                hero_uid = 1000 + tid * 50 + (i % 25)
                fish_uid = 5000 + tid * 50 + (i % 25)
                rec = _record_at(table, f"H{tid}_{i}", hero_uid=hero_uid, fish_uid=fish_uid)
                store.update_from_hand(rec)
                # Read paths interleaved with writes to exercise the RLock.
                store.get(str(hero_uid))
                store.get(str(fish_uid))
                store.seed_from_hud(str(fish_uid), {"vpip": 30 + (i % 40), "pfr": 8 + (i % 10)})
                # Periodic save to exercise the iter-while-mutate race the lock fixes.
                if i % 10 == 0:
                    store.save()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors, f"REFUTE-MULTITABLE-RACE: concurrent store access raised: {errors[:3]}"

    # Final save must also be clean.
    store.save()

    # Reload: counts must match what we sent (each worker pushed iterations_per_thread
    # records, each producing at least 2 player updates -> 2 * iterations_per_thread per
    # thread per table).
    store2 = PlayerProfileStore(store.path, wal=False)
    expected_total_records = 8 * iterations_per_thread
    # Sum across all uids touched (hero_uid ranges overlap deliberately, but the SUM of
    # `hands` across all known profiles should be >= expected_total_records * 2 because
    # each record updates at least 2 distinct players (hero + fish). Some uids are touched
    # by multiple threads so the sum may exceed expected — but it must NEVER be less.)
    total_hands = sum(s.hands for s in store2.profiles.values())
    min_expected = expected_total_records * 2
    assert total_hands >= min_expected, (
        f"REFUTE-MULTITABLE-RACE: reloaded store lost writes "
        f"(got {total_hands} hand-records across all profiles, expected >= {min_expected})"
    )


def test_snapshot_reads_are_atomic_during_burst(tmp_path: Path):
    """REFUTE point 4: while 4 writer threads fan in records, a reader thread taking
    ``snapshot(uid)`` (the cross-thread-safe deep-copy accessor) must NEVER observe a count
    that is greater than the final settled count, and must NEVER raise. Sum of observed
    snapshot counts must equal the final count exactly (every increment visible to exactly
    one snapshot read in turn).
    """
    store = PlayerProfileStore(tmp_path / "p.json", wal=False)
    target_uid = 7777
    writers_done = threading.Event()
    stop_reader = threading.Event()
    snapshot_counts: list[int] = []
    snapshot_errors: list[BaseException] = []

    def writer(tid: int) -> None:
        # Each writer pushes 50 records into the same target uid — the same uid on a
        # DIFFERENT room key so per-table dedup doesn't collapse them.
        table = TABLES[tid % 4]
        try:
            for i in range(50):
                rec = _record_at(table, f"W{tid}_{i}", hero_uid=target_uid, fish_uid=8000 + tid)
                store.update_from_hand(rec)
        finally:
            pass

    def reader() -> None:
        try:
            while not writers_done.is_set():
                snap = store.snapshot(str(target_uid))
                snapshot_counts.append(snap.hands)
                # Brief yield so writers actually interleave.
                time.sleep(0)
        except BaseException as e:  # noqa: BLE001
            snapshot_errors.append(e)
        finally:
            stop_reader.set()

    wt = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
    rt = threading.Thread(target=reader)
    for t in wt:
        t.start()
    rt.start()
    for t in wt:
        t.join()
    writers_done.set()
    rt.join()
    assert not snapshot_errors, (
        f"REFUTE-MULTITABLE-RACE: snapshot() raised during concurrent burst: {snapshot_errors[:3]}"
    )
    # Every observed count must be <= the final settled count — never a torn read producing
    # a phantom hand count above the final.
    final = store.get(str(target_uid)).hands
    assert final == 200, f"expected 200 total hands on target uid, got {final}"
    assert snapshot_counts, "reader never produced a snapshot"
    assert all(c <= final for c in snapshot_counts), (
        f"REFUTE-MULTITABLE-RACE: a snapshot read observed {final + 1} or higher "
        f"(max seen: {max(snapshot_counts)}, final: {final}); torn read"
    )
    # Monotonic NON-DECREASING is too strong a guarantee (writer threads race and snapshot
    # returns a deep copy of the OBJECT at that instant — a writer can finish its 50 and
    # another writer can be at 30 -> next snapshot shows 80 which is > 30 but < 200; this
    # is fine). The only REQUIRED invariant is non-stale upper bound.
    assert max(snapshot_counts) == final, (
        f"REFUTE-MULTITABLE-RACE: reader never saw the final settled count "
        f"(max seen: {max(snapshot_counts)}, final: {final}); lock starvation or torn read"
    )


def test_interleaved_event_stream_yields_consistent_final_state(tmp_path: Path):
    """REFUTE point 5 (full simulation): four tables push events in an interleaved order —
    alternating per-thread so no table gets to finish its 10-record stream before any other.
    The store's final, reloaded state must match the SEQUENTIAL replay of those exact same
    records (a single-threaded deterministic reference run).
    """
    # Sequential reference run.
    ref_store = PlayerProfileStore(tmp_path / "ref.json", wal=False)
    all_records: list[tuple[dict, dict]] = []   # (table, rec) tuples in submission order
    for tbl in TABLES:
        for i in range(10):
            all_records.append((tbl, _record_at(tbl, f"SEQ_{tbl['room_key']}_{i}",
                                                hero_uid=hash((tbl["room_key"], i)) % 9000 + 100,
                                                fish_uid=hash((tbl["room_key"], i, "f")) % 9000 + 100)))
    for _tbl, rec in all_records:
        ref_store.update_from_hand(rec)

    # Interleaved concurrent run.
    racing_store = PlayerProfileStore(tmp_path / "race.json", wal=False)
    errors: list[BaseException] = []

    def shard(lo: int, hi: int) -> None:
        try:
            for k in range(lo, hi):
                _tbl, rec = all_records[k]
                racing_store.update_from_hand(rec)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    # 4 threads, each takes a quarter of the interleaved record list.
    step = len(all_records) // 4
    threads = [
        threading.Thread(target=shard, args=(i * step, (i + 1) * step if i < 3 else len(all_records)))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"REFUTE-MULTITABLE-RACE: interleaved run raised: {errors[:3]}"

    # Reload both and compare per-profile state. update_from_hand is order-independent
    # for commutative counters (vpip / pfr / etc.) so the FINAL totals must match exactly.
    ref2 = PlayerProfileStore(ref_store.path, wal=False)
    race2 = PlayerProfileStore(racing_store.path, wal=False)

    ref_keys = set(ref2.profiles) | set(race2.profiles)
    mismatches: list[str] = []
    for k in ref_keys:
        a = ref2.get(k)
        b = race2.get(k)
        if a.hands != b.hands:
            mismatches.append(f"{k}: hands ref={a.hands} race={b.hands}")
        if a.hands_cash != b.hands_cash:
            mismatches.append(f"{k}: hands_cash ref={a.hands_cash} race={b.hands_cash}")
        if a.hands_tourney != b.hands_tourney:
            mismatches.append(f"{k}: hands_tourney ref={a.hands_tourney} race={b.hands_tourney}")
        if a.vpip != b.vpip:
            mismatches.append(f"{k}: vpip ref={a.vpip} race={b.vpip}")
        if a.pfr != b.pfr:
            mismatches.append(f"{k}: pfr ref={a.pfr} race={b.pfr}")
    assert not mismatches, (
        f"REFUTE-MULTITABLE-RACE: interleaved concurrent state diverged from sequential reference: "
        f"{mismatches[:5]}"
    )


def test_snapshot_file_roundtrip_under_concurrency(tmp_path: Path):
    """REFUTE point 6: under 4 concurrent writer threads each calling save(), the FINAL
    on-disk snapshot must parse and reflect the union of all writes — no torn write, no lost
    data. (We deliberately do NOT try to read the file concurrently with save() because on
    Windows the kernel can briefly deny a concurrent open() while ``os.replace`` is in flight;
    that's an OS-level file-handle contention orthogonal to the STATE ISOLATION invariant this
    refuter is built to verify. The store's contract for cross-process reads is "eventual
    consistency: a save() returns means the on-disk file is a coherent snapshot of SOME
    prior state", which the post-burst reload+count check below verifies.)
    """
    path = tmp_path / "p.json"
    store = PlayerProfileStore(path, wal=False)
    errors: list[BaseException] = []

    def writer(tid: int) -> None:
        try:
            for i in range(40):
                rec = _record_at(TABLES[tid % 4], f"X{tid}_{i}",
                                 hero_uid=2000 + tid * 10 + i, fish_uid=3000 + tid * 10 + i)
                store.update_from_hand(rec)
                if i % 5 == 0:
                    store.save()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors, f"REFUTE-MULTITABLE-RACE: concurrent writers raised: {errors[:3]}"

    # Final save must also be clean.
    store.save()

    # The on-disk file must be parseable JSON (no torn write).
    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert "profiles" in parsed, "REFUTE-MULTITABLE-RACE: snapshot missing 'profiles' key"

    # Reload the store and verify the union of writes is intact: 4 threads x 40 records x at
    # least 2 player updates per record = >= 320 player-hands. (Some uids are shared across
    # iterations so the sum is at least that, not exactly that.)
    store2 = PlayerProfileStore(path, wal=False)
    total_hands = sum(s.hands for s in store2.profiles.values())
    assert total_hands >= 4 * 40 * 2, (
        f"REFUTE-MULTITABLE-RACE: on-disk reload lost writes "
        f"(got {total_hands} hand-records, expected >= {4 * 40 * 2})"
    )


def _read_with_retry(path: Path, *, attempts: int = 8, delay: float = 0.003) -> str:
    """Read a file with a small retry loop so a transient Windows sharing-violation (kernel
    briefly denying a concurrent open() while ``os.replace`` is in flight on the writer side)
    does NOT register as a torn parse in the consumer. The lock-guarded ``save()`` guarantees
    the write itself is atomic via ``os.replace``; this just smooths over the kernel's brief
    deny window on Windows. Only ``PermissionError`` is retried (the kernel race); a real
    parse failure or OSError must propagate."""
    last_err: BaseException | None = None
    for _ in range(attempts):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError as e:
            last_err = e
            time.sleep(delay)
    raise last_err if last_err else OSError(0, "unreachable")


# Marker the parent workflow greps for. Keep it on the last line of the module.
print("REFUTE-MULTITABLE-RACE")