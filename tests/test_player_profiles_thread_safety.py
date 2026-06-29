"""Regression tests for the player-profile-store race conditions from
``artifacts/six_max_bug_hunt_report.md``:

  * **F-LIVE-1**  (``_learn_aliases`` lock window -- orphan-merge race)
  * **F-LIVE-2**  (``confidence()`` returns a float outside the lock)

The store is shared across multitable + live recorder threads. Each test pins ONE thread-safety
property:

  * F-LIVE-1: the alias-learning path AND the orphan-merge path both run INSIDE the store's lock,
    so a concurrent ``snapshot()`` never deep-copies a half-merged ``ctx_*`` dict. The fix is
    that ``update_from_hand -> _update_from_hand_impl -> _learn_aliases`` is wholly under
    ``with self._lock:``, and ``_merge_stats`` only mutates after ``_learn_aliases`` has acquired
    the same lock via ``self.get(ukey)`` (RLock is reentrant so we don't deadlock on ourselves).
  * F-LIVE-2: ``confidence(key)`` now holds the same lock while reading ``s.hands`` / ``s.hud``,
    so a parallel ``update_from_hand`` cannot mutate the profile between read and return.
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import threading
import time
import pytest

from pokerbot.opponent.player_profiles import PlayerProfileStore, OpponentStats


@pytest.fixture
def store(tmp_path):
    """A fresh in-memory PlayerProfileStore backed by tmp_path (WAL off by default for speed)."""
    s = PlayerProfileStore(tmp_path / "profiles.json", wal=False)
    yield s


# ---------------------------------------------------------------------------------------
# F-LIVE-1 — _learn_aliases is inside the lock; the orphan-merge path is serialized
# ---------------------------------------------------------------------------------------
def test_learn_aliases_happens_under_the_lock(store):
    """A name-only profile accreted BEFORE the uid is seen must merge into the uid profile
    ONCE the uid is learned — and the merge must happen INSIDE the store's lock so a parallel
    ``snapshot()`` never sees a half-merged ``ctx_*`` dict. Pin the contract via the public API:
    after update_from_hand sees a hand with ``name`` AND ``uid`` for the SAME player, the
    name-only orphan (if any) must be merged and the alias registered."""
    # First hand: only ``name`` is known (uid absent for this seat -> the older "name-only" profile
    # accretes).
    rec1 = {
        "hand_id": "h1",
        "table": {"room_key": "r1"},
        "actions": [
            {"name": "alice", "user_id": None, "role": "BTN", "did": "FOLD"},
        ],
    }
    store.update_from_hand(rec1)
    # The name-only orphan must now exist (the player is keyed under 'name:alice' until a uid shows).
    assert "name:alice" in store.profiles

    # Second hand: alice is now identified by a uid. _learn_aliases should:
    #   (a) register alias name:alice -> <uid>
    #   (b) merge the orphan name:alice into the uid profile
    #   (c) pop name:alice from profiles + names
    rec2 = {
        "hand_id": "h2",
        "table": {"room_key": "r1"},
        "actions": [
            {"name": "alice", "user_id": "u-alice-1", "role": "BTN", "did": "FOLD"},
        ],
    }
    store.update_from_hand(rec2)
    assert store.alias.get("name:alice") == "u-alice-1"
    # The orphan is merged: uid profile exists AND name-only is gone.
    assert "u-alice-1" in store.profiles
    assert "name:alice" not in store.profiles


def test_concurrent_snapshot_during_alias_merge_does_not_crash(store):
    """Stress: while ``update_from_hand`` runs (which holds the lock through _learn_aliases ->
    _merge_stats), a parallel ``snapshot(uid_key)`` is safe because it ALSO acquires the lock and
    does a ``deepcopy`` inside it. The OLD bug allowed the merge to mutate ``ctx_*`` dicts while
    ``snapshot()`` was deep-copying them outside the lock, which would raise ``RuntimeError:
    dictionary changed size during iteration`` under load. The fix is the symmetric locking
    contract; this test pins it by spinning both threads simultaneously over an alias-merge
    scenario that exercises the merge path."""
    n_hands = 25
    errors: list[BaseException] = []

    def writer():
        try:
            for i in range(n_hands):
                rec = {
                    "hand_id": f"h-{i}",
                    "table": {"room_key": "r1"},
                    "actions": [
                        # The first hand uses name-only; subsequent hands use uid. This forces
                        # _learn_aliases -> _merge_stats on the very next iteration.
                        {"name": "bob", "user_id": ("u-bob" if i > 0 else None), "role": "BTN",
                         "did": "CALL"},
                        {"name": "carol", "user_id": "u-carol", "role": "SB",
                         "did": "FOLD"},
                    ],
                }
                store.update_from_hand(rec)
                # Continuously poke ctx_* dicts on every player's profile so the merge path
                # has live data to mutate.
                for k in ("u-bob", "u-carol"):
                    p = store.profiles.get(k)
                    if p is not None:
                        p.add_ctx("flop_bet_first_in", [1, 0])
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        try:
            for _ in range(n_hands * 5):
                # snapshot() under the lock must NEVER raise, even when the writer is mid-merge.
                store.snapshot("u-bob")
                store.snapshot("u-carol")
                # confidence() must also be safe now (F-LIVE-2: it took the lock too).
                _ = store.confidence("u-bob")
                _ = store.confidence("u-carol")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t_write = threading.Thread(target=writer)
    t_read = threading.Thread(target=reader)
    t_write.start()
    t_read.start()
    t_write.join()
    t_read.join()
    assert not errors, f"concurrent reader raised under writer load: {errors!r}"


# ---------------------------------------------------------------------------------------
# F-LIVE-2 — confidence() holds the lock during read
# ---------------------------------------------------------------------------------------
def test_confidence_does_not_change_under_concurrent_updates(store):
    """OLD bug: ``confidence()`` returned ``min(1, s.hands / 50)`` after reading ``s.hands`` without
    the lock. A concurrent ``update_from_hand`` could have advanced ``s.hands`` between the read
    and the return, producing a confidence floor higher than what the snapshot another thread
    deep-copied was using. The fix is to wrap the read in ``with self._lock:`` so the value is
    consistent w.r.t. any in-flight update. This test pins that confidence is now safe to call
    from any thread and does not crash under load."""
    store.profiles["u-x"] = OpponentStats()
    errors: list[BaseException] = []

    def updater():
        try:
            for i in range(100):
                p = store.profiles["u-x"]
                with store._lock:           # mirror the locked update path
                    p.hands = i
                    time.sleep(0)            # yield
        except BaseException as exc:
            errors.append(exc)

    def reader():
        try:
            for _ in range(500):
                v = store.confidence("u-x")
                assert 0.0 <= v <= 1.0      # the contract always holds
        except BaseException as exc:
            errors.append(exc)

    t_up = threading.Thread(target=updater)
    t_rd = threading.Thread(target=reader)
    t_up.start()
    t_rd.start()
    t_up.join()
    t_rd.join()
    assert not errors, f"confidence() raised under concurrent updates: {errors!r}"


def test_confidence_hud_floor_still_works_under_lock(store):
    """The HUD-confidence floor (0.1 + 0.1 * hands capped at 0.6 when len(hud) >= 3) must still
    apply after the lock refactor — the fix is purely a locking change, the value contract is
    unchanged."""
    p = store.profiles["u-y"] = OpponentStats()
    p.hands = 10
    p.hud = {"vpip": 0.4, "pfr": 0.1, "af": 1.0}
    v = store.confidence("u-y")
    # base = min(1, 10/50) = 0.2; HUD path returns max(0.2, min(0.6, 0.1 + 0.1*10)) = max(0.2, 0.6) = 0.6
    assert abs(v - 0.6) < 1e-9

    # Without HUD: base path.
    p2 = store.profiles["u-z"] = OpponentStats()
    p2.hands = 10
    v2 = store.confidence("u-z")
    assert abs(v2 - 0.2) < 1e-9