"""Persistent per-opponent push/fold database for the AoF exploit layer.

The core IP of the exploitative engine: every push/fold an opponent makes is a labelled
observation at a fully-specified infoset (position, hand bucket, history, stack-depth — the
v2 infoset key). We store sufficient statistics ``[push_count, total_count]`` per
``(identity, infoset_key)`` across every session ever played, so the profiler can estimate
an opponent's strategy (and how exploitable they are) before the first hand of a new session.

Storage is a single JSON keyed by a persistent identity (``user_id`` when available, else
``name:<username>``), idempotent by ``hand_id``. Designed to survive **multi-tabling**:

  * **Thread-safe.** All mutators/readers take a re-entrant lock, and ``save`` serializes a
    consistent snapshot — so N tables recording concurrently never lose a ``+=`` update and
    never crash ``save`` with "dict changed size during iteration".
  * **Crash-safe.** ``save`` writes a temp file, fsyncs, and atomically ``os.replace``s; a
    corrupt/partial file on ``load`` is backed up and the DB starts fresh instead of throwing.
  * **Multi-process-safe.** ``save(merge=True)`` does a read-modify-write under a best-effort
    cross-process file lock, applying only this process's delta since the last persist — so two
    engine processes sharing one file don't clobber each other (counts are additive).

Fine for thousands of players; migrate the same accessors to sqlite past ~10k.
Play-money / own-account / bot-arena use only.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = "data/aof_player_profiles/profiles.json"
SCHEMA_VERSION = 1
SEEN_HANDS_CAP = 200_000   # dedup needs only a recent window; counts are already folded


def _spec_hash() -> str:
    from aof_cfr.infoset import infoset_key_spec_hash
    return infoset_key_spec_hash()


def identity_key(user_id: object | None, name: str | None) -> str | None:
    """Persistent identity: user_id if present, else name:<username>, else None (anonymous)."""
    if user_id:
        return str(user_id)
    if name:
        return f"name:{name}"
    return None


@contextmanager
def _file_lock(lock_path: Path, timeout: float = 10.0, poll: float = 0.03):
    """Best-effort cross-process lock via an O_EXCL lock file. Yields True if held, else False
    (caller falls back to a plain atomic write — we never block forever or raise)."""
    fd = None
    start = time.monotonic()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            try:  # steal an obviously-stale lock so a crashed process can't wedge us
                if time.monotonic() - os.path.getmtime(lock_path) > max(timeout, 30.0):
                    os.unlink(lock_path)
                    continue
            except OSError:
                pass
            if time.monotonic() - start > timeout:
                yield False
                return
            time.sleep(poll)
    try:
        yield True
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(lock_path)
        except OSError:
            pass


def _ordered_set(*iterables) -> dict:
    """An insertion-ordered set (dict keys). Earlier iterables = older entries, kept first; a
    later duplicate does NOT move an id. Used so the seen_hands recency window is FIFO by arrival,
    not lexicographic by id (hand ids are namespaced ``table_id:hid`` and do NOT sort by time)."""
    out: dict = {}
    for it in iterables:
        for x in it:
            if x not in out:
                out[x] = None
    return out


def _cap_recent(seen: dict, cap: int) -> dict:
    """Keep the most-recently-added ``cap`` ids (evict from the front), preserving order."""
    if len(seen) <= cap:
        return seen
    return dict.fromkeys(list(seen)[-cap:])


@dataclass
class AoFOpponentDB:
    path: Path = field(default_factory=lambda: Path(DEFAULT_PATH))
    # identity -> {infoset_key: [push_count, total_count]}
    counts: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    seen_hands: dict = field(default_factory=dict)   # insertion-ordered set (FIFO recency, see _ordered_set)
    _dirty: bool = False
    wal: bool = True                                  # crash journal between snapshots (see _wal_append)
    _lock: threading.RLock = field(default=None, init=False, repr=False, compare=False)
    _io_lock: threading.Lock = field(default=None, init=False, repr=False, compare=False)
    _persisted: dict = field(default=None, init=False, repr=False, compare=False)
    _wal_replaying: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._lock = threading.RLock()      # guards the in-memory state (the source of truth)
        self._io_lock = threading.Lock()    # serializes saves (no temp-file races, no merge interleave)
        self.load()

    # ---- persistence ----------------------------------------------------------------
    def load(self) -> None:
        with self._lock:
            self._persisted = {}
            raw = None
            if self.path.exists():
                try:
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                    if not isinstance(raw, dict):
                        raise ValueError("profiles root is not an object")
                except (OSError, ValueError) as exc:
                    self._backup_corrupt(exc)
                    raw = None
            if raw is not None:
                # Refuse-with-backup on an infoset-key-spec mismatch: counts keyed under a
                # different spec would silently mix incompatible cells. Legacy files without the
                # header are accepted (they predate the header, not the current spec).
                stored_spec = raw.get("infoset_key_spec_hash")
                if stored_spec is not None and stored_spec != _spec_hash():
                    self._backup_corrupt(ValueError(
                        f"infoset key spec {stored_spec} != current {_spec_hash()}"))
                else:
                    self.counts = _coerce_counts(raw.get("counts"))
                    self.names = dict(raw.get("names") or {})
                    self.seen_hands = dict.fromkeys(raw.get("seen_hands") or [])  # preserve order
                    self._persisted = copy.deepcopy(self.counts)  # baseline for delta-merge saves
        self._fold_sidecars()  # recover any lock-timeout deltas other runs left behind
        self._replay_wal()     # recover hands journaled but not yet snapshotted (hard-kill safety)

    def _backup_corrupt(self, exc: Exception) -> None:
        try:
            bad = self.path.with_suffix(self.path.suffix + ".corrupt")
            os.replace(self.path, bad)
            print(f"AoFOpponentDB: corrupt profile {self.path} ({exc}); backed up to {bad}, starting fresh")
        except OSError:
            pass

    def save(self, *, merge: bool = False) -> None:
        """Persist atomically. ``merge=True`` does a read-modify-write under a cross-process lock,
        writing this process's delta on top of whatever other processes wrote (multi-process safe).

        ``self.counts`` (lock-protected) is the in-process source of truth and is NOT mutated here,
        so concurrent ``record_*`` from other tables are never lost or double-counted by a save."""
        with self._io_lock:  # serialize saves: no temp-file race, no merge interleave
            with self._lock:
                counts_snap = copy.deepcopy(self.counts)  # consistent snapshot + matching baseline
                names_snap = dict(self.names)
                seen_snap = dict(self.seen_hands)         # ordered copy
                baseline = self._persisted or {}
                # Rotate the WAL under the SAME lock as the snapshot so the rotated file holds
                # exactly the hands captured in counts_snap; appends racing in from the ingest
                # thread land in a fresh WAL and are never discarded by this save.
                flushing = self._rotate_wal_locked()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            out_counts, out_names, out_seen = counts_snap, names_snap, seen_snap
            if merge:
                lock_path = self.path.with_suffix(self.path.suffix + ".lock")
                with _file_lock(lock_path) as held:
                    if not held:
                        # NEVER clobber other processes' writes on a lock timeout: park OUR
                        # delta in a sidecar; the next successful locked save/load folds it.
                        self._write_sidecar(_delta_counts(baseline, counts_snap),
                                            names_snap, seen_snap)
                        with self._lock:
                            self._persisted = counts_snap   # delta is parked, not pending
                            self._dirty = False
                        self._discard_flushing(flushing)    # data is durable in the sidecar now
                        return
                    disk = self._read_disk()
                    extra = self._collect_sidecars()
                    if disk:  # fold our delta over other processes' on-disk contributions
                        out_counts = _merge_counts(disk.get("counts", {}), baseline, counts_snap)
                        out_seen = _ordered_set(disk.get("seen_hands") or [], seen_snap)  # disk older, ours newer
                        out_names = dict(disk.get("names") or {}); out_names.update(names_snap)
                    for sc in extra:                          # fold parked deltas too
                        out_counts = _merge_counts(out_counts, {}, sc.get("counts", {}))
                        out_seen = _ordered_set(out_seen, sc.get("seen_hands") or [])
                        out_names.update(sc.get("names") or {})
                    self._atomic_write(out_counts, out_names, out_seen)
                    self._remove_sidecars()
            else:
                self._atomic_write(out_counts, out_names, out_seen)
            with self._lock:
                self._persisted = counts_snap  # baseline = what we just persisted our part of
                self._dirty = False
            self._discard_flushing(flushing)   # rotated WAL is now folded into the snapshot

    # -- sidecar deltas (lock-timeout fallback that cannot clobber) -----------------------
    def _sidecar_glob(self):
        return self.path.parent.glob(self.path.name + ".delta.*")

    def _write_sidecar(self, counts_delta: dict, names: dict, seen) -> None:
        sc = self.path.with_name(
            f"{self.path.name}.delta.{os.getpid()}-{threading.get_ident()}-{int(time.time()*1000)}")
        payload = {"counts": counts_delta, "names": names, "seen_hands": list(seen)}  # preserve order
        try:
            sc.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            print(f"AoFOpponentDB: merge lock timed out; parked delta in {sc.name}")
        except OSError:
            pass  # best effort — worst case the delta persists at the next successful save

    def _collect_sidecars(self) -> list[dict]:
        out = []
        for sc in self._sidecar_glob():
            try:
                raw = json.loads(sc.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    out.append(raw)
            except (OSError, ValueError):
                continue
        return out

    def _remove_sidecars(self) -> None:
        for sc in self._sidecar_glob():
            try:
                os.unlink(sc)
            except OSError:
                pass

    def _fold_sidecars(self) -> None:
        """At load: adopt parked deltas into memory (they'll persist on the next save)."""
        scs = self._collect_sidecars()
        if not scs:
            return
        with self._lock:
            for sc in scs:
                for ident, rows in (sc.get("counts") or {}).items():
                    mine = self.counts.setdefault(ident, {})
                    for k, v in rows.items():
                        cell = mine.setdefault(k, [0, 0])
                        cell[0] += int(v[0]); cell[1] += int(v[1])
                for hid in sc.get("seen_hands") or []:
                    self.seen_hands.setdefault(hid, None)
                self.names.update(sc.get("names") or {})
            self._dirty = True
        self._remove_sidecars()

    def _read_disk(self) -> dict | None:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            return None  # corrupt -> caller skips merge, just overwrites with our snapshot

    def _atomic_write(self, counts, names, seen) -> None:
        seen_list = list(seen)                          # insertion-ordered (dict keys); NOT re-sorted
        if len(seen_list) > SEEN_HANDS_CAP:             # keep the most-recent window (FIFO by arrival)
            seen_list = seen_list[-SEEN_HANDS_CAP:]
        payload = {"schema_version": SCHEMA_VERSION,
                   "infoset_key_spec_hash": _spec_hash(),
                   "counts": counts, "names": names, "seen_hands": seen_list}
        tmp = self.path.with_suffix(self.path.suffix + f".tmp{os.getpid()}-{threading.get_ident()}")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=0)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)  # atomic on same volume (POSIX + Windows)

    # -- write-ahead crash journal (lossless between snapshots) ---------------------------
    # Mirrors PlayerProfileStore's WAL: record_hand appends its delta to <path>.wal.jsonl
    # immediately (cheap), load() replays anything not yet in the snapshot (idempotent by
    # hand_id), and save() rotates+discards it. A hard kill between saves loses nothing.
    @property
    def wal_path(self) -> Path:
        return self.path.with_name(self.path.name + ".wal.jsonl")

    def _wal_append(self, entry: dict) -> None:
        """Append one journal line. Caller holds ``self._lock`` (serializes multi-table writers)."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.wal_path, "a", encoding="utf-8") as wf:
                wf.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except (OSError, TypeError):
            pass  # journaling is best-effort; the snapshot is still the source of truth

    def _rotate_wal_locked(self) -> Path | None:
        """Atomically move the active WAL aside so its lines match the just-taken snapshot; fresh
        appends create a new WAL. Returns the rotated path (discarded once the snapshot is durable)."""
        if not self.wal or not self.wal_path.exists():
            return None
        dest = self.path.with_name(
            f"{self.path.name}.wal.flushing.{os.getpid()}-{threading.get_ident()}-{int(time.time()*1000)}")
        try:
            os.replace(self.wal_path, dest)
            return dest
        except OSError:
            return None

    def _discard_flushing(self, path: Path | None) -> None:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _replay_wal(self) -> None:
        if not self.wal:
            return
        # Fold any crashed mid-flush files back into the active WAL so the next save truncates them
        # (don't apply-then-delete: a crash before that save would lose them).
        for fp in sorted(self.path.parent.glob(self.path.name + ".wal.flushing.*")):
            try:
                data = fp.read_text(encoding="utf-8")
                with open(self.wal_path, "a", encoding="utf-8") as wf:
                    wf.write(data)
                fp.unlink()
            except OSError:
                pass
        if not self.wal_path.exists():
            return
        self._wal_replaying = True   # record_hand applies to memory but does NOT re-append
        try:
            for line in self.wal_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue  # torn tail line from a mid-write kill — ignore
                obs = [tuple(o) for o in (e.get("o") or [])]
                self.record_hand(e.get("id"), obs, names=e.get("n") or None)
        except OSError:
            pass
        finally:
            self._wal_replaying = False

    # ---- mutation -------------------------------------------------------------------
    def record_action(self, identity: str, infoset_key: str, pushed: bool) -> None:
        """Fold one observed push/fold decision into the counts for ``identity``."""
        if not identity:
            return
        with self._lock:
            rows = self.counts.setdefault(identity, {})
            cell = rows.setdefault(infoset_key, [0, 0])
            cell[0] += 1 if pushed else 0
            cell[1] += 1
            self._dirty = True

    def record_hand(self, hand_id: str | None, observations: list[tuple[str, str, bool]],
                    names: dict[str, str] | None = None) -> int:
        """Idempotently fold one hand's observations ``[(identity, infoset_key, pushed)]`` (and
        optional ``{identity: display_name}``) into the profile.

        Re-recording the same ``hand_id`` is a no-op (restart / log reprocessing safe).
        Returns the number of observations recorded (0 if the hand was already seen)."""
        with self._lock:
            if hand_id is not None:
                if hand_id in self.seen_hands:
                    return 0
                self.seen_hands[hand_id] = None
                if len(self.seen_hands) > SEEN_HANDS_CAP + 50_000:   # bounded memory: keep the
                    self.seen_hands = _cap_recent(self.seen_hands, SEEN_HANDS_CAP)  # recent window (FIFO)
            n = 0
            for identity, infoset_key, pushed in observations:
                if not identity:
                    continue
                rows = self.counts.setdefault(identity, {})
                cell = rows.setdefault(infoset_key, [0, 0])
                cell[0] += 1 if pushed else 0
                cell[1] += 1
                n += 1
            if names:
                for ident, nm in names.items():
                    if ident and nm:
                        self.names[ident] = nm
            self._dirty = True
            # Crash journal (only dedup-safe hands with an id): append the delta immediately so a
            # hard kill before the next snapshot replays losslessly. Skipped during replay.
            if self.wal and not self._wal_replaying and hand_id is not None:
                self._wal_append({"id": hand_id,
                                  "o": [list(o) for o in observations],
                                  "n": names or {}})
            return n

    def set_name(self, identity: str, name: str | None) -> None:
        if identity and name:
            with self._lock:
                self.names[identity] = name

    # ---- reads (locked snapshots so they never iterate a mutating structure) --------
    def cell(self, identity: str, infoset_key: str) -> tuple[int, int]:
        """(push_count, total_count) for an infoset; (0, 0) if unseen."""
        with self._lock:
            c = self.counts.get(identity, {}).get(infoset_key)
            return (int(c[0]), int(c[1])) if c else (0, 0)

    def n_obs(self, identity: str) -> int:
        """True decision count for an identity.

        When hand-agnostic aggregate cells exist (``pos|*|hist|d`` — the live keying), count
        ONLY those: showdown reveals additionally record the same decision under its exact
        per-bucket key, and counting both would double-count decisions and inflate confidence.
        Per-bucket-only profiles (tests / offline analysis) keep the plain total."""
        with self._lock:
            rows = self.counts.get(identity, {})
            agg = sum(t for k, (_p, t) in rows.items() if "|*|" in k)
            if agg:
                return agg
            return sum(t for _, t in rows.values())

    def cells_for(self, identity: str) -> dict[str, tuple[int, int]]:
        """Snapshot of every (infoset_key -> (push, total)) for an identity — one lock grab
        instead of hundreds of ``cell()`` calls when an estimator scans buckets/depths."""
        with self._lock:
            return {k: (int(v[0]), int(v[1])) for k, v in self.counts.get(identity, {}).items()}

    def n_decisions(self, identity: str) -> int:
        """Distinct infosets observed for an identity (a cheap data-volume signal)."""
        with self._lock:
            return len(self.counts.get(identity, {}))

    def identities(self) -> list[str]:
        with self._lock:
            return list(self.counts.keys())

    @property
    def dirty(self) -> bool:
        return self._dirty


def _coerce_counts(raw) -> dict[str, dict[str, list[int]]]:
    """Defensively parse stored counts (skip malformed cells instead of crashing)."""
    out: dict[str, dict[str, list[int]]] = {}
    if not isinstance(raw, dict):
        return out
    for ident, rows in raw.items():
        if not isinstance(rows, dict):
            continue
        clean: dict[str, list[int]] = {}
        for k, v in rows.items():
            try:
                clean[k] = [int(v[0]), int(v[1])]
            except (TypeError, ValueError, IndexError, KeyError):
                continue
        if clean:
            out[str(ident)] = clean
    return out


def _merge_counts(disk: dict, baseline: dict, current: dict) -> dict:
    """disk + (current - baseline), per (identity, infoset) cell — additive sufficient stats."""
    out = copy.deepcopy(_coerce_counts(disk))
    for ident, rows in current.items():
        base_rows = baseline.get(ident, {})
        o = out.setdefault(ident, {})
        for k, cur in rows.items():
            b = base_rows.get(k, (0, 0))
            cell = o.setdefault(k, [0, 0])
            cell[0] += int(cur[0]) - int(b[0])
            cell[1] += int(cur[1]) - int(b[1])
    return out


def _delta_counts(baseline: dict, current: dict) -> dict:
    """current - baseline per cell (non-zero cells only) — this process's unsaved delta."""
    out: dict = {}
    for ident, rows in current.items():
        base_rows = baseline.get(ident, {})
        for k, cur in rows.items():
            b = base_rows.get(k, (0, 0))
            dp, dt = int(cur[0]) - int(b[0]), int(cur[1]) - int(b[1])
            if dp or dt:
                out.setdefault(ident, {})[k] = [dp, dt]
    return out
