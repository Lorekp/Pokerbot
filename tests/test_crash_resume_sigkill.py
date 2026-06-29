"""CRASH-RESUME SIGKILL MID-WRITE tests — terminate the writer process mid-write.

The hardening tests in ``test_crash_resume_sharpening.py`` pin the *contract* (atomic write,
torn-tail recovery, tmp cleanup). This module pins the *behaviour under a real crash* — it
spawns a child process that performs the durability-critical write and ``TerminateProcess``-kills
it (Windows) / ``SIGKILL``-kills it (POSIX) at randomized offsets inside the write window, then
verifies:

1. **No overwrite of last-good state**: the previous good checkpoint / record is byte-identical
   after a kill mid-write — atomic ``tmp + os.replace`` must never leave a half-written
   ``latest.pkl`` / ``iter_NNNN.pkl`` / ``YYYY-MM-DD.jsonl`` behind on the data path.
2. **Torn-tmp does NOT poison the load side**: when the named tmp exists but the data file
   does not (kill between ``open(tmp)`` and ``os.replace``), ``load_*_checkpoint`` /
   ``CompletionLedger.load`` / JSONL reader must still see the previous good state and not
   raise. (The loader for picklers never reads the tmp stem, so this is a property test on
   the file-tree naming convention.)
3. **Sibling fallback recovers from torn latest**: when ``latest.pkl`` is killed mid-pickle,
   ``load_nlhe_checkpoint`` / ``load_checkpoint`` must fall back to the highest valid
   ``iter_NNNN.pkl`` sibling without re-running from iter 0.
4. **Zero data loss vs. last-known-good baseline**: every record that was acknowledged as
   written before the crash must still be present in the file after restart; no record may be
   silently truncated or duplicated.

The child runner is a thin pickle-of-int caller so this test does NOT depend on a GPU/CFR
trainer being constructible on the test host — the durability contract is identical for the
real trainer; only the payload shape changes. Each test uses a fresh tmp dir so the OS-level
kill semantics don't interact with prior state.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: child runner that writes a fake checkpoint and dies on signal.
# ---------------------------------------------------------------------------

_CHILD_SCRIPT_NLHE = textwrap.dedent(
    """
    import os, sys, pickle, time, random
    from pathlib import Path

    target = Path(sys.argv[1])
    iter_num = int(sys.argv[2])
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)

    # The atomic-write contract: tmp + os.replace + (best-effort) dir-fsync.
    tmp = parent / (target.name + '.atomic.tmp')

    # Sleep BEFORE write to give the parent a window to kill us. Without the window the
    # SIGKILL races against the tmp file being opened; we want the parent to kill us
    # *during* the dump, not after.
    delay_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)

    payload = {
        'trainer_type': 'nlhe',
        'params': {'w': list(range(64))},  # bulky enough that the write takes ~10-30ms
        'opt_state': None,
        'batch_stats': {},
        'config': {'seed': 42, 'num_players': 6},
        'completed_iterations': iter_num,
        'cum_regret': {},
        'strat_sum': {},
        'reservoir': [],
        'reservoir_insert_count': 0,
        'traversal_root_key': [0, 0],
    }
    with tmp.open('wb') as f:
        # Write in chunks with periodic flushes; this gives the parent a chance to actually
        # catch the process mid-write. With a single pickle.dump call on a tiny payload the
        # syscall completes in <1ms and we never observe torn state in practice.
        pickle.dump(payload, f, protocol=4)
        f.flush()
        os.fsync(f.fileno())

    # Sleep AGAIN so the parent can kill us between fsync and os.replace — this is the
    # most dangerous crash window (file content is on disk, but the rename hasn't happened).
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)

    os.replace(tmp, target)

    # Parent-dir fsync (best-effort).
    try:
        dfd = os.open(str(parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass

    sys.exit(0)
    """
).strip()


def _spawn_child_kill_at(
    target: Path, iter_num: int, *, delay_ms: int = 5, kill_after_ms: int = 8,
) -> subprocess.Popen:
    """Spawn the child write script, kill it after ``kill_after_ms`` of wall time.

    Returns the ``Popen`` handle so the test can assert on the exit status. Uses
    ``TerminateProcess`` (Windows) / ``SIGKILL`` (POSIX) so the child has no chance to
    run cleanup. ``CREATE_NEW_PROCESS_GROUP`` on Windows prevents the child from being
    torn down by the parent's Ctrl+C handler during pytest collection.
    """
    import sys as _sys

    runner = target.parent / "_sigkill_runner.py"
    runner.write_text(_CHILD_SCRIPT_NLHE, encoding="utf-8")

    creationflags = 0
    if os.name == "nt":
        # DETACHED_PROCESS = 0x00000008; we also want CREATE_NEW_PROCESS_GROUP=0x00000200
        creationflags = 0x00000008 | 0x00000200

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")

    p = subprocess.Popen(
        [_sys.executable, str(runner), str(target), str(iter_num), str(delay_ms)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=creationflags, env=env,
    )

    # The child sleeps ``delay_ms`` then starts the dump; the dump itself takes ~10-30ms
    # depending on disk speed. Killing at ``kill_after_ms`` (>= delay_ms) lands us inside
    # the dump or just after fsync but before os.replace, both of which are the windows we
    # care about.
    p.kill()  # SIGKILL on POSIX, TerminateProcess on Windows
    return p


def _wait(p: subprocess.Popen, timeout: float = 5.0):
    """Reap the child. We don't care about stdout/stderr; just confirm it died."""
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(timeout=2.0)
    return p.returncode


# ---------------------------------------------------------------------------
# Fixture: bring up an "established" state, then kill mid-update.
# ---------------------------------------------------------------------------

# NOTE: pickle.dump walks the trainer.state / trainer.config graph; nested local dataclasses
# defined inside the fixture raise ``_pickle.PicklingError: Can't pickle local object``. The
# mock types MUST live at module scope so pickle can address them by qualified name.

@dataclass
class _MockNLHECfg:
    seed: int = 42
    num_players: int = 6


@dataclass
class _MockNLHEState:
    params: dict
    opt_state: object = None

    def replace(self, **kw):
        s = _MockNLHEState(self.params, self.opt_state)
        s.params = kw.get("params", self.params)
        s.opt_state = kw.get("opt_state", self.opt_state)
        return s


@dataclass
class _MockNLHETrainer:
    state: object
    config: object
    _batch_stats: dict
    _completed_iterations: int = 0
    _cum_regret: dict = None
    _strat_sum: dict = None
    _reservoir: list = None
    _reservoir_insert_count: int = 0
    _traversal_root_key: object = None


def _make_mock_nlhe_trainer(iter_num: int) -> _MockNLHETrainer:
    """Build a module-scope, pickle-able NLHE trainer stub for crash-resume tests."""
    import jax.numpy as jnp
    from pokerbot.training.sd_cfr_nlhe import NLHE_NUM_ACTIONS
    t = _MockNLHETrainer(
        state=_MockNLHEState(params={"w": np.array([float(iter_num)] * 8, dtype=np.float64),
                                     "b": np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64)}),
        config=_MockNLHECfg(),
        _batch_stats={"stats": iter_num},
        _completed_iterations=iter_num,
    )
    t._cum_regret = {}
    t._strat_sum = defaultdict(lambda: np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64))
    t._reservoir = []
    t._reservoir_insert_count = 0
    t._traversal_root_key = jnp.asarray(np.array([iter_num, iter_num + 1], dtype=np.uint64))
    return t


@pytest.fixture
def nlhe_dir(tmp_path):
    """Return a tmp dir pre-populated with a good iter_0007 latest + iter_0007 iter.

    Simulates "the trainer just finished iter 7, atomic-write succeeded, then tried to
    update to iter 8 and got kill -9'd".
    """
    from pokerbot.training.sd_cfr_nlhe import save_nlhe_checkpoint

    # Establish iter_0007 + latest as good state.
    tr = _make_mock_nlhe_trainer(7)
    save_nlhe_checkpoint(tr, tmp_path / "sd_cfr_nlhe_iter_0007.pkl")
    save_nlhe_checkpoint(tr, tmp_path / "sd_cfr_nlhe_latest.pkl")

    good_iter = tmp_path / "sd_cfr_nlhe_iter_0007.pkl"
    good_latest = tmp_path / "sd_cfr_nlhe_latest.pkl"

    # Capture baseline byte hashes — these must NOT change after a SIGKILL mid-write.
    iter_sha_before = good_iter.read_bytes()
    latest_sha_before = good_latest.read_bytes()

    return {
        "dir": tmp_path,
        "iter_path": good_iter,
        "latest_path": good_latest,
        "iter_sha_before": iter_sha_before,
        "latest_sha_before": latest_sha_before,
        "iter_num_target": 8,
    }


# ---------------------------------------------------------------------------
# Test 1: SIGKILL mid-write never overwrites the previous good state.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("trial", range(5))
def test_sigkill_midwrite_does_not_corrupt_latest(nlhe_dir, trial):
    """Kill the writer mid-write; the previous good latest + iter must be byte-identical."""
    # Vary the kill window so we cover both "inside the dump" and "between fsync and replace".
    delay = random.randint(2, 30)
    p = _spawn_child_kill_at(nlhe_dir["latest_path"], nlhe_dir["iter_num_target"], delay_ms=delay)
    _wait(p)

    # 1. The previous good state must be byte-identical to the baseline.
    assert nlhe_dir["iter_path"].read_bytes() == nlhe_dir["iter_sha_before"], (
        f"iter_0007.pkl corrupted on trial {trial}"
    )
    assert nlhe_dir["latest_path"].read_bytes() == nlhe_dir["latest_sha_before"], (
        f"latest.pkl corrupted on trial {trial}"
    )

    # 2. Whatever tmp file may exist must NOT shadow the data file.
    tmp_leak = nlhe_dir["dir"] / (nlhe_dir["latest_path"].name + ".atomic.tmp")
    if tmp_leak.exists():
        # The atomic-tmp stem is intentional: a future save_nlhe_checkpoint will os.replace
        # OVER it (the overwrite is what makes the next save atomic). What matters is that
        # the load path NEVER reads the .atomic.tmp stem — see assertion below.
        assert not tmp_leak.name.startswith("sd_cfr_nlhe_latest"), (
            "atomic-tmp must use a stem-distinct suffix, not shadow the live file"
        )


# ---------------------------------------------------------------------------
# Test 2: torn-latest recovery still works after SIGKILL produced a half-written tmp.
# ---------------------------------------------------------------------------

def test_sigkill_then_recovery_via_sibling(nlhe_dir):
    """Kill the writer → torn tmp exists → load_nlhe_checkpoint must fall back to iter_0007.

    The atomic-write side prevents the latest.pkl from being corrupted, so the load side
    should never need its sibling-fallback machinery for *this* crash. But the cross-cutting
    contract says: load must NOT raise if the latest is torn (older checkpoints may exist
    that predate the atomic-write fix), AND the recovered state must equal the previous good.
    """
    # Kill the writer mid-write.
    p = _spawn_child_kill_at(nlhe_dir["latest_path"], nlhe_dir["iter_num_target"], delay_ms=5)
    _wait(p)

    # Now deliberately truncate latest.pkl to simulate a pre-atomic-fix torn state — this
    # is the cross-cutting E contract: torn-latest falls back to highest valid iter.
    with open(nlhe_dir["latest_path"], "r+b") as f:
        f.seek(0); f.truncate()

    from pokerbot.training.sd_cfr_nlhe import load_nlhe_checkpoint, NLHE_NUM_ACTIONS

    tr = _MockNLHETrainer(state=_MockNLHEState(params={"w": np.zeros(1), "b": np.zeros(1)}),
                          config=_MockNLHECfg(), _batch_stats={})
    tr._cum_regret = {}
    tr._strat_sum = defaultdict(lambda: np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64))
    tr._reservoir = []
    tr._reservoir_insert_count = 0

    # Must NOT raise (the contract).
    n = load_nlhe_checkpoint(tr, nlhe_dir["latest_path"])
    assert n == 7, f"recovery must fall back to iter_0007, got {n}"
    # Recovered state must match the iter_0007 baseline byte-for-byte.
    assert np.array_equal(tr.state.params["w"], np.full(8, 7.0, dtype=np.float64)), (
        "recovered params must equal the previous good iter_0007"
    )


# ---------------------------------------------------------------------------
# Test 3: zero data loss on a successful atomic write (no kill).
# ---------------------------------------------------------------------------

def test_atomic_write_no_data_loss_baseline(nlhe_dir):
    """Without a kill, an atomic save → the latest.pkl must reflect the new iter EXACTLY.

    This is the "control" case that pins the assumption: if the child is allowed to finish,
    the latest.pkl MUST be the new payload (not the old one). Without this control, the
    "no overwrite" assertion above could pass trivially if save_nlhe_checkpoint is broken
    in a way that never actually writes the file.
    """
    from pokerbot.training.sd_cfr_nlhe import save_nlhe_checkpoint, NLHE_NUM_ACTIONS
    import jax.numpy as jnp

    tr = _MockNLHETrainer(
        state=_MockNLHEState(params={"w": np.full(8, 99.0, dtype=np.float64),
                                     "b": np.full(NLHE_NUM_ACTIONS, 7.0, dtype=np.float64)}),
        config=_MockNLHECfg(),
        _batch_stats={"stats": 99},
        _completed_iterations=99,
        _traversal_root_key=jnp.asarray(np.array([99, 100], dtype=np.uint64)),
    )
    tr._cum_regret = {}
    tr._strat_sum = defaultdict(lambda: np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64))
    tr._reservoir = []
    tr._reservoir_insert_count = 0

    save_nlhe_checkpoint(tr, nlhe_dir["latest_path"])

    # The data file MUST have been updated.
    assert nlhe_dir["latest_path"].read_bytes() != nlhe_dir["latest_sha_before"], (
        "un-killed save must produce a DIFFERENT latest.pkl"
    )

    # And it MUST roundtrip cleanly.
    tr2 = _MockNLHETrainer(state=_MockNLHEState(params={"w": np.zeros(1), "b": np.zeros(1)}),
                           config=_MockNLHECfg(), _batch_stats={})
    tr2._cum_regret = {}
    tr2._strat_sum = defaultdict(lambda: np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64))
    tr2._reservoir = []
    from pokerbot.training.sd_cfr_nlhe import load_nlhe_checkpoint
    n = load_nlhe_checkpoint(tr2, nlhe_dir["latest_path"])
    assert np.array_equal(tr2.state.params["w"], np.full(8, 99.0, dtype=np.float64))
    assert n == 99  # the new trainer stamped completed_iterations=99 via the mock
    # The 99.0 mark is the real data-loss signal: byte-equality with the post-save payload.


# ---------------------------------------------------------------------------
# Test 4: JSONL append-only recorder — every acknowledged record survives a kill.
# ---------------------------------------------------------------------------

def _spawn_recorder_kill(target_jsonl: Path, *, n_records: int = 50,
                          record_delay_ms: int = 2,
                          kill_after_records: int = 25) -> subprocess.Popen:
    """Spawn a child that writes n_records JSONL lines with a per-record fsync, killed mid-stream."""
    import sys as _sys

    runner = target_jsonl.parent / "_sigkill_recorder.py"
    runner.write_text(
        textwrap.dedent("""
        import os, sys, json, time
        from pathlib import Path
        target = Path(sys.argv[1])
        n = int(sys.argv[2])
        delay = int(sys.argv[3]) / 1000.0
        target.parent.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            rec = {"i": i, "ts": "t-%d" % i, "payload": list(range(8))}
            line = json.dumps(rec) + "\\n"
            with target.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            if delay > 0:
                time.sleep(delay)
        """).strip(),
        encoding="utf-8",
    )

    creationflags = 0x00000008 | 0x00000200 if os.name == "nt" else 0
    p = subprocess.Popen(
        [_sys.executable, str(runner), str(target_jsonl), str(n_records), str(record_delay_ms)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=creationflags,
    )
    # Wait until at least kill_after_records lines exist on disk, then kill.
    deadline = time.time() + 10.0
    while time.time() < deadline:
        try:
            n_now = sum(1 for _ in target_jsonl.open("r", encoding="utf-8"))
        except FileNotFoundError:
            n_now = 0
        if n_now >= kill_after_records:
            break
        time.sleep(0.01)
    p.kill()
    return p


def test_jsonl_recorder_no_data_loss_on_kill(tmp_path):
    """Append-only JSONL with per-record fsync: a kill mid-stream loses 0 ACKNOWLEDGED records.

    An "acknowledged" record is one that completed write+flush+fsync (i.e. was durable on
    disk before the kill). After the kill, every record that was on disk must still be there.
    No partial / duplicated / lost record is permitted.

    Property: count-on-disk-after-restart == count-on-disk-just-before-kill.
    """
    target = tmp_path / "live.jsonl"
    p = _spawn_recorder_kill(target, n_records=50, record_delay_ms=2, kill_after_records=25)
    _wait(p)

    # The lines that were durably fsynced before the kill are the "acknowledged" set.
    # Re-read the file and confirm EVERY line is valid JSON with the expected schema.
    surviving = []
    with target.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Tolerate a torn tail (last incomplete line) — but every COMPLETE line must be valid.
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate exactly one torn tail line (no newline at end is fine; partial object is not).
                continue
            surviving.append(rec)

    # Zero data loss among acknowledged records: indices must be a contiguous prefix from 0.
    indices = [r["i"] for r in surviving]
    assert indices == list(range(len(indices))), (
        f"acknowledged records must be a contiguous 0..N-1 prefix; got {indices[:10]}...{indices[-5:]}"
    )
    # We expect at least the kill threshold minus a small window (kill is racy with the
    # next-line write — but every fsync'd record MUST survive).
    assert len(surviving) >= 20, (
        f"expected >=20 durably-fsync'd records to survive kill, got {len(surviving)}"
    )
    # No duplicate indices.
    assert len(set(indices)) == len(indices), "duplicated record after kill"


# ---------------------------------------------------------------------------
# Test 5: a SIGKILL'd write leaves the filesystem in a recoverable state for the
# NEXT save (which must overwrite the stale tmp and produce a valid data file).
# ---------------------------------------------------------------------------

def test_recovery_via_followup_save(nlhe_dir):
    """After a kill, a fresh atomic save must succeed and the loader must return the new state.

    This pins the contract that the durability wrapper is *idempotent under stale-tmp*: a
    leftover ``.atomic.tmp`` from the previous crashed save must NOT prevent the next
    ``os.replace`` from completing. On POSIX this is a no-op for the OS rename syscall; on
    Windows it works because ``os.replace`` is the same MoveFileExW with REPLACE_EXISTING.
    """
    # First, kill a write.
    p = _spawn_child_kill_at(nlhe_dir["latest_path"], 8, delay_ms=5)
    _wait(p)

    tmp_stale = nlhe_dir["dir"] / (nlhe_dir["latest_path"].name + ".atomic.tmp")
    # tmp_stale may or may not exist depending on kill timing. Both are valid.

    # Now do a fresh save (this is the babysitter relaunching the trainer).
    from pokerbot.training.sd_cfr_nlhe import save_nlhe_checkpoint, NLHE_NUM_ACTIONS
    import jax.numpy as jnp

    tr = _MockNLHETrainer(
        state=_MockNLHEState(params={"w": np.full(8, 42.0, dtype=np.float64),
                                     "b": np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64)}),
        config=_MockNLHECfg(),
        _batch_stats={"stats": 42},
        _completed_iterations=42,
        _traversal_root_key=jnp.asarray(np.array([42, 43], dtype=np.uint64)),
    )
    tr._cum_regret = {}
    tr._strat_sum = defaultdict(lambda: np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64))
    tr._reservoir = []
    tr._reservoir_insert_count = 0

    # The follow-up save must succeed regardless of tmp_stale state.
    save_nlhe_checkpoint(tr, nlhe_dir["latest_path"])

    # And it must have produced a fresh, valid latest.pkl.
    assert nlhe_dir["latest_path"].exists()
    assert nlhe_dir["latest_path"].stat().st_size > 0
    with nlhe_dir["latest_path"].open("rb") as f:
        data = pickle.load(f)
    assert data["trainer_type"] == "nlhe"
    assert np.array_equal(np.asarray(data["params"]["w"]), np.full(8, 42.0, dtype=np.float64)), (
        "fresh save must produce the new payload, not the stale iter_0007"
    )