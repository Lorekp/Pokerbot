#!/usr/bin/env python3
"""Local smoke test for the arm-2 spot-resilience checkpoint driver (2026-07-05 operational
addendum, artifacts/blueprint_v4_arm2_amendment.md sec 10).

Renders the REAL bootstrap via launch_native_cfr_training_spot.bootstrap_body (fullhist+MRR+v3
arm-2 shape), extracts the generated drive_native_cfr.py + ckpt_upload.py heredocs, and runs them
against a FAKE filesystem-backed boto3 and a FAKE deterministic battery_native trainer:

  1. UNINTERRUPTED run  -> reference final state + expected boundary schedule (rolling CKPT_EVERY
     merged with the GATE_EVERY ladder) + S3 retention (gates kept, rolling pruned to newest 3).
  2. KILLED run         -> hard-kill the driver right after a mid-run marker lands, then plant a
     PARTIAL upload (blob, no marker) and a CORRUPT candidate (marker with wrong sha) ABOVE it.
  3. RESUME run         -> must reject the corrupt candidate, ignore the partial, resume from the
     newest VERIFIED marker, and finish with a final state IDENTICAL to run 1.
  4. DELTA run          -> hybrid-trigger mode: chain-root base + size-triggered deltas between
     iter-exact gate bases; + rebalance-recommendation flag banks one extra mid-window delta
     through the normal detached path (chunked train loop).
  5. DELTA kill+resume  -> base + contiguous verified delta chain -> identical final state.
  6. EMERGENCY FLUSH    -> reclaim flag mid-window -> synchronous delta + marker banked at a
     NON-cadence iter, exit 0; a fresh resume applies base + emergency delta -> identical final.
  7. PRESTAGE valid     -> bootstrap-prestaged base+chain consumed (S3 blobs deleted to PROVE no
     download happened) -> identical final.
  8. PRESTAGE stale     -> manifest points at a superseded marker -> clean fallback download.
  9. CHAIN CONTINUATION -> post-resume re-arm: first ckpt after resume is a DELTA (seq N+1),
     not a full base (the load-wipes-arming bug).
 10. HYBRID SIZE        -> size trigger fires mid-window at an ARBITRARY (odd) iter; chain stays
     contiguous; kill+resume -> identical final.
 11. HYBRID TIME        -> time trigger (small CKPT_SECS) fires when dirty accumulation is slow;
     rolling checkpoints land at non-gate iters; identical final.
 12. GATE EXACTNESS     -> gate bases + snapshots land ITER-EXACT even when a rolling trigger
     fired one iter earlier.
 13. MIN-GAP GUARD      -> CKPT_MIN_GAP_SECS suppresses back-to-back triggers under an
     always-firing size cap.

HYBRID ROLLING CADENCE (2026-07-06): on the delta path the rolling-checkpoint cadence is no
longer CKPT_EVERY iters but the time/size triggers (POKERBOT_CKPT_SECS /
POKERBOT_CKPT_MAX_DIRTY_GB, min-interval POKERBOT_CKPT_MIN_GAP_SECS). Delta scenarios pin all
three explicitly (see delta_env) and pad fake deltas to a fixed 60 B/dirty-row
(FAKE_DELTA_PAD_BPR), freezing the driver's EMA bytes/row at its seed so size-trigger fire
points are DETERMINISTIC: with CHUNK=1 the trigger fires when 120 * n_dirty_iters >= cap bytes
(cap 240 B => every 2nd mid-window iter counted from the last checkpoint, cap 120 B => every
mid-window iter, cap 360 B => every 3rd).

No AWS, no network, no real trainer: this validates the driver/uploader/marker/retention logic
only (the trainer-level resume bit-identity is covered by the ab_train Rust smoke; the chunked
t.train() bit-identity by the golden-gate sha check).

Usage: python scripts/blueprint2/test_arm2_ckpt_driver.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ITERS = 12
CKPT_EVERY = 2
GATE_EVERY = 5
# merged boundary schedule the driver must hit (rolling 2s + gate ladder 5s + final)
EXPECT_BOUNDARIES = [2, 4, 5, 6, 8, 10, 12]
GATES = {5, 10, 12}  # 12 = final (>= ITERS)

FAKE_BOTO3 = '''
import json, os, shutil

_ROOT = os.environ["FAKE_S3_ROOT"]

class _Body:
    def __init__(self, data):
        self._d = data
    def read(self):
        return self._d

class _Paginator:
    def paginate(self, Bucket=None, Prefix=""):
        base = os.path.join(_ROOT, Bucket)
        out = []
        if os.path.isdir(base):
            for dirpath, _, files in os.walk(base):
                for fn in files:
                    full = os.path.join(dirpath, fn)
                    key = os.path.relpath(full, base).replace(os.sep, "/")
                    if key.startswith(Prefix):
                        out.append({"Key": key, "Size": os.path.getsize(full)})
        yield {"Contents": sorted(out, key=lambda o: o["Key"])}

class _Client:
    def _p(self, b, k):
        return os.path.join(_ROOT, b, k.replace("/", os.sep))
    def put_object(self, Bucket, Key, Body):
        p = self._p(Bucket, Key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        data = Body if isinstance(Body, (bytes, bytearray)) else Body.read()
        with open(p, "wb") as f:
            f.write(data)
    def get_object(self, Bucket, Key):
        p = self._p(Bucket, Key)
        if not os.path.exists(p):
            raise FileNotFoundError(Key)
        with open(p, "rb") as f:
            return {"Body": _Body(f.read())}
    def upload_file(self, path, Bucket, Key, Config=None):
        p = self._p(Bucket, Key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        shutil.copyfile(path, p)
    def upload_fileobj(self, fobj, Bucket, Key, Config=None):
        p = self._p(Bucket, Key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            shutil.copyfileobj(fobj, f)
    def download_file(self, Bucket, Key, path, Config=None):
        shutil.copyfile(self._p(Bucket, Key), path)
    def delete_object(self, Bucket, Key):
        try:
            os.remove(self._p(Bucket, Key))
        except FileNotFoundError:
            pass
    def copy(self, src, Bucket, Key, Config=None):
        shutil.copyfile(self._p(src["Bucket"], src["Key"]), self._p(Bucket, Key))
    def get_paginator(self, name):
        return _Paginator()

def client(name, region_name=None):
    return _Client()
'''

FAKE_TRANSFER = '''
class TransferConfig:
    def __init__(self, **kw):
        pass
'''

FAKE_BN = '''
import hashlib, os, struct, time

class NlheCFRTrainer:
    """Deterministic fake: state = the SET of iteration indices trained so far. A checkpoint stores that
    set; a resumed run reproduces the uninterrupted final state IFF resume restores the exact set.

    Delta model (checkpoint-speed R2): `_dirty` is the set of iters trained since the last reset_dirty.
    A full base stores ALL iters; a delta stores the dirty set (its absolute members). Applying base +
    delta chain = union of the stored sets, which equals the straight-run set IFF the chain is contiguous
    -- exactly the property the real NLDLT001 delta has (dirty rows are absolute post-floor values).
    Serialized form carries the sorted iter set so save/load is order-independent and byte-deterministic.
    """
    MAGIC_BASE = b"FAKE0001"
    MAGIC_DELTA = b"FAKD0001"
    def __init__(self, *a, **k):
        self._iters = 0
        self._set = set()          # every iter index trained (the "table")
        self._dirty = None         # None => delta tracking unarmed; else set since last reset
    # -- helpers --
    def _blob(self, magic, members, base_iter=None, seq=None):
        payload = struct.pack("<Q", len(members))
        for m in sorted(members):
            payload += struct.pack("<Q", m)
        hdr = magic + struct.pack("<Q", self._iters)
        if magic == self.MAGIC_DELTA:
            hdr += struct.pack("<QQ", base_iter, seq)
        return hdr + payload + hashlib.sha256(payload).digest()
    def _members(self, blob, off):
        (n,) = struct.unpack("<Q", blob[off:off + 8]); off += 8
        ms = set()
        for _ in range(n):
            (m,) = struct.unpack("<Q", blob[off:off + 8]); off += 8
            ms.add(m)
        return ms
    # -- train --
    def train(self, step):
        for _ in range(int(step)):
            self._iters += 1
            self._set.add(self._iters)
            if self._dirty is not None:
                self._dirty.add(self._iters)
            time.sleep(float(os.environ.get("FAKE_TRAIN_SLEEP", "0.02")))
    # -- full base --
    def save_bytes(self):
        return self._blob(self.MAGIC_BASE, self._set)
    def save_bytes_to_file(self, path):
        b = self.save_bytes()
        with open(path, "wb") as f:
            f.write(b)
        return len(b)
    def load_bytes(self, blob):
        if not blob or blob[:8] != self.MAGIC_BASE:
            return False
        self._iters = struct.unpack("<Q", blob[8:16])[0]
        self._set = self._members(blob, 16)
        # MODEL install_loaded_tables (lib.rs ~1670): `self.tab = tab` replaces the WHOLE table,
        # and a deserialized NlheTables has dirty: None -- so an arm_delta_tracking() done BEFORE
        # the load is silently WIPED (the 2026-07-06 profile-run bug, i-0f9aa667065ac1388: every
        # save_delta_to_file returned None and each rolling checkpoint silently fell through to a
        # 60 GB full base, rate 0.194 it/s). Both load_bytes AND load_bytes_from_file (which
        # funnels here, like the real pymethods) must wipe.
        self._dirty = None
        return True
    def load_bytes_from_file(self, path):
        # Round-3 resume entrypoint (real trainer: mmap + parallel deserialize). Fake: read + load.
        try:
            with open(path, "rb") as f:
                return self.load_bytes(f.read())
        except OSError:
            return False
    # -- delta --
    def arm_delta_tracking(self):
        if self._dirty is None:
            self._dirty = set()
    def dirty_counts(self):
        n = len(self._dirty) if self._dirty is not None else 0
        return (n, n)
    def reset_dirty(self):
        if self._dirty is not None:
            self._dirty = set()
    def save_delta_to_file(self, path, base_iter, seq):
        if self._dirty is None:
            return None
        b = self._blob(self.MAGIC_DELTA, self._dirty, base_iter, seq)
        # FAKE_DELTA_PAD_BPR (hybrid-trigger sim): zero-pad the delta file to exactly BPR bytes
        # per dirty ROW (dirty_counts() reports (n, n) => 2n rows), pinning the driver's EMA
        # bytes/row at its 60 B/row seed so size-trigger fire points are deterministic. The
        # padding sits after the sha tail: apply_delta parses length-prefixed members and never
        # reads it, and the .ok marker's bytes/sha are computed from the written file, so
        # verification stays consistent.
        bpr = float(os.environ.get("FAKE_DELTA_PAD_BPR", "0"))
        if bpr > 0:
            want = int(bpr * 2 * len(self._dirty))
            if want > len(b):
                b += bytes(want - len(b))
        with open(path, "wb") as f:
            f.write(b)
        return len(b)
    @staticmethod
    def delta_header(blob):
        if not blob or blob[:8] != b"FAKD0001" or len(blob) < 32:
            return None
        iters, base_iter, seq = struct.unpack("<QQQ", blob[8:32])
        return (base_iter, iters, seq)
    def apply_delta(self, blob, base_iter, prev_seq):
        h = self.delta_header(blob)
        if h is None:
            return None
        (h_base, h_iters, h_seq) = h
        if h_base != base_iter or h_seq != prev_seq + 1 or h_iters <= self._iters:
            return None
        members = self._members(blob, 32)   # overwrite/union == last-writer-wins on the iter set
        self._set |= members
        self._iters = h_iters
        return (h_iters, h_seq)
    # -- misc --
    def iters_done(self):
        return self._iters
    def n_infosets(self):
        return 10 * self._iters
    def n_cum_infosets(self):
        return 12 * self._iters
    def is_hogwild(self):
        return False
    def save_current_strategy_bytes(self):
        return b"SNAP" + str(self._iters).encode()
    def coverage(self, hands, seed):
        return (hands, [4, 3, 2, 1], [1, 1, 1, 0], 0)
    def regret_stats(self):
        return (1, 1, 1, 31.6, 125.9, 281.8, 794.3, 1.7e5, 7.4e3, 7.4e3)
    def emd_stamp(self):
        return "fake"
    def river_ochs_stamp(self):
        return "fake"
'''


def render_driver():
    spec = importlib.util.spec_from_file_location(
        "launcher", ROOT / "scripts" / "launch_native_cfr_training_spot.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    v3 = dict(lcfr=False, signed_regrets=True, lcfr_d=250, lcfr_w=10000, prune_c=-50000.0,
              prune_start=2500, prune_explore=0.05, regret_floor=None, snapshot_avg=True,
              snap_start=5, cov_hands=8, regstats=True)
    boot = m.bootstrap_body(
        bucket="fakebucket", prefix="pokerbot/prod6000/native_cfr", region="eu-central-1",
        code_key="x/code.tar.gz", run_subdir="smoke_arm2", iters=ITERS, n_deals=4,
        num_buckets=200, depth_edges=[], checkpoint_every=CKPT_EVERY, gate_every=GATE_EVERY,
        fullhist_key=True, mrr=True, work_steal=True, v3=v3, row_budget=10**9)

    def extract(name, eof):
        mm = re.search(r"cat > /opt/pokerbot/%s <<'%s'\n(.*?)\n%s\n"
                       % (re.escape(name), eof, eof), boot, re.S)
        assert mm, f"{name} heredoc missing from rendered bootstrap"
        return mm.group(1)

    return extract("drive_native_cfr.py", "PYEOF"), extract("ckpt_upload.py", "UPEOF")


def setup_env(tmp: Path, tag: str):
    """One isolated sandbox: fake-S3 root, stub packages, rewritten driver/uploader."""
    box = tmp / tag
    (box / "opt").mkdir(parents=True)
    (box / "log").mkdir()
    (box / "s3").mkdir()
    (box / "tmp").mkdir()  # sandboxed /tmp for the spot reclaim/rebalance flag files
    stub = box / "stubs"
    (stub / "boto3" / "s3").mkdir(parents=True)
    (stub / "boto3" / "__init__.py").write_text(FAKE_BOTO3, encoding="utf-8")
    (stub / "boto3" / "s3" / "__init__.py").write_text("", encoding="utf-8")
    (stub / "boto3" / "s3" / "transfer.py").write_text(FAKE_TRANSFER, encoding="utf-8")
    (stub / "battery_native.py").write_text(FAKE_BN, encoding="utf-8")

    driver, uploader = render_driver()
    opt = str(box / "opt").replace("\\", "/")
    log = str(box / "log").replace("\\", "/")
    tmpd = str(box / "tmp").replace("\\", "/")
    for src in (driver, uploader):
        pass
    driver = (driver.replace("/opt/pokerbot", opt)
                    .replace("/var/log/ckpt_upload.log", log + "/ckpt_upload.log")
                    .replace("/var/log/native-cfr-train.log", log + "/train.log")
                    .replace("/tmp/spot_reclaim", tmpd + "/spot_reclaim")
                    .replace("/tmp/spot_rebalance", tmpd + "/spot_rebalance"))
    uploader = uploader.replace("/opt/pokerbot", opt)
    (box / "opt" / "drive_native_cfr.py").write_text(driver, encoding="utf-8")
    (box / "opt" / "ckpt_upload.py").write_text(uploader, encoding="utf-8")
    (box / "log" / "train.log").write_text("smoke train log\n", encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(stub)
    env["FAKE_S3_ROOT"] = str(box / "s3")
    # mirror the bootstrap shell's v3 env exports (v3_env_block) — the driver banner reads these
    env["POKERBOT_SNAPSHOT_AVG"] = "1"
    env["POKERBOT_ROW_BUDGET"] = "0"
    env["POKERBOT_SIGNED_REGRETS"] = "1"
    env["POKERBOT_PRUNE_C"] = "-50000.0"
    env["POKERBOT_PRUNE_START"] = "2500"
    env["POKERBOT_PRUNE_EXPLORE"] = "0.05"
    return box, env


def delta_env(env, cap_gb, secs="0", mingap="0", chunk="1", pad="60"):
    """Delta-mode env with the HYBRID rolling-trigger knobs pinned (2026-07-06). Defaults:
    time trigger OFF (secs=0), min-gap OFF (deterministic size-only cadence), CHUNK=1 (every
    iter is a mid-window trigger opportunity), fake deltas padded to 60 B/dirty-row so the
    size estimate is exactly 120 * n_dirty_iters bytes. cap_gb is the POKERBOT_CKPT_MAX_DIRTY_GB
    string: "2.4e-7" = 240 B (fires every 2nd dirty iter), "1.2e-7" = 120 B (every dirty iter),
    "3.6e-7" = 360 B (every 3rd), "0" = size trigger off."""
    return dict(env, POKERBOT_DELTA_CKPT="1", POKERBOT_CKPT_CHUNK=chunk,
                POKERBOT_CKPT_SECS=secs, POKERBOT_CKPT_MAX_DIRTY_GB=cap_gb,
                POKERBOT_CKPT_MIN_GAP_SECS=mingap, FAKE_DELTA_PAD_BPR=pad)


def s3_keys(box: Path):
    base = box / "s3" / "fakebucket"
    out = []
    if base.is_dir():
        for p in base.rglob("*"):
            if p.is_file():
                out.append(str(p.relative_to(base)).replace(os.sep, "/"))
    return sorted(out)


def marker_iters(box: Path):
    return sorted(int(re.search(r"iter_(\d+)\.nlcfr\.ok$", k).group(1))
                  for k in s3_keys(box) if k.endswith(".nlcfr.ok"))


def all_marker_iters(box: Path):
    """Base AND delta markers (iters) -- used for kill-timing in delta mode."""
    return sorted(int(re.search(r"iter_(\d+)\.(?:nlcfr|nldlt)\.ok$", k).group(1))
                  for k in s3_keys(box) if k.endswith(".nlcfr.ok") or k.endswith(".nldlt.ok"))


def final_blob(box: Path, it: int):
    p = (box / "s3" / "fakebucket" / "pokerbot" / "prod6000" / "native_cfr" / "smoke_arm2"
         / "checkpoints" / f"iter_{it:06d}.nlcfr")
    return p.read_bytes()


def wait_uploads(box: Path, want: set[int], timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if want <= set(marker_iters(box)):
            return
        time.sleep(0.2)
    raise AssertionError(f"timeout waiting for markers {want}; have {marker_iters(box)}")


def run_driver(box: Path, env, sleep="0.02", kill_after_marker=None, marker_fn=marker_iters,
               touch_flag_after=None, flag_delay=0.3):
    env = dict(env, FAKE_TRAIN_SLEEP=sleep)
    proc = subprocess.Popen([sys.executable, str(box / "opt" / "drive_native_cfr.py")],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if touch_flag_after is not None:
        # graceful-reclaim scenario: wait for a marker, touch the flag file mid-window, then let
        # the driver notice it at a chunk boundary and run its own emergency flush + exit.
        want, flag = touch_flag_after
        t0 = time.time()
        while time.time() - t0 < 120:
            if want in marker_fn(box):
                time.sleep(flag_delay)
                Path(flag).write_text("1", encoding="utf-8")
                break
            if proc.poll() is not None:
                out, _ = proc.communicate()
                raise AssertionError(f"driver exited before marker {want}:\n{out}")
            time.sleep(0.05)
        else:
            proc.kill()
            raise AssertionError(f"marker {want} never appeared (flag scenario)")
        out, _ = proc.communicate(timeout=300)
        return proc.returncode, out
    if kill_after_marker is None:
        out, _ = proc.communicate(timeout=300)
        return proc.returncode, out
    # poll for the requested marker, then hard-kill mid-training (simulated spot reclaim)
    t0 = time.time()
    while time.time() - t0 < 120:
        if kill_after_marker in marker_fn(box):
            proc.kill()
            out, _ = proc.communicate(timeout=60)
            return -9, out
        if proc.poll() is not None:
            out, _ = proc.communicate()
            raise AssertionError(f"driver exited before marker {kill_after_marker}:\n{out}")
        time.sleep(0.1)
    proc.kill()
    raise AssertionError(f"marker {kill_after_marker} never appeared")


def main():
    t_start = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="arm2_ckpt_smoke_"))
    print(f"[smoke] sandbox {tmp}", flush=True)
    try:
        # ---------- run 1: uninterrupted reference ----------
        print("[smoke] 1/13 uninterrupted reference run...", flush=True)
        box1, env1 = setup_env(tmp, "ref")
        rc, out = run_driver(box1, env1)
        assert rc == 0, f"reference run rc={rc}\n{out}"
        assert "FRESH start" in out, out
        wait_uploads(box1, {ITERS})
        ref_final = final_blob(box1, ITERS)
        m1 = set(marker_iters(box1))
        # Round-1+ direct-to-file path logs CKPT_SERIALIZE (was CKPT_LOCAL pre-round-1).
        boundaries = sorted(int(x) for x in re.findall(
            r"CKPT_(?:SERIALIZE|LOCAL|DELTA) iters_done=(\d+)", out))
        assert boundaries == EXPECT_BOUNDARIES, f"boundaries {boundaries} != {EXPECT_BOUNDARIES}"
        assert GATES <= m1, f"gate checkpoints missing: have {m1}"
        # retention: rolling (non-gate) pruned to newest 3 -> iter 2 must be gone eventually
        wait_uploads(box1, m1)  # let stragglers finish
        time.sleep(1.0)
        rolling = sorted(i for i in marker_iters(box1) if i % GATE_EVERY != 0 and i < ITERS)
        assert len(rolling) <= 3, f"rolling retention failed: {rolling}"
        assert 2 not in marker_iters(box1), f"iter 2 should be pruned: {marker_iters(box1)}"
        snaps = [k for k in s3_keys(box1) if k.endswith(".nlsnap")]
        snap_iters = sorted(int(re.search(r"snap_(\d+)\.nlsnap$", k).group(1)) for k in snaps)
        assert snap_iters == [5, 10, 12], f"snapshots must land ONLY at gate iters: {snap_iters}"
        assert "V3 effective" in out and "PRUNE_C=-50000.0" in out, "v3 prune banner missing"
        print(f"[smoke]   OK boundaries={boundaries} markers={sorted(m1)} "
              f"kept={marker_iters(box1)} snaps={snap_iters}", flush=True)

        # ---------- run 2: kill mid-run after the iter-6 marker ----------
        print("[smoke] 2/13 kill-mid-run (simulated spot reclaim after iter-6 marker)...", flush=True)
        box2, env2 = setup_env(tmp, "killed")
        rc, out = run_driver(box2, env2, sleep="0.5", kill_after_marker=6)
        assert rc != 0
        have = set(marker_iters(box2))
        assert 6 in have and ITERS not in have, f"unexpected markers after kill: {have}"
        # plant a PARTIAL upload (blob, no marker) and a CORRUPT candidate (marker w/ wrong sha)
        ck = (box2 / "s3" / "fakebucket" / "pokerbot" / "prod6000" / "native_cfr" / "smoke_arm2"
              / "checkpoints")
        (ck / "iter_000007.nlcfr").write_bytes(b"PARTIAL-UPLOAD-NO-MARKER")
        garbage = b"FAKE0001" + b"\x09" + bytes(7) + bytes(4128)
        (ck / "iter_000009.nlcfr").write_bytes(garbage)
        (ck / "iter_000009.nlcfr.ok").write_text(json.dumps(
            {"iters_done": 9, "bytes": len(garbage), "sha256": "0" * 64}), encoding="utf-8")
        print(f"[smoke]   OK killed with markers={sorted(have)}; planted partial@7 + corrupt@9",
              flush=True)

        # ---------- run 3: resume ----------
        print("[smoke] 3/13 resume run (must reject corrupt@9, ignore partial@7, resume@6)...",
              flush=True)
        rc, out = run_driver(box2, env2)
        assert rc == 0, f"resume run rc={rc}\n{out}"
        assert "RESUME candidate" in out and "rejected" in out, "corrupt candidate not exercised"
        assert re.search(r"RESUME verified base s3 iter=6", out), out
        assert "RESUMED from latest checkpoint at iters_done=6" in out, out
        wait_uploads(box2, {ITERS})
        assert final_blob(box2, ITERS) == ref_final, \
            "resumed final state != uninterrupted final state"
        print("[smoke]   OK resumed@6 -> final state IDENTICAL to the uninterrupted run", flush=True)

        # ---------- run 4: DELTA-mode uninterrupted (POKERBOT_DELTA_CKPT=1, HYBRID cadence) ----------
        # HYBRID triggers govern the rolling checkpoints: cap 240 B (est = 120 * dirty iters) fires
        # every 2nd mid-window iter counted from the last checkpoint; gates (5,10) + final (12) are
        # iter-exact full bases; the chain-root base is written by the FIRST rolling fire (iter 2).
        # Pre-touched rebalance flag banks ONE extra detached delta at iter 3 (first armed mid-window
        # chunk), which resets the dirty window, shifting the deterministic schedule to:
        #   bases [2, 5, 10, 12], deltas [3(rebal), 7, 9].
        print("[smoke] 4/13 delta-mode run (hybrid size cadence; gates iter-exact; final == "
              "uninterrupted)...", flush=True)
        box4, env4 = setup_env(tmp, "delta")
        env4 = delta_env(env4, "2.4e-7")
        (box4 / "tmp" / "spot_rebalance").write_text("1", encoding="utf-8")
        rc, out = run_driver(box4, env4)
        if os.environ.get("DELTA_DEBUG"):
            print(out, flush=True)
        assert rc == 0, f"delta run rc={rc}\n{out}"
        # post-wipe-fix banner: fresh start arms AFTER the (absent) load, in the else branch
        assert "DELTA_CKPT armed (fresh start)" in out, f"delta arm banner missing:\n{out}"
        assert "ROLLING ckpt (size" in out, f"size trigger never fired:\n{out}"
        wait_uploads(box4, {ITERS})
        assert final_blob(box4, ITERS) == ref_final, "delta-mode final base != uninterrupted final state"
        # Deltas were WRITTEN at the size-triggered rolling checkpoints (from the driver log) — and
        # bases only at the chain-root fire + gate boundaries. (Deltas rooted at a SUPERSEDED base are
        # pruned by design once a newer base lands, so we assert on the log, not the final S3 state.)
        delta_writes = sorted(int(m.group(1)) for m in
                              re.finditer(r"CKPT_DELTA iters_done=(\d+)", out))
        base_writes = sorted(int(m.group(1)) for m in
                             re.finditer(r"CKPT_SERIALIZE iters_done=(\d+)", out))
        assert base_writes == [2, 5, 10, 12], f"base schedule wrong: {base_writes}"
        assert delta_writes == [3, 7, 9], f"delta schedule wrong: {delta_writes}"
        assert all(di % GATE_EVERY != 0 for di in delta_writes), f"a gate got a delta: {delta_writes}"
        # Rebalance recommendation banked exactly ONE extra mid-window delta (a non-boundary iter).
        assert out.count("REBALANCE early-signal ckpt") == 1, \
            f"rebalance ckpt should fire exactly once:\n{out}"
        rebal_iter = int(re.search(r"REBALANCE early-signal ckpt at iters_done=(\d+)", out).group(1))
        assert rebal_iter not in EXPECT_BOUNDARIES, f"rebalance delta hit a boundary: {rebal_iter}"
        assert rebal_iter in delta_writes, f"rebalance ckpt did not write a delta: {delta_writes}"
        print(f"[smoke]   OK delta-mode: base_writes={base_writes} delta_writes={delta_writes} "
              f"rebalance@{rebal_iter}, final IDENTICAL", flush=True)

        # ---------- run 5: DELTA-mode kill + resume (base + delta chain) ----------
        # Hybrid cap 240 B, no rebalance: deterministic schedule bases [2,5,10,12], deltas [4,7,9].
        print("[smoke] 5/13 delta-mode kill + resume (base@5 + delta chain -> identical final)...",
              flush=True)
        box5, env5 = setup_env(tmp, "delta_resume")
        env5 = delta_env(env5, "2.4e-7")
        # Kill AFTER the iter-7 delta marker lands (base 5 + delta seq1@7 durable) so resume must
        # apply the delta chain. all_marker_iters counts delta markers too (iter 7 is a .nldlt.ok).
        rc, out = run_driver(box5, env5, sleep="0.5", kill_after_marker=7,
                             marker_fn=all_marker_iters)
        assert rc != 0
        have5 = set(all_marker_iters(box5))
        assert 7 in have5 and ITERS not in have5, f"unexpected markers after delta kill: {have5}"
        rc, out = run_driver(box5, env5)
        assert rc == 0, f"delta resume rc={rc}\n{out}"
        # Resume must load base 5 and apply the delta chain (seq 1 -> iter 7, + any later stragglers).
        assert re.search(r"RESUME applied delta seq=\d+", out), \
            f"delta resume did not exercise the chain:\n{out}"
        assert "base_iter=5" in out, f"resume base should be gate 5:\n{out}"
        wait_uploads(box5, {ITERS})
        assert final_blob(box5, ITERS) == ref_final, \
            "delta-mode resumed final state != uninterrupted final state"
        print("[smoke]   OK delta-mode resume -> final state IDENTICAL to the uninterrupted run",
              flush=True)

        # ---------- run 6: EMERGENCY FLUSH (graceful spot reclaim mid-window) ----------
        # spot_watch touches the reclaim flag; the CHUNKED train loop notices it mid-window, banks
        # the un-checkpointed iters as ONE synchronous delta at a NON-cadence iter (+ .ok marker),
        # and exits 0. A fresh resume must land on base + emergency delta.
        print("[smoke] 6/13 emergency flush (reclaim flag mid-window -> banked delta @ "
              "non-cadence iter)...", flush=True)
        box6, env6 = setup_env(tmp, "emergency")
        # cap 240 B: chain-root base fires at iter 2 (as before); the flag lands mid-window right
        # after its marker, so the flush banks iter 3 -- a NON-cadence iter on every axis.
        env6 = delta_env(env6, "2.4e-7")
        reclaim_flag = box6 / "tmp" / "spot_reclaim"
        rc, out = run_driver(box6, env6, sleep="2.0", flag_delay=0.2,
                             touch_flag_after=(2, str(reclaim_flag)))
        assert rc == 0, f"emergency-flush run must exit 0, rc={rc}\n{out}"
        m = re.search(r"EMERGENCY_FLUSH iters_done=(\d+) bytes=(\d+) wall=([0-9.]+)s banked", out)
        assert m, f"EMERGENCY_FLUSH line missing:\n{out}"
        flush_iter = int(m.group(1))
        assert flush_iter % CKPT_EVERY != 0, \
            f"flush should land at a NON-cadence iter, got {flush_iter}\n{out}"
        # the emergency delta's marker is on S3 with the correct chain link (base 2, next seq)
        ck6 = (box6 / "s3" / "fakebucket" / "pokerbot" / "prod6000" / "native_cfr" / "smoke_arm2"
               / "checkpoints")
        emk = json.loads((ck6 / f"iter_{flush_iter:06d}.nldlt.ok").read_text(encoding="utf-8"))
        assert emk["base_iter"] == 2 and emk["seq"] == 1 and emk["iters_done"] == flush_iter, emk
        reclaim_flag.unlink()  # reclaim over; a fresh box resumes
        rc, out = run_driver(box6, env6)
        assert rc == 0, f"post-flush resume rc={rc}\n{out}"
        assert f"RESUMED from latest checkpoint at iters_done={flush_iter}" in out, out
        assert "+1 deltas" in out, f"emergency delta not applied on resume:\n{out}"
        wait_uploads(box6, {ITERS})
        assert final_blob(box6, ITERS) == ref_final, \
            "post-emergency-flush resumed final state != uninterrupted final state"
        print(f"[smoke]   OK emergency flush banked delta@{flush_iter} (non-cadence) -> resume "
              f"IDENTICAL final", flush=True)

        # ---------- run 7: PRESTAGE valid (bootstrap-prestaged base+chain; NO blob download) ----------
        print("[smoke] 7/13 prestage-valid resume (S3 blobs deleted -> prestaged copies used)...",
              flush=True)
        box7, env7 = setup_env(tmp, "prestage")
        # cap 120 B: a delta fires at EVERY mid-window iter -> gate base@5 roots a chain at 6,7(,8..);
        # kill after the iter-7 delta marker so >= 2 chain links are durable.
        env7 = delta_env(env7, "1.2e-7")
        rc, out = run_driver(box7, env7, sleep="0.5", kill_after_marker=7,
                             marker_fn=all_marker_iters)
        assert rc != 0
        ck7 = (box7 / "s3" / "fakebucket" / "pokerbot" / "prod6000" / "native_cfr" / "smoke_arm2"
               / "checkpoints")
        # QUIESCE: the detached uploaders survive the driver kill; wait for the durable marker set
        # to go stable before snapshotting the chain (hybrid deltas land at timing-decided iters).
        prev, t0q = None, time.time()
        while time.time() - t0q < 15:
            cur = all_marker_iters(box7)
            if cur == prev:
                break
            prev = cur
            time.sleep(1.0)
        ckdir = box7 / "opt" / "ckpt"
        ckdir.mkdir(parents=True, exist_ok=True)
        pfx = "pokerbot/prod6000/native_cfr/smoke_arm2/checkpoints/"
        base_meta = json.loads((ck7 / "iter_000005.nlcfr.ok").read_text(encoding="utf-8"))
        bpath = ckdir / "prestage_base.nlcfr"
        shutil.copyfile(ck7 / "iter_000005.nlcfr", bpath)
        man = {"base": {"marker": pfx + "iter_000005.nlcfr.ok",
                        "path": str(bpath).replace("\\", "/"), "iters_done": 5,
                        "bytes": base_meta["bytes"], "sha256": base_meta["sha256"]},
               "deltas": []}
        # discover the CONTIGUOUS durable chain rooted at base 5 from the .ok markers
        chain = {}
        for mk in ck7.glob("iter_*.nldlt.ok"):
            meta = json.loads(mk.read_text(encoding="utf-8"))
            if meta.get("base_iter") == 5:
                chain[int(meta["seq"])] = int(meta["iters_done"])
        seq, chain_blobs, chain_end = 1, [], 5
        while seq in chain:
            it7_ = chain[seq]
            dmeta = json.loads((ck7 / f"iter_{it7_:06d}.nldlt.ok").read_text(encoding="utf-8"))
            dpath = ckdir / f"prestage_delta_{seq:04d}.nldlt"
            shutil.copyfile(ck7 / f"iter_{it7_:06d}.nldlt", dpath)
            man["deltas"].append({"marker": pfx + f"iter_{it7_:06d}.nldlt.ok",
                                  "path": str(dpath).replace("\\", "/"), "seq": seq,
                                  "base_iter": 5, "bytes": dmeta["bytes"],
                                  "sha256": dmeta["sha256"]})
            chain_blobs.append(f"iter_{it7_:06d}.nldlt")
            chain_end = it7_
            seq += 1
        n_chain = len(chain_blobs)
        assert n_chain >= 2, f"expected >= 2 durable chain deltas rooted at base 5: {chain}"
        (ckdir / "prestage_manifest.json").write_text(json.dumps(man), encoding="utf-8")
        (ckdir / "prestage.started").write_text("1", encoding="utf-8")
        (ckdir / "prestage.done").write_text("done", encoding="utf-8")
        # DELETE the blobs from fake S3 (markers stay): a successful resume PROVES the driver used
        # the prestaged files and never issued a blob download.
        for blob in ["iter_000005.nlcfr"] + chain_blobs:
            (ck7 / blob).unlink()
        rc, out = run_driver(box7, env7)
        assert rc == 0, f"prestage resume rc={rc}\n{out}"
        assert out.count("RESUME using prestaged") == 1 + n_chain, \
            f"expected {1 + n_chain} prestaged consumptions (base + {n_chain} deltas):\n{out}"
        assert f"RESUMED from latest checkpoint at iters_done={chain_end}" in out, out
        assert f"(base_iter=5, +{n_chain} deltas)" in out, out
        wait_uploads(box7, {ITERS})
        assert final_blob(box7, ITERS) == ref_final, \
            "prestaged resume final state != uninterrupted final state"
        print(f"[smoke]   OK prestaged base+{n_chain} deltas consumed (blobs absent on S3) -> "
              f"IDENTICAL final", flush=True)

        # ---------- run 8: PRESTAGE stale (newer marker appeared) -> clean fallback download ----------
        print("[smoke] 8/13 prestage-stale resume (manifest points at an old marker -> fresh "
              "download)...", flush=True)
        stale_path = ckdir / "prestage_base.nlcfr"
        shutil.copyfile(ck7 / "iter_000010.nlcfr", stale_path)
        m10 = json.loads((ck7 / "iter_000010.nlcfr.ok").read_text(encoding="utf-8"))
        (ckdir / "prestage_manifest.json").write_text(json.dumps(
            {"base": {"marker": pfx + "iter_000010.nlcfr.ok",
                      "path": str(stale_path).replace("\\", "/"), "iters_done": 10,
                      "bytes": m10["bytes"], "sha256": m10["sha256"]},
             "deltas": []}), encoding="utf-8")
        rc, out = run_driver(box7, env7)
        assert rc == 0, f"stale-prestage resume rc={rc}\n{out}"
        assert "RESUME using prestaged" not in out, f"stale prestage must not be consumed:\n{out}"
        assert re.search(r"RESUME dl .*iter_000012\.nlcfr ", out), \
            f"expected a fresh download of the newest base:\n{out}"
        assert "RESUMED from latest checkpoint at iters_done=12" in out, out
        print("[smoke]   OK stale prestage ignored -> fresh download of the newest base (iter 12)",
              flush=True)

        # ---------- run 9: POST-RESUME CHAIN CONTINUATION (the load-wipes-arming bug) ----------
        # The real trainer's load_bytes(_from_file) replaces the whole table (dirty -> None), so a
        # pre-load arm_delta_tracking() was silently wiped and EVERY post-resume rolling checkpoint
        # fell through to a full base (60 GB / ~250 s each on the real box). The fixed driver arms
        # AFTER load + delta apply, reset_dirty, and roots _BASE at the resumed marker -- so the
        # FIRST post-resume cadence checkpoint must be a cheap DELTA continuing the chain (seq N+1
        # on the same base), and a subsequent kill+resume must apply base + N+1 deltas.
        print("[smoke] 9/13 post-resume chain continuation (first ckpt after resume is a DELTA, "
              "seq N+1)...", flush=True)
        box9, env9 = setup_env(tmp, "chain_cont")
        # Leg 1 (cap 240 B -> deltas at 4,7): kill after the iter-7 delta marker; durable state =
        # base@5 + delta seq1@7 (N=1).
        rc, out = run_driver(box9, delta_env(env9, "2.4e-7"), sleep="0.5", kill_after_marker=7,
                             marker_fn=all_marker_iters)
        assert rc != 0
        # Leg 2 (cap 120 B -> the size trigger fires at the FIRST post-resume mid-window iter, 8):
        # a correctly re-armed driver must write a DELTA there (seq 2 on the same base 5), not a
        # full base. Killed right after that delta's marker (8) lands.
        rc, out = run_driver(box9, delta_env(env9, "1.2e-7"), sleep="1.5", kill_after_marker=8,
                             marker_fn=all_marker_iters)
        assert rc != 0
        assert "DELTA_CKPT armed post-load; chain rooted at base=5 seq=1" in out, \
            f"post-load arm banner missing/wrong:\n{out}"
        assert "RESUMED from latest checkpoint at iters_done=7 (base_iter=5, +1 deltas)" in out, out
        first_ckpt = re.search(r"CKPT_(SERIALIZE|DELTA) iters_done=(\d+)(?: base=(\d+) seq=(\d+))?",
                               out)
        assert first_ckpt, f"no checkpoint line in resume run:\n{out}"
        assert first_ckpt.group(1) == "DELTA", \
            f"FIRST post-resume checkpoint must be a DELTA, got {first_ckpt.group(0)}\n{out}"
        assert (int(first_ckpt.group(2)), int(first_ckpt.group(3)), int(first_ckpt.group(4))) \
            == (8, 5, 2), f"delta must continue the chain (iter 8, base 5, seq 2): {first_ckpt.group(0)}"
        ck9 = (box9 / "s3" / "fakebucket" / "pokerbot" / "prod6000" / "native_cfr" / "smoke_arm2"
               / "checkpoints")
        mk8 = json.loads((ck9 / "iter_000008.nldlt.ok").read_text(encoding="utf-8"))
        assert mk8["base_iter"] == 5 and mk8["seq"] == 2 and mk8["iters_done"] == 8, mk8
        # final resume: must apply base@5 + BOTH deltas (7, 8) and finish identical to run 1
        rc, out = run_driver(box9, delta_env(env9, "2.4e-7"))
        assert rc == 0, f"chain-continuation final resume rc={rc}\n{out}"
        assert "DELTA_CKPT armed post-load; chain rooted at base=5 seq=2" in out, out
        assert "RESUMED from latest checkpoint at iters_done=8 (base_iter=5, +2 deltas)" in out, out
        wait_uploads(box9, {ITERS})
        assert final_blob(box9, ITERS) == ref_final, \
            "chain-continuation resumed final state != uninterrupted final state"
        print("[smoke]   OK resume re-armed post-load: first ckpt = DELTA@8 (base 5, seq 2); "
              "kill+resume applied base+2 deltas -> IDENTICAL final", flush=True)

        # ---------- run 10: HYBRID SIZE trigger at an ARBITRARY iter + kill + resume ----------
        # cap 360 B (est = 120 * dirty iters => fires every 3rd): the chain-root base lands at
        # iter 3 -- an ODD iter no CKPT_EVERY cadence could produce -- and the base-5 chain's
        # delta at iter 8. Kill after that marker; resume must apply the chain and finish
        # byte-identical.
        print("[smoke] 10/13 hybrid SIZE trigger (arbitrary-iter ckpts; kill+resume identical)...",
              flush=True)
        box10, env10 = setup_env(tmp, "hybrid_size")
        env10 = delta_env(env10, "3.6e-7")
        rc, out = run_driver(box10, env10, sleep="0.5", kill_after_marker=8,
                             marker_fn=all_marker_iters)
        assert rc != 0
        assert "ROLLING ckpt (size" in out, f"size trigger never fired:\n{out}"
        assert re.search(r"CKPT_SERIALIZE iters_done=3 ", out), \
            f"chain-root base must land at the size-trigger iter 3 (arbitrary/odd):\n{out}"
        assert re.search(r"CKPT_DELTA iters_done=8 base=5 seq=1", out), \
            f"size-triggered delta @8 (base 5, seq 1) missing:\n{out}"
        rc, out = run_driver(box10, env10)
        assert rc == 0, f"hybrid-size resume rc={rc}\n{out}"
        assert "(base_iter=5, +1 deltas)" in out, f"chain not applied on resume:\n{out}"
        wait_uploads(box10, {ITERS})
        assert final_blob(box10, ITERS) == ref_final, \
            "hybrid-size resumed final state != uninterrupted final state"
        print("[smoke]   OK size trigger fired @3 (base) and @8 (delta, base 5 seq 1); resume "
              "IDENTICAL final", flush=True)

        # ---------- run 11: HYBRID TIME trigger (slow dirty accumulation) ----------
        # Size trigger OFF, CKPT_SECS=2 s, ~0.9 s/iter: rolling checkpoints fire on wall-clock
        # roughly every 2-3 iters at timing-decided (non-gate) iters; gates stay iter-exact.
        # Wide-margin asserts only (timing-based).
        print("[smoke] 11/13 hybrid TIME trigger (CKPT_SECS=2; rolling ckpts on wall-clock)...",
              flush=True)
        box11, env11 = setup_env(tmp, "hybrid_time")
        env11 = delta_env(env11, "0", secs="2")
        rc, out = run_driver(box11, env11, sleep="0.9")
        assert rc == 0, f"hybrid-time run rc={rc}\n{out}"
        assert out.count("ROLLING ckpt (time") >= 1, f"time trigger never fired:\n{out}"
        delta_writes = sorted(int(m.group(1)) for m in
                              re.finditer(r"CKPT_DELTA iters_done=(\d+)", out))
        base_writes = sorted(int(m.group(1)) for m in
                             re.finditer(r"CKPT_SERIALIZE iters_done=(\d+)", out))
        assert all(di % GATE_EVERY != 0 for di in delta_writes), \
            f"a gate got a delta under the time trigger: {delta_writes}"
        assert GATES <= set(base_writes), f"gate bases missing/inexact: {base_writes}"
        wait_uploads(box11, {ITERS})
        assert final_blob(box11, ITERS) == ref_final, \
            "hybrid-time final state != uninterrupted final state"
        print(f"[smoke]   OK time trigger fired {out.count('ROLLING ckpt (time')}x "
              f"(bases={base_writes} deltas={delta_writes}); final IDENTICAL", flush=True)

        # ---------- run 12: GATE EXACTNESS under rolling fires ----------
        # cap 240 B, no rebalance: deterministic schedule bases [2,5,10,12] / deltas [4,7,9] --
        # the delta at 9 fires ONE iter before gate 10, and the gate must still land ITER-EXACT
        # as a full base with the v3 extras (snapshots ONLY at gates).
        print("[smoke] 12/13 gate exactness (rolling delta 1 iter before an iter-exact gate "
              "base)...", flush=True)
        box12, env12 = setup_env(tmp, "gate_exact")
        env12 = delta_env(env12, "2.4e-7")
        rc, out = run_driver(box12, env12)
        assert rc == 0, f"gate-exactness run rc={rc}\n{out}"
        delta_writes = sorted(int(m.group(1)) for m in
                              re.finditer(r"CKPT_DELTA iters_done=(\d+)", out))
        base_writes = sorted(int(m.group(1)) for m in
                             re.finditer(r"CKPT_SERIALIZE iters_done=(\d+)", out))
        assert base_writes == [2, 5, 10, 12], f"gate bases must land iter-exact: {base_writes}"
        assert delta_writes == [4, 7, 9], f"delta schedule wrong: {delta_writes}"
        assert 9 in delta_writes and 10 in base_writes, "rolling@9 + gate@10 adjacency missing"
        wait_uploads(box12, {ITERS})
        snaps12 = sorted(int(re.search(r"snap_(\d+)\.nlsnap$", k).group(1))
                         for k in s3_keys(box12) if k.endswith(".nlsnap"))
        assert snaps12 == [5, 10, 12], \
            f"snapshots must land ONLY at gate iters despite rolling fires: {snaps12}"
        assert final_blob(box12, ITERS) == ref_final, \
            "gate-exactness final state != uninterrupted final state"
        print(f"[smoke]   OK delta@9 then FULL base@10 iter-exact; snaps={snaps12}; final "
              f"IDENTICAL", flush=True)

        # ---------- run 13: MIN-INTERVAL guard suppresses back-to-back triggers ----------
        # Always-firing size cap (1 byte) + CKPT_MIN_GAP_SECS=2 s @ ~0.5 s/iter: without the
        # guard EVERY mid-window iter would checkpoint (adjacent iters); with it, consecutive
        # rolling fires must be >= 2 iters apart (wall-clock floor ~4 iters; margin 2) and only
        # a few of the ~9 mid-window opportunities may fire. Wide-margin asserts (timing-based).
        print("[smoke] 13/13 min-interval guard (always-firing cap throttled by "
              "CKPT_MIN_GAP_SECS)...", flush=True)
        box13, env13 = setup_env(tmp, "min_gap")
        env13 = delta_env(env13, "1e-9", mingap="2.0")
        rc, out = run_driver(box13, env13, sleep="0.5")
        assert rc == 0, f"min-gap run rc={rc}\n{out}"
        roll_iters = [int(m.group(1)) for m in
                      re.finditer(r"ROLLING ckpt \(.*?\) at iters_done=(\d+)", out)]
        assert roll_iters, f"no rolling fire at all (cap=1 byte should fire once >=2s in):\n{out}"
        assert len(roll_iters) <= 5, \
            f"min-gap failed to throttle the 1-byte cap: fires at {roll_iters}"
        gaps = [b - a for a, b in zip(roll_iters, roll_iters[1:])]
        assert all(g >= 2 for g in gaps), \
            f"back-to-back rolling fires slipped past the min-gap guard: {roll_iters}"
        wait_uploads(box13, {ITERS})
        assert final_blob(box13, ITERS) == ref_final, \
            "min-gap final state != uninterrupted final state"
        print(f"[smoke]   OK min-gap throttled fires to {roll_iters} (gaps {gaps}); final "
              f"IDENTICAL", flush=True)

        print(f"[smoke] ALL PASS in {time.time() - t_start:.1f}s", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
