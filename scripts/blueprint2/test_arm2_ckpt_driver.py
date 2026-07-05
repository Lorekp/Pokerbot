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

No AWS, no network, no real trainer: this validates the driver/uploader/marker/retention logic
only (the trainer-level resume bit-identity is covered by the ab_train Rust smoke).

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
    def download_file(self, Bucket, Key, path):
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
    """Deterministic fake: state = sha256 chain over the iteration indices trained so far, so a
    resumed run reproduces the uninterrupted final state IFF resume restores iters_done exactly."""
    def __init__(self, *a, **k):
        self._iters = 0
        self._acc = hashlib.sha256(b"seed")
    def _replay_to(self, n):
        acc = hashlib.sha256(b"seed")
        for i in range(1, n + 1):
            acc.update(str(i).encode())
        return acc
    def train(self, step):
        for _ in range(int(step)):
            self._iters += 1
            self._acc.update(str(self._iters).encode())
            time.sleep(float(os.environ.get("FAKE_TRAIN_SLEEP", "0.02")))
    def save_bytes(self):
        return b"FAKE0001" + struct.pack("<Q", self._iters) + self._acc.digest() + bytes(4096)
    def load_bytes(self, blob):
        if not blob or blob[:8] != b"FAKE0001":
            return False
        self._iters = struct.unpack("<Q", blob[8:16])[0]
        self._acc = self._replay_to(self._iters)
        return True
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
    stub = box / "stubs"
    (stub / "boto3" / "s3").mkdir(parents=True)
    (stub / "boto3" / "__init__.py").write_text(FAKE_BOTO3, encoding="utf-8")
    (stub / "boto3" / "s3" / "__init__.py").write_text("", encoding="utf-8")
    (stub / "boto3" / "s3" / "transfer.py").write_text(FAKE_TRANSFER, encoding="utf-8")
    (stub / "battery_native.py").write_text(FAKE_BN, encoding="utf-8")

    driver, uploader = render_driver()
    opt = str(box / "opt").replace("\\", "/")
    log = str(box / "log").replace("\\", "/")
    for src in (driver, uploader):
        pass
    driver = (driver.replace("/opt/pokerbot", opt)
                    .replace("/var/log/ckpt_upload.log", log + "/ckpt_upload.log")
                    .replace("/var/log/native-cfr-train.log", log + "/train.log"))
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


def run_driver(box: Path, env, sleep="0.02", kill_after_marker=None):
    env = dict(env, FAKE_TRAIN_SLEEP=sleep)
    proc = subprocess.Popen([sys.executable, str(box / "opt" / "drive_native_cfr.py")],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if kill_after_marker is None:
        out, _ = proc.communicate(timeout=300)
        return proc.returncode, out
    # poll for the requested marker, then hard-kill mid-training (simulated spot reclaim)
    t0 = time.time()
    while time.time() - t0 < 120:
        if kill_after_marker in marker_iters(box):
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
        print("[smoke] 1/3 uninterrupted reference run...", flush=True)
        box1, env1 = setup_env(tmp, "ref")
        rc, out = run_driver(box1, env1)
        assert rc == 0, f"reference run rc={rc}\n{out}"
        assert "FRESH start" in out, out
        wait_uploads(box1, {ITERS})
        ref_final = final_blob(box1, ITERS)
        m1 = set(marker_iters(box1))
        boundaries = sorted(int(x) for x in re.findall(r"CKPT_LOCAL iters_done=(\d+)", out))
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
        print("[smoke] 2/3 kill-mid-run (simulated spot reclaim after iter-6 marker)...", flush=True)
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
        print("[smoke] 3/3 resume run (must reject corrupt@9, ignore partial@7, resume@6)...",
              flush=True)
        rc, out = run_driver(box2, env2)
        assert rc == 0, f"resume run rc={rc}\n{out}"
        assert "RESUME candidate" in out and "rejected" in out, "corrupt candidate not exercised"
        assert re.search(r"RESUME verified .*iter_000006\.nlcfr", out), out
        assert "RESUMED from latest checkpoint at iters_done=6" in out, out
        wait_uploads(box2, {ITERS})
        assert final_blob(box2, ITERS) == ref_final, \
            "resumed final state != uninterrupted final state"
        print("[smoke]   OK resumed@6 -> final state IDENTICAL to the uninterrupted run", flush=True)

        print(f"[smoke] ALL PASS in {time.time() - t_start:.1f}s", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
