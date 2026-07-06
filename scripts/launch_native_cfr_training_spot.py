#!/usr/bin/env python3
"""
Launch a SINGLE EC2 spot instance to train the NATIVE 6-max NLHE CFR+ blueprint (Workstream F).

Mirrors scripts/launch_sd_cfr_training_spot.py (spot launch + babysitter/resume pattern), but the
workload is the Rust ``NlheCFRTrainer`` (tools/battery_native/src/trainer.rs) rather than the JAX SD-CFR
stack. The box:
  1. installs Rust + a Python with PyO3 headers,
  2. pulls the repo tarball from S3 and ``cargo build --release --features python --lib`` the crate
     (the .so is the importable ``battery_native`` extension exposing ``NlheCFRTrainer``),
  3. runs a tiny embedded driver that constructs ``NlheCFRTrainer(num_buckets, depth_edges,
     n_deals_per_iter)``, RESUMES from the latest S3 checkpoint if one exists, then trains to
     ``--iters`` total, calling ``save_bytes()`` -> boto3 upload every ``--checkpoint-every`` iters,
  4. on spot preemption (or normal exit) the latest checkpoint is on S3, so a replacement box resumes.

Default is CPU-heavy (c6i.32xlarge, 128 vCPU) — the trainer is rayon-parallel across deals. Cost target
(plan): 5000 iters x ~200 deals ~ $12-40 spot, well under the Pluribus blueprint's <$150.

Reuses the same AWS identity + S3 prefix conventions as the SD-CFR launcher.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import sys
import tarfile
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUCKET = "video-processing-6789"
DEFAULT_PREFIX = "pokerbot/prod6000/native_cfr"  # within the IAM-allowed prod6000/* prefix
DEFAULT_REGION = "eu-central-1"
DEFAULT_PROFILE = "pokerbot-labelgen-worker"

# Ship the crate (its only build inputs) + the python serve/eval chain needed for offline validation.
# The trainer builds ON core.rs's public API; no JAX needed to TRAIN (only to LBR-validate, optional).
_INCLUDE = [
    "tools/battery_native",
    "pokerbot/search/native_solver.py",
    "pokerbot/search/tabular_blueprint_adapter.py",
]
_EXCLUDE_DIRS = ("__pycache__", ".git", "target")  # never ship the local Rust build dir (huge + stale)


def _tar_filter(ti: tarfile.TarInfo):
    parts = ti.name.split("/")
    if any(p in _EXCLUDE_DIRS for p in parts) or ti.name.endswith(".pyc"):
        return None
    return ti


def build_and_upload_code(s3, bucket: str, prefix: str, crate_src: str = "tools/battery_native",
                          emd_frozen: str = "", river_ochs: str = "") -> str:
    # (src, arcname) pairs. The crate may come from a FROZEN COPY (``crate_src``) but is ALWAYS uploaded
    # under "tools/battery_native" so the box builds it at the expected manifest path — this isolates a
    # running blueprint job from concurrent edits to the live crate (e.g. a parallel search optimization).
    includes = [
        (crate_src, "tools/battery_native"),
        ("pokerbot/search/native_solver.py", "pokerbot/search/native_solver.py"),
        ("pokerbot/search/tabular_blueprint_adapter.py", "pokerbot/search/tabular_blueprint_adapter.py"),
    ]
    # Ship the shared frozen EMD lookup so the box's trainer keys postflop buckets from the SAME bytes the
    # serve encoder will read (bit-identical train/serve). Uploaded under its given relative path.
    if emd_frozen:
        includes.append((emd_frozen, emd_frozen))
    # Blueprint-v2 river OCHS table (~10 KB): shipped so the box's trainer keys the RIVER with the
    # SAME nearest-centroid table the serve encoder will read (POKERBOT_RIVER_OCHS).
    if river_ochs:
        includes.append((river_ochs, river_ochs))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for src, arc in includes:
            p = ROOT / src
            if p.exists():
                tar.add(p, arcname=arc, filter=_tar_filter)
    buf.seek(0)
    key = f"{prefix}/code/native_cfr_code.tar.gz"
    # SKIP-IF-UNCHANGED (2026-07-06): the retry supervisor deliberately never passes --reuse-code
    # (stale-code incidents), so every relaunch cycle used to pay the 224 MB home-upstream upload.
    # Compare on the ACTUAL would-be tarball bytes instead: sha256 of the DECOMPRESSED tar content
    # (same definition as the on-box CODE_SHA: `gzip -dc | sha256sum` — the gzip envelope carries a
    # fresh mtime on every tarfile.open, so the compressed bytes never match). The sha is stored as
    # S3 object metadata on upload; a HEAD + metadata match on the next invocation skips the upload
    # exactly (content-hash, not mtime/git guesswork). --reuse-code still works as-is (never calls
    # this function).
    h = hashlib.sha256()
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        for chunk in iter(lambda: gz.read(1 << 24), b""):
            h.update(chunk)
    code_sha = h.hexdigest()
    buf.seek(0)
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        if head.get("Metadata", {}).get("code-sha") == code_sha:
            print(f"Code tarball unchanged (sha={code_sha}) -- skipping upload", flush=True)
            return key
    except ClientError:
        pass  # no existing object (or no HEAD permission) -> upload
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read(),
                  Metadata={"code-sha": code_sha})
    print(f"Uploaded code tarball -> s3://{bucket}/{key} ({buf.tell()} bytes; crate_src={crate_src}; "
          f"code-sha={code_sha})", flush=True)
    return key


def resolve_ubuntu_ami(ec2, arch: str) -> tuple[str, str]:
    """Returns (image_id, root_device_name) — the root device name is needed to override the root
    volume size/throughput via BlockDeviceMappings (checkpoint staging for the detached uploader)."""
    name_arch = "amd64" if arch == "x86_64" else "arm64"
    resp = ec2.describe_images(
        Owners=["099720109477"],  # Canonical
        Filters=[
            {"Name": "name", "Values": [f"ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-{name_arch}-server-*"]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": [arch]},
        ],
    )
    imgs = sorted(resp["Images"], key=lambda i: i["CreationDate"], reverse=True)
    if not imgs:
        raise SystemExit(f"no {arch} Ubuntu 22.04 AMI found in region")
    return imgs[0]["ImageId"], imgs[0].get("RootDeviceName", "/dev/sda1")


# Spot-capacity errors worth trying the NEXT instance type for (vs a hard config/quota error, which
# every type shares, so retrying down the chain is pointless).
_CAPACITY_ERROR_CODES = frozenset({
    "InsufficientInstanceCapacity",
    "SpotMaxPriceTooLow",
    "InstanceLimitExceeded",
    "MaxSpotInstanceCountExceeded",
    "Unsupported",  # instance type not offered in this AZ/region
})


def _spotpick_chain(instance_type: str, arch: str, n: int) -> list[str]:
    """Build a capacity-diverse fallback chain (one type per family) starting at the requested type's
    vCPU count, via the shared spotpick lib (.). Returns [instance_type] alone if
    spotpick or the catalog is unavailable, so --spotpick degrades gracefully to single-type behavior.
    The head of the chain is forced to be the explicitly-requested ``instance_type`` (so the default
    c6i.32xlarge is still tried FIRST), with the spotpick alternatives appended de-duplicated."""
    try:
        sys.path.insert(0, r".")
        import spotpick  # noqa: E402
    except Exception as e:  # noqa: BLE001
        print(f"[spotpick] unavailable ({e!r}) — falling back to single type {instance_type}", flush=True)
        return [instance_type]
    # Derive the vCPU target from the requested type via the catalog (so the chain matches its size).
    try:
        cat = {it.name: it for it in spotpick.load()}
        vcpus = cat[instance_type].vcpus if instance_type in cat else None
        if vcpus is None:
            print(f"[spotpick] {instance_type} not in catalog — single type only", flush=True)
            return [instance_type]
        # SPOTPICK_EXCLUDE_FAMILIES (comma list): keep memory-inadequate families OUT of the
        # capacity-fallback chain. The catalog carries no RAM column, so at 64 vCPU an unfiltered
        # chain includes 128 GB c-family boxes that OOM late in a ~132 GB-RSS training run — the
        # caller (babysitter env) owns the RAM floor for its workload.
        import os
        excl = [f.strip() for f in os.environ.get("SPOTPICK_EXCLUDE_FAMILIES", "").split(",")
                if f.strip()]
        if excl:
            print(f"[spotpick] excluding families from fallback chain: {','.join(excl)}", flush=True)
        alts = spotpick.chain(vcpus=vcpus, arch=arch, n=max(n, 1),
                              exclude_families=excl or None)
    except Exception as e:  # noqa: BLE001
        print(f"[spotpick] chain() failed ({e!r}) — single type {instance_type}", flush=True)
        return [instance_type]
    chain = [instance_type] + [a for a in alts if a != instance_type]
    return chain[:max(n, 1)]


def _launch_with_fallback(ec2, run_kwargs: dict, *, instance_types: list[str], spot: bool):
    """Try ``run_instances`` down ``instance_types`` until one succeeds (capacity errors advance to the
    next type; any other error is fatal). Returns (instance_id, instance_type_used). The caller bakes
    UserData/tags BEFORE this — only ``InstanceType`` changes per attempt, so a fallback box is otherwise
    byte-identical (same run-subdir => resumes the same S3 checkpoint)."""
    last_err = None
    for it in instance_types:
        kw = dict(run_kwargs, InstanceType=it)
        try:
            resp = ec2.run_instances(**kw)
            return resp["Instances"][0]["InstanceId"], it
        except ClientError as e:  # noqa: PERF203
            code = e.response.get("Error", {}).get("Code", "")
            last_err = e
            if code in _CAPACITY_ERROR_CODES and len(instance_types) > 1:
                print(f"[spotpick] {it}: {code} — trying next type", flush=True)
                continue
            raise
    raise SystemExit(f"all instance types exhausted ({instance_types}); last error: {last_err}")


def user_data_stub(*, bucket, region, run_prefix, bootstrap_key, wall_guard_secs,
                   awscli_zip="awscli-exe-linux-x86_64.zip") -> str:
    """Tiny cloud-init user-data (EC2 hard-caps user-data at 16384 bytes; the full bootstrap
    outgrew it once river-OCHS + resume logic landed). Contains ONLY: (1) STEP-0 dead-man cost
    cap — inline VERBATIM so the box is cost-bounded even if the S3 fetch below fails; (2) AWS
    CLI v2 install + fetch of the real bootstrap from S3 + exec. Bootstrap fetch failure =>
    immediate shutdown (never an idle box)."""
    floor_min = wall_guard_secs // 60 + 5   # absolute logind floor, 5 min after the wall-guard unit
    return f"""#!/bin/bash
set -uxo pipefail
export HOME="${{HOME:-/root}}"   # cloud-init runs as root with HOME UNSET; under set -u a bare $HOME is FATAL
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION={region}
mkdir -p /opt/pokerbot

# ===================== STEP 0 -- DEAD-MAN COST CAP, ARMED FIRST (before apt/fetch/anything) =====================
# TWO independent on-box terminators so the box CANNOT outlive ~{floor_min} min no matter what fails below
# (the documented 3h20m idle-bill fix; ported from launch_cpu128_probe_spot.py). These MUST be the first real
# commands so they survive a crash anywhere downstream (incl. a failed bootstrap fetch) AND a dead
# babysitter / powered-off laptop:
#  0a) ABSOLUTE FLOOR: `shutdown -h +{floor_min}` is scheduled by systemd/logind ITSELF (not a child/sleep),
#      immune to cloud-init cgroup teardown, needs no packages.
#  0b) WALL-GUARD UNIT: a systemd-run transient unit (own cgroup; NO --oneshot, whose 90s
#      DefaultTimeoutStartSec would reap the sleep) that at {wall_guard_secs}s sweeps the train log to S3 then
#      shuts down (~5 min before 0a).
/sbin/shutdown -h +{floor_min} "pokerbot native-cfr wall-guard absolute cost cap" || true
systemd-run --unit=pokerbot-wallguard --collect -p TimeoutStartSec=infinity --no-block /bin/bash -c 'sleep {wall_guard_secs} ; sync ; timeout 120 aws s3 cp /var/log/native-cfr-train.log s3://{bucket}/{run_prefix}/logs/train.WALLGUARD.log --region {region} 2>/dev/null || true ; /sbin/shutdown -h now' || true
echo "WALL_GUARD armed: absolute shutdown +{floor_min}min floor + pokerbot-wallguard unit at {wall_guard_secs}s"

# ===================== STEP 1 -- fetch + exec the real bootstrap from S3 =====================
# AWS CLI v2 (NOT apt awscli v1 — the documented botocore-breakage lesson lives in the bootstrap).
apt-get update -qq
apt-get install -y -qq curl unzip
curl -fsSL "https://awscli.amazonaws.com/{awscli_zip}" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update
hash -r
aws s3 cp s3://{bucket}/{bootstrap_key} /tmp/bootstrap.sh --region {region} || {{ echo "FATAL: bootstrap fetch failed" ; sync ; /sbin/shutdown -h now ; exit 1 ; }}
bash /tmp/bootstrap.sh
"""


def bootstrap_body(*, bucket, prefix, region, code_key, run_subdir, iters, n_deals, num_buckets,
                   depth_edges, checkpoint_every, gate_every=0, keep_alive=False, emd_frozen_rel="", emd_sha="",
                   river_ochs_rel="", river_ochs_sha="", preflop_hist_key=False,
                   fullhist_key=False, mrr=False,
                   work_steal=False, dcfr=False, dcfr_prune_threshold=-300000.0, dcfr_explore_frac=0.05,
                   v3=None, row_budget=None, no_cache_fetch=False, delta_ckpt=False, profile_run=False,
                   wall_guard_secs=21600, awscli_zip="awscli-exe-linux-x86_64.zip") -> str:
    """The full on-box bootstrap (was inline user-data until it exceeded EC2's 16KB cap).
    Uploaded to S3 by main() and exec'd by user_data_stub(). Content is otherwise the proven
    launcher script, byte-equivalent from `apt-get update` onward — STEP 0 (cost cap) lives in
    the stub and is ALREADY ARMED when this runs. The AWS CLI v2 re-install below is idempotent
    (--update) and kept so this script stays self-sufficient if ever run standalone."""
    run_prefix = f"{prefix}/{run_subdir}"
    no_cache_fetch = "1" if no_cache_fetch else "0"  # rendered into the bootstrap fetch guard
    final = "echo BOOTSTRAP_DONE_KEEP_ALIVE" if keep_alive else "shutdown -h now"
    depth_csv = ",".join(str(float(x)) for x in (depth_edges or []))
    hist_flag = "1" if preflop_hist_key else "0"
    # Blueprint-v4 Stage-0 keys (FULLHIST full-action-history key + M-RR action-thinning). ADDITIVE +
    # default-off BYTE-IDENTICAL: when BOTH flags are off ``fullhist_block`` is the empty string, so the
    # rendered bootstrap is byte-for-byte the bp2/bp3 script (the export/echo lines simply do not exist).
    # When either is on, it mirrors the PREFLOP_HIST_KEY plumbing above: export the env the trainer reads
    # (fingerprint.rs fullhist A-segment on every street + trainer::abstracted_legal M-RR action-thinning)
    # plus an echo milestone. The checkpoint MUST then be served/evaluated with the SAME env
    # (design: artifacts/blueprint_pluribus_grade_design_v2.md §5-7).
    fullhist_flag = "1" if fullhist_key else "0"
    mrr_flag = "1" if mrr else "0"
    if fullhist_key or mrr:
        fullhist_block = (
            f"export POKERBOT_FULLHIST_KEY={fullhist_flag}\n"
            f'echo "FULLHIST_KEY=$POKERBOT_FULLHIST_KEY"\n'
            f"export POKERBOT_MRR={mrr_flag}\n"
            f'echo "MRR=$POKERBOT_MRR"'
        )
    else:
        fullhist_block = ""
    # blueprint-v4 v3 TRAINER SCHEDULE (LCFR + hard negative-regret pruning + snapshot averaging +
    # coverage metric — artifacts/blueprint_v4_amendment_v3.md). ADDITIVE + default-off BYTE-IDENTICAL,
    # gated exactly like ``fullhist_block``: with ``v3=None`` all three placeholders render as "" (each is
    # attached as a SUFFIX to an existing line, so the off-path emits zero extra bytes) and the bootstrap
    # is byte-for-byte the Stage-0 script (sha-compared in the v3 session). When set, ``v3`` is a dict:
    #   lcfr(bool), signed_regrets(bool — arm 2: pruning without the LCFR discount), lcfr_d, lcfr_w,
    #   prune_c(float|None), prune_start(int|None), prune_explore,
    #   regret_floor(float|None), snapshot_avg(bool), snap_start(int), cov_hands(int), regstats(bool)
    # The env exports are read ONCE by trainer.rs (V3Cfg::from_env — the SAME OnceLock the local ab_train
    # path reads, so cloud and local runs train identically); the driver extras use the v3 pymethods
    # (save_current_strategy_bytes / coverage / regret_stats) added to NlheCFRTrainer in the same
    # commit-series, uploading NLSNAP01 snapshots next to each checkpoint and printing the
    # COVERAGE/REGSTATS lines the kill-gate + babysitter grep.
    if v3:
        exports = [
            # DELTA CHECKPOINTS (R2): driver-level knob, exported alongside the v3 envs so the
            # generated driver's DELTA_CKPT gate reads it. Training semantics unchanged (driver-sim
            # proves base+delta resume final state == uninterrupted); OFF renders "0" = R2 default.
            f"export POKERBOT_DELTA_CKPT={'1' if delta_ckpt else '0'}",
            'echo "DELTA_CKPT=$POKERBOT_DELTA_CKPT"',
            f"export POKERBOT_LCFR={'1' if v3.get('lcfr') else '0'}",
            'echo "LCFR=$POKERBOT_LCFR"',
            f"export POKERBOT_SIGNED_REGRETS={'1' if v3.get('signed_regrets') else '0'}",
            'echo "SIGNED_REGRETS=$POKERBOT_SIGNED_REGRETS"',
            f"export POKERBOT_LCFR_D={int(v3.get('lcfr_d', 250))}",
            f"export POKERBOT_LCFR_W={int(v3.get('lcfr_w', 10000))}",
        ]
        if v3.get("prune_c") is not None:
            exports.append(f"export POKERBOT_PRUNE_C={float(v3['prune_c'])}")
        if v3.get("prune_start") is not None:
            exports.append(f"export POKERBOT_PRUNE_START={int(v3['prune_start'])}")
        exports.append(f"export POKERBOT_PRUNE_EXPLORE={float(v3.get('prune_explore', 0.05))}")
        if v3.get("regret_floor") is not None:
            exports.append(f"export POKERBOT_REGRET_FLOOR={float(v3['regret_floor'])}")
        exports.append(f"export POKERBOT_SNAPSHOT_AVG={'1' if v3.get('snapshot_avg') else '0'}")
        exports.append('echo "SNAPSHOT_AVG=$POKERBOT_SNAPSHOT_AVG"')
        v3_env_block = "\n" + "\n".join(exports)
        v3_driver_fns = (
            "\n\n"
            "V3_SNAPSHOT_AVG = os.environ.get(\"POKERBOT_SNAPSHOT_AVG\", \"0\") == \"1\"\n"
            f"V3_SNAP_START = {int(v3.get('snap_start', 0))}\n"
            f"V3_COV_HANDS = {int(v3.get('cov_hands', 0))}\n"
            f"V3_REGSTATS = {bool(v3.get('regstats', False))}\n"
            "\n"
            "def _v3_ckpt_extras(t, gate=True):\n"
            "    # v3 checkpoint-cadence extras (read-only on the table): NLSNAP01 snapshot upload\n"
            "    # (rounds-2-4 averaging inputs), COVERAGE line (on-policy miss gates), REGSTATS line\n"
            "    # (prune-C calibration). Failures RETRY then continue -- never crash the run.\n"
            "    # gate=False (a rolling resume-only checkpoint between gate-cadence points) skips the\n"
            "    # snapshot/COVERAGE/REGSTATS extras so the PRE-REGISTERED measurement cadence (and the\n"
            "    # snapshot-averaging density) is unchanged; the train.log upload still runs every time\n"
            "    # (monitor freshness -- the v3-closeout lesson).\n"
            "    it = t.iters_done()\n"
            "    if gate and V3_SNAPSHOT_AVG and it >= V3_SNAP_START:\n"
            "        import io as _io\n"
            "        from boto3.s3.transfer import TransferConfig as _TC\n"
            "        blob = t.save_current_strategy_bytes()\n"
            "        key = RUN_PREFIX + f\"/snapshots/snap_{it:06d}.nlsnap\"\n"
            "        last = None\n"
            "        for attempt in range(1, 6):\n"
            "            try:\n"
            "                s3.upload_fileobj(_io.BytesIO(blob), BUCKET, key,\n"
            "                                  Config=_TC(multipart_chunksize=256 * 1024 * 1024))\n"
            "                print(f\"SNAPSHOT iters_done={it} -> s3://{BUCKET}/{key} ({len(blob)} bytes)\",\n"
            "                      flush=True)\n"
            "                break\n"
            "            except Exception as e:\n"
            "                last = e\n"
            "                print(f\"SNAPSHOT UPLOAD FAIL attempt={attempt}/5: {e!r}\", flush=True)\n"
            "                time.sleep(min(30, 5 * attempt))\n"
            "        else:\n"
            "            print(f\"SNAPSHOT UPLOAD GAVE UP iters_done={it}: {last!r} -- continuing\", flush=True)\n"
            "    if gate and V3_COV_HANDS > 0:\n"
            "        hands, dec, mis, allin = t.coverage(V3_COV_HANDS, 0xC0FFEE ^ it)\n"
            "        pct = lambda i: (100.0 * mis[i] / dec[i]) if dec[i] else 0.0\n"
            "        pf_d, pf_m = sum(dec[1:4]), sum(mis[1:4])\n"
            "        pf = (100.0 * pf_m / pf_d) if pf_d else 0.0\n"
            "        print(f\"COVERAGE iters_done={it} hands={hands} miss% pre={pct(0):.1f} \"\n"
            "              f\"flop={pct(1):.1f} turn={pct(2):.1f} river={pct(3):.1f} postflop={pf:.1f} \"\n"
            "              f\"pre_allin_runout={100.0 * allin / max(hands, 1):.1f} decisions={dec}\",\n"
            "              flush=True)\n"
            "    if gate and V3_REGSTATS:\n"
            "        rows, nneg, npos, p50, p90, p99, p999, nmax, pp99, pmax = t.regret_stats()\n"
            "        print(f\"REGSTATS iters_done={it} rows={rows} neg_vals={nneg} pos_vals={npos} \"\n"
            "              f\"neg_p50={p50:.3e} neg_p90={p90:.3e} neg_p99={p99:.3e} neg_p999={p999:.3e} \"\n"
            "              f\"neg_max={nmax:.3e} pos_p99={pp99:.3e} pos_max={pmax:.3e}\", flush=True)\n"
            "    # v3-closeout LESSON (bpv4_v3_closeout.md): upload the train log AT EVERY CHECKPOINT,\n"
            "    # not only at run end -- the v3 K1 hard-terminate killed the box before the end-of-run\n"
            "    # log upload, losing every COVERAGE/REGSTATS line (recovered only by an offline\n"
            "    # re-probe session). The tee'd log file is read as-written-so-far; a checkpoint-lag\n"
            "    # tail is caught by the next checkpoint's upload. Retry then continue -- never crash.\n"
            "    for attempt in range(1, 4):\n"
            "        try:\n"
            "            s3.upload_file(\"/var/log/native-cfr-train.log\", BUCKET,\n"
            "                           RUN_PREFIX + \"/logs/train.log\")\n"
            "            print(f\"TRAINLOG uploaded at iters_done={it}\", flush=True)\n"
            "            break\n"
            "        except Exception as e:\n"
            "            print(f\"TRAINLOG UPLOAD FAIL attempt={attempt}/3: {e!r}\", flush=True)\n"
            "            time.sleep(min(30, 5 * attempt))"
        )
        # fullhist path: rolling checkpoints run every CKPT_EVERY but the pre-registered v3
        # measurement extras (snapshot/COVERAGE/REGSTATS) only at gate-cadence iters (_is_gate).
        if fullhist_key or mrr:
            v3_ckpt_call = "\n        _v3_ckpt_extras(t, _is_gate(t.iters_done()))"
        else:
            v3_ckpt_call = "\n        _v3_ckpt_extras(t)"
    else:
        v3_env_block = ""
        v3_driver_fns = ""
        v3_ckpt_call = ""
    # G8 RAM backstop OVERRIDE (arm-2 lesson: the v3 amendment pre-registered ROW_BUDGET=550M but no
    # export existed, so the driver silently ran its 400M default — a knob-effectiveness gap of exactly
    # the tree-divergence-dead-knobs kind). ``row_budget=None`` renders "" (byte-identical bootstrap);
    # an int exports POKERBOT_ROW_BUDGET, which the driver's module-scope read picks up.
    if row_budget is not None:
        row_budget_env = (f"\nexport POKERBOT_ROW_BUDGET={int(row_budget)}"
                          '\necho "ROW_BUDGET=$POKERBOT_ROW_BUDGET"')
    else:
        row_budget_env = ""
    # Checkpoint machinery. The fullhist+M-RR arm's ``save_bytes()`` blob is ~3-58 GB, which EXCEEDS
    # S3's 5 GiB single-PutObject HARD LIMIT, and a SYNCHRONOUS upload stalls training for minutes per
    # checkpoint. SPOT-RESILIENCE redesign (arm-2 operational addendum, 2026-07-05; the bpv4_h200_arm2
    # first attempt lost 5 boxes in <2h with ZERO checkpoints because the first checkpoint was due ~61
    # min in while spot reclaims arrived ~69 min apart):
    #   * serialize -> LOCAL disk write (fast, blocking; the launcher sizes the root gp3 volume via
    #     --ckpt-volume-gb) -> DETACHED uploader process (/opt/pokerbot/ckpt_upload.py) so training
    #     continues during the multi-GB S3 upload;
    #   * the uploader writes a tiny `iter_XXXXXX.nlcfr.ok` JSON marker (size+sha256) ONLY AFTER the
    #     S3 object is complete -- resume trusts markers exclusively, so a partial upload (reclaim
    #     mid-upload) is invisible to resume;
    #   * `latest.nlcfr` is NO LONGER written on this path (a 58 GB server-side copy per checkpoint,
    #     and the babysitter used to DOWNLOAD it whole to read 16 header bytes); the babysitter now
    #     reads progress from the marker names;
    #   * S3 retention: the uploader keeps ALL gate-cadence checkpoints (GATE_EVERY, the pre-registered
    #     ladder) + the newest KEEP_ROLLING others, deleting marker-first so resume never sees a
    #     blobless marker. Local files are deleted after a durable upload.
    # Legacy path (both flags off): every placeholder renders the ORIGINAL text (``ckpt_upload_fn`` is
    # the ORIGINAL two-put_object function verbatim; ``_download_latest`` unchanged but now rendered
    # AFTER the upload fn — semantically identical). The bootstrap SHELL gains the build-cache block on
    # ALL paths (documented operational change; NOT byte-identical to pre-2026-07-05 renders).
    if fullhist_key or mrr:
        ckpt_upload_imports = ("\nimport hashlib\nimport io\nimport json\nimport signal\nimport subprocess"
                               "\nfrom boto3.s3.transfer import TransferConfig")
        ckpt_upload_fn = (
            "_XFER = TransferConfig(multipart_threshold=64 * 1024 * 1024,\n"
            "                       multipart_chunksize=256 * 1024 * 1024,\n"
            "                       max_concurrency=1, use_threads=False)\n"
            "CKPT_DIR = \"/opt/pokerbot/ckpt\"\n"
            "GATE_EVERY = " + str(int(gate_every)) + "  # pre-registered gate/snapshot cadence; 0 => every checkpoint is a gate\n"
            "KEEP_ROLLING = 3  # newest non-gate checkpoints kept on S3 (gates are kept forever)\n"
            "_UPLOADS = []\n"
            "_SPAWN_KW = {\"start_new_session\": True} if os.name == \"posix\" else {}\n"
            "\n"
            "\n"
            "def _is_gate(it):\n"
            "    return GATE_EVERY <= 0 or it % GATE_EVERY == 0 or it >= ITERS\n"
            "\n"
            "\n"
            "def _next_stop(cur):\n"
            "    # next training-loop boundary = earliest of (next CKPT_EVERY multiple, next GATE_EVERY\n"
            "    # multiple, ITERS) -- rolling resume checkpoints land every CKPT_EVERY AND the\n"
            "    # pre-registered gate ladder (e.g. 2.5k/5k/.../20k for arm-2) is always hit exactly.\n"
            "    nxt = ((cur // CKPT_EVERY) + 1) * CKPT_EVERY\n"
            "    if GATE_EVERY > 0:\n"
            "        nxt = min(nxt, ((cur // GATE_EVERY) + 1) * GATE_EVERY)\n"
            "    return min(nxt, ITERS)\n"
            "\n"
            "\n"
            "def _sha256_file(path):\n"
            "    h = hashlib.sha256()\n"
            "    with open(path, \"rb\") as f:\n"
            "        for chunk in iter(lambda: f.read(1 << 24), b\"\"):\n"
            "            h.update(chunk)\n"
            "    return h.hexdigest()\n"
            "\n"
            "\n"
            "class _BlobReader(io.RawIOBase):\n"
            "    # Non-copying seekable file-like over the save_bytes() blob. upload_fileobj streams it in\n"
            "    # 256 MB multipart parts WITHOUT io.BytesIO(blob)'s full second copy (that copy would add\n"
            "    # ~55 GB at the checkpoint RAM peak on the 256 GB box and risk OOM). use_threads=False keeps\n"
            "    # reads serial so the single shared position is race-free.\n"
            "    def __init__(self, buf):\n"
            "        self._mv = memoryview(buf)\n"
            "        self._pos = 0\n"
            "    def readable(self):\n"
            "        return True\n"
            "    def seekable(self):\n"
            "        return True\n"
            "    def readinto(self, b):\n"
            "        n = min(len(b), len(self._mv) - self._pos)\n"
            "        b[:n] = self._mv[self._pos:self._pos + n]\n"
            "        self._pos += n\n"
            "        return n\n"
            "    def seek(self, off, whence=0):\n"
            "        if whence == 0:\n"
            "            self._pos = off\n"
            "        elif whence == 1:\n"
            "            self._pos += off\n"
            "        else:\n"
            "            self._pos = len(self._mv) + off\n"
            "        return self._pos\n"
            "    def tell(self):\n"
            "        return self._pos\n"
            "\n"
            "\n"
            "def _upload_sync(blob, iters_done):\n"
            "    # In-RAM multipart upload + marker: the fallback when the local ckpt volume is\n"
            "    # unavailable/full. Retry transient S3 errors; NEVER crash the run on an upload blip.\n"
            "    key = RUN_PREFIX + f\"/checkpoints/iter_{iters_done:06d}.nlcfr\"\n"
            "    last = None\n"
            "    for attempt in range(1, 6):\n"
            "        try:\n"
            "            s3.upload_fileobj(_BlobReader(blob), BUCKET, key, Config=_XFER)\n"
            "            sha = hashlib.sha256(blob).hexdigest()\n"
            "            s3.put_object(Bucket=BUCKET, Key=key + \".ok\",\n"
            "                          Body=json.dumps({\"iters_done\": iters_done, \"bytes\": len(blob),\n"
            "                                           \"sha256\": sha}).encode())\n"
            "            print(f\"CKPT iters_done={iters_done} -> s3://{BUCKET}/{key} ({len(blob)} bytes, multipart SYNC + marker)\", flush=True)\n"
            "            return\n"
            "        except Exception as e:\n"
            "            last = e\n"
            "            print(f\"CKPT UPLOAD FAIL attempt={attempt}/5 iters_done={iters_done}: {e!r}\", flush=True)\n"
            "            time.sleep(min(30, 5 * attempt))\n"
            "    print(f\"CKPT UPLOAD GAVE UP after 5 attempts iters_done={iters_done}: {last!r} -- continuing\", flush=True)\n"
            "\n"
            "\n"
            "def _handoff_upload(path, iters_done, wait, base_iter=None, seq=None):\n"
            "    # DETACHED (or wait) S3 upload of a locally-durable checkpoint file; ckpt_upload.py writes\n"
            "    # the .ok marker AFTER the upload completes and prunes rolling checkpoints. A DELTA passes\n"
            "    # base_iter+seq (extra argv) so the uploader writes an .nldlt.ok marker carrying the chain link.\n"
            "    args = [sys.executable, \"/opt/pokerbot/ckpt_upload.py\", path, str(iters_done)]\n"
            "    if base_iter is not None:\n"
            "        args += [str(base_iter), str(seq)]\n"
            "    if wait:\n"
            "        rc = subprocess.call(args)\n"
            "        print(f\"CKPT_UPLOAD_SYNC iters_done={iters_done} rc={rc}\", flush=True)\n"
            "    else:\n"
            "        logf = open(\"/var/log/ckpt_upload.log\", \"ab\")\n"
            "        _UPLOADS.append(subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT,\n"
            "                                         **_SPAWN_KW))\n"
            "    for pr in [p for p in _UPLOADS if p.poll() is not None]:\n"
            "        _UPLOADS.remove(pr)\n"
            "\n"
            "\n"
            "def _upload(blob, iters_done, wait=False):\n"
            "    # LEGACY in-RAM path (fallback / callers that still hold a blob): local durable write from\n"
            "    # an in-memory blob, then DETACHED S3 upload. The hot path is now _ckpt's direct-to-file\n"
            "    # serialize (no PyBytes blob); this remains for the SYNC-fallback and end-of-run wait.\n"
            "    t0 = time.time()\n"
            "    try:\n"
            "        os.makedirs(CKPT_DIR, exist_ok=True)\n"
            "        path = os.path.join(CKPT_DIR, f\"iter_{iters_done:06d}.nlcfr\")\n"
            "        with open(path + \".tmp\", \"wb\") as f:\n"
            "            f.write(blob)\n"
            "            f.flush()\n"
            "            os.fsync(f.fileno())\n"
            "        os.replace(path + \".tmp\", path)\n"
            "    except OSError as e:\n"
            "        print(f\"CKPT local write failed ({e!r}) -- falling back to SYNC in-RAM upload\", flush=True)\n"
            "        _upload_sync(blob, iters_done)\n"
            "        return\n"
            "    print(f\"CKPT_LOCAL iters_done={iters_done} bytes={len(blob)} write={time.time() - t0:.1f}s -> {path}\", flush=True)\n"
            "    _handoff_upload(path, iters_done, wait)\n"
            "\n"
            "\n"
            "# DELTA CHECKPOINTS (checkpoint-speed R2, 2026-07-06). Gated by POKERBOT_DELTA_CKPT=1 (default\n"
            "# OFF => every checkpoint is a full base, BYTE-IDENTICAL to the pre-R2 driver). When ON: a GATE\n"
            "# boundary and the first checkpoint after a (re)start write a FULL BASE (kept forever / resumable\n"
            "# root); the up-to-(GATE_EVERY/CKPT_EVERY - 1) rolling checkpoints between gates write a sparse\n"
            "# NLDLT001 DELTA of only the rows touched since the base. Resume applies the newest verified base\n"
            "# + its contiguous verified delta chain. Chain length is bounded by GATE_EVERY/CKPT_EVERY (arm-2b:\n"
            "# 2500/250 = 10), so a resume replays <= 9 deltas.\n"
            "DELTA_CKPT = os.environ.get(\"POKERBOT_DELTA_CKPT\", \"0\") == \"1\"\n"
            "_BASE = {\"iter\": None, \"seq\": 0}  # current delta-chain base_iter + last delta seq (None => next ckpt is a base)\n"
            "\n"
            "\n"
            "def _ckpt(t, wait=False):\n"
            "    # CHECKPOINT-SPEED path (2026-07-06): serialize DIRECTLY to the local ckpt file in Rust\n"
            "    # (save_bytes_to_file, GIL released) -- NO ~58 GB PyBytes copy and NO redundant Python\n"
            "    # f.write(blob) pass. The in-process training pause is this single serialize+write; the S3\n"
            "    # upload stays detached. On ANY OSError (ckpt volume full/unavailable) fall back to the\n"
            "    # in-RAM save_bytes() + SYNC multipart upload so a checkpoint is never silently skipped.\n"
            "    it = t.iters_done()\n"
            "    t0 = time.time()\n"
            "    # DELTA decision: base at a gate / first-after-restart / full-dirty; delta otherwise.\n"
            "    want_delta = (DELTA_CKPT and _BASE[\"iter\"] is not None and not _is_gate(it))\n"
            "    if want_delta:\n"
            "        try:\n"
            "            os.makedirs(CKPT_DIR, exist_ok=True)\n"
            "            dpath = os.path.join(CKPT_DIR, f\"iter_{it:06d}.nldlt\")\n"
            "            nbytes = t.save_delta_to_file(dpath + \".tmp\", _BASE[\"iter\"], _BASE[\"seq\"] + 1)\n"
            "        except OSError as e:\n"
            "            print(f\"CKPT delta write failed ({e!r}) -- writing a full base instead\", flush=True)\n"
            "            nbytes = None\n"
            "        if nbytes is not None:\n"
            "            os.replace(dpath + \".tmp\", dpath)\n"
            "            _BASE[\"seq\"] += 1\n"
            "            t.reset_dirty()  # window captured & durable-locally; next window starts clean\n"
            "            (dc, ds) = t.dirty_counts()  # 0,0 right after reset -- log the pre-reset was done in _handoff\n"
            "            print(f\"CKPT_DELTA iters_done={it} base={_BASE['iter']} seq={_BASE['seq']} \"\n"
            "                  f\"bytes={nbytes} wall={time.time() - t0:.1f}s\", flush=True)\n"
            "            _handoff_upload(dpath, it, wait, base_iter=_BASE[\"iter\"], seq=_BASE[\"seq\"])\n"
            "            return\n"
            "        # delta unavailable (unarmed / full-dirty / OSError) -> fall through to a full base.\n"
            "    try:\n"
            "        os.makedirs(CKPT_DIR, exist_ok=True)\n"
            "        path = os.path.join(CKPT_DIR, f\"iter_{it:06d}.nlcfr\")\n"
            "        nbytes = t.save_bytes_to_file(path + \".tmp\")\n"
            "        os.replace(path + \".tmp\", path)\n"
            "    except OSError as e:\n"
            "        print(f\"CKPT direct-to-file failed ({e!r}) -- falling back to in-RAM save_bytes\", flush=True)\n"
            "        blob = t.save_bytes()\n"
            "        _upload(blob, it, wait=wait)\n"
            "        return\n"
            "    if DELTA_CKPT:\n"
            "        _BASE[\"iter\"] = it  # this base roots the next delta chain\n"
            "        _BASE[\"seq\"] = 0\n"
            "        t.reset_dirty()\n"
            "    print(f\"CKPT_SERIALIZE iters_done={it} bytes={nbytes} wall={time.time() - t0:.1f}s (direct-to-file)\", flush=True)\n"
            "    _handoff_upload(path, it, wait)\n"
            "\n"
            "\n"
            "# ---- GRACEFUL SPOT-RECLAIM FLUSH (2026-07-06). spot_watch.sh touches /tmp/spot_reclaim the\n"
            "# moment the 2-min IMDS reclaim notice lands (and only pkills 100 s later as a fallback).\n"
            "# Training runs in CHUNK-iter slices (~5 s reaction latency at 0.77 it/s) -- t.train(a);\n"
            "# t.train(b) is BYTE-IDENTICAL to t.train(a+b) (verified vs the arm2b golden gate sha) -- so\n"
            "# the driver notices the flag fast, banks the un-checkpointed window as ONE synchronous delta\n"
            "# (+ .ok marker + train.log), and exits 0, instead of losing up to CKPT_EVERY iters of compute\n"
            "# to the old immediate pkill. /tmp/spot_rebalance (the earlier, softer EC2 rebalance\n"
            "# recommendation) triggers one extra delta through the NORMAL detached path, then training\n"
            "# continues.\n"
            "RECLAIM_FLAG = \"/tmp/spot_reclaim\"\n"
            "REBAL_FLAG = \"/tmp/spot_rebalance\"\n"
            "CHUNK = int(os.environ.get(\"POKERBOT_CKPT_CHUNK\", \"4\"))  # iters per train slice\n"
            "_STOP = {\"reclaim\": False, \"rebal_done\": False}\n"
            "\n"
            "\n"
            "def _on_sigterm(_sig, _frm):\n"
            "    # spot_watch's fallback pkill -TERM (100 s post-notice) lands here: even if the flag file\n"
            "    # was somehow missed, the next chunk boundary still flushes if any window remains.\n"
            "    _STOP[\"reclaim\"] = True\n"
            "\n"
            "\n"
            "signal.signal(signal.SIGTERM, _on_sigterm)\n"
            "\n"
            "\n"
            "def _train_to(t, upto):\n"
            "    # Chunked training between checkpoint boundaries (bit-identical to one t.train(step)):\n"
            "    # reclaim/rebalance flags are polled every CHUNK iters instead of every CKPT_EVERY.\n"
            "    while t.iters_done() < upto:\n"
            "        if _STOP[\"reclaim\"] or os.path.exists(RECLAIM_FLAG):\n"
            "            _STOP[\"reclaim\"] = True\n"
            "            return\n"
            "        t.train(min(CHUNK, upto - t.iters_done()))\n"
            "        if (DELTA_CKPT and not _STOP[\"rebal_done\"] and _BASE[\"iter\"] is not None\n"
            "                and t.iters_done() < upto and os.path.exists(REBAL_FLAG)):\n"
            "            # strictly mid-window (iters_done < upto) so the extra delta's iter can never\n"
            "            # collide with an already-written boundary checkpoint's S3 key. Once only.\n"
            "            _STOP[\"rebal_done\"] = True\n"
            "            print(f\"REBALANCE early-signal ckpt at iters_done={t.iters_done()}\", flush=True)\n"
            "            _ckpt(t)\n"
            "\n"
            "\n"
            "def _emergency_flush(t):\n"
            "    # SPOT RECLAIM: bank the un-checkpointed window as one delta, SYNCHRONOUSLY (a detached\n"
            "    # uploader would die with the box). DELTA MODE ONLY -- a full base cannot fit the 2-min\n"
            "    # window (280 s serialize alone); non-delta mode keeps today's behavior (exit; the last\n"
            "    # cadence checkpoint is already durable). Marker JSON shape == the detached uploader's,\n"
            "    # so resume treats an emergency delta at a non-cadence iter exactly like any chain link.\n"
            "    it = t.iters_done()\n"
            "    if not DELTA_CKPT or _BASE[\"iter\"] is None:\n"
            "        print(f\"RECLAIM at iters_done={it}: no delta chain to flush \"\n"
            "              f\"(delta_ckpt={DELTA_CKPT}) -- exiting\", flush=True)\n"
            "        return\n"
            "    (dc, ds) = t.dirty_counts()\n"
            "    if dc == 0 and ds == 0:\n"
            "        print(f\"RECLAIM at iters_done={it}: window already checkpointed -- nothing to flush\",\n"
            "              flush=True)\n"
            "        return\n"
            "    t0 = time.time()\n"
            "    try:\n"
            "        os.makedirs(CKPT_DIR, exist_ok=True)\n"
            "        dpath = os.path.join(CKPT_DIR, f\"iter_{it:06d}.nldlt\")\n"
            "        nbytes = t.save_delta_to_file(dpath + \".tmp\", _BASE[\"iter\"], _BASE[\"seq\"] + 1)\n"
            "        if nbytes is None:\n"
            "            print(f\"RECLAIM flush unavailable (delta unarmed/full-dirty) at iters_done={it} \"\n"
            "                  f\"-- exiting\", flush=True)\n"
            "            return\n"
            "        os.replace(dpath + \".tmp\", dpath)\n"
            "        _BASE[\"seq\"] += 1\n"
            "        t.reset_dirty()\n"
            "        key = RUN_PREFIX + f\"/checkpoints/iter_{it:06d}.nldlt\"\n"
            "        size = os.path.getsize(dpath)\n"
            "        sha = _sha256_file(dpath)\n"
            "        s3.upload_file(dpath, BUCKET, key, Config=_DL_XFER)\n"
            "        s3.put_object(Bucket=BUCKET, Key=key + \".ok\",\n"
            "                      Body=json.dumps({\"iters_done\": it, \"bytes\": size, \"sha256\": sha,\n"
            "                                       \"base_iter\": _BASE[\"iter\"],\n"
            "                                       \"seq\": _BASE[\"seq\"]}).encode())\n"
            "    except Exception as e:\n"
            "        print(f\"RECLAIM flush FAILED at iters_done={it}: {e!r} -- exiting \"\n"
            "              f\"(last cadence checkpoint stands)\", flush=True)\n"
            "        return\n"
            "    # best-effort: give any in-flight detached uploads the remaining window (their markers\n"
            "    # complete the chain the emergency delta extends). Never blocks past ~95 s total.\n"
            "    for pr in list(_UPLOADS):\n"
            "        if pr.poll() is None:\n"
            "            try:\n"
            "                pr.wait(timeout=max(1, int(95 - (time.time() - t0))))\n"
            "            except Exception:\n"
            "                break\n"
            "    try:\n"
            "        s3.upload_file(\"/var/log/native-cfr-train.log\", BUCKET,\n"
            "                       RUN_PREFIX + \"/logs/train.log\")\n"
            "    except Exception as e:\n"
            "        print(f\"RECLAIM train.log upload failed: {e!r}\", flush=True)\n"
            "    print(f\"EMERGENCY_FLUSH iters_done={it} bytes={size} wall={time.time() - t0:.1f}s banked\",\n"
            "          flush=True)"
        )
    else:
        ckpt_upload_imports = ""
        ckpt_upload_fn = (
            "def _upload(blob, iters_done):\n"
            "    # versioned per-iter checkpoint + a `latest` pointer (atomic on S3: a single PutObject).\n"
            "    key = RUN_PREFIX + f\"/checkpoints/iter_{iters_done:06d}.nlcfr\"\n"
            "    s3.put_object(Bucket=BUCKET, Key=key, Body=blob)\n"
            "    s3.put_object(Bucket=BUCKET, Key=LATEST_KEY, Body=blob)\n"
            "    print(f\"CKPT iters_done={iters_done} -> s3://{BUCKET}/{key} ({len(blob)} bytes)\", flush=True)"
        )
    # Blueprint-v4 Stage-0 G8 ROW-BUDGET RAM backstop. ADDITIVE + default-off BYTE-IDENTICAL: gated the
    # SAME way as the multipart uploader above (emitted ONLY when fullhist_key or mrr), so with both flags
    # off ``row_budget_const`` and ``row_budget_block`` are "" and the rendered driver is byte-for-byte the
    # bp2/bp3 script. Both are attached as SUFFIXES to existing driver lines (not their own placeholder
    # line) so the off-path emits zero extra bytes (no stray blank line). ``row_budget_const`` reads the
    # env budget once at module scope (default 400_000_000 = design G8; 0 disables). ``row_budget_block``
    # runs after each training chunk: when the strat-row count (t.n_infosets(), the SAME accessor the
    # PROGRESS/CKPT prints use, backed by n_strat_infosets()) reaches the budget it writes a FINAL
    # checkpoint via the SAME multipart _upload (versioned + `latest`) and breaks, letting the driver fall
    # through to its normal end-of-run log upload + the bootstrap dead-man shutdown (NO hard crash). This
    # mirrors ab_train.rs POKERBOT_AB_ROW_CAP (checkpoint-then-exit-0, partial evidence per design 7.1.2):
    # the fullhist+M-RR table grows ~42k rows/iter with no plateau, and on a fixed-RAM box its committed
    # footprint eventually crosses the box's memory and the OS OOM-kills the run BEFORE a checkpoint lands
    # (which would permanently break spot auto-resume). Row count is a deterministic, allocation-free proxy
    # for committed memory.
    # v3 + snapshot_avg: the strat map (t.n_infosets()) is preflop-bounded (~60M rows) while
    # cum_regret keeps growing -- the G8 RAM backstop must watch BOTH maps or it never fires.
    # ``rows_expr`` renders as the ORIGINAL text when v3 is off (byte-identity, sha-compared).
    rows_expr = ("max(t.n_infosets(), t.n_cum_infosets())"
                 if (v3 and v3.get("snapshot_avg")) else "t.n_infosets()")
    if fullhist_key or mrr:
        row_budget_const = (
            "\n"
            "ROW_BUDGET = int(os.environ.get(\"POKERBOT_ROW_BUDGET\", \"400000000\"))  # G8 RAM backstop (rows); 0=off"
        )
        # G8 PRE-TRAIN guard (requirement d): a spot-resumed box that LOADED an already-at/past-budget
        # checkpoint must stop BEFORE allocating another CKPT_EVERY chunk — otherwise t.train(step) runs a
        # whole native chunk (growing the table hundreds of millions of rows further) before the post-chunk
        # check can fire, re-creating the exact OOM the backstop exists to prevent (and, if the OOM lands
        # before the post-chunk _upload, an OOM-crash-resume livelock: every resume reloads the same
        # at-budget checkpoint and dies again). Emitted as a SUFFIX on the `while` header (its own line), so
        # with both flags off it is "" and the loop is byte-for-byte the bp2/bp3 driver. Same accessor
        # (t.n_infosets() -> n_strat_infosets, the PROGRESS/CKPT print accessor) and same `>=` threshold as
        # the post-chunk check. NO re-upload here: the loaded checkpoint IS the durable at-budget one (the
        # previous box wrote it via the post-chunk _upload), and calling save_bytes() would allocate the very
        # ~55-90 GB serialize blob we are trying to avoid — print + break lets the driver fall through to its
        # end-of-run log upload + the on-box dead-man shutdown (NO hard crash). Mirrors ab_train.rs
        # POKERBOT_AB_ROW_CAP "stop before growing further" semantics (design 7.1.2).
        row_budget_pre = (
            "\n"
            "        # G8 ROW-BUDGET pre-train guard: a spot-resumed box that loaded an at/past-budget\n"
            "        # checkpoint stops BEFORE allocating another chunk (requirement d). Same accessor +\n"
            "        # threshold as the post-chunk check; the loaded checkpoint is already durable so we do\n"
            "        # NOT re-serialize (avoids the ~55-90GB blob) -- print + break, fall through to the\n"
            "        # end-of-run log upload + on-box dead-man shutdown (NO hard crash).\n"
            f"        if ROW_BUDGET > 0 and {rows_expr} >= ROW_BUDGET:\n"
            "            print(f\"ROW_BUDGET_STOP (resume/pre-train) budget={ROW_BUDGET} \"\n"
            "                  f\"rows={t.n_infosets()} iters_done={t.iters_done()} -- already at/past \"\n"
            "                  f\"budget on entry, stopping cleanly (no new chunk; last checkpoint already \"\n"
            "                  f\"durable) (G8 partial evidence, design 7.1.2)\", flush=True)\n"
            "            break"
        )
        row_budget_block = (
            "\n"
            "        # G8 ROW-BUDGET RAM backstop: checkpoint-then-stop cleanly when the table hits the budget\n"
            "        # (mirrors ab_train.rs POKERBOT_AB_ROW_CAP; partial evidence, design 7.1.2). n_infosets()\n"
            "        # is the same strat-row accessor the PROGRESS/CKPT prints use. NEVER hard-crash: write a\n"
            "        # FINAL checkpoint via the SAME multipart _upload (versioned + latest) and break so the\n"
            "        # driver falls through to its end-of-run log upload + the on-box dead-man shutdown.\n"
            f"        if ROW_BUDGET > 0 and {rows_expr} >= ROW_BUDGET:\n"
            "            _ckpt(t, wait=True)\n"
            "            print(f\"ROW_BUDGET_STOP budget={ROW_BUDGET} rows={t.n_infosets()} \"\n"
            "                  f\"iters_done={t.iters_done()} -- wrote FINAL checkpoint (versioned + latest), \"\n"
            "                  f\"stopping cleanly (G8 partial evidence, design 7.1.2)\", flush=True)\n"
            "            break"
        )
    else:
        row_budget_const = ""
        row_budget_pre = ""
        row_budget_block = ""
    # ---- spot-resilience template variables (fullhist path only; legacy renders keep the ORIGINAL
    # text verbatim so bp2/bp3 drivers stay byte-identical) ----
    if fullhist_key or mrr:
        # Resume = newest checkpoint WITH a verified .ok marker (size+sha256) -- NEVER a partial
        # upload. Legacy `latest` pointer stays as a last-resort fallback (pre-marker runs).
        #
        # RESUME-SPEED path (2026-07-06, round-3 prologue optimization): the resume prologue was
        # "download 55-100 GB from S3, then serial-deserialize" -- ~10-15 min, and every spot respawn
        # burns most of an 18-27 min lifetime on it. Two fixes here:
        #   (1) DOWNLOAD THROUGHPUT: `s3.download_file`'s DEFAULT TransferConfig is 10 concurrent 8 MB
        #       ranged GETs -- on a c6a.32xlarge (50 Gbps, 128 vCPU) that leaves the pipe mostly idle. A
        #       tuned TransferConfig (64 MB parts x 64 concurrent ranged GETs, use_threads=True) saturates
        #       many more streams and sustains multiple GB/s off a single S3 object.
        #   (2) NO PYTHON BYTES COPY: `_download_latest` now returns the on-disk PATH (not a ~55-100 GB
        #       Python `bytes`). The driver resumes via `t.load_bytes_from_file(path)` (mmap + PARALLEL
        #       deserialize, GIL released) -- removing both the full-blob Python read AND the serial
        #       single-thread rehash-storm deserialize (measured 9.6x faster load at scale).
        # sha256 is still verified (integrity of a resumed checkpoint is non-negotiable), streamed once
        # before the mmap load.
        download_latest_fn = (
            "_DL_XFER = TransferConfig(multipart_threshold=64 * 1024 * 1024,\n"
            "                          multipart_chunksize=64 * 1024 * 1024,\n"
            "                          max_concurrency=64, use_threads=True)\n"
            "\n"
            "# ---- PRESTAGED RESUME (2026-07-06): the bootstrap forks a background prestage right after\n"
            "# the awscli install (before rustup/pip/build-cache), downloading + sha-verifying the newest\n"
            "# verified base and its contiguous delta chain into CKPT_DIR, then writing an atomic manifest\n"
            "# + .done flag. That overlaps the ~114 s+ resume download with the ~2-3 min toolchain\n"
            "# prologue. The driver consumes a prestaged blob ONLY when its manifest sha/bytes still match\n"
            "# the CURRENT marker (a newer checkpoint may have landed since); any mismatch/failure falls\n"
            "# back to the normal tuned download.\n"
            "PRESTAGE_WAIT_S = 300  # bounded wait for an in-flight prestage before falling back\n"
            "_PRESTAGE = {\"man\": None, \"loaded\": False}\n"
            "\n"
            "\n"
            "def _prestage_manifest():\n"
            "    done = os.path.join(CKPT_DIR, \"prestage.done\")\n"
            "    started = os.path.join(CKPT_DIR, \"prestage.started\")\n"
            "    if not os.path.exists(done) and os.path.exists(started):\n"
            "        print(f\"RESUME prestage in progress -- waiting up to {PRESTAGE_WAIT_S}s\", flush=True)\n"
            "        t0 = time.time()\n"
            "        while time.time() - t0 < PRESTAGE_WAIT_S and not os.path.exists(done):\n"
            "            time.sleep(2)\n"
            "    if not os.path.exists(done):\n"
            "        return None\n"
            "    try:\n"
            "        with open(os.path.join(CKPT_DIR, \"prestage_manifest.json\"), \"r\") as f:\n"
            "            man = json.load(f)\n"
            "        return man if man.get(\"base\") else None\n"
            "    except Exception as e:\n"
            "        print(f\"RESUME prestage manifest unreadable: {e!r} -- ignoring\", flush=True)\n"
            "        return None\n"
            "\n"
            "\n"
            "def _prestaged_take(mk, dst):\n"
            "    # If the prestage manifest holds a verified blob for marker `mk` that STILL matches the\n"
            "    # current marker meta, move it to `dst` and return the meta; else None (caller downloads).\n"
            "    # Never raises -- every failure is a clean fallback to today's path.\n"
            "    try:\n"
            "        if not _PRESTAGE[\"loaded\"]:\n"
            "            _PRESTAGE[\"man\"] = _prestage_manifest()\n"
            "            _PRESTAGE[\"loaded\"] = True\n"
            "        man = _PRESTAGE[\"man\"]\n"
            "        if not man:\n"
            "            return None\n"
            "        ent = man[\"base\"] if man[\"base\"].get(\"marker\") == mk else None\n"
            "        if ent is None:\n"
            "            for d in man.get(\"deltas\", []):\n"
            "                if d.get(\"marker\") == mk:\n"
            "                    ent = d\n"
            "                    break\n"
            "        if ent is None:\n"
            "            return None\n"
            "        meta = json.loads(s3.get_object(Bucket=BUCKET, Key=mk)[\"Body\"].read())\n"
            "        if (ent.get(\"sha256\") != meta[\"sha256\"]\n"
            "                or int(ent.get(\"bytes\", -1)) != int(meta[\"bytes\"])):\n"
            "            print(f\"RESUME prestage STALE for {mk} (marker changed) -- downloading fresh\",\n"
            "                  flush=True)\n"
            "            return None\n"
            "        p = ent.get(\"path\")\n"
            "        if not p or not os.path.exists(p) or os.path.getsize(p) != int(meta[\"bytes\"]):\n"
            "            return None\n"
            "        os.replace(p, dst)  # prestage already sha-verified the file at download time\n"
            "        print(f\"RESUME using prestaged s3://{BUCKET}/{mk[:-3]} bytes={meta['bytes']} \"\n"
            "              f\"(download overlapped with bootstrap)\", flush=True)\n"
            "        return meta\n"
            "    except Exception as e:\n"
            "        print(f\"RESUME prestage take failed for {mk}: {e!r} -- downloading\", flush=True)\n"
            "        return None\n"
            "\n"
            "\n"
            "def _dl_verified(mk, dst):\n"
            "    # Download the blob for marker `mk` to `dst` (tuned parallel ranged GETs), verifying\n"
            "    # size+sha256; return its meta dict or raise. Shared by base + delta resume so\n"
            "    # verification is identical.\n"
            "    meta = json.loads(s3.get_object(Bucket=BUCKET, Key=mk)[\"Body\"].read())\n"
            "    key = mk[:-3]\n"
            "    t0 = time.time()\n"
            "    s3.download_file(BUCKET, key, dst, Config=_DL_XFER)\n"
            "    dl = time.time() - t0\n"
            "    sz = os.path.getsize(dst)\n"
            "    if sz != int(meta[\"bytes\"]):\n"
            "        raise ValueError(f\"size {sz} != {meta['bytes']}\")\n"
            "    sha = _sha256_file(dst)\n"
            "    if sha != meta[\"sha256\"]:\n"
            "        raise ValueError(f\"sha {sha} != {meta['sha256']}\")\n"
            "    gbps = (sz / 1e9) / dl if dl > 0 else 0.0\n"
            "    print(f\"RESUME dl s3://{BUCKET}/{key} bytes={sz} dl={dl:.1f}s ({gbps:.2f} GB/s)\",\n"
            "          flush=True)\n"
            "    return meta\n"
            "\n"
            "\n"
            "def _download_latest():\n"
            "    # Returns (base_path, base_iter) for the newest VERIFIED full base (tuned download;\n"
            "    # the driver mmap-loads the path via load_bytes_from_file), or (None, None).\n"
            "    try:\n"
            "        pg = s3.get_paginator(\"list_objects_v2\").paginate(\n"
            "            Bucket=BUCKET, Prefix=RUN_PREFIX + \"/checkpoints/\")\n"
            "        oks = sorted(o[\"Key\"] for page in pg for o in page.get(\"Contents\", [])\n"
            "                     if o[\"Key\"].endswith(\".nlcfr.ok\"))\n"
            "    except Exception as e:\n"
            "        print(f\"RESUME marker list failed: {e!r}\", flush=True)\n"
            "        oks = []\n"
            "    for mk in reversed(oks):  # iter_%06d zero-padding => lexicographic == numeric\n"
            "        try:\n"
            "            os.makedirs(CKPT_DIR, exist_ok=True)\n"
            "            path = os.path.join(CKPT_DIR, \"resume.nlcfr\")\n"
            "            meta = _prestaged_take(mk, path)\n"
            "            if meta is None:\n"
            "                meta = _dl_verified(mk, path)\n"
            "            print(f\"RESUME verified base s3 iter={meta['iters_done']} bytes={meta['bytes']} \"\n"
            "                  f\"sha256={meta['sha256'][:12]}..\", flush=True)\n"
            "            return path, int(meta[\"iters_done\"])\n"
            "        except Exception as e:\n"
            "            print(f\"RESUME candidate {mk} rejected: {e!r} -- trying next\", flush=True)\n"
            "    try:  # legacy pointer fallback (pre-marker runs only): stage to disk, return path\n"
            "        os.makedirs(CKPT_DIR, exist_ok=True)\n"
            "        path = os.path.join(CKPT_DIR, \"resume.nlcfr\")\n"
            "        s3.download_file(BUCKET, LATEST_KEY, path, Config=_DL_XFER)\n"
            "        return path, None\n"
            "    except Exception:\n"
            "        return None, None\n"
            "\n"
            "\n"
            "def _resume_apply_deltas(t, base_iter):\n"
            "    # After loading a full base at `base_iter`, apply its CONTIGUOUS verified delta chain\n"
            "    # (base_iter matches, seq = 1,2,3,... with no gap). A chain STOPS at the first missing /\n"
            "    # corrupt / non-contiguous link (fail-closed: an older-but-consistent state, never a gap).\n"
            "    # Returns the number of deltas applied. No-op unless DELTA_CKPT (the chain doesn't exist).\n"
            "    if not DELTA_CKPT or base_iter is None:\n"
            "        return 0\n"
            "    try:\n"
            "        pg = s3.get_paginator(\"list_objects_v2\").paginate(\n"
            "            Bucket=BUCKET, Prefix=RUN_PREFIX + \"/checkpoints/\")\n"
            "        dks = sorted(o[\"Key\"] for page in pg for o in page.get(\"Contents\", [])\n"
            "                     if o[\"Key\"].endswith(\".nldlt.ok\"))\n"
            "    except Exception as e:\n"
            "        print(f\"RESUME delta list failed: {e!r} -- base-only resume\", flush=True)\n"
            "        return 0\n"
            "    # Index verified deltas rooted at this base by seq (drop any whose header meta mismatches).\n"
            "    by_seq = {}\n"
            "    for mk in dks:\n"
            "        try:\n"
            "            meta = json.loads(s3.get_object(Bucket=BUCKET, Key=mk)[\"Body\"].read())\n"
            "            if int(meta.get(\"base_iter\", -1)) == base_iter:\n"
            "                by_seq[int(meta[\"seq\"])] = mk\n"
            "        except Exception:\n"
            "            pass\n"
            "    applied, prev_seq = 0, 0\n"
            "    while (prev_seq + 1) in by_seq:\n"
            "        mk = by_seq[prev_seq + 1]\n"
            "        try:\n"
            "            os.makedirs(CKPT_DIR, exist_ok=True)\n"
            "            dpath = os.path.join(CKPT_DIR, \"resume.nldlt\")\n"
            "            if _prestaged_take(mk, dpath) is None:\n"
            "                _dl_verified(mk, dpath)\n"
            "            with open(dpath, \"rb\") as f:\n"
            "                dblob = f.read()\n"
            "            os.remove(dpath)\n"
            "            res = t.apply_delta(dblob, base_iter, prev_seq)\n"
            "            if res is None:\n"
            "                print(f\"RESUME delta seq={prev_seq + 1} chain-verify REJECTED -- stopping chain\", flush=True)\n"
            "                break\n"
            "            (it_now, seq_now) = res\n"
            "            prev_seq = seq_now\n"
            "            applied += 1\n"
            "            print(f\"RESUME applied delta seq={seq_now} -> iters_done={it_now}\", flush=True)\n"
            "        except Exception as e:\n"
            "            print(f\"RESUME delta seq={prev_seq + 1} rejected: {e!r} -- stopping chain\", flush=True)\n"
            "            break\n"
            "    # After resume, the loaded base_iter roots the CURRENT chain and prev_seq is where it left off\n"
            "    # -- so new checkpoints CONTINUE the chain rather than restarting it (avoids re-writing a base).\n"
            "    _BASE[\"iter\"] = base_iter\n"
            "    _BASE[\"seq\"] = prev_seq\n"
            "    return applied"
        )
        next_stop_expr = "_next_stop(t.iters_done())"
        ckpt_stmt = "_ckpt(t)"
        # Graceful-reclaim loop shape: chunked train + flush-and-exit on the reclaim flag. The
        # emergency flush runs BEFORE the row-budget check (a full-base row-budget checkpoint can
        # never fit the reclaim window).
        train_stmt = "_train_to(t, nxt)"
        reclaim_break = (
            "\n        if _STOP[\"reclaim\"]:\n"
            "            _emergency_flush(t)\n"
            "            return"
        )
        # Resume (fullhist): download the newest verified BASE to disk (tuned parallel GETs),
        # mmap-load it via load_bytes_from_file (parallel deserialize, no full-blob Python read),
        # apply its contiguous verified delta chain, then arm delta tracking AFTER the load --
        # install_loaded_tables replaces the whole NlheTables (dirty resets to None), so an arm
        # BEFORE the load is silently WIPED and every save_delta_to_file declines to a full base
        # (profile run i-0f9aa667065ac1388, 2026-07-06: 60 GB base + ~250 s pause every 100 iters,
        # rate 0.194 it/s). Arm-after-load + reset_dirty gives the first window a clean start, and
        # _BASE roots at the RESUMED marker so the first post-restart checkpoint is a cheap DELTA
        # (the old first-ckpt-is-a-base rule predates resume-aware chains; gates still force bases,
        # so chain length stays bounded by GATE_EVERY/CKPT_EVERY + the few signal deltas).
        # When DELTA_CKPT is off, arm is skipped and _resume_apply_deltas is a no-op => same as
        # base-only resume.
        resume_block = (
            "    resume_path, base_iter = _download_latest()\n"
            "    if resume_path is not None and t.load_bytes_from_file(resume_path):\n"
            "        try:\n"
            "            os.remove(resume_path)\n"
            "        except OSError:\n"
            "            pass\n"
            "        nd = _resume_apply_deltas(t, base_iter)\n"
            "        if DELTA_CKPT:\n"
            "            t.arm_delta_tracking()  # AFTER load: load replaces the tables (unarms)\n"
            "            t.reset_dirty()\n"
            "            if base_iter is not None:\n"
            "                _BASE[\"iter\"] = base_iter\n"
            "                _BASE[\"seq\"] = nd\n"
            "                print(f\"DELTA_CKPT armed post-load; chain rooted at base={base_iter} seq={nd}\", flush=True)\n"
            "            else:\n"
            "                print(\"DELTA_CKPT armed post-load; legacy resume -> first ckpt is a full base\", flush=True)\n"
            "        # Keep the 'RESUMED from latest checkpoint at iters_done=N' prefix VERBATIM (parsed by\n"
            "        # watch_bpv4_stage0.ps1); append the delta-chain detail after it.\n"
            "        print(f\"RESUMED from latest checkpoint at iters_done={t.iters_done()} \"\n"
            "              f\"(base_iter={base_iter}, +{nd} deltas)\", flush=True)\n"
            "    else:\n"
            "        if DELTA_CKPT:\n"
            "            t.arm_delta_tracking()  # fresh start: no load will wipe it\n"
            "            print(\"DELTA_CKPT armed (fresh start); first ckpt is the chain-root base\", flush=True)\n"
            "        print(\"FRESH start (no resumable checkpoint)\", flush=True)\n"
            "    # drop any UNCONSUMED prestaged blobs (stale prestage / fresh start) -- keeps the ckpt\n"
            "    # staging volume clear for the ~58-69 GB base serializes.\n"
            "    try:\n"
            "        for fn in (os.listdir(CKPT_DIR) if os.path.isdir(CKPT_DIR) else []):\n"
            "            if fn.startswith(\"prestage_\") and (fn.endswith(\".nlcfr\") or fn.endswith(\".nldlt\")):\n"
            "                os.remove(os.path.join(CKPT_DIR, fn))\n"
            "    except OSError:\n"
            "        pass\n"
            "    # POST-RESUME TELEMETRY (2026-07-06): upload train.log NOW, not only at checkpoints --\n"
            "    # a storm-reclaimed box that never reaches its first marker otherwise leaves ZERO\n"
            "    # prologue telemetry (RESUME dl GB/s, load wall), exactly when it's needed most.\n"
            "    try:\n"
            "        s3.upload_file(\"/var/log/native-cfr-train.log\", BUCKET,\n"
            "                       RUN_PREFIX + \"/logs/train.log\")\n"
            "        print(\"TRAINLOG uploaded post-resume\", flush=True)\n"
            "    except Exception as e:\n"
            "        print(f\"post-resume TRAINLOG upload failed: {e!r}\", flush=True)"
        )
        # end-of-run: block on in-flight detached uploads BEFORE the driver exits -- the bootstrap
        # shuts the box down right after the driver, which would otherwise kill the final upload.
        ckpt_wait_block = (
            "\n    for pr in list(_UPLOADS):\n"
            "        if pr.poll() is None:\n"
            "            print(f\"WAITING for in-flight checkpoint upload pid={pr.pid}\", flush=True)\n"
            "        pr.wait()"
        )
    else:
        download_latest_fn = (
            "def _download_latest():\n"
            "    try:\n"
            "        obj = s3.get_object(Bucket=BUCKET, Key=LATEST_KEY)\n"
            "        return obj[\"Body\"].read()\n"
            "    except Exception:\n"
            "        return None"
        )
        next_stop_expr = "min(t.iters_done() + CKPT_EVERY, ITERS)"
        ckpt_stmt = "_upload(t.save_bytes(), t.iters_done())"
        ckpt_wait_block = ""
        # Legacy path: ORIGINAL single train call, no reclaim machinery (byte-identical render).
        train_stmt = "t.train(step)"
        reclaim_break = ""
        # Legacy resume (both flags off): single-blob _download_latest returning a BLOB (no delta
        # chain, no path staging on the bp2/bp3 path). Byte-identical to the pre-R2/R3 inline block.
        resume_block = (
            "    blob = _download_latest()\n"
            "    if blob is not None and t.load_bytes(blob):\n"
            "        print(f\"RESUMED from latest checkpoint at iters_done={t.iters_done()}\", flush=True)\n"
            "    else:\n"
            "        print(\"FRESH start (no resumable checkpoint)\", flush=True)"
        )
    # v3 arms: print the EFFECTIVE prune config right after the LEVERS line. The LEVERS line's
    # `prune=` value is the INERT dcfr-mode knob (cfg.dcfr_prune_threshold, IGNORED when dcfr=False
    # -- trainer.rs effective_prune_threshold), which confused the arm-2 monitor (`prune=-300000.0`
    # while POKERBOT_PRUNE_C=-50000 governed). Renders "" when v3 is off (byte-identical).
    if v3:
        v3_banner = (
            "\n    print(\"V3 effective: signed_regrets=%s lcfr=%s PRUNE_C=%s prune_start=%s \"\n"
            "          \"explore=%s floor=%s [NOTE: the LEVERS dcfr 'prune=' value above is the inert \"\n"
            "          \"dcfr-mode knob, ignored when dcfr=False]\" % (\n"
            "        os.environ.get(\"POKERBOT_SIGNED_REGRETS\", \"0\"),\n"
            "        os.environ.get(\"POKERBOT_LCFR\", \"0\"),\n"
            "        os.environ.get(\"POKERBOT_PRUNE_C\", \"(off)\"),\n"
            "        os.environ.get(\"POKERBOT_PRUNE_START\", \"(W/2 default)\"),\n"
            "        os.environ.get(\"POKERBOT_PRUNE_EXPLORE\", \"0.05\"),\n"
            "        os.environ.get(\"POKERBOT_REGRET_FLOOR\", \"(C*31/30 default)\")), flush=True)"
        )
    else:
        v3_banner = ""
    # Detached checkpoint uploader (fullhist path): spawned per checkpoint by the driver; writes the
    # .ok marker only AFTER a complete upload, prunes rolling checkpoints (keeps ALL gate-cadence +
    # newest KEEP_ROLLING others; marker deleted FIRST so resume never sees a blobless marker), then
    # deletes its local file. Renders "" (no extra bytes) on the legacy path.
    if fullhist_key or mrr:
        ckpt_uploader_file = (
            "cat > /opt/pokerbot/ckpt_upload.py <<'UPEOF'\n"
            "import hashlib, json, os, re, sys, time\n"
            "import boto3\n"
            "from boto3.s3.transfer import TransferConfig\n"
            "\n"
            "BUCKET = \"" + bucket + "\"\n"
            "RUN_PREFIX = \"" + run_prefix + "\"\n"
            "REGION = \"" + region + "\"\n"
            "GATE_EVERY = " + str(int(gate_every)) + "\n"
            "KEEP_ROLLING = 3\n"
            "\n"
            "path, it = sys.argv[1], int(sys.argv[2])\n"
            "# DELTA (checkpoint-speed R2): 2 extra argv (base_iter, seq) => this is an NLDLT001 delta;\n"
            "# key ends .nldlt and the .ok marker carries the chain link. No extra argv => full base (.nlcfr).\n"
            "is_delta = len(sys.argv) >= 5\n"
            "base_iter = int(sys.argv[3]) if is_delta else None\n"
            "seq = int(sys.argv[4]) if is_delta else None\n"
            "ext = \"nldlt\" if is_delta else \"nlcfr\"\n"
            "s3 = boto3.client(\"s3\", region_name=REGION)\n"
            "key = RUN_PREFIX + f\"/checkpoints/iter_{it:06d}.{ext}\"\n"
            "size = os.path.getsize(path)\n"
            "h = hashlib.sha256()\n"
            "with open(path, \"rb\") as f:\n"
            "    for chunk in iter(lambda: f.read(1 << 24), b\"\"):\n"
            "        h.update(chunk)\n"
            "sha = h.hexdigest()\n"
            "cfgx = TransferConfig(multipart_threshold=64 * 1024 * 1024,\n"
            "                      multipart_chunksize=256 * 1024 * 1024,\n"
            "                      max_concurrency=8, use_threads=True)\n"
            "marker = {\"iters_done\": it, \"bytes\": size, \"sha256\": sha}\n"
            "if is_delta:\n"
            "    marker[\"base_iter\"] = base_iter\n"
            "    marker[\"seq\"] = seq\n"
            "t0 = time.time()\n"
            "last = None\n"
            "for attempt in range(1, 6):\n"
            "    try:\n"
            "        s3.upload_file(path, BUCKET, key, Config=cfgx)\n"
            "        s3.put_object(Bucket=BUCKET, Key=key + \".ok\", Body=json.dumps(marker).encode())\n"
            "        print(f\"CKPT_UPLOADED kind={ext} iters_done={it} bytes={size} sha256={sha[:12]}.. \"\n"
            "              f\"wall={time.time() - t0:.0f}s -> s3://{BUCKET}/{key}\", flush=True)\n"
            "        break\n"
            "    except Exception as e:\n"
            "        last = e\n"
            "        print(f\"CKPT_UPLOAD FAIL attempt={attempt}/5 iter={it}: {e!r}\", flush=True)\n"
            "        time.sleep(min(60, 10 * attempt))\n"
            "else:\n"
            "    print(f\"CKPT_UPLOAD GAVE UP iter={it}: {last!r}\", flush=True)\n"
            "    sys.exit(1)\n"
            "# PRUNE. Full bases: keep gate-cadence forever + newest KEEP_ROLLING non-gate bases (legacy\n"
            "# rule, unchanged). Deltas: a delta is only useful with its base, so prune every delta whose\n"
            "# base_iter is OLDER than the newest surviving base's iter (its chain is superseded). Markers\n"
            "# deleted FIRST so resume never sees a blobless marker.\n"
            "try:\n"
            "    pg = s3.get_paginator(\"list_objects_v2\").paginate(\n"
            "        Bucket=BUCKET, Prefix=RUN_PREFIX + \"/checkpoints/\")\n"
            "    base_iters, delta_meta = [], []  # delta_meta: (iter, base_iter)\n"
            "    for page in pg:\n"
            "        for o in page.get(\"Contents\", []):\n"
            "            k = o[\"Key\"]\n"
            "            mb = re.search(r\"iter_(\\d+)\\.nlcfr\\.ok$\", k)\n"
            "            if mb:\n"
            "                base_iters.append(int(mb.group(1)))\n"
            "                continue\n"
            "            md = re.search(r\"iter_(\\d+)\\.nldlt\\.ok$\", k)\n"
            "            if md:\n"
            "                try:\n"
            "                    meta = json.loads(s3.get_object(Bucket=BUCKET, Key=k)[\"Body\"].read())\n"
            "                    delta_meta.append((int(md.group(1)), int(meta.get(\"base_iter\", -1))))\n"
            "                except Exception:\n"
            "                    pass\n"
            "    rolling = sorted(i for i in base_iters if GATE_EVERY > 0 and i % GATE_EVERY != 0)\n"
            "    prune_bases = set(rolling[:-KEEP_ROLLING] if len(rolling) > KEEP_ROLLING else [])\n"
            "    surviving_bases = sorted(set(base_iters) - prune_bases)\n"
            "    newest_base = surviving_bases[-1] if surviving_bases else -1\n"
            "    for i in sorted(prune_bases):\n"
            "        k = RUN_PREFIX + f\"/checkpoints/iter_{i:06d}.nlcfr\"\n"
            "        s3.delete_object(Bucket=BUCKET, Key=k + \".ok\")  # marker FIRST\n"
            "        s3.delete_object(Bucket=BUCKET, Key=k)\n"
            "        print(f\"CKPT_PRUNED base iter={i}\", flush=True)\n"
            "    for (di, dbase) in delta_meta:\n"
            "        if dbase < newest_base:  # chain rooted at a superseded base\n"
            "            k = RUN_PREFIX + f\"/checkpoints/iter_{di:06d}.nldlt\"\n"
            "            s3.delete_object(Bucket=BUCKET, Key=k + \".ok\")  # marker FIRST\n"
            "            s3.delete_object(Bucket=BUCKET, Key=k)\n"
            "            print(f\"CKPT_PRUNED delta iter={di} base={dbase}\", flush=True)\n"
            "except Exception as e:\n"
            "    print(f\"CKPT_PRUNE skipped: {e!r}\", flush=True)\n"
            "try:\n"
            "    os.remove(path)\n"
            "except OSError:\n"
            "    pass\n"
            "UPEOF"
        )
    else:
        ckpt_uploader_file = ""
    # ---- PRESTAGE + graceful SPOT-WATCH (2026-07-06, fullhist path only; legacy renders keep the
    # ORIGINAL spot_watch verbatim and no prestage => bp2/bp3 bootstraps stay byte-identical). ----
    if fullhist_key or mrr:
        # Background resume-checkpoint prestage: launched IMMEDIATELY after the awscli install
        # (instance-profile creds already work there; boto3/pip is NOT installed yet, so this is
        # stdlib-python + aws cli only). Overlaps the ~114 s+ base download with the
        # rustup/pip/build-cache/EMD prologue. Any failure writes an empty manifest => the driver
        # falls back to its own tuned download.
        prestage_py = (
            "import hashlib, json, os, subprocess, time\n"
            "\n"
            "BUCKET = \"" + bucket + "\"\n"
            "RUN_PREFIX = \"" + run_prefix + "\"\n"
            "REGION = \"" + region + "\"\n"
            "CKPT_DIR = \"/opt/pokerbot/ckpt\"\n"
            "MANIFEST = os.path.join(CKPT_DIR, \"prestage_manifest.json\")\n"
            "DONE = os.path.join(CKPT_DIR, \"prestage.done\")\n"
            "\n"
            "\n"
            "def _aws(*args, timeout=600):\n"
            "    return subprocess.run([\"aws\"] + list(args) + [\"--region\", REGION],\n"
            "                          capture_output=True, text=True, timeout=timeout)\n"
            "\n"
            "\n"
            "def _list_keys():\n"
            "    r = _aws(\"s3api\", \"list-objects-v2\", \"--bucket\", BUCKET,\n"
            "             \"--prefix\", RUN_PREFIX + \"/checkpoints/\",\n"
            "             \"--query\", \"Contents[].Key\", \"--output\", \"json\")\n"
            "    if r.returncode != 0 or not r.stdout.strip():\n"
            "        return []\n"
            "    return json.loads(r.stdout) or []\n"
            "\n"
            "\n"
            "def _read_marker(key):\n"
            "    r = _aws(\"s3\", \"cp\", \"s3://%s/%s\" % (BUCKET, key), \"-\", timeout=120)\n"
            "    if r.returncode != 0:\n"
            "        raise RuntimeError(\"marker fetch failed: \" + key)\n"
            "    return json.loads(r.stdout)\n"
            "\n"
            "\n"
            "def _fetch_verified(key, meta, dst):\n"
            "    # aws cli multipart download (max_concurrent_requests/chunksize set via aws configure\n"
            "    # in the bootstrap) + size/sha256 verify against the .ok marker.\n"
            "    t0 = time.time()\n"
            "    r = _aws(\"s3\", \"cp\", \"s3://%s/%s\" % (BUCKET, key), dst, timeout=3000)\n"
            "    if r.returncode != 0:\n"
            "        raise RuntimeError(\"blob fetch failed: \" + key)\n"
            "    sz = os.path.getsize(dst)\n"
            "    if sz != int(meta[\"bytes\"]):\n"
            "        raise ValueError(\"size %d != %s for %s\" % (sz, meta[\"bytes\"], key))\n"
            "    h = hashlib.sha256()\n"
            "    with open(dst, \"rb\") as f:\n"
            "        for chunk in iter(lambda: f.read(1 << 24), b\"\"):\n"
            "            h.update(chunk)\n"
            "    if h.hexdigest() != meta[\"sha256\"]:\n"
            "        raise ValueError(\"sha mismatch for \" + key)\n"
            "    print(\"PRESTAGE fetched %s bytes=%d wall=%.1fs\" % (key, sz, time.time() - t0),\n"
            "          flush=True)\n"
            "\n"
            "\n"
            "def main():\n"
            "    man = {\"base\": None, \"deltas\": []}\n"
            "    try:\n"
            "        keys = _list_keys()\n"
            "        os.makedirs(CKPT_DIR, exist_ok=True)\n"
            "        base = None\n"
            "        for mk in reversed(sorted(k for k in keys if k.endswith(\".nlcfr.ok\"))):\n"
            "            try:\n"
            "                meta = _read_marker(mk)\n"
            "                dst = os.path.join(CKPT_DIR, \"prestage_base.nlcfr\")\n"
            "                _fetch_verified(mk[:-3], meta, dst)\n"
            "                base = {\"marker\": mk, \"path\": dst,\n"
            "                        \"iters_done\": int(meta[\"iters_done\"]),\n"
            "                        \"bytes\": int(meta[\"bytes\"]), \"sha256\": meta[\"sha256\"]}\n"
            "                break\n"
            "            except Exception as e:\n"
            "                print(\"PRESTAGE base candidate %s rejected: %r\" % (mk, e), flush=True)\n"
            "        man[\"base\"] = base\n"
            "        if base is not None:\n"
            "            by_seq = {}\n"
            "            for mk in sorted(k for k in keys if k.endswith(\".nldlt.ok\")):\n"
            "                try:\n"
            "                    meta = _read_marker(mk)\n"
            "                    if int(meta.get(\"base_iter\", -1)) == base[\"iters_done\"]:\n"
            "                        by_seq[int(meta[\"seq\"])] = (mk, meta)\n"
            "                except Exception:\n"
            "                    pass\n"
            "            seq = 1\n"
            "            while seq in by_seq:  # contiguous verified chain only (mirrors driver resume)\n"
            "                (mk, meta) = by_seq[seq]\n"
            "                try:\n"
            "                    dst = os.path.join(CKPT_DIR, \"prestage_delta_%04d.nldlt\" % seq)\n"
            "                    _fetch_verified(mk[:-3], meta, dst)\n"
            "                    man[\"deltas\"].append({\"marker\": mk, \"path\": dst, \"seq\": seq,\n"
            "                                          \"base_iter\": base[\"iters_done\"],\n"
            "                                          \"bytes\": int(meta[\"bytes\"]),\n"
            "                                          \"sha256\": meta[\"sha256\"]})\n"
            "                except Exception as e:\n"
            "                    print(\"PRESTAGE delta seq=%d rejected: %r -- chain stops\" % (seq, e),\n"
            "                          flush=True)\n"
            "                    break\n"
            "                seq += 1\n"
            "    except Exception as e:\n"
            "        print(\"PRESTAGE failed: %r -- driver will download itself\" % (e,), flush=True)\n"
            "        man = {\"base\": None, \"deltas\": []}\n"
            "    with open(MANIFEST + \".tmp\", \"w\") as f:\n"
            "        json.dump(man, f)\n"
            "    os.replace(MANIFEST + \".tmp\", MANIFEST)\n"
            "    with open(DONE, \"w\") as f:\n"
            "        f.write(\"done\\n\")\n"
            "    print(\"PRESTAGE done base=%s deltas=%d\"\n"
            "          % ((man[\"base\"] or {}).get(\"marker\"), len(man[\"deltas\"])), flush=True)\n"
            "\n"
            "\n"
            "main()\n"
        )
        prestage_block = (
            "\n# PRESTAGE the resume checkpoint (2026-07-06): overlap the ~114 s+ base download with the\n"
            "# rustup/pip/build-cache/EMD prologue below (stdlib python + aws cli; boto3 not installed\n"
            "# yet). The driver waits bounded on prestage.done and verifies the manifest against the\n"
            "# CURRENT markers before consuming; any failure falls back to its own download.\n"
            "mkdir -p /opt/pokerbot/ckpt\n"
            "aws configure set default.s3.max_concurrent_requests 64\n"
            "aws configure set default.s3.multipart_chunksize 64MB\n"
            "cat > /opt/pokerbot/prestage.py <<'PREOF'\n"
            + prestage_py +
            "PREOF\n"
            "touch /opt/pokerbot/ckpt/prestage.started\n"
            "nohup python3 /opt/pokerbot/prestage.py > /var/log/prestage.log 2>&1 &\n"
            "echo \"PRESTAGE launched (background resume-checkpoint download)\""
        )
        # Graceful spot watch: flag-first (driver flushes within ~5 s via the chunked loop), PC-side
        # babysitter death signal (reclaim_notice.json under the run prefix -- FIXED key contract
        # with the babysitter agent, do NOT rename), pkill only as a 100 s fallback; plus the
        # earlier-warning rebalance recommendation (flag only -- one extra detached delta, no kill).
        spot_watch_block = f"""# Background SPOT-INTERRUPT handler (graceful-flush rework 2026-07-06): poll IMDSv2 for BOTH the
# 2-min reclaim notice and the earlier rebalance recommendation. On reclaim: (1) touch
# /tmp/spot_reclaim -- the chunked train loop notices within ~5 s and banks an EMERGENCY delta;
# (2) upload reclaim_notice.json (the PC babysitter's instant death signal; FIXED key contract);
# (3) pkill -TERM only after a 100 s flush grace (the driver's SIGTERM handler also flushes).
# On rebalance recommendation: touch /tmp/spot_rebalance only (no kill, no notice json).
cat > /tmp/spot_watch.sh <<'SWEOF'
TOK=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
REBAL=0
while true; do
  C=$(curl -s -o /dev/null -w "%{{http_code}}" -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/spot/instance-action)
  if [ "$C" = "200" ]; then
    echo "SPOT_RECLAIM_NOTICE"
    touch /tmp/spot_reclaim
    IID=$(curl -s -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/instance-id)
    printf '{{"instance_id": "%s", "iso_time": "%s", "action": "reclaim"}}' "$IID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > /tmp/reclaim_notice.json
    timeout 25 aws s3 cp /tmp/reclaim_notice.json s3://{bucket}/{run_prefix}/reclaim_notice.json --region {region} || true
    sleep 100
    pkill -TERM -f drive_native_cfr.py || true
    break
  fi
  if [ "$REBAL" = "0" ]; then
    R=$(curl -s -o /dev/null -w "%{{http_code}}" -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/events/recommendations/rebalance)
    if [ "$R" = "200" ]; then echo "SPOT_REBALANCE_RECOMMENDATION"; REBAL=1; touch /tmp/spot_rebalance; fi
  fi
  sleep 5
done
SWEOF
nohup bash /tmp/spot_watch.sh >/var/log/spot_watch.log 2>&1 &"""
    else:
        prestage_block = ""
        # ORIGINAL spot watch VERBATIM (bp2/bp3 byte-identity).
        spot_watch_block = f"""# Background SPOT-INTERRUPT handler: poll IMDSv2 (token required) for the reclaim notice; on notice send
# SIGTERM so the driver's last checkpoint is already on S3 (it checkpoints every {checkpoint_every} iters).
cat > /tmp/spot_watch.sh <<'SWEOF'
TOK=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
while true; do
  C=$(curl -s -o /dev/null -w "%{{http_code}}" -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/spot/instance-action)
  if [ "$C" = "200" ]; then echo "SPOT_RECLAIM_NOTICE"; pkill -TERM -f drive_native_cfr.py; break; fi
  sleep 5
done
SWEOF
nohup bash /tmp/spot_watch.sh >/var/log/spot_watch.log 2>&1 &"""
    # COMPUTE PROFILER (--profile-run, 2026-07-06): mpstat per-core utilization + pidstat on the
    # driver + ONE steady-state `perf record` sample (~8 min in, past resume + first checkpoint),
    # pushed to s3 {run_prefix}/logs/profile/ every 300 s so a mid-run reclaim keeps the evidence.
    # Rendered EMPTY when off -> every existing render stays byte-identical (appended, no newline).
    # Everything is non-fatal: a missing linux-tools for the AWS kernel just skips the perf sample.
    if profile_run:
        profile_block = f"""

# COMPUTE PROFILER (--profile-run): mpstat/pidstat sampling + one steady-state perf capture.
apt-get install -y -qq sysstat linux-tools-common "linux-tools-$(uname -r)" || echo "PROFILE: linux-tools install partial (non-fatal)"
cat > /tmp/profile_watch.sh <<'PWEOF'
PDIR=/var/log/profile; mkdir -p $PDIR
PID=""
for i in $(seq 1 120); do PID=$(pgrep -f drive_native_cfr.py | head -1); [ -n "$PID" ] && break; sleep 5; done
if [ -z "$PID" ]; then echo "PROFILE: driver never appeared"; exit 0; fi
echo "PROFILE: driver pid=$PID"
nohup mpstat -P ALL 30 > $PDIR/mpstat.log 2>&1 &
nohup pidstat -u -r 30 -p $PID > $PDIR/pidstat.log 2>&1 &
( sleep 480
  if command -v perf >/dev/null 2>&1 && kill -0 $PID 2>/dev/null; then
    perf record -F 99 -g -p $PID -o $PDIR/perf.data -- sleep 75 2> $PDIR/perf.log || true
    perf report --stdio -i $PDIR/perf.data --no-children 2>/dev/null | head -400 > $PDIR/perf_flat.txt || true
    perf report --stdio -i $PDIR/perf.data 2>/dev/null | head -400 > $PDIR/perf_children.txt || true
    rm -f $PDIR/perf.data
  else
    echo "perf unavailable or driver gone" > $PDIR/perf_flat.txt
  fi ) &
while true; do
  for f in mpstat.log pidstat.log perf_flat.txt perf_children.txt perf.log; do
    if [ -s "$PDIR/$f" ]; then timeout 60 aws s3 cp "$PDIR/$f" "s3://{bucket}/{run_prefix}/logs/profile/$f" --region {region} >/dev/null 2>&1 || true; fi
  done
  kill -0 $PID 2>/dev/null || break
  sleep 300
done
for f in mpstat.log pidstat.log perf_flat.txt perf_children.txt perf.log; do
  if [ -s "$PDIR/$f" ]; then timeout 60 aws s3 cp "$PDIR/$f" "s3://{bucket}/{run_prefix}/logs/profile/$f" --region {region} >/dev/null 2>&1 || true; fi
done
echo "PROFILE: done"
PWEOF
nohup bash /tmp/profile_watch.sh >/var/log/profile_watch.log 2>&1 &"""
    else:
        profile_block = ""
    # No output redirect (cloud-init captures stdout to /var/log/cloud-init-output.log, visible via
    # get-console-output) — same lesson the SD-CFR launcher documents.
    return f"""#!/bin/bash
set -uxo pipefail
export HOME="${{HOME:-/root}}"   # may run detached from the stub's shell; same set -u guard
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION={region}
mkdir -p /opt/pokerbot

apt-get update -qq
apt-get install -y -qq python3-pip python3-dev gzip curl unzip build-essential
# AWS CLI v2 (NOT apt awscli v1): self-contained (bundled botocore under /usr/local/aws-cli), so the driver's
# `pip install boto3` below can bump the SYSTEM botocore WITHOUT breaking the CLI. The apt awscli v1 broke
# exactly this way: pip boto3 -> newer botocore -> awscli v1 `KeyError: 'opsworkscm'` on every call.
curl -fsSL "https://awscli.amazonaws.com/{awscli_zip}" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update
hash -r
aws --version   # MUST be aws-cli/2.x; if this dies, STEP-0 still bounds the box{prestage_block}
# Rust toolchain (rustup, stable) for the native build.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"
rustc --version

mkdir -p /opt/pokerbot
aws s3 cp s3://{bucket}/{code_key} /tmp/code.tar.gz --region {region}
tar -xzf /tmp/code.tar.gz -C /opt/pokerbot
cd /opt/pokerbot

# boto3 for S3 checkpoint up/download from the driver.
python3 -m pip install --disable-pip-version-check --no-input boto3 numpy
# Build the native extension (.so) exposing NlheCFRTrainer. PyO3 abi3 -> one .so for the box's python.
export PYO3_PYTHON=$(python3 -c 'import sys;print(sys.executable)')
# target-cpu=x86-64-v3 (AVX2 + FMA, NO AVX512): autovectorizes the CFR+ [f64;11]/[bool;11] hot-loop
# kernels (regret_match, EV dot-product, strat_sum AXPY, CFR+ floor, side-pot award) to 4-wide and lets
# the hand-written SIMD emit FMA instead of SSE2-baseline scalar. It is NOT -ffast-math (no float
# reassociation), so the exact-int byte-identity goldens stay green. We deliberately do NOT use
# target-cpu=native: on an AVX512 box `native` enables AVX512, whose license-based downclock is a NET
# REGRESSION for this pointer-chasing/hashmap-bound hot path (~-30% it/s measured). v3 = AVX2+FMA with
# no AVX512 => no downclock, measured small net gain. The .so is built AND run on this same box and
# never ships to a different-CPU host, so AVX2/FMA is always present (no SIGILL). Set via RUSTFLAGS on
# the build line so it applies even though the tarball include-list omits tools/battery_native/.cargo/.
#
# BUILD CACHE (spot-resilience, 2026-07-05): the lto=fat/codegen-units=1 release build takes ~65 min,
# which the bpv4_h200_arm2 first attempt paid FIVE times (spot reclaims every ~69 min => zero training
# output). After a successful build the .so is uploaded to the shared build_cache/ S3 prefix keyed by
# sha256(DECOMPRESSED code tar | RUSTFLAGS | features | rustc -V | uname -m) — the gzip envelope is
# excluded because Python's tarfile stamps a fresh gzip mtime on every upload (same sources would
# otherwise never cache-hit). x86-64-v3 (AVX2+FMA, no AVX512) runs on every type in the spotpick
# 128-vCPU chain (c6a/c6i/c6in/c6id/m6i/c7a), so a fallback-type relaunch still hits the cache.
# On boot: try the cache first (sha256-verified; corrupt/missing => source build). Relaunch-to-training
# drops from ~65 min to ~2-5 min on a hit.
BUILD_RUSTFLAGS="-C target-cpu=x86-64-v3"
CODE_SHA=$(gzip -dc /tmp/code.tar.gz | sha256sum | awk '{{print $1}}')
BUILD_KEY=$(printf '%s|%s|%s|%s|%s' "$CODE_SHA" "$BUILD_RUSTFLAGS" "python:lib" "$(rustc -V)" "$(uname -m)" | sha256sum | awk '{{print $1}}')
CACHE_URI="s3://{bucket}/{prefix}/build_cache/$BUILD_KEY"
echo "BUILD_CACHE key=$BUILD_KEY (code_sha=$CODE_SHA rustc=$(rustc -V) arch=$(uname -m))"
CACHE_HIT=0
# NO_CACHE_FETCH=1 forces the source build (provenance incident 2026-07-06: an unidentified
# account-side pipeline reacts to code-tarball uploads and plants an untrusted .so under the
# correct cache key within ~4 min — a fetch can then never be trusted after a tarball upload).
# The post-build populate still runs, OVERWRITING any planted entry with the trusted artifact.
if [ "{no_cache_fetch}" != "1" ] \
   && aws s3 cp "$CACHE_URI/battery_native.so" /opt/pokerbot/battery_native.so --region {region} 2>/dev/null \
   && aws s3 cp "$CACHE_URI/battery_native.so.sha256" /tmp/bn_so.sha --region {region} 2>/dev/null; then
  WANT=$(awk '{{print $1}}' /tmp/bn_so.sha)
  GOT=$(sha256sum /opt/pokerbot/battery_native.so | awk '{{print $1}}')
  if [ -n "$WANT" ] && [ "$WANT" = "$GOT" ]; then
    CACHE_HIT=1
    echo "BUILD_CACHE HIT sha256=$GOT -- skipping the ~65 min source build"
  else
    echo "BUILD_CACHE CORRUPT (want=$WANT got=$GOT) -- falling back to source build"
    rm -f /opt/pokerbot/battery_native.so
  fi
else
  echo "BUILD_CACHE MISS -- source build"
fi
if [ "$CACHE_HIT" != "1" ]; then
  RUSTFLAGS="$BUILD_RUSTFLAGS" cargo build --release --features python --lib --manifest-path tools/battery_native/Cargo.toml
  # The cdylib is target/release/libbattery_native.so; import it as `battery_native`.
  SO=$(ls tools/battery_native/target/release/libbattery_native.so 2>/dev/null || ls tools/battery_native/target/release/battery_native.so 2>/dev/null)
  cp "$SO" /opt/pokerbot/battery_native.so
  # Populate the cache: .so FIRST, sha marker SECOND (a marker never exists without its blob; a torn
  # blob fails the sha verify on the next boot). Non-fatal on failure (the box's own build proceeds).
  sha256sum /opt/pokerbot/battery_native.so | awk '{{print $1}}' > /tmp/bn_so.sha
  ( aws s3 cp /opt/pokerbot/battery_native.so "$CACHE_URI/battery_native.so" --region {region} \
    && aws s3 cp /tmp/bn_so.sha "$CACHE_URI/battery_native.so.sha256" --region {region} \
    && echo "BUILD_CACHE UPLOADED -> $CACHE_URI" ) || echo "BUILD_CACHE upload failed (non-fatal)"
fi
ls -la /opt/pokerbot/battery_native.so

# RAYON FANOUT: pin to PHYSICAL cores, NOT logical (SMT). The cpu128 probe (2026-06-28,
# artifacts/cpu128_probe_verdict_20260628.md) measured the trainer is memory-latency-bound — cores sit ~69%
# idle inside the parallel traverse — so the 2nd hyperthread per core CONTENDS for shared L1/L2/load-store
# ports rather than adding compute. At identical fanout, 64 threads (1/phys core) beat 128 logical by
# +12.6% deals/s / +18.5% nodes/s at LOWER CPU. PHYS_CORES = unique (socket,core) pairs (correct on 2-socket
# boxes where core ids repeat per socket). Override by exporting RAYON_NUM_THREADS before launch.
PHYS_CORES=$(lscpu -p=SOCKET,CORE | grep -v '^#' | sort -u | wc -l)
if [ -z "${{RAYON_NUM_THREADS:-}}" ]; then export RAYON_NUM_THREADS=$PHYS_CORES; fi
echo "RAYON_NUM_THREADS=$RAYON_NUM_THREADS (physical cores=$PHYS_CORES, nproc=$(nproc); SMT-off per cpu128 probe verdict)"
export PYTHONPATH=/opt/pokerbot RUST_BACKTRACE=1
# Blueprint-v3 preflop HISTORY key. When 1, fingerprint.rs (env-gated, read once per process)
# appends b"H"+[n_limps,n_raises,opener_pos+1] to every PREFLOP infoset key, so strategically
# distinct preflop histories stop aliasing (artifacts/preflop_mispricing_diagnosis.md). The
# checkpoint MUST then be served with POKERBOT_PREFLOP_HIST_KEY=1 (ship_blueprint --preflop-hist-key
# stamps + guards it). "0" is byte-identical to the shipped key space.
export POKERBOT_PREFLOP_HIST_KEY={hist_flag}
echo "PREFLOP_HIST_KEY=$POKERBOT_PREFLOP_HIST_KEY"
{fullhist_block}{v3_env_block}{row_budget_env}
# S2: verify the shipped EMD frozen blob matches its known sha256 BEFORE training. A same-K CORRUPTED blob
# (truncated S3 round-trip) -> wrong potential-aware buckets -> a broken blueprint that the convergence-stop
# (abstraction-agnostic) cannot catch and only the final ranking gate would; the ctor's num_buckets==K check
# does NOT detect content corruption. Empty EMD_SHA (made-hand path) -> skipped. A mismatch terminates the box.
EMD_SHA="{emd_sha}"
EMD_REL="{emd_frozen_rel}"
if [ -n "$EMD_SHA" ] && [ -n "$EMD_REL" ]; then
  ACT=$(sha256sum "/opt/pokerbot/$EMD_REL" | awk '{{print $1}}')
  if [ "$ACT" != "$EMD_SHA" ]; then
    echo "FATAL: EMD blob sha mismatch expect=$EMD_SHA actual=$ACT -- aborting + terminating"; sync; /sbin/shutdown -h now; exit 1
  fi
  echo "EMD blob sha256 OK: $ACT"
fi

# Blueprint-v2: verify the shipped river-OCHS table sha256 (same corruption rationale as the EMD blob;
# a corrupted table re-maps every river bucket -> a silently broken blueprint). Empty => skipped.
ROCHS_SHA="{river_ochs_sha}"
ROCHS_REL="{river_ochs_rel}"
if [ -n "$ROCHS_SHA" ] && [ -n "$ROCHS_REL" ]; then
  ACT=$(sha256sum "/opt/pokerbot/$ROCHS_REL" | awk '{{print $1}}')
  if [ "$ACT" != "$ROCHS_SHA" ]; then
    echo "FATAL: river-OCHS table sha mismatch expect=$ROCHS_SHA actual=$ACT -- aborting + terminating"; sync; /sbin/shutdown -h now; exit 1
  fi
  echo "river-OCHS table sha256 OK: $ACT"
fi
{ckpt_uploader_file}
cat > /opt/pokerbot/drive_native_cfr.py <<'PYEOF'
import os, sys, time, traceback
import boto3{ckpt_upload_imports}
import battery_native as bn

BUCKET = "{bucket}"
RUN_PREFIX = "{run_prefix}"
REGION = "{region}"
ITERS = {iters}
N_DEALS = {n_deals}
NUM_BUCKETS = {num_buckets}
CKPT_EVERY = {checkpoint_every}{row_budget_const}
DEPTH_EDGES = [float(x) for x in "{depth_csv}".split(",") if x]
# Optional shared frozen EMD (hole,board)->bucket lookup (relative to /opt/pokerbot). When set, the
# trainer keys postflop flop/turn buckets from this table (river/preflop stay made-hand) — the SAME blob
# the serve encoder reads, so train==serve is bit-identical. Empty => the deployed made-hand path.
EMD_FROZEN_REL = "{emd_frozen_rel}"
EMD_FROZEN_PATH = ("/opt/pokerbot/" + EMD_FROZEN_REL) if EMD_FROZEN_REL else None
# Blueprint-v2 river OCHS table: when set, the trainer keys the RIVER with the OCHS nearest-centroid
# bucket (flop/turn stay on the EMD frozen blob; preflop unchanged). The blueprint MUST then be served
# with POKERBOT_RIVER_OCHS pointing at this same table (stamp-guarded). Empty => made-hand mirror river.
RIVER_OCHS_REL = "{river_ochs_rel}"
RIVER_OCHS_PATH = ("/opt/pokerbot/" + RIVER_OCHS_REL) if RIVER_OCHS_REL else None
# Throughput/convergence levers (gated default-OFF; both verified). work_steal: atomic deal-counter +
# rayon::broadcast over per-thread tables -> closes the deep-deal load imbalance (+33% deals/s @64c,
# measured). dcfr: signed-regret Discounted-CFR + active Regret-Based Pruning (~2x fewer iters to
# convergence on Leduc exact exploitability).
WORK_STEAL = {work_steal}
DCFR = {dcfr}
DCFR_PRUNE_THRESHOLD = {dcfr_prune_threshold}
DCFR_EXPLORE_FRAC = {dcfr_explore_frac}

s3 = boto3.client("s3", region_name=REGION)
LATEST_KEY = RUN_PREFIX + "/checkpoints/latest.nlcfr"

{ckpt_upload_fn}{v3_driver_fns}

{download_latest_fn}

def main():
    # full ctor: (num_buckets, depth_edges, n_deals, n_seats, prune_threshold, fast_merge, emd_frozen_path,
    #             batch_deals, hogwild, hogwild_capacity, work_steal, dcfr, dcfr_prune_threshold, dcfr_explore_frac)
    if EMD_FROZEN_PATH:
        t = bn.NlheCFRTrainer(NUM_BUCKETS, DEPTH_EDGES, N_DEALS, 6, float("-inf"), False, EMD_FROZEN_PATH,
                              0, False, 1 << 21, WORK_STEAL, DCFR, DCFR_PRUNE_THRESHOLD, DCFR_EXPLORE_FRAC,
                              river_ochs_path=RIVER_OCHS_PATH)
        print(f"EMD frozen lookup ACTIVE: {{EMD_FROZEN_PATH}} stamp={{t.emd_stamp()}}", flush=True)
    else:
        t = bn.NlheCFRTrainer(NUM_BUCKETS, DEPTH_EDGES, N_DEALS, 6, float("-inf"), False, None,
                              0, False, 1 << 21, WORK_STEAL, DCFR, DCFR_PRUNE_THRESHOLD, DCFR_EXPLORE_FRAC,
                              river_ochs_path=RIVER_OCHS_PATH)
    if RIVER_OCHS_PATH:
        print(f"river-OCHS ACTIVE: {{RIVER_OCHS_PATH}} stamp={{t.river_ochs_stamp()}}", flush=True)
    print(f"LEVERS work_steal={{WORK_STEAL}} dcfr={{DCFR}} "
          f"(prune={{DCFR_PRUNE_THRESHOLD}} explore={{DCFR_EXPLORE_FRAC}}) hogwild={{t.is_hogwild()}}", flush=True){v3_banner}
{resume_block}
    start = t.iters_done()
    print(f"TRAIN target={{ITERS}} from={{start}} n_deals={{N_DEALS}} num_buckets={{NUM_BUCKETS}} "
          f"depth_edges={{DEPTH_EDGES}}", flush=True)
    t0 = time.time()
    while t.iters_done() < ITERS:{row_budget_pre}
        nxt = {next_stop_expr}
        step = nxt - t.iters_done()
        {train_stmt}{reclaim_break}{row_budget_block}
        {ckpt_stmt}{v3_ckpt_call}
        el = time.time() - t0
        done = t.iters_done() - start
        rate = done / el if el > 0 else 0.0
        print(f"PROGRESS iters_done={{t.iters_done()}}/{{ITERS}} infosets={{t.n_infosets()}} "
              f"elapsed={{el:.0f}}s rate={{rate:.3f}} it/s", flush=True){ckpt_wait_block}
    print(f"DONE iters_done={{t.iters_done()}} infosets={{t.n_infosets()}}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
PYEOF

{spot_watch_block}{profile_block}

python3 /opt/pokerbot/drive_native_cfr.py 2>&1 | tee /var/log/native-cfr-train.log
echo "train exit ${{PIPESTATUS[0]}}"

# Upload logs via boto3.
python3 - <<'LOGEOF'
import boto3
s3 = boto3.client("s3", region_name="{region}")
for src, key in [("/var/log/cloud-init-output.log", "{run_prefix}/logs/cloud-init-output.log"),
                 ("/var/log/native-cfr-train.log", "{run_prefix}/logs/train.log")]:
    try:
        s3.upload_file(src, "{bucket}", key); print("uploaded", key)
    except Exception as e:
        print("log upload fail", src, e)
LOGEOF
{final}
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instance-type", default="c6i.32xlarge",
                   help="c6i.32xlarge=full (128 vCPU); c6i.4xlarge=cheap validate")
    p.add_argument("--arch", default="x86_64", choices=["x86_64", "arm64"])
    p.add_argument("--iters", type=int, default=5000, help="total CFR+ iterations (target)")
    p.add_argument("--n-deals", type=int, default=200, help="deals fanned out per iteration (rayon)")
    p.add_argument("--num-buckets", type=int, default=200,
                   help="postflop card-abstraction buckets (MUST equal serve's num_buckets)")
    p.add_argument("--depth-edges", default="",
                   help="comma-sep depth-bucket edges for a depth-aware preflop key (e.g. '50,90,140'); "
                        "empty = non-depth blueprint")
    p.add_argument("--checkpoint-every", "--ckpt-every", type=int, default=1000, dest="checkpoint_every",
                   help="checkpoint every N iters (preemption-bounded loss). Default 1000 (arm-2 "
                        "spot-resilience ops, 2026-07-05; was 500). On the fullhist/M-RR path these are "
                        "ROLLING resume checkpoints: written to local disk then uploaded by a DETACHED "
                        "process (training continues), marker-completed, pruned to the newest 3 + all "
                        "--gate-every checkpoints.")
    p.add_argument("--gate-every", type=int, default=0,
                   help="pre-registered gate/measurement cadence in iters (fullhist/M-RR path only): "
                        "gate-cadence checkpoints are KEPT forever (the amendment ladder, e.g. 2500 for "
                        "bpv4_h200_arm2 => 2.5k/5k/.../20k) and the v3 extras (NLSNAP01 snapshot, "
                        "COVERAGE, REGSTATS) run ONLY at gate iters, preserving the registered "
                        "measurement/snapshot-averaging density. 0 = every checkpoint is a gate "
                        "(legacy behavior).")
    p.add_argument("--ckpt-volume-gb", type=int, default=None,
                   help="root gp3 EBS volume size in GB (with 700 MiB/s provisioned throughput) for the "
                        "local checkpoint staging dir the detached uploader reads from. Default: 400 when "
                        "--fullhist-key/--mrr (blobs reach ~58 GB; uploader deletes local files after a "
                        "durable upload), else AMI default (8 GB).")
    p.add_argument("--max-hours", type=float, default=6.0,
                   help="ON-BOX absolute dead-man cost cap (hours): the box self-terminates after this no "
                        "matter what (logind floor + systemd wall-guard), independent of the babysitter/PC. "
                        "Set comfortably above the expected run (~2h) but below the babysitter's 12h cap.")
    p.add_argument("--run-subdir", default=None, help="S3 subdir for this run (default: run_<ts>)")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--instance-profile", default=DEFAULT_PROFILE)
    p.add_argument("--on-demand", action="store_true", help="use on-demand instead of spot")
    p.add_argument("--spotpick", action="store_true",
                   help="on a spot-capacity error, retry down a capacity-diverse fallback chain "
                        "(spotpick.chain at the same vCPU count) instead of hard-failing. Default OFF = "
                        "single --instance-type only (unchanged behavior).")
    p.add_argument("--spotpick-n", type=int, default=5,
                   help="max instance types in the --spotpick fallback chain (incl. the primary)")
    p.add_argument("--keep-alive", action="store_true",
                   help="don't shut down on finish (keep box for get-console-output inspection)")
    p.add_argument("--reuse-code", action="store_true",
                   help="skip the code upload; REUSE the existing S3 code tarball. The babysitter passes "
                        "this on RELAUNCHES so a locally-changing working tree (e.g. a concurrent crate "
                        "optimization) never re-uploads mid-edit and breaks a resumed box.")
    p.add_argument("--no-cache-fetch", action="store_true",
                   help="bootstrap SKIPS the build-cache fetch and always builds from source (still "
                        "populates/overwrites the cache afterwards). Use when the cache entry cannot be "
                        "trusted — e.g. the 2026-07-06 incident where an account-side pipeline plants an "
                        "untrusted .so under the correct key minutes after every code-tarball upload.")
    p.add_argument("--crate-src", default="tools/battery_native",
                   help="path to the crate source to upload (always as tools/battery_native). Point at a "
                        "FROZEN COPY to isolate a running job from concurrent edits to the live crate; "
                        "falls back to tools/battery_native if the path is missing.")
    p.add_argument("--emd-frozen", default="",
                   help="path (repo-relative) to a SHARED frozen EMD (hole,board)->bucket lookup "
                        "(precompute_emd_frozen_lookup.py output). When set, the trainer keys postflop "
                        "flop/turn buckets from it (river/preflop stay made-hand) — the SAME blob the serve "
                        "encoder reads (POKERBOT_POSTFLOP_EMD_FROZEN), so train==serve is bit-identical. "
                        "Empty = the deployed made-hand path. When set, --num-buckets MUST equal the blob's K "
                        "(the ctor enforces num_buckets==lookup.K; e.g. 160 for the K160 blob); the EMD "
                        "blueprint is a self-consistent K (serve must use the same K + EMD). Made-hand path "
                        "(empty) uses --num-buckets=200 to match serve's POSTFLOP_NUM_BUCKETS.")
    p.add_argument("--river-ochs", default="",
                   help="path (repo-relative) to the blueprint-v2 river-OCHS table (.rochs, from "
                        "scripts/blueprint2/fit_river_ochs.py). When set, the trainer keys the RIVER with "
                        "the OCHS nearest-centroid bucket (flop/turn stay on --emd-frozen; preflop "
                        "unchanged) and the checkpoint MUST be served with POKERBOT_RIVER_OCHS pointing at "
                        "the SAME table (stamp-guarded; a skew HITS wrong-semantics river rows). "
                        "Empty = the deployed made-hand mirror river.")
    p.add_argument("--preflop-hist-key", action="store_true",
                   help="blueprint-v3 (bp3_prefhist_1): export POKERBOT_PREFLOP_HIST_KEY=1 on the box so "
                        "the trainer keys PREFLOP infosets with the extra b'H'+[n_limps,n_raises,"
                        "opener_pos] bytes (fingerprint.rs, env-gated) — the fix for the preflop "
                        "betting-state ALIASING root cause (artifacts/preflop_mispricing_diagnosis.md). "
                        "The checkpoint MUST then be served with the same env: ship it via "
                        "ship_blueprint.py --preflop-hist-key (stamp + two-sided serve guard).")
    p.add_argument("--fullhist-key", action="store_true",
                   help="blueprint-v4 Stage-0 (bpv4_h200): export POKERBOT_FULLHIST_KEY=1 on the box so the "
                        "trainer keys EVERY street's infosets with the full action-history A-segment "
                        "(fingerprint.rs, env-gated) — the Pluribus-grade exact-history key "
                        "(artifacts/blueprint_pluribus_grade_design_v2.md). Pair with --mrr (the M-RR "
                        "action-thinning that makes fullhist convergent). Default off => byte-identical "
                        "bootstrap (bp2/bp3 unaffected). Serve/eval MUST use the same env.")
    p.add_argument("--mrr", action="store_true",
                   help="blueprint-v4 Stage-0: export POKERBOT_MRR=1 on the box so trainer::abstracted_legal "
                        "applies M-RR (multiplicative regret / action-thinning), collapsing the fullhist tree "
                        "from ~10^31 to a Pluribus-class ~413M rows (bpv4-mrr-relief-valve). Default off => "
                        "byte-identical bootstrap. Any in-model probe/eval MUST also set POKERBOT_MRR=1 "
                        "(train/probe abstraction skew otherwise biases the verdict).")
    p.add_argument("--work-steal", action="store_true",
                   help="atomic deal-counter + rayon::broadcast scheduler (closes the deep-deal load "
                        "imbalance; +33%% deals/s @64c measured). Statistically-equivalent, default OFF.")
    p.add_argument("--dcfr", action="store_true",
                   help="signed-regret Discounted-CFR + active Regret-Based Pruning (~2x fewer iters to "
                        "convergence on Leduc). 6-max convergence being certified; default OFF.")
    p.add_argument("--dcfr-prune-threshold", type=float, default=-300000.0,
                   help="RBP floor (bb regret units) used when --dcfr is on")
    p.add_argument("--dcfr-explore-frac", type=float, default=0.05,
                   help="fraction of iters that explore all actions (no pruning) under --dcfr; Pluribus ~0.05")
    # ---- blueprint-v4 v3 trainer schedule (artifacts/blueprint_v4_amendment_v3.md) ----
    # All default-off => byte-identical bootstrap (sha-compared). --v3-lcfr enables the Pluribus
    # Linear-CFR mode (signed regrets + iteration-denominated discount + unweighted averaging);
    # --v3-prune-c enables hard negative-regret pruning (requires --v3-lcfr); --v3-snapshot-avg
    # drops the postflop strat_sum (round-1 average stays in memory) and uploads NLSNAP01
    # current-strategy snapshots at checkpoint cadence for the offline snap_avg averager.
    p.add_argument("--v3-lcfr", action="store_true",
                   help="v3: Pluribus LCFR trainer mode (POKERBOT_LCFR=1 on the box). Mutually "
                        "exclusive with --dcfr (DCFR takes precedence if both are set).")
    p.add_argument("--v3-signed-regrets", action="store_true",
                   help="arm-2 (artifacts/blueprint_v4_arm2_amendment.md): POKERBOT_SIGNED_REGRETS=1 "
                        "on the box — signed regrets + deep regret floor + hard pruning WITHOUT the "
                        "LCFR discount; CFR+ w=t linear averaging kept. Mutually exclusive with "
                        "--v3-lcfr and --dcfr.")
    p.add_argument("--v3-lcfr-d", type=int, default=250,
                   help="v3: LCFR discount interval D in iterations (Pluribus: 10 wall-minutes)")
    p.add_argument("--v3-lcfr-w", type=int, default=10000,
                   help="v3: LCFR discount window W in iterations (Pluribus: 400 wall-minutes)")
    p.add_argument("--v3-prune-c", type=float, default=None,
                   help="v3: hard negative-regret prune threshold C in bb (calibrated default in the "
                        "amendment doc; omit = pruning off)")
    p.add_argument("--v3-prune-start", type=int, default=None,
                   help="v3: first prune iteration (default W/2 inside the trainer)")
    p.add_argument("--v3-prune-explore", type=float, default=0.05,
                   help="v3: fraction of iterations that explore fully (no pruning); Pluribus 5%%")
    p.add_argument("--v3-regret-floor", type=float, default=None,
                   help="v3: regret floor in bb (default C*31/30 inside the trainer)")
    p.add_argument("--v3-snapshot-avg", action="store_true",
                   help="v3: no-postflop-strat_sum + NLSNAP01 snapshot upload at checkpoint cadence "
                        "(POKERBOT_SNAPSHOT_AVG=1); final blueprint = snap_avg.exe offline average")
    p.add_argument("--v3-snap-start", type=int, default=None,
                   help="v3: first snapshot iteration (default 8%% of --iters, Pluribus's 800/11520 "
                        "discard-early rule)")
    p.add_argument("--v3-cov-hands", type=int, default=2048,
                   help="v3: on-policy coverage probe hands at each checkpoint (0 disables); the "
                        "COVERAGE log line feeds the pre-registered coverage gates")
    p.add_argument("--delta-ckpt", action="store_true", default=False,
                   help="R2 delta checkpoints: rolling checkpoints serialize only rows touched since "
                        "the last checkpoint (NLDLT001 sidecar); full NLCFR001 bases at gate iters. "
                        "Resume = base + verified contiguous delta chain (driver-sim: final state "
                        "byte-identical to uninterrupted). Cuts the per-checkpoint pause + upload from "
                        "full-blob-sized to dirty-set-sized -- the spot-storm lever at short cadence.")
    p.add_argument("--profile-run", action="store_true", default=False,
                   help="COMPUTE PROFILER sidecar on the box: mpstat -P ALL 30 (per-core utilization) "
                        "+ pidstat -u -r 30 on the driver (CPU%%/RSS churn) + ONE steady-state "
                        "`perf record -F 99 -g` sample ~8 min in, all pushed to "
                        "<run_prefix>/logs/profile/ every 300 s (reclaim-durable). Non-fatal "
                        "everywhere; OFF renders a byte-identical bootstrap.")
    p.add_argument("--row-budget", type=int, default=None,
                   help="G8 RAM backstop: export POKERBOT_ROW_BUDGET on the box (rows in the guarded "
                        "map — max(strat,cum) under v3 snapshot_avg, strat otherwise). Omit = the "
                        "driver's 400M default and a byte-identical bootstrap. The v3 amendment's "
                        "550M was pre-registered but never exported (found in arm-2 prep) — always "
                        "pass this explicitly for fullhist arms.")
    p.add_argument("--v3-regstats", action="store_true",
                   help="v3: print REGSTATS regret-percentile lines at checkpoint cadence")
    args = p.parse_args()

    if args.num_buckets <= 0:
        p.error("--num-buckets must be > 0 (serve uses 200)")
    if args.emd_frozen and not (ROOT / args.emd_frozen).exists():
        p.error(f"--emd-frozen {args.emd_frozen} not found (run precompute_emd_frozen_lookup.py first)")
    if args.river_ochs and not (ROOT / args.river_ochs).exists():
        p.error(f"--river-ochs {args.river_ochs} not found (run scripts/blueprint2/fit_river_ochs.py first)")
    depth_edges = [float(x) for x in args.depth_edges.split(",") if x.strip()]

    s3 = boto3.client("s3", region_name=args.region)
    ec2 = boto3.client("ec2", region_name=args.region)

    run_subdir = args.run_subdir or f"run_{int(time.time())}"
    crate_src = args.crate_src if (ROOT / args.crate_src).exists() else "tools/battery_native"
    if crate_src != args.crate_src:
        print(f"WARN: --crate-src {args.crate_src} not found; falling back to {crate_src}", flush=True)
    if args.reuse_code:
        code_key = f"{args.prefix}/code/native_cfr_code.tar.gz"
        print(f"Reusing existing code tarball -> s3://{args.bucket}/{code_key} (--reuse-code)", flush=True)
    else:
        code_key = build_and_upload_code(s3, args.bucket, args.prefix, crate_src=crate_src,
                                         emd_frozen=args.emd_frozen, river_ochs=args.river_ochs)
    ami, root_dev = resolve_ubuntu_ami(ec2, args.arch)
    print(f"AMI: {ami} (root {root_dev})  instance: {args.instance_type}  arch: {args.arch}  "
          f"region: {args.region}", flush=True)

    # S2: pass the EMD blob's expected sha256 (from its sidecar .sha256) so the box verifies it before training.
    emd_sha = ""
    if args.emd_frozen:
        sha_path = ROOT / (args.emd_frozen + ".sha256")
        if sha_path.exists():
            emd_sha = sha_path.read_text(encoding="utf-8").split()[0].strip()
            print(f"EMD blob sha256 (on-box verify armed): {emd_sha}", flush=True)
        else:
            print(f"WARN: {sha_path} missing -> on-box EMD sha check SKIPPED (blob unverified)", flush=True)
    river_ochs_sha = ""
    if args.river_ochs:
        sha_path = ROOT / (args.river_ochs + ".sha256")
        if sha_path.exists():
            river_ochs_sha = sha_path.read_text(encoding="utf-8").split()[0].strip()
            print(f"river-OCHS table sha256 (on-box verify armed): {river_ochs_sha}", flush=True)
        else:
            print(f"WARN: {sha_path} missing -> on-box river-OCHS sha check SKIPPED (table unverified)",
                  flush=True)
    # blueprint-v4 v3 config dict (None when no v3 flag is set => byte-identical bootstrap,
    # sha-compared against the pre-v3 render). snap_start default = 8% of the target iters
    # (Pluribus's 800-of-11520-minute discard-early rule, iteration-denominated).
    v3_cfg = None
    if args.v3_lcfr or args.v3_signed_regrets or args.v3_snapshot_avg:
        if args.v3_prune_c is not None and not (args.v3_lcfr or args.v3_signed_regrets):
            p.error("--v3-prune-c requires --v3-lcfr or --v3-signed-regrets "
                    "(hard pruning needs signed regrets)")
        if args.v3_lcfr and args.v3_signed_regrets:
            p.error("--v3-lcfr and --v3-signed-regrets are mutually exclusive trainer modes "
                    "(LCFR already implies signed regrets; arm 2 is the no-discount decomposition)")
        if args.dcfr and (args.v3_lcfr or args.v3_signed_regrets):
            p.error("--v3-lcfr/--v3-signed-regrets and --dcfr are mutually exclusive trainer modes")
        if args.v3_signed_regrets and args.v3_prune_start is None and args.v3_prune_c is not None:
            p.error("--v3-signed-regrets with --v3-prune-c requires an explicit --v3-prune-start "
                    "(the trainer's W/2 default is an LCFR-window quantity; arm 2 has no LCFR window)")
        v3_cfg = dict(
            lcfr=args.v3_lcfr, signed_regrets=args.v3_signed_regrets,
            lcfr_d=args.v3_lcfr_d, lcfr_w=args.v3_lcfr_w,
            prune_c=args.v3_prune_c, prune_start=args.v3_prune_start,
            prune_explore=args.v3_prune_explore, regret_floor=args.v3_regret_floor,
            snapshot_avg=args.v3_snapshot_avg,
            snap_start=(args.v3_snap_start if args.v3_snap_start is not None
                        else int(args.iters * 0.08)),
            cov_hands=args.v3_cov_hands, regstats=args.v3_regstats,
        )
        print(f"v3 trainer schedule ARMED: {v3_cfg}", flush=True)

    # B2: the on-box dead-man cost cap floor = --max-hours (absolute, independent of the babysitter/PC).
    wall_guard_secs = max(1, int(args.max_hours * 3600))
    awscli_zip = "awscli-exe-linux-aarch64.zip" if args.arch == "arm64" else "awscli-exe-linux-x86_64.zip"

    # EC2 caps user-data at 16384 bytes and the full bootstrap outgrew it (river-OCHS launch,
    # 2026-07-03). Split: full bootstrap -> S3 (uploaded on EVERY launch, incl. --reuse-code
    # relaunches — content is deterministic from the args, so a babysitter relaunch re-uploads
    # the identical script); user-data = STEP-0 cost cap + fetch-and-exec stub.
    bootstrap = bootstrap_body(
        bucket=args.bucket, prefix=args.prefix, region=args.region, code_key=code_key,
        run_subdir=run_subdir, iters=args.iters, n_deals=args.n_deals,
        num_buckets=args.num_buckets, depth_edges=depth_edges,
        checkpoint_every=args.checkpoint_every, gate_every=args.gate_every,
        keep_alive=args.keep_alive,
        emd_frozen_rel=args.emd_frozen, emd_sha=emd_sha,
        river_ochs_rel=args.river_ochs, river_ochs_sha=river_ochs_sha,
        preflop_hist_key=args.preflop_hist_key,
        fullhist_key=args.fullhist_key, mrr=args.mrr,
        work_steal=args.work_steal, dcfr=args.dcfr,
        dcfr_prune_threshold=args.dcfr_prune_threshold, dcfr_explore_frac=args.dcfr_explore_frac,
        v3=v3_cfg, row_budget=args.row_budget, no_cache_fetch=args.no_cache_fetch,
        delta_ckpt=args.delta_ckpt, profile_run=args.profile_run,
        wall_guard_secs=wall_guard_secs, awscli_zip=awscli_zip,
    )
    bootstrap_key = f"{args.prefix}/{run_subdir}/code/bootstrap.sh"
    s3.put_object(Bucket=args.bucket, Key=bootstrap_key, Body=bootstrap.encode("utf-8"))
    print(f"Uploaded bootstrap ({len(bootstrap)} bytes) -> s3://{args.bucket}/{bootstrap_key}", flush=True)
    ud = user_data_stub(
        bucket=args.bucket, region=args.region, run_prefix=f"{args.prefix}/{run_subdir}",
        bootstrap_key=bootstrap_key, wall_guard_secs=wall_guard_secs, awscli_zip=awscli_zip,
    )
    # EC2's 16384-byte cap applies to the BASE64-ENCODED user-data (that's how the original
    # inline script tripped it at ~13.5KB raw = ~18KB encoded).
    ud_b64_len = len(base64.b64encode(ud.encode("utf-8")))
    if ud_b64_len > 16000:
        raise SystemExit(f"user-data stub too large (base64 {ud_b64_len} bytes) — must stay under 16384")
    run_kwargs = dict(
        ImageId=ami,
        InstanceType=args.instance_type,
        MinCount=1,
        MaxCount=1,
        UserData=base64.b64encode(ud.encode()).decode(),
        IamInstanceProfile={"Name": args.instance_profile},
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": f"native-cfr-train-{run_subdir}"},
                {"Key": "Project", "Value": "pokerbot-native-cfr"},
                {"Key": "Run", "Value": run_subdir},
            ],
        }],
        InstanceInitiatedShutdownBehavior="terminate",
    )
    # Root-volume override: the fullhist/M-RR checkpoint path stages blobs (up to ~58 GB each) on
    # local disk for the DETACHED uploader; the AMI's 8 GB root can't hold them. gp3 with 700 MiB/s
    # provisioned throughput bounds the blocking local write to ~85 s at the largest blob
    # (~$0.04+$0.03/h prorated — negligible vs the box). Default 400 GB on fullhist/M-RR arms.
    ckpt_vol = args.ckpt_volume_gb
    if ckpt_vol is None:
        ckpt_vol = 400 if (args.fullhist_key or args.mrr) else 0
    if ckpt_vol > 0:
        run_kwargs["BlockDeviceMappings"] = [{
            "DeviceName": root_dev,
            "Ebs": {"VolumeSize": int(ckpt_vol), "VolumeType": "gp3", "Iops": 3000,
                    "Throughput": 700, "DeleteOnTermination": True},
        }]
        print(f"root volume override: {ckpt_vol} GB gp3 @700 MiB/s ({root_dev}) for checkpoint staging",
              flush=True)
    if not args.on_demand:
        run_kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"},
        }

    # InstanceType is set per-attempt by _launch_with_fallback; drop the static key so the helper owns it.
    run_kwargs.pop("InstanceType", None)
    instance_types = (
        _spotpick_chain(args.instance_type, args.arch, args.spotpick_n)
        if args.spotpick else [args.instance_type]
    )
    if len(instance_types) > 1:
        print(f"spotpick fallback chain: {instance_types}", flush=True)
    iid, used_type = _launch_with_fallback(
        ec2, run_kwargs, instance_types=instance_types, spot=not args.on_demand
    )
    print(f"Launched {iid} as {used_type} ({'spot' if not args.on_demand else 'on-demand'})", flush=True)
    print(f"Run S3: s3://{args.bucket}/{args.prefix}/{run_subdir}/", flush=True)
    print(f"  checkpoints: .../{run_subdir}/checkpoints/  | logs: .../{run_subdir}/logs/", flush=True)
    print(f"Monitor console: aws ec2 get-console-output --instance-id {iid} --region {args.region} --latest",
          flush=True)
    print(f"Terminate: aws ec2 terminate-instances --instance-ids {iid} --region {args.region}", flush=True)


if __name__ == "__main__":
    main()
