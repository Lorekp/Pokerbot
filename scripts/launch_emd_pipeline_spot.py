#!/usr/bin/env python3
"""Launch a SINGLE EC2 spot box that runs the WHOLE EMD blueprint pipeline in one cloud-init:

    build the EMD crate  ->  ensure the frozen (hole,board)->bucket table  ->  train to convergence.

This is the in-box bootstrap behind ``scripts/emd_pipeline_babysitter.ps1`` (which relaunches it on spot
preemption). It is modeled on ``scripts/launch_native_cfr_training_spot.py`` (the spot launch + S3
checkpoint resume + spotpick fallback chain), but bundles the PRECOMPUTE phase ahead of training so the
user launches ONE command for the entire pipeline.

PHASES (cloud-init, run on EVERY boot so a relaunched box is self-healing):
  0. BUILD : ``cargo build --release --features python`` of the EMD crate (runs/bp_crate_emd, uploaded as
             tools/battery_native), producing the importable ``battery_native`` extension.
  1. TABLE : compute the deterministic TABLE-ID from (centroids file sha256, n_bins, n_runouts). Check S3
             for a cached frozen table at ``{prefix}/{run}/frozen/<TABLE_ID>.emdfrzn``.
               * PRESENT -> download it (fast, seconds; NO recompute). This is the spot-resume fast path.
               * ABSENT  -> run the production precompute (scripts/precompute_emd_frozen_lookup.py,
                            n_runouts=200) which enumerates all ~16.5M canonical flop+turn spots, then
                            UPLOAD the table to S3 (cache it for every later boot/relaunch).
  2. TRAIN : drive the native ``NlheCFRTrainer(num_buckets, depth_edges, n_deals, ..., emd_frozen_path)``,
             RESUMING from the latest S3 checkpoint if one exists, checkpointing to S3 every
             ``--checkpoint-every`` iters, to ``--iters`` (the convergence stop in the babysitter ends it
             earlier). Both phases log per-unit %/throughput/ETA/heartbeat (mandatory progress logging).

SPOT RESILIENCE: on preemption the box dies; the babysitter relaunches with the SAME --run-subdir. The new
box's cloud-init RE-RUNS phase 1 (finds the cached table -> downloads, no recompute) and phase 2 RESUMES
from the latest S3 checkpoint. Nothing heavy is repeated.

The crate is ALWAYS uploaded as ``tools/battery_native`` (from ``--crate-src``, default the isolated EMD
crate) so a running pipeline is insulated from concurrent edits to the live crate.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import tarfile
import time
from pathlib import Path

import boto3

# Reuse the SAME helpers/conventions as the native-CFR launcher (spotpick chain, AMI resolve, fallback
# launch, tar filter) so this stays a thin bundle on top of the proven launch path.
from launch_native_cfr_training_spot import (  # noqa: E402
    DEFAULT_BUCKET,
    DEFAULT_PREFIX,
    DEFAULT_PROFILE,
    DEFAULT_REGION,
    _launch_with_fallback,
    _spotpick_chain,
    _tar_filter,
    resolve_ubuntu_ami,
)

ROOT = Path(__file__).resolve().parents[1]

# CRN reseed per replication. Mirrors scripts/eval/eval_emd_vs_iter62000.py — the on-box eval driver
# applies seed_for_replication() to the --seeds count it receives from --eval-seeds, so the launcher's
# only job is to forward the COUNT. Re-export the helper for any future in-launcher iteration that
# may pass a seed list directly. Per artifacts/pluribus_emd_retrain_report.md hypothesis B.
try:
    from eval.eval_emd_vs_iter62000 import BASE_SEED, seed_for_replication  # noqa: E402
except ImportError:  # pragma: no cover  -- sibling launcher modules live under scripts/eval/
    BASE_SEED = 0x5EEDBAB0  # 1592263344; "BABY" mnemonic preserved (must match eval script)

    def seed_for_replication(replication_id: int) -> int:
        """Mirror of scripts/eval/eval_emd_vs_iter62000.py.seed_for_replication (kept in sync)."""
        s = BASE_SEED ^ int(replication_id)
        if replication_id != 0:
            assert s != BASE_SEED, f"CRN reseed must differ from base for non-zero replication_id (got {s})"
        assert s >= 0, "XOR of non-negative ints must be non-negative"
        return s


def table_id(centroids_path: Path, n_bins: int, n_runouts: int) -> str:
    """Deterministic table id from the centroids file content + n_bins + n_runouts. Two runs with the
    SAME centroids + params reuse the SAME cached frozen table (the precompute is a pure function of
    these, so a content hash is a correct cache key — independent of the produced blob's own stamp)."""
    h = hashlib.sha256()
    h.update(centroids_path.read_bytes())
    h.update(f"|n_bins={n_bins}|n_runouts={n_runouts}".encode())
    return h.hexdigest()[:16]


def build_and_upload_code(s3, bucket: str, prefix: str, *, crate_src: str, centroids_rel: str) -> str:
    """Tar the EMD crate (as tools/battery_native), the WHOLE pokerbot package + the ship/eval scripts,
    AND the centroids JSON, then upload as the pipeline code tarball.

    The 5-phase pipeline needs ALL of: the precompute (PHASE 1), the trainer driver (PHASE 2 -- native
    crate + boto3 only), the ship gate (PHASE 4 -- ship_blueprint.py + its serve adapter + the EMD
    encoder), and the harsh LBR eval (PHASE 5 -- pokerbot.eval.lbr6/lbr + the engine + the EMD frozen
    serve encoder). Enumerating individual files was fragile, so we ship the whole ``pokerbot`` package
    (and the two driver scripts) and let the box import what each phase needs. Centroids are SHIPPED in
    the tarball (small) so the box can compute the table-id + run the precompute without a second
    download. ``_tar_filter`` strips bytecode/.git/large junk so this stays modest."""
    dir_includes = [
        (crate_src, "tools/battery_native"),
        ("pokerbot", "pokerbot"),                  # whole package: engine + eval + encoding + search + training
    ]
    file_includes = [
        ("scripts/emd_pipeline_onbox.sh", "scripts/emd_pipeline_onbox.sh"),    # the 5-phase on-box driver
        ("scripts/precompute_emd_frozen_lookup.py", "scripts/precompute_emd_frozen_lookup.py"),
        ("scripts/ship_blueprint.py", "scripts/ship_blueprint.py"),            # PHASE 4 SHIP gate
        ("scripts/_emd_verdict_sign.py", "scripts/_emd_verdict_sign.py"),      # PHASE 5 verdict renderer/sign logic
        ("scripts/_atomic_io.py", "scripts/_atomic_io.py"),                    # PHASE 5 atomic JSON write (CRIT — without this the eval driver crashes on import)
        ("scripts/eval/eval_emd_vs_iter62000.py", "scripts/eval/eval_emd_vs_iter62000.py"),  # PHASE 5 EVAL
        (centroids_rel, centroids_rel),
    ]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for src, arc in dir_includes:
            p = ROOT / src
            if p.exists():
                tar.add(p, arcname=arc, filter=_tar_filter)
        for src, arc in file_includes:
            p = ROOT / src
            if p.exists():
                tar.add(p, arcname=arc, filter=_tar_filter)
    buf.seek(0)
    key = f"{prefix}/code/emd_pipeline_code.tar.gz"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    print(f"Uploaded EMD pipeline code tarball -> s3://{bucket}/{key} "
          f"({buf.tell()} bytes; crate_src={crate_src})", flush=True)
    return key


def user_data(*, bucket, prefix, region, code_key, run_subdir, iters, n_deals, num_buckets,
              depth_edges, checkpoint_every, centroids_rel, n_bins, n_runouts, tid,
              iter62000_snapshot_s3, eval_seeds, eval_hands, ship_name="emd_bp1",
              stop_after_ship=False, train_profile_every=0, train_fast_merge=False,
              keep_alive=False) -> str:
    """MINIMAL cloud-init bootstrap (kept well under the 16 KB EC2 user_data limit).

    All heavy logic -- the 5 phases, markers, resume-at-first-incomplete, the trainer driver, the spot
    handler -- lives in ``scripts/emd_pipeline_onbox.sh``, SHIPPED inside the code tarball. The bootstrap
    only: installs awscli + rust prerequisites, downloads + extracts the already-uploaded code tarball,
    installs the rust toolchain + python deps, EXPORTS the phase config as env vars, and runs the on-box
    driver. This keeps user_data small and means a config/phase change never grows user_data (it grows the
    tarball instead). The config is passed via env (not argv) so the on-box script stays self-documenting.
    """
    run_prefix = f"{prefix}/{run_subdir}"
    final = "echo BOOTSTRAP_DONE_KEEP_ALIVE" if keep_alive else "shutdown -h now"
    depth_csv = ",".join(str(float(x)) for x in (depth_edges or []))
    # NOTE: doubled braces escape the f-string; the emitted bash sees single braces.
    return f"""#!/bin/bash
set -uxo pipefail
export HOME="${{HOME:-/root}}"   # cloud-init runs as root with HOME UNSET; under set -u a bare $HOME is FATAL
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION={region}

apt-get update -qq
apt-get install -y -qq python3-pip python3-dev awscli gzip curl build-essential
aws --version
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"
rustc --version

CODE_DIR=/opt/pokerbot
mkdir -p "$CODE_DIR"
aws s3 cp s3://{bucket}/{code_key} /tmp/code.tar.gz --region {region}
tar -xzf /tmp/code.tar.gz -C "$CODE_DIR"

python3 -m pip install --disable-pip-version-check --no-input \
  boto3 numpy phevaluator "jax[cpu]==0.4.35"
python3 -c "import boto3,numpy,jax,phevaluator; print('SHIP_EVAL_IMPORTS_OK', 'jax', jax.__version__)"

# Phase config -> env for the on-box driver (the heavy 5-phase logic lives in the tarball).
export BUCKET="{bucket}" PREFIX="{prefix}" REGION="{region}" RUN_SUBDIR="{run_subdir}" CODE_DIR="$CODE_DIR"
export ITERS="{iters}" N_DEALS="{n_deals}" NUM_BUCKETS="{num_buckets}" CKPT_EVERY="{checkpoint_every}"
export DEPTH_CSV="{depth_csv}" CENTROIDS_REL="{centroids_rel}" N_BINS="{n_bins}" N_RUNOUTS="{n_runouts}"
export TABLE_ID="{tid}" SHIP_NAME="{ship_name}" ITER62000_SNAPSHOT_S3="{iter62000_snapshot_s3}"
export EVAL_SEEDS="{eval_seeds}" EVAL_HANDS="{eval_hands}"
export STOP_AFTER_SHIP="{1 if stop_after_ship else 0}"
export TRAIN_PROFILE_EVERY="{int(train_profile_every)}"
export TRAIN_FAST_MERGE="{1 if train_fast_merge else 0}"

ONBOX="$CODE_DIR/scripts/emd_pipeline_onbox.sh"
test -s "$ONBOX" || {{ echo "FATAL: $ONBOX missing from the code tarball"; exit 1; }}
bash "$ONBOX"
echo "onbox exit $?"
{final}
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instance-type", default="c6a.32xlarge",
                   help="c6a.32xlarge=full (128 vCPU); c6i.4xlarge=cheap validate")
    p.add_argument("--arch", default="x86_64", choices=["x86_64", "arm64"])
    p.add_argument("--iters", type=int, default=250000, help="total CFR+ iterations (convergence backstop)")
    p.add_argument("--n-deals", type=int, default=200, help="deals fanned out per iteration (rayon)")
    p.add_argument("--num-buckets", type=int, default=200,
                   help="EMD bucket count; MUST equal the centroids' K (the frozen table's num_buckets)")
    p.add_argument("--depth-edges", default="", help="comma-sep depth edges (empty = non-depth blueprint)")
    p.add_argument("--checkpoint-every", type=int, default=2000,
                   help="checkpoint to S3 every N iters (preemption-bounded loss)")
    p.add_argument("--centroids", default="data/abstraction/emd_buckets_200.json",
                   help="EMDBucketer centroids JSON (K clusters); the frozen table is built from these")
    p.add_argument("--n-bins", type=int, default=0, help="override centroids' n_bins (0 = use the file's)")
    p.add_argument("--n-runouts", type=int, default=200,
                   help="board completions per spot for the PRODUCTION precompute (0 = use the file's)")
    p.add_argument("--run-subdir", default=None, help="S3 subdir for this run (default: run_<ts>)")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--instance-profile", default=DEFAULT_PROFILE)
    p.add_argument("--on-demand", action="store_true", help="use on-demand instead of spot")
    p.add_argument("--spotpick", action="store_true",
                   help="retry down a capacity-diverse fallback chain on a spot-capacity error")
    p.add_argument("--spotpick-n", type=int, default=5, help="max types in the --spotpick chain")
    p.add_argument("--keep-alive", action="store_true",
                   help="don't shut down on finish (keep box for get-console-output inspection)")
    p.add_argument("--reuse-code", action="store_true",
                   help="skip the code upload; REUSE the existing S3 code tarball (babysitter relaunches)")
    p.add_argument("--crate-src", default="runs/bp_crate_emd/battery_native",
                   help="crate source to upload (always as tools/battery_native); default = the isolated "
                        "EMD crate, insulating the run from concurrent live-crate edits.")
    # --- PHASE 4/5 ship + eval ---
    p.add_argument("--ship-name", default="emd_bp1",
                   help="version tag for the staged serve snapshot (checkpoints/serve_blueprints/<name>.pkl)")
    p.add_argument("--iter62000-snapshot-s3",
                   default="s3://video-processing-6789/pokerbot/prod6000/native_cfr/"
                           "bp6max_pluribus/serve/iter62000_snapshot.pkl",
                   help="the SHIPPED iter62000 serve snapshot to LBR as the verified-better baseline (CRN)")
    p.add_argument("--eval-seeds", type=int, default=12,
                   help="PHASE 5: number of CRN seeds to pool the LBR over (both blueprints share them)")
    p.add_argument("--eval-hands", type=int, default=2000,
                   help="PHASE 5: hands per seed (Python LBR is GIL-bound; lean on the CRN per-seat delta)")
    p.add_argument("--stop-after-ship", action="store_true",
                   help="stop after PHASE 4 staged serve artifact; do not enter PHASE 5 eval")
    p.add_argument("--train-profile-every", type=int, default=0,
                   help="emit TRAIN_PROFILE_JSON every N train iterations on-box (0 disables; safe default)")
    p.add_argument("--train-fast-merge", action="store_true",
                   help="enable statistical-equivalence tree-reduce merge; requires separate verifier approval")
    args = p.parse_args()

    if args.num_buckets <= 0:
        p.error("--num-buckets must be > 0")
    if args.train_profile_every < 0:
        p.error("--train-profile-every must be >= 0")
    centroids_rel = args.centroids.replace("\\", "/")
    if centroids_rel != args.centroids:
        print(f"Normalized --centroids path: {args.centroids} -> {centroids_rel}", flush=True)
    cen = ROOT / centroids_rel
    if not cen.exists():
        p.error(f"--centroids {centroids_rel} not found (run precompute_emd_buckets.py first)")
    # Resolve n_bins/n_runouts from the centroids file when 0, so the TABLE_ID matches what the box uses.
    import json
    cd = json.loads(cen.read_text(encoding="utf-8"))
    n_bins = args.n_bins or int(cd.get("n_bins", 20))
    n_runouts = args.n_runouts or int(cd.get("n_runouts", 200))
    k = len(cd.get("centroids") or [])
    if k == 0:
        p.error(f"--centroids {centroids_rel} has no fitted clusters")
    if k != args.num_buckets:
        p.error(f"--num-buckets {args.num_buckets} != centroids K={k} (the frozen table's num_buckets); "
                f"pass --num-buckets {k}")
    if args.num_buckets > 200:
        p.error(f"--num-buckets {args.num_buckets} exceeds the production K-cap of 200. K=500 is banned (overfit regression on 2026-06-25).")
    tid = table_id(cen, n_bins, n_runouts)
    depth_edges = [float(x) for x in args.depth_edges.split(",") if x.strip()]

    s3 = boto3.client("s3", region_name=args.region)
    ec2 = boto3.client("ec2", region_name=args.region)

    run_subdir = args.run_subdir or f"run_{int(time.time())}"
    crate_src = args.crate_src if (ROOT / args.crate_src).exists() else "tools/battery_native"
    if crate_src != args.crate_src:
        print(f"WARN: --crate-src {args.crate_src} not found; falling back to {crate_src}", flush=True)

    if args.reuse_code:
        code_key = f"{args.prefix}/code/emd_pipeline_code.tar.gz"
        print(f"Reusing existing code tarball -> s3://{args.bucket}/{code_key} (--reuse-code)", flush=True)
    else:
        code_key = build_and_upload_code(s3, args.bucket, args.prefix, crate_src=crate_src,
                                         centroids_rel=centroids_rel)
    ami = resolve_ubuntu_ami(ec2, args.arch)
    print(f"AMI: {ami}  instance: {args.instance_type}  arch: {args.arch}  region: {args.region}", flush=True)
    print(f"TABLE_ID={tid}  (centroids K={k}, n_bins={n_bins}, n_runouts={n_runouts})", flush=True)

    ud = user_data(
        bucket=args.bucket, prefix=args.prefix, region=args.region, code_key=code_key,
        run_subdir=run_subdir, iters=args.iters, n_deals=args.n_deals, num_buckets=args.num_buckets,
        depth_edges=depth_edges, checkpoint_every=args.checkpoint_every, centroids_rel=centroids_rel,
        n_bins=n_bins, n_runouts=n_runouts, tid=tid, keep_alive=args.keep_alive,
        iter62000_snapshot_s3=args.iter62000_snapshot_s3, eval_seeds=args.eval_seeds,
        eval_hands=args.eval_hands, ship_name=args.ship_name,
        stop_after_ship=args.stop_after_ship,
        train_profile_every=args.train_profile_every,
        train_fast_merge=args.train_fast_merge,
    )
    # HARD GUARD: EC2 user_data has a 16 KB limit. The 5-phase logic was externalized into the tarball-
    # shipped scripts/emd_pipeline_onbox.sh so this bootstrap stays tiny; assert it CANNOT silently regress
    # (the inline version previously blew past the limit -> InvalidParameterValue at run_instances). We check
    # BOTH the raw bytes (< 15000, generous headroom) and the base64-encoded bytes EC2 actually counts.
    ud_bytes = len(ud.encode())
    ud_b64_bytes = len(base64.b64encode(ud.encode()))
    _UD_RAW_LIMIT = 15000      # our self-imposed headroom under the 16384 EC2 raw limit
    if ud_bytes >= _UD_RAW_LIMIT or ud_b64_bytes >= 16384:
        raise SystemExit(
            f"FATAL: cloud-init user_data too large ({ud_bytes} raw bytes / {ud_b64_bytes} base64 bytes); "
            f"EC2 caps it at 16384. Keep user_data a MINIMAL bootstrap -- move logic into the tarball-"
            f"shipped scripts/emd_pipeline_onbox.sh, not into user_data. (limit: raw < {_UD_RAW_LIMIT}, "
            f"base64 < 16384)")
    print(f"user_data size: {ud_bytes} raw bytes / {ud_b64_bytes} base64 bytes (limit 16384) -- OK", flush=True)

    run_kwargs = dict(
        ImageId=ami, MinCount=1, MaxCount=1,
        UserData=base64.b64encode(ud.encode()).decode(),
        IamInstanceProfile={"Name": args.instance_profile},
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": f"emd-pipeline-{run_subdir}"},
                {"Key": "Project", "Value": "pokerbot-emd-pipeline"},
                {"Key": "Run", "Value": run_subdir},
            ],
        }],
        InstanceInitiatedShutdownBehavior="terminate",
    )
    if not args.on_demand:
        run_kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"},
        }

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
    print(f"  frozen: .../{run_subdir}/frozen/{tid}.emdfrzn  | checkpoints: .../checkpoints/  | logs: .../logs/",
          flush=True)
    print(f"Monitor console: aws ec2 get-console-output --instance-id {iid} --region {args.region} --latest",
          flush=True)
    print(f"Terminate: aws ec2 terminate-instances --instance-ids {iid} --region {args.region}", flush=True)


if __name__ == "__main__":
    main()
