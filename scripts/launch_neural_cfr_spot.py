#!/usr/bin/env python3
"""Launch ONE EC2 spot instance to run VR-DeepPDCFR+ (neural_cfr.py) NLHE training SATURATED on a
many-vCPU box.

This is the neural-CFR sibling of scripts/launch_sd_cfr_training_spot.py (which runs the DIFFERENT
hogwild/pool SD-CFR trainer). It REUSES that launcher's proven bootstrap pattern verbatim -- Canonical
Ubuntu AMI resolution, the offline wheelhouse install, the JAX XLA-compilation-cache restore/sync to
S3, and most importantly the THREAD-PINNING block (the fix for the prior 58-worker x uncapped-XLA-pool
thrash). The only thing that differs is the saturation TOPOLOGY for this trainer, established by
profiling (scripts/neural_cfr/scaling_curve.py):

  * A single JAX-CPU process is ~1-2 cores on the NLHE traversal scan REGARDLESS of XLA Eigen threads
    (benchmarked flat 1->12 threads). So we do NOT give one process a fat Eigen pool.
  * Instead we run the trainer with --traversal-workers W (= vcpus): W INDEPENDENT worker processes,
    each PINNED to one compute thread, generate traversals data-parallel (same estimator, disjoint RNG).
  * Thread arithmetic: parent(1) + W workers x intra_op(1) ~= W+1 compute threads. Set W = vcpus-1 so
    the total == vcpus with ZERO oversubscription (XLA Eigen off + OMP/BLAS=1 on every process).

Recommended instance: c7i.16xlarge (64 vCPU, Intel Sapphire Rapids, x86_64) or c6i.16xlarge. Compute-
bound, no GPU; pick the cheapest 64-vCPU compute-optimized spot. Set --vcpus 64 --traversal-workers 63.
"""

from __future__ import annotations

import argparse
import base64
import io
import tarfile
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUCKET = "video-processing-6789"
DEFAULT_PREFIX = "pokerbot/prod6000/neural_cfr"  # within the IAM-allowed prod6000/* prefix
DEFAULT_REGION = "eu-central-1"
DEFAULT_PROFILE = "pokerbot-labelgen-worker"

# Ship the training import chain + the prebuilt WEVA index (the trainer gates on the production index;
# see scripts/train_neural_cfr.py / assert_production_weva_index). Lean: exclude caches/heavy dirs.
_INCLUDE = [
    "pokerbot",
    "scripts/train_neural_cfr.py",
    "scripts/neural_cfr",
    "data/weva",
]


def _tar_filter(ti: tarfile.TarInfo):
    if any(part == "__pycache__" for part in ti.name.split("/")) or ti.name.endswith(".pyc"):
        return None
    return ti


def build_and_upload_code(s3, bucket: str, prefix: str) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in _INCLUDE:
            p = ROOT / rel
            if p.exists():
                tar.add(p, arcname=rel, filter=_tar_filter)
    buf.seek(0)
    key = f"{prefix}/code/neural_cfr_code.tar.gz"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    print(f"Uploaded code tarball -> s3://{bucket}/{key} ({buf.tell()} bytes)", flush=True)
    return key


def resolve_ubuntu_ami(ec2, arch: str) -> str:
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
    return imgs[0]["ImageId"]


def user_data(*, bucket, prefix, region, code_key, run_subdir, vcpus, traversal_workers, iterations,
              traversals, batch_size, hidden, arch="x86_64", wheels_dir="wheels_py310_x86",
              keep_alive=False, allow_dev_weva=False) -> str:
    run_prefix = f"{prefix}/{run_subdir}"
    final = "echo BOOTSTRAP_DONE_KEEP_ALIVE" if keep_alive else "shutdown -h now"
    weva_flag = "--allow-dev-weva" if allow_dev_weva else ""
    # The neural trainer sets its OWN per-process XLA topology via scripts/neural_cfr/saturate_env.py
    # (invoked by --vcpus): topology=single -> the PARENT process gets 1 device + Eigen pinned to 1
    # (because the real parallelism is the worker pool, not Eigen). Each worker process re-pins itself to
    # one compute thread in the pool initializer (mp_traversal._worker_init) BEFORE importing jax. So the
    # env we export here is the safe single-thread floor for any helper import; the trainer/workers
    # enforce their own pinning on top. Thread arithmetic: parent(1) + W workers(1 each) = W+1 ~= vcpus.
    return f"""#!/bin/bash
set -uxo pipefail
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION={region}
apt-get update -qq
apt-get install -y -qq python3-pip awscli gzip curl
aws --version
mkdir -p /wheels /opt/pokerbot
aws s3 cp s3://{bucket}/{prefix}/{wheels_dir}/ /wheels/ --recursive --region {region} --no-progress
echo "WHEEL_DL_DONE $(ls /wheels | wc -l) wheels"
aws s3 cp s3://{bucket}/{code_key} /tmp/code.tar.gz --region {region}
tar -xzf /tmp/code.tar.gz -C /opt/pokerbot
ls -la /opt/pokerbot/scripts/train_neural_cfr.py
# Restore the shared JAX (XLA) compilation cache so a reclaimed box skips the cold compile.
mkdir -p /tmp/jax_cache
aws s3 cp s3://{bucket}/{prefix}/jax_cache_{arch}/ /tmp/jax_cache/ --recursive --region {region} --no-progress 2>/dev/null || true
echo "JAX_CACHE_RESTORED files=$(find /tmp/jax_cache -type f 2>/dev/null | wc -l)"
PIPI="python3 -m pip install --no-index --no-deps --disable-pip-version-check --no-input"
for w in /wheels/*.whl; do echo "+++ $(basename $w)"; $PIPI "$w" || echo "WHEEL_FAIL $(basename $w)"; done
echo "WHEELS_INSTALLED"
python3 -c "import numpy,scipy,jax,jaxlib,flax,optax,chex,boto3; print('CORE IMPORTS OK','jax',jax.__version__)"
echo "=== deps install done ==="
cd /opt/pokerbot
export PYTHONPATH=/opt/pokerbot JAX_PLATFORMS=cpu JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
# THREAD PINNING (the fix for the prior 58-worker x uncapped-XLA-pool thrash): cap EVERY process
# (parent + spawn-inherited workers) to ONE compute thread. The saturation comes from the trainer's
# --traversal-workers process pool ({traversal_workers} thin single-thread workers), NOT from fat
# per-process Eigen pools. Total compute threads = 1 (parent) + {traversal_workers} (workers) ~= {vcpus}.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NPY_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
# Background syncer: push newly-compiled XLA executables to S3 every 90s for reclaim resilience.
cat > /tmp/jc_sync.py <<'JCEOF'
import time, os, boto3
s3 = boto3.client("s3", region_name="{region}")
B, P, base = "{bucket}", "{prefix}/jax_cache_{arch}", "/tmp/jax_cache"
sent = set()
while True:
    time.sleep(90)
    for root, _, files in os.walk(base):
        for f in files:
            p = os.path.join(root, f)
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            rel = os.path.relpath(p, base).replace(os.sep, "/")
            if (rel, mt) in sent:
                continue
            try:
                s3.upload_file(p, B, P + "/" + rel)
                sent.add((rel, mt))
            except Exception:
                pass
JCEOF
nohup python3 /tmp/jc_sync.py >/var/log/jc_sync.log 2>&1 &
mkdir -p /tmp/out
python3 scripts/train_neural_cfr.py \
  --vcpus {vcpus} --sat-topology single \
  --iterations {iterations} --traversals {traversals} --batch-size {batch_size} \
  --traversal-workers {traversal_workers} {weva_flag} \
  --checkpoint /tmp/out/history_value_net.pkl \
  2>&1 | tee /var/log/neural-cfr-train.log
echo "train exit ${{PIPESTATUS[0]}}"
# Upload checkpoints + logs via boto3 (host aws-cli is botocore-incompatible after the pip loop).
python3 - <<'PYEOF'
import boto3, glob, os
s3 = boto3.client("s3", region_name="{region}")
for src in glob.glob("/tmp/out/*"):
    try:
        s3.upload_file(src, "{bucket}", "{run_prefix}/checkpoints/" + os.path.basename(src)); print("up", src)
    except Exception as e:
        print("ckpt upload fail", src, e)
for src, key in [("/var/log/cloud-init-output.log", "{run_prefix}/logs/cloud-init-output.log"),
                 ("/var/log/neural-cfr-train.log", "{run_prefix}/logs/train.log")]:
    try:
        s3.upload_file(src, "{bucket}", key); print("uploaded", key)
    except Exception as e:
        print("log upload fail", src, e)
PYEOF
{final}
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instance-type", default="c7i.16xlarge",
                   help="64-vCPU compute-optimized recommended: c7i.16xlarge / c6i.16xlarge")
    p.add_argument("--arch", default="x86_64", choices=["x86_64", "arm64"])
    p.add_argument("--vcpus", type=int, default=64)
    p.add_argument("--traversal-workers", type=int, default=None,
                   help="worker processes (default vcpus-1, leaving 1 thread for the parent grad/ingest)")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--traversals", type=int, default=1024,
                   help="traversals/iter; make it a multiple of traversal-workers for even shards")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--instance-profile", default=DEFAULT_PROFILE)
    p.add_argument("--on-demand", action="store_true")
    p.add_argument("--allow-dev-weva", action="store_true", help="skip production WEVA gate (smoke only)")
    p.add_argument("--keep-alive", action="store_true", help="don't shut down on finish")
    p.add_argument("--run-subdir", default=None)
    args = p.parse_args()

    workers = args.traversal_workers if args.traversal_workers is not None else max(1, args.vcpus - 1)

    s3 = boto3.client("s3", region_name=args.region)
    ec2 = boto3.client("ec2", region_name=args.region)
    run_subdir = args.run_subdir or f"run_{int(time.time())}"
    code_key = build_and_upload_code(s3, args.bucket, args.prefix)
    ami = resolve_ubuntu_ami(ec2, args.arch)
    wheels_dir = "wheels_py310_x86" if args.arch == "x86_64" else "wheels_py310_arm64"
    print(f"AMI: {ami}  instance: {args.instance_type}  vcpus: {args.vcpus}  workers: {workers}", flush=True)

    ud = user_data(
        bucket=args.bucket, prefix=args.prefix, region=args.region, code_key=code_key,
        run_subdir=run_subdir, vcpus=args.vcpus, traversal_workers=workers,
        iterations=args.iterations, traversals=args.traversals, batch_size=args.batch_size,
        hidden=args.hidden, arch=args.arch, wheels_dir=wheels_dir, keep_alive=args.keep_alive,
        allow_dev_weva=args.allow_dev_weva,
    )
    run_kwargs = dict(
        ImageId=ami, InstanceType=args.instance_type, MinCount=1, MaxCount=1,
        UserData=base64.b64encode(ud.encode()).decode(),
        IamInstanceProfile={"Name": args.instance_profile},
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": f"neural-cfr-train-{run_subdir}"},
                {"Key": "Project", "Value": "pokerbot-neural-cfr"},
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
    resp = ec2.run_instances(**run_kwargs)
    iid = resp["Instances"][0]["InstanceId"]
    print(f"Launched {iid} ({'spot' if not args.on_demand else 'on-demand'})", flush=True)
    print(f"Run S3: s3://{args.bucket}/{args.prefix}/{run_subdir}/", flush=True)
    print(f"Monitor: aws ec2 get-console-output --instance-id {iid} --region {args.region} --latest", flush=True)
    print(f"Terminate: aws ec2 terminate-instances --instance-ids {iid} --region {args.region}", flush=True)


if __name__ == "__main__":
    main()
