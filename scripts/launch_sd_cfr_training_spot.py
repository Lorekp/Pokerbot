#!/usr/bin/env python3
"""
Launch a SINGLE EC2 spot instance to run SD-CFR HU training (single-machine, Pluribus-style).

Packages the code + SFT checkpoint to S3, launches one arm64 (Graviton) spot instance, and bootstraps
it to pip-install JAX/flax/optax/torch, pull the code, and run scripts/train_sd_cfr_hu_cloud.py with
fork-parallel traversal + S3 checkpoint/resume + spot-interrupt handling. Logs are uploaded to S3.

Validation-first: default instance is small/cheap (c6g.2xlarge) running a SHORT run, to confirm the
whole path (bootstrap, JAX+fork parallelism, S3 checkpoint) before the full c6g.16xlarge run.

Reuses the labelgen fleet's AWS identity (account 392971510945, region eu-central-1, instance
profile pokerbot-labelgen-worker) and the allowed S3 prefix pokerbot/prod6000/*.
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
DEFAULT_PREFIX = "pokerbot/prod6000/sd_cfr_hu"  # within the IAM-allowed prod6000/* prefix
DEFAULT_REGION = "eu-central-1"
DEFAULT_PROFILE = "pokerbot-labelgen-worker"

# Files to ship (the training import chain + SFT + the prebuilt WEVA index). Keep it lean; exclude
# caches/heavy unrelated dirs. data/weva (the WEVA bucket abstraction, ~121 KB) MUST be shipped so
# the cloud loads the SAME abstraction as local — otherwise WEVAMapper.default() rebuilds a different
# "minimal dev index" (and needs faiss/sklearn, which aren't installed), causing train/serve skew.
_INCLUDE = [
    "pokerbot",
    "scripts/train_sd_cfr_hu_cloud.py",
    "checkpoints/policy_sft_prod6000.pt",
    "data/weva",
]
_EXCLUDE_SUFFIX = ("__pycache__", ".pyc", ".git")


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
    key = f"{prefix}/code/sd_cfr_code.tar.gz"
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


def user_data(*, bucket, prefix, region, code_key, run_subdir, workers, iterations,
              traversals, deal_sweeps, max_depth, no_sft=False, wheels_dir="wheels_py310_x86",
              keep_alive=False, bet_buckets=None, postflop_buckets=None, betting_abstraction=False,
              client_bet_sizing=False, trainer="pool", hogwild_capacity=None, arch="arm64",
              occupancy="hu", cfr_variant="cfr_plus", vr_baseline=False,
              preflop_lossless=False, preflop_lossless_betting=False,
              depth_buckets=False, depth_bb=None, occupancy_mix=None) -> str:
    run_prefix = f"{prefix}/{run_subdir}"
    s3log = f"s3://{bucket}/{run_prefix}/logs"
    # --no-sft-init avoids the torch dependency (torch is only needed to load the .pt SFT). Skip the
    # heavy torch wheel and add the flag. Used for fast cloud-path/parallelism validation.
    skip_torch = "[[ \"$w\" == *torch* ]] && continue" if no_sft else "true"
    sft_flag = "--no-sft-init" if no_sft else ""
    bet_flag = f"--bet-buckets {bet_buckets}" if bet_buckets else ""
    client_flag = "--client-bet-sizing" if client_bet_sizing else ""
    trainer_flag = f"--trainer {trainer}"
    occupancy_flag = f"--occupancy {occupancy}"
    variant_flag = f"--cfr-variant {cfr_variant}"
    vr_flag = "--vr-baseline" if vr_baseline else ""
    cap_flag = f"--hogwild-capacity {int(hogwild_capacity)}" if hogwild_capacity else ""
    # Card (information) abstraction env: bounds the postflop card dimension so the tables saturate.
    postflop_env = (
        f"POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_POSTFLOP_MODE=fast POKERBOT_POSTFLOP_NUM_BUCKETS={int(postflop_buckets)}"
        if postflop_buckets else ""
    )
    # Betting-state abstraction (bounded key → the blueprint converges). Pairs with postflop buckets.
    if betting_abstraction:
        postflop_env = (postflop_env + " POKERBOT_BETTING_ABSTRACTION=1").strip()
    # Lossless-preflop (Pluribus-faithful): preflop keyed on the 169 canonical class + lossless preflop
    # betting history (postflop stays bucketed). Closes the certified +180 preflop leak; from-scratch
    # (changes the preflop key space). Serve MUST set the same flag — stamped into the checkpoint.
    if preflop_lossless:
        postflop_env = (postflop_env + " POKERBOT_PREFLOP_LOSSLESS=1").strip()
    # OPT-IN: also make preflop BETTING lossless (Pluribus-exact). Explodes the infoset count -> needs a
    # much larger --hogwild-capacity and more iters; default OFF (cards-only is tractable + fits).
    if preflop_lossless_betting:
        postflop_env = (postflop_env + " POKERBOT_PREFLOP_LOSSLESS_BETTING=1").strip()
    # Depth-aware blueprint (A2/A3): key a per-seat effective-STARTING-depth bucket into the PREFLOP key
    # (POKERBOT_DEPTH_BUCKETS) AND randomize each deal's starting stack over the live range
    # (POKERBOT_TRAIN_DEPTH_BB=lo,hi, log-uniform). Both are needed for a coherent depth retrain. Serve MUST
    # set POKERBOT_DEPTH_BUCKETS=1 (stamped into the checkpoint + cross-checked by check_blueprint_serve_env).
    if depth_buckets:
        postflop_env = (postflop_env + " POKERBOT_DEPTH_BUCKETS=1").strip()
    if depth_bb:
        postflop_env = (postflop_env + f" POKERBOT_TRAIN_DEPTH_BB={depth_bb}").strip()
    # Short-handed coverage (C1): randomize each deal's live-seat count/occupancy over LO,HI so the
    # blueprint prior covers 5/4/3-handed + heads-up (the engine + search already play any count; this adds
    # the trained warm-start). No abstraction-key change — position/history already separate seat counts.
    if occupancy_mix:
        postflop_env = (postflop_env + f" POKERBOT_TRAIN_OCCUPANCY_MIX={occupancy_mix}").strip()
    torch_check =("echo torch-skipped-no-sft" if no_sft
                   else 'python3 -c "import torch; print(\'torch\', torch.__version__)"')
    # Validation: keep the box alive on finish so we can inspect via get-console-output; production:
    # shut down (and the launcher sets InstanceInitiatedShutdownBehavior=terminate) to stop billing.
    final = "echo BOOTSTRAP_DONE_KEEP_ALIVE" if keep_alive else "shutdown -h now"
    # NOTE (root-cause of the 13-attempt failure): the prior bootstrap did `exec > logfile 2>&1`,
    # which HID all output from cloud-init's console/output-log capture (so get-console-output was
    # blank and we were blind) and made the apt aws-cli v1 read its own still-open stdout log. The
    # WORKING labelgen launcher (scripts/launch_labelgen_spot.py) does NOT redirect — output flows to
    # /var/log/cloud-init-output.log, visible via `aws ec2 get-console-output`. We mirror that here:
    # no redirect, plain apt awscli + aws s3 cp, no MTU/timeout cruft. We also copy the cloud-init
    # output log to S3 at the end for convenience, but the serial console is the primary channel.
    return f"""#!/bin/bash
set -uxo pipefail
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION={region}

# GRACEFUL SPOT INTERRUPT is owned by the trainer (scripts/train_sd_cfr_hu_cloud.py): a background
# thread polls the IMDSv2 spot-interruption notice WITH the required token (HttpTokens=required, so a
# tokenless poll would 401 forever — the prior inline bash handler was silently dead) and on notice
# (or SIGTERM) checkpoints the current state to S3 and exits cleanly. Combined with --checkpoint-every
# 1, a reclaim loses at most the single in-flight iteration; a replacement box (spawned by
# scripts/sd_cfr_coordinator.py) resumes bit-exactly from the latest S3 checkpoint. Nothing to do here.

apt-get update -qq
apt-get install -y -qq python3-pip awscli gzip curl
aws --version
mkdir -p /wheels /opt/pokerbot
aws s3 cp s3://{bucket}/{prefix}/{wheels_dir}/ /wheels/ --recursive --region {region} --no-progress
echo "WHEEL_DL_DONE $(ls /wheels | wc -l) wheels"
# Download+unpack the code tarball NOW, BEFORE the pip loop. The loop installs botocore 1.43 into
# system site-packages, which is INCOMPATIBLE with the apt aws-cli v1.22 (-> KeyError 'opsworkscm'),
# so any aws-cli call AFTER the loop crashes. All aws-cli DOWNLOADS must happen here; post-training
# uploads use boto3 (compatible with the new botocore) instead.
aws s3 cp s3://{bucket}/{code_key} /tmp/code.tar.gz --region {region}
tar -xzf /tmp/code.tar.gz -C /opt/pokerbot
ls -la /opt/pokerbot/scripts/train_sd_cfr_hu_cloud.py
# RECLAIM RESILIENCE: restore the shared JAX (XLA) compilation cache from S3 so a fresh/reclaimed box
# SKIPS the cold engine-JIT compile (the dominant startup->first-checkpoint cost). Keyed by arch (same
# jaxlib/AMI ⇒ cache hits). Must run here, in the aws-cli-OK zone (the pip loop below breaks aws-cli v1).
mkdir -p /tmp/jax_cache
aws s3 cp s3://{bucket}/{prefix}/jax_cache_{arch}/ /tmp/jax_cache/ --recursive --region {region} --no-progress 2>/dev/null || true
echo "JAX_CACHE_RESTORED files=$(find /tmp/jax_cache -type f 2>/dev/null | wc -l)"
PIPI="python3 -m pip install --no-index --no-deps --disable-pip-version-check --no-input"
for w in /wheels/*.whl; do {skip_torch}; echo "+++ $(basename $w)"; $PIPI "$w" || echo "WHEEL_FAIL $(basename $w)"; done
echo "WHEELS_INSTALLED"
python3 -c "import numpy,scipy,jax,jaxlib,flax,optax,chex,boto3; print('CORE IMPORTS OK','numpy',numpy.__version__,'jax',jax.__version__)"
{torch_check}
echo "=== deps install done ==="
cd /opt/pokerbot
export PYTHONPATH=/opt/pokerbot JAX_PLATFORMS=cpu JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache {postflop_env}
# THREAD PINNING (critical for many-process CPU JAX): each of the N worker processes otherwise spawns
# an XLA-Eigen threadpool (~#cores) AND an OpenBLAS threadpool (~#cores) → N×~128 threads thrashing 64
# cores (the stall). Cap every process (parent + spawn-inherited workers) to ONE compute thread; the
# lock-free design wants many THIN single-thread workers, not few fat multi-thread ones.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NPY_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
# Background syncer (boto3 — aws-cli is botocore-broken post-pip): push newly-compiled XLA executables to
# S3 every 90s so the NEXT reclaimed box restores them and skips the cold compile. The first box pays the
# compile once; every reclaim after is fast.
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
python3 scripts/train_sd_cfr_hu_cloud.py \
  --s3-bucket {bucket} --s3-prefix {run_prefix} --region {region} \
  --workers {workers} --iterations {iterations} --traversals {traversals} \
  --deal-sweeps {deal_sweeps} --max-depth {max_depth} --checkpoint-every 1 {sft_flag} {bet_flag} {client_flag} \
  {trainer_flag} {cap_flag} {occupancy_flag} {variant_flag} {vr_flag} \
  2>&1 | tee /var/log/sdcfr-train.log
echo "train exit ${{PIPESTATUS[0]}}"
# Host aws-cli is now botocore-incompatible -> upload logs via boto3.
python3 - <<'PYEOF'
import boto3
s3 = boto3.client("s3", region_name="{region}")
for src, key in [("/var/log/cloud-init-output.log", "{run_prefix}/logs/cloud-init-output.log"),
                 ("/var/log/sdcfr-train.log", "{run_prefix}/logs/train.log")]:
    try:
        s3.upload_file(src, "{bucket}", key); print("uploaded", key)
    except Exception as e:
        print("log upload fail", src, e)
PYEOF
{final}
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instance-type", default="c5a.2xlarge", help="c5a.2xlarge=validate, c5a.16xlarge=full")
    p.add_argument("--arch", default="x86_64", choices=["x86_64", "arm64"])
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--traversals", type=int, default=12)
    p.add_argument("--deal-sweeps", type=int, default=1)
    p.add_argument("--max-depth", type=int, default=32)
    p.add_argument("--bet-buckets", default=None, help="action abstraction indices e.g. '1,4' (omit=exact)")
    p.add_argument("--postflop-buckets", type=int, default=None, help="card abstraction: N postflop buckets (omit=off)")
    p.add_argument("--betting-abstraction", action="store_true", help="bounded betting-state key (converges)")
    p.add_argument("--preflop-lossless", action="store_true",
                   help="lossless preflop CARDS: 169 canonical class x position (no WEVA k-means; SMALLER "
                        "than the 10k-WEVA blueprint). Fixes the proven +180 card-merge leak. From-scratch.")
    p.add_argument("--depth-buckets", action="store_true",
                   help="A2: key an effective-starting-depth bucket into the PREFLOP key (depth-aware "
                        "blueprint); serve must also set POKERBOT_DEPTH_BUCKETS=1")
    p.add_argument("--depth-bb", default=None,
                   help="A3: randomize each deal's starting stack over LO,HI bb (e.g. '40,200', log-uniform). "
                        "Pair with --depth-buckets for the depth retrain.")
    p.add_argument("--occupancy-mix", default=None,
                   help="C1: randomize each deal's live-seat count over LO,HI (e.g. '2,6', weighted toward "
                        "fuller tables) so the blueprint prior covers short-handed/HU tables. Requires "
                        "--occupancy 6max. No key change; the engine/search already play any seat count.")
    p.add_argument("--preflop-lossless-betting", action="store_true",
                   help="ALSO make preflop BETTING lossless (Pluribus-exact). Explodes the infoset count "
                        "(~10^7-10^8) -> needs a big --hogwild-capacity + more iters. Opt-in; default OFF.")
    p.add_argument("--client-bet-sizing", action="store_true", help="CoinPoker client-matched street-aware sizing")
    p.add_argument("--trainer", choices=["pool", "hogwild"], default="pool",
                   help="pool=bulk-sync reference; hogwild=shared-memory lock-free (H5 many-core scaling)")
    p.add_argument("--hogwild-capacity", type=int, default=None, help="hogwild shared-table slot capacity")
    p.add_argument("--occupancy", choices=["hu", "6max"], default="hu",
                   help="table layout: hu (default) or 6max (Pluribus foundation target)")
    p.add_argument("--cfr-variant", choices=["cfr_plus", "dcfr"], default="cfr_plus",
                   help="cfr_plus (default) or dcfr (Discounted CFR); hogwild only")
    p.add_argument("--vr-baseline", action="store_true", help="VR-MCCFR baseline variance reduction (hogwild)")
    p.add_argument("--run-subdir", default=None, help="S3 subdir for this run (default: run_<ts>)")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--instance-profile", default=DEFAULT_PROFILE)
    p.add_argument("--on-demand", action="store_true", help="use on-demand instead of spot")
    p.add_argument("--no-sft", action="store_true", help="skip torch+SFT (faster validation)")
    p.add_argument("--keep-alive", action="store_true",
                   help="don't shut down on finish (keep box for get-console-output inspection)")
    p.add_argument("--run-subdir-ts", type=int, default=0, help="timestamp for run subdir (pass from caller)")
    args = p.parse_args()

    # Depth-retrain coherence guards (a mis-launch silently trains a wasted/incoherent table over hours):
    #   * --depth-buckets keys the depth byte ONLY inside the betting-abstraction digest branch, so without
    #     --betting-abstraction the byte is dropped at train time while the stamp still claims depth_bucketed.
    #   * --depth-buckets and --depth-bb must come together: keying depth while every deal is fixed-100bb
    #     fills one bucket; randomizing stacks without keying them folds incoherent ranges into one row.
    if args.depth_buckets and not args.betting_abstraction:
        p.error("--depth-buckets requires --betting-abstraction (the depth byte is only hashed in the "
                "betting-abstraction digest branch; otherwise it is silently dropped).")
    if bool(args.depth_buckets) != bool(args.depth_bb):
        p.error("--depth-buckets and --depth-bb must be used together (key + randomized deals); "
                f"got depth_buckets={args.depth_buckets!r} depth_bb={args.depth_bb!r}.")
    if args.occupancy_mix and args.occupancy != "6max":
        p.error("--occupancy-mix requires --occupancy 6max (it samples a subset of the 6 ring seats per "
                f"deal); got --occupancy {args.occupancy}.")

    s3 = boto3.client("s3", region_name=args.region)
    ec2 = boto3.client("ec2", region_name=args.region)

    run_subdir = args.run_subdir or f"run_{args.run_subdir_ts or int(time.time())}"
    code_key = build_and_upload_code(s3, args.bucket, args.prefix)
    ami = resolve_ubuntu_ami(ec2, args.arch)
    wheels_dir = "wheels_py310_x86" if args.arch == "x86_64" else "wheels_py310_arm64"
    print(f"AMI: {ami}  instance: {args.instance_type}  arch: {args.arch}  region: {args.region}", flush=True)

    ud = user_data(
        bucket=args.bucket, prefix=args.prefix, region=args.region, code_key=code_key,
        run_subdir=run_subdir, workers=args.workers, iterations=args.iterations,
        traversals=args.traversals, deal_sweeps=args.deal_sweeps, max_depth=args.max_depth,
        no_sft=args.no_sft, wheels_dir=wheels_dir, keep_alive=args.keep_alive,
        bet_buckets=args.bet_buckets, postflop_buckets=args.postflop_buckets,
        betting_abstraction=args.betting_abstraction, client_bet_sizing=args.client_bet_sizing,
        trainer=args.trainer, hogwild_capacity=args.hogwild_capacity, arch=args.arch,
        occupancy=args.occupancy, cfr_variant=args.cfr_variant, vr_baseline=args.vr_baseline,
        preflop_lossless=args.preflop_lossless, preflop_lossless_betting=args.preflop_lossless_betting,
        depth_buckets=args.depth_buckets, depth_bb=args.depth_bb,
        occupancy_mix=args.occupancy_mix,
    )
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
                {"Key": "Name", "Value": f"sd-cfr-hu-train-{run_subdir}"},
                {"Key": "Project", "Value": "pokerbot-sd-cfr"},
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
    print(f"  checkpoints: .../{run_subdir}/iter_*/  | logs: .../{run_subdir}/logs/", flush=True)
    print(f"Monitor console: aws ec2 get-console-output --instance-id {iid} --region {args.region} --latest", flush=True)
    print(f"Terminate: aws ec2 terminate-instances --instance-ids {iid} --region {args.region}", flush=True)


if __name__ == "__main__":
    main()
