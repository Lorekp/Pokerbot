#!/usr/bin/env python3
"""Launch ONE arm64 spot box to run the full exploitation-eval suite (run_cloud_eval_suite.py).

Mirrors the WORKING training launcher (launch_sd_cfr_training_spot.py): same account/role/region,
same wheels bootstrap (no exec-redirect — serial console stays readable), same spot mechanics.
Eval extras: ships the eval code + fitted prior/mixture data, uploads the local blueprint snapshot
(65 MB, skipped if already in S3), pip-fallbacks torch (SearchAgent import) + phevaluator (all-in
EV fast path) when not in the wheels dir.

Run from the user's terminal (their AWS creds):
  python scripts/launch_eval_suite_spot.py --run-subdir run_evalsuite1
"""

from __future__ import annotations

import argparse
import base64
import io
import tarfile
import time
from pathlib import Path

import boto3

from launch_sd_cfr_training_spot import resolve_ubuntu_ami  # same dir; proven helper

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUCKET = "video-processing-6789"
DEFAULT_PREFIX = "pokerbot/prod6000/eval_suite"   # within the IAM-allowed prod6000/* prefix
DEFAULT_REGION = "eu-central-1"
DEFAULT_PROFILE = "pokerbot-labelgen-worker"
LOCAL_BP = Path(r"checkpoints/bp6max_v2_iter200.pkl")

_INCLUDE = [
    "pokerbot",
    "scripts/eval_exploitation_battery.py",
    "scripts/run_cloud_eval_suite.py",
    "scripts/summarize_exploit_battery.py",
    "data/weva",
    "data/player_profiles/irc_population_prior.json",
    "data/player_profiles/irc_population_prior_fitted.json",
    "data/player_profiles/irc_population_percentiles.json",
    "data/player_profiles/irc_weak_mixture.json",
    "data/player_profiles/nlhe_skill_tiers.json",   # skill-tier defaults (overwritten by --mode calibrate)
    "scripts/calibrate_nlhe_tiers.py",              # --mode calibrate driver
    "checkpoints/policy_sft_prod6000.pt",   # HybridAgent base for --mode hybrid / calibrate (~1.3 MB)
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
            else:
                print(f"WARNING: {rel} missing from tarball")
    buf.seek(0)
    key = f"{prefix}/code/eval_suite_code.tar.gz"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    print(f"Uploaded code tarball -> s3://{bucket}/{key}", flush=True)
    return key


def upload_blueprint(s3, bucket: str, prefix: str) -> str:
    key = f"{prefix}/blueprint/bp6max_v2_iter200.pkl"
    size = LOCAL_BP.stat().st_size
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        if head["ContentLength"] == size:
            print(f"Blueprint already in S3 ({size} bytes) — skip upload", flush=True)
            return key
    except Exception:  # noqa: BLE001
        pass
    print(f"Uploading blueprint ({size / 1e6:.0f} MB)...", flush=True)
    s3.upload_file(str(LOCAL_BP), bucket, key)
    return key


def user_data(*, bucket, prefix, region, code_key, bp_key, run_subdir, workers, seeds, hands,
              mode="full", keep_alive=False, bs_configs="10:uniform", bs_skip_sft=False,
              bs_pooled_profiles=False) -> str:
    run_prefix = f"{prefix}/{run_subdir}"
    final = "echo BOOTSTRAP_DONE_KEEP_ALIVE" if keep_alive else "shutdown -h now"
    if mode == "calibrate":
        # Skill-tier strength ladder: the calibrate driver fans (tier,seed) across a process pool and
        # self-uploads the fitted JSON + calibrate_DONE.marker to S3 (the babysitter polls the marker).
        main_cmd = (
            f"python3 scripts/calibrate_nlhe_tiers.py --fit --allin-ev "
            f"--workers {workers} --seeds 0-{seeds - 1} --hands {hands} "
            f"--checkpoint checkpoints/policy_sft_prod6000.pt "
            f"--s3-bucket {bucket} --s3-prefix {run_prefix} --region {region} "
            f"2>&1 | tee /var/log/eval-suite.log")
    else:
        bs_flags = ""
        if mode == "basestrength":
            bs_flags = (f"--bs-pluribus-configs {bs_configs} "
                        + ("--bs-skip-sft " if bs_skip_sft else "")
                        + ("--bs-pooled-profiles " if bs_pooled_profiles else ""))
        main_cmd = (
            f"python3 scripts/run_cloud_eval_suite.py --bucket {bucket} --prefix {run_prefix} "
            f"--region {region} --workers {workers} --seeds {seeds} --hands {hands} "
            f"--bp /tmp/bp6max_v2_iter200.pkl --mode {mode} "
            f"--checkpoint checkpoints/policy_sft_prod6000.pt "
            f"{bs_flags}"
            f"2>&1 | tee /var/log/eval-suite.log")
    return f"""#!/bin/bash
set -uxo pipefail
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION={region}
apt-get update -qq
apt-get install -y -qq python3-pip awscli gzip curl
mkdir -p /wheels /opt/pokerbot
# aws-cli v1 zone (pip loop below breaks its botocore): ALL downloads happen here.
aws s3 cp s3://{bucket}/pokerbot/prod6000/sd_cfr_hu/wheels_py310_arm64/ /wheels/ --recursive --region {region} --no-progress
aws s3 cp s3://{bucket}/{code_key} /tmp/code.tar.gz --region {region}
aws s3 cp s3://{bucket}/{bp_key} /tmp/bp6max_v2_iter200.pkl --region {region}
tar -xzf /tmp/code.tar.gz -C /opt/pokerbot
ls -la /opt/pokerbot/scripts/run_cloud_eval_suite.py /tmp/bp6max_v2_iter200.pkl
PIPI="python3 -m pip install --no-index --no-deps --disable-pip-version-check --no-input"
for w in /wheels/*.whl; do echo "+++ $(basename $w)"; $PIPI "$w" || echo "WHEEL_FAIL $(basename $w)"; done
# Eval extras with PyPI fallback: torch (SearchAgent import chain) + phevaluator (all-in EV fast path).
python3 -c "import torch" 2>/dev/null || python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python3 -m pip install phevaluator 2>/dev/null || echo "phevaluator unavailable (compare_hands fallback, slower)"
python3 -c "import numpy, torch, boto3; print('IMPORTS OK')"
cd /opt/pokerbot
export PYTHONPATH=/opt/pokerbot JAX_PLATFORMS=cpu POKERBOT_BETTING_ABSTRACTION=1
# Blueprint serve keys must match the trained table postflop (the encoder bakes these at import). Without
# them the pluribus hero gets ~0% postflop blueprint coverage (silent uniform fallback). Net is invariant.
export POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_POSTFLOP_MODE=fast POKERBOT_POSTFLOP_NUM_BUCKETS=200
export POKERBOT_BLUEPRINT_SNAPSHOT=/tmp/bp6max_v2_iter200.pkl
# Thin single-thread workers (the many-process lesson from training: no nested threadpools).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NPY_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
{main_cmd}
echo "suite exit ${{PIPESTATUS[0]}}"
python3 - <<'PYEOF'
import boto3
s3 = boto3.client("s3", region_name="{region}")
for src, key in [("/var/log/cloud-init-output.log", "{run_prefix}/logs/cloud-init-output.log"),
                 ("/var/log/eval-suite.log", "{run_prefix}/logs/eval-suite.log")]:
    try:
        s3.upload_file(src, "{bucket}", key); print("uploaded", key)
    except Exception as e:
        print("log upload fail", src, e)
PYEOF
{final}
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instance-type", default="m7g.16xlarge")
    p.add_argument("--workers", type=int, default=58)
    p.add_argument("--seeds", type=int, default=18)
    p.add_argument("--hands", type=int, default=3000)
    p.add_argument("--run-subdir", default="run_evalsuite1")
    p.add_argument("--mode", choices=("full", "confirm", "hybrid", "calibrate", "basestrength"),
                   default="full")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--instance-profile", default=DEFAULT_PROFILE)
    p.add_argument("--on-demand", action="store_true")
    p.add_argument("--keep-alive", action="store_true")
    p.add_argument("--bs-configs", default="10:uniform",
                   help="--mode basestrength: pluribus strong-config sweep '<iters>:<leaf>,...' "
                        "(e.g. 10:uniform,50:gadget,150:gadget)")
    p.add_argument("--bs-skip-sft", action="store_true",
                   help="--mode basestrength: skip the SFT base (already measured)")
    p.add_argument("--bs-pooled-profiles", action="store_true",
                   help="--mode basestrength: legacy ONE-pooled-profile exploit on mixed cells "
                        "(default is per-villain targeting)")
    args = p.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    ec2 = boto3.client("ec2", region_name=args.region)
    code_key = build_and_upload_code(s3, args.bucket, args.prefix)
    bp_key = upload_blueprint(s3, args.bucket, args.prefix)
    ami = resolve_ubuntu_ami(ec2, "arm64")
    print(f"AMI: {ami}  instance: {args.instance_type}  region: {args.region}", flush=True)

    ud = user_data(bucket=args.bucket, prefix=args.prefix, region=args.region, code_key=code_key,
                   bp_key=bp_key, run_subdir=args.run_subdir, workers=args.workers,
                   seeds=args.seeds, hands=args.hands, mode=args.mode, keep_alive=args.keep_alive,
                   bs_configs=args.bs_configs, bs_skip_sft=args.bs_skip_sft,
                   bs_pooled_profiles=args.bs_pooled_profiles)
    run_kwargs = dict(
        ImageId=ami,
        InstanceType=args.instance_type,
        MinCount=1, MaxCount=1,
        UserData=base64.b64encode(ud.encode()).decode(),
        IamInstanceProfile={"Name": args.instance_profile},
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
        InstanceInitiatedShutdownBehavior="terminate",
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": f"pokerbot-eval-suite-{args.run_subdir}"},
                     {"Key": "Project", "Value": "pokerbot-sd-cfr"},
                     {"Key": "Run", "Value": args.run_subdir}],
        }],
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": 64, "VolumeType": "gp3",
                                      "DeleteOnTermination": True}}],
    )
    if not args.on_demand:
        run_kwargs["InstanceMarketOptions"] = {"MarketType": "spot",
                                               "SpotOptions": {"SpotInstanceType": "one-time"}}
    inst = ec2.run_instances(**run_kwargs)["Instances"][0]
    print(f"LAUNCHED {inst['InstanceId']} ({'on-demand' if args.on_demand else 'spot'})")
    print(f"watch:   aws ec2 get-console-output --instance-id {inst['InstanceId']} --region {args.region} --output text")
    print(f"results: aws s3 ls s3://{args.bucket}/{args.prefix}/{args.run_subdir}/results/ --region {args.region}")


if __name__ == "__main__":
    main()
