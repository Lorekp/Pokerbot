#!/usr/bin/env python3
"""Launch ONE x86_64 c6a spot box to run the multi-seed native-CFV LBR-equivalence gate.

The native CFV-under-exploitation port (POKERBOT_NATIVE_CFV) makes the Rust CFR driver reachable under
exploitation by carrying the firewalled exploit tilt AS DATA. The relaxable changes (native xoshiro RNG vs
numpy PCG64, native float reduction order) must be certified LBR/EV-equivalent before the path can carry a
live decision. That cert is HEAVY (multi-seed LBR + a per-decision sweep) and the user's PC is fragile, so
it runs on a cloud box like the other eval runs (bucket video-processing-6789, region eu-central-1).

This box (x86_64, so we BUILD the native crate from source with cargo + the python feature):
  1. bootstrap python3.10 + the existing x86 wheels (numpy/jax/torch) + boto3
  2. install rustup + cargo, build tools/battery_native --release --features python -> battery_native.so
  3. run the i64 infoset-key parity fuzz (tests/test_infoset_key_i64_parity.py) — the #1-risk gate
  4. run scripts/native_cfv_lbr_equivalence.py across N seeds (ON vs OFF) + the direct per-decision sweep
  5. upload the JSON result + logs to S3 and terminate

Run from the user's terminal (their AWS creds = the labelgen-worker user):
  python scripts/launch_native_cfv_lbr_spot.py --run-subdir run_nativecfv1 --seeds 8 --hands 300
"""
from __future__ import annotations

import argparse
import base64
import io
import tarfile
from botocore.exceptions import ClientError

# R4 HIGH 4: capacity-error allow-list (the only ClientError codes that should
# advance down the fallback chain). Any other ClientError is a real
# configuration problem (VPC, IAM, key pair, malformed user_data) and must
# surface immediately so we don't burn retries on a misconfigured launch.
_CAPACITY_ERROR_CODES = frozenset(
    {
        "InsufficientInstanceCapacity",
        "SpotMaxPriceTooLow",
        "InstanceLimitExceeded",
        "MaxSpotInstanceCountExceeded",
        "RequestLimitExceeded",
        "Unsupported",
    }
)
import time
from pathlib import Path

import boto3

from launch_sd_cfr_training_spot import resolve_ubuntu_ami

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUCKET = "video-processing-6789"
DEFAULT_PREFIX = "pokerbot/prod6000/native_cfv_lbr"
DEFAULT_REGION = "eu-central-1"
DEFAULT_PROFILE = "pokerbot-labelgen-worker"
LOCAL_BP = Path(r"checkpoints/bp6max_v2_iter200.pkl")

# The native crate must build on the box; ship the whole crate (src + Cargo.{toml,lock}). The pokerbot
# package + the runner + the parity test + the blueprint data round it out.
_INCLUDE = [
    "pokerbot",
    "scripts/native_cfv_lbr_equivalence.py",
    "scripts/native_cfv_profile.py",
    "tests/test_infoset_key_i64_parity.py",
    "tools/battery_native/Cargo.toml",
    "tools/battery_native/Cargo.lock",
    "tools/battery_native/src",
    "data/weva",
    "data/player_profiles/irc_population_prior.json",
    "data/player_profiles/irc_population_prior_fitted.json",
    "data/player_profiles/irc_population_percentiles.json",
    "data/player_profiles/irc_weak_mixture.json",
    "checkpoints/policy_sft_prod6000.pt",
]


def _tar_filter(ti: tarfile.TarInfo):
    parts = ti.name.split("/")
    if any(p == "__pycache__" for p in parts) or ti.name.endswith(".pyc"):
        return None
    if "target" in parts:  # never ship the local Rust build dir
        return None
    return ti


def build_and_upload_code(s3, bucket, prefix) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in _INCLUDE:
            p = ROOT / rel
            if p.exists():
                tar.add(p, arcname=rel, filter=_tar_filter)
            else:
                print(f"WARNING: {rel} missing from tarball")
    buf.seek(0)
    key = f"{prefix}/code/native_cfv_code.tar.gz"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    print(f"Uploaded code tarball -> s3://{bucket}/{key}", flush=True)
    return key


def upload_blueprint(s3, bucket, prefix, local_bp: Path) -> str:
    key = f"{prefix}/blueprint/{local_bp.stem}.pkl"
    size = local_bp.stat().st_size
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        if head["ContentLength"] == size:
            print(f"Blueprint already in S3 ({size} bytes) — skip upload", flush=True)
            return key
    except Exception:  # noqa: BLE001
        pass
    print(f"Uploading blueprint ({size / 1e6:.0f} MB)...", flush=True)
    s3.upload_file(str(local_bp), bucket, key)
    return key


def user_data(*, bucket, prefix, region, code_key, bp_key, run_subdir, seeds, hands, warmup, menu,
              search_iters, direct_nodes, profile_nodes, profile_iters, profile_budget_ms,
              keep_alive=False) -> str:
    run_prefix = f"{prefix}/{run_subdir}"
    final = "echo BOOTSTRAP_DONE_KEEP_ALIVE" if keep_alive else "shutdown -h now"
    result_key = f"{run_prefix}/results/native_cfv_lbr_result.json"
    profile_key = f"{run_prefix}/results/native_cfv_profile.json"
    return f"""#!/bin/bash
set -uxo pipefail
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION={region}
apt-get update -qq
apt-get install -y -qq python3-pip awscli gzip curl build-essential pkg-config
mkdir -p /wheels /opt/pokerbot
# aws-cli v1 zone first (the pip loop below breaks its botocore).
aws s3 cp s3://{bucket}/pokerbot/prod6000/sd_cfr_hu/wheels_py310_x86/ /wheels/ --recursive --region {region} --no-progress
aws s3 cp s3://{bucket}/{code_key} /tmp/code.tar.gz --region {region}
aws s3 cp s3://{bucket}/{bp_key} /tmp/bp.pkl --region {region}
tar -xzf /tmp/code.tar.gz -C /opt/pokerbot
PIPI="python3 -m pip install --no-index --no-deps --disable-pip-version-check --no-input"
for w in /wheels/*.whl; do echo "+++ $(basename $w)"; $PIPI "$w" || echo "WHEEL_FAIL $(basename $w)"; done
python3 -c "import torch" 2>/dev/null || python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python3 -m pip install phevaluator 2>/dev/null || echo "phevaluator unavailable (slower all-in EV)"
python3 -c "import numpy, boto3; print('IMPORTS OK')"
# ---- Rust toolchain + native crate build (x86_64) ----
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"
export PYO3_PYTHON=$(which python3)
cd /opt/pokerbot/tools/battery_native
cargo build --release --features python 2>&1 | tail -8
# cdylib output is lib*.so on Linux; alias to the import name battery_native.so on sys.path.
cp -v target/release/libbattery_native.so target/release/battery_native.so || true
ls -la target/release/*.so
cd /opt/pokerbot
export PYTHONPATH=/opt/pokerbot:/opt/pokerbot/tools/battery_native/target/release JAX_PLATFORMS=cpu
export POKERBOT_PREFLOP_LOSSLESS=1 POKERBOT_SEARCH_PREFLOP=1
export POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_POSTFLOP_MODE=fast POKERBOT_POSTFLOP_NUM_BUCKETS=200
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p /tmp/jax_cache
# Confirm the native SubgameSolver is importable + self-checked BEFORE the gate (else the run is vacuous).
python3 - <<'PYEOF'
from pokerbot.search import native_solver as ns
print("native available:", ns.available(), "subgame_available:", ns.native_subgame_available())
assert ns.native_subgame_available(), "NATIVE SUBGAME NOT AVAILABLE — build failed, gate would be vacuous"
PYEOF
echo "=== KEY-PARITY FUZZ (the #1-risk gate) ==="
python3 -m pytest tests/test_infoset_key_i64_parity.py -x -q 2>&1 | tee /var/log/parity.log
echo "=== MULTI-SEED LBR EQUIVALENCE GATE ==="
python3 scripts/native_cfv_lbr_equivalence.py --seeds {seeds} --hands {hands} --warmup {warmup} \
    --menu {menu} --search-iters {search_iters} --direct-nodes {direct_nodes} \
    --blueprint-snapshot /tmp/bp.pkl --out /tmp/native_cfv_lbr_result.json \
    --s3-bucket {bucket} --s3-key {result_key} --region {region} 2>&1 | tee /var/log/native-cfv-lbr.log
echo "gate exit ${{PIPESTATUS[0]}}"
echo "=== PER-DECISION LATENCY PROFILE (p50/p95 ON vs OFF + component split) ==="
python3 scripts/native_cfv_profile.py --nodes {profile_nodes} --search-iters {profile_iters} \
    --budget-ms {profile_budget_ms} --blueprint-snapshot /tmp/bp.pkl \
    --out /tmp/native_cfv_profile.json \
    --s3-bucket {bucket} --s3-key {profile_key} --region {region} 2>&1 | tee /var/log/native-cfv-profile.log
echo "profile exit ${{PIPESTATUS[0]}}"
python3 - <<'PYEOF'
import boto3
s3 = boto3.client("s3", region_name="{region}")
for src, key in [("/var/log/cloud-init-output.log", "{run_prefix}/logs/cloud-init-output.log"),
                 ("/var/log/native-cfv-lbr.log", "{run_prefix}/logs/native-cfv-lbr.log"),
                 ("/var/log/native-cfv-profile.log", "{run_prefix}/logs/native-cfv-profile.log"),
                 ("/var/log/parity.log", "{run_prefix}/logs/parity.log")]:
    try:
        s3.upload_file(src, "{bucket}", key); print("uploaded", key)
    except Exception as e:
        print("log upload fail", src, e)
PYEOF
{final}
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instance-type", default=None,
                   help="default: spotpick c6a chain (c6a.4xlarge first)")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--hands", type=int, default=300)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--menu", choices=("fc", "fcpa", "full"), default="fc")
    p.add_argument("--search-iters", type=int, default=10)
    p.add_argument("--direct-nodes", type=int, default=600)
    p.add_argument("--profile-nodes", type=int, default=600,
                   help="postflop nodes for the per-decision latency profile")
    p.add_argument("--profile-iters", type=int, default=200,
                   help="CFR iterations for the profile (200 = the live converged target)")
    p.add_argument("--profile-budget-ms", type=int, default=800,
                   help="time_budget_ms for the profile (live budget; bounds the interpreted OFF arm)")
    p.add_argument("--run-subdir", default="run_nativecfv1")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--instance-profile", default=DEFAULT_PROFILE)
    p.add_argument("--on-demand", action="store_true")
    p.add_argument("--keep-alive", action="store_true")
    p.add_argument("--local-bp", default=None)
    args = p.parse_args()

    # spotpick capacity-diverse c6a-first chain (16 vCPU x86_64). Try each type until one launches.
    chain = []
    if args.instance_type:
        chain = [args.instance_type]
    else:
        try:
            import sys
            sys.path.insert(0, r".")
            import spotpick
            # prefer c6a family but allow the diverse x86 16-vCPU fallback chain.
            c6a = spotpick.chain(16, "x86_64", n=1, families=None, exclude_families=None)
        except Exception:  # noqa: BLE001
            c6a = []
        # Force c6a.4xlarge first per the directive, then the diverse fallback.
        chain = ["c6a.4xlarge"]
        try:
            sys.path.insert(0, r".")
            import spotpick
            for t in spotpick.chain(16, "x86_64", n=5):
                if t not in chain:
                    chain.append(t)
        except Exception:  # noqa: BLE001
            chain += ["c5a.4xlarge", "m6a.4xlarge", "c5.4xlarge"]
    print("instance chain:", chain, flush=True)

    s3 = boto3.client("s3", region_name=args.region)
    ec2 = boto3.client("ec2", region_name=args.region)
    code_key = build_and_upload_code(s3, args.bucket, args.prefix)
    bp_key = upload_blueprint(s3, args.bucket, args.prefix,
                              Path(args.local_bp) if args.local_bp else LOCAL_BP)
    ami = resolve_ubuntu_ami(ec2, "x86_64")
    print(f"AMI: {ami}  region: {args.region}", flush=True)

    ud = user_data(bucket=args.bucket, prefix=args.prefix, region=args.region, code_key=code_key,
                   bp_key=bp_key, run_subdir=args.run_subdir, seeds=args.seeds, hands=args.hands,
                   warmup=args.warmup, menu=args.menu, search_iters=args.search_iters,
                   direct_nodes=args.direct_nodes, profile_nodes=args.profile_nodes,
                   profile_iters=args.profile_iters, profile_budget_ms=args.profile_budget_ms,
                   keep_alive=args.keep_alive)
    ud_b64 = base64.b64encode(ud.encode()).decode()

    last_err = None
    for itype in chain:
        run_kwargs = dict(
            ImageId=ami, InstanceType=itype, MinCount=1, MaxCount=1,
            UserData=ud_b64, IamInstanceProfile={"Name": args.instance_profile},
            MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
            InstanceInitiatedShutdownBehavior="terminate",
            TagSpecifications=[{"ResourceType": "instance",
                                "Tags": [{"Key": "Name", "Value": f"pokerbot-native-cfv-lbr-{args.run_subdir}"},
                                         {"Key": "Project", "Value": "pokerbot-sd-cfr"},
                                         {"Key": "Run", "Value": args.run_subdir}]}],
            BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                                  "Ebs": {"VolumeSize": 40, "VolumeType": "gp3",
                                          "DeleteOnTermination": True}}],
        )
        if not args.on_demand:
            run_kwargs["InstanceMarketOptions"] = {"MarketType": "spot",
                                                   "SpotOptions": {"SpotInstanceType": "one-time"}}
        try:
            inst = ec2.run_instances(**run_kwargs)["Instances"][0]
        except ClientError as e:  # R4 HIGH 4: capacity errors advance, everything else re-raises
            err_code = e.response.get("Error", {}).get("Code", "")
            if err_code in _CAPACITY_ERROR_CODES:
                print(f"  {itype} capacity error: {err_code} — trying next type", flush=True)
                last_err = e
                continue
            raise
        except Exception as e:  # noqa: BLE001
            # Real boto / network / param / KeyError: never treat as capacity
            print(f"  {itype} FATAL: {type(e).__name__}: {e}", flush=True)
            raise
        iid = inst["InstanceId"]
        print(f"LAUNCHED {iid} as {itype} ({'on-demand' if args.on_demand else 'spot'})")
        print(f"console: aws ec2 get-console-output --instance-id {iid} --region {args.region} --output text")
        print(f"result : aws s3 cp s3://{args.bucket}/{args.prefix}/{args.run_subdir}/results/native_cfv_lbr_result.json - --region {args.region}")
        print(f"profile: aws s3 cp s3://{args.bucket}/{args.prefix}/{args.run_subdir}/results/native_cfv_profile.json - --region {args.region}")
        print(f"iid={iid}")
        return
    raise SystemExit(f"all instance types failed; last error: {last_err}")


if __name__ == "__main__":
    main()
