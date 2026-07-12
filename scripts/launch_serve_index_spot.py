#!/usr/bin/env python3
"""Launch ONE small spot box that BUILDS THE TIER-1 SERVE INDEX and uploads it to S3.

Serve-flip deployment path (task #71, 2026-07-12): the certified serve object is
snap_avg(iter_126000.nlcfr preflop + NLSNAP01 snapshots 100k..125k postflop) served via the
SortedSnapshotDir mmap index (scripts/nlcfr_key_index.py --with-rows). cert8 built exactly this
on-box (serve_index_meta.json: n_strat=538848758) but the box is gone and only ~141 GB are free
locally (inputs alone are ~183 GB), so the merge+index runs in-region and ONLY the index
(~32 GB keys+rows) comes down to the serve machine.

Mechanics reuse the proven spot plumbing (resolve_ubuntu_ami / user_data_stub dead-man /
_launch_with_fallback) and the byte-identical cert7 code tarball (same snap_avg + index code the
cert runs built with). No gates/battery here — the object was cert7/cert8-gated; the local flip
re-runs serve_miss_probe on the downloaded index before serving.

Cost bound: spot one-time, terminate-on-shutdown, STEP-0 wall guard (default 4 h), poweroff on
completion AND on any failure. Expected ~2-2.5 h on an m6i.4xlarge (~$0.4-0.6 total).
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from launch_native_cfr_training_spot import (  # noqa: E402  (proven helpers, same dir)
    _launch_with_fallback, resolve_ubuntu_ami, user_data_stub,
)

BUCKET = "video-processing-6789"
REGION = "eu-central-1"
PROFILE = "pokerbot-labelgen-worker"
RUN_PREFIX = "pokerbot/prod6000/native_cfr/bpv4_tier1_faithful_1"
CODE_TARBALL = "pokerbot/prod6000/cert_pluribus/cert7/code/cert7_code.tar.gz"
NLCFR_SHA = "abcb00730f4e8671cc8ae5a27db1ac256a4f3d83437005f6cbfb30b372b1ae2f"
# cert8 ground truth for the merged object (artifacts/cert8_results/serve_index_meta.json)
EXPECT_N_STRAT = 538_848_758
EXPECT_ITERS = 126_000
SNAP_ITERS = (100000, 105000, 110000, 115000, 120000, 125000)


def _render_bootstrap() -> str:
    res = f"s3://{BUCKET}/{RUN_PREFIX}/serve_index_build"
    dest = f"s3://{BUCKET}/{RUN_PREFIX}/serve_index"
    snap_urls = " ".join(
        f"s3://{BUCKET}/{RUN_PREFIX}/snapshots/snap_{i:06d}.nlsnap" for i in SNAP_ITERS)
    return f"""#!/bin/bash
set -uxo pipefail
export HOME="${{HOME:-/root}}"
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION={REGION}
OUT=/opt/pokerbot/idx_out; mkdir -p $OUT /opt/pokerbot
fail() {{ echo "$1" > $OUT/FAILED.txt; aws s3 sync $OUT {res}/ --region {REGION} || true; sync; poweroff; exit 1; }}
stage() {{ echo "$(date -u +%s) $1" >> $OUT/stage.txt; }}
stage boot
( while true; do date -u +%s > $OUT/heartbeat.txt; aws s3 sync $OUT {res}/ --region {REGION} --quiet || true; sleep 120; done ) &

stage deps
apt-get update -qq && apt-get install -y -qq python3-pip python3-dev build-essential curl unzip || fail deps
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/aws.zip && unzip -q /tmp/aws.zip -d /tmp && /tmp/aws/install --update && hash -r
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"
python3 -m pip install --disable-pip-version-check --no-input --break-system-packages numpy \\
  || python3 -m pip install --disable-pip-version-check --no-input numpy || fail pip
aws s3 cp s3://{BUCKET}/{CODE_TARBALL} /tmp/code.tar.gz --region {REGION} || fail codefetch
tar -xzf /tmp/code.tar.gz -C /opt/pokerbot
cd /opt/pokerbot

# snap_avg only (no python feature; minutes)
stage build
cd tools/battery_native && cargo build --release --example snap_avg 2>&1 | tail -2 || fail build_snapavg
cd /opt/pokerbot

stage download
aws s3 cp s3://{BUCKET}/{RUN_PREFIX}/checkpoints/iter_126000.nlcfr /opt/pokerbot/iter_126000.nlcfr --region {REGION} || fail nlcfr
echo "{NLCFR_SHA}  /opt/pokerbot/iter_126000.nlcfr" | sha256sum -c - || fail nlcfr_sha
SNAPS=""
for SURL in {snap_urls}; do
  SBASE=$(basename $SURL)
  aws s3 cp $SURL /opt/pokerbot/$SBASE --region {REGION} || fail snapdl
  SNAPS="$SNAPS /opt/pokerbot/$SBASE"
done

stage snap_merge
tools/battery_native/target/release/examples/snap_avg /opt/pokerbot/merged_126k.nlcfr \\
  /opt/pokerbot/iter_126000.nlcfr $SNAPS 2>&1 | tee $OUT/snap_merge.log || fail snap_merge
rm -f /opt/pokerbot/iter_126000.nlcfr /opt/pokerbot/snap_*.nlsnap

stage index
python3 scripts/nlcfr_key_index.py /opt/pokerbot/merged_126k.nlcfr /opt/pokerbot/idx_new_full \\
  --with-rows 2>&1 | tee $OUT/index.log || fail index
rm -f /opt/pokerbot/merged_126k.nlcfr
python3 - <<'PY' || fail meta_check
import json
m = json.load(open('/opt/pokerbot/idx_new_full/meta.json'))
assert m["n_strat"] == {EXPECT_N_STRAT}, m
assert m["iters"] == {EXPECT_ITERS}, m
assert m["with_rows"] is True, m
print("meta OK", m["n_strat"], m["iters"])
PY

stage sha
cd /opt/pokerbot/idx_new_full && sha256sum keys_sorted.bin rows_sorted.f32.bin meta.json > sha256sums.txt || fail sha
cp sha256sums.txt $OUT/sha256sums.txt; cp meta.json $OUT/serve_index_meta.json
cd /opt/pokerbot

stage upload
aws s3 sync /opt/pokerbot/idx_new_full {dest}/ --region {REGION} || fail upload
# read-back size check (multipart sync silently truncating would poison the serve object)
aws s3 ls {dest}/ --region {REGION} | tee $OUT/s3_listing.txt

stage done
date -u +"%Y-%m-%dT%H:%M:%SZ" > $OUT/RESULT_DONE.txt
aws s3 sync $OUT {res}/ --region {REGION}
sync
poweroff
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall-hours", type=float, default=4.0)
    ap.add_argument("--instance-types",
                    default="m6i.4xlarge,m5.4xlarge,c6i.8xlarge,r6i.2xlarge")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    session = boto3.Session(region_name=REGION)
    s3 = session.client("s3")
    ec2 = session.client("ec2")

    bootstrap = _render_bootstrap()
    bkey = f"{RUN_PREFIX}/serve_index_build/code/bootstrap.sh"
    s3.put_object(Bucket=BUCKET, Key=bkey, Body=bootstrap.encode())
    print(f"uploaded bootstrap ({len(bootstrap)} bytes) -> s3://{BUCKET}/{bkey}", flush=True)
    if args.dry_run:
        print("--dry-run: not launching"); return

    ud = user_data_stub(bucket=BUCKET, region=REGION,
                        run_prefix=f"{RUN_PREFIX}/serve_index_build",
                        bootstrap_key=bkey, wall_guard_secs=int(args.wall_hours * 3600))
    ami, root_dev = resolve_ubuntu_ami(ec2, "x86_64")
    run_kwargs = dict(
        ImageId=ami, MinCount=1, MaxCount=1,
        UserData=base64.b64encode(ud.encode()).decode(),
        IamInstanceProfile={"Name": PROFILE},
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
        TagSpecifications=[{"ResourceType": "instance", "Tags": [
            {"Key": "Name", "Value": "tier1-serve-index-build"},
            {"Key": "Project", "Value": "pokerbot-serve-flip"},
        ]}],
        InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": root_dev,
                              "Ebs": {"VolumeSize": 300, "VolumeType": "gp3", "Iops": 3000,
                                      "Throughput": 500, "DeleteOnTermination": True}}],
        InstanceMarketOptions={"MarketType": "spot",
                               "SpotOptions": {"SpotInstanceType": "one-time",
                                               "InstanceInterruptionBehavior": "terminate"}},
    )
    chain = [t.strip() for t in args.instance_types.split(",") if t.strip()]
    iid, used = _launch_with_fallback(ec2, run_kwargs, instance_types=chain, spot=True)
    print(f"LAUNCHED {iid} as {used} (spot)")
    print(f"progress: s3://{BUCKET}/{RUN_PREFIX}/serve_index_build/")
    print(f"index:    s3://{BUCKET}/{RUN_PREFIX}/serve_index/")
    print(f"terminate: aws ec2 terminate-instances --instance-ids {iid} --region {REGION}")


if __name__ == "__main__":
    main()
