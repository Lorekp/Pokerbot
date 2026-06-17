#!/usr/bin/env python3
import boto3
from botocore.exceptions import ClientError

bucket = "video-processing-6789"
prefix = "pokerbot/prod6000/validate50_ec2"
region = "eu-central-1"

s3 = boto3.client("s3", region_name=region)
for key in ("cloud/manifest.json", "image_manifest.json", "work_manifest.json"):
    s3_key = f"{prefix}/{key}"
    try:
        s3.head_object(Bucket=bucket, Key=s3_key)
        print(f"EXISTS s3://{bucket}/{s3_key}")
    except ClientError as e:
        print(f"MISSING s3://{bucket}/{s3_key} ({e.response['Error']['Code']})")

ec2 = boto3.client("ec2", region_name=region)
resp = ec2.describe_instances(
    Filters=[
        {"Name": "tag:Name", "Values": ["pokerbot-validate50-ec2-gate"]},
    ]
)
found = False
for res in resp.get("Reservations", []):
    for inst in res["Instances"]:
        found = True
        print(
            "INSTANCE",
            inst["InstanceId"],
            inst["State"]["Name"],
            str(inst.get("LaunchTime", "")),
        )
        from datetime import datetime, timezone

        launch = inst["LaunchTime"]
        elapsed_min = (datetime.now(timezone.utc) - launch).total_seconds() / 60
        print(f"  elapsed_min={elapsed_min:.0f}")
if not found:
    print("No validate50-tagged EC2 instances")
