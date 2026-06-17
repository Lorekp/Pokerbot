#!/usr/bin/env python3
"""List S3 buckets (AWS credential smoke test)."""
import boto3

s3 = boto3.client("s3")
buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
print("buckets:", buckets[:10])
