#!/usr/bin/env python3
"""Launch EC2 spot workers for label generation (scaffold — validate locally first)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Launch labelgen spot fleet")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--budget-usd", type=float, default=10.0)
    p.add_argument("--allow-on-demand", action="store_true")
    p.add_argument("--max-instances", type=int, default=1)
    p.add_argument("--docker-image", type=str, required=True)
    p.add_argument("--s3-bucket", type=str, default=None)
    args = p.parse_args()

    if not args.s3_bucket:
        print(
            json.dumps(
                {
                    "status": "not_launched",
                    "reason": "S3 bucket required for cloud launch; run 50-label local-docker validation first",
                    "manifest": str(args.manifest),
                    "budget_usd": args.budget_usd,
                    "spot_only": not args.allow_on-demand,
                },
                indent=2,
            )
        )
        sys.exit(2)

    # AWS launch wiring is environment-specific; worker uses cloud_label_worker.py in Docker.
    print(
        json.dumps(
            {
                "status": "scaffold_only",
                "message": "Configure AWS credentials and Terraform under infra/labelgen/ before fleet launch",
                "manifest": str(args.manifest),
                "max_instances": args.max_instances,
                "docker_image": args.docker_image,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
