"""Unit tests for S3 claim coordination (mocked S3 — no AWS)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from pokerbot.solver.s3_coordination import S3Coordinator


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "PutObject")


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}

    def put_object(self, **kwargs) -> dict:
        key = kwargs["Key"]
        if_none = kwargs.get("IfNoneMatch")
        if_match = kwargs.get("IfMatch")
        etag = self.objects[key]["etag"] if key in self.objects else None

        if if_none == "*" and key in self.objects:
            raise _client_error("PreconditionFailed")
        if if_match is not None:
            if etag is None or if_match.strip('"') != etag.strip('"'):
                raise _client_error("PreconditionFailed")

        new_etag = f"etag-{len(self.objects) + 1}"
        self.objects[key] = {
            "body": kwargs["Body"],
            "etag": new_etag,
            "content_type": kwargs.get("ContentType"),
        }
        return {"ETag": f'"{new_etag}"'}

    def get_object(self, **kwargs) -> dict:
        key = kwargs["Key"]
        if key not in self.objects:
            raise _client_error("NoSuchKey")
        obj = self.objects[key]
        body = MagicMock()
        body.read.return_value = obj["body"]
        return {"Body": body, "ETag": f'"{obj["etag"]}"'}

    def head_object(self, **kwargs) -> dict:
        key = kwargs["Key"]
        if key not in self.objects:
            raise _client_error("404")
        return {"ETag": f'"{self.objects[key]["etag"]}"'}

    def delete_object(self, **kwargs) -> dict:
        self.objects.pop(kwargs["Key"], None)
        return {}

    def upload_file(self, _path: str, _bucket: str, key: str) -> None:
        self.put_object(Bucket=_bucket, Key=key, Body=b"parquet", ContentType="application/octet-stream")

    def get_paginator(self, _name: str) -> MagicMock:
        paginator = MagicMock()

        def paginate(**kwargs) -> list[dict]:
            prefix = kwargs.get("Prefix", "")
            contents = []
            for k in self.objects:
                if not k.startswith(prefix):
                    continue
                if k.endswith(".json") and "/completed/" in k:
                    contents.append({"Key": k})
                if k.endswith(".parquet") and "/shards/" in k:
                    contents.append({"Key": k})
            return [{"Contents": contents}] if contents else [{}]

        paginator.paginate.side_effect = paginate
        return paginator


@pytest.fixture
def coord() -> S3Coordinator:
    c = S3Coordinator(
        bucket="test-bucket",
        prefix="pokerbot/test",
        worker_id="worker-a",
        region="us-east-1",
        claim_ttl_sec=7200,
    )
    c.s3 = FakeS3()  # type: ignore[assignment]
    return c


def test_fresh_claim_wins(coord: S3Coordinator) -> None:
    result = coord.try_claim("btn_vs_bb_flop|Ah7c2s|hero=5")
    assert result.claimed is True
    assert result.reason == "claimed"
    assert coord.metrics.claims_won == 1


def test_concurrent_claim_collision(coord: S3Coordinator) -> None:
    spot = "btn_vs_bb_flop|Ah7c2s|hero=5"
    assert coord.try_claim(spot).claimed is True

    other = S3Coordinator(
        bucket=coord.bucket,
        prefix=coord.prefix,
        worker_id="worker-b",
        region=coord.region,
        claim_ttl_sec=coord.claim_ttl_sec,
    )
    other.s3 = coord.s3  # type: ignore[assignment]

    result = other.try_claim(spot)
    assert result.claimed is False
    assert result.collision is True
    assert other.metrics.claim_collisions == 1


def test_stale_claim_reclaimed(coord: S3Coordinator) -> None:
    spot = "btn_vs_bb_flop|Ah7c2s|hero=5"
    coord.claim_ttl_sec = 60
    assert coord.try_claim(spot).claimed is True

    from pokerbot.solver.label_completion import spot_key_hash

    claim_key = coord._key("claims", f"{spot_key_hash(spot)}.json")
    stale_ts = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
    stale_body = json.dumps(
        {"spot_key": spot, "worker_id": "old-worker", "claimed_at": stale_ts},
        sort_keys=True,
    ).encode()
    coord.s3.objects[claim_key]["body"] = stale_body  # type: ignore[attr-defined]

    other = S3Coordinator(
        bucket=coord.bucket,
        prefix=coord.prefix,
        worker_id="worker-b",
        region=coord.region,
        claim_ttl_sec=60,
    )
    other.s3 = coord.s3  # type: ignore[assignment]
    result = other.try_claim(spot)
    assert result.claimed is True
    assert result.reason == "reclaimed_stale"
    assert other.metrics.stale_claims_reclaimed == 1


def test_active_claim_not_reclaimed(coord: S3Coordinator) -> None:
    spot = "sb_vs_bb_flop|Kc8s2d|hero=4"
    assert coord.try_claim(spot).claimed is True

    other = S3Coordinator(
        bucket=coord.bucket,
        prefix=coord.prefix,
        worker_id="worker-b",
        region=coord.region,
        claim_ttl_sec=7200,
    )
    other.s3 = coord.s3  # type: ignore[assignment]
    result = other.try_claim(spot)
    assert result.claimed is False
    assert result.collision is True
    assert result.reason == "collision_active"


def test_release_claim_deletes_file(coord: S3Coordinator) -> None:
    spot = "bunching_postflop|Kc8s2d|hero=3"
    assert coord.try_claim(spot).claimed is True
    coord.release_claim(spot)
    from pokerbot.solver.label_completion import spot_key_hash

    claim_key = coord._key("claims", f"{spot_key_hash(spot)}.json")
    assert claim_key not in coord.s3.objects  # type: ignore[attr-defined]


def test_mark_completed_first_writer_wins(coord: S3Coordinator) -> None:
    from pathlib import Path

    from pokerbot.solver.label_completion import spot_key_hash

    spot = "btn_vs_bb_flop|6d2s8d|hero=5"
    tmp = Path(__file__).parent / "_tmp_shard.parquet"
    tmp.write_bytes(b"fake")
    try:
        coord.upload_shard(spot, tmp)
        assert coord.mark_completed(spot, exploitability_pct=0.45) is True

        other = S3Coordinator(
            bucket=coord.bucket,
            prefix=coord.prefix,
            worker_id="worker-b",
            region=coord.region,
        )
        other.s3 = coord.s3  # type: ignore[assignment]
        assert other.mark_completed(spot, exploitability_pct=0.50) is False
        assert coord.is_completed(spot) is True
    finally:
        tmp.unlink(missing_ok=True)
        digest = spot_key_hash(spot)
        coord.s3.objects.pop(coord._key("shards", f"{digest}.parquet"), None)  # type: ignore[attr-defined]


def test_upload_shard_skips_existing(coord: S3Coordinator) -> None:
    from pathlib import Path

    spot = "btn_vs_bb_flop|6d2s8d|hero=3"
    tmp = Path(__file__).parent / "_tmp_shard.parquet"
    tmp.write_bytes(b"fake")
    try:
        assert coord.upload_shard(spot, tmp) is True
        assert coord.upload_shard(spot, tmp) is False
    finally:
        tmp.unlink(missing_ok=True)
