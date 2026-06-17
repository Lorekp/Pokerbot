"""S3-backed distributed claim coordination for label workers."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

from pokerbot.solver.label_completion import spot_key_hash


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ClaimResult:
    spot_key: str
    claimed: bool
    reason: str
    collision: bool = False


@dataclass
class CoordinationMetrics:
    claims_attempted: int = 0
    claims_won: int = 0
    claim_collisions: int = 0
    already_completed: int = 0
    stale_claims_reclaimed: int = 0
    spots_solved: int = 0
    spot_interruptions: int = 0
    s3_upload_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        rate = (
            round(self.claim_collisions / self.claims_attempted, 4)
            if self.claims_attempted
            else 0.0
        )
        return {
            "claims_attempted": self.claims_attempted,
            "claims_won": self.claims_won,
            "claim_collisions": self.claim_collisions,
            "claim_collision_rate": rate,
            "already_completed": self.already_completed,
            "stale_claims_reclaimed": self.stale_claims_reclaimed,
            "spots_solved": self.spots_solved,
            "spot_interruptions": self.spot_interruptions,
            "s3_upload_failures": self.s3_upload_failures,
        }


@dataclass
class S3Coordinator:
    bucket: str
    prefix: str
    worker_id: str
    region: str = "us-east-1"
    claim_ttl_sec: int = 7200
    metrics: CoordinationMetrics = field(default_factory=CoordinationMetrics)
    _completed_cache: set[str] = field(default_factory=set, init=False)
    _cache_ts: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.s3 = boto3.client("s3", region_name=self.region)

    @property
    def completed_prefix(self) -> str:
        return f"{self.prefix}/completed"

    @property
    def claims_prefix(self) -> str:
        return f"{self.prefix}/claims"

    @property
    def shards_prefix(self) -> str:
        return f"{self.prefix}/shards"

    @property
    def metrics_prefix(self) -> str:
        return f"{self.prefix}/metrics"

    def _key(self, *parts: str) -> str:
        return "/".join([self.prefix, *parts])

    def _shard_hashes_present(self) -> set[str]:
        hashes: set[str] = set()
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.shards_prefix}/"):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(f"{self.shards_prefix}/") :]
                if rel.endswith(".parquet"):
                    hashes.add(rel[: -len(".parquet")])
        return hashes

    def refresh_completed(
        self,
        *,
        manifest_keys: list[str] | None = None,
        force: bool = False,
    ) -> set[str]:
        now = time.time()
        if (
            not force
            and manifest_keys is None
            and now - self._cache_ts < 30
            and self._completed_cache
        ):
            return self._completed_cache

        if manifest_keys is not None:
            shard_hashes = self._shard_hashes_present()
            completed = {
                key for key in manifest_keys if spot_key_hash(key) in shard_hashes
            }
            self._completed_cache = completed
            self._cache_ts = now
            return completed

        completed: set[str] = set()
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.completed_prefix}/"):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(f"{self.completed_prefix}/") :]
                if rel.endswith(".json"):
                    try:
                        body = self.s3.get_object(Bucket=self.bucket, Key=obj["Key"])["Body"].read()
                        rec = json.loads(body)
                        key = str(rec.get("spot_key", ""))
                        if key:
                            completed.add(key)
                    except (ClientError, json.JSONDecodeError):
                        continue
        self._completed_cache = completed
        self._cache_ts = now
        return completed

    def is_completed(self, spot_key: str) -> bool:
        if spot_key in self._completed_cache:
            return True
        digest = spot_key_hash(spot_key)
        try:
            self.s3.head_object(
                Bucket=self.bucket,
                Key=self._key("shards", f"{digest}.parquet"),
            )
            self._completed_cache.add(spot_key)
            return True
        except ClientError:
            return False

    def _claim_is_stale(self, claim_body: dict[str, Any]) -> bool:
        claimed_at = claim_body.get("claimed_at")
        if not claimed_at:
            return True
        try:
            ts = datetime.fromisoformat(str(claimed_at).replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            return age > self.claim_ttl_sec
        except ValueError:
            return True

    def try_claim(self, spot_key: str) -> ClaimResult:
        self.metrics.claims_attempted += 1
        if self.is_completed(spot_key):
            self.metrics.already_completed += 1
            return ClaimResult(spot_key, False, "already_completed")

        digest = spot_key_hash(spot_key)
        claim_key = self._key("claims", f"{digest}.json")
        claim_body = {
            "spot_key": spot_key,
            "worker_id": self.worker_id,
            "claimed_at": _utcnow(),
        }
        body = json.dumps(claim_body, sort_keys=True).encode()

        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=claim_key,
                Body=body,
                ContentType="application/json",
                IfNoneMatch="*",
            )
            self.metrics.claims_won += 1
            return ClaimResult(spot_key, True, "claimed")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in ("PreconditionFailed", "412"):
                return ClaimResult(spot_key, False, f"s3_error:{code}")

        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=claim_key)
            existing = json.loads(resp["Body"].read())
        except ClientError:
            self.metrics.claim_collisions += 1
            return ClaimResult(spot_key, False, "collision", collision=True)

        if not self._claim_is_stale(existing):
            self.metrics.claim_collisions += 1
            return ClaimResult(spot_key, False, "collision_active", collision=True)

        etag = resp.get("ETag", "").strip('"')
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=claim_key,
                Body=body,
                ContentType="application/json",
                IfMatch=etag,
            )
            self.metrics.stale_claims_reclaimed += 1
            self.metrics.claims_won += 1
            return ClaimResult(spot_key, True, "reclaimed_stale")
        except ClientError:
            self.metrics.claim_collisions += 1
            return ClaimResult(spot_key, False, "collision_reclaim", collision=True)

    def release_claim(self, spot_key: str) -> None:
        digest = spot_key_hash(spot_key)
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=self._key("claims", f"{digest}.json"))
        except ClientError:
            pass

    def mark_completed(
        self,
        spot_key: str,
        *,
        exploitability_pct: float | None = None,
        iterations_used: int | None = None,
        elapsed_sec: float | None = None,
        source: str = "cloud_solve",
        near_timeout: bool = False,
    ) -> bool:
        """Write completion record. Returns True if this worker was first to complete."""
        digest = spot_key_hash(spot_key)
        rec: dict[str, Any] = {
            "spot_key": spot_key,
            "completed_at": _utcnow(),
            "worker_id": self.worker_id,
            "source": source,
        }
        if exploitability_pct is not None:
            rec["exploitability_pct"] = exploitability_pct
        if iterations_used is not None:
            rec["iterations_used"] = iterations_used
        if elapsed_sec is not None:
            rec["elapsed_sec"] = elapsed_sec
        if near_timeout:
            rec["near_timeout"] = True

        body = json.dumps(rec, sort_keys=True).encode()
        completed_key = self._key("completed", f"{digest}.json")
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=completed_key,
                Body=body,
                ContentType="application/json",
                IfNoneMatch="*",
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in ("PreconditionFailed", "412"):
                raise
            self._log_duplicate_completion(spot_key)
            self._completed_cache.add(spot_key)
            self.release_claim(spot_key)
            return False

        self._completed_cache.add(spot_key)
        self.release_claim(spot_key)
        self.metrics.spots_solved += 1
        return True

    def _log_duplicate_completion(self, spot_key: str) -> None:
        event = {
            "ts": _utcnow(),
            "worker_id": self.worker_id,
            "spot_key": spot_key,
            "reason": "duplicate_completion",
        }
        key = self._key(
            "metrics",
            "duplicates",
            f"{spot_key_hash(spot_key)}_{int(time.time())}.json",
        )
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json.dumps(event).encode(),
                ContentType="application/json",
            )
        except ClientError:
            pass

    def upload_shard(self, spot_key: str, shard_path: str | os.PathLike[str]) -> bool:
        """Upload shard if not already present. Returns True if uploaded."""
        digest = spot_key_hash(spot_key)
        key = self._key("shards", f"{digest}.parquet")
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return False
        except ClientError:
            pass
        try:
            self.s3.upload_file(str(shard_path), self.bucket, key)
            return True
        except ClientError:
            self.metrics.s3_upload_failures += 1
            raise

    def log_collision_event(self, spot_key: str, reason: str) -> None:
        event = {
            "ts": _utcnow(),
            "worker_id": self.worker_id,
            "spot_key": spot_key,
            "reason": reason,
        }
        key = self._key("metrics", "collisions", f"{spot_key_hash(spot_key)}_{int(time.time())}.json")
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json.dumps(event).encode(),
                ContentType="application/json",
            )
        except ClientError:
            pass

    def flush_metrics(self) -> None:
        body = json.dumps(
            {"worker_id": self.worker_id, "updated_at": _utcnow(), **self.metrics.to_dict()},
            indent=2,
        ).encode()
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._key("metrics", "workers", f"{self.worker_id}.json"),
            Body=body,
            ContentType="application/json",
        )

    def load_work_manifest(self) -> dict[str, Any]:
        key = self._key("work_manifest.json")
        body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return json.loads(body)

    def download_file(self, s3_rel: str, local_path: str | os.PathLike[str]) -> None:
        self.s3.download_file(self.bucket, self._key(s3_rel), str(local_path))

    def upload_file(self, local_path: str | os.PathLike[str], s3_rel: str) -> None:
        self.s3.upload_file(str(local_path), self.bucket, self._key(s3_rel))
