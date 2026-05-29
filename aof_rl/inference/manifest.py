"""Checkpoint manifest loading and verification."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    obs_version: int
    checkpoint_sha256: str
    encoder: dict[str, Any]
    ood_config: dict[str, Any]
    hmac_signature: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelManifest:
        return cls(
            model_id=str(data["model_id"]),
            obs_version=int(data.get("obs_version", 3)),
            checkpoint_sha256=str(data["checkpoint_sha256"]).lower(),
            encoder=dict(data.get("encoder") or {}),
            ood_config=dict(data.get("ood_config") or {}),
            hmac_signature=data.get("hmac_signature"),
        )


def default_manifest_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".manifest.json")


def load_manifest(path: Path) -> ModelManifest:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ManifestError("manifest must be a JSON object")
    return ModelManifest.from_dict(data)


def checkpoint_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(
    checkpoint_path: Path,
    manifest: ModelManifest,
    *,
    skip_hmac: bool = False,
) -> None:
    digest = checkpoint_sha256(checkpoint_path)
    if digest != manifest.checkpoint_sha256.lower():
        raise ManifestError(
            f"checkpoint SHA-256 mismatch: expected {manifest.checkpoint_sha256}, got {digest}"
        )
    if manifest.obs_version != 3:
        raise ManifestError(f"unsupported obs_version {manifest.obs_version}")
    if not skip_hmac and manifest.hmac_signature:
        key = os.environ.get("AOF_MANIFEST_HMAC_KEY", "").encode("utf-8")
        if not key:
            raise ManifestError("manifest has hmac_signature but AOF_MANIFEST_HMAC_KEY is unset")
        payload = json.dumps(
            {
                "model_id": manifest.model_id,
                "obs_version": manifest.obs_version,
                "checkpoint_sha256": manifest.checkpoint_sha256,
                "encoder": manifest.encoder,
            },
            sort_keys=True,
        ).encode("utf-8")
        expected = hmac.new(key, payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, manifest.hmac_signature):
            raise ManifestError("manifest HMAC verification failed")


def resolve_manifest(
    checkpoint_path: Path,
    manifest_path: Path | None,
) -> ModelManifest | None:
    if manifest_path is not None:
        if not manifest_path.is_file():
            raise ManifestError(f"manifest not found: {manifest_path}")
        return load_manifest(manifest_path)
    default = default_manifest_path(checkpoint_path)
    if default.is_file():
        return load_manifest(default)
    return None
