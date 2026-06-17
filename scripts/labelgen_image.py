#!/usr/bin/env python3
"""Content-addressed labelgen Docker image hash and S3 upload helpers."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]


def _hash_file(hasher: hashlib._Hash, path: Path, *, root: Path) -> None:
    rel = path.relative_to(root).as_posix()
    hasher.update(rel.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(path.read_bytes())
    hasher.update(b"\0")


def _hash_tree(hasher: hashlib._Hash, directory: Path, *, root: Path) -> None:
    if not directory.is_dir():
        return
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        _hash_file(hasher, path, root=root)


def postflop_solver_git_rev(root: Path | None = None) -> str:
    root = root or ROOT
    lock_text = (root / "external/postflop_solver_cli/Cargo.lock").read_text(encoding="utf-8")
    match = re.search(
        r'name = "postflop-solver"[\s\S]*?source = "git\+[^#]+#([0-9a-f]+)"',
        lock_text,
    )
    if not match:
        raise RuntimeError("postflop-solver git rev not found in Cargo.lock")
    return match.group(1)


def image_content_hash(root: Path | None = None) -> str:
    """Hash Dockerfile, deps, solver source/rev, and labelgen Python paths."""
    root = root or ROOT
    hasher = hashlib.sha256()

    for rel in (
        "infra/labelgen/Dockerfile",
        "infra/labelgen/requirements.txt",
    ):
        path = root / rel
        if path.is_file():
            _hash_file(hasher, path, root=root)

    _hash_tree(hasher, root / "external/postflop_solver_cli", root=root)

    for rel in (
        "scripts/build_solver_labels.py",
        "scripts/cloud_label_worker.py",
        "pokerbot/data/label_sources.py",
    ):
        path = root / rel
        if path.is_file():
            _hash_file(hasher, path, root=root)

    for rel in ("pokerbot/solver", "pokerbot/encoding", "pokerbot/engine"):
        _hash_tree(hasher, root / rel, root=root)

    hasher.update(postflop_solver_git_rev(root).encode("utf-8"))
    return hasher.hexdigest()[:16]


def image_s3_key(prefix: str, content_hash: str) -> str:
    return f"{prefix}/images/{content_hash}.tar.gz"


def write_image_manifest(
    *,
    bucket: str,
    prefix: str,
    content_hash: str,
    image_tag: str,
    uploaded_new: bool,
) -> dict[str, Any]:
    s3_key = image_s3_key(prefix, content_hash)
    return {
        "content_hash": content_hash,
        "s3_key": s3_key,
        "s3_uri": f"s3://{bucket}/{s3_key}",
        "image_tag": image_tag,
        "uploaded_new_image": uploaded_new,
    }


def upload_labelgen_image(
    s3,
    *,
    bucket: str,
    prefix: str,
    image_tag: str,
    skip_if_exists: bool = True,
) -> dict[str, Any]:
    """Save docker image to content-addressed S3 key; upload manifest pointer."""
    content_hash = image_content_hash()
    s3_key = image_s3_key(prefix, content_hash)

    if skip_if_exists:
        try:
            s3.head_object(Bucket=bucket, Key=s3_key)
            manifest = write_image_manifest(
                bucket=bucket,
                prefix=prefix,
                content_hash=content_hash,
                image_tag=image_tag,
                uploaded_new=False,
            )
            s3.put_object(
                Bucket=bucket,
                Key=f"{prefix}/image_manifest.json",
                Body=json.dumps(manifest, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
            manifest["upload_skipped"] = True
            return manifest
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in ("404", "NoSuchKey", "NotFound"):
                raise

    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tar_path = Path(tmp.name)
    gz_path = tar_path.with_suffix(".tar.gz")
    try:
        subprocess.run(
            ["docker", "save", image_tag, "-o", str(tar_path)],
            check=True,
        )
        with open(tar_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        s3.upload_file(str(gz_path), bucket, s3_key)
    finally:
        tar_path.unlink(missing_ok=True)
        gz_path.unlink(missing_ok=True)

    manifest = write_image_manifest(
        bucket=bucket,
        prefix=prefix,
        content_hash=content_hash,
        image_tag=image_tag,
        uploaded_new=True,
    )
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/image_manifest.json",
        Body=json.dumps(manifest, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    manifest["upload_skipped"] = False
    return manifest
