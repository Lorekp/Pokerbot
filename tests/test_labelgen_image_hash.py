"""Tests for labelgen Docker image content hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.labelgen_image import image_content_hash, postflop_solver_git_rev


def test_image_content_hash_stable() -> None:
    h1 = image_content_hash()
    h2 = image_content_hash()
    assert h1 == h2
    assert len(h1) == 16


def test_image_content_hash_changes_with_solver_rev(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "infra" / "labelgen").mkdir(parents=True)
    (root / "external" / "postflop_solver_cli").mkdir(parents=True)
    (root / "pokerbot" / "solver").mkdir(parents=True)
    (root / "pokerbot" / "encoding").mkdir(parents=True)
    (root / "pokerbot" / "engine").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)

    dockerfile = root / "infra/labelgen/Dockerfile"
    dockerfile.write_text("FROM python:3.11-slim\n", encoding="utf-8")
    (root / "infra/labelgen/requirements.txt").write_text("numpy\n", encoding="utf-8")
    (root / "scripts/build_solver_labels.py").write_text("# build\n", encoding="utf-8")
    (root / "scripts/cloud_label_worker.py").write_text("# worker\n", encoding="utf-8")

    lock_a = root / "external/postflop_solver_cli/Cargo.lock"
    lock_b = root / "external/postflop_solver_cli/Cargo.lock.b"
    rev_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    rev_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    lock_template = (
        '[[package]]\nname = "postflop-solver"\n'
        'source = "git+https://github.com/b-inary/postflop-solver?branch=main#{rev}"\n'
    )
    lock_a.write_text(lock_template.format(rev=rev_a), encoding="utf-8")
    lock_b.write_text(lock_template.format(rev=rev_b), encoding="utf-8")

    h_a = image_content_hash(root)
    lock_a.write_text(lock_template.format(rev=rev_b), encoding="utf-8")
    h_b = image_content_hash(root)
    assert h_a != h_b


def test_postflop_solver_git_rev_matches_versions_lock() -> None:
    rev = postflop_solver_git_rev()
    lock_path = Path(__file__).resolve().parents[1] / "infra/labelgen/versions.lock"
    text = lock_path.read_text(encoding="utf-8")
    assert rev in text
