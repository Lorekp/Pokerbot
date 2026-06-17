"""Tests for shard-backed completion ledger."""

from __future__ import annotations

from pathlib import Path

from pokerbot.solver.cloud_spot_solver import order_pending_for_worker
from pokerbot.solver.label_completion import (
    CompletionLedger,
    analyze_shard_completion,
    analyze_shard_completion_s3,
    completed_keys_from_shard_hashes,
    has_shard,
    spot_key_hash,
    unique_manifest_keys,
)


def test_has_shard(tmp_path: Path) -> None:
    key = "btn_vs_bb_flop|Ah7c2s|hero=5"
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    (shard_dir / f"{spot_key_hash(key)}.parquet").write_bytes(b"x")
    assert has_shard(shard_dir, key)
    assert not has_shard(shard_dir, "sb_vs_bb_flop|Kc8s2d|hero=4")


def test_rebuild_ledger_from_shards(tmp_path: Path) -> None:
    keys = [
        "btn_vs_bb_flop|Ah7c2s|hero=5",
        "sb_vs_bb_flop|Kc8s2d|hero=4",
        "bunching_postflop|6d2s8d|hero=3",
    ]
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    ledger_path = tmp_path / "completed.jsonl"
    ledger_path.write_text(
        '{"spot_key": "sb_vs_bb_flop|Kc8s2d|hero=4"}\n' * 3,
        encoding="utf-8",
    )
    (shard_dir / f"{spot_key_hash(keys[0])}.parquet").write_bytes(b"a")
    (shard_dir / f"{spot_key_hash(keys[2])}.parquet").write_bytes(b"b")

    report = analyze_shard_completion(
        manifest_keys=keys,
        shard_dir=shard_dir,
        ledger_path=ledger_path,
    )
    assert report.shard_backed == 2
    assert report.ledger_done_no_shard == 1
    assert report.todo_count == 1
    assert report.duplicate_ledger_lines == 2

    ledger = CompletionLedger.rebuild_from_shards(ledger_path, shard_dir, keys)
    assert len(ledger.completed) == 2
    assert ledger_path.read_text(encoding="utf-8").count("\n") == 2


def test_order_pending_differs_by_worker() -> None:
    keys = [f"btn_vs_bb_flop|Ah7c2s|hero={i}" for i in range(6)]
    a = order_pending_for_worker(keys, "worker-a")
    b = order_pending_for_worker(keys, "worker-b")
    assert a != b
    assert sorted(a) == sorted(b)


def test_unique_manifest_keys_dedupes_stack_depth(tmp_path: Path) -> None:
    key = "btn_vs_bb_flop|Ah7c2s|hero=5"
    manifest = [key, key]
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    (shard_dir / f"{spot_key_hash(key)}.parquet").write_bytes(b"x")

    report = analyze_shard_completion(manifest_keys=manifest, shard_dir=shard_dir)
    assert report.expected_count == 1
    assert report.shard_backed == 1
    assert report.todo_count == 0


def test_analyze_shard_completion_s3_ignores_orphan_markers() -> None:
    class FakeS3:
        def get_paginator(self, _name: str):
            paginator = self

            def paginate(**kwargs):
                prefix = kwargs.get("Prefix", "")
                if prefix.endswith("/shards/"):
                    return [{"Contents": [{"Key": f"{prefix}abc123.parquet"}]}]
                return [{}]

            paginator.paginate = paginate
            return paginator

    key = "btn_vs_bb_flop|Ah7c2s|hero=5"
    digest = spot_key_hash(key)
    hashes = {digest}
    matched = completed_keys_from_shard_hashes([key], hashes)
    assert key in matched

    class FakeS3:
        def get_paginator(self, _name: str):
            paginator = self

            def paginate(**kwargs):
                prefix = kwargs.get("Prefix", "")
                if prefix.endswith("/shards/"):
                    return [{"Contents": [{"Key": f"{prefix}{digest}.parquet"}]}]
                return [{}]

            paginator.paginate = paginate
            return paginator

    report = analyze_shard_completion_s3(
        FakeS3(),
        bucket="b",
        prefix="pokerbot/test",
        manifest_keys=[key, key],
    )
    assert report.shard_backed == 1
    assert report.expected_count == 1
