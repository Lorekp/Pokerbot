"""Per-spot completion ledger, chain-log bootstrap, and shard Parquet I/O."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pokerbot.solver.export import merge_shards, write_shard

# spot_id|board6chars|hero=N
_SPOT_KEY_RE = re.compile(
    r"^(?P<spot_id>sb_vs_bb_flop|btn_vs_bb_flop|bunching_postflop)"
    r"\|(?P<board>(?:[2-9TJQKA][cdhs]){3})"
    r"\|hero=(?P<hero>[0-5])$"
)
_SOLVING_RE = re.compile(r"^\[(\d+)\] solving (.+)$")
_OK_CONVERGED_RE = re.compile(r"^\s*OK converged=True\b")


def spot_key_hash(spot_key: str) -> str:
    return hashlib.sha256(spot_key.encode("utf-8")).hexdigest()[:16]


def shard_path(shard_dir: Path, spot_key: str) -> Path:
    return shard_dir / f"{spot_key_hash(spot_key)}.parquet"


def has_shard(shard_dir: Path, spot_key: str) -> bool:
    return shard_path(shard_dir, spot_key).is_file()


def completed_keys_from_shards(shard_dir: Path, manifest_keys: list[str]) -> list[str]:
    """Return manifest keys whose Parquet shard exists (ground-truth completion)."""
    if not shard_dir.is_dir():
        return []
    return [k for k in manifest_keys if has_shard(shard_dir, k)]


def unique_manifest_keys(manifest_keys: list[str]) -> list[str]:
    """Deduplicate manifest keys (stack depth duplicates share one shard)."""
    return sorted(set(manifest_keys))


def shard_hashes_present_local(shard_dir: Path) -> set[str]:
    if not shard_dir.is_dir():
        return set()
    return {
        p.stem
        for p in shard_dir.glob("*.parquet")
        if not p.name.endswith(".tmp")
    }


def completed_keys_from_shard_hashes(
    manifest_keys: list[str], shard_hashes: set[str]
) -> set[str]:
    return {k for k in manifest_keys if spot_key_hash(k) in shard_hashes}


def shard_hashes_present_s3(s3: Any, *, bucket: str, prefix: str) -> set[str]:
    """List shard digests under prefix/shards/."""
    hashes: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    shards_prefix = f"{prefix.rstrip('/')}/shards/"
    for page in paginator.paginate(Bucket=bucket, Prefix=shards_prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(shards_prefix) :]
            if rel.endswith(".parquet"):
                hashes.add(rel[: -len(".parquet")])
    return hashes


def analyze_shard_completion_s3(
    s3: Any,
    *,
    bucket: str,
    prefix: str,
    manifest_keys: list[str],
) -> ShardCompletionReport:
    """Shard-authoritative completion report for an S3 labelgen prefix."""
    unique_keys = unique_manifest_keys(manifest_keys)
    shard_hashes = shard_hashes_present_s3(s3, bucket=bucket, prefix=prefix)
    shard_set = completed_keys_from_shard_hashes(unique_keys, shard_hashes)

    from collections import Counter

    usable_by_type: Counter[str] = Counter()
    never_by_type: Counter[str] = Counter()
    never_touched = 0
    for key in unique_keys:
        t = spot_type(key)
        if key in shard_set:
            usable_by_type[t] += 1
        else:
            never_touched += 1
            never_by_type[t] += 1

    return ShardCompletionReport(
        expected_count=len(unique_keys),
        shard_backed=len(shard_set),
        ledger_unique=0,
        ledger_lines=0,
        duplicate_ledger_lines=0,
        ledger_done_no_shard=0,
        never_touched=never_touched,
        usable_by_type=dict(sorted(usable_by_type.items())),
        ledger_done_no_shard_by_type={},
        never_touched_by_type=dict(sorted(never_by_type.items())),
        todo_count=len(unique_keys) - len(shard_set),
    )


def shard_completed_keys_s3(
    s3: Any, *, bucket: str, prefix: str, manifest_keys: list[str]
) -> set[str]:
    """Return unique manifest keys with a shard object in S3."""
    unique_keys = unique_manifest_keys(manifest_keys)
    shard_hashes = shard_hashes_present_s3(s3, bucket=bucket, prefix=prefix)
    return completed_keys_from_shard_hashes(unique_keys, shard_hashes)


def spot_type(spot_key: str) -> str:
    return spot_key.split("|", 1)[0]


@dataclass
class ShardCompletionReport:
    expected_count: int
    shard_backed: int
    ledger_unique: int
    ledger_lines: int
    duplicate_ledger_lines: int
    ledger_done_no_shard: int
    never_touched: int
    usable_by_type: dict[str, int]
    ledger_done_no_shard_by_type: dict[str, int]
    never_touched_by_type: dict[str, int]
    todo_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_count": self.expected_count,
            "shard_backed": self.shard_backed,
            "ledger_unique": self.ledger_unique,
            "ledger_lines": self.ledger_lines,
            "duplicate_ledger_lines": self.duplicate_ledger_lines,
            "ledger_done_no_shard": self.ledger_done_no_shard,
            "never_touched": self.never_touched,
            "usable_by_type": self.usable_by_type,
            "ledger_done_no_shard_by_type": self.ledger_done_no_shard_by_type,
            "never_touched_by_type": self.never_touched_by_type,
            "todo_count": self.todo_count,
        }


def analyze_shard_completion(
    *,
    manifest_keys: list[str],
    shard_dir: Path,
    ledger_path: Path | None = None,
) -> ShardCompletionReport:
    """Compare manifest vs shard files vs optional log-derived ledger (unique keys)."""
    from collections import Counter

    unique_keys = unique_manifest_keys(manifest_keys)
    shard_set = set(completed_keys_from_shards(shard_dir, unique_keys))
    ledger_unique: set[str] = set()
    ledger_lines = 0
    if ledger_path is not None and ledger_path.is_file():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            ledger_lines += 1
            rec = json.loads(line)
            key = str(rec.get("spot_key", ""))
            if key:
                ledger_unique.add(key)

    usable_by_type: Counter[str] = Counter()
    no_shard_by_type: Counter[str] = Counter()
    never_by_type: Counter[str] = Counter()

    ledger_done_no_shard = 0
    never_touched = 0
    for key in unique_keys:
        t = spot_type(key)
        if key in shard_set:
            usable_by_type[t] += 1
        elif key in ledger_unique:
            ledger_done_no_shard += 1
            no_shard_by_type[t] += 1
        else:
            never_touched += 1
            never_by_type[t] += 1

    return ShardCompletionReport(
        expected_count=len(unique_keys),
        shard_backed=len(shard_set),
        ledger_unique=len(ledger_unique),
        ledger_lines=ledger_lines,
        duplicate_ledger_lines=max(0, ledger_lines - len(ledger_unique)),
        ledger_done_no_shard=ledger_done_no_shard,
        never_touched=never_touched,
        usable_by_type=dict(sorted(usable_by_type.items())),
        ledger_done_no_shard_by_type=dict(sorted(no_shard_by_type.items())),
        never_touched_by_type=dict(sorted(never_by_type.items())),
        todo_count=len(unique_keys) - len(shard_set),
    )


def validate_spot_key(spot_key: str) -> bool:
    return _SPOT_KEY_RE.match(spot_key.strip()) is not None


def parse_spot_key(spot_key: str) -> dict[str, str] | None:
    m = _SPOT_KEY_RE.match(spot_key.strip())
    if not m:
        return None
    return {
        "spot_id": m.group("spot_id"),
        "board": m.group("board"),
        "hero": m.group("hero"),
    }


@dataclass
class BootstrapEntry:
    spot_key: str
    candidate_num: int | None = None
    exploitability_pct: float | None = None
    iterations_used: int | None = None
    elapsed_sec: float | None = None


@dataclass
class BootstrapValidationResult:
    expected_count: int | None
    parsed_count: int
    valid_count: int
    invalid_keys: list[str] = field(default_factory=list)
    unpaired_solving: list[str] = field(default_factory=list)
    duplicate_keys: list[str] = field(default_factory=list)
    in_flight_solving: list[str] = field(default_factory=list)
    entries: list[BootstrapEntry] = field(default_factory=list)
    cross_check_missing: list[str] = field(default_factory=list)
    cross_check_extra: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        if self.expected_count is not None and self.valid_count != self.expected_count:
            return False
        if self.invalid_keys or self.unpaired_solving:
            return False
        if self.cross_check_missing:
            return False
        return True


def _parse_ok_metadata(line: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = re.search(r"exploit=([\d.]+)%", line)
    if m:
        out["exploitability_pct"] = float(m.group(1))
    m = re.search(r"iters=(\d+)", line)
    if m:
        out["iterations_used"] = int(m.group(1))
    m = re.search(r"time=([\d.]+)s", line)
    if m:
        out["elapsed_sec"] = float(m.group(1))
    return out


def parse_chain_log(
    log_path: Path,
    *,
    section_marker: str = "prod6000 starting",
    expected_count: int | None = None,
    last_wins_duplicates: bool = True,
    allow_in_flight: bool = False,
) -> BootstrapValidationResult:
    """Pair ``[N] solving KEY`` lines with the next ``OK converged=True`` line."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    in_section = section_marker is None
    pending: tuple[int, str] | None = None
    by_key: dict[str, BootstrapEntry] = {}
    unpaired: list[str] = []
    duplicates: list[str] = []
    invalid: list[str] = []

    for line in lines:
        if not in_section:
            if section_marker in line:
                in_section = True
            continue

        solving_m = _SOLVING_RE.match(line.strip())
        if solving_m:
            if pending is not None:
                unpaired.append(pending[1])
            pending = (int(solving_m.group(1)), solving_m.group(2).strip())
            continue

        if pending is not None and _OK_CONVERGED_RE.match(line):
            cand_num, key = pending
            pending = None
            meta = _parse_ok_metadata(line)
            if not validate_spot_key(key):
                invalid.append(key)
                continue
            if key in by_key:
                duplicates.append(key)
                if not last_wins_duplicates:
                    continue
            by_key[key] = BootstrapEntry(
                spot_key=key,
                candidate_num=cand_num,
                exploitability_pct=meta.get("exploitability_pct"),
                iterations_used=meta.get("iterations_used"),
                elapsed_sec=meta.get("elapsed_sec"),
            )
            continue

        if pending is not None and line.strip().startswith("==="):
            unpaired.append(pending[1])
            pending = None

    in_flight: list[str] = []
    if pending is not None:
        if allow_in_flight and not unpaired:
            in_flight.append(pending[1])
        else:
            unpaired.append(pending[1])

    entries = sorted(by_key.values(), key=lambda e: e.candidate_num or 0)
    valid_count = len(entries)
    result = BootstrapValidationResult(
        expected_count=expected_count,
        parsed_count=valid_count,
        valid_count=valid_count,
        invalid_keys=invalid,
        unpaired_solving=unpaired,
        duplicate_keys=sorted(set(duplicates)),
        in_flight_solving=in_flight,
        entries=entries,
    )
    if allow_in_flight and in_flight and not unpaired and not invalid:
        # Live log: trailing in-flight spot is expected; duplicates are restart artifacts.
        pass
    return result


def cross_check_bootstrap_with_shards(
    result: BootstrapValidationResult,
    shard_dir: Path | None,
    parquet_path: Path | None,
) -> BootstrapValidationResult:
    """Cross-check bootstrap keys against shard files or a partial Parquet."""
    shard_keys: set[str] = set()
    if shard_dir is not None and shard_dir.is_dir():
        for entry in result.entries:
            if (shard_dir / f"{spot_key_hash(entry.spot_key)}.parquet").is_file():
                shard_keys.add(entry.spot_key)

    if parquet_path is not None and parquet_path.is_file():
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            spot_id = str(row.get("spot_id", ""))
            board = row.get("board_cards") or row.get("board")
            hero = row.get("hero_seat")
            if board is None or hero is None:
                continue
            if isinstance(board, (list, tuple)):
                board_str = "".join(str(c) for c in board)
            else:
                board_str = str(board).replace(" ", "")
            key = f"{spot_id}|{board_str}|hero={hero}"
            if validate_spot_key(key):
                shard_keys.add(key)

    if not shard_keys:
        return result

    bootstrapped = {e.spot_key for e in result.entries}
    result.cross_check_missing = sorted(bootstrapped - shard_keys)
    result.cross_check_extra = sorted(shard_keys - bootstrapped)
    return result


@dataclass
class CompletionLedger:
    path: Path
    completed: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> CompletionLedger:
        # CRASH-RESUME-AUDIT cross-cutting E: torn-tail-tolerant load. The original code did
        # ``path.read_text(...).splitlines()`` then json.loads(line) on EVERY line — a single torn
        # tail line raised json.JSONDecodeError and the WHOLE bootstrap died, requiring manual
        # ``rebuild_from_shards``. JSONL is line-oriented by design: skip the malformed tail and
        # log loudly. The dedup ledger is recoverable (lines already in memory); the bootstrap
        # proceeds with N-1 entries. The trailing `.atomic.tmp` file from a crashed write_atomic
        # is also ignored on load — only the renamed ``path`` is consumed.
        ledger = cls(path=path)
        if not path.is_file():
            return ledger
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return ledger
        malformed = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                malformed += 1
                continue
            key = str(rec.get("spot_key", ""))
            if key:
                ledger.completed[key] = rec
        if malformed:
            import logging
            logging.getLogger(__name__).warning(
                "CompletionLedger.load: skipped %d malformed tail line(s) in %s "
                "(torn write, recovery = rebuild_from_shards)", malformed, path,
            )
        return ledger

    def contains(self, spot_key: str) -> bool:
        return spot_key in self.completed

    def append(
        self,
        spot_key: str,
        *,
        exploitability_pct: float | None = None,
        iterations_used: int | None = None,
        elapsed_sec: float | None = None,
        source: str = "solve",
    ) -> None:
        rec = {
            "spot_key": spot_key,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
        }
        if exploitability_pct is not None:
            rec["exploitability_pct"] = exploitability_pct
        if iterations_used is not None:
            rec["iterations_used"] = iterations_used
        if elapsed_sec is not None:
            rec["elapsed_sec"] = elapsed_sec
        self.completed[spot_key] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    def merge_bootstrap(self, result: BootstrapValidationResult) -> int:
        added = 0
        for entry in result.entries:
            if entry.spot_key in self.completed:
                continue
            self.append(
                entry.spot_key,
                exploitability_pct=entry.exploitability_pct,
                iterations_used=entry.iterations_used,
                elapsed_sec=entry.elapsed_sec,
                source="bootstrap_log",
            )
            added += 1
        return added

    def keys_without_shards(self, shard_dir: Path) -> list[str]:
        missing: list[str] = []
        for key in self.completed:
            if not has_shard(shard_dir, key):
                missing.append(key)
        return sorted(missing)

    @classmethod
    def rebuild_from_shards(
        cls,
        path: Path,
        shard_dir: Path,
        manifest_keys: list[str],
        *,
        source: str = "shard_scan",
    ) -> "CompletionLedger":
        """Replace ledger contents with one record per shard-backed manifest key."""
        ledger = cls(path=path)
        now = datetime.now(timezone.utc).isoformat()
        for key in manifest_keys:
            if has_shard(shard_dir, key):
                ledger.completed[key] = {
                    "spot_key": key,
                    "completed_at": now,
                    "source": source,
                }
        ledger.write_atomic()
        return ledger

    def write_atomic(self) -> None:
        """Rewrite ledger file from in-memory completed dict (deduplicated).

        CRASH-RESUME-AUDIT #1: this method was named ``write_atomic`` but was NOT atomic — it called
        ``write_text`` directly. A power cut mid-write truncated the prod6000 bootstrap ledger and
        forced a full ``rebuild_from_shards``. Now: tmp + fsync + os.replace + parent-dir fsync,
        matching the reference in pokerbot/training/leduc_tabular_cfr.py:618-636 (the in-tree
        exemplar the original author already knew). The function name is preserved.
        """
        import os
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(self.completed[k], sort_keys=True)
            for k in sorted(self.completed)
        ]
        payload = "\n".join(lines) + ("\n" if lines else "")
        tmp = self.path.with_name(self.path.name + ".atomic.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, self.path)
        try:
            dir_fd = os.open(str(self.path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass


def bootstrap_and_validate(
    log_path: Path,
    *,
    ledger_path: Path,
    shard_dir: Path | None = None,
    parquet_path: Path | None = None,
    expected_count: int | None = None,
    section_marker: str = "prod6000 starting",
    write_ledger: bool = False,
    last_wins_duplicates: bool = True,
    allow_in_flight: bool = False,
) -> BootstrapValidationResult:
    result = parse_chain_log(
        log_path,
        section_marker=section_marker,
        expected_count=expected_count,
        last_wins_duplicates=last_wins_duplicates,
        allow_in_flight=allow_in_flight,
    )
    result = cross_check_bootstrap_with_shards(result, shard_dir, parquet_path)
    if write_ledger and result.ok:
        ledger = CompletionLedger.load(ledger_path)
        ledger.merge_bootstrap(result)
    return result


def write_completion_report(result: BootstrapValidationResult, report_path: Path) -> None:
    lines = [
        "# Chain log bootstrap validation",
        "",
        f"- Expected count: {result.expected_count}",
        f"- Valid paired completions: {result.valid_count}",
        f"- OK: {result.ok}",
        "",
    ]
    if result.invalid_keys:
        lines.extend(["## Invalid keys", ""] + [f"- `{k}`" for k in result.invalid_keys] + [""])
    if result.unpaired_solving:
        lines.extend(
            ["## Unpaired solving lines (no OK converged=True)", ""]
            + [f"- `{k}`" for k in result.unpaired_solving]
            + [""]
        )
    if result.duplicate_keys:
        lines.extend(
            ["## Duplicate keys (last-wins applied; restart artifact)", ""]
            + [f"- `{k}`" for k in result.duplicate_keys[:30]]
            + ([f"- ... and {len(result.duplicate_keys) - 30} more"] if len(result.duplicate_keys) > 30 else [])
            + [""]
        )
    if result.in_flight_solving:
        lines.extend(
            ["## In-flight (not marked complete)", ""]
            + [f"- `{k}`" for k in result.in_flight_solving]
            + [""]
        )
    if result.cross_check_missing:
        extra = ""
        if len(result.cross_check_missing) > 50:
            extra = f"\n- ... and {len(result.cross_check_missing) - 50} more"
        lines.extend(
            ["## Bootstrap keys missing from shards/parquet (informational if no shards yet)", ""]
            + [f"- `{k}`" for k in result.cross_check_missing[:50]]
            + ([extra] if extra else [])
            + [""]
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
