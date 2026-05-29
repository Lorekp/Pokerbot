"""Teacher LUT schema sidecar and config-hash validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LUT_SCHEMA_VERSION = 1

V1_PATH_SET = (
    "static_profile_open",
    "static_profile_clean_call",
    "static_profile_call_behind",
)

# Files whose bytes affect range lookups (see plan).
_RANGE_HASH_FILES = (
    "aof_ranges.py",
    "aof_ranges_chart.py",
    "aof_ranges_gto.py",
    "aof_ranges_tight.py",
)


class TeacherLutConfigMismatch(RuntimeError):
    """Raised when LUT artifact config hash does not match live training config."""


@dataclass
class TeacherLutSchema:
    lut_schema_version: int
    teacher_config_hash: str
    teacher_version: int
    use_fast: bool
    teacher_samples: int
    teacher_margin_bb: float
    teacher_min_samples: int
    stack_bb: float
    v1_path_set: tuple[str, ...]
    n_ctx: int
    n_hand: int = 169
    value_keys: tuple[str, ...] = ("q_fold_bb", "q_shove_bb", "p_fold", "edge_bb")
    build_git_commit: str | None = None
    range_file_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["v1_path_set"] = list(self.v1_path_set)
        d["value_keys"] = list(self.value_keys)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TeacherLutSchema:
        return cls(
            lut_schema_version=int(data["lut_schema_version"]),
            teacher_config_hash=str(data["teacher_config_hash"]),
            teacher_version=int(data["teacher_version"]),
            use_fast=bool(data["use_fast"]),
            teacher_samples=int(data["teacher_samples"]),
            teacher_margin_bb=float(data["teacher_margin_bb"]),
            teacher_min_samples=int(data["teacher_min_samples"]),
            stack_bb=float(data["stack_bb"]),
            v1_path_set=tuple(data["v1_path_set"]),
            n_ctx=int(data["n_ctx"]),
            n_hand=int(data.get("n_hand", 169)),
            value_keys=tuple(data.get("value_keys", ("q_fold_bb", "q_shove_bb", "p_fold", "edge_bb"))),
            build_git_commit=data.get("build_git_commit"),
            range_file_hashes=dict(data.get("range_file_hashes", {})),
        )

    def save_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> TeacherLutSchema:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def range_file_hashes(repo_root: Path | None = None) -> dict[str, str]:
    root = repo_root or _repo_root()
    out: dict[str, str] = {}
    for name in _RANGE_HASH_FILES:
        p = root / name
        if p.is_file():
            out[name] = _sha256_file(p)
    return out


def compute_teacher_config_hash(
    *,
    teacher_version: int,
    use_fast: bool,
    teacher_samples: int,
    teacher_margin_bb: float,
    teacher_min_samples: int,
    stack_bb: float,
    lut_schema_version: int = LUT_SCHEMA_VERSION,
    v1_path_set: tuple[str, ...] = V1_PATH_SET,
    repo_root: Path | None = None,
) -> str:
    """Canonical SHA-256 over teacher knobs + range file bytes."""
    root = repo_root or _repo_root()
    blob: dict[str, Any] = {
        "lut_schema_version": lut_schema_version,
        "v1_path_set": list(v1_path_set),
        "teacher_version": teacher_version,
        "use_fast": use_fast,
        "teacher_samples": teacher_samples,
        "teacher_margin_bb": teacher_margin_bb,
        "teacher_min_samples": teacher_min_samples,
        "stack_bb": stack_bb,
        "range_files": range_file_hashes(root),
    }
    canonical = json.dumps(blob, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_config_hash(
    schema: TeacherLutSchema,
    *,
    teacher_version: int,
    use_fast: bool,
    teacher_samples: int,
    teacher_margin_bb: float,
    teacher_min_samples: int,
    stack_bb: float,
    repo_root: Path | None = None,
) -> None:
    live = compute_teacher_config_hash(
        teacher_version=teacher_version,
        use_fast=use_fast,
        teacher_samples=teacher_samples,
        teacher_margin_bb=teacher_margin_bb,
        teacher_min_samples=teacher_min_samples,
        stack_bb=stack_bb,
        lut_schema_version=schema.lut_schema_version,
        v1_path_set=schema.v1_path_set,
        repo_root=repo_root,
    )
    if live != schema.teacher_config_hash:
        raise TeacherLutConfigMismatch(
            f"teacher_config_hash mismatch: artifact={schema.teacher_config_hash[:12]}… "
            f"live={live[:12]}… (range files or teacher knobs changed)"
        )
