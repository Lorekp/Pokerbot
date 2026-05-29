"""Differential comparison for non-AoF postflop solver label artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# Default float tolerance for solver probabilities and exploitability metadata.
DEFAULT_FLOAT_ATOL = 1e-9
DEFAULT_FLOAT_RTOL = 1e-6

# Manifest keys ignored in semantic / float_contract modes (timing, host, timestamps).
MANIFEST_SEMANTIC_IGNORE: frozenset[str] = frozenset(
    {
        "generated_at",
        "cli_path",
        "config_path",
        "solver_git_rev",
        "build_method",
        "solver_orchestration",
    }
)

MANIFEST_TIMING_PATHS: tuple[tuple[str, ...], ...] = (
    ("board_timing",),
    ("convergence", "median_solve_time_sec"),
    ("convergence", "max_solve_time_sec"),
    ("enumeration", "per_spot_results"),
)

PARQUET_FLOAT_COLUMNS: frozenset[str] = frozenset(
    {
        "solver_exploitability_pct",
        "solver_policy",
        "cards",
        "actions",
        "position",
        "spot_scope",
    }
)


class DiffMode(str, Enum):
    STRICT = "strict"
    SEMANTIC = "semantic"
    FLOAT_CONTRACT = "float_contract"


@dataclass
class DiffIssue:
    path: str
    message: str
    expected: Any = None
    actual: Any = None


@dataclass
class DiffResult:
    mode: DiffMode
    passed: bool
    issues: list[DiffIssue] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.passed:
            lines = [f"[{self.mode.value}] {i.path}: {i.message}" for i in self.issues]
            raise AssertionError("Label diff failed:\n" + "\n".join(lines[:50]))


def _float_equal(a: float, b: float, *, atol: float, rtol: float) -> bool:
    return bool(np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True))


def _as_sequence(val: Any) -> list[Any] | None:
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    return None


def _normalize_float_column(val: Any) -> Any:
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, list):
        return [_normalize_float_column(x) for x in val]
    return float(val)


def _nested_float_equal(
    expected: Any,
    actual: Any,
    *,
    atol: float,
    rtol: float,
) -> bool:
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return all(
            _nested_float_equal(e, a, atol=atol, rtol=rtol) for e, a in zip(expected, actual)
        )
    return _float_equal(float(expected), float(actual), atol=atol, rtol=rtol)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _parse_raw_solver_json(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_solver_json")
    if raw is None:
        return {}
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def row_sort_key(row: dict[str, Any]) -> str:
    """Stable ordering key for Parquet rows."""
    raw = _parse_raw_solver_json(row)
    spot_id = raw.get("spot_id", "")
    req = raw.get("solver_request", {})
    board = req.get("board", [])
    hero = raw.get("hero_hole", [])
    hero_pos = raw.get("hero_position", "")
    return "|".join(
        [
            str(spot_id),
            "".join(str(c) for c in board),
            "".join(str(c) for c in hero),
            str(hero_pos),
            str(row.get("oop_range", "")),
            str(row.get("ip_range", "")),
        ]
    )


def canonicalize_action_distribution(dist: dict[str, float]) -> dict[str, float]:
    return {k: float(dist[k]) for k in sorted(dist.keys())}


def canonicalize_solver_response(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    if "action_distribution" in out and isinstance(out["action_distribution"], dict):
        out["action_distribution"] = canonicalize_action_distribution(out["action_distribution"])
    if "available_actions" in out and isinstance(out["available_actions"], list):
        out["available_actions"] = list(out["available_actions"])
    return out


def _walk_manifest_drop(obj: Any, path: tuple[str, ...], drop_paths: set[tuple[str, ...]]) -> Any:
    if path in drop_paths:
        return "__IGNORED__"
    if isinstance(obj, dict):
        return {
            k: _walk_manifest_drop(v, path + (k,), drop_paths)
            for k, v in sorted(obj.items())
        }
    if isinstance(obj, list):
        return [_walk_manifest_drop(v, path + (str(i),), drop_paths) for i, v in enumerate(obj)]
    return obj


def _manifest_drop_paths(mode: DiffMode) -> set[tuple[str, ...]]:
    drops: set[tuple[str, ...]] = set()
    if mode in (DiffMode.SEMANTIC, DiffMode.FLOAT_CONTRACT):
        for key in MANIFEST_SEMANTIC_IGNORE:
            drops.add((key,))
        drops.add(("board_timing",))
        drops.add(("convergence", "median_solve_time_sec"))
        drops.add(("convergence", "max_solve_time_sec"))
        drops.add(("output", "path"))
    return drops


def _strip_per_spot_timing(per_spot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in per_spot:
        cleaned = {k: v for k, v in row.items() if k != "elapsed_sec"}
        out.append(cleaned)
    return sorted(out, key=lambda r: str(r.get("spot", "")))


def canonicalize_manifest(manifest: dict[str, Any], mode: DiffMode) -> dict[str, Any]:
    drops = _manifest_drop_paths(mode)
    cleaned = _walk_manifest_drop(manifest, (), drops)
    if isinstance(cleaned, dict):
        if mode in (DiffMode.SEMANTIC, DiffMode.FLOAT_CONTRACT):
            if "per_spot_results" in cleaned and isinstance(cleaned["per_spot_results"], list):
                cleaned["per_spot_results"] = _strip_per_spot_timing(
                    [r for r in cleaned["per_spot_results"] if isinstance(r, dict)]
                )
            if "sample_labels" in cleaned and isinstance(cleaned["sample_labels"], list):
                samples = []
                for s in cleaned["sample_labels"]:
                    if not isinstance(s, dict):
                        continue
                    sc = dict(s)
                    if "action_distribution" in sc:
                        sc["action_distribution"] = canonicalize_action_distribution(sc["action_distribution"])
                    samples.append(sc)
                cleaned["sample_labels"] = sorted(
                    samples,
                    key=lambda x: (str(x.get("spot_id", "")), str(x.get("board", ""))),
                )
            if "range_scope_distribution" in cleaned and isinstance(
                cleaned["range_scope_distribution"], dict
            ):
                rsd = cleaned["range_scope_distribution"]
                cleaned["range_scope_distribution"] = {
                    k: rsd[k] for k in sorted(rsd.keys())
                }
        return cleaned
    return manifest


def _compare_values(
    path: str,
    expected: Any,
    actual: Any,
    mode: DiffMode,
    issues: list[DiffIssue],
    *,
    float_atol: float,
    float_rtol: float,
) -> None:
    if expected == "__IGNORED__" or actual == "__IGNORED__":
        return

    if mode == DiffMode.STRICT:
        if expected != actual:
            issues.append(DiffIssue(path, "strict mismatch", expected, actual))
        return

    if isinstance(expected, dict) and isinstance(actual, dict):
        _compare_dicts(path, expected, actual, mode, issues, float_atol=float_atol, float_rtol=float_rtol)
        return

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            issues.append(
                DiffIssue(path, f"list length {len(expected)} != {len(actual)}", len(expected), len(actual))
            )
            return
        for i, (e, a) in enumerate(zip(expected, actual)):
            _compare_values(
                f"{path}[{i}]",
                e,
                a,
                mode,
                issues,
                float_atol=float_atol,
                float_rtol=float_rtol,
            )
        return

    if isinstance(expected, (int, str, bool)) or expected is None:
        if expected != actual:
            issues.append(DiffIssue(path, "value mismatch", expected, actual))
        return

    if isinstance(expected, float) or isinstance(actual, float):
        ef = float(expected) if expected is not None else 0.0
        af = float(actual) if actual is not None else 0.0
        if mode == DiffMode.FLOAT_CONTRACT or isinstance(expected, float):
            if not _float_equal(ef, af, atol=float_atol, rtol=float_rtol):
                issues.append(DiffIssue(path, "float mismatch", ef, af))
        elif expected != actual:
            issues.append(DiffIssue(path, "value mismatch", expected, actual))
        return

    if expected != actual:
        issues.append(DiffIssue(path, "value mismatch", expected, actual))


def _compare_dicts(
    path: str,
    expected: dict[str, Any],
    actual: dict[str, Any],
    mode: DiffMode,
    issues: list[DiffIssue],
    *,
    float_atol: float,
    float_rtol: float,
) -> None:
    exp_keys = set(expected.keys())
    act_keys = set(actual.keys())
    if exp_keys != act_keys:
        missing = exp_keys - act_keys
        extra = act_keys - exp_keys
        if missing or extra:
            issues.append(
                DiffIssue(
                    path,
                    f"key mismatch missing={sorted(missing)} extra={sorted(extra)}",
                    sorted(exp_keys),
                    sorted(act_keys),
                )
            )
    for key in sorted(exp_keys & act_keys):
        _compare_values(
            f"{path}.{key}",
            expected[key],
            actual[key],
            mode,
            issues,
            float_atol=float_atol,
            float_rtol=float_rtol,
        )


def compare_manifests(
    expected: dict[str, Any],
    actual: dict[str, Any],
    mode: DiffMode = DiffMode.SEMANTIC,
    *,
    float_atol: float = DEFAULT_FLOAT_ATOL,
    float_rtol: float = DEFAULT_FLOAT_RTOL,
) -> DiffResult:
    issues: list[DiffIssue] = []
    if mode == DiffMode.STRICT:
        exp_text = json.dumps(expected, sort_keys=True, indent=2)
        act_text = json.dumps(actual, sort_keys=True, indent=2)
        if exp_text != act_text:
            issues.append(DiffIssue("manifest", "byte/json strict mismatch"))
    else:
        exp_c = canonicalize_manifest(expected, mode)
        act_c = canonicalize_manifest(actual, mode)
        _compare_dicts(
            "manifest",
            exp_c if isinstance(exp_c, dict) else {},
            act_c if isinstance(act_c, dict) else {},
            DiffMode.FLOAT_CONTRACT if mode == DiffMode.FLOAT_CONTRACT else DiffMode.SEMANTIC,
            issues,
            float_atol=float_atol,
            float_rtol=float_rtol,
        )
    return DiffResult(mode=mode, passed=len(issues) == 0, issues=issues)


def _canonicalize_parquet_row(row: dict[str, Any], mode: DiffMode) -> dict[str, Any]:
    out = dict(row)
    raw = _parse_raw_solver_json(out)
    if raw:
        if "action_distribution" in raw:
            raw["action_distribution"] = canonicalize_action_distribution(raw["action_distribution"])
        if "solver_response" in raw and isinstance(raw["solver_response"], dict):
            raw["solver_response"] = canonicalize_solver_response(raw["solver_response"])
        out["raw_solver_json"] = _canonical_json(raw)
    for col in ("legal_mask",):
        if col in out:
            seq = _as_sequence(out[col])
            if seq is not None:
                out[col] = [bool(x) for x in seq]
    if mode != DiffMode.STRICT:
        for col in PARQUET_FLOAT_COLUMNS:
            if col not in out:
                continue
            out[col] = _normalize_float_column(out[col])
    return out


def compare_parquet_rows(
    expected_rows: list[dict[str, Any]],
    actual_rows: list[dict[str, Any]],
    mode: DiffMode = DiffMode.SEMANTIC,
    *,
    float_atol: float = DEFAULT_FLOAT_ATOL,
    float_rtol: float = DEFAULT_FLOAT_RTOL,
) -> DiffResult:
    issues: list[DiffIssue] = []
    exp_sorted = sorted(
        [_canonicalize_parquet_row(r, mode) for r in expected_rows],
        key=row_sort_key,
    )
    act_sorted = sorted(
        [_canonicalize_parquet_row(r, mode) for r in actual_rows],
        key=row_sort_key,
    )
    if len(exp_sorted) != len(act_sorted):
        issues.append(
            DiffIssue(
                "parquet.row_count",
                f"{len(exp_sorted)} != {len(act_sorted)}",
                len(exp_sorted),
                len(act_sorted),
            )
        )
        return DiffResult(mode=mode, passed=False, issues=issues)

    if mode == DiffMode.STRICT:
        for i, (e, a) in enumerate(zip(exp_sorted, act_sorted)):
            if _canonical_json(e) != _canonical_json(a):
                issues.append(DiffIssue(f"parquet.rows[{i}]", "strict row mismatch"))
        return DiffResult(mode=mode, passed=len(issues) == 0, issues=issues)

    for i, (e, a) in enumerate(zip(exp_sorted, act_sorted)):
        if set(e.keys()) != set(a.keys()):
            issues.append(
                DiffIssue(
                    f"parquet.rows[{i}]",
                    "column mismatch",
                    sorted(e.keys()),
                    sorted(a.keys()),
                )
            )
            continue
        for col in sorted(e.keys()):
            ev, av = e[col], a[col]
            if col in PARQUET_FLOAT_COLUMNS or col == "raw_solver_json":
                if col == "raw_solver_json":
                    try:
                        ej = json.loads(ev) if isinstance(ev, str) else ev
                        aj = json.loads(av) if isinstance(av, str) else av
                        _compare_dicts(
                            f"parquet.rows[{i}].raw_solver_json",
                            ej,
                            aj,
                            DiffMode.FLOAT_CONTRACT,
                            issues,
                            float_atol=float_atol,
                            float_rtol=float_rtol,
                        )
                    except json.JSONDecodeError as exc:
                        issues.append(DiffIssue(f"parquet.rows[{i}].raw_solver_json", str(exc)))
                elif col in PARQUET_FLOAT_COLUMNS:
                    if not _nested_float_equal(ev, av, atol=float_atol, rtol=float_rtol):
                        issues.append(DiffIssue(f"parquet.rows[{i}].{col}", "float mismatch", ev, av))
                elif (seq_ev := _as_sequence(ev)) is not None or (seq_av := _as_sequence(av)) is not None:
                    seq_ev = seq_ev if seq_ev is not None else []
                    seq_av = seq_av if seq_av is not None else []
                    if len(seq_ev) != len(seq_av):
                        issues.append(
                            DiffIssue(f"parquet.rows[{i}].{col}", "length mismatch", len(seq_ev), len(seq_av))
                        )
                    else:
                        for j, (ef, af) in enumerate(zip(seq_ev, seq_av)):
                            if not _float_equal(float(ef), float(af), atol=float_atol, rtol=float_rtol):
                                issues.append(
                                    DiffIssue(
                                        f"parquet.rows[{i}].{col}[{j}]",
                                        "float mismatch",
                                        ef,
                                        af,
                                    )
                                )
                elif isinstance(ev, list):
                    if len(ev) != len(av):
                        issues.append(
                            DiffIssue(f"parquet.rows[{i}].{col}", "length mismatch", len(ev), len(av))
                        )
                    else:
                        for j, (ef, af) in enumerate(zip(ev, av)):
                            if not _float_equal(float(ef), float(af), atol=float_atol, rtol=float_rtol):
                                issues.append(
                                    DiffIssue(
                                        f"parquet.rows[{i}].{col}[{j}]",
                                        "float mismatch",
                                        ef,
                                        af,
                                    )
                                )
                elif not _float_equal(float(ev), float(av), atol=float_atol, rtol=float_rtol):
                    issues.append(DiffIssue(f"parquet.rows[{i}].{col}", "float mismatch", ev, av))
            elif ev != av:
                issues.append(DiffIssue(f"parquet.rows[{i}].{col}", "value mismatch", ev, av))

    return DiffResult(mode=mode, passed=len(issues) == 0, issues=issues)


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def compare_label_dirs(
    expected_dir: Path,
    actual_dir: Path,
    mode: DiffMode = DiffMode.SEMANTIC,
    *,
    float_atol: float = DEFAULT_FLOAT_ATOL,
    float_rtol: float = DEFAULT_FLOAT_RTOL,
) -> DiffResult:
    """Compare two label run directories (manifest.json + train.parquet)."""
    all_issues: list[DiffIssue] = []

    exp_manifest_path = expected_dir / "manifest.json"
    act_manifest_path = actual_dir / "manifest.json"
    for label, path in (("expected", exp_manifest_path), ("actual", act_manifest_path)):
        if not path.is_file():
            all_issues.append(DiffIssue(f"{label}_dir", f"missing manifest.json at {path}"))

    manifest_result = DiffResult(mode=mode, passed=True)
    if exp_manifest_path.is_file() and act_manifest_path.is_file():
        exp_m = json.loads(exp_manifest_path.read_text(encoding="utf-8"))
        act_m = json.loads(act_manifest_path.read_text(encoding="utf-8"))
        manifest_result = compare_manifests(
            exp_m, act_m, mode, float_atol=float_atol, float_rtol=float_rtol
        )
        all_issues.extend(manifest_result.issues)

    exp_parquet = expected_dir / "train.parquet"
    act_parquet = actual_dir / "train.parquet"
    parquet_result = DiffResult(mode=mode, passed=True)
    if exp_parquet.is_file() and act_parquet.is_file():
        if mode == DiffMode.STRICT:
            if file_sha256(exp_parquet) != file_sha256(act_parquet):
                all_issues.append(DiffIssue("train.parquet", "strict byte hash mismatch"))
        else:
            parquet_result = compare_parquet_rows(
                load_parquet_rows(exp_parquet),
                load_parquet_rows(act_parquet),
                mode,
                float_atol=float_atol,
                float_rtol=float_rtol,
            )
            all_issues.extend(parquet_result.issues)
    elif exp_parquet.is_file() != act_parquet.is_file():
        all_issues.append(
            DiffIssue(
                "train.parquet",
                "presence mismatch",
                exp_parquet.is_file(),
                act_parquet.is_file(),
            )
        )

    return DiffResult(mode=mode, passed=len(all_issues) == 0, issues=all_issues)


def format_diff_report(result: DiffResult) -> str:
    lines = [f"# Label diff report ({result.mode.value})", "", f"Passed: {result.passed}", ""]
    if result.issues:
        lines.append("## Issues")
        for issue in result.issues:
            lines.append(f"- **{issue.path}**: {issue.message}")
            if issue.expected is not None:
                lines.append(f"  - expected: {issue.expected!r}")
            if issue.actual is not None:
                lines.append(f"  - actual: {issue.actual!r}")
    return "\n".join(lines)
