#!/usr/bin/env python3
"""Compare non-AoF postflop solver label artifacts (golden vs candidate)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pokerbot.solver.label_diff import (
    DEFAULT_FLOAT_ATOL,
    DEFAULT_FLOAT_RTOL,
    DiffMode,
    compare_label_dirs,
    compare_manifests,
    compare_parquet_rows,
    format_diff_report,
    load_parquet_rows,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Diff non-AoF solver label artifacts")
    p.add_argument(
        "--expected",
        type=Path,
        required=True,
        help="Expected run directory or manifest.json path",
    )
    p.add_argument(
        "--actual",
        type=Path,
        required=True,
        help="Actual run directory or manifest.json path",
    )
    p.add_argument(
        "--mode",
        choices=[m.value for m in DiffMode],
        default=DiffMode.SEMANTIC.value,
        help="strict | semantic | float_contract",
    )
    p.add_argument("--float-atol", type=float, default=DEFAULT_FLOAT_ATOL)
    p.add_argument("--float-rtol", type=float, default=DEFAULT_FLOAT_RTOL)
    p.add_argument("--report", type=Path, default=None, help="Write markdown report here")
    p.add_argument("--json", type=Path, default=None, help="Write JSON summary here")
    args = p.parse_args()

    mode = DiffMode(args.mode)

    def _resolve_dir(path: Path) -> Path:
        if path.is_file() and path.name in ("manifest.json", "train.parquet"):
            return path.parent
        return path

    exp_dir = _resolve_dir(args.expected)
    act_dir = _resolve_dir(args.actual)

    result = compare_label_dirs(
        exp_dir,
        act_dir,
        mode,
        float_atol=args.float_atol,
        float_rtol=args.float_rtol,
    )

    report = format_diff_report(result)
    print(report)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")

    if args.json:
        payload = {
            "mode": mode.value,
            "passed": result.passed,
            "issue_count": len(result.issues),
            "issues": [
                {"path": i.path, "message": i.message, "expected": i.expected, "actual": i.actual}
                for i in result.issues
            ],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not result.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
