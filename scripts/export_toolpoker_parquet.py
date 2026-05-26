#!/usr/bin/env python3
"""Convert ToolPoker solver JSONL exports to solver-annotated Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pokerbot.encoding.toolpoker_encoder import encode_toolpoker_row, encode_toolpoker_to_batch_dict
from pokerbot.encoding.weva_encoder import WEVAMapper


def main() -> None:
    p = argparse.ArgumentParser(description="Export ToolPoker JSONL to solver Parquet")
    p.add_argument("--input", type=Path, required=True, help="JSONL file (one solver response per line)")
    p.add_argument("--output", type=Path, default=Path("data/solver/train.parquet"))
    p.add_argument("--weva-index", type=Path, default=Path("data/weva"))
    args = p.parse_args()

    mapper = WEVAMapper.from_directory(args.weva_index)
    rows: list[dict] = []
    line_no = 0
    for line in args.input.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        line_no += 1
        try:
            payload = json.loads(line)
            record = encode_toolpoker_row(payload, mapper=mapper)
            rows.append(encode_toolpoker_to_batch_dict(record))
        except ValueError as exc:
            raise ValueError(f"Line {line_no}: {exc}") from exc

    if not rows:
        raise ValueError(f"No rows parsed from {args.input}")

    import pandas as pd

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(f"Wrote {len(rows)} solver rows to {args.output} (label_source=toolpoker_solver)")


if __name__ == "__main__":
    main()
