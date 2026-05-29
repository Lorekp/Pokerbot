#!/usr/bin/env python3
"""Fetch ToolPoker solver spots and append to JSONL cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pokerbot.data.toolpoker_client import ToolPokerClient, ToolPokerClientConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch ToolPoker solver data to JSONL")
    p.add_argument("--spot-id", action="append", default=[], help="Spot ID(s) to fetch")
    p.add_argument("--spot-file", default=None, help="Text file with one spot ID per line")
    p.add_argument("--output", default="data/toolpoker/fetched.jsonl")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    spot_ids = list(args.spot_id)
    if args.spot_file:
        spot_ids.extend(Path(args.spot_file).read_text(encoding="utf-8").splitlines())

    if not spot_ids:
        raise SystemExit("Provide --spot-id or --spot-file")

    cfg = ToolPokerClientConfig()
    if args.base_url:
        cfg.base_url = args.base_url
    if args.api_key:
        cfg.api_key = args.api_key
    client = ToolPokerClient(cfg)

    records = []
    for sid in spot_ids:
        sid = sid.strip()
        if not sid:
            continue
        records.append(client.fetch_spot(sid, use_cache=not args.no_cache))

    out = Path(args.output)
    client.append_jsonl(records, out)
    print(f"Wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
