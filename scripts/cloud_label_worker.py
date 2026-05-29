#!/usr/bin/env python3
"""Cloud/local label worker scaffold — polls manifest, delegates to build_solver_labels."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pokerbot.solver.label_completion import CompletionLedger


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _claim_next(manifest: dict[str, Any], worker_id: str) -> str | None:
    for key, entry in manifest.get("entries", {}).items():
        if entry.get("status") == "pending":
            entry["status"] = "claimed"
            entry["claimed_by"] = worker_id
            entry["claimed_at"] = datetime.now(timezone.utc).isoformat()
            return key
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Label generation worker (manifest poll scaffold)")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--worker-id", type=str, default=os.environ.get("HOSTNAME", "worker-1"))
    p.add_argument("--max-spots", type=int, default=1)
    args = p.parse_args()

    manifest_path = args.manifest
    ledger_path = args.output_dir / "completed.jsonl"
    claim_file = Path(os.environ.get("LABELGEN_CLAIM_FILE", "/tmp/labelgen_claim.json"))
    manifest = _load_manifest(manifest_path)
    processed = 0

    def _handle_interrupt(signum: int, _frame: Any) -> None:
        print(f"interrupt signal {signum}; releasing claim", flush=True)
        claim_file.unlink(missing_ok=True)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_interrupt)

    while processed < args.max_spots:
        key = _claim_next(manifest, args.worker_id)
        if key is None:
            print("no pending spots; exiting", flush=True)
            break
        _save_manifest(manifest_path, manifest)
        claim_file.write_text(json.dumps({"spot_key": key, "worker": args.worker_id}), encoding="utf-8")

        # Production path: invoke build_solver_labels with --resume for one spot batch.
        # See scripts/plan_label_work.py + infra/labelgen/Dockerfile for full wiring.
        ledger = CompletionLedger.load(ledger_path)
        ledger.append(key, source="claimed")
        manifest["entries"][key]["status"] = "done"
        manifest["entries"][key]["done_at"] = datetime.now(timezone.utc).isoformat()
        _save_manifest(manifest_path, manifest)
        claim_file.unlink(missing_ok=True)
        processed += 1

    print(json.dumps({"processed": processed, "note": "scaffold — wire build_solver_labels per spot"}, indent=2))


if __name__ == "__main__":
    main()
