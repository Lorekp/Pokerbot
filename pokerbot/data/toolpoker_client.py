"""ToolPoker live API client with JSONL cache (Phase 8)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass
class ToolPokerClientConfig:
    base_url: str = "https://api.toolpoker.example/v1"
    api_key: str | None = None
    cache_dir: Path = Path("data/toolpoker/cache")
    timeout_s: float = 30.0


class ToolPokerClient:
    """Fetch solver spots from ToolPoker API; cache responses as JSONL."""

    def __init__(self, config: ToolPokerClientConfig | None = None) -> None:
        self.config = config or ToolPokerClientConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, spot_id: str) -> Path:
        safe = spot_id.replace("/", "_").replace("\\", "_")
        return self.config.cache_dir / f"{safe}.jsonl"

    def fetch_spot(self, spot_id: str, *, use_cache: bool = True) -> dict[str, Any]:
        cache = self._cache_path(spot_id)
        if use_cache and cache.exists():
            # CRASH-RESUME-AUDIT #7 + recovery: read is now tolerant of a torn first line (the
            # previous crash mode was a single in-place write_text that left a partial JSON).
            # We skip empty / unparseable lines and fall through to a network re-fetch.
            try:
                text = cache.read_text(encoding="utf-8")
            except OSError:
                text = ""
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except (ValueError, json.JSONDecodeError):
                    # torn first line — fall through to re-fetch
                    break

        url = f"{self.config.base_url.rstrip('/')}/spots/{spot_id}"
        headers = {"Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = Request(url, headers=headers, method="GET")
        try:
            with urlopen(req, timeout=self.config.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except URLError as exc:
            raise ConnectionError(f"ToolPoker API unavailable: {exc}") from exc

        # CRASH-RESUME-AUDIT #7: write the cache atomically (tmp + fsync + os.replace + dir-fsync).
        # Previously ``cache.write_text`` was in-place; a power cut mid-write left a torn JSON file
        # and every subsequent fetch_spot() call raised json.loads. The atomic rewrite protects the
        # $ + rate-limit cost of an unnecessary re-fetch.
        tmp = cache.with_name(cache.name + ".atomic.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, cache)
        try:
            dir_fd = os.open(str(cache.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
        return payload

    def append_jsonl(self, records: list[dict[str, Any]], output: Path) -> Path:
        """Write solver records to JSONL for export_toolpoker_parquet.py."""
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return output
