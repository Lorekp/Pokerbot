"""ToolPoker live API client with JSONL cache (Phase 8)."""

from __future__ import annotations

import json
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
            line = cache.read_text(encoding="utf-8").strip().splitlines()[0]
            return json.loads(line)

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

        cache.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return payload

    def append_jsonl(self, records: list[dict[str, Any]], output: Path) -> Path:
        """Write solver records to JSONL for export_toolpoker_parquet.py."""
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return output
