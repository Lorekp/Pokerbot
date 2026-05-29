"""Structured JSON logging for inference server."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            payload["request_id"] = record.request_id
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            payload.update(record.extra_fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_inference_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger("aof_rl.inference")
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)


def log_request(
    logger: logging.Logger,
    *,
    request_id: str,
    path: str,
    latency_ms: float,
    status_code: int,
    ood_alert: bool | None = None,
    model_id: str | None = None,
) -> None:
    extra = {
        "request_id": request_id,
        "extra_fields": {
            "path": path,
            "latency_ms": round(latency_ms, 3),
            "status_code": status_code,
            "ood_alert": ood_alert,
            "model_id": model_id,
        },
    }
    logger.info("request_complete", extra=extra)
