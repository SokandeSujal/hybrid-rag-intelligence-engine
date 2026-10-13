"""Minimal structured event logging; payloads intentionally avoid raw user content."""
from __future__ import annotations

import json
import logging
from typing import Any


def emit_event(logger: logging.Logger, event_name: str, **fields: Any) -> None:
    """Emit a one-line JSON record suitable for parsing by log processors."""
    record = {"event": event_name, **fields}
    logger.info(json.dumps(record, sort_keys=True, default=str))
