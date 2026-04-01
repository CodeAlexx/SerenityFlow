"""Ring-buffer log handler for structured log capture.

Stores the last N log entries in memory for query via the debug API.
Attach to the root ``serenityflow`` logger at startup.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterator

__all__ = ["LogEntry", "RingBufferHandler", "get_handler", "install"]


@dataclass(slots=True)
class LogEntry:
    timestamp: str
    level: str
    component: str
    message: str


class RingBufferHandler(logging.Handler):
    """Logging handler that stores entries in a fixed-size ring buffer."""

    def __init__(self, capacity: int = 10_000):
        super().__init__()
        self.buffer: deque[LogEntry] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = LogEntry(
                timestamp=datetime.fromtimestamp(
                    record.created, tz=timezone.utc
                ).isoformat(),
                level=record.levelname,
                component=record.name.rsplit(".", 1)[-1] if "." in record.name else record.name,
                message=self.format(record),
            )
            self.buffer.append(entry)
        except Exception:
            self.handleError(record)

    def get_entries(
        self,
        n: int = 100,
        level: str | None = None,
        component: str | None = None,
    ) -> list[dict]:
        """Return the last *n* matching entries as dicts."""
        entries: Iterator[LogEntry] = iter(self.buffer)

        if level:
            min_level = getattr(logging, level.upper(), 0)
            entries = (e for e in entries if getattr(logging, e.level, 0) >= min_level)

        if component:
            comp_lower = component.lower()
            entries = (e for e in entries if comp_lower in e.component.lower())

        # Materialize then take the tail
        matched = list(entries)
        return [asdict(e) for e in matched[-n:]]

    @property
    def total(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_handler: RingBufferHandler | None = None


def get_handler() -> RingBufferHandler | None:
    return _handler


def install(capacity: int = 10_000, log_format: str | None = None) -> RingBufferHandler:
    """Create the singleton handler and attach it to the ``serenityflow`` logger."""
    global _handler
    if _handler is not None:
        return _handler

    _handler = RingBufferHandler(capacity=capacity)
    fmt = log_format or "%(message)s"
    _handler.setFormatter(logging.Formatter(fmt))

    sf_logger = logging.getLogger("serenityflow")
    sf_logger.addHandler(_handler)

    return _handler
