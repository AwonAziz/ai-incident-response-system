"""Bounded per-series rolling windows.

A single :class:`SeriesHistory` instance is shared between the collectors (which
populate the rolling statistics attached to every :class:`~src.ingestion.metric_schema.Metric`)
and the feature engineer (which turns those statistics into ML features).
One registry means train-time and serve-time features are computed from exactly
the same numbers.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from threading import RLock

from src.core.stats import RollingStats, stats_from_values

__all__ = ["SeriesHistory"]


class SeriesHistory:
    """Thread-safe registry of bounded value windows keyed by series."""

    def __init__(self, window: int = 30, min_samples: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self.min_samples = max(1, int(min_samples))
        self._values: dict[str, deque[float]] = {}
        self._first_seen: dict[str, datetime] = {}
        self._last_seen: dict[str, datetime] = {}
        self._lock = RLock()

    def record(self, series_key: str, value: float, timestamp: datetime) -> RollingStats:
        """Append ``value`` to the series window and return fresh statistics."""
        with self._lock:
            buffer = self._values.get(series_key)
            if buffer is None:
                buffer = deque(maxlen=self.window)
                self._values[series_key] = buffer
                self._first_seen[series_key] = timestamp
            buffer.append(float(value))
            self._last_seen[series_key] = timestamp
            span = max(0.0, (timestamp - self._first_seen[series_key]).total_seconds())
            return stats_from_values(list(buffer), span_seconds=span)

    def stats(self, series_key: str) -> RollingStats | None:
        """Current statistics for a series, or ``None`` if never recorded."""
        with self._lock:
            buffer = self._values.get(series_key)
            if not buffer:
                return None
            first = self._first_seen.get(series_key)
            last_seen = self._last_seen.get(series_key)
            span = 0.0
            if first is not None and last_seen is not None:
                span = max(0.0, (last_seen - first).total_seconds())
            return stats_from_values(list(buffer), span_seconds=span)

    def values(self, series_key: str) -> tuple[float, ...]:
        with self._lock:
            buffer = self._values.get(series_key)
            return tuple(buffer) if buffer else ()

    def is_ready(self, series_key: str) -> bool:
        with self._lock:
            buffer = self._values.get(series_key)
            return bool(buffer) and len(buffer) >= self.min_samples

    def keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._values)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self._first_seen.clear()
            self._last_seen.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)

    def __contains__(self, series_key: object) -> bool:
        with self._lock:
            return series_key in self._values
