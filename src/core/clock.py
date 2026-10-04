"""Clock abstraction.

Deduplication windows, SLA maths, rate limiting and auto-resolution are all
time dependent. Routing them through :class:`Clock` keeps those components
deterministic under test without patching ``datetime`` globally.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "ManualClock", "SystemClock", "sleep", "utc_now"]


def utc_now() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(timezone.utc)


def sleep(seconds: float) -> None:  # pragma: no cover - trivial
    """Real sleep, split out so clocks can override it."""
    import time

    time.sleep(max(0.0, seconds))


@runtime_checkable
class Clock(Protocol):
    """Minimal time source."""

    def now(self) -> datetime:  # pragma: no cover - protocol definition
        ...

    def sleep(self, seconds: float) -> None:  # pragma: no cover - protocol definition
        ...


class SystemClock:
    """Wall-clock time source (default)."""

    def now(self) -> datetime:
        return utc_now()

    def sleep(self, seconds: float) -> None:
        sleep(seconds)


class ManualClock:
    """Deterministic clock for tests and replay tooling."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or utc_now()

    def now(self) -> datetime:
        return self._now

    def sleep(self, seconds: float) -> datetime:
        """Advance instead of blocking, so retry backoff is instant in tests."""
        return self.advance(seconds)

    def advance(self, seconds: float) -> datetime:
        self._now = self._now + timedelta(seconds=seconds)
        return self._now

    def set(self, moment: datetime) -> None:
        self._now = moment
