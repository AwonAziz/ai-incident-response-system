"""Shared enumerations.

``Severity`` is an ``IntEnum`` so ordering comparisons work directly
(``Severity.CRITICAL > Severity.HIGH``) without a custom comparator.
"""

from __future__ import annotations

from enum import Enum, IntEnum

__all__ = ["SEVERITY_DESC", "Cloud", "IncidentStatus", "Severity"]


class Cloud(str, Enum):
    """Supported cloud providers."""

    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"

    @classmethod
    def parse(cls, value: Cloud | str) -> Cloud:
        if isinstance(value, cls):
            return value
        token = str(value).strip().lower()
        for member in cls:
            if member.value == token or member.name.lower() == token:
                return member
        raise ValueError(f"unknown cloud {value!r}; expected one of {[c.value for c in cls]}")

    @property
    def label(self) -> str:
        return {"aws": "AWS", "azure": "Azure", "gcp": "GCP"}[self.value]

    @classmethod
    def values(cls) -> list[str]:
        return [member.value for member in cls]


class Severity(IntEnum):
    """Incident severity, ordered from least to most severe."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def parse(cls, value: Severity | str | int) -> Severity:
        if isinstance(value, cls):
            return value
        token = str(value).strip().lower()
        for member in cls:
            if token in (member.name.lower(), str(int(member)), f"{member.name.lower()}-severity"):
                return member
        if token.isdigit() and 1 <= int(token) <= 4:
            return cls(int(token))
        raise ValueError(f"unknown severity {value!r}; expected one of {[s.name.lower() for s in cls]}")

    @property
    def label(self) -> str:
        return self.name

    @property
    def slug(self) -> str:
        return self.name.lower()

    def at_least(self, other: Severity) -> bool:
        return self >= other


SEVERITY_DESC: tuple[Severity, ...] = tuple(sorted(Severity, reverse=True))


class IncidentStatus(str, Enum):
    """Lifecycle state of an incident."""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"

    @classmethod
    def parse(cls, value: IncidentStatus | str) -> IncidentStatus:
        if isinstance(value, cls):
            return value
        token = str(value).strip().lower()
        for member in cls:
            if token == member.value:
                return member
        raise ValueError(f"unknown incident status {value!r}")

    @property
    def is_open(self) -> bool:
        return self in (IncidentStatus.OPEN, IncidentStatus.ACKNOWLEDGED)

    @property
    def is_closed(self) -> bool:
        return self is IncidentStatus.RESOLVED
