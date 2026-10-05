"""Notification plumbing: the result type and the base channel contract.

Split from :mod:`src.notifications.notifier` so the import graph stays one-way:

    notifier -> slack / pagerduty / email / webhook -> base
    base     -> incident_manager

The router imports channels lazily inside ``__init__`` so a deployment that only
configures Slack never imports the SMTP or PagerDuty paths, and so no channel
ends up importing the router that constructs it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from src.core.clock import Clock, SystemClock
from src.core.enums import Severity
from src.triage.incident_manager import Incident

__all__ = [
    "STATUS_FAILED",
    "STATUS_RATE_LIMITED",
    "STATUS_SENT",
    "STATUS_SKIPPED",
    "STATUS_SUPPRESSED",
    "BaseNotifier",
    "ConsoleNotifier",
    "NotificationResult",
]

logger = logging.getLogger(__name__)

STATUS_SENT = "sent"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUS_SUPPRESSED = "suppressed"
STATUS_RATE_LIMITED = "rate_limited"


@dataclass(frozen=True, slots=True)
class NotificationResult:
    """Outcome of a single delivery attempt sequence."""

    notifier: str
    incident_id: str
    severity: str
    status: str
    detail: str = ""
    attempts: int = 0
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == STATUS_SENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "notifier": self.notifier,
            "incident_id": self.incident_id,
            "severity": self.severity,
            "status": self.status,
            "detail": self.detail,
            "attempts": self.attempts,
            "duration_ms": round(self.duration_ms, 3),
        }


class BaseNotifier(ABC):
    """Common delivery machinery: severity filter, cooldown, retries, dry run."""

    name: ClassVar[str] = "base"
    default_severities: ClassVar[tuple[Severity, ...]] = tuple(Severity)

    def __init__(
        self,
        severities: Sequence[Severity] | None = None,
        *,
        timeout: float = 5.0,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
        cooldown_seconds: float = 5.0,
        clock: Clock | None = None,
    ) -> None:
        self.severities = tuple(severities) if severities is not None else tuple(self.default_severities)
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.clock: Clock = clock or SystemClock()
        self._last_sent: float | None = None
        self.stats: dict[str, int] = {"sent": 0, "failed": 0, "skipped": 0, "suppressed": 0, "rate_limited": 0}

    # ── configuration hooks ────────────────────────────────────────────
    @property
    def is_configured(self) -> bool:
        """``False`` means dry run: payloads are built but nothing is sent."""
        return True

    def handles(self, severity: Severity) -> bool:
        return severity in self.severities

    # ── delivery ───────────────────────────────────────────────────────
    def send(self, incident: Incident) -> NotificationResult:
        severity = incident.severity
        started = self.clock.now()

        def result(status: str, detail: str = "", attempts: int = 0) -> NotificationResult:
            duration = max(0.0, (self.clock.now() - started).total_seconds() * 1000.0)
            return NotificationResult(
                notifier=self.name,
                incident_id=incident.id,
                severity=incident.severity.name,
                status=status,
                detail=detail,
                attempts=attempts,
                duration_ms=duration,
            )

        if not self.handles(severity):
            self.stats["suppressed"] += 1
            return result(STATUS_SUPPRESSED, "severity not routed to this channel")
        if not self.is_configured:
            self.stats["skipped"] += 1
            return result(STATUS_SKIPPED, "channel not configured (dry run)")
        if self._in_cooldown():
            self.stats["rate_limited"] += 1
            return result(STATUS_RATE_LIMITED, "cooldown active")

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                detail = self._deliver(incident)
            except Exception as exc:
                last_error = exc
                logger.warning("%s delivery failed (attempt %d): %s", self.name, attempt, exc)
                if attempt > self.max_retries:
                    break
                self.clock.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
                continue
            self.stats["sent"] += 1
            self._last_sent = self.clock.now().timestamp()
            return result(STATUS_SENT, detail, attempts=attempt)

        self.stats["failed"] += 1
        message = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown error"
        return result(STATUS_FAILED, message, attempts=self.max_retries + 1)

    @abstractmethod
    def _deliver(self, incident: Incident) -> str:
        """Perform the actual delivery, returning a short detail string."""

    def preview(self, incident: Incident) -> dict[str, Any]:
        """Dry-run payload preview (used by ``--notify-preview`` and the API)."""
        return {"notifier": self.name, "severities": [s.name for s in self.severities], "body": self.body(incident)}

    def body(self, incident: Incident) -> Any:  # pragma: no cover - overridden where useful
        return incident.summary_line()

    # ── internals ──────────────────────────────────────────────────────
    def _in_cooldown(self) -> bool:
        if self._last_sent is None or self.cooldown_seconds <= 0:
            return False
        return (self.clock.now().timestamp() - self._last_sent) < self.cooldown_seconds




class ConsoleNotifier(BaseNotifier):
    """Human-facing terminal output; always configured."""

    name: ClassVar[str] = "console"
    default_severities: ClassVar[tuple[Severity, ...]] = tuple(Severity)

    #: ANSI colours per severity, disabled automatically by the router in quiet mode
    COLOURS: ClassVar[dict[str, str]] = {
        "CRITICAL": "\033[1;41m",
        "HIGH": "\033[1;31m",
        "MEDIUM": "\033[1;33m",
        "LOW": "\033[1;36m",
    }
    RESET: ClassVar[str] = "\033[0m"

    def __init__(self, *args: Any, quiet_mode: bool = False, colour: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.quiet_mode = bool(quiet_mode)
        self.colour = bool(colour)
        if self.quiet_mode and Severity.LOW in self.severities:
            self.severities = tuple(s for s in self.severities if s is not Severity.LOW)

    @property
    def is_configured(self) -> bool:
        return True

    def body(self, incident: Incident) -> str:
        lines = [
            incident.summary_line(),
            f"    {incident.description}",
        ]
        lines.extend(f"    hint: {hint}" for hint in incident.hints[:2])
        if incident.causes:
            primary = incident.causes[0]
            lines.append(f"    likely cause: {primary.cause} ({primary.likelihood:.0%})")
        lines.append(f"    sla: {incident.sla_minutes:.0f}m target, breach={incident.sla_breached}")
        return "\n".join(lines)

    def _deliver(self, incident: Incident) -> str:
        text = self.body(incident)
        if self.colour:
            prefix = self.COLOURS.get(incident.severity.name, "")
            suffix = self.RESET if prefix else ""
            printed = f"{prefix}{text}{suffix}"
        else:
            printed = text
        print(printed, flush=True)
        log = {
            Severity.CRITICAL: logger.critical,
            Severity.HIGH: logger.error,
            Severity.MEDIUM: logger.warning,
            Severity.LOW: logger.info,
        }[incident.severity]
        log("incident %s: %s", incident.id, incident.title)
        return "printed to console"

