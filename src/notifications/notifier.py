"""Notification base classes, console notifier and the severity router.

The router is intentionally defensive: a misconfigured or unreachable channel
must never take down the detection pipeline. Every send returns a
:class:`NotificationResult` describing what happened, and retries/backoff plus
per-channel cooldown are handled once in :meth:`BaseNotifier.send` instead of
being reimplemented per channel.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from src.core.clock import Clock, SystemClock
from src.core.enums import Severity
from src.triage.incident_manager import Incident

__all__ = ["BaseNotifier", "ConsoleNotifier", "NotificationResult", "NotificationRouter"]

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


@dataclass(slots=True)
class _RouterCounters:
    sent: int = 0
    failed: int = 0
    skipped: int = 0
    suppressed: int = 0
    rate_limited: int = 0
    by_notifier: dict[str, int] = field(default_factory=dict)

    def record(self, result: NotificationResult) -> None:
        setattr(self, result.status, getattr(self, result.status, 0) + 1)
        self.by_notifier[result.notifier] = self.by_notifier.get(result.notifier, 0) + 1


class NotificationRouter:
    """Routes incidents to channels based on severity, tracking delivery stats."""

    def __init__(
        self,
        slack_webhook: str | None = None,
        pagerduty_key: str | None = None,
        quiet_mode: bool = False,
        *,
        email_to: str | None = None,
        webhook_url: str | None = None,
        notifiers: Iterable[BaseNotifier] | None = None,
        settings: Any | None = None,
        clock: Clock | None = None,
        console: bool = True,
    ) -> None:
        self.clock: Clock = clock or SystemClock()
        self.quiet_mode = bool(quiet_mode)
        self._counters = _RouterCounters()
        self._notifiers: list[BaseNotifier] = []

        if notifiers is not None:
            for notifier in notifiers:
                self.register(notifier)
            return

        from config.settings import get_settings

        resolved = settings or get_settings()
        timeout = resolved.notifier_timeout_seconds
        retries = resolved.notifier_max_retries
        cooldown = resolved.notifier_cooldown_seconds

        if console:
            self.register(
                ConsoleNotifier(
                    quiet_mode=self.quiet_mode,
                    clock=self.clock,
                    timeout=timeout,
                    max_retries=retries,
                    cooldown_seconds=0.0,
                )
            )
        # Channels are always registered so an unconfigured one shows up as
        # "skipped" (dry run) instead of silently disappearing from the routing
        # table the dashboard renders.
        from src.notifications.email_notifier import EmailNotifier
        from src.notifications.pagerduty_notifier import PagerDutyNotifier
        from src.notifications.slack_notifier import SlackNotifier
        from src.notifications.webhook_notifier import WebhookNotifier

        self.register(
            WebhookNotifier(
                webhook_url or resolved.webhook_url,
                clock=self.clock,
                timeout=timeout,
                max_retries=retries,
                cooldown_seconds=cooldown,
            )
        )
        self.register(
            SlackNotifier(
                slack_webhook or resolved.slack_webhook_url,
                clock=self.clock,
                timeout=timeout,
                max_retries=retries,
                cooldown_seconds=cooldown,
            )
        )
        self.register(
            PagerDutyNotifier(
                pagerduty_key or resolved.pagerduty_routing_key,
                clock=self.clock,
                timeout=timeout,
                max_retries=retries,
                cooldown_seconds=cooldown,
            )
        )
        self.register(
            EmailNotifier(
                email_to or resolved.email_to,
                sender=resolved.email_from or None,
                smtp_host=resolved.email_smtp_host,
                smtp_port=resolved.email_smtp_port,
                username=resolved.email_username,
                password=resolved.email_password,
                clock=self.clock,
                timeout=timeout,
                max_retries=retries,
                cooldown_seconds=cooldown,
            )
        )

    # ── wiring ─────────────────────────────────────────────────────────
    def register(self, notifier: BaseNotifier) -> NotificationRouter:
        self._notifiers.append(notifier)
        return self

    def unregister(self, name: str) -> None:
        self._notifiers = [item for item in self._notifiers if item.name != name]

    @property
    def notifiers(self) -> tuple[BaseNotifier, ...]:
        return tuple(self._notifiers)

    def get(self, name: str) -> BaseNotifier | None:
        for notifier in self._notifiers:
            if notifier.name == name:
                return notifier
        return None

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "channels": [notifier.name for notifier in self._notifiers],
            "sent": self._counters.sent,
            "failed": self._counters.failed,
            "skipped": self._counters.skipped,
            "suppressed": self._counters.suppressed,
            "rate_limited": self._counters.rate_limited,
            "by_notifier": dict(self._counters.by_notifier),
        }

    # ── routing ────────────────────────────────────────────────────────
    def notify(self, incident: Incident) -> list[NotificationResult]:
        """Fan one incident out to every channel that handles its severity."""
        results: list[NotificationResult] = []
        for notifier in self._notifiers:
            try:
                result = notifier.send(incident)
            except Exception as exc:
                logger.exception("notifier %s raised", notifier.name)
                result = NotificationResult(
                    notifier=notifier.name,
                    incident_id=incident.id,
                    severity=incident.severity.name,
                    status=STATUS_FAILED,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            self._counters.record(result)
            results.append(result)

        if any(result.ok for result in results) and incident.severity.name not in incident.notified_severities:
            incident.notified_severities.append(incident.severity.name)
        return results

    def preview(self, incident: Incident) -> list[dict[str, Any]]:
        """What would be sent, without sending (CLI/API dry run)."""
        return [notifier.preview(incident) for notifier in self._notifiers]

    def reset_stats(self) -> None:
        self._counters = _RouterCounters()
        for notifier in self._notifiers:
            notifier.stats.update({"sent": 0, "failed": 0, "skipped": 0, "suppressed": 0, "rate_limited": 0})
