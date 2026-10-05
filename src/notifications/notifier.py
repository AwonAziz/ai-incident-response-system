"""Severity routing for notification channels.

Every send returns a :class:`~src.notifications.base.NotificationResult` rather
than raising: a misconfigured or unreachable channel must never take down the
detection pipeline. Retries, backoff and per-channel cooldown live once in
:meth:`~src.notifications.base.BaseNotifier.send` instead of being reimplemented
per channel.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from src.core.clock import Clock, SystemClock
from src.notifications.base import (
    STATUS_FAILED,
    BaseNotifier,
    ConsoleNotifier,
    NotificationResult,
)
from src.triage.incident_manager import Incident

__all__ = ["ConsoleNotifier", "NotificationRouter"]

logger = logging.getLogger(__name__)
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
