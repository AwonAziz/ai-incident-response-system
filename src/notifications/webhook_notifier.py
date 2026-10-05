"""Generic JSON webhook notifier.

Point ``WEBHOOK_URL`` at any endpoint (Slack-compatible, Splunk HEC, an internal
gateway) to receive the full incident document.
"""

from __future__ import annotations

from typing import Any, ClassVar

from src.core.enums import Severity
from src.notifications.base import BaseNotifier
from src.notifications.http_client import post_json
from src.triage.incident_manager import Incident

__all__ = ["WebhookNotifier"]


class WebhookNotifier(BaseNotifier):
    """POSTs ``incident.to_dict()`` to an arbitrary URL."""

    name: ClassVar[str] = "webhook"
    default_severities: ClassVar[tuple[Severity, ...]] = tuple(Severity)

    def __init__(self, url: str | None = None, *, headers: dict[str, str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.url = (url or "").strip()
        self.headers = headers or {}

    @property
    def is_configured(self) -> bool:
        return bool(self.url)

    def payload(self, incident: Incident) -> dict[str, Any]:
        return {
            "event": "incident.detected",
            "source": "ai-incident-response-system",
            "incident": incident.to_dict(),
        }

    def body(self, incident: Incident) -> dict[str, Any]:
        return self.payload(incident)

    def _deliver(self, incident: Incident) -> str:
        return post_json(self.url, self.payload(incident), timeout=self.timeout, headers=self.headers)
