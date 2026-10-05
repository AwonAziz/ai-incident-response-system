"""PagerDuty Events API v2 notifier."""

from __future__ import annotations

from typing import Any, ClassVar

from src.core.enums import Severity
from src.notifications.base import BaseNotifier
from src.notifications.http_client import post_json
from src.triage.incident_manager import Incident

__all__ = ["EVENTS_API_URL", "PagerDutyNotifier"]

EVENTS_API_URL = "https://events.pagerduty.com/v2/enqueue"

_EVENT_FOR_STATUS: dict[str, str] = {"open": "trigger", "acknowledged": "acknowledge", "resolved": "resolve"}


class PagerDutyNotifier(BaseNotifier):
    """Triggers PagerDuty incidents, deduplicated on the incident fingerprint."""

    name: ClassVar[str] = "pagerduty"
    default_severities: ClassVar[tuple[Severity, ...]] = (Severity.CRITICAL, Severity.HIGH)

    def __init__(self, routing_key: str | None = None, *, api_url: str = EVENTS_API_URL, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.routing_key = (routing_key or "").strip()
        self.api_url = api_url

    @property
    def is_configured(self) -> bool:
        return bool(self.routing_key)

    def dedup_key(self, incident: Incident) -> str:
        """One PagerDuty incident per (cloud, service, metric) - not per alert."""
        return f"aiops-{incident.fingerprint}"

    def payload(self, incident: Incident) -> dict[str, Any]:
        return {
            "routing_key": self.routing_key,
            "event_action": _EVENT_FOR_STATUS.get(incident.status.value, "trigger"),
            "dedup_key": self.dedup_key(incident),
            "client": "ai-incident-response-system",
            "payload": {
                "summary": f"{incident.severity.name}: {incident.title}"[:1024],
                "source": f"{incident.cloud.value}/{incident.service}",
                "severity": "critical" if incident.severity is Severity.CRITICAL else "error",
                "component": incident.metric_name,
                "group": incident.cloud.value,
                "class": incident.causes[0].category if incident.causes else "anomaly",
                "custom_details": {
                    "incident_id": incident.id,
                    "resource_id": incident.resource_id,
                    "region": incident.region,
                    "value": incident.metric_value,
                    "unit": incident.unit,
                    "score": incident.score,
                    "confidence": round(incident.confidence, 4),
                    "occurrences": incident.occurrences,
                    "hints": incident.hints,
                },
            },
        }

    def body(self, incident: Incident) -> dict[str, Any]:
        return self.payload(incident)

    def _deliver(self, incident: Incident) -> str:
        return post_json(self.api_url, self.payload(incident), timeout=self.timeout)
