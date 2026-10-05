"""Slack incoming-webhook notifier."""

from __future__ import annotations

from typing import Any, ClassVar

from src.core.enums import Severity
from src.notifications.base import BaseNotifier
from src.notifications.http_client import post_json
from src.triage.incident_manager import Incident

__all__ = ["SlackNotifier"]

_SEVERITY_EMOJI: dict[str, str] = {
    "CRITICAL": ":rotating_light:",
    "HIGH": ":large_orange_diamond:",
    "MEDIUM": ":warning:",
    "LOW": ":information_source:",
}


class SlackNotifier(BaseNotifier):
    """Posts Block Kit incident cards to a Slack incoming webhook."""

    name: ClassVar[str] = "slack"
    default_severities: ClassVar[tuple[Severity, ...]] = (
        Severity.CRITICAL,
        Severity.HIGH,
        Severity.MEDIUM,
    )

    def __init__(self, webhook_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.webhook_url = (webhook_url or "").strip()

    @property
    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    # ── payload ────────────────────────────────────────────────────────
    def blocks(self, incident: Incident) -> list[dict[str, Any]]:
        emoji = _SEVERITY_EMOJI.get(incident.severity.name, ":grey_question:")
        header = f"{emoji} {incident.severity.name} - {incident.title}"
        fields = [
            {"type": "mrkdwn", "text": f"*Cloud*\n{incident.cloud.label} ({incident.region or 'n/a'})"},
            {"type": "mrkdwn", "text": f"*Service*\n{incident.service} / `{incident.resource_id}`"},
            {"type": "mrkdwn", "text": f"*Metric*\n`{incident.metric_name}` = {incident.metric_value:.2f}{incident.unit}"},
            {"type": "mrkdwn", "text": f"*Occurrences*\n{incident.occurrences}x since {incident.created_at:%H:%M:%S}Z"},
            {"type": "mrkdwn", "text": f"*Score / confidence*\n{incident.score:.2f} / {incident.confidence:.0%}"},
            {
                "type": "mrkdwn",
                "text": f"*SLA*\n{incident.sla_minutes:.0f}m ack target"
                f"{' - AT RISK' if incident.sla_breached else ''}",
            },
        ]
        blocks: list[dict[str, Any]] = [
            {"type": "header", "text": {"type": "plain_text", "text": header[:150]}},
            {"type": "section", "text": {"type": "mrkdwn", "text": incident.description}},
            {"type": "section", "fields": fields},
        ]
        causes = [f"*{cause.cause}* ({cause.likelihood:.0%}) - {cause.evidence[0]}" for cause in incident.causes if cause.evidence]
        if causes:
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "*Likely causes*\n" + "\n".join(f"- {line}" for line in causes[:3])},
                }
            )
        breached = ", ".join(sorted({violation.get("rule_id", "") for violation in incident.violations}))
        if breached:
            blocks.append(
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": f"rules: {breached}"}],
                }
            )
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"incident `{incident.id}` | fingerprint `{incident.fingerprint}`"}],
            }
        )
        return blocks

    def payload(self, incident: Incident) -> dict[str, Any]:
        return {
            "text": f"[{incident.severity.name}] {incident.title}",
            "blocks": self.blocks(incident),
        }

    def body(self, incident: Incident) -> dict[str, Any]:
        return self.payload(incident)

    def _deliver(self, incident: Incident) -> str:
        return post_json(self.webhook_url, self.payload(incident), timeout=self.timeout)
