"""Email notifier (SMTP, STARTTLS when the server supports it)."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Any, ClassVar

from src.core.enums import Severity
from src.notifications.base import BaseNotifier
from src.triage.incident_manager import Incident

__all__ = ["EmailNotifier"]


class EmailNotifier(BaseNotifier):
    """Sends a plain-text incident digest to a comma separated recipient list."""

    name: ClassVar[str] = "email"
    default_severities: ClassVar[tuple[Severity, ...]] = (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM)

    def __init__(
        self,
        recipients: str | None = None,
        *,
        sender: str | None = None,
        smtp_host: str | None = None,
        smtp_port: int = 587,
        username: str | None = None,
        password: str | None = None,
        use_tls: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.recipients = [item.strip() for item in (recipients or "").split(",") if item.strip()]
        self.sender = (sender or "ai-incident-response@localhost").strip()
        self.smtp_host = (smtp_host or "").strip()
        self.smtp_port = int(smtp_port)
        self.username = username or ""
        self.password = password or ""
        self.use_tls = bool(use_tls)

    @property
    def is_configured(self) -> bool:
        return bool(self.recipients and self.smtp_host)

    def body(self, incident: Incident) -> str:
        lines = [
            incident.summary_line(),
            "",
            incident.description,
            "",
            "Likely causes:",
            *(f"  - {cause.cause} ({cause.likelihood:.0%}): {cause.evidence[0] if cause.evidence else 'n/a'}" for cause in incident.causes),
            "",
            "Next actions:",
            *(f"  - {hint}" for hint in incident.hints),
            "",
            f"SLA: {incident.sla_minutes:.0f}m acknowledge target"
            f"{' (AT RISK)' if incident.sla_breached else ''}",
            f"Fingerprint: {incident.fingerprint}",
        ]
        return "\n".join(lines)

    def message(self, incident: Incident) -> EmailMessage:
        message = EmailMessage()
        message["Subject"] = f"[{incident.severity.name}] {incident.title}"[:200]
        message["From"] = self.sender
        message["To"] = ", ".join(self.recipients)
        message.set_content(self.body(incident))
        return message

    def _deliver(self, incident: Incident) -> str:
        payload = self.message(incident)
        with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=self.timeout) as client:
            if self.use_tls:
                client.starttls()
            if self.username:
                client.login(self.username, self.password)
            client.send_message(payload)
        return f"sent to {len(self.recipients)} recipient(s)"
