"""Notification layer: severity routing with pluggable delivery channels.

Channels are only constructed when their credentials are present, so a
misconfigured or missing Slack/PagerDuty setup degrades to a console-only run
instead of crashing at startup. Unconfigured channels still report as
``skipped`` so the dashboard can show exactly what was not delivered.
"""

from __future__ import annotations

from src.notifications.email_notifier import EmailNotifier
from src.notifications.http_client import DeliveryError
from src.notifications.notifier import (
    BaseNotifier,
    ConsoleNotifier,
    NotificationResult,
    NotificationRouter,
)
from src.notifications.pagerduty_notifier import PagerDutyNotifier
from src.notifications.slack_notifier import SlackNotifier
from src.notifications.webhook_notifier import WebhookNotifier

__all__ = [
    "BaseNotifier",
    "ConsoleNotifier",
    "DeliveryError",
    "EmailNotifier",
    "NotificationResult",
    "NotificationRouter",
    "PagerDutyNotifier",
    "SlackNotifier",
    "WebhookNotifier",
]
