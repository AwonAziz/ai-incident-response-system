"""Notification channels, severity routing, retries and rate limiting."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.core.clock import ManualClock
from src.core.enums import Severity
from src.notifications import (
    BaseNotifier,
    ConsoleNotifier,
    EmailNotifier,
    NotificationResult,
    NotificationRouter,
    PagerDutyNotifier,
    SlackNotifier,
    WebhookNotifier,
)
from src.notifications.http_client import DeliveryError, post_json


class _RecordingNotifier(BaseNotifier):
    name = "recording"
    calls: list[object] = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls = []

    def _deliver(self, incident) -> str:
        self.calls.append(incident)
        return "ok"


class _FlakyNotifier(BaseNotifier):
    name = "flaky"
    default_severities = tuple(Severity)

    def __init__(self, failures: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.failures = failures
        self.attempts = 0

    def _deliver(self, incident) -> str:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise DeliveryError(f"attempt {self.attempts} failed")
        return "recovered"


class TestNotificationResult:
    def test_ok_flag(self) -> None:
        assert NotificationResult("slack", "INC-1", "HIGH", "sent").ok
        assert not NotificationResult("slack", "INC-1", "HIGH", "failed").ok

    def test_serialisation(self) -> None:
        payload = NotificationResult("slack", "INC-1", "HIGH", "sent", "HTTP 200", 1, 12.3456).to_dict()
        assert payload["status"] == "sent"
        assert payload["duration_ms"] == 12.346


class TestBaseNotifier:
    def test_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseNotifier()  # type: ignore[abstract]

    def test_severity_filter(self, incident, clock: ManualClock) -> None:
        notifier = _RecordingNotifier(severities=[Severity.CRITICAL], clock=clock)
        result = notifier.send(replace(incident, severity=Severity.LOW))
        assert result.status == "suppressed"
        assert notifier.calls == []

    def test_retries_then_succeeds(self, incident, clock: ManualClock) -> None:
        notifier = _FlakyNotifier(failures=1, clock=clock, max_retries=2)
        result = notifier.send(incident)
        assert result.status == "sent"
        assert result.attempts == 2
        assert notifier.attempts == 2
        assert notifier.stats["sent"] == 1

    def test_retry_budget_is_exhausted(self, incident, clock: ManualClock) -> None:
        notifier = _FlakyNotifier(failures=99, clock=clock, max_retries=1)
        result = notifier.send(incident)
        assert result.status == "failed"
        assert result.attempts == 2
        assert "DeliveryError" in result.detail

    def test_backoff_does_not_block_a_manual_clock(self, incident, clock: ManualClock) -> None:
        notifier = _FlakyNotifier(failures=1, clock=clock, max_retries=2, backoff_seconds=30.0)
        before = clock.now()
        assert notifier.send(incident).status == "sent"
        assert clock.now() > before  # time advanced instead of sleeping

    def test_cooldown_rate_limits(self, incident, clock: ManualClock) -> None:
        notifier = _RecordingNotifier(clock=clock, cooldown_seconds=30.0)
        assert notifier.send(incident).status == "sent"
        assert notifier.send(incident).status == "rate_limited"
        clock.advance(31)
        assert notifier.send(incident).status == "sent"
        assert len(notifier.calls) == 2

    def test_zero_cooldown_never_limits(self, incident, clock: ManualClock) -> None:
        notifier = _RecordingNotifier(clock=clock, cooldown_seconds=0.0)
        assert notifier.send(incident).status == "sent"
        assert notifier.send(incident).status == "sent"

    def test_unconfigured_channels_are_skipped(self, incident, clock: ManualClock) -> None:
        notifier = SlackNotifier("", clock=clock)
        result = notifier.send(incident)
        assert result.status == "skipped"
        assert "dry run" in result.detail


class TestConsoleNotifier:
    def test_prints_incident_body(self, incident, capsys) -> None:
        result = ConsoleNotifier(clock=ManualClock(), colour=False).send(incident)
        assert result.status == "sent"
        output = capsys.readouterr().out
        assert incident.id in output
        assert "hint:" in output
        assert "likely cause:" in output
        assert "sla:" in output

    def test_colour_wrapping(self, incident, capsys) -> None:
        ConsoleNotifier(clock=ManualClock(), colour=True).send(incident)
        assert "\033[1;41m" in capsys.readouterr().out

    def test_quiet_mode_drops_low_severity(self, incident, clock: ManualClock) -> None:
        notifier = ConsoleNotifier(quiet_mode=True, clock=clock)
        assert Severity.LOW not in notifier.severities
        assert notifier.send(replace(incident, severity=Severity.LOW)).status == "suppressed"
        assert notifier.send(replace(incident, severity=Severity.HIGH)).status == "sent"


class TestSlackNotifier:
    def test_payload_blocks(self, incident) -> None:
        payload = SlackNotifier("https://hooks.slack.test/x").payload(incident)
        assert payload["text"].startswith("[CRITICAL]")
        kinds = [block["type"] for block in payload["blocks"]]
        assert kinds[0] == "header"
        assert "Likely causes" in json.dumps(payload)
        assert incident.fingerprint in json.dumps(payload)

    def test_payload_includes_breached_rules(self, incident) -> None:
        incident = replace(incident, violations=[{"rule_id": "cpu_utilization.critical_high"}])
        assert "cpu_utilization.critical_high" in json.dumps(SlackNotifier("u").payload(incident))

    def test_low_severity_is_not_routed_by_default(self, incident) -> None:
        assert SlackNotifier("u").send(replace(incident, severity=Severity.LOW)).status == "suppressed"

    def test_delivery_uses_post_json(self, incident, monkeypatch) -> None:
        captured: dict = {}

        def fake_post(url, payload, timeout=5.0, headers=None):
            captured.update(url=url, payload=payload, timeout=timeout)
            return "HTTP 200"

        monkeypatch.setattr("src.notifications.slack_notifier.post_json", fake_post)
        result = SlackNotifier("https://hooks.slack.test/x", timeout=2.5, clock=ManualClock()).send(incident)
        assert result.status == "sent"
        assert captured["url"] == "https://hooks.slack.test/x"
        assert captured["timeout"] == 2.5
        assert captured["payload"]["blocks"]


class TestPagerDutyNotifier:
    def test_payload_and_dedup_key(self, incident) -> None:
        notifier = PagerDutyNotifier("routing-key")
        payload = notifier.payload(incident)
        assert payload["routing_key"] == "routing-key"
        assert payload["event_action"] == "trigger"
        assert payload["dedup_key"] == f"aiops-{incident.fingerprint}"
        assert payload["payload"]["severity"] == "critical"
        assert payload["payload"]["custom_details"]["occurrences"] == incident.occurrences

    def test_resolved_incident_resolves_the_pager(self, incident) -> None:
        from src.core.enums import IncidentStatus

        resolved = replace(incident, status=IncidentStatus.RESOLVED)
        assert PagerDutyNotifier("k").payload(resolved)["event_action"] == "resolve"

    def test_high_and_critical_only(self, incident, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.notifications.pagerduty_notifier.post_json", lambda *a, **k: "HTTP 202"
        )
        notifier = PagerDutyNotifier("k")
        assert notifier.send(replace(incident, severity=Severity.MEDIUM)).status == "suppressed"
        assert notifier.send(replace(incident, severity=Severity.HIGH)).status == "sent"

    def test_dedup_key_is_stable_across_incidents(self, incident) -> None:
        notifier = PagerDutyNotifier("k")
        twin = replace(incident, id="INC-99999", occurrences=9)
        assert notifier.dedup_key(incident) == notifier.dedup_key(twin)

    def test_unconfigured(self, incident) -> None:
        assert PagerDutyNotifier("").send(incident).status == "skipped"


class TestEmailNotifier:
    def test_body_contains_actionable_content(self, incident) -> None:
        notifier = EmailNotifier(
            "sre@example.com, oncall@example.com",
            smtp_host="smtp.example.com",
            smtp_port=587,
        )
        body = notifier.body(incident)
        assert "Next actions:" in body
        assert "Likely causes:" in body
        assert "SLA: 15m" in body
        assert notifier.recipients == ["sre@example.com", "oncall@example.com"]

    def test_message_headers(self, incident) -> None:
        notifier = EmailNotifier("sre@example.com", smtp_host="smtp.example.com", sender="bot@example.com")
        message = notifier.message(incident)
        assert message["To"] == "sre@example.com"
        assert message["From"] == "bot@example.com"
        assert message["Subject"].startswith("[CRITICAL]")

    def test_requires_recipients_and_host(self, incident) -> None:
        assert not EmailNotifier("", smtp_host="x").is_configured
        assert not EmailNotifier("a@b.c", smtp_host="").is_configured
        assert EmailNotifier("a@b.c", smtp_host="x").is_configured

    def test_unconfigured_is_skipped(self, incident) -> None:
        assert EmailNotifier("").send(incident).status == "skipped"

    def test_low_severity_not_routed(self, incident) -> None:
        notifier = EmailNotifier("a@b.c", smtp_host="x")
        assert notifier.send(replace(incident, severity=Severity.LOW)).status == "suppressed"


class TestWebhookNotifier:
    def test_payload_shape(self, incident) -> None:
        payload = WebhookNotifier("https://hooks.example.com/x").payload(incident)
        assert payload["event"] == "incident.detected"
        assert payload["incident"]["id"] == incident.id

    def test_custom_headers_are_passed(self, incident, monkeypatch) -> None:
        captured: dict = {}
        monkeypatch.setattr(
            "src.notifications.webhook_notifier.post_json",
            lambda url, payload, timeout=5.0, headers=None: captured.setdefault("headers", headers) or "HTTP 200",
        )
        WebhookNotifier("https://x.test", headers={"X-Token": "abc"}, clock=ManualClock()).send(incident)
        assert captured["headers"] == {"X-Token": "abc"}


class TestHttpClient:
    def test_maps_http_errors(self, monkeypatch) -> None:
        from urllib.error import HTTPError

        def boom(*args, **kwargs):
            raise HTTPError("u", 500, "server error", {}, None)

        monkeypatch.setattr("src.notifications.http_client.urlopen", boom)
        with pytest.raises(DeliveryError, match="HTTP 500"):
            post_json("https://x.test", {})

    def test_maps_connection_errors(self, monkeypatch) -> None:
        from urllib.error import URLError

        def boom(*args, **kwargs):
            raise URLError("unreachable")

        monkeypatch.setattr("src.notifications.http_client.urlopen", boom)
        with pytest.raises(DeliveryError, match="URLError"):
            post_json("https://x.test", {})

    def test_success_returns_status(self, monkeypatch) -> None:
        class FakeResponse:
            status = 202

            def read(self) -> bytes:
                return b"queued"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        monkeypatch.setattr("src.notifications.http_client.urlopen", lambda *a, **k: FakeResponse())
        assert "202" in post_json("https://x.test", {"a": 1})


class TestNotificationRouter:
    def test_default_channels_are_registered(self, settings, clock: ManualClock) -> None:
        router = NotificationRouter(settings=settings, clock=clock, console=True)
        assert [notifier.name for notifier in router.notifiers] == [
            "console",
            "webhook",
            "slack",
            "pagerduty",
            "email",
        ]

    def test_unconfigured_channels_report_skipped(self, incident, settings, clock: ManualClock) -> None:
        router = NotificationRouter(settings=settings, clock=clock, console=False)
        results = router.notify(incident)
        statuses = {result.notifier: result.status for result in results}
        assert statuses == {
            "webhook": "skipped",
            "slack": "skipped",
            "pagerduty": "skipped",
            "email": "skipped",
        }
        assert router.stats["skipped"] == 4

    def test_console_delivery_is_recorded(self, incident, settings, clock: ManualClock) -> None:
        router = NotificationRouter(settings=settings, clock=clock)
        results = router.notify(incident)
        assert results[0].status == "sent"
        assert router.stats["sent"] == 1
        assert router.stats["by_notifier"]["console"] == 1

    def test_severity_routing_per_channel(self, incident, settings, clock: ManualClock) -> None:
        router = NotificationRouter(settings=settings, clock=clock, console=False)
        results = router.notify(replace(incident, severity=Severity.LOW))
        statuses = {result.notifier: result.status for result in results}
        assert statuses["slack"] == "suppressed"
        assert statuses["pagerduty"] == "suppressed"
        assert statuses["email"] == "suppressed"
        assert statuses["webhook"] == "skipped"  # unconfigured, but routes everything
        assert router.stats["suppressed"] == 3

    def test_notified_severities_recorded_on_incident(self, incident, settings, clock: ManualClock) -> None:
        router = NotificationRouter(settings=settings, clock=clock)
        router.notify(incident)
        assert "CRITICAL" in incident.notified_severities
        router.notify(incident)
        assert incident.notified_severities.count("CRITICAL") == 1

    def test_exploding_notifier_does_not_break_the_router(self, incident, settings, clock: ManualClock) -> None:
        class Exploding(BaseNotifier):
            name = "boom"

            def _deliver(self, incident) -> str:
                raise RuntimeError("kaboom")

        router = NotificationRouter(settings=settings, clock=clock, console=False)
        router.register(Exploding(clock=clock, max_retries=0))
        results = router.notify(incident)
        by_name = {result.notifier: result for result in results}
        assert by_name["boom"].status == "failed"
        assert "kaboom" in by_name["boom"].detail
        assert by_name["slack"].status == "skipped"

    def test_register_and_unregister(self, settings, clock: ManualClock) -> None:
        router = NotificationRouter(settings=settings, clock=clock, console=False)
        assert router.get("slack") is not None
        router.unregister("slack")
        assert router.get("slack") is None
        assert "slack" not in [notifier.name for notifier in router.notifiers]

    def test_custom_notifier_list_is_used(self, incident, clock: ManualClock) -> None:
        custom = _RecordingNotifier(clock=clock)
        router = NotificationRouter(notifiers=[custom], clock=clock)
        assert router.notify(incident)[0].notifier == "recording"
        assert len(custom.calls) == 1

    def test_preview_does_not_send(self, incident, settings, clock: ManualClock) -> None:
        router = NotificationRouter(settings=settings, clock=clock)
        previews = router.preview(incident)
        assert [item["notifier"] for item in previews] == [notifier.name for notifier in router.notifiers]
        assert router.stats["sent"] == 0

    def test_reset_stats(self, incident, settings, clock: ManualClock) -> None:
        router = NotificationRouter(settings=settings, clock=clock)
        router.notify(incident)
        router.reset_stats()
        assert router.stats["sent"] == 0
        assert router.stats["by_notifier"] == {}

    def test_credentials_select_channels(self, settings, clock: ManualClock) -> None:
        router = NotificationRouter(
            slack_webhook="https://hooks.slack.test/x",
            pagerduty_key="key",
            settings=settings,
            clock=clock,
            console=False,
        )
        assert router.get("slack").is_configured
        assert router.get("pagerduty").is_configured
        assert not router.get("email").is_configured
