"""Dashboard rendering (no terminal required)."""

from __future__ import annotations

from dataclasses import replace

from rich.console import Console

from src.core.clock import ManualClock
from src.core.enums import Severity
from src.dashboard.live_dashboard import LiveDashboard, sparkline
from src.ingestion.metric_schema import Metric
from src.triage.incident_manager import IncidentManager


def _render(dashboard: LiveDashboard, width: int = 160) -> str:
    console = Console(width=width, record=True, force_terminal=False, no_color=True)
    console.print(dashboard.get_renderable())
    return console.export_text()


class TestSparkline:
    def test_empty_and_constant(self) -> None:
        assert sparkline([]).strip() == ""
        assert set(sparkline([3.0, 3.0, 3.0])) == {"▁"}

    def test_varied_values_use_multiple_blocks(self) -> None:
        rendered = sparkline([1.0, 5.0, 9.0])
        assert len(rendered) == 3
        assert rendered[0] < rendered[-1]

    def test_width_is_capped(self) -> None:
        assert len(sparkline(list(range(50)), width=8)) == 8


class TestLiveDashboard:
    def _dashboard(self, manager: IncidentManager, detector=None, clock=None) -> LiveDashboard:
        return LiveDashboard(manager, detector, clock=clock or ManualClock())

    def _metric(self, value: float, *, anomaly: bool = False, cloud: str = "aws") -> Metric:
        from src.core.enums import Cloud
        from src.core.stats import stats_from_values

        return Metric(
            name="cpu_utilization",
            value=value,
            unit="%",
            cloud=Cloud.parse(cloud),
            resource_id="i-1",
            service="EC2",
            region="us-east-1",
            window=stats_from_values([40.0, 42.0, 41.0, 39.0, 40.5]),
            is_anomaly=anomaly,
            confidence=0.8 if anomaly else 0.1,
        )

    def test_renders_before_any_data(self, manager: IncidentManager) -> None:
        text = _render(self._dashboard(manager))
        assert "AI INCIDENT RESPONSE" in text
        assert "awaiting telemetry" in text
        assert "no active incidents" in text
        assert "no lifecycle events yet" in text

    def test_update_metrics_aggregates_per_cloud(self, manager: IncidentManager) -> None:
        dashboard = self._dashboard(manager)
        dashboard.update_metrics([self._metric(41.0), self._metric(97.0, anomaly=True, cloud="gcp")])
        state = dashboard.state
        assert state["metrics_seen"] == 2
        assert state["anomalies_seen"] == 1
        assert state["clouds"]["aws"]["series"] == 1
        assert state["clouds"]["gcp"]["anomalies"] == 1
        assert state["ticks"] == 1

    def test_renders_metric_values_and_anomalies(self, manager: IncidentManager) -> None:
        dashboard = self._dashboard(manager)
        dashboard.update_metrics([self._metric(41.0), self._metric(97.0, anomaly=True)])
        text = _render(dashboard)
        assert "AWS" in text and "GCP" in text
        assert "ANOMALY" in text
        assert "97.0%" in text

    def test_subscribes_to_incident_events(self, manager: IncidentManager, incident) -> None:
        dashboard = self._dashboard(manager)
        manager.add(incident)
        manager.acknowledge(incident.id)
        manager.resolve(incident.id, "fixed")
        assert [item["kind"] for item in dashboard.feed] == ["created", "acknowledged", "resolved"]
        text = _render(dashboard)
        assert "RESOLVED" in text
        assert incident.id in text

    def test_active_incident_table_shows_severity(self, manager: IncidentManager, incident) -> None:
        dashboard = self._dashboard(manager)
        manager.add(incident)
        text = _render(dashboard)
        assert "INC-00001" in text
        assert "CRITICAL" in text

    def test_model_panel_with_detector(self, manager: IncidentManager, detector) -> None:
        dashboard = self._dashboard(manager, detector=detector)
        dashboard.update_metrics([self._metric(41.0)])
        text = _render(dashboard)
        assert "DETECTION MODEL" in text
        assert detector.version in text
        assert "drift" in text

    def test_model_panel_without_detector(self, manager: IncidentManager) -> None:
        text = _render(self._dashboard(manager))
        assert "not attached" in text

    def test_attach_notifiers_shows_routing_table(self, manager: IncidentManager) -> None:
        from src.notifications.notifier import ConsoleNotifier

        dashboard = self._dashboard(manager)
        dashboard.attach_notifiers([ConsoleNotifier()])
        assert "channels: console" in _render(dashboard)

    def test_resolved_incidents_leave_the_active_table(
        self,
        manager: IncidentManager,
        incident,
    ) -> None:
        dashboard = self._dashboard(manager)
        manager.add(incident)
        manager.resolve(incident.id)
        assert "no active incidents" in _render(dashboard)

    def test_severity_counts_summary(self, manager: IncidentManager, incident) -> None:
        manager.add(incident)
        manager.add(replace(incident, id="INC-00002", fingerprint="b", severity=Severity.LOW))
        text = _render(self._dashboard(manager))
        assert "CRIT" in text and "HIGH" in text and "MED" in text and "LOW" in text

    def test_state_is_json_friendly(self, manager: IncidentManager, incident) -> None:
        import json

        dashboard = self._dashboard(manager)
        manager.add(incident)
        dashboard.update_metrics([self._metric(50.0)])
        payload = json.loads(json.dumps(dashboard.state, default=str))
        assert payload["active"] == [incident.id]
        assert payload["status_counts"]["open"] == 1
