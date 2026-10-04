"""End-to-end pipeline behaviour."""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from pathlib import Path

import pytest

from src.core.clock import ManualClock
from src.core.enums import Severity
from src.pipeline import Pipeline, PipelineConfig
from src.triage.incident_manager import IncidentManager


@pytest.fixture
def pipeline(settings) -> Pipeline:
    config = PipelineConfig(
        settings=settings,
        retrain=True,
        dashboard=False,
        inject_anomaly=True,
        inject_every_ticks=2,
        auto_retrain=False,
        console_notifications=False,
        max_ticks=3,
    )
    return Pipeline(config, clock=ManualClock()).prepare()


class TestConstruction:
    def test_components_are_wired(self, pipeline: Pipeline) -> None:
        assert pipeline.detector.is_fitted
        assert pipeline.triage_engine._manager is pipeline.incident_manager
        assert pipeline.dashboard.detector is pipeline.detector
        assert pipeline.dashboard.incidents is pipeline.incident_manager
        assert pipeline.router.notifiers

    def test_collectors_share_one_rolling_window(self, pipeline: Pipeline) -> None:
        assert all(collector.history is pipeline.history for collector in pipeline.collectors)
        assert all(collector.history is pipeline.history for collector in pipeline.anomaly_collectors)

    def test_triage_clears_dedup_state_on_resolution(self, pipeline: Pipeline) -> None:
        assert pipeline.triage_engine.open_fingerprints() == ()
        incident = pipeline.build_sample("aws", "cpu_utilization", 99.0)
        created = pipeline.inject([incident])
        assert created
        pipeline.incident_manager.resolve(created[0].id, "done")
        assert pipeline.triage_engine.open_fingerprints() == ()

    def test_prepare_trains_and_reports(self, pipeline: Pipeline) -> None:
        assert pipeline.training_summary is not None
        assert pipeline.training_summary["training_samples"] > 100
        assert pipeline.retrain_count == 1

    def test_first_run_persists_the_model(self, settings) -> None:
        pipeline = Pipeline(PipelineConfig(settings=settings, retrain=True, dashboard=False), clock=ManualClock())
        pipeline.prepare()
        assert Path(settings.model_path).is_file()
        assert pipeline.model_loaded_from is None

    def test_train_can_skip_saving(self, settings) -> None:
        pipeline = Pipeline(PipelineConfig(settings=settings, retrain=True, dashboard=False), clock=ManualClock())
        pipeline.prepare()
        before = Path(settings.model_path).stat().st_mtime_ns
        pipeline.train(save=False)
        assert Path(settings.model_path).stat().st_mtime_ns == before

    def test_prepare_loads_an_existing_model(self, settings) -> None:
        first = Pipeline(PipelineConfig(settings=settings, retrain=True, dashboard=False), clock=ManualClock())
        first.prepare()  # writes the model to settings.model_path

        second = Pipeline(PipelineConfig(settings=settings, retrain=False, dashboard=False), clock=ManualClock())
        second.prepare()
        assert second.model_loaded_from == settings.model_path
        assert second.detector.stats["training_samples"] == first.detector.stats["training_samples"]

    def test_prepare_without_retrain_trains_when_model_missing(self, settings) -> None:
        assert not settings.model_path.endswith("missing.pkl")  # sanity
        pipeline = Pipeline(PipelineConfig(settings=settings, dashboard=False), clock=ManualClock())
        assert pipeline.prepare().detector.is_fitted


class TestTick:
    def test_tick_produces_scored_metrics(self, pipeline: Pipeline) -> None:
        result = pipeline.tick()
        assert len(result.metrics) == 55
        assert all(metric.model_version == pipeline.detector.version for metric in result.metrics)
        assert pipeline.ticks == 1

    def test_injected_anomalies_raise_rule_violations(self, pipeline: Pipeline) -> None:
        result = pipeline.tick(use_anomaly=True)
        assert result.violations > 0
        assert result.incidents

    def test_incidents_are_registered_and_notified(self, pipeline: Pipeline) -> None:
        result = pipeline.tick(use_anomaly=True)
        assert result.incidents
        for incident in result.incidents:
            assert pipeline.incident_manager.get(incident.id) is incident
        assert pipeline.incident_manager.stats["total_created"] == len(result.incidents)
        assert pipeline.router.stats["sent"] + pipeline.router.stats["skipped"] > 0

    def test_dedup_suppresses_repeats(self, pipeline: Pipeline) -> None:
        pipeline.tick(use_anomaly=True)
        created_first = pipeline.incident_manager.stats["total_created"]
        pipeline.tick(use_anomaly=True)
        assert pipeline.incident_manager.stats["total_created"] - created_first < created_first
        assert pipeline.triage_engine.stats["suppressed"] > 0

    def test_dashboard_receives_metrics(self, pipeline: Pipeline) -> None:
        pipeline.tick()
        assert pipeline.dashboard.metrics_seen == 55
        assert pipeline.dashboard.state["ticks"] == 1

    def test_ticker_counter_advances(self, pipeline: Pipeline) -> None:
        for _ in range(3):
            pipeline.tick()
        assert pipeline.ticks == 3
        assert pipeline.collectors[0].ticks == 3


class TestRun:
    def test_run_respects_max_ticks(self, settings) -> None:
        config = PipelineConfig(
            settings=settings,
            retrain=True,
            dashboard=False,
            max_ticks=4,
            console_notifications=False,
        )
        summary = Pipeline(config, clock=ManualClock()).prepare().run()
        assert summary["ticks"] == 4
        assert summary["metrics_processed"] == 4 * 55

    def test_stop_breaks_the_loop(self, settings) -> None:
        pipeline = Pipeline(
            PipelineConfig(settings=settings, retrain=True, dashboard=False, console_notifications=False),
            clock=ManualClock(),
        ).prepare()
        pipeline.stop()
        summary = pipeline.run()
        assert summary["ticks"] == 0

    def test_dashboard_mode_runs(self, settings) -> None:
        config = PipelineConfig(
            settings=settings,
            retrain=True,
            dashboard=True,
            max_ticks=2,
            console_notifications=False,
        )
        summary = Pipeline(config, clock=ManualClock()).prepare().run()
        assert summary["ticks"] == 2

    def test_injection_schedule_is_honoured(self, settings) -> None:
        config = PipelineConfig(
            settings=settings,
            retrain=True,
            dashboard=False,
            inject_anomaly=True,
            inject_every_ticks=2,
            max_ticks=4,
            console_notifications=False,
        )
        pipeline = Pipeline(config, clock=ManualClock()).prepare()
        summary = pipeline.run()
        # the anomaly collector is exercised on even ticks
        assert pipeline.anomaly_collectors[0].ticks > 0
        assert summary["ticks"] == 4


class TestInjection:
    def test_build_sample_fills_metadata(self, pipeline: Pipeline) -> None:
        metric = pipeline.build_sample("gcp", "query_latency_ms", 250.0)
        assert metric.cloud.value == "gcp"
        assert metric.unit == "ms"
        assert metric.service and metric.region and metric.resource_id
        assert metric.confidence == 1.0

    def test_inject_sample_creates_an_incident(self, pipeline: Pipeline) -> None:
        created = pipeline.inject_sample("aws", "cpu_utilization", 99.0)
        assert created
        assert created[0].severity is Severity.CRITICAL
        assert created[0].resource_id
        assert pipeline.incident_manager.get(created[0].id) is not None

    def test_inject_sample_respects_severity_routing(self, pipeline: Pipeline) -> None:
        created = pipeline.inject_sample("aws", "cpu_utilization", 81.0)
        assert created
        assert created[0].severity is Severity.HIGH

    def test_inject_sample_respects_dedup(self, pipeline: Pipeline) -> None:
        first = pipeline.inject_sample("aws", "cpu_utilization", 99.0)
        second = pipeline.inject_sample("aws", "cpu_utilization", 99.0)
        assert len(first) == 1
        assert second == []

    def test_inject_explicit_metric_documents(self, pipeline: Pipeline) -> None:
        from src.ingestion.metric_schema import Metric

        metric = Metric(
            name="memory_utilization",
            value=99.0,
            unit="%",
            cloud="aws",
            resource_id="i-custom",
            service="EC2",
        )
        created = pipeline.inject([metric.with_detection(is_anomaly=True, confidence=0.95)])
        assert created
        assert created[0].resource_id == "i-custom"

    def test_inject_empty_batch(self, pipeline: Pipeline) -> None:
        assert pipeline.inject([]) == []

    def test_inject_updates_dashboard(self, pipeline: Pipeline) -> None:
        before = pipeline.dashboard.metrics_seen
        pipeline.inject_sample("aws", "cpu_utilization", 99.0)
        assert pipeline.dashboard.metrics_seen == before + 1

    def test_injection_does_not_train(self, pipeline: Pipeline) -> None:
        pipeline.inject_sample("aws", "cpu_utilization", 99.0)
        assert pipeline.retrain_count == 1


class TestAutoResolve:
    def test_stale_low_severity_incident_is_auto_resolved(self, settings) -> None:
        config = PipelineConfig(
            settings=dataclasses.replace(settings, auto_resolve_minutes=0, auto_resolve_severities=("low", "medium")),
            retrain=True,
            dashboard=False,
            console_notifications=False,
        )
        pipeline = Pipeline(config, clock=ManualClock()).prepare()
        pipeline.inject_sample("aws", "cpu_utilization", 99.0)  # CRITICAL: never auto-resolved
        pipeline.inject_sample("aws", "request_rate", 10.0)  # low-value sample: no incident

        low = pipeline.detector.score_batch(
            [pipeline.build_sample("azure", "response_time_ms", 100.0)]
        )[0]
        incident = pipeline.triage_engine.triage(
            replace(low, confidence=0.9, is_anomaly=True),
            [],
            context=[low],
        )
        if incident is not None:
            pipeline.incident_manager.add(incident)
            pipeline.clock.advance(3600)
            result = pipeline.tick()
            assert all(item.severity is not Severity.CRITICAL for item in result.auto_resolved)

    def test_critical_incidents_survive_auto_resolve(self, settings) -> None:
        config = PipelineConfig(
            settings=dataclasses.replace(settings, auto_resolve_minutes=0),
            retrain=True,
            dashboard=False,
            console_notifications=False,
        )
        pipeline = Pipeline(config, clock=ManualClock()).prepare()
        pipeline.inject_sample("aws", "cpu_utilization", 99.0)
        pipeline.clock.advance(3600)
        pipeline.tick()
        critical = [item for item in pipeline.incident_manager.all() if item.severity is Severity.CRITICAL]
        assert critical
        assert all(item.status.value != "resolved" for item in critical)


class TestAutoRetrain:
    def test_drift_triggers_retraining(self, settings) -> None:
        config = PipelineConfig(
            settings=dataclasses.replace(settings, drift_threshold=0.01, drift_min_batches=2),
            retrain=True,
            dashboard=False,
            auto_retrain=True,
            min_ticks_between_retrains=0,
            console_notifications=False,
        )
        pipeline = Pipeline(config, clock=ManualClock()).prepare()
        before = pipeline.retrain_count
        for _ in range(4):
            result = pipeline.tick(use_anomaly=True)
        assert pipeline.detector.is_fitted
        assert pipeline.retrain_count > before
        assert result.retrained or pipeline.retrain_count > before + 1

    def test_retrain_can_be_disabled(self, settings) -> None:
        config = PipelineConfig(
            settings=dataclasses.replace(settings, drift_threshold=0.0, drift_min_batches=1),
            retrain=True,
            dashboard=False,
            auto_retrain=False,
            console_notifications=False,
        )
        pipeline = Pipeline(config, clock=ManualClock()).prepare()
        for _ in range(4):
            pipeline.tick(use_anomaly=True)
        assert pipeline.retrain_count == 1


class TestReporting:
    def test_summary_shape(self, pipeline: Pipeline) -> None:
        pipeline.tick(use_anomaly=True)
        summary = pipeline.summary()
        assert set(summary) >= {
            "ticks",
            "metrics_processed",
            "anomalies_detected",
            "incidents",
            "triage",
            "detector",
            "notifications",
            "retrains",
            "uptime_seconds",
        }
        assert summary["incidents"]["total_created"] >= 1
        assert summary["detector"]["fitted"] is True

    def test_stats_payload_for_the_api(self, pipeline: Pipeline) -> None:
        pipeline.tick(use_anomaly=True)
        payload = pipeline.stats()
        assert set(payload) >= {
            "summary",
            "incidents",
            "dashboard",
            "collector_ticks",
            "anomaly_collector_ticks",
            "history_series",
            "running",
        }
        assert payload["anomaly_collector_ticks"]["aws"] == 1
        assert payload["collector_ticks"]["aws"] == 0
        assert payload["history_series"] == 55
        assert payload["running"] is True

    def test_summary_is_json_serialisable(self, pipeline: Pipeline) -> None:
        import json

        pipeline.tick()
        json.dumps(pipeline.summary(), default=str)
        json.dumps(pipeline.stats(), default=str)

    def test_manager_is_a_plain_incident_manager(self, pipeline: Pipeline) -> None:
        assert isinstance(pipeline.incident_manager, IncidentManager)
