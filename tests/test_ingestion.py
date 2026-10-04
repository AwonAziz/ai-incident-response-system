"""Unified metric model, rolling history and the cloud collectors."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core.enums import Cloud
from src.ingestion import (
    AWSCollector,
    AzureCollector,
    GCPCollector,
    SeriesHistory,
    collectors_for_clouds,
)
from src.ingestion.base_collector import BaseCollector, SimulatedCollector
from src.ingestion.metric_schema import Metric, MetricSpec, ResourceSpec


class TestMetric:
    def test_normalises_cloud_and_timestamp(self, clock) -> None:
        metric = Metric(name="cpu_utilization", value="42.5", cloud="AZURE")
        assert metric.cloud is Cloud.AZURE
        assert isinstance(metric.value, float)
        assert metric.timestamp.tzinfo is not None
        assert metric.tags == {}

    def test_keys_distinguish_scope_from_series(self, warm_metric: Metric) -> None:
        assert warm_metric.series_key == "aws:i-0a1f9c4d2e7b8a31:cpu_utilization"
        assert warm_metric.scope_key == "aws:EC2:cpu_utilization"

    def test_z_score_requires_window(self, warm_metric: Metric, cold_metric: Metric) -> None:
        assert warm_metric.z_score > 5.0
        assert cold_metric.z_score == 0.0
        assert warm_metric.history_ready
        assert not cold_metric.history_ready

    def test_round_trip_serialisation(self, warm_metric: Metric) -> None:
        restored = Metric.from_dict(warm_metric.to_dict())
        assert restored.series_key == warm_metric.series_key
        assert restored.value == warm_metric.value
        assert restored.timestamp == warm_metric.timestamp
        assert restored.window is not None
        assert restored.window.count == warm_metric.window.count
        assert restored.cloud is warm_metric.cloud

    def test_with_detection_returns_a_copy(self, warm_metric: Metric) -> None:
        annotated = warm_metric.with_detection(is_anomaly=True, confidence=0.9, anomaly_score=1.5)
        assert annotated.is_anomaly
        assert annotated.confidence == 0.9
        assert not warm_metric.is_anomaly
        assert "anomaly" in annotated.describe()

    def test_describe_includes_window_context(self, warm_metric: Metric, cold_metric: Metric) -> None:
        assert "z=" in warm_metric.describe()
        assert "z=" not in cold_metric.describe()


class TestMetricSpecs:
    def test_midpoint_and_amplitude(self) -> None:
        spec = MetricSpec(
            name="m",
            label="M",
            unit="%",
            normal_low=10.0,
            normal_high=30.0,
            warn_high=80.0,
            critical_high=90.0,
        )
        assert spec.midpoint == 20.0
        assert spec.amplitude == 10.0

    def test_resource_spec_from_dict(self) -> None:
        spec = ResourceSpec.from_dict({"id": "x", "service": "EC2", "region": "eu-west-1", "tags": {"a": 1}})
        assert spec.tags == {"a": "1"}


class TestSeriesHistory:
    def test_bounded_window(self) -> None:
        history = SeriesHistory(window=3, min_samples=2)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for index in range(5):
            history.record("s", float(index), base + timedelta(seconds=index))
        assert len(history) == 1
        assert history.values("s") == (2.0, 3.0, 4.0)
        assert history.stats("s").span_seconds == 4.0

    def test_readiness_and_missing_series(self) -> None:
        history = SeriesHistory(window=10, min_samples=2)
        now = datetime.now(timezone.utc)
        assert history.stats("missing") is None
        assert history.values("missing") == ()
        assert not history.is_ready("missing")
        history.record("s", 1.0, now)
        assert not history.is_ready("s")
        history.record("s", 2.0, now)
        assert history.is_ready("s")
        assert "s" in history

    def test_shared_instance_is_reused_even_when_empty(self, history: SeriesHistory) -> None:
        collector = AWSCollector(history=history)
        assert collector.history is history
        assert len(history) == 0  # empty history is falsy; identity must still hold

    def test_clear(self) -> None:
        history = SeriesHistory()
        history.record("s", 1.0, datetime.now(timezone.utc))
        history.clear()
        assert len(history) == 0

    def test_rejects_zero_window(self) -> None:
        with pytest.raises(ValueError):
            SeriesHistory(window=0)


class TestCollectors:
    def test_one_collector_per_cloud(self) -> None:
        collectors = collectors_for_clouds(seed=1)
        assert [collector.cloud for collector in collectors] == [Cloud.AWS, Cloud.AZURE, Cloud.GCP]
        assert all(isinstance(collector, SimulatedCollector) for collector in collectors)
        assert all(isinstance(collector, BaseCollector) for collector in collectors)

    def test_collect_emits_every_resource_and_metric(self) -> None:
        collector = AWSCollector(seed=2)
        metrics = collector.collect()
        assert len(metrics) == len(collector.resources) * len(collector.specs)
        assert {metric.cloud for metric in metrics} == {Cloud.AWS}
        assert all(metric.window is None for metric in metrics)

    def test_collect_and_track_attaches_window(self, history: SeriesHistory) -> None:
        collector = GCPCollector(seed=3, history=history)
        metrics = collector.collect_and_track()
        assert all(metric.window is not None for metric in metrics)
        assert all(metric.window.count == 1 for metric in metrics)
        assert len(history) == len(metrics)
        assert collector.ticks == 1

    def test_window_grows_and_caps(self, history: SeriesHistory) -> None:
        collector = AzureCollector(seed=4, history=history, inject_anomaly=False)
        for _ in range(history.window + 5):
            collector.collect_and_track()
        counts = {metric.window.count for metric in collector.collect_and_track()}
        assert counts == {history.window}

    def test_deterministic_for_a_fixed_seed(self) -> None:
        first = [m.value for m in AWSCollector(seed=99).collect()]
        second = [m.value for m in AWSCollector(seed=99).collect()]
        assert first == second

    def test_different_seeds_diverge(self) -> None:
        first = [m.value for m in AWSCollector(seed=1).collect()]
        second = [m.value for m in AWSCollector(seed=2).collect()]
        assert first != second

    def test_values_stay_within_sane_bounds(self) -> None:
        collector = AWSCollector(seed=5)
        for _ in range(30):
            for metric in collector.collect():
                assert metric.value >= 0.0
                assert metric.value < 1000.0

    def test_anomaly_injection_raises_values_and_records_series(self) -> None:
        clean = AWSCollector(seed=6, inject_anomaly=False)
        stressed = AWSCollector(seed=6, inject_anomaly=True)
        clean_means: dict[str, list[float]] = {}
        stressed_means: dict[str, list[float]] = {}
        for _ in range(14):
            for metric in clean.collect():
                clean_means.setdefault(metric.name, []).append(metric.value)
            for metric in stressed.collect():
                stressed_means.setdefault(metric.name, []).append(metric.value)
        assert stressed.injected_series, "anomaly mode never injected anything"
        assert any(
            sum(stressed_means[name]) > sum(clean_means[name]) for name in clean_means
        ), "injected metrics were not higher than the baseline"

    def test_degradation_episodes_expire(self) -> None:
        collector = AWSCollector(seed=7, inject_anomaly=True)
        for _ in range(40):
            collector.collect()
        assert collector.degraded_resources() == ()

    def test_cloud_display_names(self) -> None:
        assert AWSCollector().display_name() == "AWS"
        assert AzureCollector().display_name() == "Azure"
        assert GCPCollector().display_name() == "GCP"

    def test_per_cloud_overrides_change_thresholds(self) -> None:
        aws = AWSCollector(seed=8).specs["error_rate"]
        gcp = GCPCollector(seed=8).specs["error_rate"]
        assert aws.critical_high < gcp.critical_high

    def test_collectors_for_clouds_subset(self) -> None:
        assert [c.cloud for c in collectors_for_clouds([Cloud.GCP])] == [Cloud.GCP]

    def test_repr_mentions_cloud(self) -> None:
        assert "aws" in repr(AWSCollector())
