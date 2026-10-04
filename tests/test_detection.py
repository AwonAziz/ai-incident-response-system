"""Feature engineering, Isolation Forest scoring, drift watch and rule engine."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.core.enums import Cloud, Severity
from src.detection import (
    FEATURE_NAMES,
    AnomalyDetector,
    DriftMonitor,
    FeatureEngineer,
    RuleEngine,
)
from src.detection.trainer import build_training_set, evaluate_on_labels, load_dataset, save_dataset
from src.ingestion.metric_schema import Metric


class TestFeatureEngineer:
    def test_cold_metric_has_neutral_statistics(self, cold_metric: Metric) -> None:
        vector = FeatureEngineer(min_samples=5).transform(cold_metric)
        assert vector.shape == (len(FEATURE_NAMES),)
        assert vector[FEATURE_NAMES.index("z_score")] == 0.0
        assert vector[FEATURE_NAMES.index("pct_change")] == 0.0
        assert vector[FEATURE_NAMES.index("value_over_mean")] == 1.0
        assert vector[FEATURE_NAMES.index("history_coverage")] == 0.0

    def test_warm_metric_exposes_statistics(self, warm_metric: Metric) -> None:
        vector = FeatureEngineer(min_samples=5).transform(warm_metric)
        expected = warm_metric.window.z_score(warm_metric.value)
        assert vector[FEATURE_NAMES.index("z_score")] == pytest.approx(min(expected, 12.0))
        assert vector[FEATURE_NAMES.index("coefficient_of_variation")] > 0
        assert vector[FEATURE_NAMES.index("history_coverage")] == 1.0

    def test_z_score_is_not_clipped_for_moderate_deviations(self, warm_metric: Metric) -> None:
        mild = replace(warm_metric, value=warm_metric.window.mean + warm_metric.window.std)
        vector = FeatureEngineer(min_samples=5).transform(mild)
        assert vector[FEATURE_NAMES.index("z_score")] == pytest.approx(1.0, rel=1e-6)

    def test_features_are_dimensionless(self, warm_metric: Metric) -> None:
        """Two metrics with wildly different units must yield comparable features."""
        engineer = FeatureEngineer.from_profiles(
            {"metrics": {"cpu_utilization": {"normal": [10, 80]}, "latency_ms": {"normal": [5, 200]}}},
            min_samples=5,
        )
        scaled = replace(warm_metric, name="latency_ms", value=warm_metric.value * 1000.0)
        assert engineer.transform(warm_metric).shape == engineer.transform(scaled).shape
        # band position, not raw value, carries the "how far out of range" signal
        assert engineer.transform(scaled)[FEATURE_NAMES.index("band_position")] > 3.0

    def test_band_position_uses_profiles(self, warm_metric: Metric) -> None:
        engineer = FeatureEngineer.from_profiles({"metrics": {"cpu_utilization": {"normal": [12, 78]}}})
        vector = engineer.transform(warm_metric)
        assert vector[FEATURE_NAMES.index("band_position")] == pytest.approx((97.5 - 12) / 66)
        assert engineer.bands == {"cpu_utilization": (12.0, 78.0)}

    def test_band_position_without_a_profile(self, warm_metric: Metric) -> None:
        assert FeatureEngineer().transform(warm_metric)[FEATURE_NAMES.index("band_position")] == 0.0

    def test_degenerate_band_is_guarded(self, warm_metric: Metric) -> None:
        engineer = FeatureEngineer(bands={"cpu_utilization": (50.0, 50.0)})
        assert engineer.transform(warm_metric)[FEATURE_NAMES.index("band_position")] == 0.0

    def test_coverage_is_proportional(self, clock) -> None:
        from src.core.stats import stats_from_values

        metric = Metric(
            name="cpu_utilization",
            value=50.0,
            timestamp=clock.now(),
            window=stats_from_values([40.0, 45.0]),
        )
        assert FeatureEngineer(min_samples=5).transform(metric)[-1] == pytest.approx(0.4)

    def test_batch_matrix_shape(self, training_data) -> None:
        samples, _ = training_data
        matrix = FeatureEngineer().transform_batch(samples[:40])
        assert matrix.shape == (40, len(FEATURE_NAMES))
        assert np.isfinite(matrix).all()

    def test_empty_batch(self) -> None:
        assert FeatureEngineer().transform_batch([]).shape == (0, len(FEATURE_NAMES))

    def test_outliers_are_clipped(self, clock) -> None:
        from src.core.stats import stats_from_values

        metric = Metric(
            name="latency_ms",
            value=1e9,
            timestamp=clock.now(),
            window=stats_from_values([10.0, 12.0, 11.0, 10.5, 11.5]),
        )
        vector = FeatureEngineer().transform(metric)
        assert abs(vector[FEATURE_NAMES.index("z_score")]) <= 12.0
        assert abs(vector[FEATURE_NAMES.index("pct_change")]) <= 500.0
        assert abs(vector[FEATURE_NAMES.index("value_over_mean")]) <= 10.0

    def test_feature_report(self, training_data) -> None:
        samples, _ = training_data
        engineer = FeatureEngineer()
        report = engineer.feature_report(engineer.transform_batch(samples[-30:]))
        assert set(report) == set(FEATURE_NAMES)
        assert report["z_score"]["max"] > report["z_score"]["min"]


class TestDriftMonitor:
    def test_no_drift_when_distribution_is_stable(self) -> None:
        monitor = DriftMonitor(threshold=0.35, min_batches=3)
        baseline = np.random.default_rng(0).normal(0.0, 1.0, size=(500, 4))
        monitor.fit_baseline(baseline)
        for _ in range(5):
            monitor.observe(np.random.default_rng(1).normal(0.0, 1.0, size=(20, 4)))
        assert monitor.drift_score < monitor.threshold
        assert not monitor.drifted

    def test_drift_when_distribution_shifts(self) -> None:
        monitor = DriftMonitor(threshold=0.35, min_batches=3)
        rng = np.random.default_rng(7)
        monitor.fit_baseline(rng.normal(0.0, 1.0, size=(500, 4)))
        for _ in range(5):
            monitor.observe(rng.normal(4.0, 1.0, size=(20, 4)))
        assert monitor.drift_score > monitor.threshold
        assert monitor.drifted

    def test_waits_for_minimum_batches(self) -> None:
        monitor = DriftMonitor(threshold=0.1, min_batches=10)
        monitor.fit_baseline(np.zeros((100, 3)))
        monitor.observe(np.ones((10, 3)))
        assert monitor.drift_score == 0.0
        assert not monitor.drifted

    def test_ignores_empty_and_uncalibrated_input(self) -> None:
        monitor = DriftMonitor()
        assert monitor.observe(np.zeros((0, 3))) == 0.0
        assert not monitor.is_calibrated
        monitor.fit_baseline(np.zeros((0, 3)))
        assert not monitor.is_calibrated

    def test_reset(self) -> None:
        monitor = DriftMonitor()
        monitor.fit_baseline(np.zeros((50, 2)))
        monitor.observe(np.zeros((5, 2)))
        monitor.reset()
        assert monitor.batches_seen == 0


class TestAnomalyDetectorTraining:
    def test_requires_enough_samples(self, warm_metric: Metric) -> None:
        detector = AnomalyDetector(n_estimators=10)
        with pytest.raises(ValueError, match="at least 50 samples"):
            detector.train([warm_metric] * 10)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"contamination": 0.0}, "contamination"),
            ({"contamination": 0.9}, "contamination"),
            ({"n_estimators": 2}, "n_estimators"),
        ],
    )
    def test_rejects_invalid_hyperparameters(self, kwargs: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            AnomalyDetector(**kwargs)

    def test_training_reports_summary(self, training_data) -> None:
        samples, _ = training_data
        detector = AnomalyDetector(n_estimators=20)
        summary = detector.train(samples)
        assert summary["training_samples"] == len(samples)
        assert summary["features"] == list(FEATURE_NAMES)
        assert summary["metrics"]
        assert detector.is_fitted
        assert detector.stats["training_samples"] == len(samples)
        assert detector.feature_summary["z_score"]["max"] > 0

    def test_rejects_wrong_matrix_width(self) -> None:
        with pytest.raises(ValueError, match="feature matrix"):
            AnomalyDetector().train_from_features(np.zeros((100, 3)))

    def test_rejects_non_metric_samples(self) -> None:
        with pytest.raises(TypeError):
            AnomalyDetector().train(["not-a-metric"])

    def test_unfitted_scoring_is_refused(self, warm_metric: Metric) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            AnomalyDetector().score_batch([warm_metric])


class TestAnomalyDetectorScoring:
    def test_scoring_annotates_metrics(self, detector: AnomalyDetector, training_data) -> None:
        samples, _ = training_data
        batch = samples[-15:]
        scored = detector.score_batch(batch)
        assert len(scored) == len(batch)
        for metric in scored:
            assert metric.anomaly_score is not None
            assert 0.0 <= metric.confidence <= 1.0
            assert metric.model_version == detector.version
            assert metric.is_anomaly is False or metric.confidence > 0.0

    def test_cold_metrics_are_never_flagged(self, detector: AnomalyDetector, cold_metric: Metric) -> None:
        cold = cold_metric.with_detection(is_anomaly=True, confidence=1.0)
        scored = detector.score_batch([cold])[0]
        assert scored.is_anomaly is False
        assert scored.confidence < 0.4  # damped by zero history coverage

    def test_injected_anomalies_score_materially_higher_than_clean_traffic(
        self,
        detector: AnomalyDetector,
    ) -> None:
        """The guarantee that matters: degraded samples stand out from the baseline.

        A single synthetic point's verdict depends on where it falls relative to
        the forest, so the property is asserted at the batch level instead.
        """
        from src.ingestion import SeriesHistory, collectors_for_clouds

        history = SeriesHistory(window=30, min_samples=5)
        clean = collectors_for_clouds(seed=4242, history=history, inject_anomaly=False)
        stressed = collectors_for_clouds(seed=4242, history=history, inject_anomaly=True)

        healthy: list[Metric] = []
        degraded: list[Metric] = []
        for _ in range(25):
            for collector in clean:
                healthy.extend(collector.collect_and_track())
            for collector in stressed:
                produced = collector.collect_and_track()
                if collector.degraded_resources():
                    degraded.extend(produced)

        assert degraded, "the simulator produced no degraded samples"
        healthy_scored = detector.score_batch(healthy)
        degraded_scored = detector.score_batch(degraded)

        healthy_confidence = sum(metric.confidence for metric in healthy_scored) / len(healthy_scored)
        degraded_confidence = sum(metric.confidence for metric in degraded_scored) / len(degraded_scored)
        assert degraded_confidence > healthy_confidence

        healthy_rate = sum(1 for metric in healthy_scored if metric.is_anomaly) / len(healthy_scored)
        degraded_rate = sum(1 for metric in degraded_scored if metric.is_anomaly) / len(degraded_scored)
        assert degraded_rate > max(5 * healthy_rate, 0.1)

    def test_confidence_is_monotonic_in_anomalousness(self, detector: AnomalyDetector) -> None:
        anchors = detector.stats["confidence_anchors"]
        assert anchors[0]["decision"] >= anchors[-1]["decision"]
        assert anchors[0]["confidence"] < anchors[-1]["confidence"]
        assert detector.confidence_for_decision(anchors[0]["decision"]) == pytest.approx(0.15, abs=1e-4)
        assert detector.confidence_for_decision(anchors[-1]["decision"] - 1.0) == pytest.approx(0.95)

    def test_empty_batch(self, detector: AnomalyDetector) -> None:
        assert detector.score_batch([]) == []

    def test_running_stats_accumulate(self, fresh_detector: AnomalyDetector, training_data) -> None:
        samples, _ = training_data
        fresh_detector.score_batch(samples[-30:])
        stats = fresh_detector.stats
        assert stats["metrics_scored"] == 30
        assert stats["batches_scored"] == 1
        assert 0.0 <= stats["anomaly_rate"] <= 1.0

    def test_healthy_sample_scores_low_confidence(self, detector: AnomalyDetector, warm_metric: Metric) -> None:
        healthy = replace(warm_metric, value=warm_metric.window.mean)
        scored = detector.score_batch([healthy])[0]
        assert not scored.is_anomaly
        assert scored.confidence < 0.4

    def test_outcomes_expose_full_detail(self, detector: AnomalyDetector, warm_metric: Metric) -> None:
        outcomes = detector.score_with_outcomes([warm_metric])
        payload = outcomes[0].to_dict()
        assert payload["series"] == warm_metric.series_key
        assert payload["is_anomaly"] is outcomes[0].is_anomaly
        assert 0.0 <= payload["confidence"] <= 1.0

    def test_evaluate_reports_classification_metrics(self, detector: AnomalyDetector) -> None:
        assert detector.evaluate([1, 1, 0, 0], [1, 0, 1, 0]) == {
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "support": 4,
        }
        assert detector.evaluate([0, 0], [0, 0])["precision"] == 0.0

    def test_reset_runtime_stats(self, detector: AnomalyDetector, training_data) -> None:
        samples, _ = training_data
        detector.score_batch(samples[-5:])
        detector.reset_runtime_stats()
        assert detector.stats["metrics_scored"] == 0


class TestPersistence:
    def test_save_load_round_trip(self, detector: AnomalyDetector, tmp_path: Path, warm_metric: Metric) -> None:
        path = tmp_path / "nested" / "model.pkl"
        detector.save(path)
        assert path.is_file()

        restored = AnomalyDetector()
        restored.load(path)
        assert restored.is_fitted
        assert restored.version == detector.version
        assert restored.stats["training_samples"] == detector.stats["training_samples"]

        original = detector.score_batch([warm_metric])[0]
        reloaded = restored.score_batch([warm_metric])[0]
        assert original.is_anomaly == reloaded.is_anomaly
        assert original.anomaly_score == pytest.approx(reloaded.anomaly_score)
        assert original.confidence == pytest.approx(reloaded.confidence)

    def test_loading_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            AnomalyDetector().load(tmp_path / "absent.pkl")

    def test_saving_unfitted_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            AnomalyDetector().save(tmp_path / "m.pkl")

    def test_future_format_is_rejected(self, tmp_path: Path, detector: AnomalyDetector) -> None:
        import joblib

        path = detector.save(tmp_path / "m.pkl")
        bundle = joblib.load(path)
        bundle["format_version"] = 999
        joblib.dump(bundle, path)
        with pytest.raises(ValueError, match="newer than supported"):
            AnomalyDetector().load(path)


class TestTrainerHelpers:
    def test_build_training_set_shares_history(self) -> None:
        samples, history = build_training_set(rounds=2, seed=5)
        assert len(history) == 55  # 11 resources x 5 metrics
        assert all(metric.window is not None for metric in samples)
        assert len(samples) == 2 * 55

    def test_evaluate_on_labels(self, detector: AnomalyDetector) -> None:
        samples, _ = build_training_set(rounds=2, seed=6)
        labels = [0] * len(samples)
        metrics = evaluate_on_labels(detector, samples, labels)
        assert metrics["support"] == len(samples)
        assert metrics["actual_positive"] == 0
        assert "mean_decision" in metrics

    def test_evaluate_rejects_length_mismatch(self, detector: AnomalyDetector) -> None:
        samples, _ = build_training_set(rounds=1, seed=6)
        with pytest.raises(ValueError, match="same length"):
            evaluate_on_labels(detector, samples, [])

    def test_dataset_round_trip(self, tmp_path: Path) -> None:
        samples, _ = build_training_set(rounds=1, seed=8)
        path = save_dataset(samples[:5], tmp_path / "dataset.json")
        restored = load_dataset(path)
        assert len(restored) == 5
        assert restored[0].series_key == samples[0].series_key

    def test_load_bare_list(self, tmp_path: Path) -> None:
        samples, _ = build_training_set(rounds=1, seed=8)
        path = save_dataset(samples[:2], tmp_path / "d.json")
        rows = json.loads(path.read_text(encoding="utf-8"))["metrics"]
        bare = tmp_path / "list.json"
        bare.write_text(json.dumps(rows), encoding="utf-8")
        assert len(load_dataset(bare)) == 2

    def test_load_missing_dataset(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nope.json")


class TestRuleEngine:
    @pytest.fixture
    def engine(self, profiles: dict) -> RuleEngine:
        return RuleEngine(profiles)

    def _metric(self, **overrides) -> Metric:
        base = {
            "name": "cpu_utilization",
            "value": 50.0,
            "unit": "%",
            "cloud": Cloud.AWS,
            "resource_id": "i-0a1f9c4d2e7b8a31",
            "service": "EC2",
            "region": "us-east-1",
        }
        base.update(overrides)
        return Metric(**base)

    def test_healthy_value_has_no_violations(self, engine: RuleEngine) -> None:
        assert engine.evaluate(self._metric(value=40.0)) == []

    def test_critical_threshold(self, engine: RuleEngine) -> None:
        violations = engine.evaluate(self._metric(value=97.0))
        critical = [v for v in violations if v.rule_id == "cpu_utilization.critical_high"]
        assert critical and critical[0].severity is Severity.CRITICAL
        assert critical[0].exceeded_by > 0

    def test_warning_threshold(self, engine: RuleEngine) -> None:
        violations = engine.evaluate(self._metric(value=80.0))
        assert any(v.rule_id == "cpu_utilization.warn_high" for v in violations)
        assert max(violations, key=lambda v: int(v.severity)).severity in (Severity.HIGH, Severity.MEDIUM)

    def test_band_rule_fires_outside_normal(self, engine: RuleEngine) -> None:
        violation = next(
            v for v in engine.evaluate(self._metric(value=5.0)) if v.rule_id.endswith("outside_normal_band")
        )
        assert violation.severity is Severity.MEDIUM

    def test_violations_are_sorted_worst_first(self, engine: RuleEngine) -> None:
        severities = [int(v.severity) for v in engine.evaluate(self._metric(value=99.0))]
        assert severities == sorted(severities, reverse=True)

    def test_zscore_rule_requires_a_warm_window(self, engine: RuleEngine) -> None:
        cold = self._metric(value=99.0)
        assert not any(v.rule_id.endswith("zscore_spike") for v in engine.evaluate(cold))

        from src.core.stats import stats_from_values

        warm = self._metric(value=99.0, window=stats_from_values([40.0, 41.0, 39.0, 40.0, 42.0]))
        assert any(v.rule_id.endswith("zscore_spike") for v in engine.evaluate(warm))

    def test_per_cloud_override_applies(self, engine: RuleEngine) -> None:
        aws = engine.evaluate(self._metric(name="error_rate", value=3.5, cloud=Cloud.AWS))
        gcp = engine.evaluate(self._metric(name="error_rate", value=3.5, cloud=Cloud.GCP))
        assert any(v.rule_id.endswith("critical_high") for v in aws)
        assert not any(v.rule_id.endswith("critical_high") for v in gcp)

    def test_unknown_metric_is_ignored(self, engine: RuleEngine) -> None:
        assert engine.evaluate(self._metric(name="not_a_metric", value=1e9)) == []

    def test_evaluate_batch_groups_by_series(self, engine: RuleEngine) -> None:
        batch = [self._metric(value=40.0), self._metric(value=99.0, name="memory_utilization")]
        grouped = engine.evaluate_batch(batch)
        assert len(grouped) == 1
        assert next(iter(grouped)).endswith("memory_utilization")

    def test_rules_for_and_known_metrics(self, engine: RuleEngine) -> None:
        rules = engine.rules_for("aws", "cpu_utilization")
        assert {rule["id"] for rule in rules} >= {
            "cpu_utilization.warn_high",
            "cpu_utilization.critical_high",
            "cpu_utilization.outside_normal_band",
            "cpu_utilization.zscore_spike",
        }
        assert "error_rate" in engine.known_metrics

    def test_violation_serialisation(self, engine: RuleEngine) -> None:
        payload = engine.evaluate(self._metric(value=99.0))[0].to_dict()
        assert payload["severity"] == "CRITICAL"
        assert payload["cloud"] == "aws"
        assert "exceeded_by_percent" in payload

    def test_spike_multiplier_tightens_the_z_rule(self, profiles: dict) -> None:
        from src.core.stats import stats_from_values

        strict = RuleEngine(profiles, spike_multiplier=0.5)
        metric = self._metric(value=82.0, window=stats_from_values([40.0, 41.0, 39.0, 40.0, 42.0]))
        assert any(v.rule_id.endswith("zscore_spike") for v in strict.evaluate(metric))

    def test_empty_profiles_yield_no_rules(self) -> None:
        assert RuleEngine({"clouds": {}}).known_metrics == ()
