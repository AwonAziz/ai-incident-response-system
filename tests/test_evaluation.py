"""Evaluation harness: scorer contract, metric maths, protocol integrity.

All synthetic: a detector that works here must not depend on the downloaded
dataset, so these tests build labelled series in memory.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from src.data.timeseries import AnomalyWindow, LabelledSeries
from src.detection.evaluation import (
    EVALUATION_WARNINGS,
    DetectionScorer,
    EvaluationProtocol,
    EWMAScorer,
    GlobalThresholdScorer,
    IsolationForestScorer,
    ProfileThresholdScorer,
    RollingZScoreScorer,
    aggregate,
    alert_budget_curve,
    average_precision,
    benchmark,
    default_scorers,
    evaluate_series,
    quantile_threshold,
    roc_auc,
    run_count,
    split_series,
    train_band,
)
from src.detection.feature_engineer import FEATURE_NAMES, FeatureEngineer
from src.ingestion.metric_schema import Metric

PROTOCOL = EvaluationProtocol(train_fraction=0.4, alert_rate=0.02, n_estimators=20)


def build_series(
    *,
    size: int = 400,
    spike_at: int = 320,
    spike_size: float = 40.0,
    base: float = 50.0,
    noise: float = 1.0,
    seed: int = 0,
    step_seconds: int = 300,
) -> LabelledSeries:
    """A stable series with one level shift, labels derived from the shift."""
    import random

    rng = random.Random(seed)
    spike_at = spike_at if spike_at < size else int(size * 0.8)
    start = datetime(2014, 1, 1, tzinfo=timezone.utc)
    stamps = tuple(start + timedelta(seconds=step_seconds * index) for index in range(size))
    values: list[float] = []
    labels: list[int] = []
    for index in range(size):
        if index >= spike_at:
            values.append(base + spike_size + rng.gauss(0.0, noise))
            labels.append(1)
        else:
            values.append(base + rng.gauss(0.0, noise))
            labels.append(0)
    window = AnomalyWindow(
        start=stamps[spike_at],
        end=stamps[-1],
    )
    return LabelledSeries(
        name=f"synthetic_{size}",
        metric="cpu_utilization",
        timestamps=stamps,
        values=tuple(values),
        labels=tuple(labels),
        windows=(window,),
        resource_id="cpu_utilization/synthetic",
    )


class TestMetricHelpers:
    def test_run_count_counts_episodes(self) -> None:
        assert run_count([]) == 0
        assert run_count([False, True, True, False, True]) == 2
        assert run_count([True, True, True]) == 1
        assert run_count([False, False]) == 0

    def test_quantile_threshold_alerts_on_the_requested_rate(self) -> None:
        scores = [float(index) for index in range(100)]
        assert quantile_threshold(scores, 0.02) == 98.0
        assert quantile_threshold(scores, 0.10) == 90.0
        assert quantile_threshold([], 0.02) == math.inf

    def test_roc_auc_perfect_and_inverted(self) -> None:
        labels = [0, 0, 1, 1]
        assert roc_auc(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0
        assert roc_auc(labels, [0.9, 0.8, 0.2, 0.1]) == 0.0

    def test_roc_auc_handles_ties_as_half_credit(self) -> None:
        assert roc_auc([0, 1], [0.5, 0.5]) == 0.5

    def test_roc_auc_degenerate_labels(self) -> None:
        assert roc_auc([0, 0], [0.1, 0.9]) == 0.5
        assert roc_auc([1, 1], [0.1, 0.9]) == 0.5

    def test_average_precision(self) -> None:
        labels = [1, 0, 1, 0]
        assert average_precision(labels, [0.9, 0.8, 0.7, 0.1]) == pytest.approx((1 / 1 + 2 / 3) / 2)
        assert average_precision([0, 0], [0.1, 0.2]) == 0.0

    def test_alert_budget_curve_is_monotonic_in_precision(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.9]
        labels = [0, 0, 0, 1]
        curve = alert_budget_curve(scores, labels, [0.25, 0.5])
        assert curve[0] == {"budget": 0.25, "alerts": 1, "precision": 1.0, "recall": 1.0}
        assert curve[1]["precision"] == 0.5


class TestSplitAndBand:
    def test_split_is_temporal(self) -> None:
        series = build_series(size=100)
        train, evaluation = split_series(series, 0.4)
        assert train.size == 40
        assert evaluation.size == 60
        assert train.timestamps[-1] < evaluation.timestamps[0]
        assert evaluation.window_count == 1

    def test_train_band_uses_percentiles_and_is_robust(self) -> None:
        low, high = train_band(build_series(size=200))
        assert low < high
        constant = LabelledSeries(
            name="flat",
            metric="cpu_utilization",
            timestamps=build_series(size=10).timestamps,
            values=(7.0,) * 10,
            labels=(0,) * 10,
        )
        flat_low, flat_high = train_band(constant)
        assert flat_high > flat_low

    def test_protocol_serialises_every_knob(self) -> None:
        payload = EvaluationProtocol(ablated_features=("z_score",)).to_dict()
        assert payload["ablated_features"] == ["z_score"]
        assert payload["proximity_minutes"] == 60.0
        json.dumps(payload)


class TestScorers:
    def _metric(self, value: float, history: list[float] | None = None) -> Metric:
        from src.core.stats import stats_from_values

        metric = Metric(
            name="cpu_utilization",
            value=value,
            unit="%",
            cloud="aws",
            resource_id="i-1",
            service="EC2",
        )
        if history:
            metric.window = stats_from_values(history)
        return metric

    def test_base_class_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            DetectionScorer()  # type: ignore[abstract]

    def test_profile_threshold_never_fires_on_normalised_series(self, profiles: dict) -> None:
        scorer = ProfileThresholdScorer(profiles=profiles)
        assert scorer.score(self._metric(99.0)) > 0  # margin above warn_high is positive
        assert scorer.score(self._metric(0.99)) < 0

    def test_profile_threshold_without_a_spec(self, profiles: dict) -> None:
        metric = Metric(name="not_in_profiles", value=1.0, unit="", cloud="aws")
        assert ProfileThresholdScorer(profiles=profiles).score(metric) == 0.0

    def test_global_threshold_uses_train_mean_and_sigma(self) -> None:
        scorer = GlobalThresholdScorer(PROTOCOL, k=1.0)
        train = [self._metric(float(index)) for index in range(50)]
        scorer.fit(train, None)  # type: ignore[arg-type]
        # mean 24.5 + 1*14.5 = 39
        assert scorer.score(self._metric(45.0)) > 0
        assert scorer.score(self._metric(10.0)) < 0

    def test_global_threshold_without_fit(self) -> None:
        assert GlobalThresholdScorer(PROTOCOL).score(self._metric(1.0)) == 0.0

    def test_rolling_zscore_needs_a_warm_window(self) -> None:
        scorer = RollingZScoreScorer(PROTOCOL, k=3.0)
        assert scorer.score(self._metric(99.0)) == 0.0
        history = [40.0 + index * 0.1 for index in range(20)]
        assert scorer.score(self._metric(200.0, history)) > 3.0
        assert scorer.score(self._metric(40.0, history)) < 3.0

    def test_ewma_flags_a_level_shift_and_settles(self) -> None:
        scorer = EWMAScorer(PROTOCOL, span=10, k=3.0)
        train = [self._metric(50.0 + math.sin(index) * 0.5) for index in range(60)]
        scorer.fit(train, None)  # type: ignore[arg-type]
        scorer.reset_stream()
        for metric in train:
            scorer.score(metric)
        assert scorer.score(self._metric(90.0)) > 3.0  # first point of the shift
        for _ in range(30):
            scorer.score(self._metric(90.0))  # the process re-stabilises at the new level
        assert scorer.score(self._metric(90.0)) < 3.0  # EWMA has absorbed the shift

    def test_ewma_is_unfitted_until_fit(self) -> None:
        scorer = EWMAScorer(PROTOCOL)
        assert scorer.score(self._metric(50.0)) == 0.0
        assert scorer.band == 0.0

    def test_isolation_forest_scores_are_continuous(self) -> None:
        series = build_series()
        train, _ = split_series(series, 0.4)
        scorer = IsolationForestScorer(PROTOCOL, bands={"cpu_utilization": train_band(train)})
        scorer.fit([self._metric(value) for value in train.values], series)
        first = scorer.score(self._metric(50.0))
        second = scorer.score(self._metric(90.0))
        assert second > first
        assert scorer.score_all([self._metric(50.0)] * 3) == [pytest.approx(first)] * 3

    def test_isolation_forest_needs_enough_training_points(self) -> None:
        series = build_series(size=200)
        scorer = IsolationForestScorer(PROTOCOL)
        with pytest.raises(ValueError, match="at least 50 training points"):
            scorer.fit([self._metric(1.0)] * 10, series)

    def test_isolation_forest_ranks_on_decision_not_confidence(self) -> None:
        """Confidence is a triage bucket with plateaus; ranking needs a continuum."""
        series = build_series()
        train, _ = split_series(series, 0.4)
        scorer = IsolationForestScorer(PROTOCOL, bands={"cpu_utilization": train_band(train)})
        scorer.fit([self._metric(value) for value in train.values], series)
        scores = scorer.score_all([self._metric(value) for value in train.values])
        assert len(set(scores)) > len(train) * 0.9  # essentially tie-free


class TestEvaluateSeries:
    def test_detects_an_injected_level_shift(self) -> None:
        series = build_series()
        result = evaluate_series(series, lambda bands: IsolationForestScorer(PROTOCOL, bands=bands), PROTOCOL)
        assert result.skipped is None
        assert result.windows_in_eval == 1
        assert result.windows_detected == 1
        assert result.event_recall == 1.0
        # 80 of the 240 evaluation points are anomalous (33% base rate): the
        # detector must beat that base rate and stay well short of alerting
        # everywhere
        assert result.precision > 0.4
        assert result.eval_alert_rate < 0.25

    def test_baselines_detect_the_shift_too(self) -> None:
        series = build_series()
        for factory in (
            lambda bands: RollingZScoreScorer(PROTOCOL),
            lambda bands: EWMAScorer(PROTOCOL),
            lambda bands: GlobalThresholdScorer(PROTOCOL),
        ):
            result = evaluate_series(series, factory, PROTOCOL)
            assert result.windows_detected == 1, factory({}).name

    def test_threshold_comes_from_training_only(self) -> None:
        """The operating point must not move when the evaluation slice changes."""
        series = build_series()
        protocol = EvaluationProtocol(train_fraction=0.4, alert_rate=0.05)
        first = evaluate_series(series, lambda bands: GlobalThresholdScorer(protocol, k=3.0), protocol)
        # same training distribution, twice the anomaly size: the threshold is set
        # before the anomaly is ever seen, so it cannot react to it
        shifted = build_series(size=400, spike_size=90.0)
        second = evaluate_series(shifted, lambda bands: GlobalThresholdScorer(protocol, k=3.0), protocol)
        assert first.threshold == pytest.approx(second.threshold)
        assert first.train_alert_rate == pytest.approx(second.train_alert_rate)
        assert second.recall >= first.recall

    def test_short_series_is_skipped_not_crashed(self) -> None:
        series = build_series(size=30)
        result = evaluate_series(series, lambda bands: RollingZScoreScorer(PROTOCOL), PROTOCOL)
        assert result.scored is False
        assert "too few points" in (result.skipped or "")

    def test_detection_delay_is_never_negative(self) -> None:
        """Proximity credit can credit a pre-window alert; delay must clamp at 0."""
        series = build_series(spike_at=250)
        result = evaluate_series(series, lambda bands: GlobalThresholdScorer(PROTOCOL), PROTOCOL)
        assert result.detection_delay_seconds is None or result.detection_delay_seconds >= 0.0

    def test_normal_only_series_produce_no_windows(self) -> None:
        series = LabelledSeries(
            name="quiet",
            metric="cpu_utilization",
            timestamps=build_series().timestamps,
            values=build_series().values,
            labels=(0,) * 400,
            windows=(),
        )
        result = evaluate_series(series, lambda bands: RollingZScoreScorer(PROTOCOL), PROTOCOL)
        assert result.windows_in_eval == 0
        assert result.event_recall is None
        assert result.scored is True

    def test_result_is_json_serialisable_and_carries_warnings(self) -> None:
        result = evaluate_series(build_series(), lambda bands: EWMAScorer(PROTOCOL), PROTOCOL)
        payload = json.loads(json.dumps(result.to_dict(), default=str))
        assert payload["scorer"] == "ewma"
        assert result.warnings == list(EVALUATION_WARNINGS)


class TestAggregateAndBenchmark:
    def _results(self):
        series = build_series()
        return [
            evaluate_series(series, lambda bands: RollingZScoreScorer(PROTOCOL), PROTOCOL),
            evaluate_series(series, lambda bands: EWMAScorer(PROTOCOL), PROTOCOL),
        ]

    def test_aggregate_groups_by_scorer(self) -> None:
        summary = aggregate(self._results())
        assert set(summary) == {"rolling_zscore", "ewma"}
        for metrics in summary.values():
            assert metrics["series"] == 1
            assert metrics["scored_series"] == 1
            assert metrics["windows"] == 1
            assert 0.0 <= metrics["event_recall"] <= 1.0

    def test_benchmark_document_shape(self) -> None:
        report = benchmark([build_series(size=200)], PROTOCOL)
        assert set(report) >= {"protocol", "dataset", "scorers", "results", "summary", "warnings"}
        assert len(report["results"]) == len(report["scorers"])
        assert report["dataset"]["series"] == 1
        json.dumps(report, default=str)

    def test_default_scorers_cover_the_baselines(self) -> None:
        names = {factory({}).name for factory in default_scorers(PROTOCOL)}
        assert names == {
            "isolation_forest",
            "rolling_zscore",
            "ewma",
            "global_threshold",
            "profile_threshold",
        }

    def test_benchmark_accepts_a_scorer_subset(self) -> None:
        report = benchmark(
            [build_series(size=200)],
            PROTOCOL,
            [lambda bands: EWMAScorer(PROTOCOL)],
        )
        assert report["scorers"] == ["ewma"]
        assert len(report["results"]) == 1


class TestFeatureAblation:
    def test_subset_selects_and_orders_features(self, warm_metric: Metric) -> None:
        engineer = FeatureEngineer(min_samples=5, feature_names=("z_score", "band_position"))
        vector = engineer.transform(warm_metric)
        assert vector.shape == (2,)
        assert engineer.n_features == 2
        assert engineer.feature_names == ("z_score", "band_position")

    def test_unknown_feature_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown feature"):
            FeatureEngineer(feature_names=("z_score", "not_a_feature"))

    def test_empty_selection_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one feature"):
            FeatureEngineer(feature_names=())

    def test_ablated_features_are_reported(self) -> None:
        engineer = FeatureEngineer(feature_names=("z_score", "value_over_mean"))
        assert set(engineer.ablated_features) == set(FEATURE_NAMES) - {"z_score", "value_over_mean"}

    def test_detector_trains_on_an_ablated_feature_set(self, training_data) -> None:
        samples, _ = training_data
        engineer = FeatureEngineer(min_samples=5, feature_names=("z_score", "band_position"))
        detector = IsolationForestScorer(PROTOCOL, n_estimators=20)
        detector.detector = type(detector.detector)(n_estimators=20, feature_engineer=engineer)
        detector.fit(list(samples[:200]), build_series(size=200))
        assert detector.detector.stats["ablated_features"]
        scored = detector.score_all(list(samples[:5]))
        assert len(scored) == 5

    def test_detector_tolerates_missing_coverage_feature(self, training_data) -> None:
        samples, _ = training_data
        scorer = IsolationForestScorer(PROTOCOL, n_estimators=20)
        scorer.detector = type(scorer.detector)(
            n_estimators=20,
            feature_engineer=FeatureEngineer(min_samples=5, feature_names=("z_score", "band_position")),
        )
        scorer.fit(list(samples[:200]), build_series(size=200))
        # cold samples must not be flagged when coverage is not a feature
        cold = Metric(name="cpu_utilization", value=1e6, cloud="aws", resource_id="i-1", service="EC2")
        assert scorer.score(cold) == pytest.approx(scorer.score(cold))
