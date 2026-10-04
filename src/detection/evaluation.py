"""Evaluation on real labelled data.

Does the detection approach work on telemetry it did not generate? Four scorers
share one interface so the comparison is apples-to-apples.

===============================  =========================================
``isolation_forest``            this project's :class:`AnomalyDetector`
``rolling_zscore``              deviation from the rolling mean, in sigma
``ewma``                        EWMA control chart - what most on-call teams
                                actually run
``global_threshold``            ``mean + k·sigma`` fitted on the train split
``profile_threshold``           the fixed ``warn_high`` values in
                                ``cloud_profiles.yaml``
===============================  =========================================

Protocol
--------

1. **Temporal split.** Training is everything before the cut, scoring is
   everything after. No shuffling, no leakage.
2. **Operating point from training only.** Every scorer's threshold is the
   ``(1 - alert_rate)`` quantile of *its own training scores*. That is how
   unsupervised detectors are compared fairly: nobody tunes a threshold on the
   evaluation split, and by construction they all alert at the same rate.
3. **Both point and event metrics.** NAB windows are wide (5-39 hours, about
   10% of each series), so a detector that fires forever scores recall 1.0 and
   precision 0.1. Reported next to point-wise precision/recall: *event recall*
   (was the window hit at all), *detection delay*, and *false alarms per day*
   outside windows.

Point-adjust - crediting every point after the first hit - is deliberately not
used: it inflates precision and recall for detectors that detect each event once.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np

from src.data.timeseries import ANOMALOUS, LabelledSeries
from src.detection.anomaly_detector import AnomalyDetector
from src.detection.feature_engineer import FEATURE_NAMES, FeatureEngineer
from src.ingestion.history import SeriesHistory
from src.ingestion.metric_schema import Metric
from src.ingestion.replay_collector import ReplayCollector

__all__ = [
    "DEFAULT_ALERT_RATE",
    "EVALUATION_WARNINGS",
    "DetectionScorer",
    "EWMAScorer",
    "EvaluationProtocol",
    "GlobalThresholdScorer",
    "IsolationForestScorer",
    "ProfileThresholdScorer",
    "RollingZScoreScorer",
    "ScorerFactory",
    "SeriesResult",
    "aggregate",
    "alert_budget_curve",
    "average_precision",
    "benchmark",
    "default_scorers",
    "evaluate_series",
    "quantile_threshold",
    "roc_auc",
    "run_count",
]

MIN_TRAIN_POINTS = 60
DEFAULT_ALERT_RATE = 0.02
DEFAULT_PROXIMITY = timedelta(hours=1)
EVALUATION_WARNINGS: tuple[str, ...] = (
    "point-wise precision is capped by benchmark window width; judge on-call impact with event "
    "recall and false alarms per day",
    "thresholds come from training quantiles over warm-window samples, never from tuning on the "
    "evaluation split",
    "each model trains on the training slice of the same series: there is no cross-series transfer, "
    "so these numbers are a lower bound on what a pooled model would achieve",
    "NAB's realAWSCloudwatch subset is deliberately hard - published leaderboards show most "
    "detectors barely beat random on it",
)

ScorerFactory = Callable[[dict[str, tuple[float, float]]], "DetectionScorer"]


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    """How the benchmark runs. Recorded verbatim in every result file."""

    train_fraction: float = 0.4
    alert_rate: float = DEFAULT_ALERT_RATE
    min_history: int = 5
    history_window: int = 30
    proximity: timedelta = DEFAULT_PROXIMITY
    contamination: float = 0.05
    n_estimators: int = 150
    random_state: int = 42
    zscore_k: float = 3.0
    ewma_span: int = 20
    ewma_k: float = 3.0
    global_k: float = 3.0
    ablated_features: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_fraction": self.train_fraction,
            "alert_rate": self.alert_rate,
            "min_history": self.min_history,
            "history_window": self.history_window,
            "proximity_minutes": self.proximity.total_seconds() / 60.0,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "zscore_k": self.zscore_k,
            "ewma_span": self.ewma_span,
            "ewma_k": self.ewma_k,
            "global_k": self.global_k,
            "ablated_features": list(self.ablated_features),
        }


# ── scorers ─────────────────────────────────────────────────────────────


class DetectionScorer(ABC):
    """Scores points of one series in stream order; higher is more anomalous.

    ``fit`` sees the training slice only. ``score`` must be causal: it may use
    the metric's rolling window and its own running state, never a future sample.
    """

    name: ClassVar[str] = "scorer"
    description: ClassVar[str] = ""

    def __init__(self, protocol: EvaluationProtocol | None = None, **options: Any) -> None:
        self.protocol = protocol or EvaluationProtocol()
        self.options = options

    def fit(self, metrics: Sequence[Metric], series: LabelledSeries) -> Self:
        """Learn parameters from the training slice."""
        return self

    def reset_stream(self) -> None:  # noqa: B027 - optional hook, stateless scorers need no override
        """Drop running state before a scored replay; fitted parameters survive.

        Optional: scorers with no running state inherit this no-op.
        """

    @abstractmethod
    def score(self, metric: Metric) -> float:
        """Score one point. Higher means more anomalous."""

    def score_all(self, metrics: Sequence[Metric]) -> list[float]:
        """Score a block of points. Overridden where batching is equivalent."""
        return [float(self.score(metric)) for metric in metrics]

    def describe(self) -> str:
        return self.description or self.name


class ProfileThresholdScorer(DetectionScorer):
    """Fixed ``warn_high`` thresholds from ``cloud_profiles.yaml``.

    Included because it shows a real finding: hand-written thresholds do not
    transfer to real traffic. NAB CPU series are normalised to 0-1, so a profile
    threshold of 78% never fires no matter how abnormal the data gets.

    The score is the signed margin above the threshold, so the scorer stays
    rankable (a 0/1 score would make ROC AUC meaningless).
    """

    name = "profile_threshold"

    def __init__(self, protocol: EvaluationProtocol | None = None, *, profiles: dict[str, Any] | None = None, **options) -> None:
        super().__init__(protocol, **options)
        if profiles is None:
            from config.settings import CLOUD_PROFILES

            profiles = CLOUD_PROFILES
        self.specs: dict[str, Any] = profiles.get("metrics", {})

    def score(self, metric: Metric) -> float:
        spec = self.specs.get(metric.name) or {}
        warn = spec.get("warn_high")
        if warn is None:
            return 0.0
        try:
            return float(metric.value) - float(warn)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0.0


class GlobalThresholdScorer(DetectionScorer):
    """``mean + k·sigma`` fitted on the training slice - the classic static alert.

    Scores the signed margin above the fitted threshold so it stays rankable.
    """

    name = "global_threshold"

    def __init__(self, protocol: EvaluationProtocol | None = None, *, k: float | None = None, **options) -> None:
        super().__init__(protocol, **options)
        self.k = self.protocol.global_k if k is None else float(k)
        self._threshold: float | None = None

    def fit(self, metrics: Sequence[Metric], series: LabelledSeries) -> Self:
        values = np.asarray([metric.value for metric in metrics], dtype=np.float64)
        if values.size:
            self._threshold = float(values.mean() + self.k * values.std())
        return self

    def score(self, metric: Metric) -> float:
        if self._threshold is None:
            return 0.0
        return float(metric.value) - self._threshold


class RollingZScoreScorer(DetectionScorer):
    """Absolute deviation from the rolling mean, in sigma."""

    name = "rolling_zscore"

    def __init__(self, protocol: EvaluationProtocol | None = None, *, k: float | None = None, **options) -> None:
        super().__init__(protocol, **options)
        self.k = self.protocol.zscore_k if k is None else float(k)

    def score(self, metric: Metric) -> float:
        window = metric.window
        if window is None or not window.is_ready(self.protocol.min_history):
            return 0.0
        return abs(window.z_score(metric.value))


class EWMAScorer(DetectionScorer):
    """Exponentially weighted moving-average control chart.

    The detector most on-call teams actually run: a running mean plus a band
    whose width follows the EWMA variance (Roberts' update). The score is the
    deviation in band-widths, so a score of ``k`` means "outside the k-sigma
    control limit" and the value drifts back down once the process re-stabilises.
    """

    name = "ewma"

    def __init__(
        self,
        protocol: EvaluationProtocol | None = None,
        *,
        span: int | None = None,
        k: float | None = None,
        **options,
    ) -> None:
        super().__init__(protocol, **options)
        self.span = self.protocol.ewma_span if span is None else int(span)
        self.k = self.protocol.ewma_k if k is None else float(k)
        self.alpha = 2.0 / (self.span + 1.0)
        self._fit_mean: float | None = None
        self._fit_std: float = 0.0
        self._mean: float | None = None
        self._variance = 0.0
        self._observations = 0

    def fit(self, metrics: Sequence[Metric], series: LabelledSeries) -> Self:
        values = np.asarray([metric.value for metric in metrics], dtype=np.float64)
        if values.size >= 2:
            self._fit_mean = float(values.mean())
            self._fit_std = float(values.std(ddof=1))
        return self

    def reset_stream(self) -> None:
        self._mean = None
        self._variance = 0.0
        self._observations = 0

    @property
    def sigma(self) -> float:
        """Current EWMA sigma estimate, falling back to the training sigma."""
        if self._observations < 2 or self._variance <= 0:
            return self._fit_std
        return math.sqrt(self._variance * self.alpha / (2.0 - self.alpha))

    @property
    def band(self) -> float:
        """Control limit in the metric's own units."""
        return self.k * self.sigma

    def score(self, metric: Metric) -> float:
        value = float(metric.value)
        if self._mean is None:
            # first scored point: seed the running state from the training slice
            self._mean = self._fit_mean if self._fit_mean is not None else value
            self._variance = self._fit_std**2
            self._observations = 1
            return 0.0
        sigma = self.sigma
        score = abs(value - self._mean) / sigma if sigma > 0 else 0.0
        delta = value - self._mean
        self._mean += self.alpha * delta
        self._observations += 1
        self._variance = (1.0 - self.alpha) * (self._variance + self.alpha * delta * delta)
        return score


class IsolationForestScorer(DetectionScorer):
    """This project's detector, trained on the training slice of the same series.

    Expected bands come from training quantiles (see
    :func:`src.data.timeseries.derive_bands`), because real telemetry has no
    universal "normal" range to take from configuration.
    """

    name = "isolation_forest"

    def __init__(
        self,
        protocol: EvaluationProtocol | None = None,
        *,
        bands: dict[str, tuple[float, float]] | None = None,
        contamination: float | None = None,
        n_estimators: int | None = None,
        **options,
    ) -> None:
        super().__init__(protocol, **options)
        self.bands = dict(bands or {})
        kept = tuple(name for name in FEATURE_NAMES if name not in self.protocol.ablated_features)
        self.detector = AnomalyDetector(
            contamination=self.protocol.contamination if contamination is None else contamination,
            n_estimators=self.protocol.n_estimators if n_estimators is None else n_estimators,
            random_state=self.protocol.random_state,
            min_history=self.protocol.min_history,
            feature_engineer=FeatureEngineer(
                min_samples=self.protocol.min_history,
                bands=self.bands,
                feature_names=kept,
            ),
        )

    def fit(self, metrics: Sequence[Metric], series: LabelledSeries) -> Self:
        if len(metrics) < 50:
            raise ValueError(f"{series.name}: need at least 50 training points, got {len(metrics)}")
        self.detector.train(list(metrics))
        return self

    def score(self, metric: Metric) -> float:
        """Rank on the continuous decision value, not on the bucketed confidence.

        ``Metric.confidence`` is deliberately coarse (it is a triage bucket for
        humans and Slack cards), and its plateaus create ties that make ROC AUC
        meaningless. Evaluation needs a continuous rank, so the negated
        ``decision_function`` is used: higher means more anomalous.
        """
        return float(self.detector.score_batch([metric])[0].anomaly_score)

    def score_all(self, metrics: Sequence[Metric]) -> list[float]:
        """Batched: the detector is stateless at inference time."""
        if not metrics:
            return []
        return [float(metric.anomaly_score or 0.0) for metric in self.detector.score_batch(list(metrics))]

    def model_stats(self) -> dict[str, Any]:
        return dict(self.detector.stats)


def default_scorers(protocol: EvaluationProtocol) -> list[ScorerFactory]:
    """Scorer factories evaluated for every series."""
    return [
        lambda bands: IsolationForestScorer(protocol, bands=bands),
        lambda bands: RollingZScoreScorer(protocol),
        lambda bands: EWMAScorer(protocol),
        lambda bands: GlobalThresholdScorer(protocol),
        lambda bands: ProfileThresholdScorer(protocol),
    ]


# ── metric helpers ──────────────────────────────────────────────────────


def run_count(flags: Sequence[bool]) -> int:
    """Number of contiguous True runs: each run is one alert episode."""
    return sum(1 for index, flag in enumerate(flags) if flag and (index == 0 or not flags[index - 1]))


def quantile_threshold(scores: Sequence[float], alert_rate: float) -> float:
    """Threshold that alerts on ``alert_rate`` of the given scores."""
    values = sorted(float(score) for score in scores)
    if not values:
        return math.inf
    keep = max(1, round(len(values) * alert_rate))
    return values[-keep]


def alert_budget_curve(
    scores: Sequence[float],
    labels: Sequence[int],
    budgets: Iterable[float],
) -> list[dict[str, float]]:
    """Precision and recall at fixed alert budgets (top-x% most anomalous)."""
    curve: list[dict[str, float]] = []
    positives = sum(1 for label in labels if label == ANOMALOUS)
    ordered = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    for budget in budgets:
        take = max(1, round(len(ordered) * budget))
        hits = sum(1 for index in ordered[:take] if labels[index] == ANOMALOUS)
        curve.append(
            {
                "budget": round(budget, 6),
                "alerts": take,
                "precision": round(hits / take, 6) if take else 0.0,
                "recall": round(hits / positives, 6) if positives else 0.0,
            }
        )
    return curve


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Average precision (area under the precision-recall curve)."""
    positives = sum(1 for label in labels if label == ANOMALOUS)
    if positives == 0 or not scores:
        return 0.0
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    hits = 0
    total = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == ANOMALOUS:
            hits += 1
            total += hits / rank
    return round(total / positives, 6)


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """ROC AUC via the rank-sum identity, with ties averaged."""
    positives = sum(1 for label in labels if label == ANOMALOUS)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in order[position : end + 1]:
            ranks[index] = average
        position = end + 1
    positive_rank_sum = sum(ranks[index] for index, label in enumerate(labels) if label == ANOMALOUS)
    return round((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives), 6)


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


# ── per-series evaluation ───────────────────────────────────────────────


@dataclass(slots=True)
class SeriesResult:
    """Outcome for one series under one scorer."""

    series: str
    metric: str
    scorer: str
    train_points: int
    eval_points: int
    scored_points: int
    windows_in_eval: int
    windows_detected: int
    points_in_windows: int
    detected_in_windows: int
    precision: float
    recall: float
    f1: float
    precision_strict: float
    false_alarms: int
    false_alarms_per_day: float
    detection_delay_seconds: float | None
    average_precision: float
    roc_auc: float
    threshold: float
    train_alert_rate: float
    eval_alert_rate: float
    skipped: str | None = None
    warnings: list[str] = field(default_factory=lambda: list(EVALUATION_WARNINGS))

    @property
    def scored(self) -> bool:
        return self.skipped is None

    @property
    def event_recall(self) -> float | None:
        if self.windows_in_eval == 0:
            return None
        return round(self.windows_detected / self.windows_in_eval, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "series": self.series,
            "metric": self.metric,
            "scorer": self.scorer,
            "skipped": self.skipped,
            "train_points": self.train_points,
            "eval_points": self.eval_points,
            "scored_points": self.scored_points,
            "windows_in_eval": self.windows_in_eval,
            "windows_detected": self.windows_detected,
            "event_recall": self.event_recall,
            "points_in_windows": self.points_in_windows,
            "detected_in_windows": self.detected_in_windows,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "precision_strict": self.precision_strict,
            "false_alarms": self.false_alarms,
            "false_alarms_per_day": self.false_alarms_per_day,
            "detection_delay_seconds": self.detection_delay_seconds,
            "average_precision": self.average_precision,
            "roc_auc": self.roc_auc,
            "threshold": self.threshold,
            "train_alert_rate": self.train_alert_rate,
            "eval_alert_rate": self.eval_alert_rate,
        }


def split_series(series: LabelledSeries, train_fraction: float) -> tuple[LabelledSeries, LabelledSeries]:
    """Cut once, by position: training first, evaluation after."""
    if len(series) < 2:
        return series, series.slice(0, 0)
    cut = max(1, min(int(len(series) * train_fraction), len(series) - 1))
    return series.slice(0, cut), series.slice(cut)


def train_band(series: LabelledSeries, lower: float = 0.01, upper: float = 0.99) -> tuple[float, float]:
    """Expected band from the training slice only (1st/99th percentile)."""
    values = sorted(series.values)
    if not values:
        return (0.0, 1.0)
    low = values[int((len(values) - 1) * lower)]
    high = values[int((len(values) - 1) * upper)]
    if high - low < 1e-12:
        low, high = low - 0.5, high + 0.5
    return (float(low), float(high))


def evaluate_series(
    series: LabelledSeries,
    factory: ScorerFactory,
    protocol: EvaluationProtocol | None = None,
) -> SeriesResult:
    """Stream one series through one scorer and score the evaluation slice.

    The rolling window is continuous across the split - the evaluation slice
    sees the same history a live detector would have - while the model and its
    operating point only ever see the training slice.
    """
    resolved = protocol or EvaluationProtocol()
    train, evaluation = split_series(series, resolved.train_fraction)
    scorer = factory({series.metric: train_band(train)})

    if len(train) < MIN_TRAIN_POINTS or len(evaluation) < resolved.min_history:
        return SeriesResult(
            series=series.name,
            metric=series.metric,
            scorer=scorer.name,
            train_points=len(train),
            eval_points=len(evaluation),
            scored_points=0,
            windows_in_eval=evaluation.window_count,
            windows_detected=0,
            points_in_windows=0,
            detected_in_windows=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            precision_strict=0.0,
            false_alarms=0,
            false_alarms_per_day=0.0,
            detection_delay_seconds=None,
            average_precision=0.0,
            roc_auc=0.0,
            threshold=0.0,
            train_alert_rate=0.0,
            eval_alert_rate=0.0,
            skipped=f"too few points (train={len(train)}, eval={len(evaluation)})",
        )

    history = SeriesHistory(window=resolved.history_window, min_samples=resolved.min_history)
    collector = ReplayCollector(series, history=history)

    train_metrics: list[Metric] = []
    while collector.index < len(train):
        batch = collector.collect_and_track()
        if not batch:
            break
        train_metrics.append(batch[0])

    scorer.fit(train_metrics, train)
    scorer.reset_stream()
    train_scores = scorer.score_all(train_metrics)
    threshold, train_alert_rate = _operating_point(train_scores, train_metrics, resolved.alert_rate, resolved)

    eval_metrics: list[Metric] = []
    while not collector.exhausted:
        batch = collector.collect_and_track()
        if not batch:
            break
        eval_metrics.append(batch[0])

    eval_scores = scorer.score_all(eval_metrics)
    labels = list(evaluation.labels)
    flags = [score >= threshold for score in eval_scores]

    return _summarise(
        series=series,
        train=train,
        evaluation=evaluation,
        scorer_name=scorer.name,
        train_scores=train_scores,
        eval_scores=eval_scores,
        labels=labels,
        flags=flags,
        threshold=threshold,
        protocol=resolved,
        train_alert_rate=train_alert_rate,
    )


def _warm_mask(metrics: Sequence[Metric], protocol: EvaluationProtocol) -> list[bool]:
    """Which points the detector is actually able to judge.

    The detector refuses to flag a sample whose rolling window is not warm, so
    those samples must not define the operating point either: including them put
    cold-start artefacts at the top of the training quantile, which suppressed
    every real detection in the evaluation slice.
    """
    return [
        metric.window is not None and metric.window.is_ready(protocol.min_history) for metric in metrics
    ]


def _operating_point(
    scores: Sequence[float],
    metrics: Sequence[Metric],
    alert_rate: float,
    protocol: EvaluationProtocol,
) -> tuple[float, float]:
    """Alert threshold (and the rate it implies on training data).

    Derived from warm samples only; falls back to all samples when the series is
    too short to have a warm window at all.
    """
    warm = [score for score, ready in zip(scores, _warm_mask(metrics, protocol), strict=True) if ready]
    reference = warm if len(warm) >= 20 else list(scores)
    threshold = quantile_threshold(reference, alert_rate)
    return threshold, _rate([score >= threshold for score in scores])


def _rate(flags: Sequence[bool]) -> float:
    return round(sum(1 for flag in flags if flag) / len(flags), 6) if flags else 0.0


def _summarise(
    *,
    series: LabelledSeries,
    train: LabelledSeries,
    evaluation: LabelledSeries,
    scorer_name: str,
    train_scores: Sequence[float],
    eval_scores: Sequence[float],
    labels: Sequence[int],
    flags: Sequence[bool],
    threshold: float,
    protocol: EvaluationProtocol,
    train_alert_rate: float,
) -> SeriesResult:
    positives = sum(1 for label in labels if label == ANOMALOUS)
    predicted = sum(1 for flag in flags if flag)
    hits = sum(1 for flag, label in zip(flags, labels, strict=True) if flag and label == ANOMALOUS)
    precision = hits / predicted if predicted else 0.0
    recall = hits / positives if positives else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    windows_detected = 0
    delays: list[float] = []
    for window in evaluation.windows:
        indices = [
            index
            for index, stamp in enumerate(evaluation.timestamps)
            if window.contains(stamp, protocol.proximity)
        ]
        if not indices:
            continue
        hit_indices = [index for index in indices if flags[index]]
        if hit_indices:
            windows_detected += 1
            # Clamp at zero: proximity credit lets a detector that was already
            # firing before the window opened count as an instant hit, which is
            # a real property but not a negative response time.
            delays.append(max(0.0, (evaluation.timestamps[hit_indices[0]] - window.start).total_seconds()))

    strict_hits = sum(
        1
        for flag, label, stamp in zip(flags, labels, evaluation.timestamps, strict=True)
        if flag and label == ANOMALOUS and any(window.contains(stamp) for window in evaluation.windows)
    )
    precision_strict = strict_hits / predicted if predicted else 0.0

    outside = [flag and label == 0 for flag, label in zip(flags, labels, strict=True)]
    false_alarms = run_count(outside)
    days = evaluation.span.total_seconds() / 86400.0

    return SeriesResult(
        series=series.name,
        metric=series.metric,
        scorer=scorer_name,
        train_points=len(train),
        eval_points=len(evaluation),
        scored_points=len(eval_scores),
        windows_in_eval=evaluation.window_count,
        windows_detected=windows_detected,
        points_in_windows=positives,
        detected_in_windows=hits,
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        precision_strict=round(precision_strict, 6),
        false_alarms=false_alarms,
        false_alarms_per_day=round(false_alarms / days, 4) if days else 0.0,
        detection_delay_seconds=_mean(delays) if delays else None,
        average_precision=average_precision(labels, eval_scores),
        roc_auc=roc_auc(labels, eval_scores),
        threshold=round(float(threshold), 6),
        train_alert_rate=train_alert_rate,
        eval_alert_rate=_rate(flags),
    )


# ── benchmark ───────────────────────────────────────────────────────────


def aggregate(results: Sequence[SeriesResult]) -> dict[str, dict[str, Any]]:
    """Macro-average per scorer over series that have an evaluation window.

    Series without an evaluation window (the benchmark ships normal-only
    controls) cannot contribute recall, so they are summarised separately as a
    pure false-alarm check.
    """
    by_scorer: dict[str, list[SeriesResult]] = {}
    for result in results:
        by_scorer.setdefault(result.scorer, []).append(result)

    summary: dict[str, dict[str, Any]] = {}
    for scorer, items in sorted(by_scorer.items()):
        scorable = [item for item in items if item.scored and item.windows_in_eval > 0]
        normal_only = [item for item in items if item.scored and item.windows_in_eval == 0]
        delays = [item.detection_delay_seconds for item in scorable if item.detection_delay_seconds is not None]
        summary[scorer] = {
            "series": len(items),
            "scored_series": len(scorable),
            "normal_only_series": len(normal_only),
            "skipped": len(items) - len(scorable) - len(normal_only),
            "windows": sum(item.windows_in_eval for item in scorable),
            "windows_detected": sum(item.windows_detected for item in scorable),
            "event_recall": _mean([item.event_recall for item in scorable if item.event_recall is not None]),
            "precision": _mean([item.precision for item in scorable]),
            "recall": _mean([item.recall for item in scorable]),
            "f1": _mean([item.f1 for item in scorable]),
            "precision_strict": _mean([item.precision_strict for item in scorable]),
            "mean_detection_delay_minutes": round(_mean(delays) / 60.0, 2) if delays else None,
            "false_alarms_per_day": _mean([item.false_alarms_per_day for item in scorable]),
            "false_alarms_per_day_normal_only": _mean([item.false_alarms_per_day for item in normal_only]),
            "average_precision": _mean([item.average_precision for item in scorable]),
            "roc_auc": _mean([item.roc_auc for item in scorable]),
        }
    return summary


def dataset_summary(series: Sequence[LabelledSeries]) -> dict[str, Any]:
    metrics: dict[str, int] = {}
    for item in series:
        metrics[item.metric] = metrics.get(item.metric, 0) + 1
    return {
        "series": len(series),
        "points": sum(item.size for item in series),
        "labelled_series": sum(1 for item in series if item.is_labelled),
        "windows": sum(item.window_count for item in series),
        "metrics": dict(sorted(metrics.items())),
        "examples": sorted({Path(item.source).name for item in series})[:3],
    }


def benchmark(
    series: Sequence[LabelledSeries],
    protocol: EvaluationProtocol | None = None,
    factories: Sequence[ScorerFactory] | None = None,
) -> dict[str, Any]:
    """Run every scorer over every series; return per-series and aggregate results."""
    resolved = protocol or EvaluationProtocol()
    scorer_factories = list(factories) if factories is not None else default_scorers(resolved)

    results: list[SeriesResult] = []
    for item in series:
        for factory in scorer_factories:
            results.append(evaluate_series(item, factory, resolved))

    return {
        "protocol": resolved.to_dict(),
        "dataset": dataset_summary(series),
        "scorers": [factory({}).name for factory in scorer_factories],
        "results": [result.to_dict() for result in results],
        "summary": aggregate(results),
        "warnings": list(EVALUATION_WARNINGS),
    }
