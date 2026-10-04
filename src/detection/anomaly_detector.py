"""Isolation Forest anomaly detector.

Design notes
------------
* Anomaly score is the negated scikit-learn ``decision_function`` so that
  *higher means more anomalous* everywhere in this codebase.
* Outlier classification uses the forest's own ``predict``.
* Confidence is interpolated from the *training* decision distribution rather
  than from an arbitrary sigmoid: the score at the model's contamination
  percentile maps to 0.5, so "just past the model's own outlier threshold"
  reads as "probably an anomaly", and the tail saturates towards 1.0. The
  anchors are stored in the model bundle, so a reloaded model scores exactly
  like the one that was trained.
* Confidence is damped by ``history_coverage`` so samples collected before the
  rolling window is warm cannot page anyone.
* ``DriftMonitor`` observes every scored batch; when the live distribution moves
  away from training, :attr:`AnomalyDetector.retrain_recommended` flips and the
  operator (or ``Pipeline``) retrains.

The trained artefact is a single joblib bundle so a model can be loaded without
the training code path or the original feature matrix.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.core.clock import utc_now
from src.detection.feature_engineer import DriftMonitor, FeatureEngineer
from src.ingestion.metric_schema import Metric

__all__ = ["MODEL_FORMAT_VERSION", "AnomalyDetector", "DetectionOutcome"]

logger = logging.getLogger(__name__)

MODEL_FORMAT_VERSION = 2
_STD_FLOOR = 1e-6

#: (percentile of the training decision distribution, confidence at/above it)
#: stored ascending by percentile; ``None`` means the extreme minimum.
CONFIDENCE_PERCENTILES: tuple[tuple[float, float], ...] = (
    (50.0, 0.15),
    (5.0, 0.50),
    (1.0, 0.75),
    (0.0, 0.95),
)


@dataclass(slots=True)
class DetectionOutcome:
    """Per-batch detection summary (one per metric in the scored batch)."""

    metric: Metric
    anomaly_score: float
    confidence: float
    is_anomaly: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "series": self.metric.series_key,
            "metric": self.metric.name,
            "value": self.metric.value,
            "anomaly_score": round(self.anomaly_score, 6),
            "confidence": round(self.confidence, 6),
            "is_anomaly": self.is_anomaly,
        }


@dataclass(slots=True)
class _RunningStats:
    scored: int = 0
    anomalies: int = 0
    confidence_sum: float = 0.0
    score_sum: float = 0.0
    batches: int = 0

    def as_dict(self) -> dict[str, Any]:
        scored = max(1, self.scored)
        return {
            "metrics_scored": self.scored,
            "batches_scored": self.batches,
            "anomalies_detected": self.anomalies,
            "anomaly_rate": round(self.anomalies / scored, 6),
            "mean_confidence": round(self.confidence_sum / scored, 6),
            "mean_anomaly_score": round(self.score_sum / scored, 6),
        }


class AnomalyDetector:
    """Unsupervised anomaly detection with confidence scoring and drift watch."""

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 150,
        random_state: int = 42,
        min_history: int = 5,
        drift_threshold: float = 0.35,
        drift_min_batches: int = 10,
        feature_engineer: FeatureEngineer | None = None,
        profiles: dict[str, Any] | None = None,
    ) -> None:
        if not 0.0 < float(contamination) < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")
        if int(n_estimators) < 10:
            raise ValueError("n_estimators must be >= 10")

        self.contamination = float(contamination)
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.min_history = max(1, int(min_history))
        self.feature_engineer = feature_engineer or self._default_engineer(profiles, self.min_history)
        self.drift_monitor = DriftMonitor(threshold=drift_threshold, min_batches=drift_min_batches)

        self._model: IsolationForest | None = None
        self._scaler: StandardScaler | None = None
        self._decision_mean = 0.0
        self._decision_std = 1.0
        self._confidence_anchors: tuple[tuple[float, float], ...] = ()
        self._trained_at: datetime | None = None
        self._n_samples = 0
        self._metrics_seen: list[str] = []
        self._running = _RunningStats()
        self._feature_summary: dict[str, dict[str, float]] = {}

    def _coverage(self, raw: np.ndarray) -> np.ndarray:
        """History coverage per row, or ones when the feature was ablated."""
        names = self.feature_engineer.feature_names
        if "history_coverage" in names:
            return raw[:, names.index("history_coverage")]
        return np.ones(raw.shape[0], dtype=np.float64)

    @staticmethod
    def _default_engineer(profiles: dict[str, Any] | None, min_samples: int) -> FeatureEngineer:
        """Feature engineer with per-metric bands taken from the YAML profiles."""
        if profiles is None:
            from config.settings import CLOUD_PROFILES

            profiles = CLOUD_PROFILES
        return FeatureEngineer.from_profiles(profiles, min_samples=min_samples)

    # ── state ──────────────────────────────────────────────────────────
    @property
    def is_fitted(self) -> bool:
        return self._model is not None and self._scaler is not None

    @property
    def version(self) -> str:
        if self._trained_at is None:
            return "unfitted"
        return f"if-{self._trained_at.strftime('%Y%m%d%H%M%S')}-{self._n_samples}"

    @property
    def stats(self) -> dict[str, Any]:
        """Everything the dashboard and the API need, in one dict."""
        payload: dict[str, Any] = {
            "fitted": self.is_fitted,
            "version": self.version,
            "trained_at": self._trained_at.isoformat() if self._trained_at else None,
            "training_samples": self._n_samples,
            "training_metrics": list(self._metrics_seen),
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "features": list(self.feature_engineer.feature_names),
            "ablated_features": list(self.feature_engineer.ablated_features),
            "min_history": self.min_history,
            "decision_threshold": float(self._model.offset_) if self.is_fitted else None,
            "decision_offset": float(self._model.offset_) if self.is_fitted else None,
            "decision_mean": round(self._decision_mean, 6),
            "decision_std": round(self._decision_std, 6),
            "confidence_anchors": [
                {"decision": round(decision, 6), "confidence": confidence}
                for decision, confidence in self._confidence_anchors
            ],
            "drift_threshold": self.drift_monitor.threshold,
            "drift_score": round(self.drift_monitor.drift_score, 6),
            "drift_batches": self.drift_monitor.batches_seen,
            "drifted": self.drift_monitor.drifted,
            "retrain_recommended": self.retrain_recommended,
        }
        payload.update(self._running.as_dict())
        return payload

    @property
    def retrain_recommended(self) -> bool:
        return self.is_fitted and self.drift_monitor.drifted

    @property
    def feature_summary(self) -> dict[str, dict[str, float]]:
        return dict(self._feature_summary)

    def require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("detector is not fitted; call train() or load() first")

    # ── training ───────────────────────────────────────────────────────
    def train(self, samples: Sequence[Metric] | Iterable[Metric] | np.ndarray) -> dict[str, Any]:
        """Fit the scaler + Isolation Forest on ``samples``.

        Accepts metrics (features are engineered on the fly) or a ready-made
        feature matrix.
        """
        matrix = self._as_matrix(samples)
        if matrix.shape[0] < 50:
            raise ValueError(f"need at least 50 samples to train, got {matrix.shape[0]}")
        return self.train_from_features(matrix)

    def train_from_features(self, matrix: np.ndarray) -> dict[str, Any]:
        expected_width = self.feature_engineer.n_features
        if matrix.ndim != 2 or matrix.shape[1] != expected_width:
            raise ValueError(f"expected (n, {expected_width}) feature matrix, got {matrix.shape}")

        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix)

        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(scaled)

        decisions = model.decision_function(scaled)
        self._decision_mean = float(np.mean(decisions))
        self._decision_std = float(max(np.std(decisions), _STD_FLOOR))
        self._confidence_anchors = self._anchors_from(decisions)

        self._model = model
        self._scaler = scaler
        self._trained_at = utc_now()
        self._n_samples = int(matrix.shape[0])
        self._metrics_seen = sorted({name for name in self._metrics_seen if name})
        self._running = _RunningStats()
        self._feature_summary = self.feature_engineer.feature_report(matrix)
        self.drift_monitor.fit_baseline(scaled)

        summary = {
            "training_samples": self._n_samples,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "decision_offset": float(model.offset_),
            "decision_mean": round(self._decision_mean, 6),
            "decision_std": round(self._decision_std, 6),
            "confidence_anchors": [
                {"decision": round(decision, 6), "confidence": confidence}
                for decision, confidence in self._confidence_anchors
            ],
            "features": list(self.feature_engineer.feature_names),
            "ablated_features": list(self.feature_engineer.ablated_features),
            "metrics": list(self._metrics_seen),
            "trained_at": self._trained_at.isoformat(),
        }
        logger.info("detector trained on %d samples (offset=%.5f)", self._n_samples, model.offset_)
        return summary

    def observe_metric_names(self, names: Iterable[str]) -> None:
        """Record which metric names contributed to training (for the UI)."""
        self._metrics_seen = sorted(set(self._metrics_seen) | {str(name) for name in names})

    # ── scoring ────────────────────────────────────────────────────────
    def score_batch(self, metrics: Sequence[Metric]) -> list[Metric]:
        """Annotate ``metrics`` with anomaly scores; returns annotated copies."""
        self.require_fitted()
        if not metrics:
            return []
        return [outcome.metric for outcome in self.score_with_outcomes(metrics)]

    def score_with_outcomes(self, metrics: Sequence[Metric]) -> list[DetectionOutcome]:
        """Score a batch and return full outcomes (used by tests and the API)."""
        self.require_fitted()
        if not metrics:
            return []
        assert self._model is not None and self._scaler is not None

        raw = self.feature_engineer.transform_batch(metrics)
        scaled = self._scaler.transform(raw)
        decisions = self._model.decision_function(scaled)
        predicted = self._model.predict(scaled)

        confidences = np.asarray(
            [self.confidence_for_decision(float(decision)) for decision in decisions],
            dtype=np.float64,
        )
        coverage = self._coverage(raw)

        self.drift_monitor.observe(scaled)

        outcomes: list[DetectionOutcome] = []
        for metric, decision, flag, confidence, cov in zip(
            metrics, decisions, predicted, confidences, coverage, strict=True
        ):
            warm = bool(cov >= 1.0)
            damped = float(confidence) * (0.35 + 0.65 * float(cov))
            is_anomaly = bool(flag == -1) and warm
            annotated = metric.with_detection(
                anomaly_score=float(-decision),
                is_anomaly=is_anomaly,
                confidence=round(damped, 6),
                model_version=self.version,
            )
            outcomes.append(
                DetectionOutcome(
                    metric=annotated,
                    anomaly_score=annotated.anomaly_score or 0.0,
                    confidence=annotated.confidence,
                    is_anomaly=is_anomaly,
                )
            )

        self._running.batches += 1
        for outcome in outcomes:
            self._running.scored += 1
            self._running.anomalies += int(outcome.is_anomaly)
            self._running.confidence_sum += outcome.confidence
            self._running.score_sum += outcome.anomaly_score

        return outcomes

    # ── evaluation (only meaningful with labels) ───────────────────────
    def confidence_for_decision(self, decision: float) -> float:
        """Map a raw ``decision_function`` value to a confidence in (0, 1].

        Linear interpolation between the anchors derived from the training
        decision distribution: the contamination percentile scores 0.5.
        """
        anchors = self._confidence_anchors
        if not anchors:
            return 0.5
        if decision >= anchors[0][0]:
            return anchors[0][1]
        if decision <= anchors[-1][0]:
            return anchors[-1][1]
        for (high_decision, high_confidence), (low_decision, low_confidence) in pairwise(anchors):
            if low_decision <= decision <= high_decision:
                span = high_decision - low_decision
                ratio = 0.0 if span <= _STD_FLOOR else (high_decision - decision) / span
                return high_confidence + (low_confidence - high_confidence) * ratio
        return anchors[0][1]  # pragma: no cover - defensive

    @staticmethod
    def _anchors_from(decisions: np.ndarray) -> tuple[tuple[float, float], ...]:
        """Confidence anchors ordered by *descending* decision value.

        ``anchors[0]`` is the busiest part of the training distribution (lowest
        confidence) and ``anchors[-1]`` is its most anomalous tail.
        """
        anchors: list[tuple[float, float]] = []
        for percentile, confidence in CONFIDENCE_PERCENTILES:
            value = float(np.min(decisions)) if percentile <= 0 else float(np.percentile(decisions, percentile))
            anchors.append((value, confidence))
        anchors.sort(key=lambda item: item[0], reverse=True)
        return tuple(anchors)

    @staticmethod
    def _anchors_from_bundle(raw: Any) -> tuple[tuple[float, float], ...]:
        if not raw:
            return ()
        pairs = [(float(item[0]), float(item[1])) for item in raw]
        return tuple(sorted(pairs, key=lambda item: item[0], reverse=True))

    def evaluate(self, y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
        """Precision / recall / F1 for a labelled batch."""
        from sklearn.metrics import f1_score, precision_score, recall_score

        return {
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
            "support": len(y_true),
        }

    def score_features(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw decisions and predictions for a feature matrix (no metrics)."""
        self.require_fitted()
        assert self._model is not None and self._scaler is not None
        scaled = self._scaler.transform(matrix)
        return self._model.decision_function(scaled), self._model.predict(scaled)

    # ── persistence ────────────────────────────────────────────────────
    def save(self, path: str | Path) -> Path:
        self.require_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "format_version": MODEL_FORMAT_VERSION,
            "model": self._model,
            "scaler": self._scaler,
            "feature_names": list(self.feature_engineer.feature_names),
            "bands": dict(self.feature_engineer.bands),
            "min_history": self.min_history,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "decision_mean": self._decision_mean,
            "decision_std": self._decision_std,
            "confidence_anchors": [list(anchor) for anchor in self._confidence_anchors],
            "trained_at": self._trained_at,
            "training_samples": self._n_samples,
            "training_metrics": list(self._metrics_seen),
            "feature_summary": self._feature_summary,
        }
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - joblib ships with scikit-learn
            raise RuntimeError("joblib is required to persist models") from exc
        joblib.dump(bundle, target, compress=3)
        logger.info("detector saved to %s", target)
        return target

    def load(self, path: str | Path) -> AnomalyDetector:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        import joblib

        bundle = joblib.load(source)
        format_version = int(bundle.get("format_version", 1))
        if format_version > MODEL_FORMAT_VERSION:
            raise ValueError(f"model format {format_version} is newer than supported {MODEL_FORMAT_VERSION}")

        self._model = bundle["model"]
        self._scaler = bundle["scaler"]
        self.min_history = int(bundle.get("min_history", self.min_history))
        existing = self.feature_engineer.bands if isinstance(self.feature_engineer, FeatureEngineer) else {}
        self.feature_engineer = FeatureEngineer(
            min_samples=self.min_history,
            bands={str(k): tuple(v) for k, v in (bundle.get("bands") or existing).items()},  # type: ignore[misc]
        )
        self.contamination = float(bundle.get("contamination", self.contamination))
        self.n_estimators = int(bundle.get("n_estimators", self.n_estimators))
        self.random_state = int(bundle.get("random_state", self.random_state))
        self._decision_mean = float(bundle.get("decision_mean", 0.0))
        self._decision_std = float(max(bundle.get("decision_std", 1.0), _STD_FLOOR))
        self._confidence_anchors = self._anchors_from_bundle(bundle.get("confidence_anchors"))
        trained_at = bundle.get("trained_at")
        self._trained_at = trained_at if isinstance(trained_at, datetime) else utc_now()
        self._n_samples = int(bundle.get("training_samples", 0))
        self._metrics_seen = list(bundle.get("training_metrics") or [])
        self._feature_summary = dict(bundle.get("feature_summary") or {})
        self._running = _RunningStats()
        logger.info("detector loaded from %s (%d training samples)", source, self._n_samples)
        return self

    def reset_runtime_stats(self) -> None:
        self._running = _RunningStats()
        self.drift_monitor.reset()

    # ── internals ──────────────────────────────────────────────────────
    def _as_matrix(self, samples: Sequence[Metric] | Iterable[Metric] | np.ndarray) -> np.ndarray:
        if isinstance(samples, np.ndarray):
            return np.asarray(samples, dtype=np.float64)
        metrics = list(samples)
        if metrics and isinstance(metrics[0], Metric):
            self.observe_metric_names(metric.name for metric in metrics)
            return self.feature_engineer.transform_batch(metrics)
        raise TypeError("samples must be a sequence of Metric or a numpy feature matrix")
