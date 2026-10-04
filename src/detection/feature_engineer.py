"""Feature engineering for the anomaly model.

Two properties matter here.

**Train/serve parity.** Features are computed by this one function from the
rolling statistics the collectors already publish, so building the training
matrix and scoring live traffic use identical code and identical inputs.

**Unit-free features.** The pipeline mixes percent, milliseconds, rps and Mbps
in one model, so raw ``value``/``mean``/``std`` columns are meaningless after a
single global scaler (a 97% CPU reading would sit *below* a 1100 rps request
rate). Every feature below is therefore dimensionless:

==================  =====================================================
``z_score``         deviation from the rolling mean, in sigma
``pct_change``      change versus the previous sample, in percent
``value_over_mean`` current value relative to its own rolling mean
``coefficient_of_variation``  rolling std / mean
``delta_over_std``  step size relative to the rolling volatility
``band_position``   0..1 inside the metric's expected band, >1 above it
``history_coverage``  fraction of the minimum required window available
==================  =====================================================

``history_coverage`` lets the model see cold-start samples instead of silently
scoring them as if they were steady state, and
:class:`~src.detection.anomaly_detector.AnomalyDetector` refuses to flag them.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from src.ingestion.metric_schema import Metric

__all__ = ["FEATURE_NAMES", "DriftMonitor", "FeatureEngineer"]

FEATURE_NAMES: tuple[str, ...] = (
    "z_score",
    "pct_change",
    "value_over_mean",
    "coefficient_of_variation",
    "delta_over_std",
    "band_position",
    "history_coverage",
)

FEATURE_INDEX = {name: position for position, name in enumerate(FEATURE_NAMES)}

_Z_CLIP = 12.0
_PCT_CLIP = 500.0
_RATIO_CLIP = 10.0
_CV_CLIP = 10.0
_BAND_CLIP = 20.0
_EPSILON = 1e-9


class FeatureEngineer:
    """Turns :class:`Metric` objects into a fixed-width dimensionless vector.

    ``feature_names`` selects and orders the features, which is what makes an
    ablation study possible: dropping ``z_score`` is ``feature_names`` without
    it, and nothing else changes.
    """

    def __init__(
        self,
        min_samples: int = 5,
        bands: dict[str, tuple[float, float]] | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.min_samples = max(1, int(min_samples))
        selected = tuple(feature_names) if feature_names is not None else FEATURE_NAMES
        unknown = [name for name in selected if name not in FEATURE_INDEX]
        if unknown:
            raise ValueError(f"unknown feature(s): {unknown}; known: {list(FEATURE_NAMES)}")
        if not selected:
            raise ValueError("at least one feature is required")
        self.feature_names = tuple(selected)
        #: metric name -> (expected_low, expected_high) used for ``band_position``
        self.bands = dict(bands or {})

    @classmethod
    def from_profiles(cls, profiles: dict[str, Any], min_samples: int = 5) -> FeatureEngineer:
        """Build bands from ``cloud_profiles.yaml`` metric definitions."""
        bands: dict[str, tuple[float, float]] = {}
        for name, spec in (profiles.get("metrics") or {}).items():
            normal = (spec or {}).get("normal") or []
            if len(normal) == 2:
                low, high = float(normal[0]), float(normal[1])
                bands[str(name)] = (min(low, high), max(low, high))
        return cls(min_samples=min_samples, bands=bands)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def ablated_features(self) -> tuple[str, ...]:
        """Features dropped relative to the full set."""
        return tuple(name for name in FEATURE_NAMES if name not in self.feature_names)

    def transform(self, metric: Metric) -> np.ndarray:
        """Feature vector for a single metric, in :attr:`feature_names` order."""
        window = metric.window
        if window is None or window.count == 0:
            cold = {
                "z_score": 0.0,
                "pct_change": 0.0,
                "value_over_mean": 1.0,
                "coefficient_of_variation": 0.0,
                "delta_over_std": 0.0,
                "band_position": self._band_position(metric.value, metric.name),
                "history_coverage": 0.0,
            }
            return np.asarray([cold[name] for name in self.feature_names], dtype=np.float64)

        coverage = min(1.0, window.count / self.min_samples)
        std = max(window.std, _EPSILON)
        values = {
            "z_score": float(np.clip(window.z_score(metric.value), -_Z_CLIP, _Z_CLIP)),
            "pct_change": float(np.clip(window.pct_change, -_PCT_CLIP, _PCT_CLIP)),
            "value_over_mean": float(np.clip(window.ratio(metric.value), -_RATIO_CLIP, _RATIO_CLIP)),
            "coefficient_of_variation": float(np.clip(window.coefficient_of_variation, 0.0, _CV_CLIP)),
            "delta_over_std": float(np.clip(window.delta / std, -_RATIO_CLIP, _RATIO_CLIP)),
            "band_position": self._band_position(metric.value, metric.name),
            "history_coverage": coverage,
        }
        return np.asarray([values[name] for name in self.feature_names], dtype=np.float64)

    def _band_position(self, value: float, metric_name: str) -> float:
        band = self.bands.get(metric_name)
        if band is None:
            return 0.0
        low, high = band
        span = high - low
        if span < _EPSILON:
            return 0.0
        return float(np.clip((value - low) / span, -_BAND_CLIP, _BAND_CLIP))

    def transform_batch(self, metrics: Sequence[Metric] | Iterable[Metric]) -> np.ndarray:
        """Feature matrix with shape ``(n_metrics, n_features)``."""
        rows = [self.transform(metric) for metric in metrics]
        if not rows:
            return np.zeros((0, self.n_features), dtype=np.float64)
        return np.vstack(rows)

    def feature_report(self, matrix: np.ndarray) -> dict[str, dict[str, float]]:
        """Per-feature mean/std/min/max, used for logging and the dashboard."""
        if matrix.size == 0:
            return {}
        report: dict[str, dict[str, float]] = {}
        for position, name in enumerate(self.feature_names):
            column = matrix[:, position]
            report[name] = {
                "mean": float(np.mean(column)),
                "std": float(np.std(column)),
                "min": float(np.min(column)),
                "max": float(np.max(column)),
            }
        return report


class DriftMonitor:
    """Population-stability style monitor over per-batch feature means.

    The detector feeds it every scored batch; once enough batches have been seen
    it reports how far the live feature distribution has moved away from the
    training distribution. ``drifted`` is what drives retraining.
    """

    def __init__(self, threshold: float = 0.35, min_batches: int = 10, window: int = 30) -> None:
        self.threshold = float(threshold)
        self.min_batches = max(1, int(min_batches))
        self._window = deque(maxlen=max(2, int(window)))
        self._baseline_mean: np.ndarray | None = None
        self._baseline_std: np.ndarray | None = None
        self._batches = 0
        self._score = 0.0

    @property
    def is_calibrated(self) -> bool:
        return self._baseline_mean is not None

    @property
    def drift_score(self) -> float:
        return self._score

    @property
    def drifted(self) -> bool:
        return self.is_calibrated and self._batches >= self.min_batches and self._score > self.threshold

    @property
    def batches_seen(self) -> int:
        return self._batches

    def fit_baseline(self, matrix: np.ndarray, epsilon: float = 1e-6) -> None:
        """Record the training-time distribution as the reference."""
        if matrix.size == 0:
            return
        self._baseline_mean = np.mean(matrix, axis=0)
        self._baseline_std = np.maximum(np.std(matrix, axis=0), epsilon)
        self._window.clear()
        self._batches = 0
        self._score = 0.0

    def observe(self, matrix: np.ndarray) -> float:
        """Add a batch to the live window and recompute the drift score."""
        if matrix.size == 0 or not self.is_calibrated:
            return self._score
        assert self._baseline_mean is not None and self._baseline_std is not None
        self._window.append(np.mean(matrix, axis=0))
        self._batches += 1
        if self._batches < self.min_batches or len(self._window) < 2:
            return self._score
        recent = np.vstack(self._window)
        shifts = np.abs(np.mean(recent, axis=0) - self._baseline_mean) / self._baseline_std
        self._score = float(np.clip(np.mean(shifts), 0.0, 1.0))
        return self._score

    def reset(self) -> None:
        self._window.clear()
        self._batches = 0
        self._score = 0.0
