"""Training helpers.

Kept out of ``AnomalyDetector`` so the detector stays focused on inference and
the training recipe (which collectors, how many rounds, which window) can be
reused by the CLI, the pipeline and the test-suite unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from config.settings import Settings, get_settings
from src.core.enums import Cloud
from src.detection.anomaly_detector import AnomalyDetector
from src.ingestion import collectors_for_clouds
from src.ingestion.history import SeriesHistory
from src.ingestion.metric_schema import Metric

__all__ = [
    "build_training_set",
    "evaluate_on_labels",
    "load_dataset",
    "save_dataset",
    "train_detector",
]


def build_training_set(
    rounds: int = 500,
    clouds: Sequence[Cloud] | None = None,
    *,
    seed: int = 42,
    inject_anomaly: bool = False,
    settings: Settings | None = None,
    history: SeriesHistory | None = None,
) -> tuple[list[Metric], SeriesHistory]:
    """Generate ``rounds`` ticks of telemetry and return it with its window state.

    ``inject_anomaly=True`` yields a training set that contains anomalies, which
    is what makes it possible to measure precision/recall later.
    """
    resolved = settings or get_settings()
    # an empty SeriesHistory is falsy (len == 0) so this must be an identity check
    series_history = history if history is not None else SeriesHistory(
        window=resolved.history_window_size,
        min_samples=resolved.min_history_samples,
    )
    collectors = collectors_for_clouds(
        clouds,
        inject_anomaly=inject_anomaly,
        seed=seed,
        history=series_history,
    )
    samples: list[Metric] = []
    for _ in range(max(1, rounds)):
        for collector in collectors:
            samples.extend(collector.collect_and_track())
    return samples, series_history


def train_detector(
    samples: Sequence[Metric],
    *,
    settings: Settings | None = None,
    contamination: float | None = None,
    n_estimators: int | None = None,
    random_state: int | None = None,
    detector: AnomalyDetector | None = None,
) -> AnomalyDetector:
    """Fit (or refit) a detector on ``samples``."""
    resolved = settings or get_settings()
    engine = detector or AnomalyDetector(
        contamination=resolved.model_contamination if contamination is None else contamination,
        n_estimators=resolved.model_n_estimators if n_estimators is None else n_estimators,
        random_state=resolved.random_seed if random_state is None else random_state,
        min_history=resolved.min_history_samples,
        drift_threshold=resolved.drift_threshold,
        drift_min_batches=resolved.drift_min_batches,
    )
    engine.train(samples)
    return engine


def evaluate_on_labels(
    detector: AnomalyDetector,
    samples: Sequence[Metric],
    labels: Sequence[int],
) -> dict[str, float]:
    """Score ``samples`` and compare against ``labels`` (0 = normal, 1 = anomaly)."""
    if len(samples) != len(labels):
        raise ValueError("samples and labels must have the same length")
    matrix = detector.feature_engineer.transform_batch(samples)
    decisions, predicted = detector.score_features(matrix)
    y_pred = [int(flag == -1) for flag in predicted]
    metrics = detector.evaluate([int(label) for label in labels], y_pred)
    metrics["mean_decision"] = round(float(decisions.mean()), 6)
    metrics["predicted_positive"] = sum(y_pred)
    metrics["actual_positive"] = sum(int(label) for label in labels)
    return metrics


def save_dataset(samples: Sequence[Metric], path: str | Path, *, limit: int | None = None) -> Path:
    """Persist metrics as compact JSON so training runs can be reproduced."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(samples)[:limit] if limit else list(samples)
    payload = {
        "version": 1,
        "generated_by": "src.detection.trainer.save_dataset",
        "count": len(rows),
        "metrics": [metric.to_dict() for metric in rows],
    }
    target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return target


def load_dataset(path: str | Path) -> list[Metric]:
    """Load a dataset written by :func:`save_dataset`.

    Accepts either ``{"metrics": [...]}`` or a bare JSON list.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload: Any = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("metrics", []) if isinstance(payload, dict) else payload
    return [Metric.from_dict(row) for row in rows]
