"""Shared fixtures.

The expensive artefacts (a trained detector, a generated baseline window) are
session scoped; everything that mutates state gets a fresh instance. Settings
are replaced per test so nothing writes to the real ``models/`` directory.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest

from config.settings import get_settings
from src.core.clock import ManualClock
from src.core.enums import Cloud, Severity
from src.detection.anomaly_detector import AnomalyDetector
from src.detection.trainer import build_training_set, train_detector
from src.ingestion import SeriesHistory
from src.ingestion.metric_schema import Metric
from src.triage.incident_manager import Incident, IncidentManager
from src.triage.root_cause import Hypothesis
from src.triage.triage_engine import TriageEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_ROUNDS = 25
TRAINING_SEED = 20260903


@pytest.fixture(scope="session")
def base_settings() -> object:
    return get_settings()


@pytest.fixture
def settings(tmp_path: Path, base_settings):
    """Settings with a temp model path, a temp database and fast defaults."""
    return dataclasses.replace(
        base_settings,
        model_path=str(tmp_path / "model.pkl"),
        database_path=str(tmp_path / "incidents.db"),
        baseline_samples=TRAINING_ROUNDS,
        model_n_estimators=20,
        collection_interval_seconds=0.0,
        history_window=20,
        min_history=5,
        dedup_window_seconds=60,
        auto_resolve_minutes=10,
        notifier_cooldown_seconds=0.0,
        log_dir=str(tmp_path / "logs"),
    )


@pytest.fixture(scope="session")
def training_data():
    samples, history = build_training_set(rounds=TRAINING_ROUNDS, seed=TRAINING_SEED)
    return samples, history


@pytest.fixture(scope="session")
def detector(training_data) -> AnomalyDetector:
    samples, _ = training_data
    return train_detector(samples, n_estimators=20, random_state=TRAINING_SEED)


@pytest.fixture
def fresh_detector(training_data) -> AnomalyDetector:
    """Detector with its own runtime counters (for cumulative-stat assertions)."""
    samples, _ = training_data
    return train_detector(samples, n_estimators=20, random_state=TRAINING_SEED)


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def manager(clock: ManualClock) -> IncidentManager:
    return IncidentManager(clock=clock)


@pytest.fixture
def triage_engine(clock: ManualClock) -> TriageEngine:
    return TriageEngine(dedup_window_seconds=60, clock=clock)


@pytest.fixture
def history(settings) -> SeriesHistory:
    return SeriesHistory(
        window=settings.history_window_size,
        min_samples=settings.min_history_samples,
    )


@pytest.fixture
def profiles(settings) -> dict:
    return settings.profiles


@pytest.fixture
def warm_metric(clock) -> Metric:
    """A realistic CPU series (cv ~0.3) with one sample far out of band.

    The window matches the shape the simulator produces, so the feature vector
    looks like a genuine degradation rather than a synthetic outlier.
    """
    from src.core.stats import stats_from_values

    window_values = [34.0, 41.0, 29.0, 47.0, 38.0, 52.0, 44.0, 31.0, 49.0, 43.0]
    return Metric(
        name="cpu_utilization",
        value=97.5,
        unit="%",
        timestamp=clock.now(),
        cloud=Cloud.AWS,
        resource_id="i-0a1f9c4d2e7b8a31",
        service="EC2",
        region="us-east-1",
        window=stats_from_values(window_values),
    )


@pytest.fixture
def cold_metric(clock) -> Metric:
    return Metric(
        name="cpu_utilization",
        value=97.5,
        unit="%",
        timestamp=clock.now(),
        cloud=Cloud.AWS,
        resource_id="i-0a1f9c4d2e7b8a31",
        service="EC2",
        region="us-east-1",
    )


@pytest.fixture
def incident(clock) -> Incident:
    return Incident(
        id="INC-00001",
        fingerprint="abc123abc123",
        title="AWS EC2 cpu utilization anomaly - CPU saturation",
        cloud=Cloud.AWS,
        service="EC2",
        resource_id="i-0a1f9c4d2e7b8a31",
        region="us-east-1",
        metric_name="cpu_utilization",
        metric_value=97.5,
        unit="%",
        severity=Severity.CRITICAL,
        score=9.5,
        confidence=0.95,
        model_confidence=0.8,
        description="cpu_utilization=97.50% on i-0a1f9c4d2e7b8a31",
        hints=["CPU saturation (60%): +9.0 sigma from the 10-sample mean of 40.05"],
        causes=[Hypothesis(cause="CPU saturation", category="capacity", likelihood=0.6, evidence=("+9.0 sigma",))],
        sla_minutes=15.0,
        estimated_resolution_minutes=12.4,
        sla_deadline=clock.now(),
        created_at=clock.now(),
        last_seen_at=clock.now(),
    )


@pytest.fixture(autouse=True)
def _quiet_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    logging.getLogger("src.notifications.http_client").setLevel(logging.CRITICAL)
