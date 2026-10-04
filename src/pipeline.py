"""Pipeline orchestration.

This is the only place that knows the order of operations:

    collect -> score (ML) -> rules -> root cause -> triage -> dedup ->
    register -> notify -> auto-resolve -> dashboard

``main.py`` is a thin CLI around :class:`Pipeline`, the control API drives the
same object, and the tests exercise ``tick()`` directly, so the wiring is
verified by tests rather than by staring at a terminal.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import SETTINGS, Settings, get_settings
from src.core.clock import Clock, SystemClock
from src.core.enums import Severity
from src.dashboard.live_dashboard import LiveDashboard
from src.detection.anomaly_detector import AnomalyDetector
from src.detection.rule_engine import RuleEngine
from src.detection.trainer import build_training_set
from src.ingestion import SeriesHistory, collectors_for_clouds
from src.ingestion.metric_schema import Metric
from src.notifications.notifier import NotificationRouter
from src.triage.incident_manager import Incident, IncidentManager
from src.triage.root_cause import generate_hints
from src.triage.triage_engine import TriageEngine, TriageSettings

__all__ = ["Pipeline", "PipelineConfig", "TickResult"]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineConfig:
    """Runtime switches for one pipeline run."""

    settings: Settings = field(default_factory=get_settings)
    seed: int | None = None
    retrain: bool = False
    dashboard: bool = True
    inject_anomaly: bool = False
    inject_every_ticks: int = 6
    max_ticks: int | None = None
    auto_retrain: bool = True
    min_ticks_between_retrains: int = 60
    console_notifications: bool = True
    api_enabled: bool = False
    api_host: str | None = None
    api_port: int | None = None
    api_token: str | None = None
    persistence: bool | None = None
    database_path: str | None = None
    restore: bool = True
    save_model: bool = True


@dataclass(slots=True)
class TickResult:
    """Everything one collection cycle produced."""

    metrics: list[Metric] = field(default_factory=list)
    anomalies: int = 0
    violations: int = 0
    incidents: list[Incident] = field(default_factory=list)
    auto_resolved: list[Incident] = field(default_factory=list)
    retrained: bool = False

    @property
    def incident_count(self) -> int:
        return len(self.incidents)


class Pipeline:
    """Wires every component together and drives the main loop."""

    def __init__(self, config: PipelineConfig | None = None, clock: Clock | None = None) -> None:
        self.config = config or PipelineConfig()
        self.settings: Settings = self.config.settings
        self.clock: Clock = clock or SystemClock()
        self.running = True
        self.ticks = 0
        self.retrain_count = 0
        self.store: Any = None
        self.restored_incidents: list[Incident] = []

        # rolling window shared by collectors, detector and rule engine
        self.history = SeriesHistory(
            window=self.settings.history_window_size,
            min_samples=self.settings.min_history_samples,
        )

        collectors_kwargs: dict[str, Any] = {
            "seed": self.config.seed if self.config.seed is not None else self.settings.random_seed,
            "history": self.history,
        }
        self.collectors = collectors_for_clouds(inject_anomaly=False, **collectors_kwargs)
        self.anomaly_collectors = collectors_for_clouds(inject_anomaly=True, **collectors_kwargs)

        self.detector = AnomalyDetector(
            contamination=self.settings.model_contamination,
            n_estimators=self.settings.model_n_estimators,
            random_state=self.settings.random_seed,
            min_history=self.settings.min_history_samples,
            drift_threshold=self.settings.drift_threshold,
            drift_min_batches=self.settings.drift_min_batches,
        )
        self.rule_engine = RuleEngine(self.settings.profiles)

        # durable incident state, restored before the first tick so dedup survives
        self.store = self._build_store()
        self.incident_manager = IncidentManager(clock=self.clock, store=self.store)
        self.restored_incidents: list[Incident] = []
        if self.store is not None and self.config.restore:
            try:
                self.restored_incidents = self.incident_manager.restore()
            except Exception as exc:
                logger.warning("could not restore incidents from %s: %s", self.store.path, exc)
        if self.restored_incidents:
            logger.info("restored %d open incident(s) from %s", len(self.restored_incidents), self.store.path)

        self.triage_engine = TriageEngine(
            triage_settings=TriageSettings.from_settings(self.settings),
            clock=self.clock,
        ).bind(self.incident_manager)
        self.incident_manager.add_listener(self.triage_engine.on_incident_event)
        self.router = NotificationRouter(
            quiet_mode=self.settings.quiet_mode,
            settings=self.settings,
            clock=self.clock,
            console=self.config.console_notifications,
        )
        self.dashboard = LiveDashboard(self.incident_manager, self.detector, clock=self.clock)
        self.dashboard.attach_notifiers(self.router.notifiers)
        self.model_loaded_from: str | None = None
        self.training_summary: dict[str, Any] | None = None

    # ── lifecycle ──────────────────────────────────────────────────────
    def prepare(self) -> Pipeline:
        """Load or train the model and warm the rolling window."""
        model_path = Path(self.settings.model_path)
        if self.config.retrain or not model_path.is_file():
            self.train()
        else:
            self.detector.load(model_path)
            self.model_loaded_from = str(model_path)
            logger.info("model loaded from %s", model_path)
            self._warm_history(rounds=max(self.settings.min_history_samples, 5))
        return self

    def train(self, *, save: bool | None = None) -> dict[str, Any]:
        """(Re)train the detector on freshly generated baseline telemetry.

        The bundle is written to ``MODEL_PATH`` by default, so a first run (or a
        drift-triggered retrain) leaves a model for the next start instead of
        paying for training every time. Pass ``save=False`` to leave the
        existing artefact untouched.
        """
        rounds = self.settings.baseline_samples
        samples, _ = build_training_set(
            rounds=rounds,
            seed=self.settings.random_seed if self.config.seed is None else self.config.seed,
            settings=self.settings,
            history=self.history,
        )
        summary = self.detector.train(samples)
        self.training_summary = summary
        self.retrain_count += 1
        should_save = self.config.save_model if save is None else bool(save)
        if should_save:
            self.detector.save(self.settings.model_path)
            self.model_loaded_from = None
        logger.info("model trained on %d samples (retrain #%d)", summary["training_samples"], self.retrain_count)
        return summary

    def _warm_history(self, rounds: int) -> None:
        """Fill the rolling window with clean samples before serving."""
        for _ in range(max(0, rounds)):
            for collector in self.collectors:
                collector.collect_and_track()

    # ── main loop ──────────────────────────────────────────────────────
    def tick(self, use_anomaly: bool = False) -> TickResult:
        """Run one full collection -> notification cycle."""
        collectors = self.anomaly_collectors if use_anomaly else self.collectors
        collected: list[Metric] = []
        for collector in collectors:
            collected.extend(collector.collect_and_track())

        scored = self.detector.score_batch(collected)
        result = TickResult(
            metrics=scored,
            anomalies=sum(1 for metric in scored if metric.is_anomaly),
        )

        for metric in scored:
            violations = self.rule_engine.evaluate(metric)
            result.violations += len(violations)
            if not violations and metric.confidence < self.settings.ml_min_confidence:
                continue
            incident = self.triage_engine.triage(
                metric,
                violations,
                generate_hints(metric, violations, scored),
                context=scored,
            )
            if incident is None:
                continue
            is_new = self.incident_manager.get(incident.id) is None
            if is_new:
                self.incident_manager.add(incident)
            self.router.notify(incident)
            result.incidents.append(incident)

        result.auto_resolved = self.incident_manager.auto_resolve_old(
            max_age_minutes=self.settings.auto_resolve_minutes,
            severities=self._auto_resolve_severities(),
        )
        self.dashboard.update_metrics(scored)
        self.ticks += 1
        if self.config.auto_retrain:
            result.retrained = self._maybe_retrain()
        return result

    def run(self) -> dict[str, Any]:
        """Run until stopped (or ``max_ticks`` reached)."""
        interval = self.settings.collection_interval_seconds
        max_ticks = self.config.max_ticks

        if not self.config.dashboard:
            logger.info("pipeline started (headless), interval=%.1fs", interval)
            while self.running and (max_ticks is None or self.ticks < max_ticks):
                self._tick_with_injection()
                self._sleep(interval)
            return self.summary()

        from rich.live import Live

        logger.info("pipeline started with live dashboard, interval=%.1fs", interval)
        with Live(self.dashboard.get_renderable(), refresh_per_second=2, screen=True) as live:
            while self.running and (max_ticks is None or self.ticks < max_ticks):
                self._tick_with_injection(live=live)
                self._sleep(interval, live=live)
        return self.summary()

    def _tick_with_injection(self, live: Any | None = None) -> TickResult:
        use_anomaly = self.config.inject_anomaly and self.ticks % max(1, self.config.inject_every_ticks) == 0
        result = self.tick(use_anomaly=use_anomaly)
        if live is not None:
            live.update(self.dashboard.get_renderable())
        if result.incidents:
            logger.info(
                "tick %d: %d new/escalated incident(s), %d anomalies, %d rule breaches",
                self.ticks,
                result.incident_count,
                result.anomalies,
                result.violations,
            )
        return result

    def _sleep(self, seconds: float, live: Any | None = None) -> None:
        """Interruptible sleep so Ctrl-C / SIGTERM stops promptly."""
        deadline = self.clock.now().timestamp() + seconds
        while self.running and self.clock.now().timestamp() < deadline:
            self.clock.sleep(min(0.25, seconds))
            if live is not None:
                live.update(self.dashboard.get_renderable())

    def stop(self) -> None:
        self.running = False

    # ── external injection (control API / tests) ───────────────────────
    def inject(self, metrics: Sequence[Metric]) -> list[Incident]:
        """Score and triage externally supplied metrics through the full path."""
        if not metrics:
            return []
        scored = self.detector.score_batch(list(metrics))
        created: list[Incident] = []
        for metric in scored:
            violations = self.rule_engine.evaluate(metric)
            incident = self.triage_engine.triage(
                metric,
                violations,
                generate_hints(metric, violations, scored),
                context=scored,
            )
            if incident is None:
                continue
            if self.incident_manager.get(incident.id) is None:
                self.incident_manager.add(incident)
            self.router.notify(incident)
            created.append(incident)
        self.dashboard.update_metrics(scored)
        return created

    def build_sample(
        self,
        cloud: str,
        metric_name: str,
        value: float,
        *,
        resource_id: str | None = None,
        service: str | None = None,
        region: str | None = None,
        confidence: float = 1.0,
    ) -> Metric:
        """Build one metric from a cloud/metric/value triple.

        Externally injected samples are treated as operator-asserted anomalies
        (``confidence`` defaults to 1.0) because they did not come from the
        simulated collectors and would otherwise be ignored while the rolling
        window is cold.
        """
        collector = next(
            (item for item in self.collectors + self.anomaly_collectors if item.cloud.value == str(cloud).lower()),
            self.collectors[0],
        )
        metric_spec = collector.specs.get(metric_name)
        resources = collector.resources
        chosen = next((item for item in resources if item.id == resource_id), None) if resource_id else None
        chosen = chosen or (resources[0] if resources else None)
        metric = Metric(
            name=metric_name,
            value=float(value),
            unit=metric_spec.unit if metric_spec else "",
            timestamp=self.clock.now(),
            cloud=collector.cloud,
            resource_id=chosen.id if chosen else str(cloud),
            service=service or (chosen.service if chosen else str(cloud)),
            region=region or (chosen.region if chosen else ""),
            tags=dict(chosen.tags) if chosen else {},
        )
        annotated = metric.with_detection(
            anomaly_score=float(value),
            is_anomaly=True,
            confidence=float(confidence),
        )
        annotated.window = self.history.stats(annotated.series_key)
        return annotated

    def inject_sample(
        self,
        cloud: str,
        metric_name: str,
        value: float,
        **kwargs: Any,
    ) -> list[Incident]:
        """Build one metric from a cloud/metric/value triple and inject it."""
        return self.inject([self.build_sample(cloud, metric_name, value, **kwargs)])

    # ── reporting ──────────────────────────────────────────────────────
    def summary(self) -> dict[str, Any]:
        incident_stats = self.incident_manager.stats
        return {
            "ticks": self.ticks,
            "metrics_processed": self.dashboard.metrics_seen,
            "anomalies_detected": self.dashboard.anomalies_seen,
            "incidents": incident_stats,
            "triage": self.triage_engine.stats,
            "detector": self.detector.stats,
            "notifications": self.router.stats,
            "retrains": self.retrain_count,
            "uptime_seconds": (self.clock.now() - self.dashboard.started_at).total_seconds(),
            "persistence": self.persistence_stats(),
        }

    def persistence_stats(self) -> dict[str, Any]:
        """What survived, where it lives, and how much has been written."""
        return {
            "enabled": self.store is not None,
            "path": str(self.store.path) if self.store else None,
            "restored_on_start": len(self.restored_incidents),
            "open_restored": [
                {"id": incident.id, "severity": incident.severity.name, "occurrences": incident.occurrences}
                for incident in self.restored_incidents
            ],
            "store": self.store.counts() if self.store else None,
        }

    def stats(self) -> dict[str, Any]:
        """Full stats payload for the control API."""
        return {
            "summary": self.summary(),
            "incidents": self.incident_manager.snapshot(),
            "dashboard": self.dashboard.state,
            "collector_ticks": {collector.cloud.value: collector.ticks for collector in self.collectors},
            "anomaly_collector_ticks": {collector.cloud.value: collector.ticks for collector in self.anomaly_collectors},
            "history_series": len(self.history),
            "running": self.running,
        }

    # ── internals ──────────────────────────────────────────────────────
    def _build_store(self):
        """Open the SQLite store when persistence is enabled and reachable."""
        enabled = self.settings.persistence_enabled if self.config.persistence is None else self.config.persistence
        if not enabled:
            logger.info("incident persistence disabled")
            return None
        from src.triage.store import IncidentStore

        store = IncidentStore(self.config.database_path or self.settings.database_path)
        if not store.healthy():
            logger.warning("incident store at %s is unusable - continuing in memory only", store.path)
            return None
        logger.info("incident persistence enabled: %s", store.path)
        return store

    def _auto_resolve_severities(self) -> tuple[Severity, ...]:
        resolved: list[Severity] = []
        for token in self.settings.auto_resolve_severities:
            try:
                resolved.append(Severity.parse(token))
            except ValueError:  # pragma: no cover - defensive
                continue
        return tuple(resolved) or (Severity.MEDIUM, Severity.LOW)

    def _maybe_retrain(self) -> bool:
        if not self.detector.retrain_recommended:
            return False
        if self.retrain_count and self.ticks < self.config.min_ticks_between_retrains * self.retrain_count:
            return False
        logger.warning("drift detected (score=%.3f) - retraining model", self.detector.stats["drift_score"])
        self.train(save=True)
        return True


def default_config(**overrides: Any) -> PipelineConfig:
    """Pipeline config seeded from module level settings (used by ``main.py``)."""
    return PipelineConfig(settings=SETTINGS, **overrides)
