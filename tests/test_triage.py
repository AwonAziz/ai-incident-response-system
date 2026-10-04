"""Root-cause analysis, triage scoring/dedup and the incident lifecycle."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.core.enums import Cloud, IncidentStatus, Severity
from src.core.stats import stats_from_values
from src.detection.rule_engine import RuleEngine
from src.ingestion.metric_schema import Metric
from src.triage import (
    Incident,
    IncidentManager,
    RootCause,
    TriageEngine,
    TriageSettings,
    analyze,
    generate_hints,
)
from src.triage.triage_engine import RULE_CONFIDENCE


@pytest.fixture
def engine(triage_engine: TriageEngine) -> TriageEngine:
    return triage_engine


@pytest.fixture
def mild_metric(warm_metric: Metric) -> Metric:
    """Same breaching value but with a *realistic* rolling window (z ~ 3)."""
    return replace(warm_metric, window=stats_from_values([30.0, 45.0, 60.0, 55.0, 40.0, 50.0, 35.0, 65.0]))


def _violation(rule_id: str, severity: Severity, weight: float = 1.0, value: float = 99.0):
    from src.core.enums import Cloud
    from src.detection.rule_engine import RuleViolation

    return RuleViolation(
        rule_id=rule_id,
        metric_name="cpu_utilization",
        cloud=Cloud.AWS,
        resource_id="i-0a1f9c4d2e7b8a31",
        service="EC2",
        severity=severity,
        direction="high",
        threshold=90.0,
        value=value,
        message=f"{rule_id} breached",
        weight=weight,
    )


class TestRootCause:
    def test_single_metric_cause_ranks_first(self, warm_metric: Metric) -> None:
        root = analyze(warm_metric, [_violation("cpu_utilization.critical_high", Severity.CRITICAL)])
        assert isinstance(root, RootCause)
        assert root.primary is not None
        assert root.primary.cause == "CPU saturation"
        assert 0.0 < root.primary.likelihood <= 1.0
        assert sum(h.likelihood for h in root.hypotheses) == pytest.approx(1.0, abs=1e-6)

    def test_correlation_rule_requires_the_metric_to_participate(self, clock) -> None:
        """A latency+error correlation must not be blamed for a disk spike."""
        latency = Metric(name="latency_ms", value=400.0, timestamp=clock.now(), service="RDS")
        error = Metric(name="error_rate", value=5.0, timestamp=clock.now(), service="RDS")
        disk = Metric(name="disk_io_utilization", value=97.0, timestamp=clock.now(), service="RDS")

        latency_root = analyze(latency, context=[latency, error.with_detection(is_anomaly=True)])
        disk_root = analyze(disk, context=[disk, error.with_detection(is_anomaly=True)])
        assert latency_root.primary is not None and latency_root.primary.cause == "Downstream dependency failure"
        assert disk_root.primary is not None
        assert disk_root.primary.cause != "Downstream dependency failure"
        assert disk_root.primary.cause == "Log volume explosion or unbounded write"

    def test_capacity_correlation(self, clock) -> None:
        cpu = Metric(name="cpu_utilization", value=97.0, timestamp=clock.now())
        memory = Metric(name="memory_utilization", value=96.0, timestamp=clock.now())
        root = analyze(cpu, context=[cpu, memory.with_detection(is_anomaly=True)])
        causes = {item.cause for item in root.hypotheses}
        assert "Resource exhaustion" in causes

    def test_threshold_breach_is_evidence(self, warm_metric: Metric) -> None:
        root = analyze(warm_metric, [_violation("cpu_utilization.critical_high", Severity.CRITICAL)])
        assert any("Threshold breach" in item.cause for item in root.hypotheses)

    def test_model_only_detection_is_evidence(self, warm_metric: Metric) -> None:
        scored = replace(warm_metric, confidence=0.9, is_anomaly=True)
        root = analyze(scored)
        assert any(item.category == "anomaly" for item in root.hypotheses)

    def test_unknown_metric_falls_back(self, clock) -> None:
        root = analyze(Metric(name="mystery_metric", value=1.0, timestamp=clock.now()))
        assert root.primary is not None
        assert root.primary.cause == "Unclassified metric anomaly"

    def test_statistics_include_z_score_when_warm(self, warm_metric: Metric, cold_metric: Metric) -> None:
        assert "z_score" in analyze(warm_metric).statistics
        assert "z_score" not in analyze(cold_metric).statistics

    def test_serialisation(self, warm_metric: Metric) -> None:
        payload = analyze(warm_metric, [_violation("cpu_utilization.critical_high", Severity.CRITICAL)]).to_dict()
        assert payload["cloud"] == "aws"
        assert payload["severity"] == "CRITICAL"
        assert payload["hypotheses"][0]["likelihood"] <= 1.0

    def test_generate_hints_are_readable(self, warm_metric: Metric) -> None:
        hints = generate_hints(warm_metric, [_violation("cpu_utilization.critical_high", Severity.CRITICAL)])
        assert 1 <= len(hints) <= 3
        assert all("%" in hint or ")" in hint for hint in hints)

    def test_hypothesis_describe(self, warm_metric: Metric) -> None:
        primary = analyze(warm_metric).primary
        assert primary is not None
        assert primary.cause in primary.describe()


class TestTriageScoring:
    def test_no_violations_and_low_confidence_is_ignored(self, engine: TriageEngine, cold_metric: Metric) -> None:
        assert engine.triage(cold_metric, [], []) is None
        assert engine.stats["below_threshold"] == 1

    def test_ml_only_detection_creates_incident(self, engine: TriageEngine, warm_metric: Metric) -> None:
        scored = replace(warm_metric, confidence=0.95, is_anomaly=True)
        incident = engine.triage(scored, [])
        assert incident is not None
        assert incident.severity in (Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)
        assert "anomaly model only" in incident.description

    def test_critical_rule_is_always_critical(self, engine: TriageEngine, mild_metric: Metric) -> None:
        incident = engine.triage(mild_metric, [_violation("cpu_utilization.critical_high", Severity.CRITICAL)])
        assert incident is not None
        assert incident.severity is Severity.CRITICAL

    def test_rule_severity_is_a_floor_not_a_ceiling(self, engine: TriageEngine, mild_metric: Metric) -> None:
        """A big z-score can escalate beyond the breached threshold."""
        incident = engine.triage(mild_metric, [_violation("cpu_utilization.warn_high", Severity.HIGH)])
        assert incident is not None
        assert incident.severity in (Severity.HIGH, Severity.CRITICAL)

    def test_severity_thresholds(self, engine: TriageEngine) -> None:
        assert engine.classify(9.0) is Severity.CRITICAL
        assert engine.classify(6.0) is Severity.HIGH
        assert engine.classify(3.5) is Severity.MEDIUM
        assert engine.classify(1.0) is Severity.LOW

    def test_score_grows_with_violation_severity(self, engine: TriageEngine, warm_metric: Metric) -> None:
        low = engine.score(warm_metric, [_violation("x", Severity.LOW)])
        critical = engine.score(warm_metric, [_violation("x", Severity.CRITICAL)])
        assert critical > low

    def test_score_includes_z_score_and_blast_radius(self, engine: TriageEngine, warm_metric: Metric) -> None:
        baseline = engine.score(warm_metric, [_violation("x", Severity.MEDIUM)])
        sibling = replace(warm_metric, name="memory_utilization", is_anomaly=True)
        with_blast = engine.score(warm_metric, [_violation("x", Severity.MEDIUM)], context=[warm_metric, sibling])
        assert with_blast > baseline

    def test_fingerprint_is_scope_based_and_stable(self, engine: TriageEngine, warm_metric: Metric) -> None:
        other_instance = replace(warm_metric, resource_id="i-different")
        assert engine.fingerprint(warm_metric) == engine.fingerprint(other_instance)
        assert engine.fingerprint(warm_metric) == TriageEngine.fingerprint(warm_metric)

    def test_fingerprint_differs_across_services(self, engine: TriageEngine, warm_metric: Metric) -> None:
        assert engine.fingerprint(warm_metric) != engine.fingerprint(replace(warm_metric, service="RDS"))


class TestTriageDeduplication:
    def test_repeat_inside_window_is_suppressed(self, engine: TriageEngine, warm_metric: Metric, clock) -> None:
        violations = [_violation("cpu_utilization.critical_high", Severity.CRITICAL)]
        first = engine.triage(warm_metric, violations)
        assert first is not None

        clock.advance(5)
        repeat = replace(warm_metric, value=98.0)
        assert engine.triage(repeat, violations) is None
        assert engine.stats["suppressed"] == 1
        assert first.occurrences == 2
        assert first.last_seen_at > first.created_at

    def test_repeat_after_window_opens_a_new_incident(self, engine: TriageEngine, warm_metric: Metric, clock) -> None:
        violations = [_violation("cpu_utilization.critical_high", Severity.CRITICAL)]
        first = engine.triage(warm_metric, violations)
        clock.advance(120)
        second = engine.triage(warm_metric, violations)
        assert first is not None and second is not None
        assert second.id != first.id
        assert second.fingerprint == first.fingerprint
        assert len(engine.open_fingerprints()) == 1

    def test_severity_escalation_returns_same_incident(self, engine: TriageEngine, mild_metric: Metric) -> None:
        mild = engine.triage(mild_metric, [_violation("cpu_utilization.warn_high", Severity.MEDIUM)])
        assert mild is not None
        escalated = engine.triage(mild_metric, [_violation("cpu_utilization.critical_high", Severity.CRITICAL)])
        assert escalated is mild
        assert escalated.severity is Severity.CRITICAL
        assert escalated.escalations == 1
        assert engine.stats["escalated"] == 1

    def test_dedup_works_without_an_incident_manager(self, engine: TriageEngine, warm_metric: Metric) -> None:
        engine.triage(warm_metric, [_violation("cpu_utilization.critical_high", Severity.CRITICAL)])
        assert engine.triage(warm_metric, [_violation("cpu_utilization.critical_high", Severity.CRITICAL)]) is None

    def test_closed_incident_stops_suppressing(
        self,
        engine: TriageEngine,
        manager: IncidentManager,
        warm_metric: Metric,
        clock,
    ) -> None:
        engine.bind(manager)
        violations = [_violation("cpu_utilization.critical_high", Severity.CRITICAL)]
        incident = engine.triage(warm_metric, violations)
        assert incident is not None
        manager.add(incident)
        manager.resolve(incident.id, "handled")

        clock.advance(1)
        fresh = engine.triage(warm_metric, violations)
        assert fresh is not None
        assert fresh.id != incident.id

    def test_bind_returns_engine_for_chaining(self, engine: TriageEngine, manager: IncidentManager) -> None:
        assert engine.bind(manager) is engine


class TestTriageIncidentContent:
    def test_incident_fields(self, engine: TriageEngine, warm_metric: Metric) -> None:
        incident = engine.triage(warm_metric, [_violation("cpu_utilization.critical_high", Severity.CRITICAL)])
        assert incident is not None
        assert incident.id == "INC-00001"
        assert incident.metric_name == "cpu_utilization"
        assert incident.metric_value == warm_metric.value
        assert incident.sla_minutes == 15.0
        assert incident.sla_deadline is not None
        assert incident.hints
        assert incident.causes
        assert incident.violations[0]["rule_id"] == "cpu_utilization.critical_high"
        assert incident.confidence == pytest.approx(RULE_CONFIDENCE["critical"], abs=0.95)
        assert incident.model_confidence == 0.0

    def test_rule_breach_sets_confidence_above_model(
        self,
        engine: TriageEngine,
        warm_metric: Metric,
    ) -> None:
        scored = replace(warm_metric, confidence=0.3)
        incident = engine.triage(scored, [_violation("cpu_utilization.warn_high", Severity.HIGH)])
        assert incident is not None
        assert incident.confidence == pytest.approx(RULE_CONFIDENCE["high"])
        assert incident.model_confidence == 0.3

    def test_confident_sla_does_not_breach(self, engine: TriageEngine, warm_metric: Metric) -> None:
        incident = engine.triage(warm_metric, [_violation("cpu_utilization.critical_high", Severity.CRITICAL)])
        assert incident is not None
        assert not incident.sla_breached

    def test_wide_blast_radius_breaches_the_sla(self, engine: TriageEngine, mild_metric: Metric) -> None:
        """A high-severity incident with several correlated signals at once is at risk."""
        siblings = [
            replace(mild_metric, name=f"signal_{index}", is_anomaly=True) for index in range(3)
        ]
        violations = [_violation(f"rule_{index}", Severity.MEDIUM) for index in range(3)]
        incident = engine.triage(mild_metric, violations, context=[mild_metric, *siblings])
        assert incident is not None
        assert incident.severity is Severity.CRITICAL
        assert incident.sla_breached
        assert incident.estimated_resolution_minutes and incident.estimated_resolution_minutes > 15.0

    def test_ids_increment(self, engine: TriageEngine, clock, warm_metric: Metric) -> None:
        violations = [_violation("cpu_utilization.critical_high", Severity.CRITICAL)]
        first = engine.triage(replace(warm_metric, service="EC2"), violations)
        second = engine.triage(replace(warm_metric, service="RDS"), violations)
        assert first is not None and second is not None
        assert (first.id, second.id) == ("INC-00001", "INC-00002")

    def test_custom_id_prefix(self, clock, warm_metric: Metric) -> None:
        engine = TriageEngine(clock=clock, id_prefix="OPS")
        incident = engine.triage(warm_metric, [_violation("cpu_utilization.critical_high", Severity.CRITICAL)])
        assert incident is not None
        assert incident.id.startswith("OPS-")

    def test_summary_line_is_human_readable(self, engine: TriageEngine, warm_metric: Metric) -> None:
        incident = engine.triage(warm_metric, [_violation("cpu_utilization.critical_high", Severity.CRITICAL)])
        assert incident is not None
        assert incident.summary_line().startswith("INC-00001 [CRITICAL]")

    def test_serialisation_is_json_safe(self, engine: TriageEngine, warm_metric: Metric) -> None:
        import json

        incident = engine.triage(warm_metric, [_violation("cpu_utilization.critical_high", Severity.CRITICAL)])
        payload = json.loads(json.dumps(incident.to_dict(), default=str))
        assert payload["severity"] == "CRITICAL"
        assert payload["cloud"] == "aws"
        assert payload["created_at"].startswith("20")

    def test_age_and_sla_countdown(self, engine: TriageEngine, warm_metric: Metric, clock) -> None:
        incident = engine.triage(warm_metric, [_violation("cpu_utilization.critical_high", Severity.CRITICAL)])
        assert incident is not None
        assert incident.age_seconds(clock.now()) == 0.0
        assert incident.time_to_breach_seconds(clock.now()) == pytest.approx(900.0)


class TestIncidentManager:
    def _incident(self, **overrides) -> Incident:
        base = {
            "id": "INC-00001",
            "fingerprint": "fp1",
            "title": "t",
            "cloud": None,
            "service": "EC2",
            "resource_id": "i-1",
            "severity": Severity.HIGH,
        }
        base.update(overrides)
        base["cloud"] = overrides.get("cloud") or Cloud.AWS
        return Incident(**base)

    def test_add_and_get(self, manager: IncidentManager) -> None:
        incident = self._incident()
        assert manager.add(incident) is incident
        assert manager.get("INC-00001") is incident
        assert manager.get("missing") is None
        assert len(manager) == 1

    def test_duplicate_ids_are_rejected(self, manager: IncidentManager) -> None:
        manager.add(self._incident())
        with pytest.raises(ValueError, match="already exists"):
            manager.add(self._incident())

    def test_active_sorted_by_severity(self, manager: IncidentManager) -> None:
        manager.add(self._incident(id="A", fingerprint="a", severity=Severity.LOW))
        manager.add(self._incident(id="B", fingerprint="b", severity=Severity.CRITICAL))
        manager.add(self._incident(id="C", fingerprint="c", severity=Severity.HIGH))
        assert [item.id for item in manager.active()] == ["B", "C", "A"]
        assert manager.counts_by_severity()["CRITICAL"] == 1

    def test_acknowledge_and_resolve(self, manager: IncidentManager, clock) -> None:
        incident = manager.add(self._incident())
        manager.acknowledge(incident.id, "looking into it")
        assert incident.status is IncidentStatus.ACKNOWLEDGED
        assert incident.acknowledged_at is not None
        manager.resolve(incident.id, "fixed")
        assert incident.status is IncidentStatus.RESOLVED
        assert incident.resolution == "fixed"
        assert manager.active() == []

    def test_unknown_incident_raises(self, manager: IncidentManager) -> None:
        with pytest.raises(KeyError):
            manager.resolve("nope")

    def test_acknowledging_a_resolved_incident_raises(self, manager: IncidentManager) -> None:
        incident = manager.add(self._incident())
        manager.resolve(incident.id)
        with pytest.raises(ValueError, match="already resolved"):
            manager.acknowledge(incident.id)

    def test_resolving_twice_is_idempotent(self, manager: IncidentManager) -> None:
        incident = manager.add(self._incident())
        manager.resolve(incident.id)
        manager.resolve(incident.id)
        assert manager.stats["total_resolved"] == 1

    def test_auto_resolve_skips_critical_and_high(self, manager: IncidentManager, clock) -> None:
        old = clock.now()
        manager.add(self._incident(id="A", fingerprint="a", severity=Severity.CRITICAL, created_at=old))
        manager.add(self._incident(id="B", fingerprint="b", severity=Severity.HIGH, created_at=old))
        clock.advance(60 * 30)
        resolved = manager.auto_resolve_old(max_age_minutes=10)
        assert resolved == []
        assert len(manager.active()) == 2

    def test_auto_resolve_closes_stale_low_severity(self, manager: IncidentManager, clock) -> None:
        stale = self._incident(id="A", fingerprint="a", severity=Severity.LOW, created_at=clock.now())
        fresh = self._incident(id="B", fingerprint="b", severity=Severity.LOW)
        manager.add(stale)
        manager.add(fresh)
        clock.advance(60 * 30)
        manager.touch(fresh, clock.now())  # still being observed
        resolved = manager.auto_resolve_old(max_age_minutes=10)
        assert [item.id for item in resolved] == ["A"]
        assert resolved[0].resolution.startswith("auto-resolved")
        assert manager.stats["total_auto_resolved"] == 1
        assert [item.id for item in manager.active()] == ["B"]

    def test_auto_resolve_uses_last_seen_not_created_at(self, manager: IncidentManager, clock) -> None:
        incident = self._incident(id="A", fingerprint="a", severity=Severity.MEDIUM)
        manager.add(incident)
        clock.advance(60 * 5)
        manager.touch(incident, clock.now())
        clock.advance(60 * 5)
        assert manager.auto_resolve_old(max_age_minutes=10) == []

    def test_find_by_fingerprint_only_matches_open(self, manager: IncidentManager) -> None:
        incident = manager.add(self._incident())
        assert manager.find_by_fingerprint("fp1") is incident
        manager.resolve(incident.id)
        assert manager.find_by_fingerprint("fp1") is None

    def test_escalate_raises_severity_and_score(self, manager: IncidentManager) -> None:
        incident = manager.add(self._incident(severity=Severity.MEDIUM))
        manager.escalate(incident, Severity.CRITICAL)
        assert incident.severity is Severity.CRITICAL
        assert incident.escalations == 1
        assert manager.stats["total_escalated"] == 1

    def test_escalate_to_lower_severity_is_ignored(self, manager: IncidentManager) -> None:
        incident = manager.add(self._incident(severity=Severity.CRITICAL))
        manager.escalate(incident, Severity.LOW)
        assert incident.severity is Severity.CRITICAL
        assert incident.escalations == 0

    def test_listeners_receive_events(self, manager: IncidentManager) -> None:
        seen: list[tuple[str, str]] = []
        manager.add_listener(lambda kind, item: seen.append((kind, item.id)))
        incident = manager.add(self._incident())
        manager.acknowledge(incident.id)
        manager.resolve(incident.id)
        assert [kind for kind, _ in seen] == ["created", "acknowledged", "resolved"]

    def test_broken_listener_does_not_break_lifecycle(self, manager: IncidentManager) -> None:
        def explode(kind: str, item: object) -> None:
            raise RuntimeError("listener bug")

        manager.add_listener(explode)
        incident = manager.add(self._incident())
        assert manager.resolve(incident.id).status is IncidentStatus.RESOLVED

    def test_listener_removal(self, manager: IncidentManager) -> None:
        seen: list[str] = []
        listener = lambda kind, item: seen.append(kind)  # noqa: E731
        manager.add_listener(listener)
        manager.remove_listener(listener)
        manager.add(self._incident())
        assert seen == []

    def test_event_feed(self, manager: IncidentManager) -> None:
        incident = manager.add(self._incident())
        manager.resolve(incident.id, "done")
        events = manager.events(10)
        assert [event.kind for event in events] == ["resolved", "created"]
        assert events[0].to_dict()["incident_id"] == incident.id

    def test_history_is_trimmed_but_open_incidents_survive(self) -> None:
        manager = IncidentManager(max_history=3)
        for index in range(3):
            manager.add(self._incident(id=f"I{index}", fingerprint=f"fp{index}"))
        assert len(manager) == 3
        # adding a fourth incident trims the oldest *closed* one
        manager.resolve("I0", "closed")
        manager.add(self._incident(id="I3", fingerprint="fp3"))
        assert manager.get("I0") is None
        assert len(manager) == 3
        assert all(manager.get(f"I{index}") is not None for index in (1, 2, 3))

    def test_open_incidents_are_never_trimmed(self) -> None:
        manager = IncidentManager(max_history=2)
        for index in range(4):
            manager.add(self._incident(id=f"I{index}", fingerprint=f"fp{index}"))
        assert len(manager) == 4

    def test_stats_and_snapshot(self, manager: IncidentManager) -> None:
        manager.add(self._incident(id="A", fingerprint="a", service="EC2"))
        manager.add(self._incident(id="B", fingerprint="b", service="RDS"))
        stats = manager.stats
        assert stats["total_created"] == 2
        assert stats["active_count"] == 2
        assert stats["open_by_service"] == {"EC2": 1, "RDS": 1}
        assert stats["open_by_cloud"] == {"aws": 2}

        snapshot = manager.snapshot(limit=1)
        assert len(snapshot["active"]) == 1
        assert snapshot["stats"]["total_created"] == 2
        assert snapshot["events"]

    def test_recent_is_newest_first(self, manager: IncidentManager) -> None:
        for index in range(3):
            manager.add(self._incident(id=f"I{index}", fingerprint=f"fp{index}"))
        assert [item.id for item in manager.recent(2)] == ["I2", "I1"]

    def test_iteration(self, manager: IncidentManager) -> None:
        manager.add(self._incident())
        assert len(list(manager)) == 1


class TestTriageSettings:
    def test_from_settings(self, settings) -> None:
        triage = TriageSettings.from_settings(settings)
        assert triage.dedup_window_seconds == settings.dedup_window_seconds
        assert triage.severity_factors["critical"] == 3.0
        assert triage.sla_targets_minutes["high"] == 60.0

    def test_dedup_window_comes_from_constructor(self, clock) -> None:
        engine = TriageEngine(dedup_window_seconds=5, clock=clock)
        assert engine.config.dedup_window_seconds == 5
        assert engine.stats["dedup_window_seconds"] == 5

    def test_rule_engine_integration_end_to_end(self, warm_metric: Metric, engine: TriageEngine) -> None:
        rules = RuleEngine()
        metric = replace(warm_metric, value=99.0)
        violations = rules.evaluate(metric)
        assert violations, "expected threshold violations for a 99% CPU reading"
        incident = engine.triage(metric, violations, generate_hints(metric, violations, [metric]))
        assert incident is not None
        assert incident.severity is Severity.CRITICAL
        assert stats_from_values([1.0]).count == 1
