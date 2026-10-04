"""Durable incident storage and restart recovery."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.core.enums import IncidentStatus, Severity
from src.triage.incident_manager import Incident, IncidentEvent, IncidentManager
from src.triage.store import SCHEMA_VERSION, IncidentStore, default_database_path


@pytest.fixture
def store(tmp_path: Path) -> IncidentStore:
    return IncidentStore(tmp_path / "incidents.db")


@pytest.fixture
def full_incident(clock) -> Incident:
    """An incident that touches every persisted column."""
    return Incident(
        id="INC-00001",
        fingerprint="fp-abc",
        title="AWS EC2 cpu utilization anomaly - CPU saturation",
        cloud="aws",
        service="EC2",
        resource_id="i-0a1f9c4d2e7b8a31",
        region="us-east-1",
        metric_name="cpu_utilization",
        metric_value=97.5,
        unit="%",
        severity=Severity.CRITICAL,
        score=9.25,
        confidence=0.95,
        model_confidence=0.81,
        description="cpu_utilization=97.50% on i-0a1f9c4d2e7b8a31",
        hints=["CPU saturation (60%)", "+9.0 sigma"],
        violations=[{"rule_id": "cpu_utilization.critical_high", "severity": "CRITICAL"}],
        occurrences=4,
        escalations=1,
        notified_severities=["CRITICAL"],
        sla_minutes=15.0,
        estimated_resolution_minutes=12.4,
        sla_deadline=clock.now(),
        sla_breached=True,
        created_at=clock.now(),
        last_seen_at=clock.now(),
        tags={"region": "us-east-1", "team": "platform"},
    )


class TestStoreRoundTrip:
    def test_missing_incident_is_none(self, store: IncidentStore) -> None:
        assert store.get("nope") is None

    def test_full_row_round_trip(self, store: IncidentStore, full_incident: Incident) -> None:
        store.save(full_incident)
        restored = store.get(full_incident.id)
        assert restored is not None
        for field in (
            "id",
            "fingerprint",
            "title",
            "cloud",
            "service",
            "resource_id",
            "region",
            "metric_name",
            "unit",
            "description",
            "resolution",
        ):
            assert getattr(restored, field) == getattr(full_incident, field), field
        assert restored.metric_value == pytest.approx(full_incident.metric_value)
        assert restored.severity is full_incident.severity
        assert restored.sla_deadline == full_incident.sla_deadline
        assert restored.hints == full_incident.hints
        assert restored.violations == full_incident.violations
        assert restored.tags == full_incident.tags
        assert restored.sla_breached is True
        assert restored.occurrences == 4
        assert restored.escalations == 1

    def test_causes_survive_the_round_trip(self, store: IncidentStore, full_incident: Incident) -> None:
        from src.triage.root_cause import Hypothesis

        full_incident.causes = [Hypothesis(cause="CPU saturation", category="capacity", likelihood=0.6, evidence=("+9 sigma",))]
        store.save(full_incident)
        restored = store.get(full_incident.id)
        assert restored is not None
        assert restored.causes[0].cause == "CPU saturation"
        assert restored.causes[0].category == "capacity"
        assert restored.causes[0].likelihood == pytest.approx(0.6)
        assert restored.causes[0].evidence == ("+9 sigma",)

    def test_save_is_an_upsert(self, store: IncidentStore, full_incident: Incident) -> None:
        store.save(full_incident)
        store.save(replace(full_incident, occurrences=9, severity=Severity.HIGH))
        restored = store.get(full_incident.id)
        assert restored is not None
        assert restored.occurrences == 9
        assert restored.severity is Severity.HIGH
        assert store.counts()["incidents"] == 1

    def test_save_many(self, store: IncidentStore, full_incident: Incident) -> None:
        store.save_many([full_incident, replace(full_incident, id="INC-00002", fingerprint="fp-def")])
        assert store.counts()["incidents"] == 2
        store.save_many([])
        assert store.counts()["incidents"] == 2

    def test_optional_datetime_columns(self, store: IncidentStore, full_incident: Incident) -> None:
        store.save(replace(full_incident, resolved_at=full_incident.created_at, acknowledged_at=full_incident.created_at))
        restored = store.get(full_incident.id)
        assert restored is not None
        assert restored.resolved_at == full_incident.created_at

    def test_json_columns_are_never_none(self, store: IncidentStore, clock) -> None:
        bare = Incident(id="INC-9", fingerprint="f", title="t", cloud="gcp", service="GKE", resource_id="")
        store.save(bare)
        restored = store.get("INC-9")
        assert restored is not None
        assert restored.hints == []
        assert restored.causes == []
        assert restored.violations == []
        assert restored.tags == {}

    def test_corrupt_json_column_is_discarded(self, store: IncidentStore, full_incident: Incident) -> None:
        store.save(full_incident)
        with store.connection() as connection:
            connection.execute("UPDATE incidents SET hints = 'not json' WHERE id = ?", (full_incident.id,))
        restored = store.get(full_incident.id)
        assert restored is not None
        assert restored.hints == []

    def test_delete_incident(self, store: IncidentStore, clock, full_incident: Incident) -> None:
        store.save(full_incident)
        store.append_event(IncidentEvent(clock.now(), "created", full_incident.id, "CRITICAL", "t"))
        store.delete_incident(full_incident.id)
        assert store.get(full_incident.id) is None
        assert store.events(10, incident_id=full_incident.id) == []


class TestStoreQueries:
    def test_open_incidents_excludes_resolved(self, store: IncidentStore, clock) -> None:
        base = Incident(id="INC-1", fingerprint="a", title="t", cloud="aws", service="EC2", resource_id="i-1")
        store.save(base)
        store.save(replace(base, id="INC-2", fingerprint="b", status=IncidentStatus.RESOLVED))
        open_rows = store.open_incidents()
        assert [item.id for item in open_rows] == ["INC-1"]

    def test_all_incidents_newest_first(self, store: IncidentStore, clock) -> None:
        first = Incident(
            id="INC-1",
            fingerprint="a",
            title="t",
            cloud="aws",
            service="EC2",
            resource_id="i-1",
            created_at=clock.now(),
        )
        store.save(first)
        clock.advance(60)
        store.save(replace(first, id="INC-2", fingerprint="b", created_at=clock.now()))
        assert [item.id for item in store.all_incidents()] == ["INC-2", "INC-1"]
        assert len(store.all_incidents(limit=1)) == 1

    def test_events_are_newest_first_and_filterable(self, store: IncidentStore, clock) -> None:
        store.append_events(
            [
                IncidentEvent(clock.now(), "created", "INC-1", "HIGH", "first"),
                IncidentEvent(clock.now(), "acknowledged", "INC-1", "HIGH", "second"),
                IncidentEvent(clock.now(), "resolved", "INC-2", "LOW", "other"),
            ]
        )
        assert store.events(10, incident_id="INC-1")[0].message == "second"
        assert len(store.events(10)) == 3
        store.append_events([])

    def test_counts_and_size(self, store: IncidentStore, full_incident: Incident) -> None:
        store.save(full_incident)
        payload = store.counts()
        assert payload["incidents"] == 1
        assert payload["open_incidents"] == 1
        assert payload["path"].endswith("incidents.db")
        assert payload["size_bytes"] > 0

    def test_healthy_and_vacuum(self, store: IncidentStore, full_incident: Incident) -> None:
        store.save(full_incident)
        assert store.healthy()
        store.vacuum()
        assert store.counts()["incidents"] == 1

    def test_schema_version_is_recorded(self, store: IncidentStore) -> None:
        with store.connection() as connection:
            row = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        assert row["value"] == str(SCHEMA_VERSION)

    def test_reopening_keeps_the_schema(self, tmp_path: Path, full_incident: Incident) -> None:
        path = tmp_path / "incidents.db"
        IncidentStore(path).save(full_incident)
        assert IncidentStore(path).get(full_incident.id) is not None

    def test_default_path_is_under_data(self) -> None:
        assert default_database_path().endswith(str(Path("data") / "incidents.db"))


class TestManagerPersistence:
    def test_manager_writes_through(self, store: IncidentStore, clock, full_incident: Incident) -> None:
        manager = IncidentManager(clock=clock, store=store)
        manager.add(full_incident)
        assert store.get(full_incident.id) is not None
        assert manager.persisted is True
        assert manager.stats["persistence"]["enabled"] is True

    def test_lifecycle_changes_are_persisted(self, store: IncidentStore, clock, full_incident: Incident) -> None:
        manager = IncidentManager(clock=clock, store=store)
        manager.add(full_incident)
        manager.acknowledge(full_incident.id)
        assert store.get(full_incident.id).status is IncidentStatus.ACKNOWLEDGED
        manager.resolve(full_incident.id, "done")
        assert store.get(full_incident.id).status is IncidentStatus.RESOLVED
        assert store.get(full_incident.id).resolution == "done"

    def test_escalation_is_persisted(self, store: IncidentStore, clock, full_incident: Incident) -> None:
        manager = IncidentManager(clock=clock, store=store)
        manager.add(full_incident)
        manager.escalate(full_incident, Severity.CRITICAL)
        assert store.get(full_incident.id).escalations == 1

    def test_repeats_are_written_in_batches_not_on_every_touch(
        self,
        store: IncidentStore,
        clock,
        full_incident: Incident,
    ) -> None:
        manager = IncidentManager(clock=clock, store=store)
        manager.add(full_incident)
        for _ in range(5):
            manager.touch(full_incident, clock.now())
        assert store.get(full_incident.id).occurrences == 4  # nothing flushed yet
        manager.touch(full_incident, clock.now())  # reaches 10 -> flushed
        assert store.get(full_incident.id).occurrences == 10

    def test_events_are_persisted(self, store: IncidentStore, clock, full_incident: Incident) -> None:
        manager = IncidentManager(clock=clock, store=store)
        manager.add(full_incident)
        manager.acknowledge(full_incident.id)
        manager.resolve(full_incident.id, "done")
        kinds = [event.kind for event in store.events(10)]
        assert kinds[:3] == ["resolved", "acknowledged", "created"]

    def test_storage_failure_does_not_break_triage(self, clock, full_incident: Incident) -> None:
        class BrokenStore:
            path = "memory://broken"

            def healthy(self) -> bool:
                return True

            def save(self, incident: Incident) -> None:
                raise RuntimeError("disk full")

            def append_event(self, event: IncidentEvent) -> None:
                raise RuntimeError("disk full")

        manager = IncidentManager(clock=clock, store=BrokenStore())  # type: ignore[arg-type]
        manager.add(full_incident)
        manager.acknowledge(full_incident.id)
        assert manager.get(full_incident.id).status is IncidentStatus.ACKNOWLEDGED

    def test_no_store_means_no_persistence_calls(self, clock, full_incident: Incident) -> None:
        manager = IncidentManager(clock=clock)
        manager.add(full_incident)
        assert manager.persisted is False
        assert manager.restore() == []
        assert manager.stats["persistence"]["enabled"] is False


class TestRestartRecovery:
    def _incident(self, clock, identifier: str, fingerprint: str, severity=Severity.HIGH) -> Incident:
        return Incident(
            id=identifier,
            fingerprint=fingerprint,
            title=f"incident {identifier}",
            cloud="aws",
            service="EC2",
            resource_id="i-1",
            severity=severity,
            created_at=clock.now(),
            last_seen_at=clock.now(),
        )

    def test_open_incidents_survive_a_restart(self, tmp_path: Path, clock) -> None:
        path = tmp_path / "incidents.db"
        first = IncidentManager(clock=clock, store=IncidentStore(path))
        first.add(self._incident(clock, "INC-1", "fp-1", Severity.CRITICAL))
        first.add(self._incident(clock, "INC-2", "fp-2"))
        first.resolve("INC-2", "handled")

        second = IncidentManager(clock=clock, store=IncidentStore(path))
        restored = second.restore()
        assert [item.id for item in restored] == ["INC-1"]
        assert restored[0].severity is Severity.CRITICAL
        assert second.restored_incidents == 1
        assert second.stats["persistence"]["restored"] == 1

    def test_resolved_incidents_are_not_resurrected(self, tmp_path: Path, clock) -> None:
        path = tmp_path / "incidents.db"
        first = IncidentManager(clock=clock, store=IncidentStore(path))
        first.add(self._incident(clock, "INC-1", "fp-1"))
        first.resolve("INC-1", "handled")

        second = IncidentManager(clock=clock, store=IncidentStore(path))
        assert second.restore() == []
        assert second.active() == []

    def test_restored_incident_is_acknowledged_and_solvable(self, tmp_path: Path, clock) -> None:
        path = tmp_path / "incidents.db"
        first = IncidentManager(clock=clock, store=IncidentStore(path))
        first.add(self._incident(clock, "INC-1", "fp-1"))

        second = IncidentManager(clock=clock, store=IncidentStore(path))
        second.restore()
        second.acknowledge("INC-1", "on it")
        assert second.get("INC-1").status is IncidentStatus.ACKNOWLEDGED
        second.resolve("INC-1", "closed after restart")
        third = IncidentManager(clock=clock, store=IncidentStore(path))
        assert third.restore() == []

    def test_dedup_state_survives_a_restart(self, tmp_path: Path, clock, warm_metric) -> None:
        """The repeat that a restart must not re-alert on."""
        from src.detection.rule_engine import RuleViolation
        from src.triage import TriageEngine

        path = tmp_path / "incidents.db"
        violations = [
            RuleViolation(
                rule_id="cpu_utilization.critical_high",
                metric_name="cpu_utilization",
                cloud=warm_metric.cloud,
                resource_id=warm_metric.resource_id,
                service=warm_metric.service,
                severity=Severity.CRITICAL,
                direction="high",
                threshold=92.0,
                value=warm_metric.value,
                message="breached",
            )
        ]

        manager = IncidentManager(clock=clock, store=IncidentStore(path))
        engine = TriageEngine(dedup_window_seconds=300, clock=clock).bind(manager)
        created = engine.triage(warm_metric, violations)
        assert created is not None
        manager.add(created)

        # "restart": fresh manager + engine over the same database
        restarted_manager = IncidentManager(clock=clock, store=IncidentStore(path))
        restarted_manager.restore()
        restarted_engine = TriageEngine(dedup_window_seconds=300, clock=clock).bind(restarted_manager)

        assert len(restarted_engine.open_fingerprints()) == 1
        repeat = restarted_engine.triage(warm_metric, violations)
        assert repeat is None, "a restored incident must keep suppressing its own repeat"
        assert restarted_manager.get(created.id).occurrences == 2

    def test_restored_history_is_capped(self, tmp_path: Path, clock) -> None:
        path = tmp_path / "incidents.db"
        first = IncidentManager(clock=clock, store=IncidentStore(path))
        for index in range(5):
            first.add(self._incident(clock, f"INC-{index}", f"fp-{index}"))
        second = IncidentManager(clock=clock, store=IncidentStore(path), max_history=3)
        assert len(second.restore()) == 5  # the store keeps them; the manager trims as it goes
        assert len(second.all()[:3]) == 3
