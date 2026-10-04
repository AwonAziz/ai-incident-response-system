"""SQLite persistence for incidents and their lifecycle events.

The in-memory :class:`~src.triage.incident_manager.IncidentManager` is fast and
simple, but a restart loses everything: the incidents an operator is looking at,
and the deduplication state that stops the same problem re-alerting. This store
closes that gap.

Design choices:

* **Stdlib only.** ``sqlite3`` ships with Python; the project deliberately has no
  database dependency.
* **Connection per operation.** The control API serves requests on its own
  threads, and a shared connection would need cross-thread plumbing. Opening a
  connection is cheap, WAL mode makes concurrent readers safe, and it removes a
  whole class of "database is locked" / "SQLite objects created in a thread can
  only be used in that thread" failures.
* **The database is a cache of authoritative state.** Every write is a full-row
  upsert of the incident as the manager holds it, so the store can never drift
  from the live object, and a replayed incident is byte-identical to the one the
  dashboard rendered.
* **Only open incidents are restored.** Resolved history is kept for the record,
  but a restart should not resurrect a page an operator already dismissed.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.enums import Cloud, IncidentStatus, Severity
from src.triage.incident_manager import Incident, IncidentEvent
from src.triage.root_cause import Hypothesis

__all__ = ["SCHEMA_VERSION", "IncidentStore", "default_database_path"]

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

INCIDENT_COLUMNS = (
    "id",
    "fingerprint",
    "title",
    "cloud",
    "service",
    "resource_id",
    "region",
    "metric_name",
    "metric_value",
    "unit",
    "severity",
    "score",
    "confidence",
    "model_confidence",
    "status",
    "description",
    "hints",
    "causes",
    "violations",
    "occurrences",
    "escalations",
    "notified_severities",
    "sla_minutes",
    "estimated_resolution_minutes",
    "sla_deadline",
    "sla_breached",
    "created_at",
    "last_seen_at",
    "acknowledged_at",
    "resolved_at",
    "resolution",
    "tags",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS incidents (
    id                            TEXT PRIMARY KEY,
    fingerprint                   TEXT NOT NULL,
    title                         TEXT NOT NULL,
    cloud                         TEXT NOT NULL,
    service                       TEXT NOT NULL,
    resource_id                   TEXT NOT NULL DEFAULT '',
    region                        TEXT NOT NULL DEFAULT '',
    metric_name                   TEXT NOT NULL DEFAULT '',
    metric_value                  REAL NOT NULL DEFAULT 0,
    unit                          TEXT NOT NULL DEFAULT '',
    severity                      TEXT NOT NULL,
    score                         REAL NOT NULL DEFAULT 0,
    confidence                    REAL NOT NULL DEFAULT 0,
    model_confidence              REAL NOT NULL DEFAULT 0,
    status                        TEXT NOT NULL,
    description                   TEXT NOT NULL DEFAULT '',
    hints                         TEXT NOT NULL DEFAULT '[]',
    causes                        TEXT NOT NULL DEFAULT '[]',
    violations                    TEXT NOT NULL DEFAULT '[]',
    occurrences                   INTEGER NOT NULL DEFAULT 1,
    escalations                   INTEGER NOT NULL DEFAULT 0,
    notified_severities           TEXT NOT NULL DEFAULT '[]',
    sla_minutes                   REAL,
    estimated_resolution_minutes  REAL,
    sla_deadline                  TEXT,
    sla_breached                  INTEGER NOT NULL DEFAULT 0,
    created_at                    TEXT NOT NULL,
    last_seen_at                  TEXT NOT NULL,
    acknowledged_at               TEXT,
    resolved_at                   TEXT,
    resolution                    TEXT,
    tags                          TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_incidents_fingerprint ON incidents(fingerprint);
CREATE INDEX IF NOT EXISTS idx_incidents_created ON incidents(created_at);

CREATE TABLE IF NOT EXISTS incident_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          TEXT NOT NULL,
    kind        TEXT NOT NULL,
    incident_id TEXT NOT NULL,
    severity    TEXT NOT NULL,
    message     TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_events_incident ON incident_events(incident_id);
CREATE INDEX IF NOT EXISTS idx_events_at ON incident_events(at);
"""


def default_database_path() -> str:
    from config.settings import PROJECT_ROOT

    return str(Path(PROJECT_ROOT) / "data" / "incidents.db")


def _dump(value: Any) -> str:
    return json.dumps(value, default=str)


def _load(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:  # pragma: no cover - corrupted row
        logger.warning("discarding unparseable JSON column")
        return fallback


def _stamp(value: datetime | None) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _moment(raw: str | None) -> datetime | None:
    return datetime.fromisoformat(raw) if raw else None


class IncidentStore:
    """Durable home for incidents and their events."""

    def __init__(self, path: str | Path | None = None, *, timeout: float = 5.0) -> None:
        self.path = Path(path) if path is not None else Path(default_database_path())
        self.timeout = float(timeout)
        self._initialised = False

    # ── plumbing ────────────────────────────────────────────────────────
    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=self.timeout, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            if not self._initialised:
                connection.executescript(SCHEMA)
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(SCHEMA_VERSION),),
                )
                self._initialised = True
            yield connection
        finally:
            connection.close()

    def healthy(self) -> bool:
        try:
            with self.connection() as connection:
                connection.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error as exc:  # pragma: no cover - environment dependent
            logger.warning("incident store unavailable: %s", exc)
            return False

    # ── writes ──────────────────────────────────────────────────────────
    def save(self, incident: Incident) -> None:
        """Insert or update the full row for ``incident``."""
        row = self._row(incident)
        placeholders = ", ".join("?" for _ in INCIDENT_COLUMNS)
        columns = ", ".join(INCIDENT_COLUMNS)
        updates = ", ".join(f"{column}=excluded.{column}" for column in INCIDENT_COLUMNS if column != "id")
        with self.connection() as connection:
            connection.execute(
                f"INSERT INTO incidents ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(id) DO UPDATE SET {updates}",
                row,
            )

    def save_many(self, incidents: Sequence[Incident]) -> None:
        if not incidents:
            return
        with self.connection() as connection:
            self._save_with(connection, incidents)

    def append_event(self, event: IncidentEvent) -> None:
        with self.connection() as connection:
            self._append_event_with(connection, event)

    def append_events(self, events: Sequence[IncidentEvent]) -> None:
        if not events:
            return
        with self.connection() as connection:
            for event in events:
                self._append_event_with(connection, event)

    def delete_incident(self, incident_id: str) -> None:
        with self.connection() as connection:
            connection.execute("DELETE FROM incidents WHERE id = ?", (incident_id,))
            connection.execute("DELETE FROM incident_events WHERE incident_id = ?", (incident_id,))

    # ── reads ───────────────────────────────────────────────────────────
    def get(self, incident_id: str) -> Incident | None:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        return self._incident(row) if row else None

    def open_incidents(self) -> list[Incident]:
        """Incidents still open, oldest first - what a restart restores."""
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM incidents WHERE status != ? ORDER BY created_at ASC",
                (IncidentStatus.RESOLVED.value,),
            ).fetchall()
        return [self._incident(row) for row in rows]

    def all_incidents(self, limit: int | None = None) -> list[Incident]:
        query = "SELECT * FROM incidents ORDER BY created_at DESC"
        with self.connection() as connection:
            rows = connection.execute(query).fetchall()
        incidents = [self._incident(row) for row in rows]
        return incidents[:limit] if limit else incidents

    def events(self, limit: int = 50, incident_id: str | None = None) -> list[IncidentEvent]:
        query = "SELECT * FROM incident_events"
        params: tuple[Any, ...] = ()
        if incident_id:
            query += " WHERE incident_id = ?"
            params = (incident_id,)
        query += " ORDER BY at DESC, id DESC LIMIT ?"
        with self.connection() as connection:
            rows = connection.execute(query, (*params, int(limit))).fetchall()
        return [
            IncidentEvent(
                at=_moment(row["at"]) or datetime.now().astimezone(),
                kind=row["kind"],
                incident_id=row["incident_id"],
                severity=row["severity"],
                message=row["message"],
            )
            for row in rows
        ]

    def counts(self) -> dict[str, Any]:
        with self.connection() as connection:
            incidents = connection.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
            events = connection.execute("SELECT COUNT(*) FROM incident_events").fetchone()[0]
            open_rows = connection.execute(
                "SELECT COUNT(*) FROM incidents WHERE status != ?",
                (IncidentStatus.RESOLVED.value,),
            ).fetchone()[0]
        return {
            "path": str(self.path),
            "incidents": incidents,
            "open_incidents": open_rows,
            "events": events,
            "size_bytes": self.path.stat().st_size if self.path.is_file() else 0,
        }

    def vacuum(self) -> None:
        with self.connection() as connection:
            connection.execute("VACUUM")

    # ── internals ───────────────────────────────────────────────────────
    def _save_with(self, connection: sqlite3.Connection, incidents: Sequence[Incident]) -> None:
        columns = ", ".join(INCIDENT_COLUMNS)
        placeholders = ", ".join("?" for _ in INCIDENT_COLUMNS)
        connection.executemany(
            f"INSERT INTO incidents ({columns}) VALUES ({placeholders}) "
            "ON CONFLICT(id) DO UPDATE SET "
            + ", ".join(f"{column}=excluded.{column}" for column in INCIDENT_COLUMNS if column != "id"),
            [self._row(incident) for incident in incidents],
        )

    def _append_event_with(self, connection: sqlite3.Connection, event: IncidentEvent) -> None:
        connection.execute(
            "INSERT INTO incident_events (at, kind, incident_id, severity, message) VALUES (?, ?, ?, ?, ?)",
            (event.at.isoformat(), event.kind, event.incident_id, event.severity, event.message),
        )

    @staticmethod
    def _row(incident: Incident) -> tuple[Any, ...]:
        values: dict[str, Any] = {
            "id": incident.id,
            "fingerprint": incident.fingerprint,
            "title": incident.title,
            "cloud": Cloud.parse(incident.cloud).value,
            "service": incident.service,
            "resource_id": incident.resource_id,
            "region": incident.region,
            "metric_name": incident.metric_name,
            "metric_value": float(incident.metric_value),
            "unit": incident.unit,
            "severity": Severity.parse(incident.severity).name,
            "score": float(incident.score),
            "confidence": float(incident.confidence),
            "model_confidence": float(incident.model_confidence),
            "status": IncidentStatus.parse(incident.status).value,
            "description": incident.description,
            "hints": _dump(list(incident.hints)),
            "causes": _dump([cause.to_dict() for cause in incident.causes]),
            "violations": _dump(list(incident.violations)),
            "occurrences": int(incident.occurrences),
            "escalations": int(incident.escalations),
            "notified_severities": _dump(list(incident.notified_severities)),
            "sla_minutes": incident.sla_minutes,
            "estimated_resolution_minutes": incident.estimated_resolution_minutes,
            "sla_deadline": _stamp(incident.sla_deadline),
            "sla_breached": int(bool(incident.sla_breached)),
            "created_at": _stamp(incident.created_at),
            "last_seen_at": _stamp(incident.last_seen_at),
            "acknowledged_at": _stamp(incident.acknowledged_at),
            "resolved_at": _stamp(incident.resolved_at),
            "resolution": incident.resolution,
            "tags": _dump(dict(incident.tags)),
        }
        return tuple(values[column] for column in INCIDENT_COLUMNS)

    @staticmethod
    def _incident(row: sqlite3.Row) -> Incident:
        return Incident(
            id=row["id"],
            fingerprint=row["fingerprint"],
            title=row["title"],
            cloud=Cloud.parse(row["cloud"]),
            service=row["service"],
            resource_id=row["resource_id"],
            region=row["region"],
            metric_name=row["metric_name"],
            metric_value=row["metric_value"],
            unit=row["unit"],
            severity=Severity.parse(row["severity"]),
            score=row["score"],
            confidence=row["confidence"],
            model_confidence=row["model_confidence"],
            status=IncidentStatus.parse(row["status"]),
            description=row["description"],
            hints=list(_load(row["hints"], [])),
            causes=[
                Hypothesis(
                    cause=item.get("cause", ""),
                    category=item.get("category", "unknown"),
                    likelihood=float(item.get("likelihood", 0.0)),
                    evidence=tuple(item.get("evidence") or ()),
                )
                for item in _load(row["causes"], [])
            ],
            violations=list(_load(row["violations"], [])),
            occurrences=row["occurrences"],
            escalations=row["escalations"],
            notified_severities=list(_load(row["notified_severities"], [])),
            sla_minutes=row["sla_minutes"],
            estimated_resolution_minutes=row["estimated_resolution_minutes"],
            sla_deadline=_moment(row["sla_deadline"]),
            sla_breached=bool(row["sla_breached"]),
            created_at=_moment(row["created_at"]) or datetime.now().astimezone(),
            last_seen_at=_moment(row["last_seen_at"]) or datetime.now().astimezone(),
            acknowledged_at=_moment(row["acknowledged_at"]),
            resolved_at=_moment(row["resolved_at"]),
            resolution=row["resolution"],
            tags=dict(_load(row["tags"], {})),
        )
