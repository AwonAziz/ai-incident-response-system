"""Incident records and lifecycle management.

:class:`IncidentManager` is the single source of truth for incident state. It is
thread-safe (the control API reads it from its own thread) and emits lifecycle
events so the dashboard can maintain a live feed without polling internals.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import TYPE_CHECKING, Any

from src.core.clock import Clock, SystemClock, utc_now
from src.core.enums import Cloud, IncidentStatus, Severity
from src.triage.root_cause import Hypothesis

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.triage.store import IncidentStore

logger = logging.getLogger(__name__)

__all__ = ["Incident", "IncidentEvent", "IncidentListener", "IncidentManager"]

IncidentListener = Callable[[str, "Incident"], None]


@dataclass(slots=True)
class Incident:
    """One incident, from detection to resolution."""

    id: str
    fingerprint: str
    title: str
    cloud: Cloud
    service: str
    resource_id: str
    region: str = ""
    metric_name: str = ""
    metric_value: float = 0.0
    unit: str = ""
    severity: Severity = Severity.LOW
    score: float = 0.0
    confidence: float = 0.0
    model_confidence: float = 0.0
    status: IncidentStatus = IncidentStatus.OPEN
    description: str = ""
    hints: list[str] = field(default_factory=list)
    causes: list[Hypothesis] = field(default_factory=list)
    violations: list[dict[str, Any]] = field(default_factory=list)
    occurrences: int = 1
    escalations: int = 0
    notified_severities: list[str] = field(default_factory=list)
    sla_minutes: float | None = None
    estimated_resolution_minutes: float | None = None
    sla_deadline: datetime | None = None
    sla_breached: bool = False
    created_at: datetime = field(default_factory=utc_now)
    last_seen_at: datetime = field(default_factory=utc_now)
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    resolution: str | None = None
    tags: dict[str, str] = field(default_factory=dict)

    # ── derived ────────────────────────────────────────────────────────
    @property
    def is_open(self) -> bool:
        return self.status.is_open

    def age_seconds(self, now: datetime) -> float:
        return max(0.0, (now - self.created_at).total_seconds())

    def time_to_breach_seconds(self, now: datetime) -> float | None:
        if self.sla_deadline is None or self.status.is_closed:
            return None
        return (self.sla_deadline - now).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cloud"] = self.cloud.value
        data["severity"] = self.severity.name
        data["status"] = self.status.value
        data["causes"] = [cause.to_dict() for cause in self.causes]
        for key in ("created_at", "last_seen_at", "acknowledged_at", "resolved_at", "sla_deadline"):
            value = data.get(key)
            data[key] = value.isoformat() if isinstance(value, datetime) else None
        return data

    def summary_line(self) -> str:
        return (
            f"{self.id} [{self.severity.name}] {self.cloud.value}/{self.service} "
            f"{self.metric_name}={self.metric_value:.2f}{self.unit} x{self.occurrences} - {self.title}"
        )


@dataclass(frozen=True, slots=True)
class IncidentEvent:
    """Lifecycle event appended to the incident feed."""

    at: datetime
    kind: str
    incident_id: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at.isoformat(),
            "kind": self.kind,
            "incident_id": self.incident_id,
            "severity": self.severity,
            "message": self.message,
        }


class IncidentManager:
    """Registry + lifecycle for incidents.

    Optionally backed by an :class:`~src.triage.store.IncidentStore`: every
    state change is written through, and :meth:`restore` brings open incidents
    back after a restart so an operator does not lose the page they were looking
    at.

    ``persist_every`` bounds the write amplification of :meth:`touch`: a
    repeated observation only bumps a counter, so the row is flushed when the
    occurrence count reaches a multiple of this value (set it to 1 to flush
    every repeat).
    """

    def __init__(
        self,
        *,
        max_history: int = 500,
        clock: Clock | None = None,
        store: IncidentStore | None = None,
        persist_every: int = 10,
    ) -> None:
        self.clock = clock or SystemClock()
        self.max_history = max(1, int(max_history))
        self.store = store
        self.persist_every = max(1, int(persist_every))
        self.restored_incidents = 0
        self.persisted = False
        self._incidents: dict[str, Incident] = {}
        self._order: list[str] = []
        self._events: list[IncidentEvent] = []
        self._listeners: list[IncidentListener] = []
        self._lock = RLock()
        self._counters = {
            "total_created": 0,
            "total_acknowledged": 0,
            "total_resolved": 0,
            "total_escalated": 0,
            "total_auto_resolved": 0,
        }

    # ── persistence ─────────────────────────────────────────────────────
    def restore(self) -> list[Incident]:
        """Reload open incidents from the store. Returns what came back."""
        if self.store is None:
            return []
        restored = self.store.open_incidents()
        with self._lock:
            for incident in restored:
                self._incidents[incident.id] = incident
                self._order.append(incident.id)
                self._counters["total_created"] += 1
            self.restored_incidents = len(restored)
        return restored

    def _persist(self, incident: Incident) -> None:
        if self.store is None:
            return
        try:
            self.store.save(incident)
            self.persisted = True
        except Exception as exc:
            logger.warning("could not persist incident %s: %s", incident.id, exc)

    def _persist_event(self, event: IncidentEvent) -> None:
        if self.store is None:
            return
        try:
            self.store.append_event(event)
        except Exception as exc:
            logger.warning("could not persist event for %s: %s", event.incident_id, exc)

    # ── wiring ─────────────────────────────────────────────────────────
    def add_listener(self, listener: IncidentListener) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(self, listener: IncidentListener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    # ── mutations ──────────────────────────────────────────────────────
    def add(self, incident: Incident, *, notify: bool = True) -> Incident:
        """Register a new incident (the triage engine owns construction)."""
        now = self.clock.now()
        if not incident.created_at:
            incident.created_at = now
        incident.last_seen_at = incident.last_seen_at or now
        with self._lock:
            if incident.id in self._incidents:
                raise ValueError(f"incident {incident.id} already exists")
            self._incidents[incident.id] = incident
            self._order.append(incident.id)
            self._counters["total_created"] += 1
            self._record_event("created", incident, incident.title)
            self._trim()
            self._persist(incident)
        if notify:
            self._emit("created", incident)
        return incident

    def acknowledge(self, incident_id: str, note: str | None = None) -> Incident:
        with self._lock:
            incident = self._require(incident_id)
            if incident.status.is_closed:
                raise ValueError(f"incident {incident_id} is already resolved")
            incident.status = IncidentStatus.ACKNOWLEDGED
            incident.acknowledged_at = self.clock.now()
            self._counters["total_acknowledged"] += 1
            self._record_event("acknowledged", incident, note or "acknowledged by operator")
            self._persist(incident)
        self._emit("acknowledged", incident)
        return incident

    def resolve(self, incident_id: str, note: str | None = None, *, auto: bool = False) -> Incident:
        with self._lock:
            incident = self._require(incident_id)
            if incident.status.is_closed:
                return incident
            incident.status = IncidentStatus.RESOLVED
            incident.resolved_at = self.clock.now()
            incident.resolution = note or "resolved"
            self._counters["total_resolved"] += 1
            if auto:
                self._counters["total_auto_resolved"] += 1
            self._record_event("resolved", incident, incident.resolution)
            self._persist(incident)
        self._emit("resolved", incident)
        return incident

    def escalate(self, incident: Incident, severity: Severity, now: datetime | None = None) -> Incident:
        """Raise severity on an existing incident (dedup escalation path)."""
        with self._lock:
            if severity <= incident.severity:
                return incident
            previous = incident.severity
            incident.severity = severity
            incident.escalations += 1
            incident.score = max(incident.score, severity * 2.5)
            moment = now or self.clock.now()
            incident.last_seen_at = moment
            self._counters["total_escalated"] += 1
            self._record_event("escalated", incident, f"{previous.name} -> {severity.name}")
            self._persist(incident)
        self._emit("escalated", incident)
        return incident

    def touch(self, incident: Incident, now: datetime | None = None) -> Incident:
        """Record a repeated observation of an open incident."""
        with self._lock:
            incident.occurrences += 1
            incident.last_seen_at = now or self.clock.now()
            # A repeat with no state change is not worth a database write.
            if incident.occurrences % self.persist_every == 0:
                self._persist(incident)
        return incident

    def auto_resolve_old(
        self,
        max_age_minutes: float = 10.0,
        severities: Sequence[Severity] = (Severity.MEDIUM, Severity.LOW),
    ) -> list[Incident]:
        """Resolve stale incidents of low-severity tiers.

        CRITICAL/HIGH are never auto-resolved: closing them automatically would
        hide an unacknowledged outage.
        """
        resolved: list[Incident] = []
        now = self.clock.now()
        cutoff = timedelta(minutes=float(max_age_minutes))
        allowed = set(severities)
        with self._lock:
            candidates = [
                incident
                for incident in self.active()
                if incident.severity in allowed and now - incident.last_seen_at >= cutoff
            ]
        for incident in candidates:
            resolved.append(
                self.resolve(
                    incident.id,
                    note=f"auto-resolved after {max_age_minutes:g}m without new signals",
                    auto=True,
                )
            )
        return resolved

    # ── queries ────────────────────────────────────────────────────────
    def get(self, incident_id: str) -> Incident | None:
        with self._lock:
            return self._incidents.get(incident_id)

    def require(self, incident_id: str) -> Incident:
        with self._lock:
            return self._require(incident_id)

    def find_by_fingerprint(self, fingerprint: str) -> Incident | None:
        with self._lock:
            for incident_id in reversed(self._order):
                incident = self._incidents[incident_id]
                if incident.fingerprint == fingerprint and incident.is_open:
                    return incident
            return None

    def active(self) -> list[Incident]:
        with self._lock:
            items = [self._incidents[key] for key in self._order if self._incidents[key].is_open]
        items.sort(key=lambda item: (-int(item.severity), item.created_at))
        return items

    def all(self) -> list[Incident]:
        with self._lock:
            return [self._incidents[key] for key in self._order]

    def recent(self, limit: int = 10) -> list[Incident]:
        with self._lock:
            return [self._incidents[key] for key in self._order[-int(limit) :]][::-1]

    def events(self, limit: int = 25) -> list[IncidentEvent]:
        with self._lock:
            return list(self._events[-int(limit) :])[::-1]

    def counts_by_severity(self) -> dict[str, int]:
        counts = {severity.name: 0 for severity in Severity}
        for incident in self.active():
            counts[incident.severity.name] += 1
        return counts

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._counters)
            payload["active_count"] = sum(1 for item in self._incidents.values() if item.is_open)
            payload["total_tracked"] = len(self._incidents)
            payload["by_severity"] = self.counts_by_severity()
            payload["open_by_cloud"] = self._count_by("cloud")
            payload["open_by_service"] = self._count_by("service")
            payload["persistence"] = {
                "enabled": self.store is not None,
                "written": self.persisted,
                "restored": self.restored_incidents,
                "path": str(self.store.path) if self.store else None,
            }
            return payload

    def snapshot(self, limit: int = 25) -> dict[str, Any]:
        """JSON-ready view used by the control API and the CLI summary."""
        return {
            "stats": self.stats,
            "active": [incident.to_dict() for incident in self.active()[:limit]],
            "recent": [incident.to_dict() for incident in self.recent(limit)],
            "events": [event.to_dict() for event in self.events(limit)],
        }

    # ── internals ──────────────────────────────────────────────────────
    def _count_by(self, attribute: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for incident in self.active():
            raw = getattr(incident, attribute)
            key = raw.value if isinstance(raw, (Cloud, Severity)) else str(raw)
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: -item[1]))

    def _require(self, incident_id: str) -> Incident:
        incident = self._incidents.get(incident_id)
        if incident is None:
            raise KeyError(f"unknown incident {incident_id}")
        return incident

    def _record_event(self, kind: str, incident: Incident, message: str) -> None:
        event = IncidentEvent(
            at=self.clock.now(),
            kind=kind,
            incident_id=incident.id,
            severity=incident.severity.name,
            message=message,
        )
        self._events.append(event)
        if len(self._events) > self.max_history:
            del self._events[: len(self._events) - self.max_history]
        self._persist_event(event)

    def _trim(self) -> None:
        while len(self._order) > self.max_history:
            oldest = self._order[0]
            candidate = self._incidents.get(oldest)
            if candidate is not None and candidate.is_open:
                break
            self._order.pop(0)
            self._incidents.pop(oldest, None)
            if self.store is not None and candidate is not None:
                try:
                    self.store.delete_incident(oldest)
                except Exception as exc:
                    logger.warning("could not prune incident %s: %s", oldest, exc)

    def _emit(self, kind: str, incident: Incident) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(kind, incident)
            except Exception:
                continue

    def __len__(self) -> int:
        with self._lock:
            return len(self._incidents)

    def __iter__(self) -> Iterable[Incident]:  # pragma: no cover - convenience
        return iter(self.all())
