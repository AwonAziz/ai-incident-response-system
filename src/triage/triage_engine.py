"""Triage engine: scoring, severity classification, deduplication, SLA maths.

An incident is only created when something actionable happened:

* at least one rule violation, or
* the ML model flagged the sample with confidence >= ``ml_min_confidence``.

Identical problems (same cloud + service + metric) collapse onto the same
fingerprint inside the dedup window, which is what prevents alert storms. A
repeat that is *more severe* escalates the existing incident instead of opening
a new one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from config.settings import Settings
from src.core.clock import Clock, SystemClock
from src.core.enums import Cloud, Severity
from src.ingestion.metric_schema import Metric
from src.triage.incident_manager import Incident, IncidentManager
from src.triage.root_cause import RootCause, analyze, generate_hints

__all__ = ["TriageEngine", "TriageSettings"]

DEFAULT_SEVERITY_THRESHOLDS: dict[str, float] = {"critical": 8.0, "high": 5.5, "medium": 3.0}
DEFAULT_SLA_TARGETS: dict[str, float] = {"critical": 15.0, "high": 60.0, "medium": 240.0, "low": 1440.0}
#: base time-to-resolve estimate (minutes) before the confidence adjustment;
#: deliberately below the SLA target so `sla_breached` flags *uncertain* cases
#: instead of every high-severity incident.
BASE_RESOLUTION_MINUTES: dict[str, float] = {"critical": 12.0, "high": 40.0, "medium": 120.0, "low": 480.0}
#: confidence implied by a static threshold breach of each severity; a CRITICAL
#: threshold breach is a definitive detection, so it outranks a weak model score
#: when estimating time-to-resolve.
RULE_CONFIDENCE: dict[str, float] = {"critical": 0.95, "high": 0.8, "medium": 0.6, "low": 0.4}


@dataclass(frozen=True, slots=True)
class TriageSettings:
    """Everything the engine needs to score an incident."""

    dedup_window_seconds: int = 60
    ml_min_confidence: float = 0.5
    critical_threshold: float = 8.0
    high_threshold: float = 5.5
    medium_threshold: float = 3.0
    severity_factors: dict[str, float] | None = None
    sla_targets_minutes: dict[str, float] | None = None
    blast_radius_cap: int = 3
    z_score_bonus: float = 0.5

    @classmethod
    def from_settings(cls, settings: Settings) -> TriageSettings:
        return cls(
            dedup_window_seconds=settings.dedup_window_seconds,
            ml_min_confidence=settings.ml_min_confidence,
            critical_threshold=settings.critical_score,
            high_threshold=settings.high_score,
            medium_threshold=settings.medium_score,
            severity_factors=settings.severity_factors,
            sla_targets_minutes=settings.sla_targets_minutes,
        )


class TriageEngine:
    """Turns detections into deduplicated, prioritised incidents."""

    def __init__(
        self,
        dedup_window_seconds: int = 60,
        *,
        triage_settings: TriageSettings | None = None,
        settings: Settings | None = None,
        clock: Clock | None = None,
        id_prefix: str = "INC",
    ) -> None:
        base = triage_settings or (TriageSettings.from_settings(settings) if settings else None)
        self.config = base or TriageSettings(dedup_window_seconds=int(dedup_window_seconds))
        self.clock = clock or SystemClock()
        self.id_prefix = id_prefix
        self._sequence = 0
        self._open: dict[str, Incident] = {}
        self._manager: IncidentManager | None = None
        self._counters = {"triaged": 0, "created": 0, "suppressed": 0, "escalated": 0, "below_threshold": 0}

    # ── wiring ─────────────────────────────────────────────────────────
    def bind(self, manager: IncidentManager) -> TriageEngine:
        """Give the engine access to incident state (status, superseding).

        Also seeds the dedup registry from whatever the manager already holds, so
        incidents restored from disk after a restart keep suppressing repeats
        instead of re-alerting as new ones.
        """
        self._manager = manager
        for incident in manager.active():
            self._open.setdefault(incident.fingerprint, incident)
        return self

    @property
    def stats(self) -> dict[str, Any]:
        payload = dict(self._counters)
        payload["open_fingerprints"] = len(self._open)
        payload["dedup_window_seconds"] = self.config.dedup_window_seconds
        return payload

    def open_fingerprints(self) -> tuple[str, ...]:
        return tuple(self._open)

    # ── fingerprinting ─────────────────────────────────────────────────
    @staticmethod
    def fingerprint(metric: Metric) -> str:
        raw = metric.scope_key
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    # ── scoring ────────────────────────────────────────────────────────
    def severity_factors(self) -> dict[str, float]:
        return self.config.severity_factors or {"critical": 3.0, "high": 2.0, "medium": 1.2, "low": 0.5}

    def score(
        self,
        metric: Metric,
        violations: Sequence[Any] = (),
        context: Sequence[Metric] = (),
    ) -> float:
        """Blended rule + ML + statistical score (higher is worse)."""
        factors = self.severity_factors()
        total = 0.0
        for violation in violations:
            severity = getattr(violation, "severity", Severity.LOW)
            weight = float(getattr(violation, "weight", 1.0))
            total += factors.get(severity.slug, 1.0) * weight

        if not violations:
            # ML-only detection: confidence is the whole signal
            total += 3.0 * metric.confidence

        if metric.window is not None and metric.window.count > 1:
            total += min(abs(metric.window.z_score(metric.value)), 8.0) * self.config.z_score_bonus

        siblings = self.blast_radius(metric, context)
        total += siblings * 1.2

        return round(total, 4)

    def classify(self, score: float) -> Severity:
        config = self.config
        if score >= config.critical_threshold:
            return Severity.CRITICAL
        if score >= config.high_threshold:
            return Severity.HIGH
        if score >= config.medium_threshold:
            return Severity.MEDIUM
        return Severity.LOW

    def severity_for(self, metric: Metric, violations: Sequence[Any], context: Sequence[Metric] = ()) -> Severity:
        """Severity is the worst of the blended score and the strongest rule.

        The floor matters for predictability: a ``critical_high`` breach is
        always CRITICAL regardless of how quiet the metric's history is.
        """
        severity = self.classify(self.score(metric, violations, context))
        for violation in violations:
            candidate = getattr(violation, "severity", None)
            if isinstance(candidate, Severity) and candidate > severity:
                severity = candidate
        return severity

    def blast_radius(self, metric: Metric, context: Sequence[Metric] = ()) -> int:
        """Number of ML-flagged sibling signals on the same resource/service."""
        return min(len(self._breaching_siblings(metric, context)), self.config.blast_radius_cap)

    @staticmethod
    def _breaching_siblings(metric: Metric, context: Sequence[Metric]) -> list[Metric]:
        """Sibling metrics the model flagged on the same resource or service.

        Rule breaches are already counted individually in :meth:`score`, so the
        blast-radius bonus only considers ML-flagged siblings to avoid
        double-counting the same signal.
        """
        return [
            sibling
            for sibling in context
            if sibling.series_key != metric.series_key
            and sibling.name != metric.name
            and (sibling.resource_id == metric.resource_id or sibling.service == metric.service)
            and sibling.is_anomaly
        ]

    # ── main entry point ───────────────────────────────────────────────
    def triage(
        self,
        metric: Metric,
        violations: Sequence[Any] = (),
        hints: Sequence[str] | None = None,
        *,
        context: Sequence[Metric] = (),
    ) -> Incident | None:
        """Return a new/escalated incident, or ``None`` when nothing to do."""
        self._counters["triaged"] += 1
        if not violations and metric.confidence < self.config.ml_min_confidence:
            self._counters["below_threshold"] += 1
            return None

        score = self.score(metric, violations, context)
        severity = self.severity_for(metric, violations, context)
        fingerprint = self.fingerprint(metric)
        now = self.clock.now()
        existing = self._lookup(fingerprint)

        if existing is not None:
            within_window = (now - existing.last_seen_at).total_seconds() < self.config.dedup_window_seconds
            if within_window:
                if self._manager is not None:
                    self._manager.touch(existing, now)
                else:
                    existing.occurrences += 1
                    existing.last_seen_at = now
                if severity > existing.severity:
                    existing.severity = severity
                    existing.escalations += 1
                    existing.score = max(existing.score, score)
                    existing.confidence = max(existing.confidence, metric.confidence)
                    self._counters["escalated"] += 1
                    return existing
                self._counters["suppressed"] += 1
                return None
            # Same problem, outside the dedup window: close the stale incident
            # and open a fresh one so SLA timers restart.
            if self._manager is not None:
                self._manager.resolve(existing.id, note="superseded by a new detection", auto=True)

        incident = self._build_incident(
            metric=metric,
            violations=violations,
            hints=hints,
            score=score,
            severity=severity,
            fingerprint=fingerprint,
            now=now,
            context=context,
            blast=self.blast_radius(metric, context),
        )
        self._open[fingerprint] = incident
        self._counters["created"] += 1
        return incident

    def on_incident_event(self, kind: str, incident: Incident) -> None:
        """``IncidentManager`` listener: drop dedup state once an incident closes."""
        if kind == "resolved":
            self.on_resolved(incident)

    def on_resolved(self, fingerprint: str | Incident) -> None:
        """Forget a fingerprint so the next detection opens a fresh incident."""
        key = fingerprint if isinstance(fingerprint, str) else fingerprint.fingerprint
        self._open.pop(key, None)

    # ── construction ───────────────────────────────────────────────────
    def _build_incident(
        self,
        *,
        metric: Metric,
        violations: Sequence[Any],
        hints: Sequence[str] | None,
        score: float,
        severity: Severity,
        fingerprint: str,
        now: datetime,
        context: Sequence[Metric],
        blast: int = 0,
    ) -> Incident:
        root_cause: RootCause = analyze(metric, violations, context)
        hint_list = list(hints) if hints else generate_hints(metric, violations, context)
        sla_targets = self.config.sla_targets_minutes or DEFAULT_SLA_TARGETS
        sla_minutes = float(sla_targets.get(severity.slug, DEFAULT_SLA_TARGETS["low"]))
        confidence = self._detection_confidence(metric, violations, severity)
        # time-to-resolve grows with cause uncertainty (low confidence) and with
        # the number of correlated signals that have to be triaged
        estimate = (
            BASE_RESOLUTION_MINUTES.get(severity.slug, 480.0)
            * (1.0 + 0.5 * (1.0 - confidence))
            * (1.0 + 0.15 * blast)
        )
        deadline = now + _minutes(sla_minutes)

        self._sequence += 1
        return Incident(
            id=f"{self.id_prefix}-{self._sequence:05d}",
            fingerprint=fingerprint,
            title=self._title(metric, severity, root_cause),
            cloud=Cloud.parse(metric.cloud),
            service=metric.service,
            resource_id=metric.resource_id,
            region=metric.region,
            metric_name=metric.name,
            metric_value=metric.value,
            unit=metric.unit,
            severity=severity,
            score=score,
            confidence=confidence,
            model_confidence=metric.confidence,
            description=self._description(metric, violations),
            hints=hint_list[:5],
            causes=list(root_cause.top(3)),
            violations=[violation.to_dict() for violation in violations],
            notified_severities=[severity.name],
            sla_minutes=sla_minutes,
            estimated_resolution_minutes=round(estimate, 2),
            sla_deadline=deadline,
            sla_breached=estimate > sla_minutes,
            created_at=now,
            last_seen_at=now,
            tags={"region": metric.region} if metric.region else {},
        )

    @staticmethod
    def _detection_confidence(
        metric: Metric,
        violations: Sequence[Any],
        severity: Severity,
    ) -> float:
        """How sure we are that this is a real problem.

        Rule-driven detections inherit the confidence of the strongest
        threshold breach; ML-only detections use the model score.
        """
        rule_confidence = 0.0
        for violation in violations:
            candidate = getattr(violation, "severity", None)
            if isinstance(candidate, Severity):
                rule_confidence = max(rule_confidence, RULE_CONFIDENCE.get(candidate.slug, 0.4))
        return round(min(1.0, max(metric.confidence, rule_confidence, RULE_CONFIDENCE.get(severity.slug, 0.4) if violations else 0.0)), 4)

    @staticmethod
    def _title(metric: Metric, severity: Severity, root_cause: RootCause) -> str:
        primary = root_cause.primary
        cause = primary.cause if primary else "metric anomaly"
        label = metric.name.replace("_", " ")
        return f"{metric.cloud.label} {metric.service} {label} anomaly - {cause}"

    @staticmethod
    def _description(metric: Metric, violations: Sequence[Any]) -> str:
        unit = metric.unit
        head = f"{metric.name}={metric.value:.2f}{unit} on {metric.resource_id} ({metric.cloud.value}/{metric.region})"
        if violations:
            breached = ", ".join(str(violation.rule_id) for violation in violations[:4])
            return f"{head}. Breached: {breached}"
        return f"{head}. Flagged by the anomaly model only (confidence {metric.confidence:.0%})."

    # ── helpers ────────────────────────────────────────────────────────
    def _lookup(self, fingerprint: str) -> Incident | None:
        """The currently open incident for a fingerprint, if any.

        The engine owns the dedup registry so suppression works even when no
        incident manager is bound; the manager is only consulted to notice that
        an incident was closed (acknowledged/resolved/auto-resolved) elsewhere.
        """
        incident = self._open.get(fingerprint)
        if incident is None:
            return None
        if self._manager is not None:
            current = self._manager.get(incident.id)
            if current is None or current.status.is_closed:
                self._open.pop(fingerprint, None)
                return None
        return incident


def _minutes(value: float) -> timedelta:
    return timedelta(minutes=value)
