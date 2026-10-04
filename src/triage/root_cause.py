"""Root-cause hypotheses.

Detection says *something* is wrong; this module tries to say *what kind* of
thing, using three evidence sources:

1. the breaching metric itself (``latency_ms`` -> dependency / lock contention),
2. the other signals in the same batch (latency **and** errors together is a
   very different story than latency alone),
3. the rolling statistics (how far out of band, how fast it moved).

Output is a ranked list of :class:`Hypothesis` objects with explicit evidence
strings, plus a flat list of human-readable hints for the dashboard/Slack.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.core.enums import Severity
from src.ingestion.metric_schema import Metric

__all__ = ["SIGNAL_RULES", "Hypothesis", "RootCause", "analyze", "generate_hints"]


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A candidate explanation with a normalised likelihood."""

    cause: str
    category: str
    likelihood: float
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "category": self.category,
            "likelihood": round(self.likelihood, 4),
            "evidence": list(self.evidence),
        }

    def describe(self) -> str:
        return f"{self.cause} ({self.likelihood:.0%}) - {'; '.join(self.evidence)}"


@dataclass(frozen=True, slots=True)
class RootCause:
    """Ranked hypotheses for one breaching metric."""

    series_key: str
    metric_name: str
    cloud: str
    severity: Severity
    hypotheses: tuple[Hypothesis, ...] = ()
    statistics: dict[str, float] = field(default_factory=dict)

    @property
    def primary(self) -> Hypothesis | None:
        return self.hypotheses[0] if self.hypotheses else None

    def top(self, count: int = 3) -> tuple[Hypothesis, ...]:
        return self.hypotheses[:count]

    def to_dict(self) -> dict[str, Any]:
        return {
            "series": self.series_key,
            "metric": self.metric_name,
            "cloud": self.cloud,
            "severity": self.severity.name,
            "statistics": {key: round(value, 4) for key, value in self.statistics.items()},
            "hypotheses": [item.to_dict() for item in self.hypotheses],
        }


@dataclass(frozen=True, slots=True)
class SignalRule:
    """Maps a set of simultaneously breaching metric names to a likely cause."""

    signals: tuple[str, ...]
    cause: str
    category: str
    weight: float
    evidence: str


#: Cross-signal correlation rules, most specific first.
SIGNAL_RULES: tuple[SignalRule, ...] = (
    SignalRule(
        signals=("latency_ms", "error_rate"),
        cause="Downstream dependency failure",
        category="dependency",
        weight=3.0,
        evidence="latency and error rate are elevated together, which points at a dependency rather than capacity",
    ),
    SignalRule(
        signals=("response_time_ms", "error_rate"),
        cause="Downstream dependency failure",
        category="dependency",
        weight=3.0,
        evidence="response time and error rate rise together - inspect upstream calls",
    ),
    SignalRule(
        signals=("query_latency_ms", "error_rate"),
        cause="Database degradation",
        category="data",
        weight=3.0,
        evidence="query latency and errors co-occur - check locks, plan regressions and connection pools",
    ),
    SignalRule(
        signals=("cpu_utilization", "memory_utilization"),
        cause="Resource exhaustion",
        category="capacity",
        weight=2.4,
        evidence="CPU and memory are both saturated, which usually means a leak or a runaway workload",
    ),
    SignalRule(
        signals=("request_rate", "latency_ms"),
        cause="Traffic surge beyond capacity",
        category="capacity",
        weight=2.2,
        evidence="request rate and latency rise together - autoscaling may be lagging",
    ),
    SignalRule(
        signals=("request_rate", "response_time_ms"),
        cause="Traffic surge beyond capacity",
        category="capacity",
        weight=2.2,
        evidence="inbound traffic and response time rise together",
    ),
    SignalRule(
        signals=("network_io_mbps", "latency_ms"),
        cause="Network path saturation",
        category="network",
        weight=2.0,
        evidence="network throughput and latency moved together - suspect cross-AZ or egress limits",
    ),
)

#: Single-metric causes, keyed by metric name.
METRIC_CAUSES: dict[str, tuple[tuple[str, str, float], ...]] = {
    "cpu_utilization": (
        ("CPU saturation", "capacity", 2.6),
        ("Hot path / runaway process", "workload", 2.0),
        ("Throttling from quota limits", "quota", 1.4),
    ),
    "memory_utilization": (
        ("Memory leak or unbounded cache growth", "workload", 2.6),
        ("Heap pressure / GC thrash", "runtime", 2.0),
        ("Instance undersized for the workload", "capacity", 1.5),
    ),
    "latency_ms": (
        ("Downstream dependency latency", "dependency", 2.4),
        ("Connection pool saturation", "dependency", 2.0),
        ("Garbage collection pause", "runtime", 1.5),
    ),
    "error_rate": (
        ("Failed deployment or configuration change", "deployment", 2.6),
        ("Upstream dependency returning errors", "dependency", 2.2),
        ("Input validation / client error surge", "client", 1.4),
    ),
    "disk_io_utilization": (
        ("Log volume explosion or unbounded write", "storage", 2.4),
        ("Backup or batch job competing for IO", "storage", 2.0),
        ("Filesystem checkpoint churn", "storage", 1.4),
    ),
    "request_rate": (
        ("Traffic anomaly (attack or misrouted client)", "traffic", 2.4),
        ("Cache miss storm", "traffic", 1.8),
        ("Legitimate load increase", "traffic", 1.2),
    ),
    "response_time_ms": (
        ("Downstream service slowdown", "dependency", 2.4),
        ("Cold start / scale-out events", "runtime", 1.8),
        ("Network hop added to the path", "network", 1.4),
    ),
    "network_io_mbps": (
        ("Unexpected egress or data transfer job", "network", 2.4),
        ("Cross-AZ traffic shift", "network", 1.8),
        ("Load balancer hotspot", "network", 1.2),
    ),
    "query_latency_ms": (
        ("Slow query or missing index after plan change", "data", 2.6),
        ("Lock contention", "data", 2.0),
        ("Database CPU saturation", "capacity", 1.6),
    ),
}

_DEFAULT_CAUSES: tuple[tuple[str, str, float], ...] = (("Unclassified metric anomaly", "unknown", 1.0),)


def _statistics(metric: Metric) -> dict[str, float]:
    stats: dict[str, float] = {
        "value": metric.value,
        "confidence": metric.confidence,
    }
    window = metric.window
    if window is not None and window.count > 1:
        stats["z_score"] = window.z_score(metric.value)
        stats["ratio_to_mean"] = window.ratio(metric.value)
        stats["pct_change"] = window.pct_change
        stats["rolling_mean"] = window.mean
        stats["samples"] = float(window.count)
    return stats


def _statistical_evidence(metric: Metric) -> list[str]:
    evidence: list[str] = []
    window = metric.window
    if window is None or window.count < 2:
        return evidence
    z_score = window.z_score(metric.value)
    if abs(z_score) >= 1.0:
        evidence.append(f"{z_score:+.1f} sigma from the {window.count}-sample mean of {window.mean:.2f}")
    ratio = window.ratio(metric.value)
    if ratio >= 1.25:
        evidence.append(f"{ratio:.2f}x its rolling average")
    if abs(window.pct_change) >= 25.0:
        direction = "up" if window.pct_change > 0 else "down"
        evidence.append(f"{direction} {abs(window.pct_change):.0f}% versus the previous sample")
    return evidence


def _breached_signals(metric: Metric, violations: Sequence[Any], context: Sequence[Metric]) -> set[str]:
    signals = {metric.name}
    for violation in violations:
        signals.add(getattr(violation, "metric_name", ""))
    for sibling in context:
        if sibling.series_key == metric.series_key:
            continue
        if sibling.resource_id != metric.resource_id and sibling.service != metric.service:
            continue
        if sibling.is_anomaly:
            signals.add(sibling.name)
    signals.discard("")
    return signals


def analyze(
    metric: Metric,
    violations: Sequence[Any] = (),
    context: Sequence[Metric] = (),
) -> RootCause:
    """Rank root-cause hypotheses for a breaching metric."""
    signals = _breached_signals(metric, violations, context)
    stat_evidence = _statistical_evidence(metric)
    stats = _statistics(metric)

    raw: dict[tuple[str, str], tuple[float, list[str]]] = {}

    def add(cause: str, category: str, weight: float, evidence: Sequence[str]) -> None:
        key = (cause, category)
        existing = raw.get(key)
        if existing is None:
            raw[key] = (float(weight), [str(item) for item in evidence])
            return
        combined = list(existing[1])
        for item in evidence:
            text = str(item)
            if text not in combined:
                combined.append(text)
        raw[key] = (existing[0] + 0.25 * float(weight), combined)

    for cause, category, weight in METRIC_CAUSES.get(metric.name, _DEFAULT_CAUSES):
        add(cause, category, weight, ())

    for rule in SIGNAL_RULES:
        # a correlation rule only explains *this* metric when the metric is part
        # of the correlated set; otherwise it is context, not a cause.
        if metric.name in rule.signals and set(rule.signals) <= signals:
            add(rule.cause, rule.category, rule.weight, (rule.evidence,))

    for violation in violations:
        message = getattr(violation, "message", "")
        rule_id = getattr(violation, "rule_id", "")
        severity = getattr(violation, "severity", None)
        if message:
            add(
                f"Threshold breach: {rule_id}",
                "threshold",
                1.2 + 0.4 * int(severity or Severity.LOW) / 4.0,
                (f"{message} (severity {getattr(severity, 'name', 'LOW')})",),
            )

    if metric.confidence > 0:
        add(
            "Pattern unseen during training",
            "anomaly",
            1.0 + 1.8 * metric.confidence,
            (f"model confidence {metric.confidence:.0%} for an untrained pattern",),
        )

    total = sum(weight for weight, _ in raw.values()) or 1.0
    hypotheses = tuple(
        Hypothesis(cause=cause, category=category, likelihood=weight / total, evidence=tuple(evidence))
        for (cause, category), (weight, evidence) in sorted(
            raw.items(), key=lambda item: (-item[1][0], item[0][0])
        )
    )

    severity = Severity.LOW
    for violation in violations:
        candidate = getattr(violation, "severity", None)
        if isinstance(candidate, Severity) and candidate > severity:
            severity = candidate

    enriched: list[Hypothesis] = []
    for hypothesis in hypotheses:
        evidence = list(hypothesis.evidence)
        if not evidence and stat_evidence:
            evidence = list(stat_evidence)
        enriched.append(
            Hypothesis(
                cause=hypothesis.cause,
                category=hypothesis.category,
                likelihood=hypothesis.likelihood,
                evidence=tuple(evidence),
            )
        )

    return RootCause(
        series_key=metric.series_key,
        metric_name=metric.name,
        cloud=metric.cloud.value,
        severity=severity,
        hypotheses=tuple(enriched),
        statistics=stats,
    )


def generate_hints(
    metric: Metric,
    violations: Sequence[Any] = (),
    context: Sequence[Metric] = (),
    *,
    limit: int = 3,
) -> list[str]:
    """Flat, human-readable root-cause hints (dashboard, Slack, email)."""
    root_cause = analyze(metric, violations, context)
    hints: list[str] = []
    for hypothesis in root_cause.top(limit):
        if hypothesis.evidence:
            hints.append(f"{hypothesis.cause} ({hypothesis.likelihood:.0%}): {hypothesis.evidence[0]}")
        else:
            hints.append(f"{hypothesis.cause} ({hypothesis.likelihood:.0%})")
    return hints
