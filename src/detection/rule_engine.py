"""Static threshold rules.

The ML model catches *unusual* shapes; the rule engine catches *known bad*
absolute values. Both feed triage, which is why an incident can be raised by
either or by both (the hybrid in the README is literal, not aspirational).

Every threshold comes from ``cloud_profiles.yaml`` and may be overridden per
cloud (``clouds.aws.overrides.error_rate.critical_high``), which is how a
payments workload gets a stricter error budget than a batch job.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from src.core.enums import Cloud, Severity
from src.ingestion.metric_schema import Metric

__all__ = ["RuleEngine", "RuleViolation"]


@dataclass(frozen=True, slots=True)
class RuleViolation:
    """A single threshold breach."""

    rule_id: str
    metric_name: str
    cloud: Cloud
    resource_id: str
    service: str
    severity: Severity
    direction: str  # "high" | "low" | "spike"
    threshold: float
    value: float
    message: str
    weight: float = 1.0

    @property
    def exceeded_by(self) -> float:
        if abs(self.threshold) < 1e-9:
            return 0.0
        return (self.value - self.threshold) / abs(self.threshold) * 100.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cloud"] = self.cloud.value
        data["severity"] = self.severity.name
        data["exceeded_by_percent"] = round(self.exceeded_by, 3)
        return data

    def describe(self) -> str:
        return f"[{self.severity.name}] {self.rule_id}: {self.message}"


class RuleEngine:
    """Evaluates a metric against its profile thresholds."""

    def __init__(self, profiles: dict[str, Any] | None = None, *, spike_multiplier: float = 1.0) -> None:
        if profiles is None:
            from config.settings import CLOUD_PROFILES

            profiles = CLOUD_PROFILES
        self.profiles = profiles
        self.spike_multiplier = float(spike_multiplier)
        self._rules = self._compile_rules()

    # ── public API ─────────────────────────────────────────────────────
    def evaluate(self, metric: Metric) -> list[RuleViolation]:
        """All threshold breaches for one metric sample (worst first)."""
        rules = self._rules.get((metric.cloud, metric.name))
        if not rules:
            return []
        violations: list[RuleViolation] = []
        for rule in rules:
            violation = self._apply(rule, metric)
            if violation is not None:
                violations.append(violation)
        violations.sort(key=lambda item: (-int(item.severity), -item.exceeded_by))
        return violations

    def evaluate_batch(self, metrics: Iterable[Metric]) -> dict[str, list[RuleViolation]]:
        """Map of series key -> violations for a whole batch."""
        result: dict[str, list[RuleViolation]] = {}
        for metric in metrics:
            found = self.evaluate(metric)
            if found:
                result[metric.series_key] = found
        return result

    def rules_for(self, cloud: Cloud | str, metric_name: str) -> list[dict[str, Any]]:
        """Compiled rules for a cloud/metric pair (used by the API and tests)."""
        return list(self._rules.get((Cloud.parse(cloud), metric_name), []))

    @property
    def known_metrics(self) -> tuple[str, ...]:
        return tuple(sorted({name for _, name in self._rules}))

    # ── compilation ────────────────────────────────────────────────────
    def _compile_rules(self) -> dict[tuple[Cloud, str], list[dict[str, Any]]]:
        compiled: dict[tuple[Cloud, str], list[dict[str, Any]]] = {}
        clouds = self.profiles.get("clouds", {}) or {}
        for cloud_key, cloud_profile in clouds.items():
            try:
                cloud = Cloud.parse(cloud_key)
            except ValueError:  # pragma: no cover - defensive
                continue
            overrides = (cloud_profile.get("overrides") or {}) if isinstance(cloud_profile, dict) else {}
            metric_names = list((cloud_profile or {}).get("metrics") or self.profiles.get("metrics", {}))
            for metric_name in metric_names:
                base = dict(self.profiles.get("metrics", {}).get(metric_name) or {})
                base.update(overrides.get(metric_name) or {})
                if not base:
                    continue
                compiled[(cloud, metric_name)] = self._rules_for_metric(metric_name, base)
        return compiled

    def _rules_for_metric(self, metric_name: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
        label = str(spec.get("label", metric_name))
        unit = str(spec.get("unit", ""))
        weight = float(spec.get("weight", 1.0))
        rules: list[dict[str, Any]] = []

        critical_high = spec.get("critical_high")
        warn_high = spec.get("warn_high")
        if critical_high is not None:
            rules.append(
                {
                    "id": f"{metric_name}.critical_high",
                    "direction": "high",
                    "threshold": float(critical_high),
                    "severity": Severity.CRITICAL,
                    "weight": weight,
                    "message": f"{label} {critical_high}{unit} is critical (breached)",
                }
            )
        if warn_high is not None:
            rules.append(
                {
                    "id": f"{metric_name}.warn_high",
                    "direction": "high",
                    "threshold": float(warn_high),
                    "severity": Severity.HIGH,
                    "weight": weight,
                    "message": f"{label} above {warn_high}{unit}",
                }
            )

        critical_low = spec.get("critical_low")
        warn_low = spec.get("warn_low")
        if critical_low is not None:
            rules.append(
                {
                    "id": f"{metric_name}.critical_low",
                    "direction": "low",
                    "threshold": float(critical_low),
                    "severity": Severity.CRITICAL,
                    "weight": weight,
                    "message": f"{label} below {critical_low}{unit} is critical (starved)",
                }
            )
        if warn_low is not None:
            rules.append(
                {
                    "id": f"{metric_name}.warn_low",
                    "direction": "low",
                    "threshold": float(warn_low),
                    "severity": Severity.HIGH,
                    "weight": weight,
                    "message": f"{label} below {warn_low}{unit}",
                }
            )

        normal = spec.get("normal") or []
        if len(normal) == 2:
            rules.append(
                {
                    "id": f"{metric_name}.outside_normal_band",
                    "direction": "band",
                    "low": float(normal[0]),
                    "high": float(normal[1]),
                    "severity": Severity.MEDIUM,
                    "weight": weight,
                    "message": f"{label} outside expected {normal[0]}-{normal[1]}{unit}",
                }
            )

        spike_z = spec.get("spike_z")
        if spike_z is not None:
            rules.append(
                {
                    "id": f"{metric_name}.zscore_spike",
                    "direction": "spike",
                    "threshold": float(spike_z) * self.spike_multiplier,
                    "severity": Severity.MEDIUM,
                    "weight": weight,
                    "message": f"{label} deviates more than {float(spike_z) * self.spike_multiplier:g} sigma",
                }
            )
        return rules

    def _apply(self, rule: dict[str, Any], metric: Metric) -> RuleViolation | None:
        direction = rule["direction"]
        value = metric.value
        threshold = float(rule.get("threshold", 0.0))

        if direction == "high":
            if value < rule["threshold"]:
                return None
        elif direction == "low":
            if value > rule["threshold"]:
                return None
        elif direction == "band":
            if rule["low"] <= value <= rule["high"]:
                return None
            threshold = rule["low"] if value < rule["low"] else rule["high"]
        elif direction == "spike":
            window = metric.window
            if window is None or not window.is_ready(5):
                return None
            if abs(window.z_score(value)) < rule["threshold"]:
                return None
            threshold = rule["threshold"]
        else:  # pragma: no cover - defensive
            return None

        return RuleViolation(
            rule_id=str(rule["id"]),
            metric_name=metric.name,
            cloud=metric.cloud,
            resource_id=metric.resource_id,
            service=metric.service,
            severity=rule["severity"],
            direction=direction,
            threshold=float(threshold),
            value=float(value),
            message=str(rule["message"]),
            weight=float(rule.get("weight", 1.0)),
        )
