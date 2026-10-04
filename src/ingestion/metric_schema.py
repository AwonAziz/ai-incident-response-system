"""Unified metric data model.

Every collector - simulated or real - emits :class:`Metric` objects so the rest
of the pipeline never has to care where telemetry came from. Detection results
are attached to the same object (``anomaly_score`` / ``is_anomaly`` /
``confidence``) rather than returned out of band, which keeps the collector ->
rule engine -> triage call chain free of parallel structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from src.core.clock import utc_now
from src.core.enums import Cloud
from src.core.stats import RollingStats

__all__ = ["Metric", "MetricSpec", "ResourceSpec"]


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    """A monitored resource declared in ``cloud_profiles.yaml``."""

    id: str
    service: str
    region: str
    tags: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceSpec:
        return cls(
            id=str(data.get("id", "unknown")),
            service=str(data.get("service", "unknown")),
            region=str(data.get("region", "unknown")),
            tags={str(k): str(v) for k, v in (data.get("tags") or {}).items()},
        )


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Threshold/normalisation metadata for one metric name."""

    name: str
    label: str
    unit: str
    normal_low: float
    normal_high: float
    warn_high: float
    critical_high: float
    warn_low: float | None = None
    critical_low: float | None = None
    weight: float = 1.0
    spike_z: float | None = None

    @property
    def midpoint(self) -> float:
        return (self.normal_low + self.normal_high) / 2.0

    @property
    def amplitude(self) -> float:
        return max(1e-6, (self.normal_high - self.normal_low) / 2.0)


@dataclass(slots=True)
class Metric:
    """A single telemetry sample plus its detection annotations."""

    name: str
    value: float
    unit: str = ""
    timestamp: datetime = field(default_factory=utc_now)
    cloud: Cloud = Cloud.AWS
    resource_id: str = ""
    service: str = ""
    region: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    # filled in by the collector that owns the rolling window
    window: RollingStats | None = None

    # filled in by the detector
    anomaly_score: float | None = None
    is_anomaly: bool = False
    confidence: float = 0.0
    model_version: str | None = None

    def __post_init__(self) -> None:
        self.cloud = Cloud.parse(self.cloud)
        self.value = float(self.value)
        self.unit = self.unit or ""
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=utc_now().tzinfo)
        if not self.tags:
            self.tags = {}

    # ── identity ───────────────────────────────────────────────────────
    @property
    def series_key(self) -> str:
        """Stable key for the rolling window / feature history of this series."""
        return f"{self.cloud.value}:{self.resource_id}:{self.name}"

    @property
    def scope_key(self) -> str:
        """Key used for incident deduplication (service level, not instance)."""
        return f"{self.cloud.value}:{self.service}:{self.name}"

    @property
    def history_ready(self) -> bool:
        return self.window is not None and self.window.count > 0

    @property
    def z_score(self) -> float:
        return self.window.z_score(self.value) if self.window is not None else 0.0

    # ── transforms ─────────────────────────────────────────────────────
    def with_detection(
        self,
        *,
        anomaly_score: float | None = None,
        is_anomaly: bool = False,
        confidence: float = 0.0,
        model_version: str | None = None,
    ) -> Metric:
        """Return a copy carrying detection annotations."""
        return replace(
            self,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            confidence=confidence,
            model_version=model_version,
        )

    def describe(self) -> str:
        parts = [f"{self.cloud.value}/{self.service}", f"{self.resource_id}", f"{self.name}={self.value:.2f}{self.unit}"]
        if self.window is not None and self.window.count > 1:
            parts.append(f"z={self.z_score:+.2f}")
        if self.is_anomaly:
            parts.append(f"anomaly(conf={self.confidence:.2f})")
        return " ".join(parts)

    # ── serialisation ──────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "cloud": self.cloud.value,
            "resource_id": self.resource_id,
            "service": self.service,
            "region": self.region,
            "tags": dict(self.tags),
            "window": self.window.to_dict() if self.window is not None else None,
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "confidence": self.confidence,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metric:
        timestamp = data.get("timestamp")
        parsed = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else utc_now()
        window_data = data.get("window")
        window = None
        if isinstance(window_data, dict):
            fields = {
                key: float(value)
                for key, value in window_data.items()
                if key in RollingStats.__slots__ and value is not None
            }
            count = int(window_data.get("count", 0))
            window = RollingStats(count=count, **{key: value for key, value in fields.items() if key != "count"})
        return cls(
            name=str(data["name"]),
            value=float(data["value"]),
            unit=str(data.get("unit", "")),
            timestamp=parsed,
            cloud=Cloud.parse(str(data.get("cloud", "aws"))),
            resource_id=str(data.get("resource_id", "")),
            service=str(data.get("service", "")),
            region=str(data.get("region", "")),
            tags={str(k): str(v) for k, v in (data.get("tags") or {}).items()},
            window=window,
            anomaly_score=data.get("anomaly_score"),
            is_anomaly=bool(data.get("is_anomaly", False)),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            model_version=data.get("model_version"),
        )
