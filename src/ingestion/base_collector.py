"""Collector interface and the telemetry simulator.

:class:`BaseCollector` defines the contract every collector honours
(``collect`` / ``collect_and_track``). :class:`SimulatedCollector` implements a
deterministic, seeded telemetry generator on top of the YAML profiles, so the
per-cloud subclasses stay declarative and a real cloud SDK collector can be
dropped in later without touching the pipeline.

Simulation model, per ``(resource, metric)`` series:

* a slow diurnal sine wave over the tick counter (load is not constant),
* gaussian noise proportional to the metric's normal band,
* optional "degradation" episodes where a whole resource is pushed out of band
  for a few ticks, which is what auto-resolution and dedup logic is tested on.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import pi, sin
from typing import Any, ClassVar
from zlib import crc32

from config.settings import CLOUD_PROFILES
from src.core.clock import Clock, SystemClock
from src.core.enums import Cloud
from src.core.stats import RollingStats
from src.ingestion.history import SeriesHistory
from src.ingestion.metric_schema import Metric, MetricSpec, ResourceSpec

__all__ = ["BaseCollector", "SimulatedCollector"]

WAVE_PERIOD_TICKS = 24.0


class BaseCollector(ABC):
    """Abstract telemetry collector."""

    cloud: ClassVar[Cloud]

    def __init__(
        self,
        *,
        inject_anomaly: bool = False,
        seed: int | None = None,
        history: SeriesHistory | None = None,
        profiles: dict[str, Any] | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.profiles = profiles if profiles is not None else CLOUD_PROFILES
        defaults = self.profiles.get("defaults", {})
        # an empty SeriesHistory is falsy (len == 0) so this must be an identity check
        self.history = history if history is not None else SeriesHistory(
            window=int(defaults.get("history_window", 30)),
            min_samples=int(defaults.get("min_history", 5)),
        )
        self.clock: Clock = clock or SystemClock()
        self.inject_anomaly = bool(inject_anomaly)
        self.ticks = 0
        self.seed = seed
        self._rng = random.Random(self._resolve_seed(seed))

    @staticmethod
    def _resolve_seed(seed: int | None) -> int:
        return 0 if seed is None else int(seed)

    @abstractmethod
    def collect(self) -> list[Metric]:
        """Produce one batch of metrics without touching the rolling window."""

    def collect_and_track(self) -> list[Metric]:
        """Produce a batch and attach rolling statistics to every metric."""
        metrics = self.collect()
        for metric in metrics:
            metric.window = self.record(metric)
        self.ticks += 1
        return metrics

    def record(self, metric: Metric) -> RollingStats:
        return self.history.record(metric.series_key, metric.value, metric.timestamp)

    def recent_values(self, series_key: str) -> tuple[float, ...]:
        return self.history.values(series_key)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<{type(self).__name__} cloud={self.cloud.value} ticks={self.ticks}>"


class SimulatedCollector(BaseCollector):
    """Seeded metric simulator driven by ``cloud_profiles.yaml``."""

    #: how often a new degradation episode starts (ticks), anomaly mode only
    anomaly_period: ClassVar[int] = 6
    #: how many ticks a degradation episode lasts
    anomaly_duration: ClassVar[tuple[int, int]] = (1, 3)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._degraded: dict[str, int] = {}
        self._last_injection_tick: int = -self.anomaly_period
        self.injected_series: list[str] = []
        self._specs = self._metric_specs()
        self._resources = self._resource_specs()

    # ── profile access ─────────────────────────────────────────────────
    def _cloud_profile(self) -> dict[str, Any]:
        return dict(self.profiles.get("clouds", {}).get(self.cloud.value, {}))

    def display_name(self) -> str:
        return str(self._cloud_profile().get("display_name", self.cloud.label))

    def _metric_names(self) -> Sequence[str]:
        names = self._cloud_profile().get("metrics") or list(self.profiles.get("metrics", {}))
        return [str(name) for name in names]

    def _metric_specs(self) -> dict[str, MetricSpec]:
        specs: dict[str, MetricSpec] = {}
        overrides = self._cloud_profile().get("overrides", {}) or {}
        for name in self._metric_names():
            base = dict(self.profiles.get("metrics", {}).get(name, {}))
            base.update(overrides.get(name, {}) or {})
            if not base:
                continue
            specs[name] = MetricSpec(
                name=name,
                label=str(base.get("label", name)),
                unit=str(base.get("unit", "")),
                normal_low=float(base.get("normal", [0.0, 1.0])[0]),
                normal_high=float(base.get("normal", [0.0, 1.0])[1]),
                warn_high=float(base.get("warn_high", float("inf"))),
                critical_high=float(base.get("critical_high", float("inf"))),
                warn_low=(float(base["warn_low"]) if base.get("warn_low") is not None else None),
                critical_low=(float(base["critical_low"]) if base.get("critical_low") is not None else None),
                weight=float(base.get("weight", 1.0)),
                spike_z=(float(base["spike_z"]) if base.get("spike_z") is not None else None),
            )
        return specs

    def _resource_specs(self) -> list[ResourceSpec]:
        raw = self._cloud_profile().get("resources", []) or []
        return [ResourceSpec.from_dict(item) for item in raw]

    @property
    def resources(self) -> list[ResourceSpec]:
        return list(self._resources)

    @property
    def specs(self) -> dict[str, MetricSpec]:
        return dict(self._specs)

    # ── simulation ─────────────────────────────────────────────────────
    def collect(self) -> list[Metric]:
        now = self.clock.now()
        if self.inject_anomaly:
            self._maybe_start_degradation()

        metrics: list[Metric] = []
        for resource in self._resources:
            remaining = self._degraded.get(resource.id, 0)
            for name in self._metric_names():
                spec = self._specs.get(name)
                if spec is None:
                    continue
                value = self._sample(spec, resource, degraded=remaining > 0)
                metrics.append(
                    Metric(
                        name=spec.name,
                        value=round(value, 4),
                        unit=spec.unit,
                        timestamp=now,
                        cloud=self.cloud,
                        resource_id=resource.id,
                        service=resource.service,
                        region=resource.region,
                        tags=dict(resource.tags),
                    )
                )
        self.advance_degradations()
        return metrics

    def _sample(self, spec: MetricSpec, resource: ResourceSpec, *, degraded: bool) -> float:
        centre = spec.midpoint
        amplitude = spec.amplitude
        phase = self._phase(resource.id, spec.name)
        wave = amplitude * 0.3 * sin(2 * pi * self.ticks / WAVE_PERIOD_TICKS + phase)
        noise_sigma = max(amplitude * 0.14, abs(centre) * 0.04)
        value = centre + wave + self._rng.gauss(0.0, noise_sigma)

        if degraded:
            headroom = max(spec.critical_high - centre, amplitude)
            value += self._rng.uniform(0.55, 1.25) * headroom

        ceiling = max(spec.critical_high * 1.35, spec.normal_high * 1.35)
        floor = max(0.0, spec.normal_low * 0.4)
        return max(floor, min(ceiling, value))

    def _phase(self, resource_id: str, metric_name: str) -> float:
        digest = crc32(f"{resource_id}|{metric_name}|{self.seed}".encode())
        return (digest % 1000) / 1000.0 * 2 * pi

    def _maybe_start_degradation(self) -> None:
        if self.ticks - self._last_injection_tick < self.anomaly_period:
            return
        self._last_injection_tick = self.ticks
        if not self._resources:
            return
        target = self._rng.choice(self._resources)
        low, high = self.anomaly_duration
        self._degraded[target.id] = self._rng.randint(low, high)
        self.injected_series.append(f"{self.cloud.value}:{target.id}")

    def advance_degradations(self) -> None:
        """Age every active degradation episode by one tick."""
        for resource_id, remaining in list(self._degraded.items()):
            if remaining <= 1:
                del self._degraded[resource_id]
            else:
                self._degraded[resource_id] = remaining - 1

    def clear_injections(self) -> None:
        self.injected_series.clear()

    def degraded_resources(self) -> tuple[str, ...]:
        return tuple(self._degraded)
