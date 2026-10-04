"""Replay a real time series through the live pipeline.

:class:`ReplayCollector` is a :class:`~src.ingestion.base_collector.BaseCollector`
like the simulators, so a real series exercises exactly the code the production
path uses: ``Metric`` construction, the shared rolling window, feature
engineering, rule evaluation, triage. It additionally exposes the ground-truth
label for the point it just emitted, which is what makes evaluation possible.

Labels are ``1`` inside a benchmark anomaly window and ``0`` elsewhere; the
``-1`` label exists to say "unknown" and is never produced by :func:`relabel`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from src.core.clock import Clock, SystemClock
from src.core.enums import Cloud
from src.data.timeseries import ANOMALOUS, NORMAL, LabelledSeries
from src.ingestion.base_collector import BaseCollector
from src.ingestion.metric_schema import Metric

__all__ = ["ReplayCollector"]


class ReplayCollector(BaseCollector):
    """Emit the points of a :class:`LabelledSeries`, in order, one tick each."""

    cloud: ClassVar[Cloud] = Cloud.AWS

    def __init__(
        self,
        series: LabelledSeries,
        *,
        history=None,
        profiles: dict[str, Any] | None = None,
        clock: Clock | None = None,
        start: int = 0,
        stop: int | None = None,
    ) -> None:
        if profiles is None:
            from config.settings import CLOUD_PROFILES

            profiles = CLOUD_PROFILES
        super().__init__(
            inject_anomaly=False,
            seed=0,
            history=history,
            profiles=profiles,
            clock=clock or SystemClock(),
        )
        self.series = series
        self.cloud = series.cloud
        self._start = max(0, int(start))
        self._stop = len(series) if stop is None else min(int(stop), len(series))
        self._index = self._start
        self.last_label: int | None = None
        self.emitted = 0

    # ── state ──────────────────────────────────────────────────────────
    @property
    def index(self) -> int:
        """Position of the next point to emit."""
        return self._index

    @property
    def total(self) -> int:
        return self._stop - self._start

    @property
    def remaining(self) -> int:
        return max(0, self._stop - self._index)

    @property
    def exhausted(self) -> bool:
        return self._index >= self._stop

    @property
    def metrics(self) -> Sequence[str]:
        return (self.series.metric,)

    def labels(self) -> tuple[int, ...]:
        """Ground-truth labels for the replayed slice."""
        return tuple(self.series.labels[self._start : self._stop])

    # ── BaseCollector contract ─────────────────────────────────────────
    def collect(self) -> list[Metric]:
        if self.exhausted:
            return []
        position = self._index
        self._index += 1
        self.emitted += 1
        label = self.series.labels[position]
        self.last_label = label
        return [
            Metric(
                name=self.series.metric,
                value=self.series.values[position],
                unit=self.series.unit,
                timestamp=self.series.timestamps[position],
                cloud=self.series.cloud,
                resource_id=self.series.resource_id,
                service=self.series.service,
                region=self.series.region,
                tags={"dataset": self.series.name, "label": str(label)},
            )
        ]

    def recent_values(self, series_key: str) -> tuple[float, ...]:
        return self.history.values(series_key)

    # ── helpers for evaluation ─────────────────────────────────────────
    @staticmethod
    def is_anomalous(label: int | None) -> bool:
        return label == ANOMALOUS

    @staticmethod
    def is_normal(label: int | None) -> bool:
        return label == NORMAL

    def label_at(self, position: int) -> int:
        return self.series.labels[position]

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<ReplayCollector {self.series.name} {self._index}/{self._stop}>"
