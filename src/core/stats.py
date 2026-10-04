"""Rolling window statistics for a single metric series."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, sqrt

__all__ = ["RollingStats", "stats_from_values"]

_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class RollingStats:
    """Summary of the most recent samples of one metric series.

    Attributes are populated even for very short windows so downstream code can
    decide whether the sample count is trustworthy via :meth:`is_ready`.
    """

    count: int
    mean: float
    std: float
    minimum: float
    maximum: float
    last: float
    delta: float
    pct_change: float
    span_seconds: float

    @property
    def variance(self) -> float:
        return self.std * self.std

    @property
    def coefficient_of_variation(self) -> float:
        if abs(self.mean) < _EPSILON:
            return 0.0
        return self.std / abs(self.mean)

    def is_ready(self, min_samples: int) -> bool:
        return self.count >= max(1, min_samples)

    def z_score(self, value: float | None = None) -> float:
        """Deviation of ``value`` (default: latest) in standard deviations."""
        target = self.last if value is None else value
        if self.std < _EPSILON or self.count < 2:
            return 0.0
        return (target - self.mean) / self.std

    def ratio(self, value: float | None = None) -> float:
        """``value`` divided by the rolling mean (guarded against ~0 means)."""
        target = self.last if value is None else value
        if abs(self.mean) < _EPSILON:
            return 0.0
        return target / self.mean

    def to_dict(self) -> dict[str, float | int]:
        data = asdict(self)
        data["z_score"] = self.z_score()
        data["coefficient_of_variation"] = self.coefficient_of_variation
        return data


def stats_from_values(values: list[float], span_seconds: float = 0.0) -> RollingStats:
    """Compute :class:`RollingStats` from a list of samples (last sample wins)."""
    if not values:
        return RollingStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    count = len(values)
    total = 0.0
    for item in values:
        total += item
    mean = total / count

    variance = 0.0
    if count > 1:
        accumulator = 0.0
        for item in values:
            deviation = item - mean
            accumulator += deviation * deviation
        variance = accumulator / (count - 1)

    std = sqrt(variance)
    if not isfinite(std) or std < _EPSILON:
        std = 0.0

    last = values[-1]
    previous = values[-2] if count > 1 else last
    delta = last - previous
    pct_change = 0.0 if abs(previous) < _EPSILON else delta / abs(previous) * 100.0

    return RollingStats(
        count=count,
        mean=mean,
        std=std,
        minimum=min(values),
        maximum=max(values),
        last=last,
        delta=delta,
        pct_change=pct_change,
        span_seconds=span_seconds,
    )
