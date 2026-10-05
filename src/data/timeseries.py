"""Real time-series data for evaluation.

The pipeline itself is verified against a seeded simulator, which proves the
plumbing but says nothing about whether the detection approach works. This
module loads **real** telemetry with published ground truth so the approach can
be measured.

Dataset: Numenta's Anomaly Benchmark (NAB), ``realAWSCloudwatch`` subset - 17
real AWS CloudWatch series (EC2/RDS CPU utilisation, disk write bytes, network
in, ELB request count) with anomaly windows defined by the benchmark authors.
Series CSVs hold ``timestamp,value``; the labels live in
``labels/combined_windows.json`` keyed by ``<dataset>/<file>.csv``.

Two rules are enforced here so results stay honest:

* **The split is temporal.** ``split_time`` cuts a series once, by position.
  A random split would let the model learn from the future.
* **Anything derived from data comes from the training segment only.**
  Expected bands (``derive_bands``) use training quantiles, never the full
  series.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.core.enums import Cloud

__all__ = [
    "METRIC_NAMES",
    "NAB_BRANCH",
    "NAB_DATASET",
    "NAB_REPO",
    "UNIT_LABELS",
    "AnomalyWindow",
    "LabelledSeries",
    "derive_bands",
    "load_nab_dataset",
    "load_nab_windows",
    "load_series",
    "parse_timestamp",
    "split_time",
]

NAB_REPO = "numenta/NAB"
NAB_BRANCH = "master"
NAB_DATASET = "realAWSCloudwatch"
TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")

#: NAB file stem prefix -> unified metric name (matches the simulated profiles)
METRIC_NAMES: dict[str, str] = {
    "ec2_cpu_utilization": "cpu_utilization",
    "rds_cpu_utilization": "cpu_utilization",
    "ec2_disk_write_bytes": "disk_write_bytes",
    "ec2_network_in": "network_in_bytes",
    "networkin": "network_in_bytes",
    "elb_request_count": "request_count",
    "grok_asg": "grok_request_count",
}

#: display units only - every ML feature is dimensionless
UNIT_LABELS: dict[str, str] = {
    "cpu_utilization": "%",
    "disk_write_bytes": "bytes/s",
    "network_in_bytes": "bytes/s",
    "request_count": "req/s",
    "grok_request_count": "req/s",
}

ANOMALOUS = 1
NORMAL = 0


@dataclass(frozen=True, slots=True)
class AnomalyWindow:
    """A labelled interval in which the benchmark considers the series anomalous."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"window ends before it starts: {self.start} -> {self.end}")

    def contains(self, moment: datetime, pad: timedelta = timedelta(0)) -> bool:
        return (self.start - pad) <= moment <= (self.end + pad)

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    def to_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}

    @classmethod
    def from_strings(cls, start: str, end: str) -> AnomalyWindow:
        return cls(parse_timestamp(start), parse_timestamp(end))


@dataclass(frozen=True, slots=True)
class LabelledSeries:
    """One real time series with per-point anomaly labels."""

    name: str
    metric: str
    timestamps: tuple[datetime, ...]
    values: tuple[float, ...]
    labels: tuple[int, ...]
    windows: tuple[AnomalyWindow, ...] = ()
    resource_id: str = ""
    service: str = "EC2"
    region: str = "unknown"
    cloud: Cloud = Cloud.AWS
    unit: str = ""
    source: str = ""
    dropped_points: int = 0

    def __post_init__(self) -> None:
        if not (len(self.timestamps) == len(self.values) == len(self.labels)):
            raise ValueError(
                f"{self.name}: ragged series "
                f"({len(self.timestamps)} ts / {len(self.values)} values / {len(self.labels)} labels)"
            )

    # ── basics ─────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[tuple[datetime, float, int]]:
        return iter(zip(self.timestamps, self.values, self.labels, strict=True))

    @property
    def size(self) -> int:
        return len(self.values)

    @property
    def step(self) -> timedelta | None:
        if len(self.timestamps) < 2:
            return None
        return self.timestamps[1] - self.timestamps[0]

    @property
    def step_seconds(self) -> float:
        step = self.step
        return step.total_seconds() if step else 0.0

    @property
    def start(self) -> datetime | None:
        return self.timestamps[0] if self.timestamps else None

    @property
    def end(self) -> datetime | None:
        return self.timestamps[-1] if self.timestamps else None

    @property
    def span(self) -> timedelta:
        if not self.timestamps:
            return timedelta(0)
        return self.timestamps[-1] - self.timestamps[0]

    @property
    def positive_labels(self) -> int:
        return sum(1 for label in self.labels if label == ANOMALOUS)

    @property
    def window_count(self) -> int:
        return len(self.windows)

    @property
    def is_labelled(self) -> bool:
        return self.window_count > 0

    def index_of(self, moment: datetime) -> int | None:
        for position, stamp in enumerate(self.timestamps):
            if stamp == moment:
                return position
        return None

    # ── slicing ────────────────────────────────────────────────────────
    def slice(self, start: int, stop: int | None = None) -> LabelledSeries:
        """Positional slice that keeps only the windows fully inside it."""
        stop = len(self) if stop is None else stop
        start = max(0, start)
        stop = max(start, min(stop, len(self)))
        timestamps = self.timestamps[start:stop]
        if timestamps:
            lower, upper = timestamps[0], timestamps[-1]
            windows = tuple(item for item in self.windows if lower <= item.start and item.end <= upper)
        else:
            windows = ()
        return replace(
            self,
            timestamps=timestamps,
            values=self.values[start:stop],
            labels=self.labels[start:stop],
            windows=windows,
        )

    def relabel(self, pad: timedelta = timedelta(0)) -> LabelledSeries:
        """Recompute labels from the (possibly padded) windows."""
        labels = tuple(
            ANOMALOUS if any(window.contains(stamp, pad) for window in self.windows) else NORMAL
            for stamp in self.timestamps
        )
        return replace(self, labels=labels)

    def to_records(self) -> list[dict[str, Any]]:
        """Row-per-point records, JSON friendly.

        Deliberately not a ``pandas.DataFrame``: the loader is stdlib-only, and
        a convenience accessor that can crash the interpreter inside a native
        string-array backend is not worth a 40 MB dependency nobody else needs.
        """
        return [
            {"timestamp": stamp.isoformat(), "value": value, "label": label}
            for stamp, value, label in zip(self.timestamps, self.values, self.labels, strict=True)
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "resource_id": self.resource_id,
            "points": self.size,
            "dropped_points": self.dropped_points,
            "windows": self.window_count,
            "anomalous_points": self.positive_labels,
            "anomalous_fraction": round(self.positive_labels / self.size, 6) if self.size else 0.0,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "step_seconds": self.step_seconds,
            "window_hours": [round(item.duration.total_seconds() / 3600, 2) for item in self.windows],
            "source": self.source,
        }


def parse_timestamp(value: str) -> datetime:
    """Parse a NAB timestamp into an aware UTC datetime.

    Accepts NAB's ``%Y-%m-%d %H:%M:%S[.%f]`` and ISO-8601 (with or without an
    offset), so anything this module serialises can be read back. The trailing
    ``Z`` is rewritten by hand because ``datetime.fromisoformat`` only learned to
    parse it in Python 3.11.
    """
    token = value.strip()
    normalised = f"{token[:-1]}+00:00" if token.endswith(("Z", "z")) else token
    try:
        parsed = datetime.fromisoformat(normalised)
    except ValueError:
        for fmt in TIMESTAMP_FORMATS:
            try:
                naive = datetime.strptime(token, fmt)
            except ValueError:
                continue
            return naive.replace(tzinfo=timezone.utc)
        raise ValueError(f"unrecognised timestamp: {value!r}") from None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def metric_name_for(file_stem: str) -> str:
    """Map a NAB file stem onto a unified metric name.

    Substring matching, longest alias first, so both ``ec2_cpu_utilization_*``
    and ``iio_us-east-1_i-a2eb1cd9_NetworkIn`` resolve to a known metric.
    """
    lowered = file_stem.lower()
    for alias, metric in sorted(METRIC_NAMES.items(), key=lambda item: -len(item[0])):
        if alias in lowered:
            return metric
    return lowered.split("_anomaly")[0] or lowered


def unit_for(metric: str) -> str:
    return UNIT_LABELS.get(metric, "")


def load_series(
    csv_path: str | Path,
    windows: Sequence[AnomalyWindow] = (),
    *,
    drop_missing: bool = True,
    max_gap_multiple: int | None = None,
) -> LabelledSeries:
    """Load one NAB-style CSV (``timestamp,value``) and attach window labels.

    ``drop_missing`` removes rows with an empty/NaN value. When
    ``max_gap_multiple`` is given, rows further than that multiple of the modal
    sampling step from their predecessor are also removed, so a multi-hour hole
    in the feed does not become a fake "jump" the detector reacts to.
    """
    path = Path(csv_path)
    name = path.stem
    metric = metric_name_for(name)

    rows: list[tuple[datetime, float]] = []
    dropped = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames or (), path)
        for row in reader:
            raw_value = (row.get("value") or "").strip()
            if drop_missing and not _is_number(raw_value):
                dropped += 1
                continue
            rows.append((parse_timestamp(row["timestamp"]), float(raw_value)))

    if not rows:
        raise ValueError(f"{path.name}: no usable rows")

    timestamps = [item[0] for item in rows]
    # The sampling step has to come from the whole file: a partial estimate would
    # make every early row look like an enormous gap.
    step = _modal_step(timestamps)
    limit = timedelta(seconds=step.total_seconds() * max_gap_multiple) if max_gap_multiple else None

    if limit is not None:
        kept: list[tuple[datetime, float]] = []
        previous: datetime | None = None
        for stamp, value in rows:
            if previous is not None and stamp - previous > limit:
                dropped += 1
                continue
            kept.append((stamp, value))
            previous = stamp
        rows = kept
        timestamps = [item[0] for item in rows]

    resolved = tuple(sorted(windows, key=lambda item: item.start))
    series = LabelledSeries(
        name=name,
        metric=metric,
        timestamps=tuple(timestamps),
        values=tuple(value for _, value in rows),
        labels=tuple(NORMAL for _ in timestamps),
        windows=resolved,
        resource_id=f"{metric}/{name}",
        service=_service_for(metric),
        region="unknown",
        unit=unit_for(metric),
        source=path.as_posix(),
        dropped_points=dropped,
    )
    return series.relabel()


def load_nab_windows(windows_path: str | Path, dataset: str = NAB_DATASET) -> dict[str, tuple[AnomalyWindow, ...]]:
    """Read ``labels/combined_windows.json`` into per-file windows."""
    payload = json.loads(Path(windows_path).read_text(encoding="utf-8"))
    windows: dict[str, tuple[AnomalyWindow, ...]] = {}
    prefix = f"{dataset}/"
    for key, ranges in payload.items():
        if not key.startswith(prefix):
            continue
        stem = Path(key).stem
        windows[stem] = tuple(AnomalyWindow.from_strings(start, end) for start, end in ranges)
    return windows


def load_nab_dataset(
    root: str | Path,
    windows_path: str | Path,
    dataset: str = NAB_DATASET,
    *,
    max_gap_multiple: int | None = None,
    include_unlabelled: bool = True,
) -> list[LabelledSeries]:
    """Load the series of a NAB dataset directory.

    ``include_unlabelled`` keeps series the benchmark ships without an anomaly
    window (a control series). They contribute nothing to recall but are the
    purest possible false-alarm test, so the benchmark keeps them and reports
    them separately.
    """
    directory = Path(root)
    labels = load_nab_windows(windows_path, dataset=dataset)
    series: list[LabelledSeries] = []
    for path in sorted(directory.glob("*.csv")):
        windows = labels.get(path.stem, ())
        if not windows and not include_unlabelled:
            continue
        series.append(load_series(path, windows, max_gap_multiple=max_gap_multiple))
    if not series:
        raise FileNotFoundError(f"no labelled series found in {directory} for dataset {dataset!r}")
    return series


def split_time(series: LabelledSeries, train_fraction: float = 0.6) -> tuple[LabelledSeries, LabelledSeries]:
    """Split once, by position: training first, evaluation after.

    No shuffling, no interleaving - a model must never see the future.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    cut = int(len(series) * train_fraction)
    cut = max(1, min(cut, len(series) - 1)) if len(series) > 1 else 1
    return series.slice(0, cut), series.slice(cut)


def derive_bands(
    series: Iterable[LabelledSeries],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> dict[str, tuple[float, float]]:
    """Expected value band per metric from the **training** segment only.

    Real telemetry has no universal "normal" band, so ``band_position`` is
    calibrated per metric on data the model is allowed to see.
    """
    buckets: dict[str, list[float]] = {}
    for item in series:
        buckets.setdefault(item.metric, []).extend(item.values)
    bands: dict[str, tuple[float, float]] = {}
    for metric, values in buckets.items():
        if not values:
            continue
        ordered = sorted(values)
        low = ordered[min(len(ordered) - 1, int(len(ordered) * lower_quantile))]
        high = ordered[min(len(ordered) - 1, int(len(ordered) * upper_quantile))]
        if high - low < 1e-12:  # constant series: widen slightly so the band is valid
            low, high = low - 0.5, high + 0.5
        bands[metric] = (float(low), float(high))
    return bands


def _require_columns(fieldnames: Sequence[str], path: Path) -> None:
    missing = {"timestamp", "value"} - {name.strip().lower() for name in fieldnames}
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {sorted(missing)}; found {list(fieldnames)}")


def _is_number(token: str) -> bool:
    """Finite number? ``NaN`` and ``inf`` are treated as missing data.

    ``float("NaN")`` parses happily but poisons every mean, std and z-score in
    the pipeline, so it is filtered here rather than trusted downstream.
    """
    if not token:
        return False
    try:
        value = float(token)
    except ValueError:
        return False
    return math.isfinite(value)


def _modal_step(timestamps: Sequence[datetime]) -> timedelta:
    """Most common sampling interval in the series so far."""
    if len(timestamps) < 3:
        return timedelta(seconds=1)
    steps: dict[float, int] = {}
    for index in range(1, len(timestamps)):
        seconds = (timestamps[index] - timestamps[index - 1]).total_seconds()
        steps[seconds] = steps.get(seconds, 0) + 1
    best = max(steps.items(), key=lambda item: (item[1], -item[0]))[0]
    return timedelta(seconds=best or 1.0)


def _service_for(metric: str) -> str:
    return {
        "cpu_utilization": "EC2",
        "disk_write_bytes": "EBS",
        "network_in_bytes": "VPC",
        "request_count": "ELB",
        "grok_request_count": "ELB",
    }.get(metric, "EC2")


@dataclass(slots=True)
class DatasetManifest:
    """Fetch bookkeeping, written next to the cached data."""

    dataset: str
    source_url: str
    files: dict[str, str] = field(default_factory=dict)  # name -> sha256
    downloaded_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_url": self.source_url,
            "downloaded_at": self.downloaded_at,
            "files": dict(sorted(self.files.items())),
        }
