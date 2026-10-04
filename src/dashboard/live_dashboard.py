"""Live terminal dashboard (Rich).

The dashboard is a pure consumer: it renders whatever the pipeline pushes into
it and never mutates pipeline state. It also subscribes to incident lifecycle
events so the feed column updates without polling.

``get_renderable()`` returns a fresh Rich renderable on every call, which is
what ``rich.live.Live`` expects.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rich.align import Align
from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.core.clock import Clock, SystemClock
from src.core.enums import Cloud, IncidentStatus, Severity
from src.ingestion.metric_schema import Metric
from src.triage.incident_manager import Incident, IncidentManager

__all__ = ["SEVERITY_STYLE", "LiveDashboard"]

SEVERITY_STYLE: dict[str, str] = {
    "CRITICAL": "bold white on red",
    "HIGH": "bold red",
    "MEDIUM": "yellow",
    "LOW": "cyan",
}

SEVERITY_COLOR: dict[str, str] = {
    "CRITICAL": "red",
    "HIGH": "dark_orange",
    "MEDIUM": "yellow",
    "LOW": "cyan",
}

SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float], width: int = 12) -> str:
    """Tiny unicode sparkline; renders a flat line for constant/empty input."""
    if not values:
        return " " * width
    sample = list(values)[-width:]
    low = min(sample)
    high = max(sample)
    span = high - low
    if span <= 1e-9:
        return SPARK_BLOCKS[0] * len(sample)
    scale = (len(SPARK_BLOCKS) - 1) / span
    return "".join(SPARK_BLOCKS[int((value - low) * scale)] for value in sample)


@dataclass(slots=True)
class _SeriesView:
    """Latest state for one metric series, rendered as a table row."""

    series_key: str
    metric_name: str
    label: str
    value: float
    unit: str
    z_score: float
    is_anomaly: bool
    confidence: float
    history: deque[float] = field(default_factory=lambda: deque(maxlen=12))


@dataclass(slots=True)
class _CloudView:
    metrics_seen: int = 0
    anomalies: int = 0
    series: dict[str, _SeriesView] = field(default_factory=dict)


class LiveDashboard:
    """Renders pipeline state as a set of Rich panels."""

    def __init__(
        self,
        incident_manager: IncidentManager,
        detector: Any | None = None,
        *,
        clock: Clock | None = None,
        subscribe: bool = True,
    ) -> None:
        self.incidents = incident_manager
        self.detector = detector
        self.clock: Clock = clock or SystemClock()
        self.started_at = self.clock.now()
        self.ticks = 0
        self.metrics_seen = 0
        self.anomalies_seen = 0
        self.feed: deque[dict[str, Any]] = deque(maxlen=12)
        self._clouds: dict[Cloud, _CloudView] = {cloud: _CloudView() for cloud in Cloud}
        self._last_update = self.started_at
        if subscribe:
            self.incidents.add_listener(self.record_event)

    # ── inputs ─────────────────────────────────────────────────────────
    def update_metrics(self, metrics: Sequence[Metric]) -> None:
        """Absorb a scored batch."""
        self.ticks += 1
        self._last_update = self.clock.now()
        for metric in metrics:
            view = self._clouds.setdefault(metric.cloud, _CloudView())
            view.metrics_seen += 1
            self.metrics_seen += 1
            if metric.is_anomaly:
                view.anomalies += 1
                self.anomalies_seen += 1
            series = view.series.get(metric.series_key)
            if series is None:
                series = _SeriesView(
                    series_key=metric.series_key,
                    metric_name=metric.name,
                    label=metric.name.replace("_", " "),
                    value=metric.value,
                    unit=metric.unit,
                    z_score=metric.z_score,
                    is_anomaly=metric.is_anomaly,
                    confidence=metric.confidence,
                )
                view.series[metric.series_key] = series
            else:
                series.value = metric.value
                series.z_score = metric.z_score
                series.is_anomaly = metric.is_anomaly
                series.confidence = metric.confidence
            series.history.append(metric.value)

    def record_event(self, kind: str, incident: Incident) -> None:
        """Incident lifecycle listener."""
        message = {
            "created": incident.title,
            "acknowledged": f"acknowledged by operator ({incident.id})",
            "resolved": f"resolved: {incident.resolution}",
            "escalated": f"escalated to {incident.severity.name} ({incident.id})",
        }.get(kind, kind)
        self.feed.append(
            {
                "at": self.clock.now(),
                "kind": kind,
                "severity": incident.severity.name,
                "incident_id": incident.id,
                "message": message,
            }
        )

    # ── rendering ──────────────────────────────────────────────────────
    def get_renderable(self) -> RenderableType:
        return Group(
            self._header(),
            Columns(self._cloud_panels(), equal=True, expand=True),
            Columns([self._model_panel(), self._incidents_panel()], equal=True, expand=True),
            self._feed_panel(),
            self._footer(),
        )

    def _header(self) -> RenderableType:
        uptime = (self.clock.now() - self.started_at).total_seconds()
        stats = self.incidents.stats
        title = Text()
        title.append(" AI INCIDENT RESPONSE ", style="bold black on cyan")
        title.append(f"  up {_duration(uptime)}   tick {self.ticks}   metrics {self.metrics_seen}")
        title.append(f"   anomalies {self.anomalies_seen}", style="bold yellow" if self.anomalies_seen else "")
        title.append(
            f"   active {stats['active_count']}   created {stats['total_created']}   resolved {stats['total_resolved']}"
        )
        return Panel(title, border_style="cyan", padding=(0, 1))

    def _cloud_panels(self) -> list[RenderableType]:
        panels: list[RenderableType] = []
        for cloud in Cloud:
            view = self._clouds[cloud]
            table = Table(box=None, expand=True, pad_edge=False, show_edge=False)
            table.add_column("metric", style="dim", no_wrap=True)
            table.add_column("value", justify="right", no_wrap=True)
            table.add_column("z", justify="right", no_wrap=True)
            table.add_column("trend", no_wrap=True)
            table.add_column("", no_wrap=True)

            hottest = sorted(view.series.values(), key=lambda item: -abs(item.z_score))[:5]
            for series in hottest or []:
                style = SEVERITY_COLOR["CRITICAL"] if series.is_anomaly else None
                table.add_row(
                    series.label,
                    Text(f"{series.value:.1f}{series.unit}", style=style or ""),
                    Text(f"{series.z_score:+.1f}", style=style or ""),
                    sparkline(list(series.history)),
                    Text("ANOMALY", style="bold red") if series.is_anomaly else Text(""),
                )
            if not view.series:
                table.add_row("awaiting telemetry", "", "", "", "")

            anomaly_style = "bold red" if view.anomalies else "dim green"
            footer = Text.assemble(
                f"{view.metrics_seen} metrics   ",
                (f"{view.anomalies} anomalies", anomaly_style),
            )
            panels.append(Panel(Group(table, footer), title=f"[bold]{cloud.label}[/]", border_style="blue", padding=(0, 1)))
        return panels

    def _model_panel(self) -> RenderableType:
        table = Table(box=None, expand=True, pad_edge=False, show_edge=False)
        table.add_column("model stat", style="dim", no_wrap=True)
        table.add_column("value", justify="right", no_wrap=True)

        if self.detector is None:
            table.add_row("detector", "not attached")
        else:
            stats = self.detector.stats
            table.add_row("version", str(stats.get("version")))
            table.add_row("training samples", f"{stats.get('training_samples', 0)}")
            table.add_row("trees / contamination", f"{stats.get('n_estimators')} @ {stats.get('contamination')}")
            table.add_row("anomaly rate", f"{float(stats.get('anomaly_rate', 0.0)) * 100:.2f}%")
            table.add_row("mean confidence", f"{float(stats.get('mean_confidence', 0.0)) * 100:.1f}%")
            drift = float(stats.get("drift_score", 0.0))
            drift_text = Text(
                f"{drift:.3f}",
                style="bold red" if stats.get("drifted") else "green",
            )
            table.add_row("drift", drift_text)
            if stats.get("retrain_recommended"):
                table.add_row("action", Text("drift detected - retrain advised", style="bold yellow"))

        routing = [notifier.name for notifier in getattr(self, "_notifiers", ())] or ["console"]
        return Panel(
            table,
            title="[bold]DETECTION MODEL[/]",
            border_style="magenta",
            padding=(0, 1),
            subtitle=f"[dim]channels: {', '.join(routing)}[/]",
        )

    def _incidents_panel(self) -> RenderableType:
        table = Table(box=None, expand=True, pad_edge=False, show_edge=False)
        table.add_column("id", style="dim", no_wrap=True)
        table.add_column("sev", no_wrap=True)
        table.add_column("target", no_wrap=True)
        table.add_column("age", justify="right", no_wrap=True)
        table.add_column("x", justify="right", no_wrap=True)

        active = self.incidents.active()[:8]
        now = self.clock.now()
        for incident in active:
            table.add_row(
                incident.id,
                Text(incident.severity.name, style=SEVERITY_STYLE[incident.severity.name]),
                Text(f"{incident.cloud.value}/{incident.service}", overflow="ellipsis", no_wrap=True),
                _duration(incident.age_seconds(now)),
                str(incident.occurrences),
            )
        if not active:
            table.add_row("-", Text("no active incidents", style="dim"), "", "", "")
        counts = self.incidents.counts_by_severity()
        summary = Text.assemble(
            ("CRIT ", "bold red"),
            (str(counts["CRITICAL"]), "red"),
            ("  HIGH ", "bold red"),
            (str(counts["HIGH"]), "dark_orange"),
            ("  MED ", "bold yellow"),
            (str(counts["MEDIUM"]), "yellow"),
            ("  LOW ", "bold cyan"),
            (str(counts["LOW"]), "cyan"),
        )
        return Panel(Group(table, summary), title="[bold]ACTIVE INCIDENTS[/]", border_style="red", padding=(0, 1))

    def _feed_panel(self) -> RenderableType:
        table = Table(box=None, expand=True, pad_edge=False, show_edge=False)
        table.add_column("time", style="dim", no_wrap=True)
        table.add_column("event", no_wrap=True)
        table.add_column("detail", overflow="ellipsis")

        items = list(self.feed)[-6:][::-1]
        for item in items:
            at = item["at"]
            style = SEVERITY_COLOR.get(item["severity"], "dim")
            table.add_row(
                at.strftime("%H:%M:%S"),
                Text(item["kind"].upper(), style=style),
                Text(f"[{item['incident_id']}] {item['message']}", style=style),
            )
        if not items:
            table.add_row("-", Text("no lifecycle events yet", style="dim"), "")
        return Panel(table, title="[bold]INCIDENT FEED[/]", border_style="yellow", padding=(0, 1))

    def _footer(self) -> RenderableType:
        return Align.center(
            Text(
                "ctrl-c to stop   |   metrics/5s   |   anomalies feed detection -> triage -> notifications",
                style="dim",
            )
        )

    # ── helpers ────────────────────────────────────────────────────────
    def attach_notifiers(self, notifiers: Sequence[Any]) -> None:
        """Show the live routing table in the model panel subtitle."""
        self._notifiers = list(notifiers)

    @property
    def state(self) -> dict[str, Any]:
        """Machine readable snapshot (API/debug)."""
        return {
            "ticks": self.ticks,
            "metrics_seen": self.metrics_seen,
            "anomalies_seen": self.anomalies_seen,
            "started_at": self.started_at.isoformat(),
            "last_update": self._last_update.isoformat(),
            "clouds": {
                cloud.value: {
                    "metrics_seen": view.metrics_seen,
                    "anomalies": view.anomalies,
                    "series": len(view.series),
                }
                for cloud, view in self._clouds.items()
            },
            "status_counts": {
                status.value: sum(1 for item in self.incidents.all() if item.status is status)
                for status in IncidentStatus
            },
            "active": [incident.id for incident in self.incidents.active()],
        }


def _duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def severity_badge(severity: Severity) -> Text:  # pragma: no cover - display helper
    return Text(severity.name, style=SEVERITY_STYLE[severity.name])
