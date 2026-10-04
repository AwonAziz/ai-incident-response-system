"""Generate (and optionally inject) anomalous telemetry.

Examples:
    python scripts/simulate_incident.py                          # one AWS spike, JSON on stdout
    python scripts/simulate_incident.py --cloud all --count 3
    python scripts/simulate_incident.py --metric error_rate --value 7.5 --format text
    python scripts/simulate_incident.py --push http://localhost:8080/inject
    python scripts/simulate_incident.py --loop 6 --interval 10 --push http://host:8080/inject
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from src.ingestion import SeriesHistory, collectors_for_clouds
from src.ingestion.metric_schema import Metric


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Simulate multi-cloud incidents")
    parser.add_argument("--cloud", default="aws", choices=["aws", "azure", "gcp", "all"], help="target cloud")
    parser.add_argument("--count", type=int, default=1, help="number of metrics to emit")
    parser.add_argument("--metric", default=None, help="metric name (default: any metric of the chosen cloud)")
    parser.add_argument("--value", type=float, default=None, help="explicit value instead of a generated spike")
    parser.add_argument("--seed", type=int, default=settings.random_seed, help="RNG seed")
    parser.add_argument("--warmup", type=int, default=10, help="clean ticks to run first so windows are realistic")
    parser.add_argument("--format", default="json", choices=["json", "text"], help="stdout format")
    parser.add_argument("--push", default=None, metavar="URL", help="POST the metrics to a control API instead of stdout")
    parser.add_argument(
        "--token",
        default=None,
        help="bearer token for the target API (defaults to $API_TOKEN)",
    )
    parser.add_argument("--loop", type=int, default=1, help="repeat the injection N times (0 = forever)")
    parser.add_argument("--interval", type=float, default=10.0, help="seconds between loops")
    return parser


def generate_metrics(
    cloud: str = "aws",
    count: int = 1,
    *,
    metric_name: str | None = None,
    value: float | None = None,
    seed: int = 42,
    warmup: int = 10,
    profiles: dict[str, Any] | None = None,
) -> list[Metric]:
    """Produce ``count`` anomalous metrics for one cloud.

    A shared :class:`SeriesHistory` is warmed with clean ticks first so the
    emitted metrics carry realistic rolling statistics (and therefore usable
    z-scores), exactly like the running pipeline would see them.
    """
    from src.core.enums import Cloud

    clouds = [Cloud.parse(item) for item in (Cloud.values() if cloud == "all" else [cloud])]
    history = SeriesHistory(window=30, min_samples=5)
    collectors = collectors_for_clouds(
        clouds,
        seed=seed,
        history=history,
        profiles=profiles,
    )

    for _ in range(max(0, warmup)):
        for collector in collectors:
            collector.collect_and_track()

    spikes = collectors_for_clouds(
        clouds,
        inject_anomaly=True,
        seed=seed,
        history=history,
        profiles=profiles,
    )

    emitted: list[Metric] = []
    while len(emitted) < max(1, count):
        for collector in spikes:
            produced = collector.collect_and_track()
            degraded = set(collector.degraded_resources())
            # only emit metrics that belong to a resource the simulator actually
            # degraded, otherwise "spikes" look like ordinary samples
            stressed = [item for item in produced if item.resource_id in degraded]
            if metric_name:
                stressed = [item for item in stressed if item.name == metric_name]
                if not stressed:
                    stressed = [item for item in produced if item.name == metric_name]
            if not stressed:
                continue
            for metric in stressed:
                if value is not None:
                    metric.value = float(value)
                emitted.append(metric.with_detection(is_anomaly=True, confidence=1.0, anomaly_score=metric.value))
    return emitted[: max(1, count)]


def _format(metrics: Sequence[Metric], style: str) -> str:
    if style == "json":
        return json.dumps([metric.to_dict() for metric in metrics], indent=2)
    lines = []
    for metric in metrics:
        window = metric.window
        z = f"{metric.z_score:+.2f}" if window else "n/a"
        lines.append(
            f"[{metric.cloud.value.upper()}] {metric.service}/{metric.resource_id} "
            f"{metric.name}={metric.value:.2f}{metric.unit} "
            f"(z={z}, samples={window.count if window else 0})"
        )
    return "\n".join(lines)


def push(metrics: Sequence[Metric], url: str, timeout: float = 5.0, token: str | None = None) -> str:
    """POST metrics to a control API ``/inject`` endpoint."""
    from src.notifications.http_client import post_json

    headers = {"Authorization": f"Bearer {token}"} if token else None
    return post_json(url, {"metrics": [metric.to_dict() for metric in metrics]}, timeout=timeout, headers=headers)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    runs = max(0, args.loop)
    iteration = 0
    while True:
        iteration += 1
        metrics = generate_metrics(
            args.cloud,
            args.count,
            metric_name=args.metric,
            value=args.value,
            seed=args.seed + iteration,
            warmup=args.warmup,
            profiles=settings.profiles,
        )
        if args.push:
            try:
                detail = push(metrics, args.push, token=args.token or os.getenv("API_TOKEN"))
                print(f"iteration {iteration}: {detail}", flush=True)
            except Exception as exc:  # the CLI reports, it does not crash
                print(f"iteration {iteration}: push failed - {exc}", file=sys.stderr, flush=True)
                return 1
        else:
            print(_format(metrics, args.format), flush=True)
        if runs and iteration >= runs:
            return 0
        time.sleep(max(0.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
