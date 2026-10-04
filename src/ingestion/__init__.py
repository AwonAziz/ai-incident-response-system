"""Telemetry ingestion layer.

Public surface::

    from src.ingestion import AWSCollector, AzureCollector, GCPCollector
    from src.ingestion import Metric, SeriesHistory

Swapping the simulator for a real cloud SDK collector only requires subclassing
:class:`~src.ingestion.base_collector.BaseCollector`; the pipeline consumes
:class:`~src.ingestion.metric_schema.Metric` objects either way.
"""

from __future__ import annotations

from src.core.enums import Cloud
from src.ingestion.aws_collector import AWSCollector
from src.ingestion.azure_collector import AzureCollector
from src.ingestion.base_collector import BaseCollector, SimulatedCollector
from src.ingestion.gcp_collector import GCPCollector
from src.ingestion.history import SeriesHistory
from src.ingestion.metric_schema import Metric, MetricSpec, ResourceSpec
from src.ingestion.replay_collector import ReplayCollector

COLLECTORS: dict[Cloud, type[SimulatedCollector]] = {
    Cloud.AWS: AWSCollector,
    Cloud.AZURE: AzureCollector,
    Cloud.GCP: GCPCollector,
}


def collectors_for_clouds(
    clouds: list[Cloud] | tuple[Cloud, ...] | None = None,
    **kwargs: object,
) -> list[SimulatedCollector]:
    """Instantiate one collector per requested cloud."""
    selected = list(clouds) if clouds else list(COLLECTORS)
    return [COLLECTORS[Cloud.parse(cloud)](**kwargs) for cloud in selected]


__all__ = [
    "COLLECTORS",
    "AWSCollector",
    "AzureCollector",
    "BaseCollector",
    "GCPCollector",
    "Metric",
    "MetricSpec",
    "ReplayCollector",
    "ResourceSpec",
    "SeriesHistory",
    "SimulatedCollector",
    "collectors_for_clouds",
]
