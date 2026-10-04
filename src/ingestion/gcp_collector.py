"""GCP telemetry collector (GKE / Compute Engine / Cloud SQL)."""

from __future__ import annotations

from typing import ClassVar

from src.core.enums import Cloud
from src.ingestion.base_collector import SimulatedCollector

__all__ = ["GCPCollector"]


class GCPCollector(SimulatedCollector):
    """Emits CPU, memory, network-I/O, query-latency and error telemetry."""

    cloud: ClassVar[Cloud] = Cloud.GCP
