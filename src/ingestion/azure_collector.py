"""Azure telemetry collector (AKS / Virtual Machines / App Service)."""

from __future__ import annotations

from typing import ClassVar

from src.core.enums import Cloud
from src.ingestion.base_collector import SimulatedCollector

__all__ = ["AzureCollector"]


class AzureCollector(SimulatedCollector):
    """Emits CPU, memory, request-rate, response-time and error telemetry."""

    cloud: ClassVar[Cloud] = Cloud.AZURE
