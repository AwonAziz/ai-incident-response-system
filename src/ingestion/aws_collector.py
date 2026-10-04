"""AWS telemetry collector (EC2 / Lambda / RDS / ELB)."""

from __future__ import annotations

from typing import ClassVar

from src.core.enums import Cloud
from src.ingestion.base_collector import SimulatedCollector

__all__ = ["AWSCollector"]


class AWSCollector(SimulatedCollector):
    """Emits CPU, memory, latency, error-rate and disk-I/O telemetry for AWS."""

    cloud: ClassVar[Cloud] = Cloud.AWS
