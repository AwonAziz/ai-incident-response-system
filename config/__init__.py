"""Runtime configuration for the AI incident response system."""

from __future__ import annotations

from . import settings

#: telemetry profiles are data (``cloud_profiles.yaml``), not a module
__all__ = ["settings"]
