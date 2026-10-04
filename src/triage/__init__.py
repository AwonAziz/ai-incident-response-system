"""Triage layer: root-cause analysis, severity scoring, incident lifecycle."""

from __future__ import annotations

from src.triage.incident_manager import Incident, IncidentEvent, IncidentManager
from src.triage.root_cause import Hypothesis, RootCause, analyze, generate_hints
from src.triage.triage_engine import TriageEngine, TriageSettings

__all__ = [
    "Hypothesis",
    "Incident",
    "IncidentEvent",
    "IncidentManager",
    "RootCause",
    "TriageEngine",
    "TriageSettings",
    "analyze",
    "generate_hints",
]
