"""Core primitives: enums, clocks, rolling statistics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core.clock import ManualClock, SystemClock, utc_now
from src.core.enums import SEVERITY_DESC, Cloud, IncidentStatus, Severity
from src.core.stats import RollingStats, stats_from_values


class TestCloud:
    def test_parses_common_spellings(self) -> None:
        assert Cloud.parse("aws") is Cloud.AWS
        assert Cloud.parse(" Azure ") is Cloud.AZURE
        assert Cloud.parse("GCP") is Cloud.GCP
        assert Cloud.parse(Cloud.AWS) is Cloud.AWS

    def test_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="unknown cloud"):
            Cloud.parse("digitalocean")

    def test_labels(self) -> None:
        assert [cloud.label for cloud in Cloud] == ["AWS", "Azure", "GCP"]
        assert Cloud.values() == ["aws", "azure", "gcp"]


class TestSeverity:
    def test_ordering(self) -> None:
        assert Severity.CRITICAL > Severity.HIGH > Severity.MEDIUM > Severity.LOW
        assert Severity.LOW.at_least(Severity.LOW)
        assert not Severity.HIGH.at_least(Severity.CRITICAL)

    def test_parse_accepts_names_and_numbers(self) -> None:
        assert Severity.parse("critical") is Severity.CRITICAL
        assert Severity.parse("4") is Severity.CRITICAL
        assert Severity.parse(Severity.LOW) is Severity.LOW
        with pytest.raises(ValueError, match="unknown severity"):
            Severity.parse("catastrophic")

    def test_desc_tuple_is_worst_first(self) -> None:
        assert SEVERITY_DESC == (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW)


class TestIncidentStatus:
    def test_openness(self) -> None:
        assert IncidentStatus.OPEN.is_open
        assert IncidentStatus.ACKNOWLEDGED.is_open
        assert IncidentStatus.RESOLVED.is_closed

    def test_parse(self) -> None:
        assert IncidentStatus.parse("resolved") is IncidentStatus.RESOLVED
        with pytest.raises(ValueError):
            IncidentStatus.parse("snoozed")


class TestClocks:
    def test_system_clock_is_utc_aware(self) -> None:
        now = SystemClock().now()
        assert now.tzinfo is not None
        assert now <= utc_now() + timedelta(seconds=1)

    def test_manual_clock_advances_instead_of_blocking(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        clock = ManualClock(start)
        assert clock.now() == start
        clock.sleep(2.5)
        assert clock.now() == start + timedelta(seconds=2.5)
        clock.set(start)
        assert clock.now() == start


class TestRollingStats:
    def test_empty_is_neutral(self) -> None:
        stats = stats_from_values([])
        assert stats.count == 0
        assert stats.z_score() == 0.0
        assert stats.ratio() == 0.0
        assert not stats.is_ready(3)

    def test_sample_statistics(self) -> None:
        stats = stats_from_values([10.0, 12.0, 14.0, 16.0])
        assert stats.count == 4
        assert stats.mean == pytest.approx(13.0)
        assert stats.minimum == 10.0
        assert stats.maximum == 16.0
        assert stats.last == 16.0
        assert stats.delta == pytest.approx(2.0)
        assert stats.pct_change == pytest.approx(14.2857, rel=1e-3)

    def test_z_score_and_ratio(self) -> None:
        stats = stats_from_values([10.0, 10.0, 10.0, 14.0])
        assert stats.z_score(14.0) == pytest.approx(1.5)
        assert stats.ratio(14.0) == pytest.approx(14.0 / 11.0)
        assert stats.is_ready(4)
        assert not stats.is_ready(5)

    def test_constant_series_has_no_infinite_z(self) -> None:
        stats = stats_from_values([5.0, 5.0, 5.0])
        assert stats.std == 0.0
        assert stats.z_score(500.0) == 0.0
        assert stats.coefficient_of_variation == 0.0

    def test_zero_mean_is_guarded(self) -> None:
        stats = stats_from_values([0.0, 0.0, 0.0])
        assert stats.ratio() == 0.0
        assert stats.z_score(99.0) == 0.0
        assert stats.pct_change == 0.0

    def test_to_dict_is_json_friendly(self) -> None:
        payload = stats_from_values([1.0, 2.0]).to_dict()
        assert payload["count"] == 2
        assert "z_score" in payload
        assert all(isinstance(value, (int, float)) for value in payload.values())

    def test_dataclass_is_immutable(self) -> None:
        stats = stats_from_values([1.0])
        with pytest.raises(AttributeError):
            stats.mean = 5.0  # type: ignore[misc]
        assert isinstance(stats, RollingStats)
