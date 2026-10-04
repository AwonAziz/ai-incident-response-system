"""Configuration loading and the YAML telemetry profiles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from config.settings import (
    CLOUD_PROFILES,
    SETTINGS,
    Settings,
    get_settings,
    load_cloud_profiles,
    setup_logging,
)


class TestEnvParsing:
    def test_typed_defaults_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in ("COLLECTION_INTERVAL", "DEDUP_WINDOW", "QUIET_MODE", "MODEL_CONTAMINATION"):
            monkeypatch.delenv(key, raising=False)
        settings = get_settings()
        assert settings.collection_interval_seconds == 5.0
        assert settings.dedup_window_seconds == 60
        assert settings.quiet_mode is False

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("12", 12), ("12.7", 12), ("not-a-number", 60), ("", 60), ("  ", 60)],
    )
    def test_int_coercion_falls_back(self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: int) -> None:
        monkeypatch.setenv("DEDUP_WINDOW", raw)
        assert get_settings().dedup_window_seconds == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("1.5", 1.5), ("bogus", 5.0), ("0", 0.0)],
    )
    def test_float_coercion(self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: float) -> None:
        monkeypatch.setenv("COLLECTION_INTERVAL", raw)
        assert get_settings().collection_interval_seconds == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("true", True), ("YES", True), ("on", True), ("1", True), ("false", False), ("0", False), ("junk", False)],
    )
    def test_bool_coercion(self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
        monkeypatch.setenv("QUIET_MODE", raw)
        assert get_settings().quiet_mode is expected

    def test_notification_secrets_are_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/x")
        monkeypatch.setenv("PAGERDUTY_ROUTING_KEY", "key-123")
        settings = get_settings()
        assert settings.slack_webhook_url.endswith("/x")
        assert settings.pagerduty_routing_key == "key-123"

    def test_as_dict_hides_profiles(self) -> None:
        payload = SETTINGS.as_dict()
        assert "profiles" not in payload
        assert json.dumps(payload, default=str)


class TestProfiles:
    def test_repository_profiles_load(self) -> None:
        profiles = load_cloud_profiles()
        assert set(profiles["clouds"]) == {"aws", "azure", "gcp"}
        assert "cpu_utilization" in profiles["metrics"]
        assert profiles["metrics"]["cpu_utilization"]["warn_high"] == 78
        assert profiles["defaults"]["min_history"] == 5

    def test_cloud_overrides_only_replace_the_keys_they_declare(self) -> None:
        aws_error_rate = CLOUD_PROFILES["clouds"]["aws"]["overrides"]["error_rate"]
        base = CLOUD_PROFILES["metrics"]["error_rate"]
        assert set(aws_error_rate) == {"critical_high"}
        assert aws_error_rate["critical_high"] < base["critical_high"]
        assert "error_rate" not in (CLOUD_PROFILES["clouds"]["gcp"].get("overrides") or {})

    def test_every_referenced_metric_is_defined(self) -> None:
        for cloud, profile in CLOUD_PROFILES["clouds"].items():
            for metric_name in profile["metrics"]:
                assert metric_name in CLOUD_PROFILES["metrics"], f"{cloud}:{metric_name} undefined"

    def test_every_resource_has_an_id_service_and_region(self) -> None:
        for cloud, profile in CLOUD_PROFILES["clouds"].items():
            assert profile["resources"], f"{cloud} has no resources"
            for resource in profile["resources"]:
                assert resource["id"] and resource["service"] and resource["region"]

    def test_missing_file_falls_back_to_defaults(self, tmp_path: Path) -> None:
        profiles = load_cloud_profiles(tmp_path / "nope.yaml")
        assert profiles["clouds"] == {}
        assert profiles["defaults"]["history_window"] == 30

    def test_broken_yaml_falls_back_instead_of_raising(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.yaml"
        broken.write_text("clouds: [unclosed", encoding="utf-8")
        assert load_cloud_profiles(broken)["defaults"]["min_history"] == 5

    def test_profiles_helpers(self, settings: Settings) -> None:
        assert settings.history_window_size == 20
        assert settings.min_history_samples == 5
        assert settings.severity_factors["critical"] == 3.0
        assert settings.sla_targets_minutes["critical"] == 15.0


class TestLoggingSetup:
    def test_creates_log_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "logs"
        setup_logging("DEBUG", target)
        assert (target / "system.log").exists()
        setup_logging("INFO", target)  # idempotent (force=True)

    def test_is_idempotent(self, tmp_path: Path) -> None:
        setup_logging("INFO", tmp_path)
        setup_logging("INFO", tmp_path)


class TestConsoleEncoding:
    def test_enable_utf8_console_is_safe_to_call(self) -> None:
        """The dashboard draws box lines; a cp1252 stdout must not raise."""
        from config.settings import enable_utf8_console

        assert isinstance(enable_utf8_console(), bool)
        assert sys.stdout is not None
