"""Configuration loading.

Two layers feed the system:

* ``cloud_profiles.yaml`` — declarative telemetry profiles (services, metric
  bands, thresholds, weights). It is data, so it is versionable and editable
  without touching code.
* environment variables / ``.env`` — deployment specific knobs (timing, model
  hyper-parameters, notification credentials).

Every environment value is read through a typed helper so a malformed ``.env``
degrades to the documented default instead of crashing the pipeline at import
time. Module level constants are kept because they are the public config API
(``from config.settings import COLLECTION_INTERVAL_SECONDS``).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
PROFILES_PATH = CONFIG_DIR / "cloud_profiles.yaml"
DOTENV_PATH = PROJECT_ROOT / ".env"

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n", ""}


def _load_dotenv() -> None:
    """Load ``.env`` if python-dotenv is available. Never overrides real env vars."""
    try:  # pragma: no cover - trivial guard
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional dependency
        return
    if DOTENV_PATH.is_file():
        load_dotenv(DOTENV_PATH, override=False)


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None else raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(float(raw.strip()))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    token = raw.strip().lower()
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    return default


DEFAULT_PROFILES: dict[str, Any] = {
    "version": 1,
    "defaults": {"history_window": 30, "min_history": 5},
    "metrics": {},
    "severity_factors": {"critical": 3.0, "high": 2.0, "medium": 1.2, "low": 0.5},
    "sla_targets_minutes": {"critical": 15, "high": 60, "medium": 240, "low": 1440},
    "clouds": {},
}


def load_cloud_profiles(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load telemetry profiles from YAML, merging over the built-in defaults.

    A malformed or missing file yields the defaults instead of an exception so
    the pipeline degrades rather than dies at import time.
    """
    profiles = {key: (dict(value) if isinstance(value, dict) else value) for key, value in DEFAULT_PROFILES.items()}
    target = Path(path) if path is not None else PROFILES_PATH
    try:
        import yaml
    except ImportError:  # pragma: no cover - optional dependency
        return profiles
    try:
        with target.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return profiles
    if not isinstance(loaded, dict):
        return profiles

    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(profiles.get(key), dict):
            profiles[key].update(value)
        else:
            profiles[key] = value

    merged_defaults = dict(DEFAULT_PROFILES["defaults"])
    merged_defaults.update(profiles.get("defaults") or {})
    profiles["defaults"] = merged_defaults
    return profiles


@dataclass(frozen=True, slots=True)
class Settings:
    """Typed view of every runtime knob."""

    # pipeline timing
    collection_interval_seconds: float = 5.0
    dedup_window_seconds: int = 60
    auto_resolve_minutes: int = 10
    inject_every_ticks: int = 6

    # detection
    model_contamination: float = 0.02
    model_n_estimators: int = 150
    baseline_samples: int = 500
    history_window: int = 30
    min_history: int = 5
    drift_threshold: float = 0.35
    drift_min_batches: int = 10
    ml_min_confidence: float = 0.6

    # triage
    critical_score: float = 8.0
    high_score: float = 5.5
    medium_score: float = 3.0
    auto_resolve_severities: tuple[str, ...] = ("medium", "low")

    # notifications
    slack_webhook_url: str = ""
    pagerduty_routing_key: str = ""
    email_to: str = ""
    email_from: str = ""
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    webhook_url: str = ""
    quiet_mode: bool = False
    notifier_timeout_seconds: float = 5.0
    notifier_max_retries: int = 2
    notifier_cooldown_seconds: float = 5.0

    # control API
    api_enabled: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_token: str = ""

    # persistence
    persistence_enabled: bool = True
    database_path: str = str(PROJECT_ROOT / "data" / "incidents.db")
    restore_on_start: bool = True

    # misc
    model_path: str = str(PROJECT_ROOT / "models" / "isolation_forest.pkl")
    sample_dataset_path: str = str(PROJECT_ROOT / "data" / "sample" / "baseline_metrics.json")
    log_dir: str = str(PROJECT_ROOT / "logs")
    log_level: str = "INFO"
    random_seed: int = 42

    profiles: dict[str, Any] = field(default_factory=load_cloud_profiles)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("profiles", None)
        return data

    # ── profile helpers ────────────────────────────────────────────────
    @property
    def history_window_size(self) -> int:
        """Rolling window length (env > YAML default)."""
        return max(2, int(self.history_window))

    @property
    def min_history_samples(self) -> int:
        """Samples required before ML scoring is trusted (env > YAML default)."""
        return max(1, int(self.min_history))

    @property
    def severity_factors(self) -> dict[str, float]:
        factors = self.profiles.get("severity_factors") or DEFAULT_PROFILES["severity_factors"]
        return {key: float(value) for key, value in factors.items()}

    @property
    def sla_targets_minutes(self) -> dict[str, float]:
        targets = self.profiles.get("sla_targets_minutes") or DEFAULT_PROFILES["sla_targets_minutes"]
        return {key: float(value) for key, value in targets.items()}


def get_settings(profiles: dict[str, Any] | None = None) -> Settings:
    """Build :class:`Settings` from the environment (plus optional overrides)."""
    _load_dotenv()
    base = load_cloud_profiles()
    return Settings(
        collection_interval_seconds=_env_float("COLLECTION_INTERVAL", 5.0),
        dedup_window_seconds=_env_int("DEDUP_WINDOW", 60),
        auto_resolve_minutes=_env_int("AUTO_RESOLVE_MINUTES", 10),
        inject_every_ticks=_env_int("INJECT_EVERY_TICKS", 6),
        model_contamination=_env_float("MODEL_CONTAMINATION", 0.02),
        model_n_estimators=_env_int("MODEL_N_ESTIMATORS", 150),
        baseline_samples=_env_int("BASELINE_SAMPLES", 500),
        history_window=_env_int("HISTORY_WINDOW", base["defaults"].get("history_window", 30)),
        min_history=_env_int("MIN_HISTORY", base["defaults"].get("min_history", 5)),
        drift_threshold=_env_float("DRIFT_THRESHOLD", 0.35),
        drift_min_batches=_env_int("DRIFT_MIN_BATCHES", 10),
        ml_min_confidence=_env_float("ML_MIN_CONFIDENCE", 0.6),
        slack_webhook_url=_env_str("SLACK_WEBHOOK_URL", ""),
        pagerduty_routing_key=_env_str("PAGERDUTY_ROUTING_KEY", ""),
        email_to=_env_str("EMAIL_TO", ""),
        email_from=_env_str("EMAIL_FROM", ""),
        email_smtp_host=_env_str("EMAIL_SMTP_HOST", ""),
        email_smtp_port=_env_int("EMAIL_SMTP_PORT", 587),
        email_username=_env_str("EMAIL_USERNAME", ""),
        email_password=_env_str("EMAIL_PASSWORD", ""),
        webhook_url=_env_str("WEBHOOK_URL", ""),
        quiet_mode=_env_bool("QUIET_MODE", False),
        notifier_timeout_seconds=_env_float("NOTIFIER_TIMEOUT_SECONDS", 5.0),
        notifier_max_retries=_env_int("NOTIFIER_MAX_RETRIES", 2),
        notifier_cooldown_seconds=_env_float("NOTIFIER_COOLDOWN_SECONDS", 5.0),
        api_enabled=_env_bool("API_ENABLED", False),
        api_host=_env_str("API_HOST", "0.0.0.0"),
        api_port=_env_int("API_PORT", 8080),
        api_token=_env_str("API_TOKEN", ""),
        persistence_enabled=_env_bool("PERSISTENCE_ENABLED", True),
        database_path=_env_str("DATABASE_PATH", str(PROJECT_ROOT / "data" / "incidents.db")),
        restore_on_start=_env_bool("RESTORE_ON_START", True),
        model_path=_env_str("MODEL_PATH", str(PROJECT_ROOT / "models" / "isolation_forest.pkl")),
        sample_dataset_path=_env_str(
            "SAMPLE_DATASET_PATH", str(PROJECT_ROOT / "data" / "sample" / "baseline_metrics.json")
        ),
        log_dir=_env_str("LOG_DIR", str(PROJECT_ROOT / "logs")),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        random_seed=_env_int("RANDOM_SEED", 42),
        profiles=profiles if profiles is not None else base,
    )


SETTINGS = get_settings()

# ── Module level config API ────────────────────────────────────────────
COLLECTION_INTERVAL_SECONDS = SETTINGS.collection_interval_seconds
DEDUP_WINDOW_SECONDS = SETTINGS.dedup_window_seconds
AUTO_RESOLVE_MINUTES = SETTINGS.auto_resolve_minutes
INJECT_EVERY_TICKS = SETTINGS.inject_every_ticks

MODEL_CONTAMINATION = SETTINGS.model_contamination
MODEL_N_ESTIMATORS = SETTINGS.model_n_estimators
BASELINE_SAMPLES = SETTINGS.baseline_samples
HISTORY_WINDOW = SETTINGS.history_window
MIN_HISTORY = SETTINGS.min_history
DRIFT_THRESHOLD = SETTINGS.drift_threshold
DRIFT_MIN_BATCHES = SETTINGS.drift_min_batches
ML_MIN_CONFIDENCE = SETTINGS.ml_min_confidence

SLACK_WEBHOOK_URL = SETTINGS.slack_webhook_url
PAGERDUTY_ROUTING_KEY = SETTINGS.pagerduty_routing_key
EMAIL_TO = SETTINGS.email_to
EMAIL_SMTP_HOST = SETTINGS.email_smtp_host
WEBHOOK_URL = SETTINGS.webhook_url
QUIET_MODE = SETTINGS.quiet_mode
NOTIFIER_TIMEOUT_SECONDS = SETTINGS.notifier_timeout_seconds
NOTIFIER_MAX_RETRIES = SETTINGS.notifier_max_retries
NOTIFIER_COOLDOWN_SECONDS = SETTINGS.notifier_cooldown_seconds

API_ENABLED = SETTINGS.api_enabled
API_HOST = SETTINGS.api_host
API_PORT = SETTINGS.api_port
API_TOKEN = SETTINGS.api_token

PERSISTENCE_ENABLED = SETTINGS.persistence_enabled
DATABASE_PATH = SETTINGS.database_path
RESTORE_ON_START = SETTINGS.restore_on_start

MODEL_PATH = SETTINGS.model_path
SAMPLE_DATASET_PATH = SETTINGS.sample_dataset_path
LOG_DIR = SETTINGS.log_dir
LOG_LEVEL = SETTINGS.log_level
RANDOM_SEED = SETTINGS.random_seed

CLOUD_PROFILES = SETTINGS.profiles


def enable_utf8_console() -> bool:
    """Make stdout/stderr UTF-8 tolerant so Rich can draw box lines and sparklines.

    Legacy Windows consoles default to cp1252 and raise ``UnicodeEncodeError``
    on the dashboard's block characters. Returns ``True`` when a stream was
    reconfigured.
    """
    import sys

    changed = False
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # pragma: no cover - non-standard stream
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
            changed = True
        except (ValueError, OSError, LookupError):  # pragma: no cover - already utf-8 / closed
            continue
    return changed


def setup_logging(level: str | None = None, log_dir: str | os.PathLike[str] | None = None) -> None:
    """Configure root logging once, writing to ``logs/system.log`` and stdout."""
    import logging

    resolved_level = (level or LOG_LEVEL or "INFO").upper()
    directory = Path(log_dir or LOG_DIR)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        file_handler: logging.Handler = logging.FileHandler(directory / "system.log", encoding="utf-8")
    except OSError:  # pragma: no cover - read-only filesystems
        file_handler = logging.NullHandler()

    logging.basicConfig(
        level=getattr(logging, resolved_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[file_handler, logging.StreamHandler()],
        force=True,
    )
