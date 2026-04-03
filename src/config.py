"""
Config loader for FX-Leopard.
Reads config/config.yaml and exposes a typed config object.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VolatilityConfig:
    atr_multiplier: float = 1.5
    pip_spike_threshold: int = 30
    spike_window_minutes: int = 5


@dataclass
class NotificationsConfig:
    channel: str = "telegram"
    bot_token: str = ""
    chat_id: str = ""
    dedup_window_seconds: int = 60
    rate_limit_delay_seconds: float = 1.5
    max_retries: int = 3


@dataclass
class ApiKeysConfig:
    twelvedata: str = ""
    openai: str = ""
    newsapi: str = ""


@dataclass
class LoggingConfig:
    signal_db: str = "data/signals.db"
    log_level: str = "INFO"


@dataclass
class CalendarConfig:
    pre_event_alert_minutes: int = 15
    post_event_check_delay_minutes: int = 2
    min_impact: str = "high"
    refresh_interval_minutes: int = 60
    sources: List[str] = field(default_factory=lambda: ["forexfactory_json"])


@dataclass
class AppConfig:
    pairs: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    confluence_threshold: float = 7.0
    watch_threshold: float = 5.0
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    api_keys: ApiKeysConfig = field(default_factory=ApiKeysConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    calendar: CalendarConfig = field(default_factory=CalendarConfig)


def _resolve_env(value: str) -> str:
    """Resolve ${ENV_VAR} placeholders from environment variables."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.environ.get(env_var, "")
    return value


def load_config(path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to the config YAML file. Defaults to config/config.yaml
              relative to the project root.

    Returns:
        AppConfig instance with all settings populated.
    """
    if path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, "config", "config.yaml")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    api_keys_raw = raw.get("api_keys", {})
    notifications_raw = raw.get("notifications", {})

    volatility_raw = raw.get("volatility", {})
    volatility = VolatilityConfig(
        atr_multiplier=volatility_raw.get("atr_multiplier", 1.5),
        pip_spike_threshold=volatility_raw.get("pip_spike_threshold", 30),
        spike_window_minutes=volatility_raw.get("spike_window_minutes", 5),
    )

    notifications = NotificationsConfig(
        channel=notifications_raw.get("channel", "telegram"),
        bot_token=_resolve_env(notifications_raw.get("bot_token", "")),
        chat_id=_resolve_env(notifications_raw.get("chat_id", "")),
        dedup_window_seconds=int(notifications_raw.get("dedup_window_seconds", 60)),
        rate_limit_delay_seconds=float(notifications_raw.get("rate_limit_delay_seconds", 1.5)),
        max_retries=int(notifications_raw.get("max_retries", 3)),
    )

    api_keys = ApiKeysConfig(
        twelvedata=_resolve_env(api_keys_raw.get("twelvedata", "")),
        openai=_resolve_env(api_keys_raw.get("openai", "")),
        newsapi=_resolve_env(api_keys_raw.get("newsapi", "")),
    )

    logging_raw = raw.get("logging", {})
    logging_cfg = LoggingConfig(
        signal_db=logging_raw.get("signal_db", "data/signals.db"),
        log_level=logging_raw.get("log_level", "INFO"),
    )

    calendar_raw = raw.get("calendar", {})
    calendar = CalendarConfig(
        pre_event_alert_minutes=calendar_raw.get("pre_event_alert_minutes", 15),
        post_event_check_delay_minutes=calendar_raw.get("post_event_check_delay_minutes", 2),
        min_impact=calendar_raw.get("min_impact", "high"),
        refresh_interval_minutes=calendar_raw.get("refresh_interval_minutes", 60),
        sources=calendar_raw.get("sources", ["forexfactory_json"]),
    )

    return AppConfig(
        pairs=raw.get("pairs", []),
        timeframes=raw.get("timeframes", []),
        confluence_threshold=float(raw.get("confluence_threshold", 7.0)),
        watch_threshold=float(raw.get("watch_threshold", 5.0)),
        volatility=volatility,
        notifications=notifications,
        api_keys=api_keys,
        logging=logging_cfg,
        calendar=calendar,
    )
