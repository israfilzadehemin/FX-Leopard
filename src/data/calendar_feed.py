"""
Economic calendar feed for FX-Leopard.

Fetches upcoming high-impact economic events from ForexFactory JSON feed,
schedules pre-event Telegram alerts before each event, and fires post-event
alerts after data is released — giving the trader time to prepare and react.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, List, Optional

import requests
from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ForexFactory JSON feed URL
# ---------------------------------------------------------------------------

FOREXFACTORY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# ---------------------------------------------------------------------------
# Country → affected FX pairs mapping
# ---------------------------------------------------------------------------

COUNTRY_PAIRS_MAP: Dict[str, List[str]] = {
    "USD": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "USOIL", "XAUUSD"],
    "EUR": ["EURUSD", "EURGBP"],
    "GBP": ["GBPUSD", "EURGBP"],
    "JPY": ["USDJPY"],
    "AUD": ["AUDUSD"],
    "CAD": ["USDCAD", "UKOIL"],
    "CHF": ["USDCHF"],
    "NZD": ["NZDUSD"],
}

# Country flag emoji map for alert formatting
COUNTRY_FLAGS: Dict[str, str] = {
    "USD": "🇺🇸",
    "EUR": "🇪🇺",
    "GBP": "🇬🇧",
    "JPY": "🇯🇵",
    "AUD": "🇦🇺",
    "CAD": "🇨🇦",
    "CHF": "🇨🇭",
    "NZD": "🇳🇿",
}

# Impact level ordering for filtering
IMPACT_LEVELS = {"high": 3, "medium": 2, "low": 1}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EconomicEvent:
    """Structured representation of a single economic calendar event."""

    title: str
    country: str
    datetime: str           # ISO-8601 UTC string, e.g. "2026-04-03T12:30:00Z"
    impact: str             # "high" | "medium" | "low"
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    affected_pairs: List[str] = field(default_factory=list)

    def get_datetime(self) -> datetime:
        """Parse the ISO-8601 datetime string into a timezone-aware datetime."""
        dt_str = self.datetime
        # Handle both "Z" suffix and "+00:00" suffix
        if dt_str.endswith("Z"):
            dt_str = dt_str[:-1] + "+00:00"
        return datetime.fromisoformat(dt_str).astimezone(timezone.utc)

    def is_released(self) -> bool:
        """Return True if the event time has already passed."""
        return datetime.now(timezone.utc) >= self.get_datetime()


# ---------------------------------------------------------------------------
# Alert formatters
# ---------------------------------------------------------------------------


def format_pre_event_alert(event: EconomicEvent, minutes_until: int) -> str:
    """
    Format a pre-event Telegram alert message.

    Args:
        event: The upcoming economic event.
        minutes_until: How many minutes until the event fires.

    Returns:
        Formatted alert string ready for Telegram.
    """
    flag = COUNTRY_FLAGS.get(event.country, "🌐")
    pairs_str = ", ".join(event.affected_pairs) if event.affected_pairs else "—"

    try:
        event_time = event.get_datetime()
        time_str = event_time.strftime("%H:%M UTC")
    except Exception:
        time_str = event.datetime

    lines = [
        f"📅 UPCOMING HIGH IMPACT EVENT",
        "",
        f"{flag} {event.title} ({event.country})",
        f"⏰ In {minutes_until} minutes — {time_str}",
        "",
    ]

    if event.forecast is not None:
        lines.append(f"📊 Forecast: {event.forecast}")
    if event.previous is not None:
        lines.append(f"📊 Previous: {event.previous}")
    if event.forecast is not None or event.previous is not None:
        lines.append("")

    lines += [
        f"⚡ Affected pairs: {pairs_str}",
        "",
        "⚠️ Consider tightening stops or staying out until dust settles.",
    ]

    return "\n".join(lines)


def format_post_event_alert(event: EconomicEvent) -> str:
    """
    Format a post-event release Telegram alert message.

    Classifies the release as BEAT / MISSED / INLINE relative to forecast,
    then generates a directional market reaction hint.

    Args:
        event: The economic event with ``actual`` value populated.

    Returns:
        Formatted alert string ready for Telegram.
    """
    flag = COUNTRY_FLAGS.get(event.country, "🌐")
    actual = event.actual or "N/A"

    # --- Determine beat/miss/inline ---
    result_label, direction = _classify_release(event)

    # --- Build market reaction hint ---
    reaction_line = _build_reaction_line(event, direction)

    lines = [
        "📰 EVENT RELEASED",
        "",
        f"{flag} {event.title} ({event.country})",
        "",
        f"{'✅' if result_label == 'BEAT' else '❌' if result_label == 'MISSED' else '➡️'} "
        f"Actual:   {actual}  ← {result_label}",
    ]

    if event.forecast is not None:
        lines.append(f"📊 Forecast: {event.forecast}")
    if event.previous is not None:
        lines.append(f"📊 Previous: {event.previous}")

    if reaction_line:
        lines += ["", reaction_line]

    return "\n".join(lines)


def _classify_release(event: EconomicEvent):
    """
    Compare actual vs forecast and return (label, direction) tuple.

    direction: +1 means stronger-than-expected (bullish country currency),
               -1 means weaker-than-expected, 0 means inline.
    """
    if event.actual is None or event.forecast is None:
        return "N/A", 0

    actual_val = _parse_numeric(event.actual)
    forecast_val = _parse_numeric(event.forecast)

    if actual_val is None or forecast_val is None:
        return "N/A", 0

    if actual_val > forecast_val:
        return "BEAT", 1
    elif actual_val < forecast_val:
        return "MISSED", -1
    else:
        return "INLINE", 0


def _parse_numeric(value: str) -> Optional[float]:
    """Strip common suffixes (K, M, B, %) and parse to float."""
    if not value:
        return None
    v = value.strip().replace(",", "").upper()
    multiplier = 1.0
    if v.endswith("B"):
        multiplier = 1_000_000_000
        v = v[:-1]
    elif v.endswith("M"):
        multiplier = 1_000_000
        v = v[:-1]
    elif v.endswith("K"):
        multiplier = 1_000
        v = v[:-1]
    elif v.endswith("%"):
        v = v[:-1]
    try:
        return float(v) * multiplier
    except ValueError:
        return None


def _build_reaction_line(event: EconomicEvent, direction: int) -> str:
    """Produce a one-line market reaction hint string."""
    if direction == 0 or not event.affected_pairs:
        return ""

    country = event.country
    pairs = event.affected_pairs

    if direction == 1:
        strength = "strengthen"
        pair_hints = []
        for pair in pairs:
            if pair.startswith(country):
                pair_hints.append(f"{pair} (buy)")
            elif pair.endswith(country):
                pair_hints.append(f"{pair} (sell)")
            else:
                pair_hints.append(f"{pair} (sell)")
        verb = "📈"
    else:
        strength = "weaken"
        pair_hints = []
        for pair in pairs:
            if pair.startswith(country):
                pair_hints.append(f"{pair} (sell)")
            elif pair.endswith(country):
                pair_hints.append(f"{pair} (buy)")
            else:
                pair_hints.append(f"{pair} (buy)")
        verb = "📉"

    hints_str = ", ".join(pair_hints[:5])  # cap to 5 for readability
    return f"{verb} {country} likely to {strength}. Watch: {hints_str}"


# ---------------------------------------------------------------------------
# CalendarFeed
# ---------------------------------------------------------------------------


class CalendarFeed:
    """
    Fetches and monitors the economic calendar from ForexFactory.

    Usage::

        def on_alert(message: str):
            print(message)  # hand off to Telegram notifier

        feed = CalendarFeed(config=cfg_dict, on_alert=on_alert)
        feed.start()   # begins background scheduler
        # ...
        feed.stop()

    Configuration keys (nested under ``calendar:`` in config.yaml)::

        pre_event_alert_minutes: 15
        post_event_check_delay_minutes: 2
        min_impact: high        # high | medium | low
        refresh_interval_minutes: 60
        sources:
          - forexfactory_json
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        on_alert: Optional[Callable[[str], None]] = None,
    ) -> None:
        cfg = config or {}
        cal_cfg = cfg.get("calendar", {})

        self._pre_alert_minutes: int = int(cal_cfg.get("pre_event_alert_minutes", 15))
        self._post_delay_minutes: int = int(cal_cfg.get("post_event_check_delay_minutes", 2))
        self._min_impact: str = cal_cfg.get("min_impact", "high").lower()
        self._refresh_interval: int = int(cal_cfg.get("refresh_interval_minutes", 60))
        self._on_alert: Callable[[str], None] = on_alert or (lambda msg: logger.info(msg))

        self._events: List[EconomicEvent] = []
        self._scheduled_event_ids: set = set()
        self._scheduler: BackgroundScheduler = BackgroundScheduler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background scheduler and do an initial calendar refresh."""
        self._scheduler.start()

        # Initial fetch
        self.refresh_calendar()

        # Periodic refresh
        self._scheduler.add_job(
            self.refresh_calendar,
            trigger="interval",
            minutes=self._refresh_interval,
            id="calendar_refresh",
            replace_existing=True,
        )

        logger.info(
            "CalendarFeed started — pre-alert %d min, min_impact=%s, refresh every %d min",
            self._pre_alert_minutes,
            self._min_impact,
            self._refresh_interval,
        )

    def stop(self) -> None:
        """Shut down the background scheduler gracefully."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        logger.info("CalendarFeed stopped")

    def refresh_calendar(self) -> None:
        """Fetch the calendar, parse events, and schedule alerts."""
        logger.info("Refreshing economic calendar…")
        try:
            events = self._fetch_events()
        except Exception as exc:
            logger.warning("Calendar fetch failed: %s", exc)
            return

        self._events = events
        self._schedule_alerts(events)
        logger.info("Calendar refreshed — %d qualifying events loaded", len(events))

    def get_events(self) -> List[EconomicEvent]:
        """Return the current list of parsed economic events."""
        return list(self._events)

    # ------------------------------------------------------------------
    # Fetching & parsing
    # ------------------------------------------------------------------

    def _fetch_events(self) -> List[EconomicEvent]:
        """
        Fetch events from ForexFactory JSON feed and return parsed events
        filtered by minimum impact level.
        """
        raw_events = self._fetch_forexfactory()
        events: List[EconomicEvent] = []
        for raw in raw_events:
            event = self._parse_event(raw)
            if event is not None and self._meets_impact_filter(event):
                events.append(event)
        return events

    def _fetch_forexfactory(self) -> List[dict]:
        """Download and return the raw JSON list from ForexFactory."""
        response = requests.get(FOREXFACTORY_URL, timeout=15)
        response.raise_for_status()
        return response.json()

    def _parse_event(self, raw: dict) -> Optional[EconomicEvent]:
        """Parse a single raw ForexFactory dict into an EconomicEvent."""
        try:
            title = (raw.get("title") or "").strip()
            country = (raw.get("country") or "").strip().upper()
            impact = (raw.get("impact") or "low").strip().lower()
            forecast = raw.get("forecast") or None
            previous = raw.get("previous") or None
            actual = raw.get("actual") or None

            # Parse the date field from ForexFactory format
            date_str = _parse_ff_datetime(raw)
            if not date_str:
                return None

            affected_pairs = COUNTRY_PAIRS_MAP.get(country, [])

            return EconomicEvent(
                title=title,
                country=country,
                datetime=date_str,
                impact=impact,
                forecast=forecast if forecast else None,
                previous=previous if previous else None,
                actual=actual if actual else None,
                affected_pairs=list(affected_pairs),
            )
        except Exception as exc:
            logger.debug("Failed to parse event %r: %s", raw, exc)
            return None

    def _meets_impact_filter(self, event: EconomicEvent) -> bool:
        """Return True if the event impact level meets the minimum threshold."""
        min_level = IMPACT_LEVELS.get(self._min_impact, 3)
        event_level = IMPACT_LEVELS.get(event.impact, 0)
        return event_level >= min_level

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _schedule_alerts(self, events: List[EconomicEvent]) -> None:
        """Schedule pre-event and post-event alert jobs for each event."""
        now = datetime.now(timezone.utc)

        for event in events:
            try:
                event_time = event.get_datetime()
            except Exception:
                continue

            # --- Pre-event alert ---
            pre_alert_time = event_time - timedelta(minutes=self._pre_alert_minutes)
            pre_job_id = f"pre_{event.country}_{event.title}_{event.datetime}"

            if pre_alert_time > now and pre_job_id not in self._scheduled_event_ids:
                self._scheduler.add_job(
                    self._fire_pre_event_alert,
                    trigger="date",
                    run_date=pre_alert_time,
                    args=[event],
                    id=pre_job_id,
                    replace_existing=True,
                )
                self._scheduled_event_ids.add(pre_job_id)
                logger.debug("Scheduled pre-alert for '%s' at %s", event.title, pre_alert_time)

            # --- Post-event alert ---
            post_alert_time = event_time + timedelta(minutes=self._post_delay_minutes)
            post_job_id = f"post_{event.country}_{event.title}_{event.datetime}"

            if post_alert_time > now and post_job_id not in self._scheduled_event_ids:
                self._scheduler.add_job(
                    self._fire_post_event_alert,
                    trigger="date",
                    run_date=post_alert_time,
                    args=[event],
                    id=post_job_id,
                    replace_existing=True,
                )
                self._scheduled_event_ids.add(post_job_id)
                logger.debug("Scheduled post-alert for '%s' at %s", event.title, post_alert_time)

    def _fire_pre_event_alert(self, event: EconomicEvent) -> None:
        """Called by the scheduler N minutes before an event."""
        message = format_pre_event_alert(event, self._pre_alert_minutes)
        logger.info("Firing pre-event alert: %s", event.title)
        self._on_alert(message)

    def _fire_post_event_alert(self, event: EconomicEvent) -> None:
        """Called by the scheduler after an event — re-fetches to get actual value."""
        logger.info("Firing post-event check: %s", event.title)
        try:
            updated_event = self._refetch_event(event)
        except Exception as exc:
            logger.warning("Post-event refetch failed for '%s': %s", event.title, exc)
            updated_event = event

        message = format_post_event_alert(updated_event)
        self._on_alert(message)

    def _refetch_event(self, event: EconomicEvent) -> EconomicEvent:
        """
        Re-fetch the calendar and return an updated copy of the event with
        the ``actual`` field populated (if released).
        """
        raw_events = self._fetch_forexfactory()
        for raw in raw_events:
            candidate = self._parse_event(raw)
            if candidate is None:
                continue
            if candidate.title == event.title and candidate.country == event.country \
                    and candidate.datetime == event.datetime:
                return candidate
        # No updated data found — return original
        return event


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_ff_datetime(raw: dict) -> Optional[str]:
    """
    Parse the ForexFactory event date/time into an ISO-8601 UTC string.

    ForexFactory JSON contains ``"date"`` (e.g. ``"04-03-2026"``) and
    ``"time"`` (e.g. ``"8:30am"``).  If ``"time"`` is empty the event is
    considered an all-day event and midnight UTC is used.
    """
    date_str = raw.get("date", "")
    time_str = raw.get("time", "")

    if not date_str:
        return None

    try:
        # Try ISO format first (YYYY-MM-DDTHH:MM:SSZ)
        if "T" in date_str:
            if not date_str.endswith("Z"):
                date_str = date_str + "Z"
            return date_str

        # ForexFactory format: "04-03-2026" and "8:30am"
        if not time_str or time_str.strip() == "":
            time_str = "12:00am"

        dt_str = f"{date_str} {time_str}"
        dt = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p")
        dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        try:
            # Try just the date
            dt = datetime.strptime(date_str.strip(), "%m-%d-%Y")
            dt = dt.replace(tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return None
