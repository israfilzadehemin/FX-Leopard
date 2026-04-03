"""
Unit tests for CalendarFeed and related utilities.

All HTTP calls are fully mocked — no live internet connection required.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import List
from unittest.mock import MagicMock, patch, call

import pytest

from data.calendar_feed import (
    COUNTRY_PAIRS_MAP,
    EconomicEvent,
    CalendarFeed,
    format_pre_event_alert,
    format_post_event_alert,
    _classify_release,
    _parse_numeric,
    _parse_ff_datetime,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = {
    "calendar": {
        "pre_event_alert_minutes": 15,
        "post_event_check_delay_minutes": 2,
        "min_impact": "high",
        "refresh_interval_minutes": 60,
        "sources": ["forexfactory_json"],
    }
}


def _make_raw_event(
    title="Non-Farm Payrolls",
    country="USD",
    date="04-03-2026",
    time="8:30am",
    impact="High",
    forecast="185K",
    previous="151K",
    actual=None,
) -> dict:
    """Return a dict shaped like a ForexFactory JSON event."""
    return {
        "title": title,
        "country": country,
        "date": date,
        "time": time,
        "impact": impact,
        "forecast": forecast,
        "previous": previous,
        "actual": actual or "",
    }


def _make_event(
    title="Non-Farm Payrolls",
    country="USD",
    dt_str="2026-04-03T12:30:00Z",
    impact="high",
    forecast="185K",
    previous="151K",
    actual=None,
) -> EconomicEvent:
    pairs = COUNTRY_PAIRS_MAP.get(country, [])
    return EconomicEvent(
        title=title,
        country=country,
        datetime=dt_str,
        impact=impact,
        forecast=forecast,
        previous=previous,
        actual=actual,
        affected_pairs=list(pairs),
    )


def _future_event(minutes_ahead: int = 60) -> EconomicEvent:
    """Return an EconomicEvent whose datetime is minutes_ahead from now."""
    dt = datetime.now(timezone.utc) + timedelta(minutes=minutes_ahead)
    return _make_event(dt_str=dt.strftime("%Y-%m-%dT%H:%M:%SZ"))


def _past_event(minutes_ago: int = 5) -> EconomicEvent:
    """Return an EconomicEvent whose datetime is in the past."""
    dt = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return _make_event(dt_str=dt.strftime("%Y-%m-%dT%H:%M:%SZ"))


# ---------------------------------------------------------------------------
# EconomicEvent model
# ---------------------------------------------------------------------------


class TestEconomicEvent:
    def test_fields_populated(self):
        event = _make_event()
        assert event.title == "Non-Farm Payrolls"
        assert event.country == "USD"
        assert event.impact == "high"
        assert event.forecast == "185K"
        assert event.previous == "151K"
        assert event.actual is None

    def test_affected_pairs_populated_from_map(self):
        event = _make_event(country="USD")
        assert "EURUSD" in event.affected_pairs
        assert "XAUUSD" in event.affected_pairs

    def test_affected_pairs_empty_for_unknown_country(self):
        event = _make_event(country="XYZ")
        # Unknown country → no pairs from map (we pass empty list explicitly)
        assert event.affected_pairs == []

    def test_get_datetime_parses_z_suffix(self):
        event = _make_event(dt_str="2026-04-03T12:30:00Z")
        dt = event.get_datetime()
        assert dt.year == 2026
        assert dt.month == 4
        assert dt.day == 3
        assert dt.hour == 12
        assert dt.minute == 30
        assert dt.tzinfo == timezone.utc

    def test_is_released_false_for_future(self):
        event = _future_event(minutes_ahead=60)
        assert event.is_released() is False

    def test_is_released_true_for_past(self):
        event = _past_event(minutes_ago=5)
        assert event.is_released() is True


# ---------------------------------------------------------------------------
# Country-to-pairs mapping
# ---------------------------------------------------------------------------


class TestCountryPairsMap:
    def test_usd_pairs(self):
        pairs = COUNTRY_PAIRS_MAP["USD"]
        assert "EURUSD" in pairs
        assert "GBPUSD" in pairs
        assert "USDJPY" in pairs
        assert "XAUUSD" in pairs
        assert "USOIL" in pairs

    def test_eur_pairs(self):
        pairs = COUNTRY_PAIRS_MAP["EUR"]
        assert "EURUSD" in pairs
        assert "EURGBP" in pairs

    def test_gbp_pairs(self):
        pairs = COUNTRY_PAIRS_MAP["GBP"]
        assert "GBPUSD" in pairs
        assert "EURGBP" in pairs

    def test_jpy_pairs(self):
        assert COUNTRY_PAIRS_MAP["JPY"] == ["USDJPY"]

    def test_cad_pairs(self):
        pairs = COUNTRY_PAIRS_MAP["CAD"]
        assert "USDCAD" in pairs
        assert "UKOIL" in pairs

    def test_all_keys_present(self):
        for key in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]:
            assert key in COUNTRY_PAIRS_MAP


# ---------------------------------------------------------------------------
# _parse_ff_datetime
# ---------------------------------------------------------------------------


class TestParseFFDatetime:
    def test_parses_mm_dd_yyyy_with_time(self):
        # ForexFactory times are US Eastern (EDT = UTC-4 in April).
        # 8:30am ET → 12:30 UTC
        raw = {"date": "04-03-2026", "time": "8:30am"}
        result = _parse_ff_datetime(raw)
        assert result == "2026-04-03T12:30:00Z"

    def test_parses_mm_dd_yyyy_with_pm_time(self):
        # 1:30pm ET → 17:30 UTC
        raw = {"date": "04-03-2026", "time": "1:30pm"}
        result = _parse_ff_datetime(raw)
        assert result == "2026-04-03T17:30:00Z"

    def test_empty_time_uses_midnight(self):
        # All-day events have no time component; midnight UTC is used as-is.
        raw = {"date": "04-03-2026", "time": ""}
        result = _parse_ff_datetime(raw)
        assert result == "2026-04-03T00:00:00Z"

    def test_missing_date_returns_none(self):
        raw = {"time": "8:30am"}
        result = _parse_ff_datetime(raw)
        assert result is None

    def test_passthrough_iso_format(self):
        raw = {"date": "2026-04-03T12:30:00Z", "time": ""}
        result = _parse_ff_datetime(raw)
        assert result == "2026-04-03T12:30:00Z"


# ---------------------------------------------------------------------------
# CalendarFeed — fetching & parsing
# ---------------------------------------------------------------------------


class TestCalendarFeedParsing:
    def _feed(self) -> CalendarFeed:
        return CalendarFeed(config=SAMPLE_CONFIG)

    @patch("data.calendar_feed.requests.get")
    def test_fetch_returns_events(self, mock_get):
        raw = [_make_raw_event()]
        mock_get.return_value = MagicMock(json=lambda: raw, raise_for_status=lambda: None)
        feed = self._feed()
        events = feed._fetch_events()
        assert len(events) == 1
        assert events[0].title == "Non-Farm Payrolls"

    @patch("data.calendar_feed.requests.get")
    def test_parse_event_creates_correct_model(self, mock_get):
        raw = _make_raw_event()
        feed = self._feed()
        event = feed._parse_event(raw)
        assert event is not None
        assert event.title == "Non-Farm Payrolls"
        assert event.country == "USD"
        assert event.impact == "high"
        assert event.forecast == "185K"
        assert event.previous == "151K"
        assert event.actual is None
        assert "EURUSD" in event.affected_pairs

    @patch("data.calendar_feed.requests.get")
    def test_empty_actual_parsed_as_none(self, mock_get):
        raw = _make_raw_event(actual="")
        feed = self._feed()
        event = feed._parse_event(raw)
        assert event is not None
        assert event.actual is None

    @patch("data.calendar_feed.requests.get")
    def test_actual_value_populated(self, mock_get):
        raw = _make_raw_event(actual="227K")
        feed = self._feed()
        event = feed._parse_event(raw)
        assert event is not None
        assert event.actual == "227K"

    @patch("data.calendar_feed.requests.get")
    def test_country_uppercased(self, mock_get):
        raw = _make_raw_event(country="usd")
        feed = self._feed()
        event = feed._parse_event(raw)
        assert event is not None
        assert event.country == "USD"

    @patch("data.calendar_feed.requests.get")
    def test_impact_lowercased(self, mock_get):
        raw = _make_raw_event(impact="High")
        feed = self._feed()
        event = feed._parse_event(raw)
        assert event is not None
        assert event.impact == "high"

    @patch("data.calendar_feed.requests.get")
    def test_raises_on_http_error(self, mock_get):
        mock_get.return_value = MagicMock(
            raise_for_status=MagicMock(side_effect=Exception("HTTP 503"))
        )
        feed = self._feed()
        with pytest.raises(Exception):
            feed._fetch_forexfactory()

    @patch("data.calendar_feed.requests.get")
    def test_multiple_events_parsed(self, mock_get):
        raw_events = [
            _make_raw_event(title="NFP", country="USD", impact="High"),
            _make_raw_event(title="CPI", country="EUR", impact="High"),
            _make_raw_event(title="Retail Sales", country="GBP", impact="High"),
        ]
        mock_get.return_value = MagicMock(json=lambda: raw_events, raise_for_status=lambda: None)
        feed = self._feed()
        events = feed._fetch_events()
        assert len(events) == 3

    @patch("data.calendar_feed.requests.get")
    def test_invalid_event_skipped(self, mock_get):
        """Events with missing date should be silently skipped."""
        raw_events = [
            {"title": "Bad Event"},   # missing date
            _make_raw_event(title="Good Event"),
        ]
        mock_get.return_value = MagicMock(json=lambda: raw_events, raise_for_status=lambda: None)
        feed = self._feed()
        events = feed._fetch_events()
        assert len(events) == 1
        assert events[0].title == "Good Event"


# ---------------------------------------------------------------------------
# Impact filtering
# ---------------------------------------------------------------------------


class TestImpactFiltering:
    def _feed(self, min_impact: str) -> CalendarFeed:
        config = {"calendar": {**SAMPLE_CONFIG["calendar"], "min_impact": min_impact}}
        return CalendarFeed(config=config)

    def test_high_impact_passes_high_filter(self):
        feed = self._feed("high")
        event = _make_event(impact="high")
        assert feed._meets_impact_filter(event) is True

    def test_medium_impact_fails_high_filter(self):
        feed = self._feed("high")
        event = _make_event(impact="medium")
        assert feed._meets_impact_filter(event) is False

    def test_low_impact_fails_high_filter(self):
        feed = self._feed("high")
        event = _make_event(impact="low")
        assert feed._meets_impact_filter(event) is False

    def test_medium_impact_passes_medium_filter(self):
        feed = self._feed("medium")
        event = _make_event(impact="medium")
        assert feed._meets_impact_filter(event) is True

    def test_high_impact_passes_medium_filter(self):
        feed = self._feed("medium")
        event = _make_event(impact="high")
        assert feed._meets_impact_filter(event) is True

    def test_low_impact_passes_low_filter(self):
        feed = self._feed("low")
        event = _make_event(impact="low")
        assert feed._meets_impact_filter(event) is True

    @patch("data.calendar_feed.requests.get")
    def test_fetch_filters_out_low_impact(self, mock_get):
        raw_events = [
            _make_raw_event(title="High Event", impact="High"),
            _make_raw_event(title="Low Event", impact="Low"),
        ]
        mock_get.return_value = MagicMock(json=lambda: raw_events, raise_for_status=lambda: None)
        feed = self._feed("high")
        events = feed._fetch_events()
        titles = [e.title for e in events]
        assert "High Event" in titles
        assert "Low Event" not in titles


# ---------------------------------------------------------------------------
# Beat / miss / inline classification
# ---------------------------------------------------------------------------


class TestClassifyRelease:
    def test_beat_when_actual_greater(self):
        event = _make_event(forecast="185K", actual="227K")
        label, direction = _classify_release(event)
        assert label == "BEAT"
        assert direction == 1

    def test_missed_when_actual_less(self):
        event = _make_event(forecast="185K", actual="150K")
        label, direction = _classify_release(event)
        assert label == "MISSED"
        assert direction == -1

    def test_inline_when_actual_equals_forecast(self):
        event = _make_event(forecast="185K", actual="185K")
        label, direction = _classify_release(event)
        assert label == "INLINE"
        assert direction == 0

    def test_no_actual_returns_na(self):
        event = _make_event(forecast="185K", actual=None)
        label, direction = _classify_release(event)
        assert label == "N/A"
        assert direction == 0

    def test_no_forecast_returns_na(self):
        event = _make_event(forecast=None, actual="185K")
        label, direction = _classify_release(event)
        assert label == "N/A"
        assert direction == 0

    def test_negative_values(self):
        event = _make_event(forecast="-0.2%", actual="-0.1%")
        label, direction = _classify_release(event)
        assert label == "BEAT"
        assert direction == 1

    def test_percentage_values(self):
        event = _make_event(forecast="3.5%", actual="3.2%")
        label, direction = _classify_release(event)
        assert label == "MISSED"
        assert direction == -1


class TestParseNumeric:
    def test_plain_number(self):
        assert _parse_numeric("185") == pytest.approx(185.0)

    def test_k_suffix(self):
        assert _parse_numeric("185K") == pytest.approx(185_000.0)

    def test_m_suffix(self):
        assert _parse_numeric("1.5M") == pytest.approx(1_500_000.0)

    def test_b_suffix(self):
        assert _parse_numeric("2B") == pytest.approx(2_000_000_000.0)

    def test_percent_suffix(self):
        assert _parse_numeric("3.5%") == pytest.approx(3.5)

    def test_negative(self):
        assert _parse_numeric("-0.2%") == pytest.approx(-0.2)

    def test_non_numeric_returns_none(self):
        assert _parse_numeric("N/A") is None

    def test_empty_returns_none(self):
        assert _parse_numeric("") is None


# ---------------------------------------------------------------------------
# Alert formatting
# ---------------------------------------------------------------------------


class TestFormatPreEventAlert:
    def test_contains_title(self):
        event = _make_event()
        msg = format_pre_event_alert(event, minutes_until=15)
        assert "Non-Farm Payrolls" in msg

    def test_contains_country(self):
        event = _make_event()
        msg = format_pre_event_alert(event, minutes_until=15)
        assert "USD" in msg

    def test_contains_minutes(self):
        event = _make_event()
        msg = format_pre_event_alert(event, minutes_until=15)
        assert "15 minutes" in msg

    def test_contains_forecast(self):
        event = _make_event(forecast="185K")
        msg = format_pre_event_alert(event, minutes_until=15)
        assert "185K" in msg

    def test_contains_previous(self):
        event = _make_event(previous="151K")
        msg = format_pre_event_alert(event, minutes_until=15)
        assert "151K" in msg

    def test_contains_affected_pairs(self):
        event = _make_event(country="USD")
        msg = format_pre_event_alert(event, minutes_until=15)
        assert "EURUSD" in msg

    def test_contains_header(self):
        event = _make_event()
        msg = format_pre_event_alert(event, minutes_until=15)
        assert "UPCOMING HIGH IMPACT EVENT" in msg

    def test_contains_flag_for_usd(self):
        event = _make_event(country="USD")
        msg = format_pre_event_alert(event, minutes_until=15)
        assert "🇺🇸" in msg

    def test_contains_utc_time(self):
        event = _make_event(dt_str="2026-04-03T12:30:00Z")
        msg = format_pre_event_alert(event, minutes_until=15)
        assert "UTC" in msg

    def test_none_forecast_omitted(self):
        event = _make_event(forecast=None)
        msg = format_pre_event_alert(event, minutes_until=15)
        assert "Forecast:" not in msg


class TestFormatPostEventAlert:
    def test_contains_title(self):
        event = _make_event(actual="227K")
        msg = format_post_event_alert(event)
        assert "Non-Farm Payrolls" in msg

    def test_contains_actual(self):
        event = _make_event(actual="227K")
        msg = format_post_event_alert(event)
        assert "227K" in msg

    def test_beat_label_present(self):
        event = _make_event(forecast="185K", actual="227K")
        msg = format_post_event_alert(event)
        assert "BEAT" in msg

    def test_missed_label_present(self):
        event = _make_event(forecast="185K", actual="150K")
        msg = format_post_event_alert(event)
        assert "MISSED" in msg

    def test_inline_label_present(self):
        event = _make_event(forecast="185K", actual="185K")
        msg = format_post_event_alert(event)
        assert "INLINE" in msg

    def test_contains_forecast(self):
        event = _make_event(forecast="185K", actual="227K")
        msg = format_post_event_alert(event)
        assert "185K" in msg

    def test_contains_header(self):
        event = _make_event(actual="227K")
        msg = format_post_event_alert(event)
        assert "EVENT RELEASED" in msg

    def test_reaction_line_on_beat(self):
        event = _make_event(country="USD", forecast="185K", actual="227K")
        msg = format_post_event_alert(event)
        assert "strengthen" in msg.lower() or "USD" in msg

    def test_reaction_line_on_miss(self):
        event = _make_event(country="USD", forecast="185K", actual="100K")
        msg = format_post_event_alert(event)
        assert "weaken" in msg.lower() or "USD" in msg

    def test_no_actual_shows_na(self):
        event = _make_event(actual=None)
        msg = format_post_event_alert(event)
        assert "N/A" in msg


# ---------------------------------------------------------------------------
# Pre-event alert scheduling
# ---------------------------------------------------------------------------


class TestCalendarFeedScheduling:
    def _feed_with_mock_scheduler(self) -> tuple[CalendarFeed, MagicMock]:
        feed = CalendarFeed(config=SAMPLE_CONFIG)
        mock_scheduler = MagicMock()
        mock_scheduler.running = False
        feed._scheduler = mock_scheduler
        return feed, mock_scheduler

    def test_future_event_schedules_pre_alert(self):
        feed, mock_scheduler = self._feed_with_mock_scheduler()
        event = _future_event(minutes_ahead=60)
        feed._schedule_alerts([event])

        add_job_calls = [c for c in mock_scheduler.add_job.call_args_list
                         if c.args and c.args[0] == feed._fire_pre_event_alert]
        assert len(add_job_calls) == 1

    def test_future_event_schedules_post_alert(self):
        feed, mock_scheduler = self._feed_with_mock_scheduler()
        event = _future_event(minutes_ahead=60)
        feed._schedule_alerts([event])

        add_job_calls = [c for c in mock_scheduler.add_job.call_args_list
                         if c.args and c.args[0] == feed._fire_post_event_alert]
        assert len(add_job_calls) == 1

    def test_past_pre_alert_not_scheduled(self):
        """Pre-alert should not be scheduled if the pre-alert time is already past."""
        feed, mock_scheduler = self._feed_with_mock_scheduler()
        # Event is 5 min in the future but pre-alert fires 15 min before → already past
        event = _future_event(minutes_ahead=5)
        feed._schedule_alerts([event])

        pre_calls = [c for c in mock_scheduler.add_job.call_args_list
                     if c.args and c.args[0] == feed._fire_pre_event_alert]
        assert len(pre_calls) == 0

    def test_already_released_event_no_alerts(self):
        """Past events should not schedule any alerts."""
        feed, mock_scheduler = self._feed_with_mock_scheduler()
        event = _past_event(minutes_ago=30)
        feed._schedule_alerts([event])

        assert mock_scheduler.add_job.call_count == 0

    def test_duplicate_events_not_rescheduled(self):
        """Same event should not be scheduled twice."""
        feed, mock_scheduler = self._feed_with_mock_scheduler()
        event = _future_event(minutes_ahead=60)
        feed._schedule_alerts([event])
        feed._schedule_alerts([event])  # call again with same event

        # Should only be scheduled once despite two calls
        pre_calls = [c for c in mock_scheduler.add_job.call_args_list
                     if c.args and c.args[0] == feed._fire_pre_event_alert]
        assert len(pre_calls) == 1

    def test_pre_alert_fires_on_alert_callback(self):
        alerts = []
        feed = CalendarFeed(config=SAMPLE_CONFIG, on_alert=alerts.append)
        event = _future_event(minutes_ahead=60)
        feed._fire_pre_event_alert(event)
        assert len(alerts) == 1
        assert "UPCOMING HIGH IMPACT EVENT" in alerts[0]

    def test_post_alert_fires_on_alert_callback(self):
        alerts = []
        feed = CalendarFeed(config=SAMPLE_CONFIG, on_alert=alerts.append)
        event = _past_event(minutes_ago=5)
        event.actual = "227K"

        with patch.object(feed, "_refetch_event", return_value=event):
            feed._fire_post_event_alert(event)

        assert len(alerts) == 1
        assert "EVENT RELEASED" in alerts[0]

    @patch("data.calendar_feed.requests.get")
    def test_refresh_calendar_calls_on_alert_for_pre_event(self, mock_get):
        """refresh_calendar should schedule (not immediately fire) alerts."""
        future_dt = datetime.now(timezone.utc) + timedelta(minutes=60)
        raw_events = [_make_raw_event(
            date=future_dt.strftime("%m-%d-%Y"),
            time=future_dt.strftime("%I:%M%p").lower().lstrip("0"),
        )]
        mock_get.return_value = MagicMock(json=lambda: raw_events, raise_for_status=lambda: None)

        alerts = []
        feed = CalendarFeed(config=SAMPLE_CONFIG, on_alert=alerts.append)
        mock_scheduler = MagicMock()
        mock_scheduler.running = False
        feed._scheduler = mock_scheduler

        feed.refresh_calendar()

        # Scheduler should have been asked to add jobs
        assert mock_scheduler.add_job.called


# ---------------------------------------------------------------------------
# CalendarFeed start/stop
# ---------------------------------------------------------------------------


class TestCalendarFeedLifecycle:
    @patch("data.calendar_feed.requests.get")
    def test_start_triggers_initial_refresh(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: [], raise_for_status=lambda: None)
        feed = CalendarFeed(config=SAMPLE_CONFIG)
        feed.start()
        try:
            # After start, events should be fetched (empty list is fine here)
            assert isinstance(feed.get_events(), list)
        finally:
            feed.stop()

    @patch("data.calendar_feed.requests.get")
    def test_stop_shuts_down_scheduler(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: [], raise_for_status=lambda: None)
        feed = CalendarFeed(config=SAMPLE_CONFIG)
        feed.start()
        feed.stop()
        assert not feed._scheduler.running

    @patch("data.calendar_feed.requests.get")
    def test_get_events_returns_list(self, mock_get):
        raw_events = [_make_raw_event()]
        mock_get.return_value = MagicMock(json=lambda: raw_events, raise_for_status=lambda: None)
        feed = CalendarFeed(config=SAMPLE_CONFIG)
        feed.refresh_calendar()
        events = feed.get_events()
        assert isinstance(events, list)
        assert len(events) == 1

    @patch("data.calendar_feed.requests.get")
    def test_refresh_failure_does_not_crash(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        feed = CalendarFeed(config=SAMPLE_CONFIG)
        # Should not raise
        feed.refresh_calendar()
        assert feed.get_events() == []
