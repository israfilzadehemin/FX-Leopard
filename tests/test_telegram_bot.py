"""
Unit tests for TelegramNotifier (src/notifications/telegram_bot.py).

All Telegram API calls are fully mocked — no real bot token required.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from analysis.models import TradeSignal, VolatilitySignal
from data.calendar_feed import EconomicEvent
from notifications.telegram_bot import (
    TelegramNotifier,
    _signal_key,
    format_calendar_alert_md,
    format_signal_alert,
    format_startup_message,
    format_volatility_alert_md,
    format_watch_alert,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_TS = "2026-04-03T14:00:00Z"


def _plain(msg: str) -> str:
    """Strip MarkdownV2 backslash escapes for easier assertion matching."""
    return msg.replace("\\", "")


def _make_signal_alert() -> TradeSignal:
    return TradeSignal(
        symbol="EURUSD",
        timeframe="H1",
        timestamp=_TS,
        signal_type="SIGNAL",
        direction="BUY",
        score=8.2,
        entry_zone=[1.0838, 1.0845],
        stop_loss=1.0808,
        take_profit=1.0908,
        rr_ratio=2.3,
        sl_pips=30.0,
        tp_pips=70.0,
        confluences=[
            "EMA alignment bullish (M15/H1/H4)",
            "RSI recovering from oversold",
            "MACD bullish crossover",
        ],
        invalidation="H1 close below 1.0805",
        signal_age_seconds=12,
    )


def _make_watch_signal() -> TradeSignal:
    return TradeSignal(
        symbol="GBPUSD",
        timeframe="H4",
        timestamp=_TS,
        signal_type="WATCH",
        direction="BUY",
        score=6.1,
        confluences=[
            "EMA alignment bullish",
            "RSI approaching oversold",
            "Price near key support 1.2640",
        ],
    )


def _make_volatility_signal() -> VolatilitySignal:
    return VolatilitySignal(
        symbol="GBPUSD",
        timestamp=_TS,
        trigger="pip_spike",
        spike_detected=True,
        magnitude=45.0,
        direction="bearish",
        price_before=1.2680,
        price_now=1.2635,
        atr_ratio=1.71,
        window_seconds=300,
        pips_moved=45.0,
        probable_catalyst="Pending news scan",
    )


def _make_calendar_event() -> EconomicEvent:
    return EconomicEvent(
        title="Non-Farm Payrolls",
        country="USD",
        datetime="2026-04-03T12:30:00Z",
        impact="high",
        forecast="185K",
        previous="151K",
        affected_pairs=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USOIL"],
    )


def _make_notifier(**kwargs) -> TelegramNotifier:
    defaults = dict(
        bot_token="fake-token",
        chat_id="123456",
        dedup_window_seconds=60,
        rate_limit_delay_seconds=0.0,
        max_retries=3,
    )
    defaults.update(kwargs)
    return TelegramNotifier(**defaults)


# ---------------------------------------------------------------------------
# Formatter tests
# ---------------------------------------------------------------------------


class TestFormatSignalAlert:
    def test_contains_symbol(self):
        msg = format_signal_alert(_make_signal_alert())
        assert "EURUSD" in msg

    def test_contains_direction(self):
        msg = format_signal_alert(_make_signal_alert())
        assert "BUY" in msg

    def test_contains_score(self):
        msg = format_signal_alert(_make_signal_alert())
        assert "8.2" in _plain(msg)

    def test_contains_entry_zone(self):
        msg = _plain(format_signal_alert(_make_signal_alert()))
        assert "1.0838" in msg
        assert "1.0845" in msg

    def test_contains_stop_loss(self):
        msg = _plain(format_signal_alert(_make_signal_alert()))
        assert "1.0808" in msg

    def test_contains_take_profit(self):
        msg = _plain(format_signal_alert(_make_signal_alert()))
        assert "1.0908" in msg

    def test_contains_rr_ratio(self):
        msg = _plain(format_signal_alert(_make_signal_alert()))
        assert "2.3" in msg

    def test_contains_confluences(self):
        msg = _plain(format_signal_alert(_make_signal_alert()))
        assert "EMA alignment bullish" in msg
        assert "MACD bullish crossover" in msg

    def test_contains_invalidation(self):
        msg = _plain(format_signal_alert(_make_signal_alert()))
        assert "1.0805" in msg

    def test_contains_signal_age(self):
        msg = format_signal_alert(_make_signal_alert())
        assert "12" in msg

    def test_contains_header(self):
        msg = _plain(format_signal_alert(_make_signal_alert()))
        assert "FX-LEOPARD" in msg or "LEOPARD" in msg


class TestFormatWatchAlert:
    def test_contains_symbol(self):
        msg = format_watch_alert(_make_watch_signal())
        assert "GBPUSD" in msg

    def test_contains_timeframe(self):
        msg = format_watch_alert(_make_watch_signal())
        assert "H4" in msg

    def test_contains_score(self):
        msg = format_watch_alert(_make_watch_signal())
        assert "6.1" in msg

    def test_contains_confluences(self):
        msg = format_watch_alert(_make_watch_signal())
        assert "EMA alignment bullish" in msg

    def test_contains_watch_header(self):
        msg = format_watch_alert(_make_watch_signal())
        assert "WATCH" in msg

    def test_contains_confirmation_line(self):
        msg = format_watch_alert(_make_watch_signal())
        assert "confirm" in msg.lower()


class TestFormatVolatilityAlert:
    def test_contains_symbol(self):
        msg = format_volatility_alert_md(_make_volatility_signal())
        assert "GBPUSD" in msg

    def test_contains_pips(self):
        msg = format_volatility_alert_md(_make_volatility_signal())
        assert "45" in msg

    def test_contains_prices(self):
        msg = _plain(format_volatility_alert_md(_make_volatility_signal()))
        assert "1.268" in msg
        assert "1.2635" in msg

    def test_contains_atr_ratio(self):
        msg = _plain(format_volatility_alert_md(_make_volatility_signal()))
        assert "1.71" in msg

    def test_contains_catalyst(self):
        msg = format_volatility_alert_md(_make_volatility_signal())
        assert "news" in msg.lower() or "catalyst" in msg.lower()

    def test_contains_volatility_header(self):
        msg = format_volatility_alert_md(_make_volatility_signal())
        assert "VOLATILITY" in msg


class TestFormatCalendarAlert:
    def test_contains_event_title(self):
        msg = format_calendar_alert_md(_make_calendar_event(), minutes_until=15)
        assert "Non" in msg  # "Non-Farm Payrolls" — hyphen may be escaped

    def test_contains_country(self):
        msg = format_calendar_alert_md(_make_calendar_event(), minutes_until=15)
        assert "USD" in msg

    def test_contains_minutes_until(self):
        msg = format_calendar_alert_md(_make_calendar_event(), minutes_until=15)
        assert "15" in msg

    def test_contains_forecast(self):
        msg = format_calendar_alert_md(_make_calendar_event(), minutes_until=15)
        assert "185K" in msg

    def test_contains_previous(self):
        msg = format_calendar_alert_md(_make_calendar_event(), minutes_until=15)
        assert "151K" in msg

    def test_contains_affected_pairs(self):
        msg = format_calendar_alert_md(_make_calendar_event(), minutes_until=15)
        assert "EURUSD" in msg

    def test_contains_calendar_header(self):
        msg = format_calendar_alert_md(_make_calendar_event(), minutes_until=15)
        assert "EVENT" in msg or "CALENDAR" in msg or "UPCOMING" in msg

    def test_without_minutes(self):
        msg = format_calendar_alert_md(_make_calendar_event())
        assert "Non" in msg


class TestFormatStartupMessage:
    def test_basic_format(self):
        msg = format_startup_message(
            pairs=["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"],
            timeframes=["M5", "M15", "H1", "H4", "D1"],
            threshold=7.0,
        )
        assert "EURUSD" in msg
        assert "7.0" in msg

    def test_more_indicator(self):
        pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]
        msg = format_startup_message(pairs=pairs, timeframes=["H1"], threshold=7.0)
        assert "+2 more" in msg

    def test_three_or_fewer_pairs_no_more(self):
        msg = format_startup_message(
            pairs=["EURUSD", "GBPUSD"],
            timeframes=["H1"],
            threshold=7.0,
        )
        assert "more" not in msg


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_first_send_not_duplicate(self):
        notifier = _make_notifier()
        sig = _make_signal_alert()
        assert not notifier._is_duplicate(sig)

    def test_second_send_within_window_is_duplicate(self):
        notifier = _make_notifier(dedup_window_seconds=60)
        sig = _make_signal_alert()
        notifier._mark_sent(sig)
        assert notifier._is_duplicate(sig)

    def test_different_signal_not_duplicate(self):
        notifier = _make_notifier()
        sig1 = _make_signal_alert()
        sig2 = _make_watch_signal()
        notifier._mark_sent(sig1)
        assert not notifier._is_duplicate(sig2)

    def test_after_window_expires_not_duplicate(self):
        notifier = _make_notifier(dedup_window_seconds=1)
        sig = _make_signal_alert()
        notifier._mark_sent(sig)
        # Manually backdate the cache entry
        key = _signal_key(sig)
        notifier._dedup_cache[key] = time.monotonic() - 2.0
        assert not notifier._is_duplicate(sig)

    def test_volatility_dedup_key(self):
        notifier = _make_notifier()
        vs = _make_volatility_signal()
        notifier._mark_sent(vs)
        assert notifier._is_duplicate(vs)

    def test_calendar_dedup_key(self):
        notifier = _make_notifier()
        ev = _make_calendar_event()
        notifier._mark_sent(ev)
        assert notifier._is_duplicate(ev)


# ---------------------------------------------------------------------------
# Message queue / ordering tests
# ---------------------------------------------------------------------------


class TestMessageQueue:
    @pytest.mark.asyncio
    async def test_queue_processes_in_order(self):
        notifier = _make_notifier(rate_limit_delay_seconds=0.0)
        sent_texts = []

        async def fake_send_with_retry(text, parse_mode):
            sent_texts.append(text)

        notifier._send_with_retry = fake_send_with_retry

        mock_bot = AsyncMock()
        notifier._bot = mock_bot

        await notifier.start()
        await notifier.send_raw("first")
        await notifier.send_raw("second")
        await notifier.send_raw("third")

        # Give the worker time to process
        await asyncio.sleep(0.2)
        await notifier.stop()

        assert sent_texts[:3] == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_queue_drains_on_stop(self):
        notifier = _make_notifier(rate_limit_delay_seconds=0.0)
        sent_count = []

        async def fake_send_with_retry(text, parse_mode):
            sent_count.append(1)

        notifier._send_with_retry = fake_send_with_retry
        notifier._bot = AsyncMock()

        await notifier.start()
        for _ in range(5):
            await notifier.send_raw("msg")

        await notifier.stop()
        assert len(sent_count) == 5


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(self):
        from telegram.error import TelegramError

        notifier = _make_notifier(max_retries=3, rate_limit_delay_seconds=0.0)
        call_count = 0

        async def flaky_send(*_, **__):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TelegramError("rate limited")

        mock_bot = AsyncMock()
        mock_bot.send_message = flaky_send
        notifier._bot = mock_bot

        # Patch asyncio.sleep to speed up test
        with patch("notifications.telegram_bot.asyncio.sleep", new=AsyncMock()):
            await notifier._send_with_retry("hello", "MarkdownV2")

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        from telegram.error import TelegramError

        notifier = _make_notifier(max_retries=3, rate_limit_delay_seconds=0.0)

        async def always_fail(*_, **__):
            raise TelegramError("always fails")

        mock_bot = AsyncMock()
        mock_bot.send_message = always_fail
        notifier._bot = mock_bot

        with patch("notifications.telegram_bot.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(TelegramError):
                await notifier._send_with_retry("hello", "MarkdownV2")

    @pytest.mark.asyncio
    async def test_retry_count_matches_max_retries(self):
        from telegram.error import TelegramError

        notifier = _make_notifier(max_retries=2, rate_limit_delay_seconds=0.0)
        call_count = 0

        async def count_calls(*_, **__):
            nonlocal call_count
            call_count += 1
            raise TelegramError("fail")

        mock_bot = AsyncMock()
        mock_bot.send_message = count_calls
        notifier._bot = mock_bot

        with patch("notifications.telegram_bot.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(TelegramError):
                await notifier._send_with_retry("hello", "MarkdownV2")

        assert call_count == 2


# ---------------------------------------------------------------------------
# send_signal / send_volatility / send_calendar deduplication integration
# ---------------------------------------------------------------------------


class TestSendMethodDeduplication:
    @pytest.mark.asyncio
    async def test_send_signal_suppresses_duplicate(self):
        notifier = _make_notifier()
        notifier._bot = AsyncMock()
        await notifier.start()

        sig = _make_signal_alert()

        with patch.object(notifier, "_queue") as mock_queue:
            mock_queue.put = AsyncMock()
            # First send — should enqueue
            await notifier.send_signal(sig)
            assert mock_queue.put.call_count == 1

            # Second send within window — should be suppressed
            await notifier.send_signal(sig)
            assert mock_queue.put.call_count == 1  # still 1

        await notifier.stop()

    @pytest.mark.asyncio
    async def test_send_volatility_suppresses_duplicate(self):
        notifier = _make_notifier()
        notifier._bot = AsyncMock()
        await notifier.start()

        vs = _make_volatility_signal()

        with patch.object(notifier, "_queue") as mock_queue:
            mock_queue.put = AsyncMock()
            await notifier.send_volatility(vs)
            assert mock_queue.put.call_count == 1

            await notifier.send_volatility(vs)
            assert mock_queue.put.call_count == 1

        await notifier.stop()

    @pytest.mark.asyncio
    async def test_send_calendar_suppresses_duplicate(self):
        notifier = _make_notifier()
        notifier._bot = AsyncMock()
        await notifier.start()

        ev = _make_calendar_event()

        with patch.object(notifier, "_queue") as mock_queue:
            mock_queue.put = AsyncMock()
            await notifier.send_calendar(ev, minutes_until=15)
            assert mock_queue.put.call_count == 1

            await notifier.send_calendar(ev, minutes_until=15)
            assert mock_queue.put.call_count == 1

        await notifier.stop()
