"""
Unit tests for PriceFeed and config loader.

WebSocket connections are fully mocked — no real API key is required.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from config import load_config, AppConfig
from data.price_feed import CandleBuffer, PriceFeed, TIMEFRAME_SECONDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = {
    "pairs": ["EURUSD", "GBPUSD", "XAUUSD"],
    "timeframes": ["M5", "M15", "H1"],
    "confluence_threshold": 7.0,
    "watch_threshold": 5.0,
    "volatility": {
        "atr_multiplier": 1.5,
        "pip_spike_threshold": 30,
        "spike_window_minutes": 5,
    },
    "notifications": {
        "channel": "telegram",
        "bot_token": "test_bot_token",
        "chat_id": "test_chat_id",
    },
    "api_keys": {
        "twelvedata": "test_api_key",
        "openai": "test_openai_key",
        "newsapi": "test_newsapi_key",
    },
    "logging": {"signal_db": "data/signals.db", "log_level": "INFO"},
    "calendar": {"pre_event_alert_minutes": 15, "min_impact": "high"},
}


def write_temp_config(data: dict, tmp_path) -> str:
    """Write a dict as YAML to a temp file inside pytest's tmp_path and return its path."""
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# Config loader tests
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_pairs(self, tmp_path):
        path = write_temp_config(SAMPLE_CONFIG, tmp_path)
        cfg = load_config(path)
        assert cfg.pairs == ["EURUSD", "GBPUSD", "XAUUSD"]

    def test_loads_timeframes(self, tmp_path):
        path = write_temp_config(SAMPLE_CONFIG, tmp_path)
        cfg = load_config(path)
        assert cfg.timeframes == ["M5", "M15", "H1"]

    def test_loads_api_key(self, tmp_path):
        path = write_temp_config(SAMPLE_CONFIG, tmp_path)
        cfg = load_config(path)
        assert cfg.api_keys.twelvedata == "test_api_key"

    def test_loads_thresholds(self, tmp_path):
        path = write_temp_config(SAMPLE_CONFIG, tmp_path)
        cfg = load_config(path)
        assert cfg.confluence_threshold == 7.0
        assert cfg.watch_threshold == 5.0

    def test_loads_volatility_settings(self, tmp_path):
        path = write_temp_config(SAMPLE_CONFIG, tmp_path)
        cfg = load_config(path)
        assert cfg.volatility.atr_multiplier == 1.5
        assert cfg.volatility.pip_spike_threshold == 30

    def test_env_var_resolution(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TWELVEDATA_API_KEY", "resolved_key")
        data = {**SAMPLE_CONFIG, "api_keys": {"twelvedata": "${TWELVEDATA_API_KEY}"}}
        path = write_temp_config(data, tmp_path)
        cfg = load_config(path)
        assert cfg.api_keys.twelvedata == "resolved_key"

    def test_env_var_missing_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        data = {**SAMPLE_CONFIG, "api_keys": {"twelvedata": "${NONEXISTENT_VAR}"}}
        path = write_temp_config(data, tmp_path)
        cfg = load_config(path)
        assert cfg.api_keys.twelvedata == ""

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_returns_app_config_instance(self, tmp_path):
        path = write_temp_config(SAMPLE_CONFIG, tmp_path)
        cfg = load_config(path)
        assert isinstance(cfg, AppConfig)


# ---------------------------------------------------------------------------
# CandleBuffer tests
# ---------------------------------------------------------------------------


class TestCandleBuffer:
    def _buf(self, timeframe="M5"):
        return CandleBuffer("EURUSD", timeframe, TIMEFRAME_SECONDS[timeframe])

    def test_first_tick_does_not_emit(self):
        buf = self._buf()
        result = buf.update(1.0821, 0)
        assert result is None

    def test_same_candle_ticks_do_not_emit(self):
        buf = self._buf()
        buf.update(1.0821, 0)
        result = buf.update(1.0830, 100)
        assert result is None

    def test_new_period_emits_candle(self):
        buf = self._buf()
        duration = TIMEFRAME_SECONDS["M5"]
        buf.update(1.0821, 0)
        buf.update(1.0845, 60)
        candle = buf.update(1.0838, duration)  # crosses into next period
        assert candle is not None

    def test_candle_ohlcv_values(self):
        buf = self._buf()
        duration = TIMEFRAME_SECONDS["M5"]
        buf.update(1.0821, 0)
        buf.update(1.0850, 30)   # high
        buf.update(1.0810, 60)   # low
        buf.update(1.0830, 90)   # close of first candle
        candle = buf.update(1.0840, duration)

        assert candle["open"] == pytest.approx(1.0821)
        assert candle["high"] == pytest.approx(1.0850)
        assert candle["low"] == pytest.approx(1.0810)
        assert candle["close"] == pytest.approx(1.0830)
        assert candle["volume"] == 4

    def test_candle_metadata(self):
        buf = self._buf()
        duration = TIMEFRAME_SECONDS["M5"]
        buf.update(1.0821, 0)
        candle = buf.update(1.0838, duration)

        assert candle["symbol"] == "EURUSD"
        assert candle["timeframe"] == "M5"
        assert "timestamp" in candle

    def test_candle_timestamp_format(self):
        buf = self._buf()
        duration = TIMEFRAME_SECONDS["M5"]
        ts_start = 1700000000
        buf.update(1.0821, ts_start)
        candle = buf.update(1.0838, ts_start + duration)

        # Should be ISO 8601 UTC format
        datetime.strptime(candle["timestamp"], "%Y-%m-%dT%H:%M:%SZ")

    def test_multiple_candles(self):
        buf = self._buf()
        duration = TIMEFRAME_SECONDS["M5"]
        candles = []
        for i in range(4):
            result = buf.update(1.08 + i * 0.001, i * duration)
            if result is not None:
                candles.append(result)

        assert len(candles) == 3  # ticks 1,2,3 each close previous candle


# ---------------------------------------------------------------------------
# PriceFeed tests
# ---------------------------------------------------------------------------


class TestPriceFeedSymbols:
    def test_symbol_list_matches_config(self):
        symbols = ["EURUSD", "GBPUSD", "XAUUSD"]
        feed = PriceFeed(api_key="key", symbols=symbols, timeframes=["M5"])
        assert feed.symbols == symbols

    def test_unknown_timeframes_are_ignored(self):
        feed = PriceFeed(api_key="key", symbols=["EURUSD"], timeframes=["M5", "BADTF"])
        assert "BADTF" not in feed.timeframes
        assert "M5" in feed.timeframes

    def test_buffers_created_for_all_combinations(self):
        symbols = ["EURUSD", "GBPUSD"]
        timeframes = ["M5", "M15"]
        feed = PriceFeed(api_key="key", symbols=symbols, timeframes=timeframes)
        for sym in symbols:
            assert sym in feed._buffers
            for tf in timeframes:
                assert tf in feed._buffers[sym]


@pytest.mark.asyncio
class TestPriceFeedMessageHandling:
    async def test_price_event_triggers_buffer_update(self):
        received = []

        async def on_candle(candle):
            received.append(candle)

        feed = PriceFeed(
            api_key="key",
            symbols=["EURUSD"],
            timeframes=["M5"],
            on_candle=on_candle,
        )

        duration = TIMEFRAME_SECONDS["M5"]
        ts0 = 1700000000

        # First tick — no candle yet
        msg1 = json.dumps({
            "event": "price",
            "symbol": "EURUSD",
            "price": "1.0821",
            "timestamp": ts0,
        })
        await feed._handle_message(msg1)
        assert len(received) == 0

        # Second tick crosses candle boundary — emits completed candle
        msg2 = json.dumps({
            "event": "price",
            "symbol": "EURUSD",
            "price": "1.0838",
            "timestamp": ts0 + duration,
        })
        await feed._handle_message(msg2)
        assert len(received) == 1
        assert received[0]["symbol"] == "EURUSD"
        assert received[0]["timeframe"] == "M5"

    async def test_unknown_symbol_is_ignored(self):
        feed = PriceFeed(api_key="key", symbols=["EURUSD"], timeframes=["M5"])
        msg = json.dumps({
            "event": "price",
            "symbol": "UNKNOWN",
            "price": "1.0",
            "timestamp": 1700000000,
        })
        # Should not raise
        await feed._handle_message(msg)

    async def test_invalid_json_is_ignored(self):
        feed = PriceFeed(api_key="key", symbols=["EURUSD"], timeframes=["M5"])
        # Should not raise
        await feed._handle_message("not valid json{{")

    async def test_on_candle_callback_invoked(self):
        called_with = []

        async def cb(candle):
            called_with.append(candle)

        feed = PriceFeed(
            api_key="key",
            symbols=["EURUSD"],
            timeframes=["M5"],
            on_candle=cb,
        )
        duration = TIMEFRAME_SECONDS["M5"]
        ts0 = 1700000000

        await feed._handle_message(json.dumps({
            "event": "price", "symbol": "EURUSD",
            "price": "1.08", "timestamp": ts0,
        }))
        await feed._handle_message(json.dumps({
            "event": "price", "symbol": "EURUSD",
            "price": "1.09", "timestamp": ts0 + duration,
        }))

        assert len(called_with) == 1
        assert called_with[0]["open"] == pytest.approx(1.08)
        assert called_with[0]["close"] == pytest.approx(1.08)


@pytest.mark.asyncio
class TestPriceFeedReconnection:
    async def test_reconnects_on_connection_error(self):
        """PriceFeed should retry after a connection failure."""
        call_count = 0

        feed = PriceFeed(api_key="key", symbols=["EURUSD"], timeframes=["M5"])
        feed._running = True

        async def fake_connect_and_stream():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Simulated disconnect")
            # Stop after 3rd call
            feed._running = False

        with patch.object(feed, "_connect_and_stream", side_effect=fake_connect_and_stream), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await feed.run()

        assert call_count == 3

    async def test_backoff_increases_on_repeated_failures(self):
        """Sleep durations should double up to BACKOFF_MAX."""
        from data.price_feed import BACKOFF_INITIAL, BACKOFF_FACTOR, BACKOFF_MAX

        sleep_calls = []
        call_count = 0

        feed = PriceFeed(api_key="key", symbols=["EURUSD"], timeframes=["M5"])

        async def fake_connect_and_stream():
            nonlocal call_count
            call_count += 1
            if call_count < 5:
                raise ConnectionError("Simulated disconnect")
            feed._running = False

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)

        with patch.object(feed, "_connect_and_stream", side_effect=fake_connect_and_stream), \
             patch("asyncio.sleep", side_effect=fake_sleep):
            await feed.run()

        # Verify doubling backoff
        assert sleep_calls[0] == BACKOFF_INITIAL
        assert sleep_calls[1] == BACKOFF_INITIAL * BACKOFF_FACTOR
        for s in sleep_calls:
            assert s <= BACKOFF_MAX

    async def test_stop_sets_running_false(self):
        feed = PriceFeed(api_key="key", symbols=["EURUSD"], timeframes=["M5"])
        feed._running = True

        mock_ws = AsyncMock()
        feed._ws = mock_ws
        await feed.stop()

        assert feed._running is False
        mock_ws.close.assert_called_once()
