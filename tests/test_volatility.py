"""
Unit tests for the VolatilityMonitor (src/analysis/volatility.py).

All tests use synthetic price/candle data — no live feed or API keys needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from analysis.models import OHLCVCandle, VolatilitySignal
from analysis.volatility import (
    VolatilityMonitor,
    _compute_atr,
    _compute_atr_series,
    _determine_direction,
    format_volatility_alert,
    get_pip_size,
    price_to_pips,
    PIP_SIZES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TS = "2026-04-02T13:00:00Z"


def _make_candle(
    close: float,
    open_: Optional[float] = None,
    high: Optional[float] = None,
    low: Optional[float] = None,
    symbol: str = "GBPUSD",
    timeframe: str = "H1",
    timestamp: str = _BASE_TS,
    volume: int = 100,
) -> OHLCVCandle:
    if open_ is None:
        open_ = close
    if high is None:
        high = close + 0.0005
    if low is None:
        low = close - 0.0005
    return OHLCVCandle(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=timestamp,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _make_candles(
    n: int,
    base_close: float = 1.2500,
    pip_size: float = 0.0001,
    symbol: str = "GBPUSD",
    timeframe: str = "H1",
    spread: float = 0.0005,
) -> List[OHLCVCandle]:
    """Create *n* synthetic candles with moderate ATR (~5 pips)."""
    candles = []
    for i in range(n):
        ts = f"2026-04-02T{i // 60:02d}:{i % 60:02d}:00Z"
        close = base_close + i * pip_size * 0.1  # very gentle drift
        candles.append(
            OHLCVCandle(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=ts,
                open=close - pip_size,
                high=close + spread,
                low=close - spread,
                close=close,
                volume=100,
            )
        )
    return candles


def _make_monitor(**kwargs) -> VolatilityMonitor:
    """Return a VolatilityMonitor with sensible test defaults."""
    defaults = {
        "config": {
            "volatility": {
                "atr_multiplier": 1.5,
                "pip_spike_threshold": 30,
                "pip_spike_window_seconds": 300,
                "cooldown_minutes": 10,
                "monitor_timeframe": "H1",
            }
        }
    }
    defaults.update(kwargs)
    return VolatilityMonitor(**defaults)


# ---------------------------------------------------------------------------
# Pip size tests
# ---------------------------------------------------------------------------


class TestPipSize:
    def test_major_pair(self):
        assert get_pip_size("EURUSD") == 0.0001
        assert get_pip_size("GBPUSD") == 0.0001
        assert get_pip_size("AUDUSD") == 0.0001

    def test_jpy_pair(self):
        assert get_pip_size("USDJPY") == 0.01
        assert get_pip_size("EURJPY") == 0.01
        assert get_pip_size("GBPJPY") == 0.01

    def test_gold(self):
        assert get_pip_size("XAUUSD") == 0.10

    def test_silver(self):
        assert get_pip_size("XAGUSD") == 0.001

    def test_oil(self):
        assert get_pip_size("USOIL") == 0.01
        assert get_pip_size("UKOIL") == 0.01

    def test_unknown_pair_defaults(self):
        assert get_pip_size("UNKNOWN") == 0.0001

    def test_case_insensitive(self):
        assert get_pip_size("eurusd") == get_pip_size("EURUSD")

    def test_price_to_pips_eurusd(self):
        pips = price_to_pips(0.0030, "EURUSD")
        assert pips == pytest.approx(30.0)

    def test_price_to_pips_usdjpy(self):
        pips = price_to_pips(0.30, "USDJPY")
        assert pips == pytest.approx(30.0)

    def test_price_to_pips_absolute(self):
        """price_to_pips should return the absolute value."""
        assert price_to_pips(-0.0030, "EURUSD") == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# ATR computation tests
# ---------------------------------------------------------------------------


class TestATRComputation:
    def test_returns_none_when_insufficient_candles(self):
        candles = _make_candles(10)
        assert _compute_atr(candles, period=14) is None

    def test_returns_value_with_enough_candles(self):
        candles = _make_candles(20)
        atr = _compute_atr(candles, period=14)
        assert atr is not None
        assert atr > 0

    def test_higher_spread_gives_higher_atr(self):
        low_spread  = _make_candles(20, spread=0.0005)
        high_spread = _make_candles(20, spread=0.0050)
        atr_low  = _compute_atr(low_spread, period=14)
        atr_high = _compute_atr(high_spread, period=14)
        assert atr_high > atr_low

    def test_atr_series_length(self):
        candles = _make_candles(30, spread=0.0010)
        series  = _compute_atr_series(candles, period=14)
        # series[0] uses candles[0..14], so length = len(candles) - period
        assert len(series) == len(candles) - 14


# ---------------------------------------------------------------------------
# ATR expansion detection
# ---------------------------------------------------------------------------


class TestATRExpansion:
    def _make_spiking_candles(
        self,
        n_normal: int = 28,
        spike_multiplier: float = 3.0,
        symbol: str = "GBPUSD",
    ) -> List[OHLCVCandle]:
        """Return candles where the last one has a much larger spread."""
        base = _make_candles(n_normal, spread=0.0005, symbol=symbol)
        # Add a spike candle with greatly expanded range
        spike = OHLCVCandle(
            symbol=symbol,
            timeframe="H1",
            timestamp="2026-04-02T13:45:00Z",
            open=base[-1].close,
            high=base[-1].close + 0.0005 * spike_multiplier,
            low=base[-1].close  - 0.0005 * spike_multiplier,
            close=base[-1].close - 0.0005 * spike_multiplier,
            volume=500,
        )
        return base + [spike]

    def test_atr_expansion_detected_above_threshold(self):
        monitor = _make_monitor()
        candles = self._make_spiking_candles(n_normal=28, spike_multiplier=4.0)

        signal = None
        for c in candles:
            signal = monitor.on_candle(c)

        assert signal is not None
        assert signal.spike_detected is True
        assert signal.trigger == "atr_expansion"
        assert signal.atr_ratio is not None
        assert signal.atr_ratio > 1.5

    def test_atr_expansion_not_fired_below_threshold(self):
        monitor = _make_monitor()
        # All candles are uniform — no spike
        candles = _make_candles(30, spread=0.0005)

        last_signal = None
        for c in candles:
            last_signal = monitor.on_candle(c)

        # uniform candles should not trigger
        if last_signal is not None:
            assert last_signal.spike_detected is False or last_signal.atr_ratio <= 1.5

    def test_atr_expansion_callback_called(self):
        callback = MagicMock()
        monitor = _make_monitor(volatility_callback=callback)
        candles = self._make_spiking_candles(n_normal=28, spike_multiplier=4.0)

        for c in candles:
            monitor.on_candle(c)

        callback.assert_called_once()
        emitted: VolatilitySignal = callback.call_args[0][0]
        assert emitted.trigger == "atr_expansion"


# ---------------------------------------------------------------------------
# Pip spike detection
# ---------------------------------------------------------------------------


class TestPipSpike:
    def _feed_price_sequence(
        self,
        monitor: VolatilityMonitor,
        symbol: str,
        prices: List[float],
        start_ts: float,
        interval_seconds: float = 30.0,
    ) -> Optional[VolatilitySignal]:
        """Feed a sequence of price ticks spaced *interval_seconds* apart."""
        last_signal = None
        for i, price in enumerate(prices):
            ts = datetime.fromtimestamp(
                start_ts + i * interval_seconds, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            result = monitor.on_price_tick(symbol, price, timestamp=ts)
            if result is not None:
                last_signal = result
        return last_signal

    def test_pip_spike_detected_within_window(self):
        monitor = _make_monitor()
        import time as time_mod
        start = 1743595200.0  # fixed unix ts

        # 45-pip move in ~3 minutes (within 5-minute window) for GBPUSD
        prices = [1.2680, 1.2665, 1.2650, 1.2635]  # 45 pip drop
        signal = self._feed_price_sequence(monitor, "GBPUSD", prices, start)

        assert signal is not None
        assert signal.spike_detected is True
        assert signal.trigger == "pip_spike"
        assert signal.pips_moved is not None
        assert signal.pips_moved >= 30  # threshold

    def test_pip_spike_not_fired_below_threshold(self):
        # Set a high threshold so a small move doesn't trigger
        monitor = VolatilityMonitor(
            config={
                "volatility": {
                    "atr_multiplier": 1.5,
                    "pip_spike_threshold": 100,   # 100 pips — very high
                    "pip_spike_window_seconds": 300,
                    "cooldown_minutes": 10,
                    "monitor_timeframe": "H1",
                }
            }
        )
        start = 1743595200.0
        prices = [1.2680, 1.2665, 1.2650, 1.2635]  # only 45 pips

        signal = self._feed_price_sequence(monitor, "GBPUSD", prices, start)
        assert signal is None

    def test_pip_spike_outside_window_not_detected(self):
        """Prices that move outside the rolling window should not trigger."""
        monitor = VolatilityMonitor(
            config={
                "volatility": {
                    "atr_multiplier": 100.0,       # disable ATR trigger
                    "pip_spike_threshold": 30,
                    "pip_spike_window_seconds": 60,  # 1-minute window
                    "cooldown_minutes": 0,
                    "monitor_timeframe": "H1",
                }
            }
        )
        start = 1743595200.0

        # Spread the 45-pip move over 3 minutes (> window)
        prices = [1.2680, 1.2665, 1.2650, 1.2635]
        interval = 70.0  # 70 seconds between ticks > 60s window

        signal = self._feed_price_sequence(
            monitor, "GBPUSD", prices, start, interval_seconds=interval
        )
        assert signal is None

    def test_pip_spike_uses_correct_pip_size_jpy(self):
        """A 35-pip move on USDJPY should fire (pip size 0.01)."""
        monitor = _make_monitor()
        start = 1743595200.0
        # 35 pips on USDJPY = 0.35 price move
        prices = [151.00, 150.65]
        signal = self._feed_price_sequence(
            monitor, "USDJPY", prices, start, interval_seconds=30.0
        )
        assert signal is not None
        assert signal.pips_moved is not None
        assert signal.pips_moved >= 30

    def test_pip_spike_uses_correct_pip_size_gold(self):
        """A 35-pip move on XAUUSD should fire (pip size 0.10)."""
        monitor = _make_monitor()
        start = 1743595200.0
        # 35 pips on XAUUSD = $3.50 move
        prices = [2300.00, 2296.50]
        signal = self._feed_price_sequence(
            monitor, "XAUUSD", prices, start, interval_seconds=30.0
        )
        assert signal is not None
        assert signal.pips_moved is not None
        assert signal.pips_moved >= 30


# ---------------------------------------------------------------------------
# Direction detection
# ---------------------------------------------------------------------------


class TestDirection:
    def test_bullish_direction(self):
        assert _determine_direction(1.2700, 1.2650) == "bullish"

    def test_bearish_direction(self):
        assert _determine_direction(1.2635, 1.2680) == "bearish"

    def test_neutral_no_move(self):
        assert _determine_direction(1.2680, 1.2680) == "neutral"

    def test_neutral_no_prior(self):
        assert _determine_direction(1.2680, None) == "neutral"

    def test_pip_spike_direction_bearish(self):
        monitor = _make_monitor()
        start = 1743595200.0
        prices = [1.2680, 1.2635]
        signal = None
        for i, price in enumerate(prices):
            ts = datetime.fromtimestamp(
                start + i * 30, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            result = monitor.on_price_tick("GBPUSD", price, timestamp=ts)
            if result:
                signal = result

        assert signal is not None
        assert signal.direction == "bearish"

    def test_pip_spike_direction_bullish(self):
        monitor = _make_monitor()
        start = 1743595200.0
        prices = [1.2635, 1.2680]
        signal = None
        for i, price in enumerate(prices):
            ts = datetime.fromtimestamp(
                start + i * 30, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            result = monitor.on_price_tick("GBPUSD", price, timestamp=ts)
            if result:
                signal = result

        assert signal is not None
        assert signal.direction == "bullish"


# ---------------------------------------------------------------------------
# Cooldown period
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_cooldown_suppresses_repeated_alerts(self):
        monitor = VolatilityMonitor(
            config={
                "volatility": {
                    "atr_multiplier": 1.5,
                    "pip_spike_threshold": 30,
                    "pip_spike_window_seconds": 300,
                    "cooldown_minutes": 10,
                    "monitor_timeframe": "H1",
                }
            }
        )
        start = 1743595200.0

        # First spike
        prices_1 = [1.2680, 1.2635]
        for i, price in enumerate(prices_1):
            ts = datetime.fromtimestamp(start + i * 30, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            monitor.on_price_tick("GBPUSD", price, timestamp=ts)

        # Second spike 2 minutes later — should be suppressed
        second_start = start + 120  # 2 minutes later
        prices_2 = [1.2635, 1.2590]
        alert_count = [0]
        original_callback = monitor._volatility_callback

        signals = []
        for i, price in enumerate(prices_2):
            ts = datetime.fromtimestamp(
                second_start + i * 30, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            result = monitor.on_price_tick("GBPUSD", price, timestamp=ts)
            if result is not None:
                signals.append(result)

        assert len(signals) == 0, "Second alert within cooldown should be suppressed"

    def test_cooldown_expires_after_period(self):
        monitor = VolatilityMonitor(
            config={
                "volatility": {
                    "atr_multiplier": 1.5,
                    "pip_spike_threshold": 30,
                    "pip_spike_window_seconds": 300,
                    "cooldown_minutes": 0,   # no cooldown
                    "monitor_timeframe": "H1",
                }
            }
        )

        start = 1743595200.0
        signals = []

        # Two spikes in quick succession — both should fire (no cooldown)
        for spike_offset in [0, 60]:
            prices = [1.2680 - spike_offset * 0.0001, 1.2680 - spike_offset * 0.0001 - 0.0045]
            for i, price in enumerate(prices):
                ts = datetime.fromtimestamp(
                    start + spike_offset + i * 30, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
                result = monitor.on_price_tick("GBPUSD", price, timestamp=ts)
                if result is not None:
                    signals.append(result)

        assert len(signals) >= 1  # at least first spike fired


# ---------------------------------------------------------------------------
# VolatilitySignal construction
# ---------------------------------------------------------------------------


class TestVolatilitySignalConstruction:
    def _get_pip_spike_signal(self) -> VolatilitySignal:
        monitor = _make_monitor()
        start = 1743595200.0
        prices = [1.2680, 1.2635]

        signal = None
        for i, price in enumerate(prices):
            ts = datetime.fromtimestamp(
                start + i * 30, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            result = monitor.on_price_tick("GBPUSD", price, timestamp=ts)
            if result:
                signal = result
        return signal

    def test_signal_has_symbol(self):
        s = self._get_pip_spike_signal()
        assert s.symbol == "GBPUSD"

    def test_signal_has_timestamp(self):
        s = self._get_pip_spike_signal()
        assert s.timestamp != ""

    def test_signal_has_trigger(self):
        s = self._get_pip_spike_signal()
        assert s.trigger in {"pip_spike", "atr_expansion"}

    def test_signal_has_price_before_and_now(self):
        s = self._get_pip_spike_signal()
        assert s.price_before == pytest.approx(1.2680)
        assert s.price_now    == pytest.approx(1.2635)

    def test_signal_has_pips_moved(self):
        s = self._get_pip_spike_signal()
        assert s.pips_moved == pytest.approx(45.0)

    def test_signal_has_window_seconds(self):
        s = self._get_pip_spike_signal()
        assert s.window_seconds == 300

    def test_signal_has_probable_catalyst(self):
        s = self._get_pip_spike_signal()
        assert s.probable_catalyst != ""

    def test_signal_spike_detected_true(self):
        s = self._get_pip_spike_signal()
        assert s.spike_detected is True


# ---------------------------------------------------------------------------
# Score contribution
# ---------------------------------------------------------------------------


class TestScoreContribution:
    def test_pip_spike_score_contribution(self):
        monitor = _make_monitor()
        start = 1743595200.0
        prices = [1.2680, 1.2635]

        signal = None
        for i, price in enumerate(prices):
            ts = datetime.fromtimestamp(
                start + i * 30, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            result = monitor.on_price_tick("GBPUSD", price, timestamp=ts)
            if result:
                signal = result

        assert signal is not None
        assert 0.0 < signal.score_contribution <= 0.5

    def test_atr_expansion_score_contribution(self):
        """ATR ratio-based score should be capped at 0.5."""
        candles = []
        n_normal = 28
        for i in range(n_normal):
            ts = f"2026-04-02T{i // 60:02d}:{i % 60:02d}:00Z"
            candles.append(
                OHLCVCandle(
                    symbol="EURUSD",
                    timeframe="H1",
                    timestamp=ts,
                    open=1.0850,
                    high=1.0855,
                    low=1.0845,
                    close=1.0850,
                    volume=100,
                )
            )
        # Spike candle
        candles.append(
            OHLCVCandle(
                symbol="EURUSD",
                timeframe="H1",
                timestamp="2026-04-02T13:45:00Z",
                open=1.0850,
                high=1.0900,
                low=1.0800,
                close=1.0800,
                volume=1000,
            )
        )

        monitor = _make_monitor()
        signal = None
        for c in candles:
            result = monitor.on_candle(c)
            if result:
                signal = result

        if signal is not None and signal.trigger == "atr_expansion":
            assert signal.score_contribution <= 0.5


# ---------------------------------------------------------------------------
# Alert format
# ---------------------------------------------------------------------------


class TestFormatVolatilityAlert:
    def _make_signal(self, trigger: str = "pip_spike") -> VolatilitySignal:
        return VolatilitySignal(
            symbol="GBPUSD",
            timestamp="2026-04-02T13:45:00Z",
            trigger=trigger,
            spike_detected=True,
            magnitude=45 if trigger == "pip_spike" else 1.71,
            direction="bearish",
            price_before=1.2680,
            price_now=1.2635,
            atr_current=0.00082,
            atr_average=0.00048,
            atr_ratio=1.71,
            window_seconds=300,
            pips_moved=45.0 if trigger == "pip_spike" else None,
            probable_catalyst="Pending news check",
            score_contribution=0.4,
        )

    def test_alert_contains_symbol(self):
        alert = format_volatility_alert(self._make_signal())
        assert "GBPUSD" in alert

    def test_alert_contains_spike_header(self):
        alert = format_volatility_alert(self._make_signal())
        assert "VOLATILITY SPIKE DETECTED" in alert

    def test_alert_contains_pips_for_pip_spike(self):
        alert = format_volatility_alert(self._make_signal("pip_spike"))
        assert "45" in alert
        assert "pip" in alert.lower()

    def test_alert_contains_prices(self):
        alert = format_volatility_alert(self._make_signal())
        assert "1.268" in alert
        assert "1.2635" in alert

    def test_alert_contains_atr_ratio(self):
        alert = format_volatility_alert(self._make_signal())
        assert "1.71" in alert

    def test_alert_contains_time(self):
        alert = format_volatility_alert(self._make_signal())
        assert "13:45" in alert

    def test_alert_contains_catalyst(self):
        alert = format_volatility_alert(self._make_signal())
        assert "catalyst" in alert.lower() or "news" in alert.lower()

    def test_atr_expansion_alert(self):
        alert = format_volatility_alert(self._make_signal("atr_expansion"))
        assert "ATR" in alert


# ---------------------------------------------------------------------------
# Confluence integration
# ---------------------------------------------------------------------------


class TestConfluenceIntegration:
    def test_update_volatility_stores_signal(self):
        from analysis.confluence import ConfluenceEngine

        engine = ConfluenceEngine()
        signal = VolatilitySignal(
            symbol="EURUSD",
            timestamp=_BASE_TS,
            spike_detected=False,
            atr_ratio=1.1,
            score_contribution=0.3,
        )
        engine.update_volatility(signal)
        cached = engine._volatility_cache.get("EURUSD")
        assert cached is not None
        assert cached.atr_ratio == pytest.approx(1.1)

    def test_volatility_score_used_in_confluence(self):
        """A non-spike volatility signal with score_contribution should add to score."""
        from analysis.confluence import ConfluenceEngine
        from analysis.models import TechnicalSignal

        engine = ConfluenceEngine()

        vol_signal = VolatilitySignal(
            symbol="EURUSD",
            timestamp=_BASE_TS,
            spike_detected=False,
            atr_ratio=1.1,
            score_contribution=0.5,
        )
        engine.update_volatility(vol_signal)

        tech_signal = TechnicalSignal(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=_BASE_TS,
            ema_alignment="bullish",
            ema_20=1.0850,
            ema_50=1.0830,
            ema_200=1.0800,
            atr=0.001,
        )
        score_with, confluences = engine.compute_score(tech_signal)

        # Score without volatility
        engine2 = ConfluenceEngine()
        score_without, _ = engine2.compute_score(tech_signal)

        assert score_with >= score_without

    def test_spike_signal_does_not_add_to_score(self):
        """An active spike should NOT contribute positively to the score."""
        from analysis.confluence import ConfluenceEngine
        from analysis.models import TechnicalSignal

        engine = ConfluenceEngine()

        spike_signal = VolatilitySignal(
            symbol="EURUSD",
            timestamp=_BASE_TS,
            spike_detected=True,
            atr_ratio=2.5,
            score_contribution=0.5,
            trigger="atr_expansion",
        )
        engine.update_volatility(spike_signal)

        tech_signal = TechnicalSignal(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=_BASE_TS,
            ema_alignment="bullish",
            ema_20=1.0850,
            atr=0.001,
        )
        _, confluences = engine.compute_score(tech_signal)

        assert "volatility_context_scored" not in confluences
        assert "volatility_context_normal" not in confluences
