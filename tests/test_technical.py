"""
Unit tests for the TechnicalEngine and related analysis models.

All tests use synthetic OHLCV data — no live feed or API keys required.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from analysis.models import (
    OHLCVCandle,
    SentimentSignal,
    TechnicalSignal,
    TradeSignal,
    VolatilitySignal,
)
from analysis.technical import (
    TechnicalEngine,
    _detect_candle_patterns,
    _detect_ema_alignment,
    _detect_macd_crossover,
    _detect_rsi_divergence,
    _find_swing_levels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candle(
    close: float,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    timestamp: str = "2026-04-02T00:00:00Z",
    volume: int = 100,
) -> OHLCVCandle:
    if open_ is None:
        open_ = close
    if high is None:
        high = max(open_, close) + 0.0002
    if low is None:
        low = min(open_, close) - 0.0002
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


def _build_rising_candles(n: int, start: float = 1.08, step: float = 0.001) -> List[dict]:
    """Generate n steadily rising candles as dicts (PriceFeed format)."""
    candles = []
    for i in range(n):
        c = start + i * step
        candles.append(
            {
                "symbol": "EURUSD",
                "timeframe": "H1",
                "timestamp": f"2026-04-{(i // 24) + 1:02d}T{i % 24:02d}:00:00Z",
                "open": c - step / 2,
                "high": c + step / 2,
                "low": c - step,
                "close": c,
                "volume": 100,
            }
        )
    return candles


def _build_falling_candles(n: int, start: float = 1.10, step: float = 0.001) -> List[dict]:
    """Generate n steadily falling candles as dicts."""
    candles = []
    for i in range(n):
        c = start - i * step
        candles.append(
            {
                "symbol": "EURUSD",
                "timeframe": "H1",
                "timestamp": f"2026-04-{(i // 24) + 1:02d}T{i % 24:02d}:00:00Z",
                "open": c + step / 2,
                "high": c + step,
                "low": c - step / 2,
                "close": c,
                "volume": 100,
            }
        )
    return candles


def _feed_candles(engine: TechnicalEngine, candles: List[dict]) -> List[TechnicalSignal]:
    signals = []
    for c in candles:
        s = engine.on_candle(c)
        if s is not None:
            signals.append(s)
    return signals


# ---------------------------------------------------------------------------
# Models tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_ohlcv_candle_fields(self):
        c = _make_candle(1.08)
        assert c.symbol == "EURUSD"
        assert c.close == pytest.approx(1.08)

    def test_technical_signal_defaults(self):
        s = TechnicalSignal(symbol="EURUSD", timeframe="H1", timestamp="2026-04-02T00:00:00Z")
        assert s.ema_alignment == "mixed"
        assert s.trending is False
        assert s.rsi_oversold is False
        assert s.events == []

    def test_technical_signal_to_dict_structure(self):
        s = TechnicalSignal(symbol="EURUSD", timeframe="H1", timestamp="2026-04-02T00:00:00Z")
        d = s.to_dict()
        assert "symbol" in d
        assert "indicators" in d
        assert "events" in d
        expected_keys = {
            "ema_20", "ema_50", "ema_200", "ema_alignment", "adx", "trending",
            "rsi", "rsi_oversold", "rsi_overbought", "rsi_divergence",
            "macd_line", "macd_signal", "macd_histogram", "macd_crossover",
            "bb_upper", "bb_middle", "bb_lower", "bb_squeeze",
            "atr", "atr_ratio", "support_levels", "resistance_levels", "candle_patterns",
        }
        assert expected_keys == set(d["indicators"].keys())

    def test_sentiment_signal_stub(self):
        s = SentimentSignal(symbol="EURUSD", timestamp="2026-04-02T00:00:00Z")
        assert s.sentiment == "neutral"

    def test_volatility_signal_stub(self):
        v = VolatilitySignal(symbol="EURUSD", timestamp="2026-04-02T00:00:00Z")
        assert v.spike_detected is False

    def test_trade_signal_stub(self):
        t = TradeSignal(symbol="EURUSD", timeframe="H1", timestamp="2026-04-02T00:00:00Z")
        assert t.signal_type == "IGNORE"


# ---------------------------------------------------------------------------
# TechnicalEngine — basic operation
# ---------------------------------------------------------------------------


class TestTechnicalEngineBasics:
    def test_returns_none_on_first_candle(self):
        engine = TechnicalEngine()
        result = engine.on_candle(_make_candle(1.08).__dict__ | {
            "symbol": "EURUSD", "timeframe": "H1", "timestamp": "2026-04-02T00:00:00Z",
            "open": 1.08, "high": 1.082, "low": 1.079, "close": 1.08, "volume": 100,
        })
        assert result is None

    def test_accepts_dict_candle(self):
        engine = TechnicalEngine()
        candles = _build_rising_candles(10)
        signals = _feed_candles(engine, candles)
        assert len(signals) >= 1

    def test_accepts_ohlcv_candle_object(self):
        engine = TechnicalEngine()
        for i in range(5):
            c = _make_candle(1.08 + i * 0.001, timestamp=f"2026-04-02T0{i}:00:00Z")
            engine.on_candle(c)

    def test_callback_invoked(self):
        received = []
        engine = TechnicalEngine(on_signal=lambda s: received.append(s))
        signals = _feed_candles(engine, _build_rising_candles(30))
        assert len(received) == len(signals)

    def test_callback_receives_technical_signal(self):
        received = []
        engine = TechnicalEngine(on_signal=lambda s: received.append(s))
        _feed_candles(engine, _build_rising_candles(30))
        assert all(isinstance(s, TechnicalSignal) for s in received)

    def test_signal_has_correct_symbol_and_timeframe(self):
        engine = TechnicalEngine()
        signals = _feed_candles(engine, _build_rising_candles(30))
        for s in signals:
            assert s.symbol == "EURUSD"
            assert s.timeframe == "H1"

    def test_get_history_returns_candles(self):
        engine = TechnicalEngine()
        _feed_candles(engine, _build_rising_candles(10))
        hist = engine.get_history("EURUSD", "H1")
        assert len(hist) == 10
        assert all(isinstance(c, OHLCVCandle) for c in hist)

    def test_get_history_missing_symbol_returns_empty(self):
        engine = TechnicalEngine()
        assert engine.get_history("UNKNOWN", "H1") == []

    def test_history_capped_at_max(self):
        engine = TechnicalEngine(max_history=50)
        _feed_candles(engine, _build_rising_candles(100))
        assert len(engine.get_history("EURUSD", "H1")) == 50

    def test_multiple_symbols_tracked_independently(self):
        engine = TechnicalEngine()
        for i in range(10):
            for sym in ["EURUSD", "GBPUSD"]:
                c = {
                    "symbol": sym, "timeframe": "H1",
                    "timestamp": f"2026-04-02T{i:02d}:00:00Z",
                    "open": 1.08, "high": 1.082, "low": 1.078, "close": 1.08 + i * 0.001,
                    "volume": 100,
                }
                engine.on_candle(c)
        assert len(engine.get_history("EURUSD", "H1")) == 10
        assert len(engine.get_history("GBPUSD", "H1")) == 10


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------


class TestIndicatorComputation:
    def _get_last_signal(self, candles: List[dict]) -> TechnicalSignal:
        engine = TechnicalEngine()
        signals = _feed_candles(engine, candles)
        assert signals, "No signals produced — need more candles"
        return signals[-1]

    def test_ema_values_computed(self):
        sig = self._get_last_signal(_build_rising_candles(60))
        assert sig.ema_20 is not None
        assert sig.ema_50 is not None

    def test_ema_200_requires_sufficient_history(self):
        # With only 60 candles, EMA-200 should be None
        sig = self._get_last_signal(_build_rising_candles(60))
        assert sig.ema_200 is None

    def test_ema_200_present_with_enough_history(self):
        sig = self._get_last_signal(_build_rising_candles(210))
        assert sig.ema_200 is not None

    def test_rsi_in_valid_range(self):
        sig = self._get_last_signal(_build_rising_candles(30))
        assert sig.rsi is not None
        assert 0.0 <= sig.rsi <= 100.0

    def test_adx_non_negative(self):
        sig = self._get_last_signal(_build_rising_candles(30))
        if sig.adx is not None:
            assert sig.adx >= 0.0

    def test_macd_values_computed(self):
        sig = self._get_last_signal(_build_rising_candles(40))
        assert sig.macd_line is not None
        assert sig.macd_signal is not None
        assert sig.macd_histogram is not None

    def test_bb_upper_greater_than_lower(self):
        sig = self._get_last_signal(_build_rising_candles(30))
        if sig.bb_upper is not None and sig.bb_lower is not None:
            assert sig.bb_upper > sig.bb_lower

    def test_atr_positive(self):
        sig = self._get_last_signal(_build_rising_candles(30))
        if sig.atr is not None:
            assert sig.atr > 0.0

    def test_atr_ratio_positive(self):
        sig = self._get_last_signal(_build_rising_candles(40))
        if sig.atr_ratio is not None:
            assert sig.atr_ratio > 0.0


# ---------------------------------------------------------------------------
# EMA alignment detection
# ---------------------------------------------------------------------------


class TestEMAAlignment:
    def test_bullish_alignment(self):
        assert _detect_ema_alignment(1.10, 1.08, 1.06) == "bullish"

    def test_bearish_alignment(self):
        assert _detect_ema_alignment(1.06, 1.08, 1.10) == "bearish"

    def test_mixed_alignment(self):
        assert _detect_ema_alignment(1.09, 1.06, 1.08) == "mixed"

    def test_none_value_returns_mixed(self):
        assert _detect_ema_alignment(None, 1.08, 1.06) == "mixed"

    def test_engine_reports_bullish_on_rising_market(self):
        # 210 strongly rising candles → EMA20 > EMA50 > EMA200
        engine = TechnicalEngine()
        signals = _feed_candles(engine, _build_rising_candles(210, step=0.002))
        last = signals[-1]
        assert last.ema_alignment == "bullish"

    def test_engine_reports_bearish_on_falling_market(self):
        # 210 strongly falling candles → EMA20 < EMA50 < EMA200
        engine = TechnicalEngine()
        signals = _feed_candles(engine, _build_falling_candles(210, step=0.002))
        last = signals[-1]
        assert last.ema_alignment == "bearish"


# ---------------------------------------------------------------------------
# RSI flags
# ---------------------------------------------------------------------------


class TestRSIFlags:
    def _engine_with_signals(self, candles):
        engine = TechnicalEngine()
        return _feed_candles(engine, candles)

    def test_rsi_overbought_flag_on_strong_rally(self):
        # 50 rapidly rising candles — RSI should eventually go overbought
        signals = self._engine_with_signals(_build_rising_candles(50, step=0.005))
        rsi_values = [s.rsi for s in signals if s.rsi is not None]
        assert any(r > 70 for r in rsi_values), "Expected at least one overbought RSI"

    def test_rsi_oversold_flag_on_strong_sell(self):
        # 50 rapidly falling candles — RSI should eventually go oversold
        signals = self._engine_with_signals(_build_falling_candles(50, step=0.005))
        rsi_values = [s.rsi for s in signals if s.rsi is not None]
        assert any(r < 30 for r in rsi_values), "Expected at least one oversold RSI"

    def test_rsi_overbought_attribute_matches_value(self):
        signals = self._engine_with_signals(_build_rising_candles(50, step=0.005))
        for s in signals:
            if s.rsi is not None:
                if s.rsi > 70:
                    assert s.rsi_overbought is True
                else:
                    assert s.rsi_overbought is False

    def test_rsi_oversold_attribute_matches_value(self):
        signals = self._engine_with_signals(_build_falling_candles(50, step=0.005))
        for s in signals:
            if s.rsi is not None:
                if s.rsi < 30:
                    assert s.rsi_oversold is True
                else:
                    assert s.rsi_oversold is False


# ---------------------------------------------------------------------------
# RSI divergence detection
# ---------------------------------------------------------------------------


class TestRSIDivergence:
    def test_bullish_divergence_detected(self):
        # Price makes lower low, RSI makes higher low → bullish divergence
        closes = [1.10, 1.09, 1.08, 1.07, 1.06, 1.05, 1.04, 1.03, 1.02, 1.01,
                  1.00, 0.99, 0.98, 0.97, 0.96]
        # RSI rises despite lower close
        rsi_vals = [50.0, 48.0, 46.0, 44.0, 42.0, 40.0, 42.0, 44.0, 45.0, 46.0,
                    47.0, 48.0, 49.0, 50.0, 52.0]
        result = _detect_rsi_divergence(closes, rsi_vals, lookback=14)
        assert result == "bullish"

    def test_bearish_divergence_detected(self):
        # Price makes higher high, RSI makes lower high → bearish divergence
        closes = [1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09,
                  1.10, 1.11, 1.12, 1.13, 1.14]
        rsi_vals = [60.0, 62.0, 64.0, 65.0, 64.0, 62.0, 61.0, 60.0, 59.0, 58.0,
                    57.0, 56.0, 55.0, 54.0, 53.0]
        result = _detect_rsi_divergence(closes, rsi_vals, lookback=14)
        assert result == "bearish"

    def test_no_divergence_returns_none(self):
        closes = list(range(15))
        rsi_vals = [float(x * 3) for x in range(15)]
        result = _detect_rsi_divergence(closes, rsi_vals, lookback=14)
        assert result is None

    def test_insufficient_data_returns_none(self):
        assert _detect_rsi_divergence([1.0, 1.1], [50.0, 55.0], lookback=14) is None


# ---------------------------------------------------------------------------
# MACD crossover detection
# ---------------------------------------------------------------------------


class TestMACDCrossover:
    def test_bullish_crossover_detected(self):
        assert _detect_macd_crossover(-0.0001, 0.0001) == "bullish"

    def test_bearish_crossover_detected(self):
        assert _detect_macd_crossover(0.0001, -0.0001) == "bearish"

    def test_no_crossover_same_sign(self):
        assert _detect_macd_crossover(0.0001, 0.0002) is None
        assert _detect_macd_crossover(-0.0001, -0.0002) is None

    def test_none_input_returns_none(self):
        assert _detect_macd_crossover(None, 0.001) is None
        assert _detect_macd_crossover(0.001, None) is None

    def test_engine_reports_macd_crossover_event(self):
        # Build candles that will generate a MACD histogram sign change
        engine = TechnicalEngine()
        # Rising then falling: should cause histogram to flip
        rising = _build_rising_candles(40, step=0.002)
        falling = _build_falling_candles(40, start=rising[-1]["close"], step=0.002)
        all_candles = rising + falling
        # Re-stamp timestamps sequentially
        for i, c in enumerate(all_candles):
            c["timestamp"] = f"2026-04-{(i // 24) + 1:02d}T{i % 24:02d}:00:00Z"
        signals = _feed_candles(engine, all_candles)
        crossovers = [s for s in signals if s.macd_crossover is not None]
        assert len(crossovers) >= 1


# ---------------------------------------------------------------------------
# Bollinger Band squeeze
# ---------------------------------------------------------------------------


class TestBBSqueeze:
    def test_squeeze_detected_on_low_volatility(self):
        # Flat price → very tight bands → squeeze
        flat_candles = []
        for i in range(30):
            flat_candles.append({
                "symbol": "EURUSD", "timeframe": "H1",
                "timestamp": f"2026-04-02T{i % 24:02d}:00:00Z",
                "open": 1.0800, "high": 1.0801, "low": 1.0799, "close": 1.0800,
                "volume": 100,
            })
        engine = TechnicalEngine()
        signals = _feed_candles(engine, flat_candles)
        squeeze_signals = [s for s in signals if s.bb_squeeze]
        assert len(squeeze_signals) > 0

    def test_no_squeeze_on_high_volatility(self):
        # Very volatile price → wide bands → no squeeze
        engine = TechnicalEngine()
        np.random.seed(99)
        volatile = []
        price = 1.08
        for i in range(30):
            price += np.random.choice([-1, 1]) * 0.01
            volatile.append({
                "symbol": "EURUSD", "timeframe": "H1",
                "timestamp": f"2026-04-02T{i % 24:02d}:00:00Z",
                "open": price - 0.005, "high": price + 0.008,
                "low": price - 0.008, "close": price,
                "volume": 100,
            })
        signals = _feed_candles(engine, volatile)
        # At least the final signal should not be squeezed
        if signals:
            assert not signals[-1].bb_squeeze


# ---------------------------------------------------------------------------
# Support & Resistance
# ---------------------------------------------------------------------------


class TestSupportResistance:
    def test_swing_highs_identified(self):
        highs = [1.08, 1.10, 1.09, 1.11, 1.09, 1.12, 1.10, 1.08]
        lows = [1.07, 1.08, 1.07, 1.09, 1.08, 1.10, 1.08, 1.07]
        resistance, _ = _find_swing_levels(highs, lows)
        assert len(resistance) > 0

    def test_swing_lows_identified(self):
        highs = [1.08, 1.10, 1.09, 1.11, 1.09, 1.12, 1.10, 1.08]
        lows = [1.07, 1.08, 1.07, 1.09, 1.08, 1.10, 1.08, 1.07]
        _, support = _find_swing_levels(highs, lows)
        assert len(support) > 0

    def test_empty_data_returns_empty_lists(self):
        r, s = _find_swing_levels([], [])
        assert r == []
        assert s == []

    def test_engine_populates_levels_after_enough_candles(self):
        engine = TechnicalEngine()
        # Oscillating candles will produce swing points
        candles = []
        for i in range(60):
            price = 1.08 + 0.002 * math.sin(i * 0.5)
            candles.append({
                "symbol": "EURUSD", "timeframe": "H1",
                "timestamp": f"2026-04-02T{i % 24:02d}:00:00Z",
                "open": price - 0.0005, "high": price + 0.001,
                "low": price - 0.001, "close": price, "volume": 100,
            })
        signals = _feed_candles(engine, candles)
        last = signals[-1]
        assert len(last.support_levels) > 0 or len(last.resistance_levels) > 0

    def test_resistance_above_support(self):
        engine = TechnicalEngine()
        candles = []
        for i in range(60):
            price = 1.08 + 0.002 * math.sin(i * 0.5)
            candles.append({
                "symbol": "EURUSD", "timeframe": "H1",
                "timestamp": f"2026-04-02T{i % 24:02d}:00:00Z",
                "open": price, "high": price + 0.001,
                "low": price - 0.001, "close": price, "volume": 100,
            })
        signals = _feed_candles(engine, candles)
        last = signals[-1]
        if last.support_levels and last.resistance_levels:
            assert max(last.support_levels) <= min(last.resistance_levels)


# ---------------------------------------------------------------------------
# Candlestick pattern detection
# ---------------------------------------------------------------------------


class TestCandlePatterns:
    def _candle(self, o, h, l, c) -> OHLCVCandle:
        return OHLCVCandle(
            symbol="EURUSD", timeframe="H1", timestamp="2026-04-02T00:00:00Z",
            open=o, high=h, low=l, close=c, volume=100,
        )

    def test_bullish_engulfing_detected(self):
        # Previous: bearish candle (open > close)
        prev = self._candle(o=1.085, h=1.086, l=1.082, c=1.083)
        # Current: bullish candle that engulfs previous (open < prev close, close > prev open)
        curr = self._candle(o=1.082, h=1.088, l=1.081, c=1.087)
        patterns = _detect_candle_patterns(curr, prev)
        assert "bullish_engulfing" in patterns

    def test_bearish_engulfing_detected(self):
        # Previous: bullish candle
        prev = self._candle(o=1.083, h=1.087, l=1.082, c=1.086)
        # Current: bearish candle that engulfs previous
        curr = self._candle(o=1.087, h=1.088, l=1.080, c=1.082)
        patterns = _detect_candle_patterns(curr, prev)
        assert "bearish_engulfing" in patterns

    def test_hammer_detected(self):
        # Small bullish body near top, long lower wick
        o, c = 1.0850, 1.0860  # small bullish body
        h = 1.0862              # tiny upper wick
        l = 1.0820              # long lower wick (4× body)
        candle = self._candle(o=o, h=h, l=l, c=c)
        patterns = _detect_candle_patterns(candle, None)
        assert "hammer" in patterns

    def test_shooting_star_detected(self):
        # Small bearish body near bottom, long upper wick
        o, c = 1.0860, 1.0850  # small bearish body
        h = 1.0895              # long upper wick
        l = 1.0848              # tiny lower wick
        candle = self._candle(o=o, h=h, l=l, c=c)
        patterns = _detect_candle_patterns(candle, None)
        assert "shooting_star" in patterns

    def test_pin_bar_detected(self):
        # Very small body, long wick on one side
        o, c = 1.0850, 1.0852  # tiny body
        h = 1.0853
        l = 1.0800              # very long lower wick (~10× body)
        candle = self._candle(o=o, h=h, l=l, c=c)
        patterns = _detect_candle_patterns(candle, None)
        assert "pin_bar" in patterns

    def test_inside_bar_detected(self):
        prev = self._candle(o=1.083, h=1.090, l=1.080, c=1.086)
        # Current completely inside previous range
        curr = self._candle(o=1.084, h=1.088, l=1.082, c=1.085)
        patterns = _detect_candle_patterns(curr, prev)
        assert "inside_bar" in patterns

    def test_no_pattern_on_normal_candle(self):
        prev = self._candle(o=1.083, h=1.085, l=1.082, c=1.084)
        curr = self._candle(o=1.084, h=1.086, l=1.083, c=1.085)
        patterns = _detect_candle_patterns(curr, prev)
        # Normal candles should not trigger bullish/bearish engulfing
        assert "bullish_engulfing" not in patterns
        assert "bearish_engulfing" not in patterns

    def test_engine_reports_patterns_in_signal(self):
        engine = TechnicalEngine()
        # Feed enough candles then a hammer
        base = _build_rising_candles(20)
        # Craft a hammer at the end
        hammer = {
            "symbol": "EURUSD", "timeframe": "H1",
            "timestamp": "2026-04-10T00:00:00Z",
            "open": 1.0850, "high": 1.0862, "low": 1.0820, "close": 1.0860,
            "volume": 100,
        }
        signals = _feed_candles(engine, base + [hammer])
        last = signals[-1]
        assert isinstance(last.candle_patterns, list)


# ---------------------------------------------------------------------------
# Events list population
# ---------------------------------------------------------------------------


class TestEventsPopulation:
    def test_overbought_event_added(self):
        engine = TechnicalEngine()
        signals = _feed_candles(engine, _build_rising_candles(50, step=0.005))
        overbought_events = [
            s for s in signals if "rsi_overbought" in s.events
        ]
        assert len(overbought_events) >= 1

    def test_oversold_event_added(self):
        engine = TechnicalEngine()
        signals = _feed_candles(engine, _build_falling_candles(50, step=0.005))
        oversold_events = [s for s in signals if "rsi_oversold" in s.events]
        assert len(oversold_events) >= 1

    def test_macd_crossover_event_added(self):
        engine = TechnicalEngine()
        rising = _build_rising_candles(40, step=0.002)
        falling = _build_falling_candles(40, start=rising[-1]["close"], step=0.002)
        all_candles = rising + falling
        for i, c in enumerate(all_candles):
            c["timestamp"] = f"2026-04-{(i // 24) + 1:02d}T{i % 24:02d}:00:00Z"
        signals = _feed_candles(engine, all_candles)
        crossover_events = [
            s for s in signals
            if any("macd_crossover" in e for e in s.events)
        ]
        assert len(crossover_events) >= 1

    def test_bb_squeeze_event_added(self):
        flat_candles = [
            {
                "symbol": "EURUSD", "timeframe": "H1",
                "timestamp": f"2026-04-02T{i % 24:02d}:00:00Z",
                "open": 1.0800, "high": 1.0801, "low": 1.0799, "close": 1.0800,
                "volume": 100,
            }
            for i in range(30)
        ]
        engine = TechnicalEngine()
        signals = _feed_candles(engine, flat_candles)
        squeeze_events = [s for s in signals if "bb_squeeze" in s.events]
        assert len(squeeze_events) > 0

    def test_candle_pattern_events_added(self):
        engine = TechnicalEngine()
        base = _build_rising_candles(20)
        hammer = {
            "symbol": "EURUSD", "timeframe": "H1",
            "timestamp": "2026-04-10T00:00:00Z",
            "open": 1.0850, "high": 1.0862, "low": 1.0820, "close": 1.0860,
            "volume": 100,
        }
        signals = _feed_candles(engine, base + [hammer])
        last = signals[-1]
        for pattern in last.candle_patterns:
            assert pattern in last.events

    def test_events_is_list(self):
        engine = TechnicalEngine()
        signals = _feed_candles(engine, _build_rising_candles(10))
        for s in signals:
            assert isinstance(s.events, list)
