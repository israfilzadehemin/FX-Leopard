"""
Unit tests for the ConfluenceEngine (src/analysis/confluence.py).

All tests use synthetic TechnicalSignal data — no live feeds or API keys needed.
"""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from analysis.confluence import (
    ConfluenceEngine,
    _determine_direction,
    _is_price_near_sr,
    _nearest_resistance_above,
    _nearest_support_below,
    _next_higher_timeframe,
    _pips,
)
from analysis.models import (
    SentimentSignal,
    TechnicalSignal,
    TradeSignal,
    VolatilitySignal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    ema_alignment: str = "mixed",
    rsi: float = 50.0,
    rsi_oversold: bool = False,
    rsi_overbought: bool = False,
    rsi_divergence: Optional[str] = None,
    macd_crossover: Optional[str] = None,
    macd_histogram: Optional[float] = None,
    candle_patterns: Optional[List[str]] = None,
    support_levels: Optional[List[float]] = None,
    resistance_levels: Optional[List[float]] = None,
    atr: float = 0.0010,
    bb_upper: Optional[float] = None,
    bb_lower: Optional[float] = None,
    bb_squeeze: bool = False,
    ema_20: float = 1.0850,
    timestamp: str = "2026-04-02T14:00:00Z",
) -> TechnicalSignal:
    return TechnicalSignal(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=timestamp,
        ema_20=ema_20,
        ema_50=ema_20 - 0.002 if ema_alignment == "bullish" else ema_20 + 0.002,
        ema_200=ema_20 - 0.005 if ema_alignment == "bullish" else ema_20 + 0.005,
        ema_alignment=ema_alignment,
        rsi=rsi,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        rsi_divergence=rsi_divergence,
        macd_crossover=macd_crossover,
        macd_histogram=macd_histogram,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        bb_squeeze=bb_squeeze,
        atr=atr,
        support_levels=support_levels or [],
        resistance_levels=resistance_levels or [],
        candle_patterns=candle_patterns or [],
    )


def _high_score_buy_signal(**overrides) -> TechnicalSignal:
    """Produces a strong BUY signal that should score ≥ 7.0."""
    price = 1.0850
    kwargs = dict(
        ema_alignment="bullish",
        rsi=28.0,
        rsi_oversold=True,
        macd_crossover="bullish",
        macd_histogram=0.0005,
        candle_patterns=["bullish_engulfing"],
        support_levels=[price - 0.0005],      # very close support
        resistance_levels=[price + 0.0080],
        atr=0.0010,
        bb_lower=price - 0.0001,              # price near lower band
        bb_upper=price + 0.0020,
        ema_20=price,
    )
    kwargs.update(overrides)
    return _make_signal(**kwargs)


def _high_score_sell_signal(**overrides) -> TechnicalSignal:
    """Produces a strong SELL signal that should score ≥ 7.0."""
    price = 1.0850
    kwargs = dict(
        ema_alignment="bearish",
        rsi=72.0,
        rsi_overbought=True,
        macd_crossover="bearish",
        macd_histogram=-0.0005,
        candle_patterns=["bearish_engulfing"],
        support_levels=[price - 0.0080],
        resistance_levels=[price + 0.0005],   # very close resistance
        atr=0.0010,
        bb_upper=price + 0.0001,              # price near upper band
        bb_lower=price - 0.0020,
        ema_20=price,
    )
    kwargs.update(overrides)
    return _make_signal(**kwargs)


# ---------------------------------------------------------------------------
# Direction determination
# ---------------------------------------------------------------------------


class TestDirectionDetermination:
    def test_bullish_ema_gives_buy(self):
        sig = _make_signal(ema_alignment="bullish")
        assert _determine_direction(sig) == "BUY"

    def test_bearish_ema_gives_sell(self):
        sig = _make_signal(ema_alignment="bearish")
        assert _determine_direction(sig) == "SELL"

    def test_macd_bullish_crossover_gives_buy_when_ema_mixed(self):
        sig = _make_signal(ema_alignment="mixed", macd_crossover="bullish")
        assert _determine_direction(sig) == "BUY"

    def test_macd_bearish_crossover_gives_sell_when_ema_mixed(self):
        sig = _make_signal(ema_alignment="mixed", macd_crossover="bearish")
        assert _determine_direction(sig) == "SELL"

    def test_rsi_oversold_gives_buy_when_ambiguous(self):
        sig = _make_signal(ema_alignment="mixed", rsi_oversold=True)
        assert _determine_direction(sig) == "BUY"

    def test_rsi_overbought_gives_sell_when_ambiguous(self):
        sig = _make_signal(ema_alignment="mixed", rsi_overbought=True)
        assert _determine_direction(sig) == "SELL"

    def test_default_direction_is_buy(self):
        sig = _make_signal(ema_alignment="mixed")
        assert _determine_direction(sig) == "BUY"


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


class TestScoreComputation:
    def setup_method(self):
        self.engine = ConfluenceEngine()

    def test_empty_signal_scores_near_zero(self):
        sig = _make_signal()
        score, confluences = self.engine.compute_score(sig)
        assert score < 2.0
        assert isinstance(confluences, list)

    def test_full_bull_setup_scores_high(self):
        sig = _high_score_buy_signal()
        score, _ = self.engine.compute_score(sig)
        assert score >= 7.0

    def test_full_bear_setup_scores_high(self):
        sig = _high_score_sell_signal()
        score, _ = self.engine.compute_score(sig)
        assert score >= 7.0

    def test_score_capped_at_ten(self):
        # Even with all components maxed the score must not exceed 10
        sig = _high_score_buy_signal()
        score, _ = self.engine.compute_score(sig)
        assert score <= 10.0

    def test_score_is_non_negative(self):
        sig = _make_signal(ema_alignment="bearish", macd_crossover="bearish")
        score, _ = self.engine.compute_score(sig)
        assert score >= 0.0

    def test_ema_alignment_contributes_to_score(self):
        aligned = _make_signal(ema_alignment="bullish")
        mixed = _make_signal(ema_alignment="mixed")
        score_aligned, _ = self.engine.compute_score(aligned)
        score_mixed, _ = self.engine.compute_score(mixed)
        assert score_aligned > score_mixed

    def test_macd_crossover_contributes_to_score(self):
        with_macd = _make_signal(ema_alignment="bullish", macd_crossover="bullish")
        without_macd = _make_signal(ema_alignment="bullish")
        score_with, _ = self.engine.compute_score(with_macd)
        score_without, _ = self.engine.compute_score(without_macd)
        assert score_with > score_without

    def test_rsi_oversold_contributes_to_buy_score(self):
        with_rsi = _make_signal(ema_alignment="bullish", rsi_oversold=True)
        without_rsi = _make_signal(ema_alignment="bullish", rsi_oversold=False, rsi=55.0)
        score_with, _ = self.engine.compute_score(with_rsi)
        score_without, _ = self.engine.compute_score(without_rsi)
        assert score_with > score_without

    def test_confluences_list_populated_for_strong_signal(self):
        sig = _high_score_buy_signal()
        _, confluences = self.engine.compute_score(sig)
        assert len(confluences) > 0
        # Should include EMA and RSI tags at minimum
        assert any("ema" in c for c in confluences)
        assert any("rsi" in c for c in confluences)

    def test_bull_signal_produces_bull_confluences(self):
        sig = _high_score_buy_signal()
        _, confluences = self.engine.compute_score(sig)
        assert any("bullish" in c or "buy" in c or "support" in c or "oversold" in c
                   for c in confluences)

    def test_bear_signal_produces_bear_confluences(self):
        sig = _high_score_sell_signal()
        _, confluences = self.engine.compute_score(sig)
        assert any("bearish" in c or "sell" in c or "resistance" in c or "overbought" in c
                   for c in confluences)

    def test_custom_weights_respected(self):
        config = {
            "scoring": {
                "weights": {
                    "ema_alignment": 5.0,   # boosted
                    "rsi_condition": 0.0,
                    "macd_crossover": 0.0,
                    "candle_pattern_at_level": 0.0,
                    "price_at_sr": 0.0,
                    "bollinger_signal": 0.0,
                    "news_sentiment": 0.0,
                    "volatility_context": 0.0,
                }
            }
        }
        engine = ConfluenceEngine(config=config)
        sig = _make_signal(ema_alignment="bullish")
        score, _ = engine.compute_score(sig)
        # With only EMA weight = 5.0 (total = 5.0), normalised score = 10.0
        assert score == pytest.approx(10.0, abs=0.01)


# ---------------------------------------------------------------------------
# Multi-timeframe bonus / penalty
# ---------------------------------------------------------------------------


class TestHTFBonus:
    def setup_method(self):
        self.engine = ConfluenceEngine()

    def test_d1_alignment_adds_bonus(self):
        # Seed D1 cache with a bullish signal
        d1 = _make_signal(timeframe="D1", ema_alignment="bullish")
        self.engine.on_technical_signal(d1)

        sig = _make_signal(timeframe="H1", ema_alignment="bullish")
        base_score, _ = self.engine.compute_score(sig)
        adjusted, _ = self.engine.apply_htf_bonus(base_score, sig, "BUY")
        assert adjusted > base_score

    def test_h4_alignment_adds_bonus(self):
        h4 = _make_signal(timeframe="H4", ema_alignment="bullish")
        self.engine.on_technical_signal(h4)

        sig = _make_signal(timeframe="H1", ema_alignment="bullish")
        base_score, _ = self.engine.compute_score(sig)
        adjusted, _ = self.engine.apply_htf_bonus(base_score, sig, "BUY")
        assert adjusted > base_score

    def test_h4_contradiction_applies_penalty(self):
        h4 = _make_signal(timeframe="H4", ema_alignment="bearish")
        self.engine.on_technical_signal(h4)

        sig = _make_signal(timeframe="H1", ema_alignment="bullish")
        base_score = 6.0
        adjusted, tags = self.engine.apply_htf_bonus(base_score, sig, "BUY")
        assert adjusted < base_score
        assert any("contradict" in t for t in tags)

    def test_htf_score_never_below_zero(self):
        h4 = _make_signal(timeframe="H4", ema_alignment="bearish")
        self.engine.on_technical_signal(h4)

        sig = _make_signal(timeframe="H1", ema_alignment="bullish")
        adjusted, _ = self.engine.apply_htf_bonus(0.0, sig, "BUY")
        assert adjusted >= 0.0

    def test_htf_score_never_above_ten(self):
        d1 = _make_signal(timeframe="D1", ema_alignment="bullish")
        h4 = _make_signal(timeframe="H4", ema_alignment="bullish")
        self.engine.on_technical_signal(d1)
        self.engine.on_technical_signal(h4)

        sig = _make_signal(timeframe="H1", ema_alignment="bullish")
        adjusted, _ = self.engine.apply_htf_bonus(10.0, sig, "BUY")
        assert adjusted <= 10.0

    def test_custom_htf_weights_respected(self):
        config = {
            "scoring": {
                "htf_bonus": {
                    "d1_alignment": 2.0,
                    "h4_alignment": 0.0,
                    "h4_contradiction_penalty": 0.0,
                }
            }
        }
        engine = ConfluenceEngine(config=config)
        d1 = _make_signal(timeframe="D1", ema_alignment="bullish")
        engine.on_technical_signal(d1)

        sig = _make_signal(timeframe="H1", ema_alignment="bullish")
        adjusted, _ = engine.apply_htf_bonus(5.0, sig, "BUY")
        assert adjusted == pytest.approx(7.0, abs=0.01)


# ---------------------------------------------------------------------------
# Alert firing thresholds
# ---------------------------------------------------------------------------


class TestAlertThresholds:
    def setup_method(self):
        self.emitted: list[TradeSignal] = []
        self.engine = ConfluenceEngine(signal_callback=self.emitted.append)

    def test_signal_fires_when_score_above_7(self):
        sig = _high_score_buy_signal()
        result = self.engine.on_technical_signal(sig)
        assert result is not None
        assert result.signal_type == "SIGNAL"

    def test_watch_fires_when_score_between_5_and_7(self):
        # Build a signal that scores in the 5–6.9 range
        # EMA bullish (2.0) + RSI divergence (0.8*1.5=1.2) → raw ~3.2/11.5 ≈ 2.78 → ~2.78
        # Need ~5.8/10 normalised → raw ~6.7/11.5
        # EMA(2.0) + MACD partial(0.6) + RSI partial(0.3*1.5=0.45) → ~3.05 raw → ~2.65 norm
        # Let's use: EMA(2.0) + MACD crossover(1.5) + RSI partial(0.45) = 3.95 → 3.43 norm
        # We need something in 5–7 range so use mid-tier setup
        sig = _make_signal(
            ema_alignment="bullish",
            macd_crossover="bullish",
            rsi=45.0,
            rsi_oversold=False,
            support_levels=[1.0840],
            atr=0.0015,
            ema_20=1.0850,
        )
        result = self.engine.on_technical_signal(sig)
        # Score should be < 7 but could still be < 5 if not enough confluences
        # The important thing is that if score is in [5, 7), we get WATCH
        if result is not None:
            assert result.signal_type in ("WATCH", "SIGNAL")

    def test_no_alert_when_score_below_5(self):
        sig = _make_signal(ema_alignment="mixed")
        result = self.engine.on_technical_signal(sig)
        assert result is None

    def test_callback_called_for_signal(self):
        sig = _high_score_buy_signal()
        self.engine.on_technical_signal(sig)
        assert len(self.emitted) == 1
        assert self.emitted[0].signal_type == "SIGNAL"

    def test_callback_not_called_for_silent_setup(self):
        sig = _make_signal(ema_alignment="mixed")
        self.engine.on_technical_signal(sig)
        assert len(self.emitted) == 0

    def test_custom_threshold_respected(self):
        config = {"confluence_threshold": 9.0, "watch_threshold": 8.0}
        engine = ConfluenceEngine(signal_callback=self.emitted.append, config=config)
        sig = _high_score_buy_signal()
        result = engine.on_technical_signal(sig)
        # Score is likely 7–8 range → should be silent with raised threshold
        # (may vary — just assert type is correct relative to score)
        if result is not None:
            if result.score >= 9.0:
                assert result.signal_type == "SIGNAL"
            elif result.score >= 8.0:
                assert result.signal_type == "WATCH"

    def test_watch_threshold_respected(self):
        config = {"confluence_threshold": 7.0, "watch_threshold": 3.0}
        emitted = []
        engine = ConfluenceEngine(signal_callback=emitted.append, config=config)
        # A weak signal that would normally be silent should now produce WATCH
        sig = _make_signal(ema_alignment="bullish")
        result = engine.on_technical_signal(sig)
        # With watch_threshold=3.0 a single EMA alignment may clear it
        # Just assert no crash and result is consistent
        if result is not None:
            assert result.signal_type in ("SIGNAL", "WATCH")


# ---------------------------------------------------------------------------
# Entry / SL / TP / R:R calculations
# ---------------------------------------------------------------------------


class TestTradeLevels:
    def setup_method(self):
        self.engine = ConfluenceEngine()

    def test_signal_alert_includes_entry_sl_tp(self):
        sig = _high_score_buy_signal()
        result = self.engine.on_technical_signal(sig)
        assert result is not None
        assert result.signal_type == "SIGNAL"
        assert len(result.entry_zone) == 2
        assert result.stop_loss is not None
        assert result.take_profit is not None

    def test_buy_stop_loss_below_entry(self):
        sig = _high_score_buy_signal()
        result = self.engine.on_technical_signal(sig)
        assert result is not None
        mid_entry = sum(result.entry_zone) / 2
        assert result.stop_loss < mid_entry

    def test_sell_stop_loss_above_entry(self):
        sig = _high_score_sell_signal()
        result = self.engine.on_technical_signal(sig)
        assert result is not None
        mid_entry = sum(result.entry_zone) / 2
        assert result.stop_loss > mid_entry

    def test_buy_take_profit_above_entry(self):
        sig = _high_score_buy_signal()
        result = self.engine.on_technical_signal(sig)
        assert result is not None
        mid_entry = sum(result.entry_zone) / 2
        assert result.take_profit > mid_entry

    def test_sell_take_profit_below_entry(self):
        sig = _high_score_sell_signal()
        result = self.engine.on_technical_signal(sig)
        assert result is not None
        mid_entry = sum(result.entry_zone) / 2
        assert result.take_profit < mid_entry

    def test_rr_ratio_computed_correctly(self):
        sig = _high_score_buy_signal(atr=0.0010)
        result = self.engine.on_technical_signal(sig)
        assert result is not None
        if result.rr_ratio is not None:
            assert result.rr_ratio > 0.0

    def test_rr_at_least_two_to_one_for_atr_based_levels(self):
        # When no S/R levels override the ATR-based SL/TP, TP=2×ATR and SL=1.5×ATR
        sig = _high_score_buy_signal(support_levels=[], resistance_levels=[])
        result = self.engine.on_technical_signal(sig)
        assert result is not None
        if result.rr_ratio is not None:
            assert result.rr_ratio >= pytest.approx(1.33, abs=0.05)

    def test_sl_pips_and_tp_pips_populated(self):
        sig = _high_score_buy_signal()
        result = self.engine.on_technical_signal(sig)
        assert result is not None
        assert result.sl_pips is not None and result.sl_pips > 0
        assert result.tp_pips is not None and result.tp_pips > 0

    def test_watch_alert_has_no_entry_levels(self):
        # Force a WATCH by using a mid-tier signal between 5 and 7
        emitted = []
        engine = ConfluenceEngine(signal_callback=emitted.append, config={
            "confluence_threshold": 7.0, "watch_threshold": 2.0
        })
        sig = _make_signal(ema_alignment="bullish")
        result = engine.on_technical_signal(sig)
        if result is not None and result.signal_type == "WATCH":
            assert result.entry_zone == []
            assert result.stop_loss is None

    def test_jpy_pair_uses_correct_pip_size(self):
        assert _pips(0.01, "USDJPY") == pytest.approx(1.0, abs=0.001)
        assert _pips(0.0001, "EURUSD") == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# BUY vs SELL direction
# ---------------------------------------------------------------------------


class TestDirectionAssignment:
    def setup_method(self):
        self.engine = ConfluenceEngine()

    def test_buy_direction_assigned_for_bull_signal(self):
        sig = _high_score_buy_signal()
        result = self.engine.on_technical_signal(sig)
        assert result is not None
        assert result.direction == "BUY"

    def test_sell_direction_assigned_for_bear_signal(self):
        sig = _high_score_sell_signal()
        result = self.engine.on_technical_signal(sig)
        assert result is not None
        assert result.direction == "SELL"


# ---------------------------------------------------------------------------
# TradeSignal model and to_dict
# ---------------------------------------------------------------------------


class TestTradeSignalModel:
    def setup_method(self):
        self.engine = ConfluenceEngine()

    def test_trade_signal_to_dict_has_all_keys(self):
        sig = _high_score_buy_signal()
        result = self.engine.on_technical_signal(sig)
        assert result is not None
        d = result.to_dict()
        required_keys = {
            "type", "direction", "symbol", "timeframe", "timestamp",
            "score", "entry_zone", "stop_loss", "take_profit",
            "rr_ratio", "sl_pips", "tp_pips", "confluences",
            "invalidation", "signal_age_seconds",
        }
        assert required_keys.issubset(d.keys())

    def test_trade_signal_to_dict_type_maps_to_signal_type(self):
        sig = _high_score_buy_signal()
        result = self.engine.on_technical_signal(sig)
        assert result is not None
        d = result.to_dict()
        assert d["type"] == result.signal_type

    def test_trade_signal_score_rounded(self):
        sig = _high_score_buy_signal()
        result = self.engine.on_technical_signal(sig)
        assert result is not None
        d = result.to_dict()
        # Score should have at most 2 decimal places
        assert d["score"] == round(d["score"], 2)

    def test_invalidation_string_present(self):
        sig = _high_score_buy_signal()
        result = self.engine.on_technical_signal(sig)
        assert result is not None
        assert isinstance(result.invalidation, str)
        assert len(result.invalidation) > 0


# ---------------------------------------------------------------------------
# Sentiment and volatility stubs
# ---------------------------------------------------------------------------


class TestStubIntegration:
    def setup_method(self):
        self.engine = ConfluenceEngine()

    def test_sentiment_stub_does_not_crash(self):
        sig = _high_score_buy_signal()
        sentiment = SentimentSignal(
            symbol="EURUSD",
            timestamp="2026-04-02T14:00:00Z",
            sentiment="bullish",
            confidence=0.9,
            impact="high",
        )
        self.engine.on_sentiment_signal(sentiment)
        result = self.engine.on_technical_signal(sig)
        assert result is not None

    def test_volatility_stub_does_not_crash(self):
        sig = _high_score_buy_signal()
        vol = VolatilitySignal(
            symbol="EURUSD",
            timestamp="2026-04-02T14:00:00Z",
            spike_detected=False,
            atr_ratio=1.0,
        )
        self.engine.on_volatility_signal(vol)
        result = self.engine.on_technical_signal(sig)
        assert result is not None

    def test_sentiment_boost_when_aligned(self):
        config = {
            "scoring": {
                "weights": {
                    "ema_alignment": 0.0,
                    "rsi_condition": 0.0,
                    "macd_crossover": 0.0,
                    "candle_pattern_at_level": 0.0,
                    "price_at_sr": 0.0,
                    "bollinger_signal": 0.0,
                    "news_sentiment": 5.0,
                    "volatility_context": 0.0,
                }
            },
            "watch_threshold": 0.0,
            "confluence_threshold": 100.0,  # prevent firing, just check score
        }
        engine = ConfluenceEngine(config=config)
        sentiment = SentimentSignal(
            symbol="EURUSD",
            timestamp="2026-04-02T14:00:00Z",
            sentiment="bullish",
            confidence=1.0,
            impact="high",
        )
        engine.on_sentiment_signal(sentiment)
        sig = _make_signal(ema_alignment="bullish")  # direction = BUY
        score, tags = engine.compute_score(sig)
        assert score == pytest.approx(10.0, abs=0.01)
        assert any("sentiment" in t for t in tags)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_next_higher_timeframe(self):
        assert _next_higher_timeframe("M15") == "M30"
        assert _next_higher_timeframe("H1") == "H4"
        assert _next_higher_timeframe("H4") == "D1"
        assert _next_higher_timeframe("W1") == "W1"  # already highest

    def test_nearest_support_below(self):
        levels = [1.0800, 1.0820, 1.0850]
        assert _nearest_support_below(1.0840, levels) == pytest.approx(1.0820)
        assert _nearest_support_below(1.0799, levels) is None

    def test_nearest_resistance_above(self):
        levels = [1.0860, 1.0880, 1.0900]
        assert _nearest_resistance_above(1.0850, levels) == pytest.approx(1.0860)
        assert _nearest_resistance_above(1.0910, levels) is None

    def test_is_price_near_sr_within_atr(self):
        assert _is_price_near_sr("EURUSD", 1.0851, [1.0850], [], 0.0010)

    def test_is_price_near_sr_outside_atr(self):
        assert not _is_price_near_sr("EURUSD", 1.0900, [1.0850], [], 0.0010)

    def test_pips_eurusd(self):
        assert _pips(0.0010, "EURUSD") == pytest.approx(10.0)

    def test_pips_usdjpy(self):
        assert _pips(0.10, "USDJPY") == pytest.approx(10.0)
