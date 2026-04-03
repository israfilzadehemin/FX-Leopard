"""
TechnicalEngine — Multi-timeframe technical indicator computation for FX-Leopard.

Receives completed OHLCV candles from the PriceFeed, maintains a rolling
history buffer per symbol per timeframe, computes a comprehensive set of
technical indicators using pandas-ta, and emits a TechnicalSignal after
every candle close.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

from analysis.models import OHLCVCandle, TechnicalSignal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_HISTORY = 200           # Rolling candle buffer length per symbol/timeframe
SR_LOOKBACK = 50            # Candles to scan for support/resistance swing points
ADX_TREND_THRESHOLD = 25.0  # ADX value above which a market is considered trending
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
BB_SQUEEZE_PCT = 0.002      # BBW (band-width %) threshold for squeeze
ATR_AVG_PERIOD = 14         # Period for ATR rolling average used in atr_ratio
RSI_DIVERGENCE_LOOKBACK = 14  # Candles to look back for RSI divergence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar(value) -> Optional[float]:
    """Safely convert a pandas scalar / numpy scalar / NaN to float or None."""
    if value is None:
        return None
    try:
        v = float(value)
        return None if np.isnan(v) else round(v, 6)
    except (TypeError, ValueError):
        return None


def _detect_ema_alignment(
    ema20: Optional[float],
    ema50: Optional[float],
    ema200: Optional[float],
) -> str:
    """Return 'bullish', 'bearish', or 'mixed' based on EMA ordering."""
    if None in (ema20, ema50, ema200):
        return "mixed"
    if ema20 > ema50 > ema200:
        return "bullish"
    if ema20 < ema50 < ema200:
        return "bearish"
    return "mixed"


def _detect_rsi_divergence(
    closes: List[float],
    rsi_values: List[float],
    lookback: int = RSI_DIVERGENCE_LOOKBACK,
) -> Optional[str]:
    """
    Detect regular RSI divergence over the last ``lookback`` candles.

    Bullish divergence: the current close sets a new low relative to the
    prior window, but RSI at the current bar is *higher* than it was at
    the prior price low — momentum is improving despite lower price.

    Bearish divergence: the current close sets a new high relative to the
    prior window, but RSI at the current bar is *lower* than it was at
    the prior price high — momentum is fading despite higher price.

    Returns 'bullish', 'bearish', or None.
    """
    if len(closes) < lookback + 1 or len(rsi_values) < lookback + 1:
        return None

    # Slice the last (lookback + 1) bars so the final bar is "current".
    price_window = closes[-(lookback + 1):]
    rsi_window = rsi_values[-(lookback + 1):]

    current_close = price_window[-1]
    current_rsi = rsi_window[-1]

    # Prior window: all bars except the current one.
    prior_prices = price_window[:-1]
    prior_rsis = rsi_window[:-1]

    # Bullish: current close is below the prior-window low, but RSI is higher
    # than it was at that prior low (momentum diverges positively).
    prior_low_idx = prior_prices.index(min(prior_prices))
    if current_close < prior_prices[prior_low_idx] and current_rsi > prior_rsis[prior_low_idx]:
        return "bullish"

    # Bearish: current close is above the prior-window high, but RSI is lower
    # than it was at that prior high (momentum diverges negatively).
    prior_high_idx = prior_prices.index(max(prior_prices))
    if current_close > prior_prices[prior_high_idx] and current_rsi < prior_rsis[prior_high_idx]:
        return "bearish"

    return None


def _detect_macd_crossover(
    histogram_prev: Optional[float],
    histogram_curr: Optional[float],
) -> Optional[str]:
    """
    Detect a MACD histogram crossover (zero-line cross of histogram).

    Returns 'bullish' (histogram crosses above zero), 'bearish' (crosses below),
    or None if no crossover.
    """
    if histogram_prev is None or histogram_curr is None:
        return None
    if histogram_prev <= 0 and histogram_curr > 0:
        return "bullish"
    if histogram_prev >= 0 and histogram_curr < 0:
        return "bearish"
    return None


def _find_swing_levels(
    highs: List[float],
    lows: List[float],
    lookback: int = SR_LOOKBACK,
) -> Tuple[List[float], List[float]]:
    """
    Identify swing high / swing low levels over the last ``lookback`` candles.

    A swing high is a bar whose high is greater than both its neighbours.
    A swing low is a bar whose low is less than both its neighbours.

    Returns (resistance_levels, support_levels) each sorted and deduplicated.
    """
    h = highs[-lookback:] if len(highs) >= lookback else highs[:]
    l = lows[-lookback:] if len(lows) >= lookback else lows[:]

    resistance: List[float] = []
    support: List[float] = []

    for i in range(1, len(h) - 1):
        if h[i] > h[i - 1] and h[i] > h[i + 1]:
            resistance.append(round(h[i], 6))
        if l[i] < l[i - 1] and l[i] < l[i + 1]:
            support.append(round(l[i], 6))

    # Deduplicate within a tolerance of 0.01 % of the last close price
    def _dedup(levels: List[float]) -> List[float]:
        if not levels:
            return []
        out: List[float] = [levels[0]]
        for v in levels[1:]:
            if all(abs(v - x) / max(abs(x), 1e-10) > 0.0001 for x in out):
                out.append(v)
        return sorted(out)

    return _dedup(sorted(resistance, reverse=True)), _dedup(sorted(support))


def _detect_candle_patterns(candle: OHLCVCandle, prev: Optional[OHLCVCandle]) -> List[str]:
    """
    Detect common single-candle and two-candle patterns.

    Patterns detected:
    - bullish_engulfing / bearish_engulfing
    - hammer
    - shooting_star
    - pin_bar  (long wick, small body on either end)
    - inside_bar
    """
    patterns: List[str] = []

    o, h, l, c = candle.open, candle.high, candle.low, candle.close
    body = abs(c - o)
    full_range = h - l
    upper_wick = h - max(c, o)
    lower_wick = min(c, o) - l

    if full_range == 0:
        return patterns

    body_ratio = body / full_range

    # --- Hammer (bullish reversal) ---
    # Small body near the top, long lower wick (at least 2× body), tiny upper wick
    if (
        body_ratio < 0.35
        and lower_wick >= 2 * body
        and upper_wick <= 0.1 * full_range
        and c > o  # bullish candle preferred
    ):
        patterns.append("hammer")

    # --- Shooting Star (bearish reversal) ---
    # Small body near the bottom, long upper wick (at least 2× body), tiny lower wick
    if (
        body_ratio < 0.35
        and upper_wick >= 2 * body
        and lower_wick <= 0.1 * full_range
        and c < o  # bearish candle preferred
    ):
        patterns.append("shooting_star")

    # --- Pin Bar (either direction, generic long-wick doji-like) ---
    # Very small body with a wick at least 3× the body on one side
    if body_ratio < 0.25 and (upper_wick >= 3 * body or lower_wick >= 3 * body):
        patterns.append("pin_bar")

    # Two-candle patterns require a previous candle
    if prev is not None:
        po, ph, pl, pc = prev.open, prev.high, prev.low, prev.close
        prev_body = abs(pc - po)
        prev_bearish = pc < po
        curr_bullish = c > o
        curr_bearish = c < o

        # --- Bullish Engulfing ---
        if (
            prev_bearish
            and curr_bullish
            and o <= pc
            and c >= po
            and body > prev_body * 0.8
        ):
            patterns.append("bullish_engulfing")

        # --- Bearish Engulfing ---
        if (
            not prev_bearish
            and curr_bearish
            and o >= pc
            and c <= po
            and body > prev_body * 0.8
        ):
            patterns.append("bearish_engulfing")

        # --- Inside Bar ---
        if h <= ph and l >= pl:
            patterns.append("inside_bar")

    return patterns


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class TechnicalEngine:
    """
    Computes technical indicators for all symbol/timeframe combinations.

    Usage::

        def on_signal(signal: TechnicalSignal):
            print(signal.to_dict())

        engine = TechnicalEngine(on_signal=on_signal)
        engine.on_candle(candle_dict)   # called by PriceFeed callback

    The ``on_candle`` method accepts either an ``OHLCVCandle`` dataclass or
    the plain ``dict`` format emitted by ``CandleBuffer`` in price_feed.py.
    """

    def __init__(
        self,
        on_signal: Optional[Callable[[TechnicalSignal], None]] = None,
        max_history: int = MAX_HISTORY,
    ) -> None:
        self._on_signal = on_signal
        self._max_history = max_history

        # history[symbol][timeframe] → deque of OHLCVCandle (oldest first)
        self._history: Dict[str, Dict[str, Deque[OHLCVCandle]]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def on_candle(self, candle) -> Optional[TechnicalSignal]:
        """
        Process a completed OHLCV candle.

        Accepts either an OHLCVCandle dataclass or the plain dict produced
        by CandleBuffer (keys: symbol, timeframe, open, high, low, close,
        volume, timestamp).

        Returns the TechnicalSignal that was computed and emitted, or None
        if there was insufficient history to compute indicators.
        """
        if isinstance(candle, dict):
            candle = OHLCVCandle(
                symbol=candle["symbol"],
                timeframe=candle["timeframe"],
                timestamp=candle["timestamp"],
                open=float(candle["open"]),
                high=float(candle["high"]),
                low=float(candle["low"]),
                close=float(candle["close"]),
                volume=int(candle["volume"]),
            )

        self._append_candle(candle)
        signal = self._compute_signal(candle.symbol, candle.timeframe, candle.timestamp)

        if signal is not None and self._on_signal is not None:
            try:
                self._on_signal(signal)
            except Exception as exc:
                logger.error("on_signal callback raised: %s", exc)

        return signal

    def get_history(self, symbol: str, timeframe: str) -> List[OHLCVCandle]:
        """Return a copy of the candle history for a symbol/timeframe pair."""
        try:
            return list(self._history[symbol][timeframe])
        except KeyError:
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_candle(self, candle: OHLCVCandle) -> None:
        sym, tf = candle.symbol, candle.timeframe
        if sym not in self._history:
            self._history[sym] = {}
        if tf not in self._history[sym]:
            self._history[sym][tf] = deque(maxlen=self._max_history)
        self._history[sym][tf].append(candle)

    def _build_dataframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        history = list(self._history[symbol][timeframe])
        return pd.DataFrame(
            {
                "open": [c.open for c in history],
                "high": [c.high for c in history],
                "low": [c.low for c in history],
                "close": [c.close for c in history],
                "volume": [c.volume for c in history],
            }
        )

    def _compute_signal(
        self, symbol: str, timeframe: str, timestamp: str
    ) -> Optional[TechnicalSignal]:
        history = list(self._history[symbol][timeframe])
        if len(history) < 2:
            return None

        df = self._build_dataframe(symbol, timeframe)
        signal = TechnicalSignal(symbol=symbol, timeframe=timeframe, timestamp=timestamp)
        events: List[str] = []

        closes = df["close"].tolist()
        highs = df["high"].tolist()
        lows = df["low"].tolist()

        # --- EMA ---
        ema20_s = ta.ema(df["close"], length=20)
        ema50_s = ta.ema(df["close"], length=50)
        ema200_s = ta.ema(df["close"], length=200)

        signal.ema_20 = _scalar(ema20_s.iloc[-1]) if ema20_s is not None else None
        signal.ema_50 = _scalar(ema50_s.iloc[-1]) if ema50_s is not None else None
        signal.ema_200 = _scalar(ema200_s.iloc[-1]) if ema200_s is not None else None

        signal.ema_alignment = _detect_ema_alignment(
            signal.ema_20, signal.ema_50, signal.ema_200
        )

        # --- ADX ---
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_df is not None and not adx_df.empty:
            adx_col = [c for c in adx_df.columns if c.startswith("ADX_")]
            if adx_col:
                signal.adx = _scalar(adx_df[adx_col[0]].iloc[-1])
                signal.trending = bool(
                    signal.adx is not None and signal.adx > ADX_TREND_THRESHOLD
                )

        # --- RSI ---
        rsi_s = ta.rsi(df["close"], length=14)
        rsi_values: List[float] = []
        if rsi_s is not None:
            rsi_values = [
                v for v in rsi_s.tolist() if v is not None and not (isinstance(v, float) and np.isnan(v))
            ]
            signal.rsi = _scalar(rsi_s.iloc[-1])

        if signal.rsi is not None:
            signal.rsi_oversold = signal.rsi < RSI_OVERSOLD
            signal.rsi_overbought = signal.rsi > RSI_OVERBOUGHT
            if signal.rsi_oversold:
                events.append("rsi_oversold")
            if signal.rsi_overbought:
                events.append("rsi_overbought")
            # Track recovery from extremes
            if len(rsi_values) >= 2:
                prev_rsi = rsi_values[-2]
                if prev_rsi < RSI_OVERSOLD and signal.rsi >= RSI_OVERSOLD:
                    events.append("rsi_recovering_from_oversold")
                if prev_rsi > RSI_OVERBOUGHT and signal.rsi <= RSI_OVERBOUGHT:
                    events.append("rsi_recovering_from_overbought")

        signal.rsi_divergence = _detect_rsi_divergence(closes, rsi_values)
        if signal.rsi_divergence:
            events.append(f"rsi_divergence_{signal.rsi_divergence}")

        # --- MACD ---
        macd_df = ta.macd(df["close"])
        prev_histogram: Optional[float] = None
        if macd_df is not None and not macd_df.empty:
            line_col = [c for c in macd_df.columns if "MACD_" in c and "MACDh" not in c and "MACDs" not in c]
            hist_col = [c for c in macd_df.columns if "MACDh_" in c]
            sig_col = [c for c in macd_df.columns if "MACDs_" in c]

            if line_col:
                signal.macd_line = _scalar(macd_df[line_col[0]].iloc[-1])
            if hist_col:
                signal.macd_histogram = _scalar(macd_df[hist_col[0]].iloc[-1])
                prev_val = macd_df[hist_col[0]].iloc[-2] if len(macd_df) >= 2 else None
                prev_histogram = _scalar(prev_val)
            if sig_col:
                signal.macd_signal = _scalar(macd_df[sig_col[0]].iloc[-1])

        signal.macd_crossover = _detect_macd_crossover(prev_histogram, signal.macd_histogram)
        if signal.macd_crossover:
            events.append(f"macd_crossover_{signal.macd_crossover}")

        # --- Bollinger Bands ---
        bb_df = ta.bbands(df["close"], length=20)
        if bb_df is not None and not bb_df.empty:
            lower_col = [c for c in bb_df.columns if c.startswith("BBL_")]
            mid_col = [c for c in bb_df.columns if c.startswith("BBM_")]
            upper_col = [c for c in bb_df.columns if c.startswith("BBU_")]
            bw_col = [c for c in bb_df.columns if c.startswith("BBB_")]

            if lower_col:
                signal.bb_lower = _scalar(bb_df[lower_col[0]].iloc[-1])
            if mid_col:
                signal.bb_middle = _scalar(bb_df[mid_col[0]].iloc[-1])
            if upper_col:
                signal.bb_upper = _scalar(bb_df[upper_col[0]].iloc[-1])

            # Squeeze: band-width percentage (BBB) below threshold
            if bw_col:
                bbw = _scalar(bb_df[bw_col[0]].iloc[-1])
                if bbw is not None and signal.bb_middle is not None and signal.bb_middle != 0:
                    # BBB is already a percentage of middle band in pandas-ta
                    signal.bb_squeeze = bbw < (BB_SQUEEZE_PCT * 100)
                    if signal.bb_squeeze:
                        events.append("bb_squeeze")

        # --- ATR ---
        atr_s = ta.atr(df["high"], df["low"], df["close"], length=14)
        if atr_s is not None:
            signal.atr = _scalar(atr_s.iloc[-1])
            valid_atrs = atr_s.dropna()
            if len(valid_atrs) >= ATR_AVG_PERIOD and signal.atr is not None:
                atr_avg = float(valid_atrs.iloc[-ATR_AVG_PERIOD:].mean())
                if atr_avg > 0:
                    signal.atr_ratio = round(signal.atr / atr_avg, 4)
                    if signal.atr_ratio > 1.5:
                        events.append("atr_spike")

        # --- Support & Resistance ---
        signal.resistance_levels, signal.support_levels = _find_swing_levels(highs, lows)

        # --- Candlestick Patterns ---
        current_candle = history[-1]
        prev_candle = history[-2] if len(history) >= 2 else None
        signal.candle_patterns = _detect_candle_patterns(current_candle, prev_candle)
        for pattern in signal.candle_patterns:
            events.append(pattern)

        signal.events = events

        logger.debug(
            "TechnicalSignal — %s %s: ema_align=%s rsi=%.1f adx=%.1f events=%s",
            symbol,
            timeframe,
            signal.ema_alignment,
            signal.rsi or 0,
            signal.adx or 0,
            events,
        )

        return signal
