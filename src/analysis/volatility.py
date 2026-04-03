"""
Volatility & Price Spike Monitor for FX-Leopard.

Watches all configured instruments in real time and fires a VolatilitySignal
the moment an abnormal price move or volatility expansion is detected.

Two detection modes
-------------------
A. ATR Expansion
   Computes ATR(14) and a rolling 14-period average of ATR values.
   If current ATR > atr_multiplier × average ATR → spike detected.

B. Pip Spike
   Tracks price movement within a rolling time window (default 5 minutes).
   If price moves more than pip_spike_threshold pips → spike detected.

Integration
-----------
- Passes VolatilitySignal to ConfluenceEngine.update_volatility()
- Passes VolatilitySignal to the notification layer for standalone alerts
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, List, Optional, Tuple

import pandas as pd

from analysis.models import OHLCVCandle, VolatilitySignal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pip sizes — instrument-aware
# ---------------------------------------------------------------------------

PIP_SIZES: Dict[str, float] = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "AUDUSD": 0.0001,
    "USDCAD": 0.0001,
    "USDCHF": 0.0001,
    "NZDUSD": 0.0001,
    "USDJPY": 0.01,
    "EURJPY": 0.01,
    "GBPJPY": 0.01,
    "AUDJPY": 0.01,
    "CADJPY": 0.01,
    "NZDJPY": 0.01,
    "CHFJPY": 0.01,
    "XAUUSD": 0.10,
    "XAGUSD": 0.001,
    "USOIL":  0.01,
    "UKOIL":  0.01,
}

_DEFAULT_PIP_SIZE = 0.0001


def get_pip_size(symbol: str) -> float:
    """Return the pip unit size for *symbol* (instrument-aware)."""
    return PIP_SIZES.get(symbol.upper(), _DEFAULT_PIP_SIZE)


def price_to_pips(price_diff: float, symbol: str) -> float:
    """Convert an absolute price difference to pips for *symbol*."""
    pip = get_pip_size(symbol)
    return abs(price_diff) / pip


# ---------------------------------------------------------------------------
# ATR helpers
# ---------------------------------------------------------------------------

def _compute_atr(candles: List[OHLCVCandle], period: int = 14) -> Optional[float]:
    """
    Compute the most recent ATR value from a list of OHLCVCandle objects.

    Returns *None* if there are fewer than *period + 1* candles.
    """
    if len(candles) < period + 1:
        return None

    closes = [c.close for c in candles]
    highs  = [c.high  for c in candles]
    lows   = [c.low   for c in candles]

    tr_values: List[float] = []
    for i in range(1, len(candles)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        tr_values.append(tr)

    if len(tr_values) < period:
        return None

    # Wilder-smoothed ATR
    atr = sum(tr_values[:period]) / period
    for tr in tr_values[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def _compute_atr_series(candles: List[OHLCVCandle], period: int = 14) -> List[float]:
    """
    Return a list of ATR values (one per candle from index *period* onwards).

    Used to compute the rolling average of ATR values.
    """
    if len(candles) < period + 1:
        return []

    closes = [c.close for c in candles]
    highs  = [c.high  for c in candles]
    lows   = [c.low   for c in candles]

    tr_values: List[float] = []
    for i in range(1, len(candles)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        tr_values.append(tr)

    if len(tr_values) < period:
        return []

    atr_series: List[float] = []
    atr = sum(tr_values[:period]) / period
    atr_series.append(atr)
    for tr in tr_values[period:]:
        atr = (atr * (period - 1) + tr) / period
        atr_series.append(atr)
    return atr_series


# ---------------------------------------------------------------------------
# Per-symbol state
# ---------------------------------------------------------------------------

class _SymbolState:
    """Holds rolling buffers and cooldown info for a single symbol."""

    def __init__(self, pip_spike_window_seconds: int) -> None:
        # Rolling candle buffer (keep enough for ATR + averaging)
        self.candles: List[OHLCVCandle] = []

        # Rolling tick/price buffer for pip-spike detection:
        # each entry is (timestamp_unix_seconds, price)
        self.pip_window_seconds = pip_spike_window_seconds
        self.price_ticks: Deque[Tuple[float, float]] = deque()

        # Cooldown tracking
        self.last_alert_ts: Optional[float] = None

    def add_candle(self, candle: OHLCVCandle, max_candles: int = 60) -> None:
        self.candles.append(candle)
        if len(self.candles) > max_candles:
            self.candles = self.candles[-max_candles:]

    def add_price_tick(self, price: float, ts_seconds: float) -> None:
        self.price_ticks.append((ts_seconds, price))
        # Evict ticks outside the rolling window
        cutoff = ts_seconds - self.pip_window_seconds
        while self.price_ticks and self.price_ticks[0][0] < cutoff:
            self.price_ticks.popleft()


# ---------------------------------------------------------------------------
# VolatilityMonitor
# ---------------------------------------------------------------------------


class VolatilityMonitor:
    """
    Watches configured symbols for ATR expansion and pip spikes.

    Parameters
    ----------
    config : dict
        Loaded ``config.yaml`` dictionary.  The relevant sub-key is
        ``volatility`` with fields:

        * ``atr_multiplier``       (float, default 1.5)
        * ``pip_spike_threshold``  (int,   default 30  pips)
        * ``pip_spike_window_seconds`` (int, default 300 s)
        * ``cooldown_minutes``     (int,   default 10 min)
        * ``monitor_timeframe``    (str,   default "H1")

    volatility_callback : callable, optional
        Receives a :class:`~analysis.models.VolatilitySignal` when either
        condition fires.  Typically wired to
        ``ConfluenceEngine.update_volatility``.

    notification_callback : callable, optional
        Receives the formatted Telegram alert string for standalone
        VOLATILITY notifications.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        volatility_callback: Optional[Callable[[VolatilitySignal], None]] = None,
        notification_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        cfg = (config or {}).get("volatility", config or {})

        self._atr_multiplier: float     = float(cfg.get("atr_multiplier", 1.5))
        self._pip_spike_threshold: float = float(cfg.get("pip_spike_threshold", 30))
        self._pip_window_seconds: int   = int(cfg.get("pip_spike_window_seconds", 300))
        self._cooldown_seconds: float   = float(cfg.get("cooldown_minutes", 10)) * 60
        self._monitor_timeframe: str    = str(cfg.get("monitor_timeframe", "H1"))

        self._volatility_callback    = volatility_callback
        self._notification_callback  = notification_callback

        # Per-symbol state
        self._states: Dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------
    # Public feed methods
    # ------------------------------------------------------------------

    def on_candle(self, candle) -> Optional[VolatilitySignal]:
        """
        Process a completed OHLCV candle.

        Accepts either an :class:`~analysis.models.OHLCVCandle` dataclass or
        the plain ``dict`` format emitted by ``CandleBuffer`` in price_feed.py
        (keys: symbol, timeframe, open, high, low, close, volume, timestamp).

        Only candles matching ``monitor_timeframe`` are used for ATR
        expansion checks.  All candles update the pip-spike price tick
        with the candle close.

        Returns the emitted :class:`~analysis.models.VolatilitySignal`, or
        ``None`` if no alert was fired.
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

        state = self._get_state(candle.symbol)

        # Parse candle timestamp to unix seconds
        ts_s = _parse_ts(candle.timestamp)

        # Update pip-spike buffer with this close price
        state.add_price_tick(candle.close, ts_s)

        # Only run ATR analysis on the monitored timeframe
        if candle.timeframe == self._monitor_timeframe:
            state.add_candle(candle)

        return self._evaluate(candle.symbol, candle.close, candle.timestamp, ts_s)

    def on_price_tick(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[str] = None,
    ) -> Optional[VolatilitySignal]:
        """
        Process a real-time price tick (bid or mid price).

        Used for pip-spike detection between candle closes.

        Returns the emitted :class:`~analysis.models.VolatilitySignal`, or
        ``None`` if no alert was fired.
        """
        ts_str = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ts_s   = _parse_ts(ts_str)

        state = self._get_state(symbol)
        state.add_price_tick(price, ts_s)

        return self._evaluate(symbol, price, ts_str, ts_s)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        symbol: str,
        current_price: float,
        timestamp: str,
        ts_s: float,
    ) -> Optional[VolatilitySignal]:
        """Check both conditions; emit a signal if either fires."""
        state = self._get_state(symbol)

        # Cooldown guard
        if self._in_cooldown(state, ts_s):
            return None

        # --- Condition A: ATR expansion ---
        atr_current: Optional[float] = None
        atr_baseline: Optional[float] = None  # ATR(14) — the "normal" volatility level
        atr_ratio:    Optional[float] = None
        atr_spike     = False

        if len(state.candles) >= 15:  # need at least period+1
            candles = state.candles
            # atr_baseline = Wilder ATR(14) computed from the last 15 candles.
            # This represents the "normal" volatility level for the symbol.
            atr_baseline = _compute_atr(candles[-15:], period=14)
            # atr_current = True Range of the most recent candle.
            # Comparing TR directly against the smoothed baseline is more responsive
            # to single-candle spikes than comparing two Wilder-smoothed values.
            last  = candles[-1]
            prev  = candles[-2]
            atr_current = max(
                last.high - last.low,
                abs(last.high - prev.close),
                abs(last.low  - prev.close),
            )
            if atr_baseline is not None and atr_baseline > 0:
                atr_ratio = atr_current / atr_baseline
                if atr_ratio > self._atr_multiplier:
                    atr_spike = True

        # --- Condition B: Pip spike ---
        pip_spike   = False
        price_before: Optional[float] = None
        pips_moved   = 0.0

        if len(state.price_ticks) >= 2:
            oldest_ts, oldest_price = state.price_ticks[0]
            price_before = oldest_price
            pips_moved = price_to_pips(current_price - price_before, symbol)
            if pips_moved >= self._pip_spike_threshold:
                pip_spike = True

        if not (atr_spike or pip_spike):
            return None

        # --- Build signal ---
        trigger = "atr_expansion" if atr_spike else "pip_spike"
        if atr_spike and pip_spike:
            trigger = "atr_expansion"  # ATR expansion takes precedence

        direction = _determine_direction(current_price, price_before)

        # score_contribution: max 0.5, scaled by atr_ratio (capped at 2×)
        score_contribution = 0.0
        if atr_ratio is not None:
            score_contribution = min(0.5, 0.5 * (atr_ratio / 2.0))
        elif pip_spike:
            score_contribution = 0.4

        signal = VolatilitySignal(
            symbol=symbol,
            timestamp=timestamp,
            trigger=trigger,
            spike_detected=True,
            magnitude=round(atr_ratio if trigger == "atr_expansion" else pips_moved, 4),
            direction=direction,
            price_before=price_before,
            price_now=current_price,
            atr_current=atr_current,
            atr_average=atr_baseline,   # expose baseline as atr_average for consumers
            atr_ratio=atr_ratio,
            window_seconds=self._pip_window_seconds,
            pips_moved=round(pips_moved, 1),  # always include for context
            probable_catalyst="Pending news check",
            score_contribution=round(score_contribution, 4),
            catalyst="",
        )

        logger.info(
            "⚡ Volatility spike detected: %s trigger=%s magnitude=%.4f direction=%s",
            symbol, trigger, signal.magnitude, direction,
        )

        # Update cooldown
        state.last_alert_ts = ts_s

        # Fire callbacks
        if self._volatility_callback is not None:
            self._volatility_callback(signal)

        if self._notification_callback is not None:
            alert_text = format_volatility_alert(signal)
            self._notification_callback(alert_text)

        return signal

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState(self._pip_window_seconds)
        return self._states[symbol]

    def _in_cooldown(self, state: _SymbolState, ts_s: float) -> bool:
        if state.last_alert_ts is None:
            return False
        return (ts_s - state.last_alert_ts) < self._cooldown_seconds


# ---------------------------------------------------------------------------
# Alert formatter
# ---------------------------------------------------------------------------


def format_volatility_alert(signal: VolatilitySignal) -> str:
    """
    Return the Telegram-ready plain-text alert string for a VolatilitySignal.

    Example output::

        ⚡ VOLATILITY SPIKE DETECTED

        📍 GBPUSD — Pip Spike
        📉 Moved 45 pips in under 5 minutes
        💰 From 1.2680 → 1.2635

        📊 ATR ratio: 1.71x average
        🕐 13:45 UTC

        ⚠️ Possible catalyst: Pending news scan
        👀 Monitor for continuation or reversal setup.
    """
    trigger_label = "ATR Expansion" if signal.trigger == "atr_expansion" else "Pip Spike"
    direction_arrow = "📉" if signal.direction == "bearish" else "📈"

    # Time portion only
    try:
        dt = datetime.fromisoformat(signal.timestamp.replace("Z", "+00:00"))
        time_str = dt.strftime("%H:%M UTC")
    except ValueError:
        time_str = signal.timestamp

    lines = [
        "⚡ VOLATILITY SPIKE DETECTED",
        "",
        f"📍 {signal.symbol} — {trigger_label}",
    ]

    if signal.trigger == "pip_spike" and signal.pips_moved is not None:
        window_min = signal.window_seconds // 60
        lines.append(
            f"{direction_arrow} Moved {signal.pips_moved:.0f} pips"
            f" in under {window_min} minutes"
        )
    elif signal.trigger == "atr_expansion" and signal.atr_ratio is not None:
        lines.append(
            f"{direction_arrow} ATR expanded {signal.atr_ratio:.2f}x vs average"
        )

    if signal.price_before is not None and signal.price_now is not None:
        lines.append(f"💰 From {signal.price_before:.5g} → {signal.price_now:.5g}")

    lines.append("")

    if signal.atr_ratio is not None:
        lines.append(f"📊 ATR ratio: {signal.atr_ratio:.2f}x average")

    lines.append(f"🕐 {time_str}")
    lines.append("")

    catalyst = signal.probable_catalyst or signal.catalyst or "Pending news scan"
    lines.append(f"⚠️ Possible catalyst: {catalyst}")
    lines.append("👀 Monitor for continuation or reversal setup.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_ts(timestamp: str) -> float:
    """Parse an ISO-8601 UTC timestamp string to a Unix timestamp (float)."""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc).timestamp()


def _determine_direction(current_price: float, price_before: Optional[float]) -> str:
    """Return "bullish", "bearish", or "neutral" based on price movement."""
    if price_before is None:
        return "neutral"
    if current_price > price_before:
        return "bullish"
    if current_price < price_before:
        return "bearish"
    return "neutral"
