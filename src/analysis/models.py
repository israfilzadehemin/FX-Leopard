"""
Shared data models for the FX-Leopard analysis pipeline.

These dataclasses are used across all analysis modules so that each layer
speaks the same language when passing signals between components.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OHLCVCandle:
    """A completed OHLCV candle for a single symbol and timeframe."""

    symbol: str
    timeframe: str
    timestamp: str          # ISO-8601 UTC string, e.g. "2026-04-02T14:00:00Z"
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class TechnicalSignal:
    """
    Full snapshot of all computed technical indicators for one candle close.

    Emitted by TechnicalEngine after every completed candle and forwarded
    to the confluence scoring engine.
    """

    symbol: str
    timeframe: str
    timestamp: str          # ISO-8601 UTC string matching the candle

    # --- Trend ---
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    ema_alignment: str = "mixed"        # "bullish" | "bearish" | "mixed"
    adx: Optional[float] = None
    trending: bool = False              # ADX > 25

    # --- Momentum ---
    rsi: Optional[float] = None
    rsi_oversold: bool = False
    rsi_overbought: bool = False
    rsi_divergence: Optional[str] = None   # "bullish" | "bearish" | None

    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    macd_crossover: Optional[str] = None   # "bullish" | "bearish" | None

    # --- Volatility ---
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_squeeze: bool = False

    atr: Optional[float] = None
    atr_ratio: Optional[float] = None      # current ATR / 14-period ATR average

    # --- Structure ---
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    candle_patterns: List[str] = field(default_factory=list)

    # --- Event log ---
    events: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict suitable for JSON serialisation or logging."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp,
            "indicators": {
                "ema_20": self.ema_20,
                "ema_50": self.ema_50,
                "ema_200": self.ema_200,
                "ema_alignment": self.ema_alignment,
                "adx": self.adx,
                "trending": self.trending,
                "rsi": self.rsi,
                "rsi_oversold": self.rsi_oversold,
                "rsi_overbought": self.rsi_overbought,
                "rsi_divergence": self.rsi_divergence,
                "macd_line": self.macd_line,
                "macd_signal": self.macd_signal,
                "macd_histogram": self.macd_histogram,
                "macd_crossover": self.macd_crossover,
                "bb_upper": self.bb_upper,
                "bb_middle": self.bb_middle,
                "bb_lower": self.bb_lower,
                "bb_squeeze": self.bb_squeeze,
                "atr": self.atr,
                "atr_ratio": self.atr_ratio,
                "support_levels": self.support_levels,
                "resistance_levels": self.resistance_levels,
                "candle_patterns": self.candle_patterns,
            },
            "events": self.events,
        }


# ---------------------------------------------------------------------------
# Stubs for future issues
# ---------------------------------------------------------------------------


@dataclass
class SentimentSignal:
    """
    Output from the News/GPT sentiment engine (Issue #4 stub).

    Fields will be expanded when Issue #4 is implemented.
    """

    symbol: str
    timestamp: str
    sentiment: str = "neutral"      # "bullish" | "bearish" | "neutral"
    confidence: float = 0.0
    impact: str = "low"             # "high" | "medium" | "low"
    headlines: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class VolatilitySignal:
    """
    Output from the Volatility & Spike Monitor (Issue #6 stub).

    Fields will be expanded when Issue #6 is implemented.
    """

    symbol: str
    timestamp: str
    spike_detected: bool = False
    atr_ratio: Optional[float] = None
    pips_moved: Optional[float] = None
    window_minutes: int = 5
    catalyst: str = ""


@dataclass
class TradeSignal:
    """
    Final aggregated output from the Confluence Scoring Engine (Issue #3 stub).

    Fields will be expanded when Issue #3 is implemented.
    """

    symbol: str
    timeframe: str
    timestamp: str
    direction: str = "neutral"      # "long" | "short" | "neutral"
    score: float = 0.0
    signal_type: str = "IGNORE"     # "SIGNAL" | "WATCH" | "IGNORE"
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: List[str] = field(default_factory=list)
