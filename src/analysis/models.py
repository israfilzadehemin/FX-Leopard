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
    Output from the News/GPT sentiment engine (Issue #4).

    ``direction`` is the canonical field for the bias ("bullish" | "bearish" | "neutral").
    ``sentiment`` is kept as an alias for backward compatibility with the
    confluence engine and existing tests.
    ``strength`` expresses how strong the directional bias is (0.0–1.0).
    ``score_contribution`` is the pre-computed contribution to the confluence
    score (max 1.5, matching the ``news_sentiment`` weight).
    """

    symbol: str
    timestamp: str
    direction: str = "neutral"          # "bullish" | "bearish" | "neutral"
    sentiment: str = "neutral"          # alias kept for backward compat
    strength: float = 0.0               # 0.0–1.0
    confidence: float = 0.0             # 0.0–1.0
    impact: str = "low"                 # "high" | "medium" | "low"
    headlines: List[str] = field(default_factory=list)
    summary: str = ""
    score_contribution: float = 0.0     # pre-computed confluence score (max 1.5)


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
    Final aggregated output from the Confluence Scoring Engine.

    Emitted for SIGNAL (score ≥ 7.0) and WATCH (score 5.0–6.9) alerts.
    IGNORE alerts are not emitted — they are silently dropped.
    """

    # --- Identity ---
    symbol: str
    timeframe: str
    timestamp: str                      # ISO-8601 UTC string

    # --- Classification ---
    signal_type: str = "IGNORE"         # "SIGNAL" | "WATCH" | "IGNORE"
    direction: str = "BUY"              # "BUY" | "SELL"
    score: float = 0.0                  # Normalised confluence score 0.0–10.0

    # --- Entry / Exit levels (populated for SIGNAL only) ---
    entry_zone: List[float] = field(default_factory=list)   # [low, high]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    rr_ratio: Optional[float] = None
    sl_pips: Optional[float] = None
    tp_pips: Optional[float] = None

    # --- Reasoning ---
    confluences: List[str] = field(default_factory=list)
    invalidation: str = ""

    # --- Timing ---
    signal_age_seconds: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict suitable for JSON serialisation or Telegram."""
        return {
            "type": self.signal_type,
            "direction": self.direction,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp,
            "score": round(self.score, 2),
            "entry_zone": self.entry_zone,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "rr_ratio": self.rr_ratio,
            "sl_pips": self.sl_pips,
            "tp_pips": self.tp_pips,
            "confluences": self.confluences,
            "invalidation": self.invalidation,
            "signal_age_seconds": self.signal_age_seconds,
        }
