"""
ConfluenceEngine — Aggregates multi-source signals into a unified score and
fires SIGNAL / WATCH trade alerts for FX-Leopard.

Scoring breakdown (configurable via config.yaml):

  Component                  Max Score
  ─────────────────────────────────────
  EMA alignment              2.0
  RSI condition              1.5
  MACD crossover             1.5
  Candle pattern at level    2.0
  Price at S/R               1.5
  Bollinger Band signal      1.0
  News sentiment             1.5  (stub — always 0 until Issue #4 merged)
  Volatility context         0.5  (stub — always 0 until Issue #6 merged)
  ─────────────────────────────────────
  Total possible            11.5  → normalised to 10.0

Multi-timeframe bonuses/penalties are then applied on top of the normalised
score before comparing against the SIGNAL / WATCH thresholds.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from analysis.models import SentimentSignal, TechnicalSignal, TradeSignal, VolatilitySignal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default scoring weights (can be overridden via config.yaml)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "ema_alignment": 2.0,
    "rsi_condition": 1.5,
    "macd_crossover": 1.5,
    "candle_pattern_at_level": 2.0,
    "price_at_sr": 1.5,
    "bollinger_signal": 1.0,
    "news_sentiment": 1.5,
    "volatility_context": 0.5,
}

DEFAULT_HTF_BONUS: Dict[str, float] = {
    "d1_alignment": 0.5,
    "h4_alignment": 0.3,
    "h4_contradiction_penalty": -1.0,
}

# Maximum raw score when every component is maxed out
_MAX_RAW_SCORE: float = sum(DEFAULT_WEIGHTS.values())  # 11.5

# Thresholds
DEFAULT_SIGNAL_THRESHOLD: float = 7.0
DEFAULT_WATCH_THRESHOLD: float = 5.0

# Pip sizes used when calculating sl_pips / tp_pips
_JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "NZDJPY", "CHFJPY"}
_COMMODITY_PAIRS = {"XAUUSD", "XAGUSD", "USOIL", "UKOIL"}


def _pip_value(symbol: str) -> float:
    """Return the pip-unit size for a given symbol."""
    if symbol in _JPY_PAIRS:
        return 0.01
    if symbol in _COMMODITY_PAIRS:
        return 0.1
    return 0.0001


def _pips(price_diff: float, symbol: str) -> float:
    """Convert a price difference to pips for the given symbol."""
    return abs(price_diff) / _pip_value(symbol)


# ---------------------------------------------------------------------------
# ConfluenceEngine
# ---------------------------------------------------------------------------


class ConfluenceEngine:
    """
    Aggregates TechnicalSignal (+ optional SentimentSignal and
    VolatilitySignal) into a single confluenc score and fires a
    TradeSignal callback when the score clears the configured thresholds.

    Usage::

        engine = ConfluenceEngine(
            signal_callback=my_notification_handler,
            config=loaded_yaml_dict,
        )
        engine.on_technical_signal(tech_signal)

    The callback receives a :class:`~analysis.models.TradeSignal` whose
    ``signal_type`` is ``"SIGNAL"`` or ``"WATCH"``.  Silent setups
    (score < watch_threshold) are not forwarded.
    """

    def __init__(
        self,
        signal_callback: Optional[Callable[[TradeSignal], None]] = None,
        config: Optional[Dict] = None,
    ) -> None:
        self._callback = signal_callback
        cfg = config or {}

        # Load scoring weights
        weights_cfg: Dict[str, float] = (
            cfg.get("scoring", {}).get("weights", {})
        )
        self._weights: Dict[str, float] = {**DEFAULT_WEIGHTS, **weights_cfg}

        htf_cfg: Dict[str, float] = (
            cfg.get("scoring", {}).get("htf_bonus", {})
        )
        self._htf: Dict[str, float] = {**DEFAULT_HTF_BONUS, **htf_cfg}

        # Thresholds
        self._signal_threshold: float = float(
            cfg.get("confluence_threshold", DEFAULT_SIGNAL_THRESHOLD)
        )
        self._watch_threshold: float = float(
            cfg.get("watch_threshold", DEFAULT_WATCH_THRESHOLD)
        )

        # Most-recent HTF snapshots per symbol, used for MTF bonuses
        self._htf_cache: Dict[str, Dict[str, TechnicalSignal]] = {}

        # Latest sentiment/volatility snapshots per symbol (stubs)
        self._sentiment_cache: Dict[str, SentimentSignal] = {}
        self._volatility_cache: Dict[str, VolatilitySignal] = {}

    # ------------------------------------------------------------------
    # Public feed methods
    # ------------------------------------------------------------------

    def on_technical_signal(self, signal: TechnicalSignal) -> Optional[TradeSignal]:
        """
        Main entry point.  Called by TechnicalEngine after every candle close.

        Caches higher-timeframe signals for MTF bonus computation, then
        computes a confluenced score and emits a TradeSignal if appropriate.

        Returns the emitted TradeSignal, or None if the setup was silent.
        """
        # Cache for HTF bonus lookup
        sym = signal.symbol
        if sym not in self._htf_cache:
            self._htf_cache[sym] = {}
        self._htf_cache[sym][signal.timeframe] = signal

        return self._evaluate(signal)

    def on_sentiment_signal(self, sentiment: SentimentSignal) -> None:
        """Cache the latest sentiment signal for a symbol (Issue #4 hook)."""
        self._sentiment_cache[sentiment.symbol] = sentiment

    def update_sentiment(self, signal: SentimentSignal) -> None:
        """Alias for :meth:`on_sentiment_signal` — preferred entry point for Issue #4."""
        self.on_sentiment_signal(signal)

    def on_volatility_signal(self, volatility: VolatilitySignal) -> None:
        """Cache the latest volatility signal for a symbol (Issue #6 hook)."""
        self._volatility_cache[volatility.symbol] = volatility

    def update_volatility(self, signal: VolatilitySignal) -> None:
        """
        Preferred entry point for Issue #6 — VolatilityMonitor calls this.

        Stores the signal so that the next :meth:`on_technical_signal` call
        for the same symbol picks it up when computing the confluence score.
        """
        self.on_volatility_signal(signal)

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def compute_score(
        self,
        signal: TechnicalSignal,
        sentiment: Optional[SentimentSignal] = None,
        volatility: Optional[VolatilitySignal] = None,
    ) -> tuple[float, List[str]]:
        """
        Compute the raw (pre-normalisation, pre-HTF-bonus) component scores.

        Returns ``(normalised_score_0_to_10, confluences_list)``.
        """
        direction = _determine_direction(signal)
        raw = 0.0
        confluences: List[str] = []

        # 1. EMA alignment
        ema_score, ema_tags = self._score_ema(signal, direction)
        raw += ema_score
        confluences.extend(ema_tags)

        # 2. RSI condition
        rsi_score, rsi_tags = self._score_rsi(signal, direction)
        raw += rsi_score
        confluences.extend(rsi_tags)

        # 3. MACD crossover
        macd_score, macd_tags = self._score_macd(signal, direction)
        raw += macd_score
        confluences.extend(macd_tags)

        # 4. Candle pattern at key level
        cp_score, cp_tags = self._score_candle_pattern(signal, direction)
        raw += cp_score
        confluences.extend(cp_tags)

        # 5. Price at S/R
        sr_score, sr_tags = self._score_price_at_sr(signal, direction)
        raw += sr_score
        confluences.extend(sr_tags)

        # 6. Bollinger Band signal
        bb_score, bb_tags = self._score_bollinger(signal, direction)
        raw += bb_score
        confluences.extend(bb_tags)

        # 7. News sentiment (stub — zero until Issue #4)
        sent_score, sent_tags = self._score_sentiment(signal, direction, sentiment)
        raw += sent_score
        confluences.extend(sent_tags)

        # 8. Volatility context (stub — zero until Issue #6)
        vol_score, vol_tags = self._score_volatility(signal, volatility)
        raw += vol_score
        confluences.extend(vol_tags)

        # Normalise to 0–10
        max_possible = sum(self._weights.values()) or _MAX_RAW_SCORE
        normalised = min(raw / max_possible * 10.0, 10.0)

        return normalised, confluences

    def apply_htf_bonus(
        self,
        base_score: float,
        signal: TechnicalSignal,
        direction: str,
    ) -> tuple[float, List[str]]:
        """
        Apply multi-timeframe bonus/penalty on top of the base normalised score.

        Returns ``(adjusted_score, extra_confluence_tags)``.
        """
        sym = signal.symbol
        htf_signals = self._htf_cache.get(sym, {})
        extra: List[str] = []
        adjustment = 0.0

        # D1 alignment
        d1 = htf_signals.get("D1")
        if d1 is not None:
            if d1.ema_alignment == direction.lower() + "ish" or d1.ema_alignment == (
                "bullish" if direction == "BUY" else "bearish"
            ):
                bonus = self._htf.get("d1_alignment", DEFAULT_HTF_BONUS["d1_alignment"])
                adjustment += bonus
                extra.append(f"d1_trend_aligns_{direction.lower()}")

        # H4 alignment or contradiction
        h4 = htf_signals.get("H4")
        if h4 is not None:
            h4_bullish = h4.ema_alignment == "bullish"
            h4_bearish = h4.ema_alignment == "bearish"
            if (direction == "BUY" and h4_bullish) or (direction == "SELL" and h4_bearish):
                bonus = self._htf.get("h4_alignment", DEFAULT_HTF_BONUS["h4_alignment"])
                adjustment += bonus
                extra.append(f"h4_trend_aligns_{direction.lower()}")
            elif (direction == "BUY" and h4_bearish) or (direction == "SELL" and h4_bullish):
                penalty = self._htf.get(
                    "h4_contradiction_penalty",
                    DEFAULT_HTF_BONUS["h4_contradiction_penalty"],
                )
                adjustment += penalty
                extra.append("h4_trend_contradicts_signal")

        adjusted = max(0.0, min(base_score + adjustment, 10.0))
        return adjusted, extra

    # ------------------------------------------------------------------
    # Component scorers
    # ------------------------------------------------------------------

    def _score_ema(
        self, signal: TechnicalSignal, direction: str
    ) -> tuple[float, List[str]]:
        max_w = self._weights.get("ema_alignment", DEFAULT_WEIGHTS["ema_alignment"])
        tags: List[str] = []
        if signal.ema_alignment == "bullish" and direction == "BUY":
            tags.append("ema_alignment_bullish")
            return max_w, tags
        if signal.ema_alignment == "bearish" and direction == "SELL":
            tags.append("ema_alignment_bearish")
            return max_w, tags
        return 0.0, tags

    def _score_rsi(
        self, signal: TechnicalSignal, direction: str
    ) -> tuple[float, List[str]]:
        max_w = self._weights.get("rsi_condition", DEFAULT_WEIGHTS["rsi_condition"])
        tags: List[str] = []
        score = 0.0
        if direction == "BUY":
            if signal.rsi_oversold:
                score = max_w
                tags.append("rsi_recovering_from_oversold")
            elif signal.rsi_divergence == "bullish":
                score = max_w * 0.8
                tags.append("rsi_bullish_divergence")
            elif signal.rsi is not None and signal.rsi < 50:
                score = max_w * 0.3
        else:  # SELL
            if signal.rsi_overbought:
                score = max_w
                tags.append("rsi_retreating_from_overbought")
            elif signal.rsi_divergence == "bearish":
                score = max_w * 0.8
                tags.append("rsi_bearish_divergence")
            elif signal.rsi is not None and signal.rsi > 50:
                score = max_w * 0.3
        return score, tags

    def _score_macd(
        self, signal: TechnicalSignal, direction: str
    ) -> tuple[float, List[str]]:
        max_w = self._weights.get("macd_crossover", DEFAULT_WEIGHTS["macd_crossover"])
        tags: List[str] = []
        if direction == "BUY" and signal.macd_crossover == "bullish":
            tags.append("macd_crossover_bullish")
            return max_w, tags
        if direction == "SELL" and signal.macd_crossover == "bearish":
            tags.append("macd_crossover_bearish")
            return max_w, tags
        # Partial score: histogram momentum agrees with direction
        if signal.macd_histogram is not None:
            if (direction == "BUY" and signal.macd_histogram > 0) or (
                direction == "SELL" and signal.macd_histogram < 0
            ):
                return max_w * 0.4, tags
        return 0.0, tags

    def _score_candle_pattern(
        self, signal: TechnicalSignal, direction: str
    ) -> tuple[float, List[str]]:
        max_w = self._weights.get(
            "candle_pattern_at_level", DEFAULT_WEIGHTS["candle_pattern_at_level"]
        )
        tags: List[str] = []
        bullish_patterns = {
            "bullish_engulfing",
            "hammer",
            "morning_star",
            "piercing_line",
            "bullish_harami",
            "dragonfly_doji",
        }
        bearish_patterns = {
            "bearish_engulfing",
            "shooting_star",
            "evening_star",
            "dark_cloud_cover",
            "bearish_harami",
            "gravestone_doji",
        }
        neutral_patterns = {"inside_bar", "doji", "pin_bar"}

        found_patterns: List[str] = []
        for pattern in signal.candle_patterns:
            p = pattern.lower().replace(" ", "_")
            if direction == "BUY" and p in bullish_patterns:
                found_patterns.append(p)
            elif direction == "SELL" and p in bearish_patterns:
                found_patterns.append(p)
            elif p in neutral_patterns:
                found_patterns.append(p)

        if not found_patterns:
            return 0.0, tags

        # Check if candle is near a key S/R level
        near_level = _is_price_near_sr(
            signal.symbol,
            signal.ema_20 or 0.0,  # current price approximation
            signal.support_levels,
            signal.resistance_levels,
            signal.atr,
        )
        for p in found_patterns:
            tag = f"{'bullish' if direction == 'BUY' else 'bearish'}_{p}"
            if near_level:
                tag += f"_{signal.timeframe.lower()}"
            tags.append(tag)

        score = max_w if near_level else max_w * 0.5
        return score, tags

    def _score_price_at_sr(
        self, signal: TechnicalSignal, direction: str
    ) -> tuple[float, List[str]]:
        max_w = self._weights.get("price_at_sr", DEFAULT_WEIGHTS["price_at_sr"])
        tags: List[str] = []
        # Use EMA20 as a proxy for current price (accurate enough for scoring)
        price = signal.ema_20
        if price is None:
            return 0.0, tags

        near = _is_price_near_sr(
            signal.symbol,
            price,
            signal.support_levels,
            signal.resistance_levels,
            signal.atr,
        )
        if near:
            level_type = "support" if direction == "BUY" else "resistance"
            tags.append(f"price_at_key_{level_type}")
            return max_w, tags
        return 0.0, tags

    def _score_bollinger(
        self, signal: TechnicalSignal, direction: str
    ) -> tuple[float, List[str]]:
        max_w = self._weights.get("bollinger_signal", DEFAULT_WEIGHTS["bollinger_signal"])
        tags: List[str] = []
        price = signal.ema_20
        if price is None or signal.bb_upper is None or signal.bb_lower is None:
            return 0.0, tags

        if signal.bb_squeeze:
            tags.append("bb_squeeze_detected")
            return max_w * 0.5, tags

        if direction == "BUY" and price <= signal.bb_lower:
            tags.append("price_at_bb_lower_band")
            return max_w, tags
        if direction == "SELL" and price >= signal.bb_upper:
            tags.append("price_at_bb_upper_band")
            return max_w, tags
        return 0.0, tags

    def _score_sentiment(
        self,
        signal: TechnicalSignal,
        direction: str,
        sentiment: Optional[SentimentSignal],
    ) -> tuple[float, List[str]]:
        """Score news sentiment contribution for the given direction."""
        max_w = self._weights.get("news_sentiment", DEFAULT_WEIGHTS["news_sentiment"])
        tags: List[str] = []
        if sentiment is None:
            sentiment = self._sentiment_cache.get(signal.symbol)
        if sentiment is None:
            return 0.0, tags

        # Use the canonical direction field; fall back to legacy sentiment field
        sent_direction = sentiment.direction if sentiment.direction != "neutral" else sentiment.sentiment

        aligned = (direction == "BUY" and sent_direction == "bullish") or (
            direction == "SELL" and sent_direction == "bearish"
        )
        if not aligned:
            return 0.0, tags

        # If score_contribution is pre-computed, honour it (capped at max weight)
        if sentiment.score_contribution > 0.0:
            score = min(sentiment.score_contribution, max_w)
        else:
            score = max_w * min(sentiment.confidence, 1.0)

        if score > 0:
            tags.append(f"news_sentiment_{sent_direction}")
        return score, tags

    def _score_volatility(
        self,
        signal: TechnicalSignal,
        volatility: Optional[VolatilitySignal],
    ) -> tuple[float, List[str]]:
        """
        Score the volatility context contribution.

        Scoring rules
        -------------
        * If a pre-computed ``score_contribution`` is present on the signal,
          use it directly (capped at ``max_w``).
        * Favourable context (no spike, ATR ratio 0.8–1.4): full score.
        * Active spike detected: zero — the market is too noisy for a clean
          entry (the spike monitor handles its own standalone alert).
        """
        max_w = self._weights.get(
            "volatility_context", DEFAULT_WEIGHTS["volatility_context"]
        )
        tags: List[str] = []
        if volatility is None:
            volatility = self._volatility_cache.get(signal.symbol)
        if volatility is None:
            return 0.0, tags

        # If the VolatilityMonitor pre-computed a contribution, honour it
        if volatility.score_contribution > 0.0 and not volatility.spike_detected:
            score = min(volatility.score_contribution, max_w)
            tags.append("volatility_context_scored")
            return score, tags

        # Favourable context: moderate volatility (not a spike)
        if not volatility.spike_detected and volatility.atr_ratio is not None:
            if 0.8 <= volatility.atr_ratio <= 1.4:
                tags.append("volatility_context_normal")
                return max_w, tags

        return 0.0, tags

    # ------------------------------------------------------------------
    # Entry / SL / TP computation
    # ------------------------------------------------------------------

    def _compute_trade_levels(
        self,
        signal: TechnicalSignal,
        direction: str,
    ) -> tuple[List[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Calculate entry zone, stop loss, take profit, R:R, sl_pips, tp_pips.

        Returns:
            (entry_zone, stop_loss, take_profit, rr_ratio, sl_pips, tp_pips)
        """
        price = signal.ema_20
        atr = signal.atr

        if price is None or atr is None or atr == 0:
            return [], None, None, None, None, None

        pip = _pip_value(signal.symbol)

        if direction == "BUY":
            entry_low = round(price - atr * 0.3, 5)
            entry_high = round(price, 5)
            entry_zone = [entry_low, entry_high]

            # SL: below nearest support or 1.5× ATR
            sl = _nearest_support_below(price, signal.support_levels)
            if sl is None or (price - sl) > atr * 2:
                sl = round(price - atr * 1.5, 5)
            else:
                sl = round(sl - atr * 0.2, 5)  # small buffer below support

            # TP: above nearest resistance or 2× ATR minimum
            tp = _nearest_resistance_above(price, signal.resistance_levels)
            min_tp = round(price + atr * 2.0, 5)
            if tp is None or tp < min_tp:
                tp = min_tp
            else:
                tp = round(tp, 5)
        else:  # SELL
            entry_low = round(price, 5)
            entry_high = round(price + atr * 0.3, 5)
            entry_zone = [entry_low, entry_high]

            sl = _nearest_resistance_above(price, signal.resistance_levels)
            if sl is None or (sl - price) > atr * 2:
                sl = round(price + atr * 1.5, 5)
            else:
                sl = round(sl + atr * 0.2, 5)

            tp = _nearest_support_below(price, signal.support_levels)
            min_tp = round(price - atr * 2.0, 5)
            if tp is None or tp > min_tp:
                tp = min_tp
            else:
                tp = round(tp, 5)

        sl_dist = abs(price - sl)
        tp_dist = abs(tp - price)
        rr = round(tp_dist / sl_dist, 2) if sl_dist > 0 else None
        sl_pips_val = round(_pips(sl_dist, signal.symbol), 1)
        tp_pips_val = round(_pips(tp_dist, signal.symbol), 1)

        return entry_zone, sl, tp, rr, sl_pips_val, tp_pips_val

    # ------------------------------------------------------------------
    # Invalidation string
    # ------------------------------------------------------------------

    @staticmethod
    def _build_invalidation(
        signal: TechnicalSignal,
        direction: str,
        stop_loss: Optional[float],
    ) -> str:
        tf_above = _next_higher_timeframe(signal.timeframe)
        if stop_loss is not None:
            move = "closes below" if direction == "BUY" else "closes above"
            return f"{tf_above} candle {move} {stop_loss}"
        return f"{tf_above} structure reversal"

    # ------------------------------------------------------------------
    # Evaluation orchestration
    # ------------------------------------------------------------------

    def _evaluate(self, signal: TechnicalSignal) -> Optional[TradeSignal]:
        direction = _determine_direction(signal)
        base_score, confluences = self.compute_score(signal)
        final_score, htf_tags = self.apply_htf_bonus(base_score, signal, direction)
        confluences = confluences + htf_tags

        if final_score < self._watch_threshold:
            logger.debug(
                "Silent setup %s %s — score=%.2f", signal.symbol, signal.timeframe, final_score
            )
            return None

        sig_type = "SIGNAL" if final_score >= self._signal_threshold else "WATCH"

        entry_zone: List[float] = []
        sl = tp = rr = sl_pips = tp_pips = None
        if sig_type == "SIGNAL":
            entry_zone, sl, tp, rr, sl_pips, tp_pips = self._compute_trade_levels(
                signal, direction
            )

        invalidation = self._build_invalidation(signal, direction, sl)

        now_ts = _utc_now()
        try:
            signal_ts = datetime.fromisoformat(signal.timestamp.replace("Z", "+00:00"))
            age = int((datetime.now(tz=timezone.utc) - signal_ts).total_seconds())
        except Exception:
            age = 0

        trade_signal = TradeSignal(
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            timestamp=now_ts,
            signal_type=sig_type,
            direction=direction,
            score=round(final_score, 2),
            entry_zone=entry_zone,
            stop_loss=sl,
            take_profit=tp,
            rr_ratio=rr,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            confluences=confluences,
            invalidation=invalidation,
            signal_age_seconds=age,
        )

        logger.info(
            "%s %s %s — score=%.2f  confluences=%s",
            sig_type,
            signal.symbol,
            direction,
            final_score,
            confluences,
        )

        if self._callback is not None:
            self._callback(trade_signal)

        return trade_signal


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _determine_direction(signal: TechnicalSignal) -> str:
    """
    Determine BUY or SELL bias from the technical signal.

    Priority: EMA alignment → MACD crossover → RSI condition.
    Defaults to BUY when ambiguous (caller should check final score).
    """
    if signal.ema_alignment == "bullish":
        return "BUY"
    if signal.ema_alignment == "bearish":
        return "SELL"
    if signal.macd_crossover == "bullish":
        return "BUY"
    if signal.macd_crossover == "bearish":
        return "SELL"
    if signal.rsi_oversold:
        return "BUY"
    if signal.rsi_overbought:
        return "SELL"
    return "BUY"


def _is_price_near_sr(
    symbol: str,
    price: float,
    support_levels: List[float],
    resistance_levels: List[float],
    atr: Optional[float],
) -> bool:
    """Return True if *price* is within ATR of any support or resistance level."""
    if not atr or atr == 0:
        threshold = price * 0.002  # 0.2% as fallback
    else:
        threshold = atr
    all_levels = list(support_levels) + list(resistance_levels)
    return any(abs(price - lvl) <= threshold for lvl in all_levels)


def _nearest_support_below(price: float, levels: List[float]) -> Optional[float]:
    candidates = [lvl for lvl in levels if lvl < price]
    return max(candidates) if candidates else None


def _nearest_resistance_above(price: float, levels: List[float]) -> Optional[float]:
    candidates = [lvl for lvl in levels if lvl > price]
    return min(candidates) if candidates else None


def _next_higher_timeframe(tf: str) -> str:
    order = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
    try:
        idx = order.index(tf)
        return order[min(idx + 1, len(order) - 1)]
    except ValueError:
        return tf


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
