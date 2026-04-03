"""
SentimentEngine for FX-Leopard.

Continuously fetches live financial news, uses OpenAI GPT (gpt-4o-mini) to
parse sentiment per currency / instrument, and feeds scored
:class:`~analysis.models.SentimentSignal` objects into the confluence engine
via callback.

Polling is driven by APScheduler (configurable interval, default 2 minutes).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from apscheduler.schedulers.background import BackgroundScheduler
from openai import APIError, OpenAI

from analysis.models import SentimentSignal
from data.news_feed import NewsFeed, NewsHeadline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Symbol / currency constants
# ---------------------------------------------------------------------------

SUPPORTED_SYMBOLS: List[str] = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "XAUUSD", "XAGUSD", "USOIL", "UKOIL",
]

# (base_currency, quote_currency) for each supported pair.
# Commodities that have no FX quote currency use None.
PAIR_CURRENCIES: Dict[str, Tuple[str, Optional[str]]] = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"),
    "USDCHF": ("USD", "CHF"),
    "AUDUSD": ("AUD", "USD"),
    "USDCAD": ("USD", "CAD"),
    "NZDUSD": ("NZD", "USD"),
    "XAUUSD": ("XAU", "USD"),
    "XAGUSD": ("XAG", "USD"),
    "USOIL": ("OIL", None),
    "UKOIL": ("OIL", None),
}

# ---------------------------------------------------------------------------
# GPT prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a professional FX market analyst. Analyze the following financial "
    "news headlines and return a JSON object assessing the impact on FX pairs "
    "and commodities.\n\n"
    "For each headline batch:\n"
    "- Identify which currencies and instruments are affected\n"
    "- Rate sentiment direction (bullish/bearish/neutral) per currency\n"
    "- Rate strength (0.0-1.0) and confidence (0.0-1.0)\n"
    "- Classify overall impact: high | medium | low\n"
    "- Write a 1-sentence summary of the key catalyst\n"
    "- List affected FX symbols from: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, "
    "USDCAD, NZDUSD, XAUUSD, XAGUSD, USOIL, UKOIL\n\n"
    "Return ONLY valid JSON. No explanation outside the JSON.\n\n"
    "Example output:\n"
    '{"affected_symbols": ["EURUSD", "XAUUSD"], '
    '"sentiment": {'
    '"USD": {"direction": "bearish", "strength": 0.75, "confidence": 0.85}, '
    '"EUR": {"direction": "bullish", "strength": 0.60, "confidence": 0.80}'
    "}, "
    '"impact": "high", '
    '"summary": "Weak US ADP data signals soft NFP, USD broadly sold off", '
    '"catalyst": "ADP Employment Change missed forecast by 45K"}'
)

# ---------------------------------------------------------------------------
# SentimentEngine
# ---------------------------------------------------------------------------


class SentimentEngine:
    """
    News intelligence layer for FX-Leopard.

    Fetches live financial news from NewsAPI and RSS feeds, batches headlines
    to GPT-4o-mini for structured sentiment scoring, maps currency-level
    sentiment to FX pair directional bias, and emits
    :class:`~analysis.models.SentimentSignal` objects via the provided
    ``signal_callback``.

    Usage::

        engine = SentimentEngine(
            signal_callback=confluence_engine.update_sentiment,
            config=loaded_yaml_dict,
        )
        engine.start()   # begins background polling

    ``signal_callback`` receives one :class:`SentimentSignal` per affected
    symbol each poll cycle.
    """

    def __init__(
        self,
        signal_callback: Optional[Callable[[SentimentSignal], None]] = None,
        config: Optional[dict] = None,
    ) -> None:
        cfg = config or {}
        self._callback = signal_callback

        # --- LLM config ---
        llm_cfg = cfg.get("llm", {})
        api_keys = cfg.get("api_keys", {})
        openai_key: str = llm_cfg.get("api_key", "") or api_keys.get("openai", "")
        self._model: str = llm_cfg.get("model", "gpt-4o-mini")
        self._max_tokens: int = int(llm_cfg.get("max_tokens", 500))
        self._temperature: float = float(llm_cfg.get("temperature", 0.1))

        # --- News polling config ---
        news_cfg = cfg.get("news", {})
        self._poll_interval: int = int(news_cfg.get("poll_interval_seconds", 120))
        self._max_headlines: int = int(news_cfg.get("max_headlines_per_batch", 20))

        # --- OpenAI client ---
        self._client: Optional[OpenAI] = OpenAI(api_key=openai_key) if openai_key else None

        # --- News feed ---
        self._feed = NewsFeed(config=cfg)

        # --- Scheduler ---
        self._scheduler = BackgroundScheduler()
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background news polling and run an initial analysis immediately."""
        if self._running:
            return
        self._running = True
        self._scheduler.add_job(
            self._poll_and_analyze,
            "interval",
            seconds=self._poll_interval,
            id="sentiment_poll",
        )
        self._scheduler.start()
        logger.info("SentimentEngine started (poll_interval=%ds)", self._poll_interval)
        self._poll_and_analyze()

    def stop(self) -> None:
        """Stop the background polling scheduler."""
        if self._running and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        self._running = False
        logger.info("SentimentEngine stopped")

    # ------------------------------------------------------------------
    # Public analysis API
    # ------------------------------------------------------------------

    def analyze_headlines(self, headlines: List[NewsHeadline]) -> List[SentimentSignal]:
        """
        Send *headlines* to GPT and return a :class:`SentimentSignal` per
        affected FX symbol.

        This method is intentionally public so it can be called directly in
        tests with mocked headline lists — no scheduler required.
        """
        if not headlines:
            return []
        if self._client is None:
            logger.warning("No OpenAI API key configured — skipping sentiment analysis")
            return []

        headline_text = "\n".join(
            f"{i + 1}. {h.title} ({h.source})"
            for i, h in enumerate(headlines[: self._max_headlines])
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Headlines:\n{headline_text}"},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            return self._parse_gpt_response(content, headlines)
        except APIError as exc:
            logger.error("OpenAI API error during sentiment analysis: %s", exc)
            return []
        except Exception as exc:
            logger.error("Unexpected error during sentiment analysis: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _poll_and_analyze(self) -> None:
        """Fetch new headlines and run GPT sentiment analysis."""
        try:
            headlines = self._feed.fetch_headlines()
            if not headlines:
                logger.debug("No new headlines to analyze")
                return
            logger.info("Analyzing %d new headlines for sentiment", len(headlines))
            signals = self.analyze_headlines(headlines)
            for signal in signals:
                logger.info(
                    "SentimentSignal: %s %s strength=%.2f confidence=%.2f impact=%s",
                    signal.symbol,
                    signal.direction,
                    signal.strength,
                    signal.confidence,
                    signal.impact,
                )
                if self._callback is not None:
                    self._callback(signal)
        except Exception as exc:
            logger.error("Error during sentiment poll cycle: %s", exc)

    def _parse_gpt_response(
        self,
        content: str,
        headlines: List[NewsHeadline],
    ) -> List[SentimentSignal]:
        """Parse a GPT JSON response into :class:`SentimentSignal` objects."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse GPT response as JSON: %s\nContent: %.500s", exc, content
            )
            return []

        affected_symbols: List[str] = data.get("affected_symbols") or []
        sentiment_map: Dict[str, dict] = data.get("sentiment") or {}
        impact: str = data.get("impact") or "low"
        summary: str = data.get("summary") or ""
        headline_titles = [h.title for h in headlines[: self._max_headlines]]
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        signals: List[SentimentSignal] = []
        for symbol in affected_symbols:
            if symbol not in SUPPORTED_SYMBOLS:
                logger.debug("Ignoring unsupported symbol from GPT: %s", symbol)
                continue

            direction, strength, confidence = self._resolve_pair_sentiment(
                symbol, sentiment_map
            )

            # Skip neutral / near-zero signals
            if direction == "neutral" and strength < 0.05:
                continue

            score_contribution = _compute_score_contribution(
                direction, strength, confidence, impact
            )

            signal = SentimentSignal(
                symbol=symbol,
                timestamp=timestamp,
                direction=direction,
                sentiment=direction,    # keep backward-compat alias in sync
                strength=strength,
                confidence=confidence,
                impact=impact,
                summary=summary,
                headlines=headline_titles[:5],
                score_contribution=score_contribution,
            )
            signals.append(signal)

        return signals

    def _resolve_pair_sentiment(
        self,
        symbol: str,
        sentiment_map: Dict[str, dict],
    ) -> Tuple[str, float, float]:
        """
        Derive the net direction / strength / confidence for an FX pair.

        If the GPT response contains a direct entry for the symbol (e.g.
        ``"XAUUSD"``), that is used as-is.  Otherwise the base and quote
        currency sentiments are combined: a bullish base OR bearish quote
        yields a bullish pair bias and vice-versa.
        """
        # Direct symbol-level override from GPT
        if symbol in sentiment_map:
            s = sentiment_map[symbol]
            return (
                str(s.get("direction") or "neutral"),
                float(s.get("strength") or 0.0),
                float(s.get("confidence") or 0.0),
            )

        currencies = PAIR_CURRENCIES.get(symbol)
        if not currencies:
            return "neutral", 0.0, 0.0

        base_ccy, quote_ccy = currencies

        base_sent = sentiment_map.get(base_ccy) or {}
        base_dir = str(base_sent.get("direction") or "neutral")
        base_strength = float(base_sent.get("strength") or 0.0)
        base_conf = float(base_sent.get("confidence") or 0.0)

        if quote_ccy is None:
            # Commodity with no FX quote (USOIL, UKOIL) — use base directly
            return base_dir, base_strength, base_conf

        quote_sent = sentiment_map.get(quote_ccy) or {}
        quote_dir = str(quote_sent.get("direction") or "neutral")
        quote_strength = float(quote_sent.get("strength") or 0.0)
        quote_conf = float(quote_sent.get("confidence") or 0.0)

        # Net score: base bullish raises pair; quote bullish lowers pair
        net = _dir_score(base_dir) * base_strength - _dir_score(quote_dir) * quote_strength

        if abs(net) < 0.05:
            direction = "neutral"
        elif net > 0:
            direction = "bullish"
        else:
            direction = "bearish"

        confs = [c for c in (base_conf, quote_conf) if c > 0.0]
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        return direction, min(abs(net), 1.0), avg_conf


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _dir_score(direction: str) -> float:
    """Map a direction string to a signed numeric score."""
    if direction == "bullish":
        return 1.0
    if direction == "bearish":
        return -1.0
    return 0.0


def _compute_score_contribution(
    direction: str,
    strength: float,
    confidence: float,
    impact: str,
) -> float:
    """
    Compute the pre-scaled score contribution for the confluence engine.

    Maximum possible value is 1.5 (matching the ``news_sentiment`` weight).
    """
    if direction == "neutral":
        return 0.0
    impact_mult = {"high": 1.0, "medium": 0.7, "low": 0.4}.get(impact, 0.5)
    raw = 1.5 * strength * confidence * impact_mult
    return round(min(raw, 1.5), 3)
