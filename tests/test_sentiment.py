"""
Unit tests for the News Fetch + GPT Sentiment Engine (Issue #4).

All external API calls (NewsAPI, RSS feeds, OpenAI) are mocked — no real
API keys are required to run these tests.
"""

from __future__ import annotations

import json
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from analysis.models import SentimentSignal
from analysis.sentiment import (
    SentimentEngine,
    _compute_score_contribution,
    _dir_score,
    SUPPORTED_SYMBOLS,
    PAIR_CURRENCIES,
)
from data.news_feed import NewsFeed, NewsHeadline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_headline(
    title: str = "Fed signals slower rate cuts",
    source: str = "Reuters",
    published_at: str = "2026-04-02T13:30:00Z",
    url: str = "https://example.com/article",
) -> NewsHeadline:
    return NewsHeadline(title=title, source=source, published_at=published_at, url=url)


def _gpt_response(
    affected_symbols: List[str],
    sentiment: dict,
    impact: str = "high",
    summary: str = "Test summary",
) -> str:
    """Build a valid GPT JSON response string."""
    return json.dumps(
        {
            "affected_symbols": affected_symbols,
            "sentiment": sentiment,
            "impact": impact,
            "summary": summary,
            "catalyst": "Test catalyst",
        }
    )


# ---------------------------------------------------------------------------
# NewsFeed — deduplication
# ---------------------------------------------------------------------------


class TestNewsFeedDeduplication:
    def setup_method(self):
        self.feed = NewsFeed(config={})

    def test_duplicate_headlines_are_filtered(self):
        """The same headline title should only appear once across two calls."""
        headlines_batch_1 = [
            _make_headline("Fed holds rates steady"),
            _make_headline("Dollar drops on weak data"),
        ]
        headlines_batch_2 = [
            _make_headline("Fed holds rates steady"),   # duplicate
            _make_headline("New headline appears"),     # new
        ]

        # Manually register first batch as seen
        for h in headlines_batch_1:
            self.feed._seen_hashes.add(h.hash)

        # Simulate second fetch — only new headlines should pass
        result: List[NewsHeadline] = []
        for h in headlines_batch_2:
            if h.hash not in self.feed._seen_hashes:
                self.feed._seen_hashes.add(h.hash)
                result.append(h)

        assert len(result) == 1
        assert result[0].title == "New headline appears"

    def test_headline_hash_is_deterministic(self):
        """The same title always produces the same hash."""
        h1 = _make_headline("Some headline")
        h2 = _make_headline("Some headline")
        assert h1.hash == h2.hash

    def test_different_titles_produce_different_hashes(self):
        h1 = _make_headline("Headline A")
        h2 = _make_headline("Headline B")
        assert h1.hash != h2.hash

    def test_clear_seen_cache_resets_dedup(self):
        h = _make_headline("Recurring headline")
        self.feed._seen_hashes.add(h.hash)
        assert h.hash in self.feed._seen_hashes
        self.feed.clear_seen_cache()
        assert h.hash not in self.feed._seen_hashes

    @patch("data.news_feed.requests.get")
    def test_newsapi_fetch_returns_headlines(self, mock_get):
        """NewsAPI response is parsed into NewsHeadline objects."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = MagicMock()
        mock_get.return_value.json.return_value = {
            "articles": [
                {
                    "title": "Fed signals slower rate cuts",
                    "source": {"name": "Reuters"},
                    "publishedAt": "2026-04-02T13:30:00Z",
                    "url": "https://reuters.com/article1",
                },
                {
                    "title": "[Removed]",  # should be filtered out
                    "source": {"name": "NewsAPI"},
                    "publishedAt": "",
                    "url": "",
                },
            ]
        }
        feed = NewsFeed(config={"api_keys": {"newsapi": "test-key"}, "news": {"sources": ["newsapi"]}})
        headlines = feed._fetch_newsapi()
        assert len(headlines) == 1
        assert headlines[0].title == "Fed signals slower rate cuts"
        assert headlines[0].source == "Reuters"

    @patch("data.news_feed.requests.get")
    def test_newsapi_failure_returns_empty_list(self, mock_get):
        """A network error during NewsAPI fetch returns an empty list (no crash)."""
        mock_get.side_effect = Exception("Connection timeout")
        feed = NewsFeed(config={"api_keys": {"newsapi": "test-key"}, "news": {"sources": ["newsapi"]}})
        headlines = feed._fetch_newsapi()
        assert headlines == []

    @patch("data.news_feed.feedparser.parse")
    def test_rss_fetch_returns_headlines(self, mock_parse):
        """RSS feed entries are parsed into NewsHeadline objects."""
        mock_entry = MagicMock()
        mock_entry.get = lambda key, default="": {
            "title": "Dollar weakens on jobs data miss",
            "link": "https://reuters.com/rss/1",
        }.get(key, default)
        mock_entry.published_parsed = None
        mock_entry.published = "Thu, 02 Apr 2026 13:30:00 +0000"

        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]
        mock_parse.return_value = mock_feed

        feed = NewsFeed()
        headlines = feed._fetch_rss("https://example.com/rss", "test_rss")
        assert len(headlines) == 1
        assert headlines[0].title == "Dollar weakens on jobs data miss"
        assert headlines[0].source == "test_rss"

    @patch("data.news_feed.feedparser.parse")
    def test_rss_failure_returns_empty_list(self, mock_parse):
        mock_parse.side_effect = Exception("Parse error")
        feed = NewsFeed()
        headlines = feed._fetch_rss("https://broken.com/rss", "broken")
        assert headlines == []

    def test_max_headlines_per_batch_respected(self):
        """fetch_headlines returns at most max_headlines_per_batch items."""
        feed = NewsFeed(config={"news": {"max_headlines_per_batch": 3}})
        # Inject 10 unique headlines via mock
        with patch.object(feed, "_fetch_rss") as mock_rss:
            mock_rss.return_value = [
                _make_headline(f"Headline {i}") for i in range(10)
            ]
            feed._sources = ["reuters_rss"]
            results = feed.fetch_headlines()
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# SentimentEngine — GPT response parsing
# ---------------------------------------------------------------------------


class TestGPTResponseParsing:
    def setup_method(self):
        self.engine = SentimentEngine(config={})

    def test_parse_valid_response(self):
        """A valid GPT response is parsed into SentimentSignal objects."""
        gpt_json = _gpt_response(
            affected_symbols=["EURUSD"],
            sentiment={
                "USD": {"direction": "bearish", "strength": 0.75, "confidence": 0.85},
                "EUR": {"direction": "bullish", "strength": 0.60, "confidence": 0.80},
            },
        )
        headlines = [_make_headline("ADP misses by 45K")]
        signals = self.engine._parse_gpt_response(gpt_json, headlines)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.symbol == "EURUSD"
        assert sig.direction == "bullish"
        assert sig.sentiment == "bullish"       # backward-compat alias
        assert 0.0 < sig.strength <= 1.0
        assert 0.0 < sig.confidence <= 1.0
        assert sig.impact == "high"
        assert sig.score_contribution > 0.0
        assert "ADP misses by 45K" in sig.headlines

    def test_parse_invalid_json_returns_empty(self):
        """Malformed JSON from GPT returns an empty list (no crash)."""
        headlines = [_make_headline()]
        signals = self.engine._parse_gpt_response("not valid json {{{", headlines)
        assert signals == []

    def test_unsupported_symbol_is_ignored(self):
        """Symbols not in SUPPORTED_SYMBOLS are silently dropped."""
        gpt_json = _gpt_response(
            affected_symbols=["EURUSD", "FAKEPAIR"],
            sentiment={"EUR": {"direction": "bullish", "strength": 0.7, "confidence": 0.8}},
        )
        signals = self.engine._parse_gpt_response(gpt_json, [_make_headline()])
        symbols = [s.symbol for s in signals]
        assert "FAKEPAIR" not in symbols

    def test_neutral_low_strength_signal_is_filtered(self):
        """Neutral signals with near-zero strength are not emitted."""
        gpt_json = json.dumps(
            {
                "affected_symbols": ["EURUSD"],
                "sentiment": {
                    "EUR": {"direction": "neutral", "strength": 0.01, "confidence": 0.5},
                    "USD": {"direction": "neutral", "strength": 0.01, "confidence": 0.5},
                },
                "impact": "low",
                "summary": "Nothing happened",
                "catalyst": "",
            }
        )
        signals = self.engine._parse_gpt_response(gpt_json, [_make_headline()])
        assert signals == []

    def test_signal_fields_populated(self):
        """All SentimentSignal fields are populated from GPT response."""
        gpt_json = _gpt_response(
            affected_symbols=["GBPUSD"],
            sentiment={
                "GBP": {"direction": "bullish", "strength": 0.8, "confidence": 0.9},
                "USD": {"direction": "bearish", "strength": 0.6, "confidence": 0.7},
            },
            impact="high",
            summary="Strong UK PMI beats expectations",
        )
        signals = self.engine._parse_gpt_response(gpt_json, [_make_headline("UK PMI beats")])
        assert len(signals) == 1
        sig = signals[0]
        assert sig.summary == "Strong UK PMI beats expectations"
        assert sig.impact == "high"
        assert sig.timestamp != ""
        assert len(sig.headlines) > 0


# ---------------------------------------------------------------------------
# Currency-to-pair mapping
# ---------------------------------------------------------------------------


class TestCurrencyToPairMapping:
    def setup_method(self):
        self.engine = SentimentEngine(config={})

    def test_usd_bearish_eur_bullish_gives_eurusd_bullish(self):
        """Classic USD selloff + EUR bid = EURUSD bullish."""
        sentiment_map = {
            "USD": {"direction": "bearish", "strength": 0.75, "confidence": 0.85},
            "EUR": {"direction": "bullish", "strength": 0.60, "confidence": 0.80},
        }
        direction, strength, confidence = self.engine._resolve_pair_sentiment(
            "EURUSD", sentiment_map
        )
        assert direction == "bullish"
        assert strength > 0.0
        assert confidence > 0.0

    def test_usd_bullish_eur_bearish_gives_eurusd_bearish(self):
        sentiment_map = {
            "USD": {"direction": "bullish", "strength": 0.70, "confidence": 0.80},
            "EUR": {"direction": "bearish", "strength": 0.50, "confidence": 0.75},
        }
        direction, _, _ = self.engine._resolve_pair_sentiment("EURUSD", sentiment_map)
        assert direction == "bearish"

    def test_usd_bullish_gives_usdjpy_bullish(self):
        """USD/JPY pair: USD is base, so USD bullish = pair bullish."""
        sentiment_map = {
            "USD": {"direction": "bullish", "strength": 0.8, "confidence": 0.9},
        }
        direction, _, _ = self.engine._resolve_pair_sentiment("USDJPY", sentiment_map)
        assert direction == "bullish"

    def test_gbp_bullish_usd_neutral_gives_gbpusd_bullish(self):
        sentiment_map = {
            "GBP": {"direction": "bullish", "strength": 0.7, "confidence": 0.8},
        }
        direction, _, _ = self.engine._resolve_pair_sentiment("GBPUSD", sentiment_map)
        assert direction == "bullish"

    def test_xauusd_direct_sentiment_used_when_provided(self):
        """If GPT provides XAUUSD directly, that takes precedence."""
        sentiment_map = {
            "XAUUSD": {"direction": "bullish", "strength": 0.9, "confidence": 0.95},
            "USD": {"direction": "bullish", "strength": 0.8, "confidence": 0.9},
        }
        direction, strength, confidence = self.engine._resolve_pair_sentiment(
            "XAUUSD", sentiment_map
        )
        assert direction == "bullish"
        assert strength == pytest.approx(0.9)

    def test_oil_no_quote_currency_uses_base_directly(self):
        """USOIL has no quote currency — OIL sentiment is used directly."""
        sentiment_map = {
            "OIL": {"direction": "bearish", "strength": 0.6, "confidence": 0.7},
        }
        direction, strength, confidence = self.engine._resolve_pair_sentiment(
            "USOIL", sentiment_map
        )
        assert direction == "bearish"
        assert strength == pytest.approx(0.6)

    def test_both_currencies_neutral_gives_neutral(self):
        sentiment_map = {
            "EUR": {"direction": "neutral", "strength": 0.0, "confidence": 0.5},
            "USD": {"direction": "neutral", "strength": 0.0, "confidence": 0.5},
        }
        direction, strength, _ = self.engine._resolve_pair_sentiment("EURUSD", sentiment_map)
        assert direction == "neutral"

    def test_unknown_symbol_returns_neutral(self):
        direction, strength, confidence = self.engine._resolve_pair_sentiment(
            "UNKNOWN", {}
        )
        assert direction == "neutral"
        assert strength == 0.0


# ---------------------------------------------------------------------------
# Score contribution calculation
# ---------------------------------------------------------------------------


class TestScoreContribution:
    def test_high_impact_bullish_full_strength(self):
        score = _compute_score_contribution("bullish", 1.0, 1.0, "high")
        assert score == pytest.approx(1.5, abs=0.01)

    def test_medium_impact_reduces_score(self):
        score_high = _compute_score_contribution("bullish", 1.0, 1.0, "high")
        score_medium = _compute_score_contribution("bullish", 1.0, 1.0, "medium")
        assert score_medium < score_high

    def test_low_impact_smallest_score(self):
        score_low = _compute_score_contribution("bullish", 1.0, 1.0, "low")
        score_med = _compute_score_contribution("bullish", 1.0, 1.0, "medium")
        assert score_low < score_med

    def test_neutral_direction_gives_zero(self):
        score = _compute_score_contribution("neutral", 1.0, 1.0, "high")
        assert score == 0.0

    def test_score_capped_at_1_5(self):
        # Even if values somehow exceed bounds the result should not exceed 1.5
        score = _compute_score_contribution("bullish", 2.0, 2.0, "high")
        assert score <= 1.5

    def test_zero_confidence_gives_zero(self):
        score = _compute_score_contribution("bullish", 1.0, 0.0, "high")
        assert score == 0.0

    def test_zero_strength_gives_zero(self):
        score = _compute_score_contribution("bullish", 0.0, 1.0, "high")
        assert score == 0.0


# ---------------------------------------------------------------------------
# GPT API error handling
# ---------------------------------------------------------------------------


class TestGPTErrorHandling:
    @patch("analysis.sentiment.OpenAI")
    def test_api_error_returns_empty_list(self, mock_openai_cls):
        """An OpenAI APIError is caught and returns an empty list."""
        from openai import APIError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        engine = SentimentEngine(config={"api_keys": {"openai": "test-key"}})
        mock_client.chat.completions.create.side_effect = APIError(
            message="rate limit",
            request=MagicMock(),
            body=None,
        )
        headlines = [_make_headline("Rate limit test")]
        signals = engine.analyze_headlines(headlines)
        assert signals == []

    @patch("analysis.sentiment.OpenAI")
    def test_unexpected_exception_returns_empty_list(self, mock_openai_cls):
        """Any unexpected exception is caught gracefully."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        engine = SentimentEngine(config={"api_keys": {"openai": "test-key"}})
        mock_client.chat.completions.create.side_effect = RuntimeError("unexpected")
        signals = engine.analyze_headlines([_make_headline()])
        assert signals == []

    def test_no_api_key_returns_empty_list(self):
        """When no OpenAI key is configured, analyze_headlines returns empty."""
        engine = SentimentEngine(config={})   # no key → _client is None
        signals = engine.analyze_headlines([_make_headline()])
        assert signals == []

    @patch("analysis.sentiment.OpenAI")
    def test_empty_headlines_list_returns_empty(self, mock_openai_cls):
        """No headlines → no GPT call → empty result."""
        mock_openai_cls.return_value = MagicMock()
        engine = SentimentEngine(config={"api_keys": {"openai": "test-key"}})
        signals = engine.analyze_headlines([])
        assert signals == []


# ---------------------------------------------------------------------------
# Confluence engine integration — update_sentiment / score_contribution
# ---------------------------------------------------------------------------


class TestConfluenceIntegration:
    """
    Verify that SentimentSignal.score_contribution flows correctly into the
    ConfluenceEngine scorer.
    """

    def test_update_sentiment_alias(self):
        """ConfluenceEngine.update_sentiment is an alias for on_sentiment_signal."""
        from analysis.confluence import ConfluenceEngine

        engine = ConfluenceEngine()
        signal = SentimentSignal(
            symbol="EURUSD",
            timestamp="2026-04-02T14:00:00Z",
            direction="bullish",
            sentiment="bullish",
            strength=0.8,
            confidence=0.9,
            impact="high",
            score_contribution=1.2,
        )
        # Both methods should cache the signal without error
        engine.update_sentiment(signal)
        assert engine._sentiment_cache.get("EURUSD") is signal

    def test_score_contribution_used_in_scoring(self):
        """
        When a SentimentSignal with a pre-computed score_contribution is cached,
        the confluence scorer uses that value (capped at the news_sentiment weight).
        """
        from analysis.confluence import ConfluenceEngine
        from analysis.models import TechnicalSignal

        config = {
            "scoring": {
                "weights": {
                    "ema_alignment": 0.0,
                    "rsi_condition": 0.0,
                    "macd_crossover": 0.0,
                    "candle_pattern_at_level": 0.0,
                    "price_at_sr": 0.0,
                    "bollinger_signal": 0.0,
                    "news_sentiment": 1.5,
                    "volatility_context": 0.0,
                }
            },
            "watch_threshold": 0.0,
            "confluence_threshold": 100.0,
        }
        engine = ConfluenceEngine(config=config)

        signal = SentimentSignal(
            symbol="EURUSD",
            timestamp="2026-04-02T14:00:00Z",
            direction="bullish",
            sentiment="bullish",
            strength=0.8,
            confidence=0.9,
            impact="high",
            score_contribution=1.5,     # max possible
        )
        engine.update_sentiment(signal)

        tech = TechnicalSignal(
            symbol="EURUSD",
            timeframe="H1",
            timestamp="2026-04-02T14:00:00Z",
            ema_alignment="bullish",
        )
        score, tags = engine.compute_score(tech)
        assert score == pytest.approx(10.0, abs=0.1)
        assert any("sentiment" in t for t in tags)

    def test_backward_compat_sentiment_field(self):
        """Old-style SentimentSignal (only sentiment= set) still works in scorer."""
        from analysis.confluence import ConfluenceEngine
        from analysis.models import TechnicalSignal

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
            "confluence_threshold": 100.0,
        }
        engine = ConfluenceEngine(config=config)
        # Create with legacy field only (direction stays "neutral")
        signal = SentimentSignal(
            symbol="EURUSD",
            timestamp="2026-04-02T14:00:00Z",
            sentiment="bullish",
            confidence=1.0,
            impact="high",
        )
        engine.on_sentiment_signal(signal)

        tech = TechnicalSignal(
            symbol="EURUSD",
            timeframe="H1",
            timestamp="2026-04-02T14:00:00Z",
            ema_alignment="bullish",
        )
        score, tags = engine.compute_score(tech)
        assert score == pytest.approx(10.0, abs=0.01)
        assert any("sentiment" in t for t in tags)


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_dir_score_bullish(self):
        assert _dir_score("bullish") == 1.0

    def test_dir_score_bearish(self):
        assert _dir_score("bearish") == -1.0

    def test_dir_score_neutral(self):
        assert _dir_score("neutral") == 0.0

    def test_dir_score_unknown(self):
        assert _dir_score("undefined") == 0.0
