"""
News feed fetcher for FX-Leopard.

Polls NewsAPI and RSS feeds, returning deduplicated :class:`NewsHeadline`
objects ready for GPT sentiment analysis.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import feedparser
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RSS feed URLs
# ---------------------------------------------------------------------------

RSS_FEED_URLS: Dict[str, str] = {
    "forexlive_rss": "https://www.forexlive.com/feed/news",
    "fxstreet_rss": "https://www.fxstreet.com/rss/news",
    "marketwatch_rss": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "ft_rss": "https://www.ft.com/rss/home/uk",
}

NEWSAPI_URL = "https://newsapi.org/v2/top-headlines"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class NewsHeadline:
    """A single news headline fetched from any source."""

    title: str
    source: str
    published_at: str   # ISO-8601 UTC string, e.g. "2026-04-02T13:30:00Z"
    url: str
    hash: str = field(default="")

    def __post_init__(self) -> None:
        if not self.hash:
            self.hash = hashlib.md5(self.title.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# NewsFeed
# ---------------------------------------------------------------------------


class NewsFeed:
    """
    Fetches and deduplicates financial news headlines from multiple sources.

    Sources (configurable via ``config.yaml``)::

        news:
          sources:
            - forexlive_rss
            - fxstreet_rss
            - marketwatch_rss
            # newsapi  # optional: enable if you have a paid plan

    Deduplication is based on the MD5 hash of the headline title so the same
    story is never returned twice across successive :meth:`fetch_headlines`
    calls.  Call :meth:`clear_seen_cache` to reset the deduplication window.

    NewsAPI daily-limit tracking prevents exceeding the free-tier cap of 100
    requests/day.  The counter resets automatically at UTC midnight.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        news_cfg = cfg.get("news", {})

        self._newsapi_key: str = cfg.get("api_keys", {}).get("newsapi", "")
        self._sources: List[str] = news_cfg.get(
            "sources", ["forexlive_rss", "fxstreet_rss"]
        )
        self._max_per_batch: int = int(news_cfg.get("max_headlines_per_batch", 20))
        self._newsapi_daily_limit: int = int(news_cfg.get("newsapi_daily_limit", 90))
        self._seen_hashes: Set[str] = set()

        # Daily request counter for NewsAPI (resets at UTC midnight)
        self._newsapi_request_count: int = 0
        self._newsapi_count_date: Optional[str] = None  # "YYYY-MM-DD"
        self._newsapi_skip_today: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_headlines(self) -> List[NewsHeadline]:
        """
        Fetch headlines from all configured sources and return those not
        seen before (deduplicated by headline hash).

        Returns at most ``max_headlines_per_batch`` new headlines.
        """
        raw: List[NewsHeadline] = []

        if "newsapi" in self._sources:
            raw.extend(self._fetch_newsapi())

        for source_key, feed_url in RSS_FEED_URLS.items():
            if source_key in self._sources:
                raw.extend(self._fetch_rss(feed_url, source_key))

        new_headlines: List[NewsHeadline] = []
        for headline in raw:
            if headline.hash not in self._seen_hashes:
                self._seen_hashes.add(headline.hash)
                new_headlines.append(headline)

        return new_headlines[: self._max_per_batch]

    def clear_seen_cache(self) -> None:
        """Reset the deduplication cache so all headlines are treated as new."""
        self._seen_hashes.clear()

    # ------------------------------------------------------------------
    # Source fetchers
    # ------------------------------------------------------------------

    def _reset_newsapi_counter_if_new_day(self) -> None:
        """Reset the daily counter and skip flag when the UTC date changes."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._newsapi_count_date != today:
            self._newsapi_request_count = 0
            self._newsapi_count_date = today
            self._newsapi_skip_today = False

    def _fetch_newsapi(self) -> List[NewsHeadline]:
        """Fetch English business headlines from NewsAPI."""
        if not self._newsapi_key:
            logger.debug("NewsAPI key not configured — skipping")
            return []

        self._reset_newsapi_counter_if_new_day()

        if self._newsapi_skip_today:
            logger.debug("NewsAPI daily limit reached — using RSS-only today")
            return []

        if self._newsapi_request_count >= self._newsapi_daily_limit:
            logger.warning(
                "NewsAPI daily limit reached (%d/%d) — switching to RSS-only for today",
                self._newsapi_request_count,
                self._newsapi_daily_limit,
            )
            self._newsapi_skip_today = True
            return []

        try:
            response = requests.get(
                NEWSAPI_URL,
                params={
                    "category": "business",
                    "language": "en",
                    "apiKey": self._newsapi_key,
                    "pageSize": self._max_per_batch,
                },
                timeout=10,
            )

            if response.status_code == 429:
                logger.warning(
                    "NewsAPI returned HTTP 429 (rate limit) — switching to RSS-only for today"
                )
                self._newsapi_skip_today = True
                return []

            response.raise_for_status()
            self._newsapi_request_count += 1

            data = response.json()
            headlines: List[NewsHeadline] = []
            for article in data.get("articles", []):
                title = (article.get("title") or "").strip()
                if not title or title == "[Removed]":
                    continue
                published = article.get("publishedAt") or ""
                source = (article.get("source") or {}).get("name") or "NewsAPI"
                url = article.get("url") or ""
                headlines.append(
                    NewsHeadline(
                        title=title,
                        source=source,
                        published_at=published,
                        url=url,
                    )
                )
            return headlines
        except Exception as exc:
            logger.warning("NewsAPI fetch failed: %s", exc)
            return []

    def _fetch_rss(self, feed_url: str, source_name: str) -> List[NewsHeadline]:
        """Parse an RSS feed and return its headlines."""
        try:
            feed = feedparser.parse(feed_url)
            headlines: List[NewsHeadline] = []
            for entry in feed.entries:
                title = (entry.get("title") or "").strip()
                if not title:
                    continue
                published = _parse_rss_date(entry)
                url = entry.get("link") or ""
                headlines.append(
                    NewsHeadline(
                        title=title,
                        source=source_name,
                        published_at=published,
                        url=url,
                    )
                )
            return headlines
        except Exception as exc:
            logger.warning("RSS fetch from %s failed: %s", feed_url, exc)
            return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_rss_date(entry: object) -> str:
    """Extract and normalise a published-date from a feedparser entry."""
    parsed = getattr(entry, "published_parsed", None)
    if parsed is not None:
        try:
            dt = datetime(*parsed[:6], tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
    return getattr(entry, "published", "") or ""
