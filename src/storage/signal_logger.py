"""
SignalLogger for FX-Leopard.

Records every alert fired by the system into a local SQLite database for
later review, performance tracking, and simple analytics.

Thread-safe: uses a threading.Lock to serialise writes and thread-local
connections for reads so that the same logger can be used from both the
main async loop and APScheduler background threads.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from analysis.models import SentimentSignal, TradeSignal
from data.calendar_feed import EconomicEvent

logger = logging.getLogger(__name__)

# Default database path (relative to the project root)
DEFAULT_DB_PATH = "data/signals.db"


class SignalLogger:
    """
    Persists every FX-Leopard alert into a SQLite database.

    Usage::

        sl = SignalLogger()                   # uses data/signals.db
        sl = SignalLogger(":memory:")         # in-memory (useful for tests)
        sl = SignalLogger("path/to/db.db")

        row_id = sl.log_signal(trade_signal)
        row_id = sl.log_news_event(sentiment_signal)
        row_id = sl.log_calendar_event(economic_event)

        rows   = sl.get_signals(symbol="EURUSD", limit=50)
        stats  = sl.get_summary(since="2026-03-01")
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        self._write_lock = threading.Lock()

        # For in-memory databases every thread must share the same connection,
        # because each new sqlite3.connect(":memory:") creates a separate DB.
        # For file-backed databases we also share one connection (with
        # check_same_thread=False) to avoid complexity while still being
        # safe under the write lock.
        self._conn: sqlite3.Connection = sqlite3.connect(
            db_path, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row

        # Initialise the schema once (creates tables if they don't exist)
        self._init_db()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """Return the shared SQLite connection."""
        return self._conn

    def _init_db(self) -> None:
        """Create the database tables if they do not exist yet."""
        with self._write_lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT,
                    signal_type TEXT NOT NULL,
                    direction TEXT,
                    score REAL,
                    entry_low REAL,
                    entry_high REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    rr_ratio REAL,
                    sl_pips REAL,
                    tp_pips REAL,
                    confluences TEXT,
                    invalidation TEXT,
                    summary TEXT,
                    raw_payload TEXT,
                    fired_at TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS news_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    headline TEXT,
                    source TEXT,
                    affected_symbols TEXT,
                    sentiment TEXT,
                    impact TEXT,
                    summary TEXT,
                    published_at TEXT,
                    processed_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS calendar_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    country TEXT,
                    event_datetime TEXT,
                    impact TEXT,
                    forecast TEXT,
                    previous TEXT,
                    actual TEXT,
                    beat_miss TEXT,
                    affected_pairs TEXT,
                    alerted_at TEXT
                );
                """
            )
            self._conn.commit()
            logger.debug("SignalLogger: database schema initialised at %s", self._db_path)

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_signal(self, trade_signal: TradeSignal) -> int:
        """
        Persist a :class:`~analysis.models.TradeSignal` to the *signals* table.

        Returns the newly inserted row id.
        """
        entry_zone = trade_signal.entry_zone or []
        entry_low = entry_zone[0] if len(entry_zone) > 0 else None
        entry_high = entry_zone[1] if len(entry_zone) > 1 else None

        row = (
            trade_signal.symbol,
            trade_signal.timeframe,
            trade_signal.signal_type,
            trade_signal.direction,
            trade_signal.score,
            entry_low,
            entry_high,
            trade_signal.stop_loss,
            trade_signal.take_profit,
            trade_signal.rr_ratio,
            trade_signal.sl_pips,
            trade_signal.tp_pips,
            json.dumps(trade_signal.confluences),
            trade_signal.invalidation,
            None,  # summary — not on TradeSignal model
            json.dumps(trade_signal.to_dict()),
            trade_signal.timestamp,
        )

        sql = """
            INSERT INTO signals (
                symbol, timeframe, signal_type, direction, score,
                entry_low, entry_high, stop_loss, take_profit, rr_ratio,
                sl_pips, tp_pips, confluences, invalidation, summary,
                raw_payload, fired_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        row_id = self._execute_write(sql, row)
        logger.debug(
            "SignalLogger: logged %s %s %s (row_id=%d)",
            trade_signal.signal_type,
            trade_signal.symbol,
            trade_signal.direction,
            row_id,
        )
        return row_id

    def log_news_event(self, sentiment_signal: SentimentSignal) -> int:
        """
        Persist a :class:`~analysis.models.SentimentSignal` to the
        *news_events* table.

        Returns the newly inserted row id.
        """
        headline = sentiment_signal.headlines[0] if sentiment_signal.headlines else None
        affected_symbols = json.dumps([sentiment_signal.symbol])

        row = (
            headline,
            None,   # source — not on SentimentSignal model
            affected_symbols,
            sentiment_signal.direction,
            sentiment_signal.impact,
            sentiment_signal.summary,
            sentiment_signal.timestamp,
        )

        sql = """
            INSERT INTO news_events (
                headline, source, affected_symbols, sentiment,
                impact, summary, published_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        row_id = self._execute_write(sql, row)
        logger.debug(
            "SignalLogger: logged news_event for %s (row_id=%d)",
            sentiment_signal.symbol,
            row_id,
        )
        return row_id

    def log_calendar_event(self, economic_event: EconomicEvent) -> int:
        """
        Persist an :class:`~data.calendar_feed.EconomicEvent` to the
        *calendar_events* table.

        Returns the newly inserted row id.
        """
        beat_miss = _classify_beat_miss(economic_event)
        affected_pairs = json.dumps(economic_event.affected_pairs)
        alerted_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        row = (
            economic_event.title,
            economic_event.country,
            economic_event.datetime,
            economic_event.impact,
            economic_event.forecast,
            economic_event.previous,
            economic_event.actual,
            beat_miss,
            affected_pairs,
            alerted_at,
        )

        sql = """
            INSERT INTO calendar_events (
                title, country, event_datetime, impact, forecast,
                previous, actual, beat_miss, affected_pairs, alerted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        row_id = self._execute_write(sql, row)
        logger.debug(
            "SignalLogger: logged calendar_event '%s' %s (row_id=%d)",
            economic_event.title,
            economic_event.country,
            row_id,
        )
        return row_id

    # ------------------------------------------------------------------
    # Query / reporting methods
    # ------------------------------------------------------------------

    def get_signals(
        self,
        since: Optional[str] = None,
        symbol: Optional[str] = None,
        signal_type: Optional[str] = None,
        direction: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Fetch signal rows with optional filtering.

        Args:
            since:       ISO8601 date string, e.g. ``"2026-03-01"``.  Only
                         signals fired on or after this date are returned.
            symbol:      Filter to a specific trading symbol, e.g. ``"EURUSD"``.
            signal_type: Filter to ``"SIGNAL"`` or ``"WATCH"``.
            direction:   Filter to ``"BUY"`` or ``"SELL"``.
            limit:       Maximum number of rows to return (default 100).

        Returns:
            List of dicts, one per matching row.
        """
        clauses: List[str] = []
        params: List = []

        if since:
            clauses.append("fired_at >= ?")
            params.append(since)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if signal_type:
            clauses.append("signal_type = ?")
            params.append(signal_type)
        if direction:
            clauses.append("direction = ?")
            params.append(direction)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM signals {where} ORDER BY fired_at DESC LIMIT ?"
        params.append(limit)

        # All operations (reads and writes) are serialised through the same
        # lock because Python's sqlite3 Connection object is not safe to use
        # from multiple threads simultaneously even for reads when check_same_thread
        # is disabled.  A single lock is the simplest correct approach.
        with self._write_lock:
            cur = self._conn.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def get_summary(self, since: Optional[str] = None) -> Dict:
        """
        Return aggregated statistics over the stored signals.

        Args:
            since: ISO8601 date string.  When provided only signals fired on
                   or after this date are included.

        Returns:
            Dict with keys: ``total_signals``, ``by_type``, ``by_symbol``,
            ``by_direction``, ``avg_score``, ``avg_rr``, ``period``.
        """
        where = "WHERE fired_at >= ?" if since else ""
        params: List = [since] if since else []

        # Serialised via the same lock used for writes — see get_signals for rationale.
        with self._write_lock:
            # Total
            total = self._conn.execute(
                f"SELECT COUNT(*) FROM signals {where}", params
            ).fetchone()[0]

            # By type
            by_type: Dict[str, int] = {}
            for row in self._conn.execute(
                f"SELECT signal_type, COUNT(*) as cnt FROM signals {where} GROUP BY signal_type",
                params,
            ):
                by_type[row[0]] = row[1]

            # By symbol
            by_symbol: Dict[str, int] = {}
            for row in self._conn.execute(
                f"SELECT symbol, COUNT(*) as cnt FROM signals {where} GROUP BY symbol ORDER BY cnt DESC",
                params,
            ):
                by_symbol[row[0]] = row[1]

            # By direction
            by_direction: Dict[str, int] = {}
            for row in self._conn.execute(
                f"SELECT direction, COUNT(*) as cnt FROM signals {where} GROUP BY direction",
                params,
            ):
                by_direction[row[0]] = row[1]

            # Averages
            avg_row = self._conn.execute(
                f"SELECT AVG(score), AVG(rr_ratio) FROM signals {where}", params
            ).fetchone()
            avg_score = round(avg_row[0], 2) if avg_row[0] is not None else 0.0
            avg_rr = round(avg_row[1], 2) if avg_row[1] is not None else 0.0

            # Period string
            period_row = self._conn.execute(
                f"SELECT MIN(fired_at), MAX(fired_at) FROM signals {where}", params
            ).fetchone()

        if period_row[0] and period_row[1]:
            period = f"{period_row[0][:10]} to {period_row[1][:10]}"
        elif since:
            today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
            period = f"{since} to {today}"
        else:
            period = "all time"

        return {
            "total_signals": total,
            "by_type": by_type,
            "by_symbol": by_symbol,
            "by_direction": by_direction,
            "avg_score": avg_score,
            "avg_rr": avg_rr,
            "period": period,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_write(self, sql: str, params: tuple) -> int:
        """Execute a write statement under the lock. Returns lastrowid."""
        with self._write_lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur.lastrowid


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _classify_beat_miss(event: EconomicEvent) -> Optional[str]:
    """Return 'beat' | 'miss' | 'inline' | None based on actual vs forecast."""
    if event.actual is None or event.forecast is None:
        return None
    actual_val = _parse_numeric(event.actual)
    forecast_val = _parse_numeric(event.forecast)
    if actual_val is None or forecast_val is None:
        return None
    if actual_val > forecast_val:
        return "beat"
    if actual_val < forecast_val:
        return "miss"
    return "inline"


def _parse_numeric(value: str) -> Optional[float]:
    """Strip common suffixes and parse to float (mirrors calendar_feed logic)."""
    if not value:
        return None
    v = value.strip().replace(",", "").upper()
    multiplier = 1.0
    if v.endswith("B"):
        multiplier = 1_000_000_000
        v = v[:-1]
    elif v.endswith("M"):
        multiplier = 1_000_000
        v = v[:-1]
    elif v.endswith("K"):
        multiplier = 1_000
        v = v[:-1]
    elif v.endswith("%"):
        v = v[:-1]
    try:
        return float(v) * multiplier
    except ValueError:
        return None
