"""
Unit tests for SignalLogger — src/storage/signal_logger.py.

Uses in-memory SQLite (:memory:) so no file I/O is required.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

import pytest

from analysis.models import SentimentSignal, TradeSignal
from data.calendar_feed import EconomicEvent
from storage.signal_logger import SignalLogger, _classify_beat_miss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sl() -> SignalLogger:
    """Return a fresh in-memory SignalLogger for each test."""
    return SignalLogger(db_path=":memory:")


def _make_trade_signal(**kwargs) -> TradeSignal:
    defaults = dict(
        symbol="EURUSD",
        timeframe="H1",
        timestamp="2026-04-01T12:00:00Z",
        signal_type="SIGNAL",
        direction="BUY",
        score=7.5,
        entry_zone=[1.0800, 1.0810],
        stop_loss=1.0750,
        take_profit=1.0900,
        rr_ratio=2.0,
        sl_pips=50.0,
        tp_pips=100.0,
        confluences=["ema_alignment_bullish", "rsi_recovering_from_oversold"],
        invalidation="H4 candle closes below 1.0750",
    )
    defaults.update(kwargs)
    return TradeSignal(**defaults)


def _make_sentiment_signal(**kwargs) -> SentimentSignal:
    defaults = dict(
        symbol="EURUSD",
        timestamp="2026-04-01T12:00:00Z",
        direction="bullish",
        sentiment="bullish",
        strength=0.75,
        confidence=0.80,
        impact="high",
        headlines=["ECB raises rates unexpectedly"],
        summary="ECB surprise hike pushes EUR higher",
        score_contribution=1.2,
    )
    defaults.update(kwargs)
    return SentimentSignal(**defaults)


def _make_economic_event(**kwargs) -> EconomicEvent:
    defaults = dict(
        title="US Non-Farm Payrolls",
        country="USD",
        datetime="2026-04-03T12:30:00Z",
        impact="high",
        forecast="200K",
        previous="180K",
        actual=None,
        affected_pairs=["EURUSD", "GBPUSD", "USDJPY"],
    )
    defaults.update(kwargs)
    return EconomicEvent(**defaults)


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_tables_created(self, sl: SignalLogger) -> None:
        """All three tables must exist after initialisation."""
        conn = sl._get_connection()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "signals" in tables
        assert "news_events" in tables
        assert "calendar_events" in tables

    def test_signals_table_empty(self, sl: SignalLogger) -> None:
        rows = sl.get_signals()
        assert rows == []

    def test_summary_empty_db(self, sl: SignalLogger) -> None:
        stats = sl.get_summary()
        assert stats["total_signals"] == 0
        assert stats["by_type"] == {}
        assert stats["avg_score"] == 0.0


# ---------------------------------------------------------------------------
# log_signal tests
# ---------------------------------------------------------------------------


class TestLogSignal:
    def test_returns_row_id(self, sl: SignalLogger) -> None:
        ts = _make_trade_signal()
        row_id = sl.log_signal(ts)
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_row_persisted(self, sl: SignalLogger) -> None:
        ts = _make_trade_signal()
        row_id = sl.log_signal(ts)
        rows = sl.get_signals()
        assert len(rows) == 1
        row = rows[0]
        assert row["id"] == row_id
        assert row["symbol"] == "EURUSD"
        assert row["timeframe"] == "H1"
        assert row["signal_type"] == "SIGNAL"
        assert row["direction"] == "BUY"
        assert row["score"] == pytest.approx(7.5)
        assert row["entry_low"] == pytest.approx(1.0800)
        assert row["entry_high"] == pytest.approx(1.0810)
        assert row["stop_loss"] == pytest.approx(1.0750)
        assert row["take_profit"] == pytest.approx(1.0900)
        assert row["rr_ratio"] == pytest.approx(2.0)
        assert row["sl_pips"] == pytest.approx(50.0)
        assert row["tp_pips"] == pytest.approx(100.0)
        assert row["fired_at"] == "2026-04-01T12:00:00Z"

    def test_confluences_stored_as_json(self, sl: SignalLogger) -> None:
        ts = _make_trade_signal()
        sl.log_signal(ts)
        row = sl.get_signals()[0]
        parsed = json.loads(row["confluences"])
        assert "ema_alignment_bullish" in parsed

    def test_raw_payload_stored(self, sl: SignalLogger) -> None:
        ts = _make_trade_signal()
        sl.log_signal(ts)
        row = sl.get_signals()[0]
        payload = json.loads(row["raw_payload"])
        assert payload["symbol"] == "EURUSD"

    def test_empty_entry_zone(self, sl: SignalLogger) -> None:
        ts = _make_trade_signal(entry_zone=[])
        sl.log_signal(ts)
        row = sl.get_signals()[0]
        assert row["entry_low"] is None
        assert row["entry_high"] is None

    def test_watch_signal(self, sl: SignalLogger) -> None:
        ts = _make_trade_signal(signal_type="WATCH", score=5.5)
        row_id = sl.log_signal(ts)
        row = sl.get_signals()[0]
        assert row["signal_type"] == "WATCH"
        assert row_id >= 1

    def test_multiple_rows(self, sl: SignalLogger) -> None:
        sl.log_signal(_make_trade_signal(symbol="EURUSD"))
        sl.log_signal(_make_trade_signal(symbol="GBPUSD"))
        sl.log_signal(_make_trade_signal(symbol="XAUUSD"))
        rows = sl.get_signals(limit=10)
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# log_news_event tests
# ---------------------------------------------------------------------------


class TestLogNewsEvent:
    def test_returns_row_id(self, sl: SignalLogger) -> None:
        ss = _make_sentiment_signal()
        row_id = sl.log_news_event(ss)
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_row_persisted(self, sl: SignalLogger) -> None:
        ss = _make_sentiment_signal()
        sl.log_news_event(ss)
        conn = sl._get_connection()
        rows = conn.execute("SELECT * FROM news_events").fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["headline"] == "ECB raises rates unexpectedly"
        assert row["sentiment"] == "bullish"
        assert row["impact"] == "high"
        assert row["summary"] == "ECB surprise hike pushes EUR higher"
        assert row["published_at"] == "2026-04-01T12:00:00Z"

    def test_affected_symbols_json(self, sl: SignalLogger) -> None:
        ss = _make_sentiment_signal(symbol="GBPUSD")
        sl.log_news_event(ss)
        conn = sl._get_connection()
        row = dict(conn.execute("SELECT * FROM news_events").fetchone())
        symbols = json.loads(row["affected_symbols"])
        assert "GBPUSD" in symbols

    def test_no_headlines(self, sl: SignalLogger) -> None:
        ss = _make_sentiment_signal(headlines=[])
        row_id = sl.log_news_event(ss)
        conn = sl._get_connection()
        row = dict(conn.execute(
            "SELECT * FROM news_events WHERE id=?", (row_id,)
        ).fetchone())
        assert row["headline"] is None


# ---------------------------------------------------------------------------
# log_calendar_event tests
# ---------------------------------------------------------------------------


class TestLogCalendarEvent:
    def test_returns_row_id(self, sl: SignalLogger) -> None:
        ev = _make_economic_event()
        row_id = sl.log_calendar_event(ev)
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_row_persisted(self, sl: SignalLogger) -> None:
        ev = _make_economic_event()
        sl.log_calendar_event(ev)
        conn = sl._get_connection()
        row = dict(conn.execute("SELECT * FROM calendar_events").fetchone())
        assert row["title"] == "US Non-Farm Payrolls"
        assert row["country"] == "USD"
        assert row["impact"] == "high"
        assert row["forecast"] == "200K"
        assert row["previous"] == "180K"
        assert row["actual"] is None

    def test_beat_miss_none_when_no_actual(self, sl: SignalLogger) -> None:
        ev = _make_economic_event(actual=None)
        sl.log_calendar_event(ev)
        conn = sl._get_connection()
        row = dict(conn.execute("SELECT * FROM calendar_events").fetchone())
        assert row["beat_miss"] is None

    def test_beat_miss_beat(self, sl: SignalLogger) -> None:
        ev = _make_economic_event(actual="250K", forecast="200K")
        sl.log_calendar_event(ev)
        conn = sl._get_connection()
        row = dict(conn.execute("SELECT * FROM calendar_events").fetchone())
        assert row["beat_miss"] == "beat"

    def test_beat_miss_miss(self, sl: SignalLogger) -> None:
        ev = _make_economic_event(actual="150K", forecast="200K")
        sl.log_calendar_event(ev)
        conn = sl._get_connection()
        row = dict(conn.execute("SELECT * FROM calendar_events").fetchone())
        assert row["beat_miss"] == "miss"

    def test_beat_miss_inline(self, sl: SignalLogger) -> None:
        ev = _make_economic_event(actual="200K", forecast="200K")
        sl.log_calendar_event(ev)
        conn = sl._get_connection()
        row = dict(conn.execute("SELECT * FROM calendar_events").fetchone())
        assert row["beat_miss"] == "inline"

    def test_affected_pairs_json(self, sl: SignalLogger) -> None:
        ev = _make_economic_event()
        sl.log_calendar_event(ev)
        conn = sl._get_connection()
        row = dict(conn.execute("SELECT * FROM calendar_events").fetchone())
        pairs = json.loads(row["affected_pairs"])
        assert "EURUSD" in pairs
        assert "GBPUSD" in pairs


# ---------------------------------------------------------------------------
# get_signals filtering tests
# ---------------------------------------------------------------------------


class TestGetSignals:
    def _seed(self, sl: SignalLogger) -> None:
        sl.log_signal(_make_trade_signal(
            symbol="EURUSD", direction="BUY", signal_type="SIGNAL",
            timestamp="2026-03-15T08:00:00Z",
        ))
        sl.log_signal(_make_trade_signal(
            symbol="EURUSD", direction="SELL", signal_type="WATCH",
            timestamp="2026-03-20T10:00:00Z",
        ))
        sl.log_signal(_make_trade_signal(
            symbol="GBPUSD", direction="BUY", signal_type="SIGNAL",
            timestamp="2026-04-01T14:00:00Z",
        ))
        sl.log_signal(_make_trade_signal(
            symbol="XAUUSD", direction="SELL", signal_type="SIGNAL",
            timestamp="2026-04-02T09:00:00Z",
        ))

    def test_no_filter(self, sl: SignalLogger) -> None:
        self._seed(sl)
        rows = sl.get_signals(limit=100)
        assert len(rows) == 4

    def test_filter_by_symbol(self, sl: SignalLogger) -> None:
        self._seed(sl)
        rows = sl.get_signals(symbol="EURUSD")
        assert all(r["symbol"] == "EURUSD" for r in rows)
        assert len(rows) == 2

    def test_filter_by_signal_type(self, sl: SignalLogger) -> None:
        self._seed(sl)
        rows = sl.get_signals(signal_type="WATCH")
        assert all(r["signal_type"] == "WATCH" for r in rows)
        assert len(rows) == 1

    def test_filter_by_direction(self, sl: SignalLogger) -> None:
        self._seed(sl)
        rows = sl.get_signals(direction="SELL")
        assert all(r["direction"] == "SELL" for r in rows)
        assert len(rows) == 2

    def test_filter_by_since(self, sl: SignalLogger) -> None:
        self._seed(sl)
        rows = sl.get_signals(since="2026-04-01")
        # Only signals with fired_at >= "2026-04-01"
        assert len(rows) == 2
        for r in rows:
            assert r["fired_at"] >= "2026-04-01"

    def test_filter_combined(self, sl: SignalLogger) -> None:
        self._seed(sl)
        rows = sl.get_signals(symbol="EURUSD", direction="BUY")
        assert len(rows) == 1
        assert rows[0]["symbol"] == "EURUSD"
        assert rows[0]["direction"] == "BUY"

    def test_limit(self, sl: SignalLogger) -> None:
        self._seed(sl)
        rows = sl.get_signals(limit=2)
        assert len(rows) == 2

    def test_returns_list_of_dicts(self, sl: SignalLogger) -> None:
        self._seed(sl)
        rows = sl.get_signals(limit=1)
        assert isinstance(rows, list)
        assert isinstance(rows[0], dict)


# ---------------------------------------------------------------------------
# get_summary tests
# ---------------------------------------------------------------------------


class TestGetSummary:
    def _seed(self, sl: SignalLogger) -> None:
        sl.log_signal(_make_trade_signal(
            symbol="EURUSD", direction="BUY", signal_type="SIGNAL",
            score=8.0, rr_ratio=2.5, timestamp="2026-03-15T08:00:00Z",
        ))
        sl.log_signal(_make_trade_signal(
            symbol="EURUSD", direction="SELL", signal_type="WATCH",
            score=6.0, rr_ratio=1.5, timestamp="2026-03-20T10:00:00Z",
        ))
        sl.log_signal(_make_trade_signal(
            symbol="GBPUSD", direction="BUY", signal_type="SIGNAL",
            score=7.0, rr_ratio=2.0, timestamp="2026-04-01T14:00:00Z",
        ))

    def test_total_signals(self, sl: SignalLogger) -> None:
        self._seed(sl)
        stats = sl.get_summary()
        assert stats["total_signals"] == 3

    def test_by_type(self, sl: SignalLogger) -> None:
        self._seed(sl)
        stats = sl.get_summary()
        assert stats["by_type"]["SIGNAL"] == 2
        assert stats["by_type"]["WATCH"] == 1

    def test_by_symbol(self, sl: SignalLogger) -> None:
        self._seed(sl)
        stats = sl.get_summary()
        assert stats["by_symbol"]["EURUSD"] == 2
        assert stats["by_symbol"]["GBPUSD"] == 1

    def test_by_direction(self, sl: SignalLogger) -> None:
        self._seed(sl)
        stats = sl.get_summary()
        assert stats["by_direction"]["BUY"] == 2
        assert stats["by_direction"]["SELL"] == 1

    def test_avg_score(self, sl: SignalLogger) -> None:
        self._seed(sl)
        stats = sl.get_summary()
        expected = round((8.0 + 6.0 + 7.0) / 3, 2)
        assert stats["avg_score"] == pytest.approx(expected, abs=0.01)

    def test_avg_rr(self, sl: SignalLogger) -> None:
        self._seed(sl)
        stats = sl.get_summary()
        expected = round((2.5 + 1.5 + 2.0) / 3, 2)
        assert stats["avg_rr"] == pytest.approx(expected, abs=0.01)

    def test_since_filter(self, sl: SignalLogger) -> None:
        self._seed(sl)
        stats = sl.get_summary(since="2026-04-01")
        assert stats["total_signals"] == 1

    def test_period_string(self, sl: SignalLogger) -> None:
        self._seed(sl)
        stats = sl.get_summary()
        assert "2026-03-15" in stats["period"]
        assert "2026-04-01" in stats["period"]

    def test_period_with_since(self, sl: SignalLogger) -> None:
        self._seed(sl)
        stats = sl.get_summary(since="2026-04-01")
        assert "2026-04-01" in stats["period"]

    def test_empty_db(self, sl: SignalLogger) -> None:
        stats = sl.get_summary()
        assert stats["total_signals"] == 0
        assert stats["avg_score"] == 0.0
        assert stats["avg_rr"] == 0.0


# ---------------------------------------------------------------------------
# Thread safety tests
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_writes(self, sl: SignalLogger) -> None:
        """Multiple threads writing concurrently should not corrupt the DB."""
        errors: list[Exception] = []
        rows_written = 50

        def write_signals() -> None:
            try:
                for i in range(rows_written // 5):
                    ts = _make_trade_signal(
                        symbol=f"SYM{i}",
                        score=float(i),
                    )
                    sl.log_signal(ts)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write_signals) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        count = sl._get_connection().execute(
            "SELECT COUNT(*) FROM signals"
        ).fetchone()[0]
        assert count == rows_written

    def test_concurrent_reads_and_writes(self, sl: SignalLogger) -> None:
        """Reads should not block on concurrent writes."""
        errors: list[Exception] = []

        def write() -> None:
            for _ in range(10):
                try:
                    sl.log_signal(_make_trade_signal())
                except Exception as exc:
                    errors.append(exc)

        def read() -> None:
            for _ in range(10):
                try:
                    sl.get_signals(limit=100)
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=write) for _ in range(3)] + [
            threading.Thread(target=read) for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# _classify_beat_miss helper
# ---------------------------------------------------------------------------


class TestClassifyBeatMiss:
    def test_beat(self) -> None:
        ev = _make_economic_event(actual="250K", forecast="200K")
        assert _classify_beat_miss(ev) == "beat"

    def test_miss(self) -> None:
        ev = _make_economic_event(actual="150K", forecast="200K")
        assert _classify_beat_miss(ev) == "miss"

    def test_inline(self) -> None:
        ev = _make_economic_event(actual="200K", forecast="200K")
        assert _classify_beat_miss(ev) == "inline"

    def test_no_actual(self) -> None:
        ev = _make_economic_event(actual=None, forecast="200K")
        assert _classify_beat_miss(ev) is None

    def test_no_forecast(self) -> None:
        ev = _make_economic_event(actual="200K", forecast=None)
        assert _classify_beat_miss(ev) is None

    def test_percentage(self) -> None:
        ev = _make_economic_event(actual="3.2%", forecast="3.0%")
        assert _classify_beat_miss(ev) == "beat"
