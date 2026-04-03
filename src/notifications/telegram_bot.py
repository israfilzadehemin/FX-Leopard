"""
Telegram Bot Notification Layer for FX-Leopard.

Receives all signal types from the analysis pipeline and delivers
rich, well-formatted Markdown alerts to a configured Telegram chat
or channel via python-telegram-bot.

Features
--------
- Formats SIGNAL, WATCH, VOLATILITY, CALENDAR alerts with distinct styles
- Async message queue prevents Telegram rate-limit breaches
- Retry logic (up to max_retries attempts with exponential-ish backoff)
- Deduplication — the same signal is suppressed within dedup_window_seconds
- Startup message sent automatically on first run
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from telegram import Bot
from telegram.error import TelegramError

from analysis.models import TradeSignal, VolatilitySignal
from data.calendar_feed import EconomicEvent, COUNTRY_FLAGS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Startup message helper
# ---------------------------------------------------------------------------


def format_startup_message(
    pairs: List[str],
    timeframes: List[str],
    threshold: float,
) -> str:
    """
    Build the bot startup Markdown message.

    Example::

        🐆 FX-Leopard is online
        Watching: EURUSD, GBPUSD, USDJPY... (+8 more)
        Timeframes: M5, M15, H1, H4, D1
        Signal threshold: 7.0/10
    """
    displayed = pairs[:3]
    extra = len(pairs) - len(displayed)

    pairs_str = ", ".join(displayed)
    if extra > 0:
        pairs_str += f"... (+{extra} more)"

    tf_str = ", ".join(timeframes)

    return (
        f"🐆 *FX\\-Leopard is online*\n"
        f"Watching: {pairs_str}\n"
        f"Timeframes: {tf_str}\n"
        f"Signal threshold: {threshold}/10"
    )


# ---------------------------------------------------------------------------
# Alert formatters
# ---------------------------------------------------------------------------


def _escape_md(text: str) -> str:
    """Escape special MarkdownV2 characters in plain text segments."""
    special = r"\_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{c}" if c in special else c for c in str(text))


def format_signal_alert(signal: TradeSignal) -> str:
    """
    Format a SIGNAL (full trade setup) alert in Telegram MarkdownV2.

    Example::

        🐆 *FX-LEOPARD SIGNAL*

        🟢 *BUY EURUSD* — H1
        📊 Score: *8.2 / 10*
        🕐 14:00 UTC

        💰 *Entry Zone:* 1.0838 – 1.0845
        🛑 *Stop Loss:*  1.0808  (-30 pips)
        🎯 *Take Profit:* 1.0908  (+70 pips)
        ⚖️ *R:R Ratio:*  1 : 2.3

        ✅ *Confluences:*
        • EMA alignment bullish ...

        ❌ *Invalidation:* H1 close below 1.0805

        _Signal age: 12 seconds_
    """
    direction_emoji = "🟢" if signal.direction == "BUY" else "🔴"

    try:
        dt = datetime.fromisoformat(signal.timestamp.replace("Z", "+00:00"))
        time_str = dt.strftime("%H:%M UTC")
    except ValueError:
        time_str = _escape_md(signal.timestamp)

    lines: List[str] = [
        "🐆 *FX\\-LEOPARD SIGNAL*",
        "",
        f"{direction_emoji} *{signal.direction} {_escape_md(signal.symbol)}* — {_escape_md(signal.timeframe)}",
        f"📊 Score: *{signal.score:.1f} / 10*",
        f"🕐 {_escape_md(time_str)}",
        "",
    ]

    # Entry zone
    if len(signal.entry_zone) >= 2:
        lo = signal.entry_zone[0]
        hi = signal.entry_zone[1]
        lines.append(
            f"💰 *Entry Zone:* {_escape_md(f'{lo:.5g}')} – {_escape_md(f'{hi:.5g}')}"
        )

    # Stop loss
    if signal.stop_loss is not None:
        sl_pips_str = (
            f" \\(\\-{signal.sl_pips:.0f} pips\\)" if signal.sl_pips else ""
        )
        lines.append(
            f"🛑 *Stop Loss:*  {_escape_md(f'{signal.stop_loss:.5g}')}{sl_pips_str}"
        )

    # Take profit
    if signal.take_profit is not None:
        tp_pips_str = (
            f" \\(\\+{signal.tp_pips:.0f} pips\\)" if signal.tp_pips else ""
        )
        lines.append(
            f"🎯 *Take Profit:* {_escape_md(f'{signal.take_profit:.5g}')}{tp_pips_str}"
        )

    # R:R ratio
    if signal.rr_ratio is not None:
        lines.append(f"⚖️ *R:R Ratio:*  1 : {_escape_md(f'{signal.rr_ratio:.1f}')}")

    lines.append("")

    # Confluences
    if signal.confluences:
        lines.append("✅ *Confluences:*")
        for c in signal.confluences:
            lines.append(f"• {_escape_md(c)}")
        lines.append("")

    # Invalidation
    if signal.invalidation:
        lines.append(f"❌ *Invalidation:* {_escape_md(signal.invalidation)}")
        lines.append("")

    lines.append(f"_Signal age: {signal.signal_age_seconds} seconds_")

    return "\n".join(lines)


def format_watch_alert(signal: TradeSignal) -> str:
    """
    Format a WATCH (developing setup) alert in Telegram MarkdownV2.

    Example::

        🐆 *FX-LEOPARD WATCH*

        🟡 *GBPUSD — H4* developing setup
        📊 Score: *6.1 / 10*

        👀 Setup forming — waiting for confirmation
        • EMA alignment bullish
        ...

        ⏳ Watch for bullish candle close to confirm.
    """
    try:
        dt = datetime.fromisoformat(signal.timestamp.replace("Z", "+00:00"))
        dt.strftime("%H:%M UTC")  # validate timestamp parses correctly
    except ValueError:
        pass

    lines: List[str] = [
        "🐆 *FX\\-LEOPARD WATCH*",
        "",
        f"🟡 *{_escape_md(signal.symbol)} — {_escape_md(signal.timeframe)}* developing setup",
        f"📊 Score: *{signal.score:.1f} / 10*",
        "",
        "👀 Setup forming — waiting for confirmation",
    ]

    for c in signal.confluences:
        lines.append(f"• {_escape_md(c)}")

    lines.append("")

    confirm_verb = "bullish" if signal.direction == "BUY" else "bearish"
    lines.append(f"⏳ Watch for {_escape_md(confirm_verb)} candle close to confirm\\.")

    return "\n".join(lines)


def format_volatility_alert_md(signal: VolatilitySignal) -> str:
    """
    Format a VOLATILITY SPIKE alert in Telegram MarkdownV2.

    Example::

        ⚡ *VOLATILITY SPIKE — GBPUSD*

        📉 Moved *45 pips* in under 5 minutes
        💰 1.2680 → 1.2635
        📊 ATR ratio: *1.71x* average
        🕐 13:45 UTC

        ⚠️ Possible catalyst: Pending news scan
        👀 Monitor for continuation or reversal.
    """
    direction_arrow = "📉" if signal.direction == "bearish" else "📈"

    try:
        dt = datetime.fromisoformat(signal.timestamp.replace("Z", "+00:00"))
        time_str = dt.strftime("%H:%M UTC")
    except ValueError:
        time_str = _escape_md(signal.timestamp)

    lines: List[str] = [
        f"⚡ *VOLATILITY SPIKE — {_escape_md(signal.symbol)}*",
        "",
    ]

    if signal.trigger == "pip_spike" and signal.pips_moved is not None:
        window_min = signal.window_seconds // 60
        lines.append(
            f"{direction_arrow} Moved *{signal.pips_moved:.0f} pips* in under {window_min} minutes"
        )
    elif signal.trigger == "atr_expansion" and signal.atr_ratio is not None:
        lines.append(
            f"{direction_arrow} ATR expanded *{signal.atr_ratio:.2f}x* vs average"
        )
    else:
        pips = signal.magnitude
        lines.append(f"{direction_arrow} Moved *{pips:.0f} pips*")

    if signal.price_before is not None and signal.price_now is not None:
        lines.append(
            f"💰 {_escape_md(f'{signal.price_before:.5g}')} → {_escape_md(f'{signal.price_now:.5g}')}"
        )

    if signal.atr_ratio is not None:
        lines.append(f"📊 ATR ratio: *{signal.atr_ratio:.2f}x* average")

    lines.append(f"🕐 {_escape_md(time_str)}")
    lines.append("")

    catalyst = signal.probable_catalyst or signal.catalyst or "Pending news scan"
    lines.append(f"⚠️ Possible catalyst: {_escape_md(catalyst)}")
    lines.append("👀 Monitor for continuation or reversal\\.")

    return "\n".join(lines)


def format_calendar_alert_md(event: EconomicEvent, minutes_until: Optional[int] = None) -> str:
    """
    Format a CALENDAR (pre-event) alert in Telegram MarkdownV2.

    Example::

        📅 *UPCOMING HIGH IMPACT EVENT*

        🇺🇸 *Non-Farm Payrolls* (USD)
        ⏰ In *15 minutes* — 12:30 UTC

        📊 Forecast: 185K | Previous: 151K

        ⚡ Affected: EURUSD, GBPUSD, USDJPY, XAUUSD, USOIL

        ⚠️ Consider tightening stops or standing aside.
    """
    flag = COUNTRY_FLAGS.get(event.country, "🌐")

    try:
        event_time = event.get_datetime()
        time_str = event_time.strftime("%H:%M UTC")
    except Exception:
        time_str = _escape_md(event.datetime)

    lines: List[str] = [
        "📅 *UPCOMING HIGH IMPACT EVENT*",
        "",
        f"{flag} *{_escape_md(event.title)}* \\({_escape_md(event.country)}\\)",
    ]

    if minutes_until is not None:
        lines.append(f"⏰ In *{minutes_until} minutes* — {_escape_md(time_str)}")
    else:
        lines.append(f"⏰ {_escape_md(time_str)}")

    lines.append("")

    # Forecast / Previous on one line when both are available
    if event.forecast is not None and event.previous is not None:
        lines.append(
            f"📊 Forecast: {_escape_md(str(event.forecast))} \\| Previous: {_escape_md(str(event.previous))}"
        )
    elif event.forecast is not None:
        lines.append(f"📊 Forecast: {_escape_md(str(event.forecast))}")
    elif event.previous is not None:
        lines.append(f"📊 Previous: {_escape_md(str(event.previous))}")

    lines.append("")

    pairs_str = ", ".join(event.affected_pairs) if event.affected_pairs else "—"
    lines.append(f"⚡ Affected: {_escape_md(pairs_str)}")
    lines.append("")
    lines.append("⚠️ Consider tightening stops or standing aside\\.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------


def _signal_key(obj: Any) -> str:
    """
    Return a stable hash key for *obj* used by the deduplication cache.

    For TradeSignal: symbol + timeframe + direction + score bucket.
    For VolatilitySignal: symbol + trigger.
    For EconomicEvent: title + datetime.
    For plain str: the string itself.
    """
    if isinstance(obj, TradeSignal):
        raw = f"TRADE:{obj.symbol}:{obj.timeframe}:{obj.direction}:{int(obj.score)}"
    elif isinstance(obj, VolatilitySignal):
        raw = f"VOL:{obj.symbol}:{obj.trigger}"
    elif isinstance(obj, EconomicEvent):
        raw = f"CAL:{obj.title}:{obj.datetime}"
    else:
        raw = str(obj)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# TelegramNotifier
# ---------------------------------------------------------------------------


@dataclass
class _QueueItem:
    """An item waiting in the send queue."""

    text: str
    parse_mode: str = "MarkdownV2"


class TelegramNotifier:
    """
    Async Telegram notification delivery layer for FX-Leopard.

    Usage::

        notifier = TelegramNotifier(cfg.notifications)
        await notifier.start()               # starts the queue worker
        await notifier.send_signal(signal)   # enqueues a TradeSignal alert
        await notifier.stop()                # drains the queue and shuts down
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        dedup_window_seconds: int = 60,
        rate_limit_delay_seconds: float = 1.5,
        max_retries: int = 3,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._dedup_window = dedup_window_seconds
        self._rate_delay = rate_limit_delay_seconds
        self._max_retries = max_retries

        self._bot: Optional[Bot] = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

        # Deduplication cache: key → last-sent timestamp (monotonic)
        self._dedup_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise the Bot and start the background queue worker."""
        self._bot = Bot(token=self._bot_token)
        self._worker_task = asyncio.create_task(self._queue_worker())
        logger.info("TelegramNotifier started (chat_id=%s)", self._chat_id)

    async def stop(self) -> None:
        """Drain the queue and stop the worker gracefully."""
        if self._worker_task:
            # Signal the worker to stop after draining
            await self._queue.join()
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("TelegramNotifier stopped")

    # ------------------------------------------------------------------
    # Public send methods
    # ------------------------------------------------------------------

    async def send_signal(self, signal: TradeSignal) -> None:
        """Enqueue a SIGNAL or WATCH TradeSignal alert."""
        if self._is_duplicate(signal):
            logger.debug("Duplicate signal suppressed: %s %s", signal.symbol, signal.signal_type)
            return

        if signal.signal_type == "SIGNAL":
            text = format_signal_alert(signal)
        else:
            text = format_watch_alert(signal)

        self._mark_sent(signal)
        await self._queue.put(_QueueItem(text=text))

    async def send_volatility(self, signal: VolatilitySignal) -> None:
        """Enqueue a VOLATILITY alert."""
        if self._is_duplicate(signal):
            logger.debug("Duplicate volatility alert suppressed: %s", signal.symbol)
            return
        text = format_volatility_alert_md(signal)
        self._mark_sent(signal)
        await self._queue.put(_QueueItem(text=text))

    async def send_calendar(
        self, event: EconomicEvent, minutes_until: Optional[int] = None
    ) -> None:
        """Enqueue a CALENDAR pre-event alert."""
        if self._is_duplicate(event):
            logger.debug("Duplicate calendar alert suppressed: %s", event.title)
            return
        text = format_calendar_alert_md(event, minutes_until=minutes_until)
        self._mark_sent(event)
        await self._queue.put(_QueueItem(text=text))

    async def send_startup_message(
        self,
        pairs: List[str],
        timeframes: List[str],
        threshold: float,
    ) -> None:
        """Send the startup banner immediately (bypasses dedup)."""
        text = format_startup_message(pairs, timeframes, threshold)
        await self._queue.put(_QueueItem(text=text))

    async def send_raw(self, text: str) -> None:
        """Enqueue an arbitrary pre-formatted message (no dedup)."""
        await self._queue.put(_QueueItem(text=text))

    # ------------------------------------------------------------------
    # Queue worker
    # ------------------------------------------------------------------

    async def _queue_worker(self) -> None:
        """Background coroutine that drains the queue respecting rate limits."""
        while True:
            item: _QueueItem = await self._queue.get()
            try:
                await self._send_with_retry(item.text, item.parse_mode)
            except Exception as exc:
                logger.error("Failed to send Telegram message after retries: %s", exc)
            finally:
                self._queue.task_done()
            # Rate limiting: wait between messages
            await asyncio.sleep(self._rate_delay)

    async def _send_with_retry(self, text: str, parse_mode: str) -> None:
        """
        Attempt to send *text* to Telegram up to *max_retries* times.

        Raises the last exception if all attempts fail.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                await self._bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode=parse_mode,
                )
                return
            except TelegramError as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    wait = 2.0 * attempt  # 2s, 4s, ...
                    logger.warning(
                        "Telegram send attempt %d/%d failed (%s). Retrying in %.0fs…",
                        attempt,
                        self._max_retries,
                        exc,
                        wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        "All %d Telegram send attempts failed: %s",
                        self._max_retries,
                        exc,
                    )
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Deduplication helpers
    # ------------------------------------------------------------------

    def _is_duplicate(self, obj: Any) -> bool:
        """Return True if *obj* was already sent within the dedup window."""
        key = _signal_key(obj)
        last_sent = self._dedup_cache.get(key)
        if last_sent is None:
            return False
        return (time.monotonic() - last_sent) < self._dedup_window

    def _mark_sent(self, obj: Any) -> None:
        """Record *obj* as sent at the current monotonic time."""
        key = _signal_key(obj)
        self._dedup_cache[key] = time.monotonic()
        # Evict all entries that have aged beyond the dedup window to prevent
        # the cache from growing without bound in long-running deployments.
        now = time.monotonic()
        expired = [k for k, ts in self._dedup_cache.items() if (now - ts) >= self._dedup_window]
        for k in expired:
            del self._dedup_cache[k]
