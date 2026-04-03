"""
FX-Leopard 🐆
Main entry point — starts all engines and runs the agent loop.

Component wiring
----------------
PriceFeed → TechnicalEngine → ConfluenceEngine → TelegramNotifier
                                      ↑
SentimentEngine  ──────────────────────┤
CalendarFeed  ──── (pre-event alerts) → TelegramNotifier
VolatilityMonitor ─ (volatility alerts) → TelegramNotifier + ConfluenceEngine
"""

import asyncio
import logging
import signal as _signal
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from config import load_config
from analysis.confluence import ConfluenceEngine
from analysis.models import SentimentSignal, TradeSignal, VolatilitySignal
from analysis.sentiment import SentimentEngine
from analysis.technical import TechnicalEngine
from analysis.volatility import VolatilityMonitor
from data.calendar_feed import CalendarFeed
from data.price_feed import PriceFeed
from notifications.telegram_bot import TelegramNotifier
from storage.signal_logger import SignalLogger

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fx-leopard")


def _load_raw_config(cfg_path: str | None = None) -> Dict[str, Any]:
    """Load the raw YAML dict (used for engines that accept a dict config)."""
    import os

    if cfg_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_path = os.path.join(base_dir, "config", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


async def main() -> None:
    logger.info("🐆 FX-Leopard starting up...")

    cfg = load_config()
    raw_cfg = _load_raw_config()

    # ------------------------------------------------------------------
    # 0. Signal Logger — persists every alert to SQLite
    # ------------------------------------------------------------------
    signal_logger = SignalLogger()
    logger.info("SignalLogger initialised")

    # ------------------------------------------------------------------
    # 1. Telegram Notifier
    # ------------------------------------------------------------------
    notifier = TelegramNotifier(
        bot_token=cfg.notifications.bot_token,
        chat_id=cfg.notifications.chat_id,
        dedup_window_seconds=cfg.notifications.dedup_window_seconds,
        rate_limit_delay_seconds=cfg.notifications.rate_limit_delay_seconds,
        max_retries=cfg.notifications.max_retries,
    )
    await notifier.start()

    # ------------------------------------------------------------------
    # 2. Confluence Engine — routes scored trade signals to notifier
    # ------------------------------------------------------------------
    loop = asyncio.get_running_loop()

    def on_trade_signal(signal: TradeSignal) -> None:
        loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(notifier.send_signal(signal))
        )

    confluence = ConfluenceEngine(
        signal_callback=on_trade_signal,
        config=raw_cfg,
        signal_logger=signal_logger,
    )

    # ------------------------------------------------------------------
    # 3. Technical Engine — feeds ConfluenceEngine
    # ------------------------------------------------------------------
    tech_engine = TechnicalEngine(
        on_signal=confluence.on_technical_signal,
    )

    # ------------------------------------------------------------------
    # 4. Volatility Monitor
    # ------------------------------------------------------------------

    def on_volatility_signal(vs: VolatilitySignal) -> None:
        confluence.update_volatility(vs)
        loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(notifier.send_volatility(vs))
        )

    vol_monitor = VolatilityMonitor(
        config=raw_cfg,
        volatility_callback=on_volatility_signal,
    )

    # ------------------------------------------------------------------
    # 5. Price Feed — drives TechnicalEngine + VolatilityMonitor
    # ------------------------------------------------------------------

    async def on_candle(candle: Any) -> None:
        tech_engine.on_candle(candle)
        if hasattr(candle, "symbol"):
            vol_monitor.on_candle(candle)

    feed = PriceFeed(
        api_key=cfg.api_keys.twelvedata,
        symbols=cfg.pairs,
        timeframes=cfg.timeframes,
        on_candle=on_candle,
    )

    # ------------------------------------------------------------------
    # 6. Sentiment Engine
    # ------------------------------------------------------------------
    sentiment_engine = SentimentEngine(
        signal_callback=confluence.update_sentiment,
        config=raw_cfg,
        signal_logger=signal_logger,
    )
    sentiment_engine.start()

    # ------------------------------------------------------------------
    # 7. Calendar Feed
    # ------------------------------------------------------------------

    def on_calendar_alert(message: str) -> None:
        loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(notifier.send_raw(message))
        )

    calendar_feed = CalendarFeed(
        config=raw_cfg,
        on_alert=on_calendar_alert,
        signal_logger=signal_logger,
    )
    calendar_feed.start()

    # ------------------------------------------------------------------
    # 8. Startup message
    # ------------------------------------------------------------------
    await notifier.send_startup_message(
        pairs=cfg.pairs,
        timeframes=cfg.timeframes,
        threshold=cfg.confluence_threshold,
    )
    logger.info(
        "🐆 FX-Leopard online — %d pairs, %d timeframes",
        len(cfg.pairs),
        len(cfg.timeframes),
    )

    # ------------------------------------------------------------------
    # 9. Graceful shutdown
    # ------------------------------------------------------------------
    stop_event = asyncio.Event()

    def _handle_shutdown(*_: Any) -> None:
        logger.info("Shutdown signal received — stopping FX-Leopard...")
        loop.call_soon_threadsafe(stop_event.set)

    for sig in (_signal.SIGTERM, _signal.SIGINT):
        loop.add_signal_handler(sig, _handle_shutdown)

    # Start the price feed in the background
    feed_task = asyncio.create_task(feed.run())

    try:
        await stop_event.wait()
    finally:
        feed_task.cancel()
        try:
            await feed_task
        except asyncio.CancelledError:
            pass

        sentiment_engine.stop()
        calendar_feed.stop()
        await notifier.stop()
        logger.info("🐆 FX-Leopard shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())


