"""
FX-Leopard 🐆
Main entry point — starts the WebSocket price feed and logs incoming candles.
"""

import asyncio
import logging

from dotenv import load_dotenv

from config import load_config
from data.price_feed import PriceFeed

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fx-leopard")


async def on_candle(candle: dict) -> None:
    logger.info(
        "📊 Candle: %s %s | O:%.5f H:%.5f L:%.5f C:%.5f V:%d @ %s",
        candle["symbol"],
        candle["timeframe"],
        candle["open"],
        candle["high"],
        candle["low"],
        candle["close"],
        candle["volume"],
        candle["timestamp"],
    )


async def main() -> None:
    logger.info("🐆 FX-Leopard starting up...")

    cfg = load_config()
    logger.info("Config loaded — %d pairs, %d timeframes", len(cfg.pairs), len(cfg.timeframes))

    feed = PriceFeed(
        api_key=cfg.api_keys.twelvedata,
        symbols=cfg.pairs,
        timeframes=cfg.timeframes,
        on_candle=on_candle,
    )

    logger.info("Starting price feed...")
    await feed.run()


if __name__ == "__main__":
    asyncio.run(main())
