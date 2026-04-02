"""
PriceFeed — Real-time price data ingestion layer for FX-Leopard.

Connects to the TwelveData WebSocket API, subscribes to configured
symbols, builds OHLCV candle buffers per timeframe, and emits
completed candles via an async callback.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Awaitable

import websockets

logger = logging.getLogger(__name__)

# TwelveData WebSocket endpoint
TWELVEDATA_WS_URL = "wss://ws.twelvedata.com/v1/quotes/price"

# Timeframe durations in seconds
TIMEFRAME_SECONDS: Dict[str, int] = {
    "M5": 5 * 60,
    "M15": 15 * 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
    "D1": 24 * 60 * 60,
}

# Reconnection backoff settings
BACKOFF_INITIAL = 1
BACKOFF_MAX = 60
BACKOFF_FACTOR = 2


class CandleBuffer:
    """
    Accumulates price ticks and emits a completed OHLCV candle dict
    whenever the timeframe boundary is crossed.
    """

    def __init__(self, symbol: str, timeframe: str, duration_seconds: int) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.duration = duration_seconds
        self._reset()

    def _reset(self) -> None:
        self.open: Optional[float] = None
        self.high: Optional[float] = None
        self.low: Optional[float] = None
        self.close: Optional[float] = None
        self.volume: int = 0
        self.candle_start: Optional[int] = None  # epoch seconds

    def _candle_open_time(self, ts: int) -> int:
        """Return the epoch second of the candle period that contains ts."""
        return (ts // self.duration) * self.duration

    def update(self, price: float, timestamp: int) -> Optional[Dict]:
        """
        Feed a tick into the buffer.

        Args:
            price: The latest price.
            timestamp: Unix epoch seconds for this tick.

        Returns:
            A completed candle dict if the candle period has closed,
            otherwise None.
        """
        candle_open = self._candle_open_time(timestamp)

        if self.candle_start is None:
            # First tick ever — start a new candle
            self.candle_start = candle_open
            self.open = price
            self.high = price
            self.low = price
            self.close = price
            self.volume = 1
            return None

        if candle_open == self.candle_start:
            # Same candle — update OHLCV
            self.high = max(self.high, price)
            self.low = min(self.low, price)
            self.close = price
            self.volume += 1
            return None

        # New candle period — emit the completed candle
        completed = self._build_candle()
        # Start the new candle
        self.candle_start = candle_open
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = 1
        return completed

    def _build_candle(self) -> Dict:
        ts_str = datetime.fromtimestamp(
            self.candle_start, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "timestamp": ts_str,
        }


class PriceFeed:
    """
    Connects to TwelveData WebSocket, subscribes to symbols from config,
    builds per-symbol per-timeframe OHLCV candle buffers, and invokes
    an async callback whenever a candle is completed.

    Usage::

        async def on_candle(candle: dict):
            print(candle)

        feed = PriceFeed(api_key="...", symbols=[...], timeframes=[...],
                         on_candle=on_candle)
        await feed.run()
    """

    def __init__(
        self,
        api_key: str,
        symbols: List[str],
        timeframes: List[str],
        on_candle: Optional[Callable[[Dict], Awaitable[None]]] = None,
    ) -> None:
        self.api_key = api_key
        self.symbols = symbols
        self.timeframes = [tf for tf in timeframes if tf in TIMEFRAME_SECONDS]
        self.on_candle = on_candle
        self._running = False
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

        # Build candle buffers: {symbol: {timeframe: CandleBuffer}}
        self._buffers: Dict[str, Dict[str, CandleBuffer]] = {
            symbol: {
                tf: CandleBuffer(symbol, tf, TIMEFRAME_SECONDS[tf])
                for tf in self.timeframes
            }
            for symbol in self.symbols
        }

        unknown = set(timeframes) - set(TIMEFRAME_SECONDS)
        if unknown:
            logger.warning("Unknown timeframes ignored: %s", unknown)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Start the price feed. Runs until cancelled.
        Reconnects automatically on disconnect with exponential backoff.
        """
        self._running = True
        backoff = BACKOFF_INITIAL
        while self._running:
            try:
                await self._connect_and_stream()
                backoff = BACKOFF_INITIAL  # reset on clean exit
            except asyncio.CancelledError:
                logger.info("PriceFeed cancelled — shutting down.")
                self._running = False
                break
            except Exception as exc:
                logger.error("PriceFeed error: %s", exc)

            if self._running:
                logger.info("Reconnecting in %ds...", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * BACKOFF_FACTOR, BACKOFF_MAX)

    async def stop(self) -> None:
        """Gracefully stop the price feed."""
        self._running = False
        if self._ws is not None:
            await self._ws.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _connect_and_stream(self) -> None:
        url = f"{TWELVEDATA_WS_URL}?apikey={self.api_key}"
        # TwelveData only supports API key authentication via URL query param.
        # The key is not logged — only the base URL is included in log messages.
        logger.info("Connecting to TwelveData WebSocket (%s)...", TWELVEDATA_WS_URL)
        async with websockets.connect(url) as ws:
            self._ws = ws
            logger.info("Connected. Subscribing to %d symbol(s)...", len(self.symbols))
            await self._subscribe(ws)
            async for raw_msg in ws:
                if not self._running:
                    break
                await self._handle_message(raw_msg)

    async def _subscribe(self, ws) -> None:
        payload = {
            "action": "subscribe",
            "params": {
                "symbols": ",".join(self.symbols),
            },
        }
        await ws.send(json.dumps(payload))
        logger.info("Subscription sent for: %s", self.symbols)

    async def _handle_message(self, raw_msg: str) -> None:
        try:
            msg = json.loads(raw_msg)
        except json.JSONDecodeError:
            logger.warning("Received non-JSON message: %s", raw_msg)
            return

        event = msg.get("event")

        if event == "price":
            symbol = msg.get("symbol")
            price_str = msg.get("price")
            ts_str = msg.get("timestamp")

            if not symbol or price_str is None or ts_str is None:
                return

            try:
                price = float(price_str)
                # TwelveData sends timestamp as Unix seconds (int) or ISO string
                if isinstance(ts_str, (int, float)):
                    ts = int(ts_str)
                else:
                    ts = int(
                        datetime.fromisoformat(
                            ts_str.replace("Z", "+00:00")
                        ).timestamp()
                    )
            except (ValueError, TypeError) as exc:
                logger.warning("Could not parse tick for %s: %s", symbol, exc)
                return

            await self._process_tick(symbol, price, ts)

        elif event == "subscribe-status":
            status = msg.get("status")
            logger.info("Subscription status: %s", status)
            if status == "error":
                logger.error("Subscription error: %s", msg.get("message"))

        elif event == "heartbeat":
            logger.debug("Heartbeat received.")

        else:
            logger.debug("Unhandled event '%s': %s", event, msg)

    async def _process_tick(self, symbol: str, price: float, timestamp: int) -> None:
        if symbol not in self._buffers:
            logger.debug("Tick for untracked symbol: %s", symbol)
            return

        for tf, buf in self._buffers[symbol].items():
            candle = buf.update(price, timestamp)
            if candle is not None:
                logger.info(
                    "Candle closed — %s %s O:%.5f H:%.5f L:%.5f C:%.5f V:%d @ %s",
                    symbol,
                    tf,
                    candle["open"],
                    candle["high"],
                    candle["low"],
                    candle["close"],
                    candle["volume"],
                    candle["timestamp"],
                )
                if self.on_candle is not None:
                    try:
                        await self.on_candle(candle)
                    except Exception as exc:
                        logger.error("on_candle callback raised: %s", exc)
