"""
FX-Leopard 🐆
AI-powered FX market surveillance agent.
Entry point — starts all engines and runs the agent loop.
"""

import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("fx-leopard")


async def main():
    logger.info("🐆 FX-Leopard starting up...")
    # Components will be wired here as they are built
    # See GitHub Issues for implementation tasks
    logger.info("All engines running. Watching the market... 🐆")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
"""
