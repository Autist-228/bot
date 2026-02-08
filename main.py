import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SCAN_INTERVAL_SECONDS
from utils.database import init_db, get_signal_stats, get_active_tracked_tokens
from modules.signal_engine import scan_and_score
from modules.telegram_bot import (
    send_message,
    format_signal_alert,
    format_stats_message,
    format_tracked_tokens,
)
from modules.price_tracker import update_all_prices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("main")


async def run_scan_cycle():
    try:
        signals = await scan_and_score()
        if signals:
            for sig in signals:
                msg = format_signal_alert(sig)
                await send_message(msg)
                await asyncio.sleep(1)
            logger.info("Sent %d signal alerts", len(signals))
        else:
            logger.info("No new signals this cycle")
    except Exception as e:
        logger.error("Scan cycle error: %s", e, exc_info=True)


async def run_price_update():
    try:
        await update_all_prices()
    except Exception as e:
        logger.error("Price update error: %s", e, exc_info=True)


async def handle_commands():
    from httpx import AsyncClient
    offset = 0
    api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

    async with AsyncClient(timeout=30) as client:
        while True:
            try:
                resp = await client.get(
                    f"{api_url}/getUpdates",
                    params={"offset": offset, "timeout": 5},
                )
                data = resp.json()
                updates = data.get("result", [])

                for update in updates:
                    offset = update["update_id"] + 1
                    message = update.get("message", {})
                    text = message.get("text", "")
                    chat_id = str(message.get("chat", {}).get("id", ""))

                    if chat_id != TELEGRAM_CHAT_ID:
                        continue

                    if text == "/start":
                        await send_message(
                            "<b>Solana Smart Money Bot</b>\n\n"
                            "Commands:\n"
                            "/scan - Manual scan\n"
                            "/stats - Signal statistics\n"
                            "/tracked - Active tracked tokens\n"
                            "/help - Help",
                            chat_id=chat_id,
                        )
                    elif text == "/scan":
                        await send_message("Scanning...", chat_id=chat_id)
                        await run_scan_cycle()
                    elif text == "/stats":
                        stats = await get_signal_stats()
                        await send_message(
                            format_stats_message(stats), chat_id=chat_id
                        )
                    elif text == "/tracked":
                        tokens = await get_active_tracked_tokens()
                        await send_message(
                            format_tracked_tokens(tokens), chat_id=chat_id
                        )
                    elif text == "/help":
                        await send_message(
                            "<b>Commands:</b>\n"
                            "/scan - Run manual scan now\n"
                            "/stats - Show signal statistics\n"
                            "/tracked - Show active tracked tokens\n"
                            "/help - This message\n\n"
                            "Bot automatically scans every "
                            f"{SCAN_INTERVAL_SECONDS}s and sends alerts "
                            "when strong signals are detected.",
                            chat_id=chat_id,
                        )
            except Exception as e:
                logger.error("Command handler error: %s", e)
                await asyncio.sleep(5)


async def scheduler_loop():
    cycle = 0
    while True:
        cycle += 1
        logger.info("=== Scan cycle #%d ===", cycle)
        await run_scan_cycle()

        if cycle % 5 == 0:
            await run_price_update()

        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


async def main():
    logger.info("Initializing database...")
    await init_db()

    logger.info("Starting Solana Smart Money Bot...")
    await send_message(
        "<b>Bot started!</b>\n"
        f"Scan interval: {SCAN_INTERVAL_SECONDS}s\n"
        "Send /help for commands."
    )

    await asyncio.gather(
        scheduler_loop(),
        handle_commands(),
    )


if __name__ == "__main__":
    asyncio.run(main())
