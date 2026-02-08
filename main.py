import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    SCAN_INTERVAL_SECONDS,
    WALLET_PRIVATE_KEY,
    DRY_RUN,
    BANK_SOL,
    POSITION_CHECK_INTERVAL,
)
from utils.database import init_db, get_signal_stats, get_active_tracked_tokens
from modules.signal_engine import scan_and_score
from modules.telegram_bot import (
    send_message,
    format_signal_alert,
    format_stats_message,
    format_tracked_tokens,
    format_trade_open,
    format_trade_close,
    format_portfolio,
)
from modules.price_tracker import update_all_prices
from modules.trader import JupiterTrader
from modules.position_manager import PositionManager
from modules.watchlist import Watchlist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("main")

trader = JupiterTrader(private_key=WALLET_PRIVATE_KEY, dry_run=DRY_RUN)
pm = PositionManager(trader=trader, bank_sol=BANK_SOL)
watchlist = Watchlist()


async def run_scan_cycle():
    try:
        signals = await scan_and_score()
        if signals:
            for sig in signals:
                msg = format_signal_alert(sig)
                await send_message(msg)
                await asyncio.sleep(0.3)
                await watchlist.add(sig)
            logger.info("Sent %d signals to watchlist", len(signals))
        else:
            logger.info("No new signals this cycle")
    except Exception as e:
        logger.error("Scan cycle error: %s", e, exc_info=True)


async def run_watchlist_check():
    try:
        confirmed = await watchlist.check()
        for sig in confirmed:
            wdata = sig.get("watchlist_data", {})
            await send_message(
                f"<b>CONFIRMED: ${sig['token']['symbol']}</b>\n"
                f"Watched {wdata.get('watch_time_sec', 0)}s, grew +{wdata.get('growth_during_watch', 0):.1f}%\n"
                f"Buying now!"
            )
            pos = await pm.open_position(sig)
            if pos:
                await send_message(format_trade_open(pos.to_dict()))
    except Exception as e:
        logger.error("Watchlist check error: %s", e, exc_info=True)


async def run_position_check():
    try:
        old_closed = len(pm.closed_positions)
        await pm.check_positions()
        new_closed = pm.closed_positions[old_closed:]
        for c in new_closed:
            await send_message(format_trade_close(c))
            await asyncio.sleep(0.3)
    except Exception as e:
        logger.error("Position check error: %s", e, exc_info=True)


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
                        mode = "DRY RUN" if DRY_RUN else "LIVE"
                        await send_message(
                            f"<b>Solana Smart Money Bot [{mode}]</b>\n\n"
                            "Commands:\n"
                            "/scan - Manual scan\n"
                            "/stats - Signal statistics\n"
                            "/portfolio - Trading portfolio\n"
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
                    elif text == "/portfolio":
                        stats = pm.get_stats()
                        await send_message(
                            format_portfolio(stats), chat_id=chat_id
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
                            "/portfolio - Trading portfolio & PnL\n"
                            "/tracked - Show active tracked tokens\n"
                            "/help - This message\n\n"
                            "Bot automatically scans every "
                            f"{SCAN_INTERVAL_SECONDS}s and sends alerts "
                            "when strong signals are detected.\n"
                            f"Trading mode: <b>{'DRY RUN' if DRY_RUN else 'LIVE'}</b>",
                            chat_id=chat_id,
                        )
            except Exception as e:
                logger.error("Command handler error: %s", e)
                await asyncio.sleep(5)


async def watchlist_position_loop():
    while True:
        await run_watchlist_check()
        await run_position_check()
        await asyncio.sleep(POSITION_CHECK_INTERVAL)


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

    mode = "DRY RUN" if DRY_RUN else "LIVE"
    logger.info("Starting Solana Smart Money Bot [%s]...", mode)
    await send_message(
        f"<b>Bot started! [{mode}]</b>\n"
        f"Bank: {BANK_SOL} SOL\n"
        f"Scan interval: {SCAN_INTERVAL_SECONDS}s\n"
        "Send /help for commands."
    )

    await asyncio.gather(
        scheduler_loop(),
        watchlist_position_loop(),
        handle_commands(),
    )


if __name__ == "__main__":
    asyncio.run(main())
