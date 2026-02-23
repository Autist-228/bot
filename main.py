#!/usr/bin/env python3
"""
Entry point: runs Sniper + Telegram bot together.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from logging.handlers import RotatingFileHandler

from config import BET_SIZE_USD, DATA_DIR, MAX_CONCURRENT, TELEGRAM_BOT_TOKEN
from sniper import Sniper


def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    log_path = os.path.join(DATA_DIR, "bot.log")
    fh = RotatingFileHandler(log_path, maxBytes=50 * 1024 * 1024, backupCount=3)
    fh.setFormatter(fmt)
    root.addHandler(fh)


_setup_logging()
log = logging.getLogger("main")

_shutdown_event: asyncio.Event | None = None


def _request_shutdown() -> None:
    if _shutdown_event:
        _shutdown_event.set()


async def run() -> None:
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    sniper = Sniper()

    tg_app = None
    if TELEGRAM_BOT_TOKEN:
        from telegram_bot import build_app

        tg_app = build_app(sniper)
        log.info("Telegram bot enabled")
    else:
        log.warning("TELEGRAM_BOT_TOKEN not set, running without Telegram")

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_shutdown)

    sniper._tasks = [
        asyncio.create_task(sniper._ws_listener()),
        asyncio.create_task(sniper._position_checker()),
        asyncio.create_task(sniper._stats_printer()),
        asyncio.create_task(sniper._sol_price_updater()),
        asyncio.create_task(sniper._memory_cleaner()),
    ]

    log.info(
        "Maximum Sniper starting | balance=$%.2f | bet=$%.2f | max_concurrent=%d",
        sniper.paper_balance,
        BET_SIZE_USD,
        MAX_CONCURRENT,
    )

    if tg_app:
        await tg_app.initialize()
        await tg_app.start()
        if tg_app.updater:
            await tg_app.updater.start_polling(drop_pending_updates=True)
        log.info("Telegram polling started")

    await _shutdown_event.wait()
    log.info("Shutdown requested")

    sniper.stop()
    for t in sniper._tasks:
        try:
            await asyncio.wait_for(t, timeout=5)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    if tg_app:
        if tg_app.updater and tg_app.updater.running:
            await tg_app.updater.stop()
        if tg_app.running:
            await tg_app.stop()
        await tg_app.shutdown()

    sniper._save_state()
    sniper._save_blacklist()
    log.info(
        "Final: bal=$%.2f | pnl=$%+.2f | trades=%d | wins=%d",
        sniper.paper_balance,
        sniper.total_pnl,
        sniper.total_trades,
        sniper.total_wins,
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
