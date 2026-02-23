from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from config import BET_SIZE_USD, STARTING_BALANCE, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

if TYPE_CHECKING:
    from sniper import Sniper

log = logging.getLogger("telegram")


def _fmt_usd(v: float) -> str:
    return f"{v:+.2f}"


def _build_main_screen(bot: Sniper) -> str:
    uptime = time.time() - bot.start_time
    hrs = uptime / 3600
    wr = (bot.total_wins / bot.total_trades * 100) if bot.total_trades > 0 else 0.0
    per_day = (bot.total_pnl / hrs * 24) if hrs > 0.01 else 0.0
    open_pos = bot._open_count()
    pending = sum(1 for p in bot.positions.values() if p.phase == "pending")
    confirmed = sum(1 for p in bot.positions.values() if p.phase == "confirmed")

    status_emoji = "\U0001f7e2" if bot._running else "\U0001f534"
    bal_emoji = "\U0001f4b0"
    chart_emoji = "\U0001f4c8" if bot.total_pnl >= 0 else "\U0001f4c9"
    trades_emoji = "\U0001f3af"
    clock_emoji = "\u23f0"
    fire_emoji = "\U0001f525"
    folder_emoji = "\U0001f4c2"
    shield_emoji = "\U0001f6e1"

    if hrs < 1:
        uptime_str = f"{int(uptime // 60)}m"
    else:
        uptime_str = f"{hrs:.1f}h"

    lines = [
        f"{status_emoji} *MAXIMUM SNIPER* {'`RUNNING`' if bot._running else '`STOPPED`'}",
        "",
        f"{bal_emoji} *Paper Balance:* `${bot.paper_balance:.2f}`",
        f"{chart_emoji} *PnL:* `${_fmt_usd(bot.total_pnl)}` ({_fmt_usd(bot.total_pnl / max(STARTING_BALANCE, 0.01) * 100)}%)",
        "",
        f"{trades_emoji} *Trades:* {bot.total_trades} (W:{bot.total_wins} / L:{bot.total_trades - bot.total_wins})",
        f"{fire_emoji} *Win Rate:* {wr:.1f}%",
        f"{chart_emoji} *$/day:* `${per_day:.2f}`",
        "",
        f"{folder_emoji} *Open:* {open_pos} (P:{pending} C:{confirmed})",
        f"{shield_emoji} *Bet:* ${BET_SIZE_USD:.2f} | *Blacklist:* {len(bot.dev_blacklist)}",
        f"{clock_emoji} *Uptime:* {uptime_str} | *Seen:* {bot.tokens_seen}",
    ]

    if bot.total_trades > 0:
        lines.append("")
        lines.append(f"\U0001f3c6 *Best:* `${bot.best_trade:+.4f}` | *Worst:* `${bot.worst_trade:.4f}`")

    lines.append("")
    lines.append("\u26a0\ufe0f _Paper trading mode_")

    return "\n".join(lines)


def _main_keyboard(bot: Sniper) -> InlineKeyboardMarkup:
    if bot._running:
        toggle = InlineKeyboardButton("\u23f8 Stop", callback_data="stop")
    else:
        toggle = InlineKeyboardButton("\u25b6\ufe0f Start", callback_data="start")
    return InlineKeyboardMarkup([
        [toggle, InlineKeyboardButton("\U0001f504 Refresh", callback_data="refresh")],
        [InlineKeyboardButton("\U0001f4b0 Set Balance", callback_data="balance_info")],
    ])


async def _send_main(update: Update | None, context: ContextTypes.DEFAULT_TYPE, bot: Sniper) -> None:
    text = _build_main_screen(bot)
    kb = _main_keyboard(bot)
    if update and update.message:
        await update.message.reply_text(text, parse_mode="Markdown", reply_markup=kb)
    elif update and update.callback_query:
        try:
            await update.callback_query.edit_message_text(text, parse_mode="Markdown", reply_markup=kb)
        except Exception:
            pass


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot: Sniper = context.bot_data["sniper"]
    await _send_main(update, context, bot)


async def cb_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    await query.answer()
    bot: Sniper = context.bot_data["sniper"]
    data = query.data

    if data == "refresh":
        await _send_main(update, context, bot)
    elif data == "stop":
        bot.stop()
        await _send_main(update, context, bot)
    elif data == "start":
        if not bot._running:
            bot._running = True
            loop = asyncio.get_running_loop()
            bot._tasks = [
                loop.create_task(bot._ws_listener()),
                loop.create_task(bot._position_checker()),
                loop.create_task(bot._stats_printer()),
                loop.create_task(bot._sol_price_updater()),
                loop.create_task(bot._memory_cleaner()),
            ]
        await _send_main(update, context, bot)
    elif data == "balance_info":
        text = (
            "\U0001f4b0 *Set Balance*\n\n"
            f"Current: `${bot.paper_balance:.2f}`\n\n"
            "Send: `/balance 100` to set $100"
        )
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("\u25c0 Back", callback_data="refresh")],
        ])
        try:
            await query.edit_message_text(text, parse_mode="Markdown", reply_markup=kb)
        except Exception:
            pass


async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot: Sniper = context.bot_data["sniper"]
    if context.args:
        try:
            new_bal = float(context.args[0])
            if new_bal > 0:
                bot.paper_balance = new_bal
                bot.reserved = 0.0
                await update.message.reply_text(f"\U0001f4b0 Balance set to `${new_bal:.2f}`", parse_mode="Markdown")
                return
        except ValueError:
            pass
    await update.message.reply_text(
        f"\U0001f4b0 Current balance: `${bot.paper_balance:.2f}`\n\nUsage: `/balance 100`",
        parse_mode="Markdown",
    )


async def _auto_update(context: ContextTypes.DEFAULT_TYPE) -> None:
    bot: Sniper = context.bot_data["sniper"]
    chat_id = context.bot_data.get("chat_id", "")
    if not chat_id:
        return
    msg_id = context.bot_data.get("main_msg_id")
    text = _build_main_screen(bot)
    kb = _main_keyboard(bot)
    try:
        if msg_id:
            await context.bot.edit_message_text(
                text, chat_id=chat_id, message_id=msg_id,
                parse_mode="Markdown", reply_markup=kb,
            )
        else:
            msg = await context.bot.send_message(
                chat_id, text, parse_mode="Markdown", reply_markup=kb,
            )
            context.bot_data["main_msg_id"] = msg.message_id
    except Exception as exc:
        if "message is not modified" not in str(exc):
            log.warning("Auto-update failed: %s", exc)


async def _post_init(app: Application) -> None:
    chat_id = app.bot_data.get("chat_id", "")
    if chat_id:
        bot: Sniper = app.bot_data["sniper"]
        text = _build_main_screen(bot)
        kb = _main_keyboard(bot)
        try:
            msg = await app.bot.send_message(
                chat_id, text, parse_mode="Markdown", reply_markup=kb,
            )
            app.bot_data["main_msg_id"] = msg.message_id
        except Exception as e:
            log.warning("Failed to send startup message: %s", e)


def build_app(sniper: Sniper) -> Application:
    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(_post_init)
        .build()
    )
    app.bot_data["sniper"] = sniper
    app.bot_data["chat_id"] = TELEGRAM_CHAT_ID

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("balance", cmd_balance))
    app.add_handler(CallbackQueryHandler(cb_handler))

    app.job_queue.run_repeating(_auto_update, interval=60, first=60)

    return app
