import asyncio
import logging
import os
import time
from datetime import datetime, timezone

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)
from telegram.constants import ParseMode

log = logging.getLogger("telegram_bot")

PERIODS = {
    "30m": (1800, "30 мин"),
    "1h": (3600, "1 час"),
    "3h": (10800, "3 часа"),
    "6h": (21600, "6 часов"),
    "12h": (43200, "12 часов"),
    "24h": (86400, "24 часа"),
}


class SniperTelegramBot:
    def __init__(self, token: str, chat_id: int, state: dict):
        self.token = token
        self.chat_id = chat_id
        self.state = state
        self.app = None
        self.current_period = "1h"
        self.session_active = False
        self.session_start = 0.0
        self.session_signals_snapshot = 0
        self.main_message_id = None
        self.session_message_id = None
        self._update_task = None
        self._current_screen = "main"

    def _get_signals(self):
        return self.state.get("signals", [])

    def _get_tokens(self):
        return self.state.get("tokens", {})

    def _get_stats(self):
        return self.state.get("stats", {})

    def _get_errors(self):
        return self.state.get("errors", [])

    def _get_model_info(self):
        return self.state.get("model_info", {})

    def _format_uptime(self):
        start = self._get_stats().get("start", 0)
        if start == 0:
            return "0 сек"
        elapsed = time.time() - start
        days = int(elapsed // 86400)
        hours = int((elapsed % 86400) // 3600)
        mins = int((elapsed % 3600) // 60)
        parts = []
        if days > 0:
            parts.append(f"{days}д")
        if hours > 0:
            parts.append(f"{hours}ч")
        parts.append(f"{mins}мин")
        return " ".join(parts)

    def _signals_for_period(self):
        period_sec = PERIODS[self.current_period][0]
        cutoff = time.time() - period_sec
        return [s for s in self._get_signals() if s.get("signal_time", 0) >= cutoff]

    def _session_signals(self):
        if not self.session_active:
            return []
        return [s for s in self._get_signals()
                if s.get("signal_time", 0) >= self.session_start]

    def _calc_pnl(self, sigs):
        total_pnl = 0.0
        total_invested = 0
        wins = 0
        losses = 0
        bet = self.state.get("bet_size", 3.0)
        for s in sigs:
            if s.get("status") == "CLOSED":
                cpnl = s.get("close_pnl_pct", 0) or 0
                total_pnl += bet * cpnl / 100
                total_invested += 1
                if cpnl > 0:
                    wins += 1
                else:
                    losses += 1
            elif s.get("status") == "ACTIVE":
                pnl = s.get("pnl_pct")
                if pnl is not None:
                    total_pnl += bet * pnl / 100
                total_invested += 1
        return {
            "pnl_usd": total_pnl,
            "invested": total_invested * bet,
            "invested_count": total_invested,
            "wins": wins,
            "losses": losses,
            "pnl_pct": (total_pnl / (total_invested * bet) * 100) if total_invested > 0 else 0,
        }

    def _build_main_text(self):
        stats = self._get_stats()
        model = self._get_model_info()
        errors = self._get_errors()
        period_sigs = self._signals_for_period()
        period_name = PERIODS[self.current_period][1]
        pnl = self._calc_pnl(period_sigs)

        ws_ok = self.state.get("ws_connected", False)
        ml_ok = self.state.get("ml_running", False)
        learn_ok = self.state.get("learner_running", False)

        ws_icon = "\u2705" if ws_ok else "\u274c"
        ml_icon = "\u2705" if ml_ok else "\u274c"
        learn_icon = "\u2705" if learn_ok else "\u274c"

        err_count = len([e for e in errors if time.time() - e.get("ts", 0) < 86400])

        last_train_ago = ""
        lt = model.get("last_train_ts", 0)
        if lt > 0:
            ago = int(time.time() - lt)
            if ago < 60:
                last_train_ago = f"{ago} сек назад"
            else:
                last_train_ago = f"{ago // 60} мин назад"
        else:
            last_train_ago = "ещё не было"

        model_size = model.get("file_size_kb", 0)
        total_samples = model.get("total_samples", 0)
        cycles = model.get("cycles", 0)
        loss = model.get("loss", 0)
        rockets_found = model.get("rockets_found", 0)
        rockets_missed = model.get("rockets_missed", 0)

        wr = 0
        if pnl["wins"] + pnl["losses"] > 0:
            wr = pnl["wins"] / (pnl["wins"] + pnl["losses"]) * 100

        tokens_scanned = stats.get("total", 0)

        lines = [
            "\U0001f916 <b>СТАТУС БОТА: РАБОТАЕТ</b>",
            f"\u23f1 Аптайм: {self._format_uptime()}",
            f"\U0001f4ca Период: {period_name}",
            "",
            "\u2501\u2501\u2501 <b>МОДЕЛЬ</b> \u2501\u2501\u2501",
            f"\U0001f4c1 entry_model.pt ({model_size} KB)",
            f"\U0001f9e0 Циклов обучения: {cycles}",
            f"\U0001f4c9 Loss: {loss:.4f}" if loss > 0 else "\U0001f4c9 Loss: ---",
            f"\U0001f37d Данных сожрано: {total_samples:,}",
            f"\U0001f680 Ракет: {rockets_found} | Пропущено: {rockets_missed}",
            f"\U0001f4c5 Посл. обучение: {last_train_ago}",
            "",
            f"\u2501\u2501\u2501 <b>МОНИТОР ({period_name})</b> \u2501\u2501\u2501",
            f"\U0001f4e1 Сигналов: {pnl['invested_count']}",
            f"\U0001f4b0 Вложено: ${pnl['invested']:.2f}",
            f"\U0001f4ca P&L: ${pnl['pnl_usd']:+.2f} ({pnl['pnl_pct']:+.1f}%)",
            f"\U0001f3c6 Винрейт: {wr:.0f}% ({pnl['wins']}W / {pnl['losses']}L)",
            f"\U0001f50d Отсканировано: {tokens_scanned:,} токенов",
            "",
            "\u2501\u2501\u2501 <b>ЗДОРОВЬЕ</b> \u2501\u2501\u2501",
            f"{ws_icon} WebSocket | {ml_icon} ML сканер | {learn_icon} Обучение",
        ]

        if err_count > 0:
            lines.append(f"\u26a0\ufe0f Ошибки за 24ч: {err_count}")
        else:
            lines.append("\u2705 Ошибок нет")

        return "\n".join(lines)

    def _build_main_keyboard(self):
        errors = self._get_errors()
        err_count = len([e for e in errors if time.time() - e.get("ts", 0) < 86400])

        if self.session_active:
            row1 = [InlineKeyboardButton("\u23f9 Стоп сессии", callback_data="session_stop")]
        else:
            row1 = [InlineKeyboardButton("\u25b6\ufe0f Старт сессии", callback_data="session_start")]

        period_name = PERIODS[self.current_period][1]
        row2 = [
            InlineKeyboardButton("\U0001f4cb Все токены", callback_data="tokens_0"),
            InlineKeyboardButton("\U0001f9e0 Модель", callback_data="model"),
        ]
        row3 = [
            InlineKeyboardButton(f"\u23f0 Период: {period_name}", callback_data="period"),
        ]
        if err_count > 0:
            row3.append(InlineKeyboardButton(f"\u26a0\ufe0f Ошибки ({err_count})", callback_data="errors"))
        row3.append(InlineKeyboardButton("\U0001f504 Обновить", callback_data="refresh"))

        rows = [row1, row2, row3]
        if self.session_active:
            rows.insert(1, [InlineKeyboardButton("\U0001f4ca Сессия", callback_data="session_view")])

        return InlineKeyboardMarkup(rows)

    def _build_session_text(self):
        sigs = self._session_signals()
        if not sigs:
            elapsed = int(time.time() - self.session_start)
            return (
                f"\U0001f4ca <b>СЕССИЯ (активна {elapsed // 60} мин)</b>\n\n"
                "Сигналов пока нет. Ожидание..."
            )

        elapsed = int(time.time() - self.session_start)
        pnl = self._calc_pnl(sigs)
        wr = 0
        if pnl["wins"] + pnl["losses"] > 0:
            wr = pnl["wins"] / (pnl["wins"] + pnl["losses"]) * 100

        lines = [
            f"\U0001f4ca <b>СЕССИЯ (активна {elapsed // 60} мин)</b>",
            "",
            f"\U0001f4b0 Вложено: ${pnl['invested']:.2f}",
            f"\U0001f4ca P&L: ${pnl['pnl_usd']:+.2f} ({pnl['pnl_pct']:+.1f}%)",
            f"\U0001f3c6 Винрейт: {wr:.0f}% ({pnl['wins']}W / {pnl['losses']}L)",
            f"\U0001f4e1 Сигналов: {len(sigs)}",
            "",
            "\u2501\u2501\u2501 <b>ТОП-5 СИГНАЛОВ</b> \u2501\u2501\u2501",
        ]

        sorted_sigs = sorted(sigs, key=lambda s: s.get("close_pnl_pct") or s.get("pnl_pct") or 0, reverse=True)
        for i, s in enumerate(sorted_sigs[:5]):
            if s["status"] == "CLOSED":
                cpnl = s.get("close_pnl_pct", 0) or 0
                bet = self.state.get("bet_size", 3.0)
                usd = bet * cpnl / 100
                icon = "\U0001f7e2" if cpnl > 0 else "\U0001f534"
                reason = s.get("close_reason", "")
                lines.append(f"{i+1}. {icon} {s['symbol']} | {cpnl:+.1f}% (${usd:+.2f}) | {reason}")
            else:
                pnl_val = s.get("pnl_pct")
                conf = s.get("ml_confidence", 0)
                if pnl_val is not None:
                    icon = "\U0001f7e1"
                    lines.append(f"{i+1}. {icon} {s['symbol']} | {pnl_val:+.1f}% (открыт) | conf={conf:.0f}%")
                else:
                    lines.append(f"{i+1}. \U0001f7e1 {s['symbol']} | ожидание... | conf={conf:.0f}%")

        return "\n".join(lines)

    def _build_session_keyboard(self):
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("\U0001f4cb Все токены сессии", callback_data="session_tokens_0")],
            [
                InlineKeyboardButton("\u23f9 Стоп сессии", callback_data="session_stop"),
                InlineKeyboardButton("\U0001f3e0 Главная", callback_data="home"),
            ],
        ])

    def _build_tokens_text(self, page: int, session_only: bool = False):
        per_page = 4
        if session_only:
            sigs = self._session_signals()
            title = "ТОКЕНЫ СЕССИИ"
        else:
            sigs = self._signals_for_period()
            period_name = PERIODS[self.current_period][1]
            title = f"СИГНАЛЫ ({period_name})"

        sorted_sigs = sorted(sigs, key=lambda s: s.get("signal_time", 0), reverse=True)
        total_pages = max(1, (len(sorted_sigs) + per_page - 1) // per_page)
        page = max(0, min(page, total_pages - 1))
        start = page * per_page
        page_sigs = sorted_sigs[start:start + per_page]

        lines = [f"\U0001f4cb <b>{title}</b> (стр. {page + 1}/{total_pages})", ""]

        if not page_sigs:
            lines.append("Сигналов нет.")
        else:
            bet = self.state.get("bet_size", 3.0)
            for i, s in enumerate(page_sigs):
                idx = start + i + 1
                sym = s.get("symbol", "???")
                conf = s.get("ml_confidence", 0)
                sig_time = s.get("signal_time", 0)
                time_str = datetime.fromtimestamp(sig_time, tz=timezone.utc).strftime("%H:%M:%S") if sig_time else "---"

                if s["status"] == "CLOSED":
                    cpnl = s.get("close_pnl_pct", 0) or 0
                    usd = bet * cpnl / 100
                    icon = "\U0001f7e2" if cpnl > 0 else "\U0001f534"
                    reason = s.get("close_reason", "")
                    lines.append(f"{idx}. {sym} | {icon} {cpnl:+.1f}% (${usd:+.2f})")
                    lines.append(f"   conf={conf:.0f}% | {reason} | {time_str}")
                else:
                    pnl_val = s.get("pnl_pct")
                    if pnl_val is not None:
                        usd = bet * pnl_val / 100
                        lines.append(f"{idx}. {sym} | \U0001f7e1 {pnl_val:+.1f}% (${usd:+.2f})")
                    else:
                        lines.append(f"{idx}. {sym} | \U0001f7e1 ожидание...")
                    lines.append(f"   conf={conf:.0f}% | открыт | {time_str}")
                lines.append("")

        return "\n".join(lines), page, total_pages

    def _build_tokens_keyboard(self, page: int, total_pages: int, session_only: bool = False):
        prefix = "session_tokens" if session_only else "tokens"
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton("\u25c0\ufe0f Назад", callback_data=f"{prefix}_{page - 1}"))
        nav.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="noop"))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton("Вперёд \u25b6\ufe0f", callback_data=f"{prefix}_{page + 1}"))
        return InlineKeyboardMarkup([
            nav,
            [InlineKeyboardButton("\U0001f3e0 Главная", callback_data="home")],
        ])

    def _build_model_text(self):
        model = self._get_model_info()
        model_size = model.get("file_size_kb", 0)
        last_save = model.get("last_save_time", "---")
        cycles = model.get("cycles", 0)
        loss = model.get("loss", 0)
        initial_loss = model.get("initial_loss", 0)
        total_samples = model.get("total_samples", 0)
        total_wins = model.get("total_wins", 0)
        rockets_missed = model.get("rockets_missed", 0)
        avg_missed_pnl = model.get("avg_missed_pnl", 0)

        win_pct = (total_wins / total_samples * 100) if total_samples > 0 else 0

        lines = [
            "\U0001f9e0 <b>СТАТИСТИКА МОДЕЛИ</b>",
            "",
            f"\U0001f4c1 Файл: entry_model.pt",
            f"\U0001f4be Размер: {model_size} KB",
            f"\U0001f4c5 Посл. сохранение: {last_save}",
            "",
            "\u2501\u2501\u2501 <b>ОБУЧЕНИЕ</b> \u2501\u2501\u2501",
            f"\U0001f504 Циклов: {cycles}",
            f"\U0001f4c9 Loss: {loss:.4f}" if loss > 0 else "\U0001f4c9 Loss: ---",
            f"\U0001f4c8 Начальный loss: {initial_loss:.4f}" if initial_loss > 0 else "",
            f"\U0001f4ca Samples за всё время: {total_samples:,}",
            f"\U0001f3c6 Из них побед: {total_wins:,} ({win_pct:.0f}%)",
            "",
            "\u2501\u2501\u2501 <b>ПРОПУЩЕННЫЕ РАКЕТЫ</b> \u2501\u2501\u2501",
            f"\U0001f680 Найдено за всё время: {rockets_missed}",
            f"\U0001f4c8 Средний рост: +{avg_missed_pnl:.0f}%" if avg_missed_pnl > 0 else "",
            "\U0001f37d Все скормлены в модель",
            "",
            "\u2501\u2501\u2501 <b>ВОЗНАГРАЖДЕНИЯ</b> \u2501\u2501\u2501",
            "\u2265200%: вес 10.0",
            "\u2265100%: вес 7.0",
            "\u226550%: вес 5.0",
            "\u226520%: вес 3.0",
            "\u22655%: вес 2.0",
            "\u2264-10%: вес 2.0",
            "остальное: вес 1.0",
        ]
        return "\n".join([l for l in lines if l is not None])

    def _build_model_keyboard(self):
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("\U0001f3e0 Главная", callback_data="home")],
        ])

    def _build_errors_text(self):
        errors = self._get_errors()
        recent = [e for e in errors if time.time() - e.get("ts", 0) < 86400]
        recent = sorted(recent, key=lambda e: e["ts"], reverse=True)[:20]

        lines = ["\u26a0\ufe0f <b>ПОСЛЕДНИЕ ОШИБКИ</b>", ""]

        if not recent:
            lines.append("\u2705 Ошибок за последние 24ч нет!")
        else:
            for e in recent:
                t = datetime.fromtimestamp(e["ts"], tz=timezone.utc).strftime("%H:%M:%S")
                msg = e.get("msg", "???")
                lines.append(f"{t} \u2014 {msg}")

            lines.append(f"\nВсего за 24ч: {len(recent)}")

        return "\n".join(lines)

    def _build_errors_keyboard(self):
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("\U0001f3e0 Главная", callback_data="home")],
        ])

    def _build_period_text(self):
        return "\u23f0 <b>Выберите период для главного экрана:</b>"

    def _build_period_keyboard(self):
        row1 = [
            InlineKeyboardButton("30 мин", callback_data="set_period_30m"),
            InlineKeyboardButton("1 час", callback_data="set_period_1h"),
            InlineKeyboardButton("3 часа", callback_data="set_period_3h"),
        ]
        row2 = [
            InlineKeyboardButton("6 часов", callback_data="set_period_6h"),
            InlineKeyboardButton("12 часов", callback_data="set_period_12h"),
            InlineKeyboardButton("24 часа", callback_data="set_period_24h"),
        ]
        row3 = [InlineKeyboardButton("\U0001f3e0 Главная", callback_data="home")]
        return InlineKeyboardMarkup([row1, row2, row3])

    async def _send_or_edit(self, chat_id, text, keyboard, message_id=None):
        try:
            if message_id:
                try:
                    await self.app.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=text,
                        reply_markup=keyboard,
                        parse_mode=ParseMode.HTML,
                    )
                    return message_id
                except Exception:
                    pass
            msg = await self.app.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=keyboard,
                parse_mode=ParseMode.HTML,
            )
            return msg.message_id
        except Exception as exc:
            log.warning("Telegram send error: %s", exc)
            return message_id

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = self._build_main_text()
        kb = self._build_main_keyboard()
        msg = await update.message.reply_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
        self.main_message_id = msg.message_id

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = self._build_main_text()
        kb = self._build_main_keyboard()
        msg = await update.message.reply_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
        self.main_message_id = msg.message_id

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data
        chat_id = query.message.chat_id
        msg_id = query.message.message_id

        if data == "home" or data == "refresh":
            self._current_screen = "main"
            text = self._build_main_text()
            kb = self._build_main_keyboard()
            self.main_message_id = await self._send_or_edit(chat_id, text, kb, msg_id)

        elif data == "session_start":
            self._current_screen = "session"
            self.session_active = True
            self.session_start = time.time()
            self.session_signals_snapshot = len(self._get_signals())
            text = self._build_session_text()
            kb = self._build_session_keyboard()
            self.session_message_id = await self._send_or_edit(chat_id, text, kb, msg_id)

        elif data == "session_stop":
            self._current_screen = "main"
            self.session_active = False
            text = self._build_main_text()
            kb = self._build_main_keyboard()
            self.main_message_id = await self._send_or_edit(chat_id, text, kb, msg_id)

        elif data == "session_view":
            self._current_screen = "session"
            text = self._build_session_text()
            kb = self._build_session_keyboard()
            self.session_message_id = await self._send_or_edit(chat_id, text, kb, msg_id)

        elif data.startswith("tokens_"):
            self._current_screen = "tokens"
            page = int(data.split("_")[1])
            text, page, total = self._build_tokens_text(page, session_only=False)
            kb = self._build_tokens_keyboard(page, total, session_only=False)
            await self._send_or_edit(chat_id, text, kb, msg_id)

        elif data.startswith("session_tokens_"):
            self._current_screen = "tokens"
            page = int(data.split("_")[2])
            text, page, total = self._build_tokens_text(page, session_only=True)
            kb = self._build_tokens_keyboard(page, total, session_only=True)
            await self._send_or_edit(chat_id, text, kb, msg_id)

        elif data == "model":
            self._current_screen = "model"
            text = self._build_model_text()
            kb = self._build_model_keyboard()
            await self._send_or_edit(chat_id, text, kb, msg_id)

        elif data == "errors":
            self._current_screen = "errors"
            text = self._build_errors_text()
            kb = self._build_errors_keyboard()
            await self._send_or_edit(chat_id, text, kb, msg_id)

        elif data == "period":
            self._current_screen = "period"
            text = self._build_period_text()
            kb = self._build_period_keyboard()
            await self._send_or_edit(chat_id, text, kb, msg_id)

        elif data.startswith("set_period_"):
            self._current_screen = "main"
            period_key = data.replace("set_period_", "")
            if period_key in PERIODS:
                self.current_period = period_key
            text = self._build_main_text()
            kb = self._build_main_keyboard()
            self.main_message_id = await self._send_or_edit(chat_id, text, kb, msg_id)

        elif data == "noop":
            pass

    async def _auto_update_loop(self):
        while True:
            await asyncio.sleep(10)
            try:
                if self._current_screen == "main" and self.main_message_id and self.chat_id:
                    text = self._build_main_text()
                    kb = self._build_main_keyboard()
                    try:
                        await self.app.bot.edit_message_text(
                            chat_id=self.chat_id,
                            message_id=self.main_message_id,
                            text=text,
                            reply_markup=kb,
                            parse_mode=ParseMode.HTML,
                        )
                    except Exception:
                        pass

                if self._current_screen == "session" and self.session_active and self.session_message_id and self.chat_id:
                    text = self._build_session_text()
                    kb = self._build_session_keyboard()
                    try:
                        await self.app.bot.edit_message_text(
                            chat_id=self.chat_id,
                            message_id=self.session_message_id,
                            text=text,
                            reply_markup=kb,
                            parse_mode=ParseMode.HTML,
                        )
                    except Exception:
                        pass
            except Exception as exc:
                log.warning("Auto-update error: %s", exc)

    async def start(self):
        if not self.token:
            log.warning("TELEGRAM_BOT_TOKEN not set, skipping Telegram bot")
            return

        self.app = Application.builder().token(self.token).build()
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CallbackQueryHandler(self.button_handler))

        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)

        self._update_task = asyncio.create_task(self._auto_update_loop())
        log.info("Telegram bot started (chat_id=%s)", self.chat_id)

    async def stop(self):
        if self._update_task:
            self._update_task.cancel()
        if self.app:
            try:
                await self.app.updater.stop()
                await self.app.stop()
                await self.app.shutdown()
            except Exception:
                pass
        log.info("Telegram bot stopped")
