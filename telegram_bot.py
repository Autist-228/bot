import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta

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

    def _get_batch_state(self):
        return self.state.get("batch_state", {})

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
        bs = self._get_batch_state()
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

        model_size = model.get("file_size_kb", 0)
        loss = model.get("loss", 0)
        cycles = model.get("cycles", 0)
        total_samples = model.get("total_samples", 0)

        batches_eaten = bs.get("total_batches", 0)
        tokens_fed = bs.get("total_tokens_fed", 0)

        cur_id = bs.get("current_id", 1)
        cur_start = bs.get("current_start", 0)
        cur_count = bs.get("current_count", 0)
        batch_elapsed = int(time.time() - cur_start) if cur_start > 0 else 0
        batch_mins = batch_elapsed // 60
        batch_secs = batch_elapsed % 60
        batch_pct = min(100, int(batch_elapsed / 1800 * 100)) if cur_start > 0 else 0

        check_id = bs.get("checking_id", 0)
        check_prog = bs.get("checking_progress", 0)
        check_total = bs.get("checking_total", 0)
        check_samples = bs.get("checking_samples", 0)

        last_acc = 0.0
        history = bs.get("history", [])
        if history:
            last_acc = history[-1].get("accuracy", 0)

        wr = 0
        if pnl["wins"] + pnl["losses"] > 0:
            wr = pnl["wins"] / (pnl["wins"] + pnl["losses"]) * 100

        tokens_scanned = stats.get("total", 0)

        lines = [
            "\U0001f916 <b>СТАТУС БОТА: РАБОТАЕТ</b>",
            f"\u23f1 Аптайм: {self._format_uptime()}",
            "",
            "\u2501\u2501\u2501 <b>\U0001f9e0 НС1 (ВХОД)</b> \u2501\u2501\u2501",
            f"\U0001f504 Циклов: {cycles} | Съедено: {total_samples:,}",
            f"\U0001f4c9 Loss: {loss:.4f} | Acc: {last_acc:.1f}%" if loss > 0 else "\U0001f4c9 Loss: --- | Acc: ---",
        ]

        if check_id > 0:
            check_pct = int(check_prog / max(1, check_total) * 100)
            lines.append(f"\U0001f50d Проверка #{check_id}: {check_prog}/{check_total} ({check_pct}%) | {check_samples} годных")
        else:
            lines.append(f"\u23f3 Сбор #{cur_id}: {cur_count} ток ({batch_mins}:{batch_secs:02d} / 30:00, {batch_pct}%)")

        exit_info = self.state.get("exit_model_info", {})
        ex_cycles = exit_info.get("cycles", 0)
        ex_loss = exit_info.get("loss", 0)
        ex_samples = exit_info.get("total_samples", 0)
        ex_sigs = exit_info.get("total_signals_used", 0)

        all_sigs = self._get_signals()
        active_sigs = len([s for s in all_sigs if s.get("status") == "ACTIVE"])
        closed_sigs = len([s for s in all_sigs if s.get("status") == "CLOSED"])

        shadow = self.state.get("shadow_stats", {})
        all_trades = shadow.get("trades", [])
        period_sec = PERIODS[self.current_period][0]
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=period_sec)
        cutoff_iso = cutoff.isoformat()
        period_trades = [t for t in all_trades if t.get("time", "") >= cutoff_iso]
        sh_total = len(period_trades)
        sh_wins = len([t for t in period_trades if (t.get("ns2_pnl") or 0) > 0])
        sh_losses = sh_total - sh_wins
        sh_pnl = sum(t.get("ns2_usd", 0) or 0 for t in period_trades)
        sh_ns2_better = len([t for t in period_trades if (t.get("ns2_pnl") is not None and t.get("ns2_pnl", 0) > t.get("rules_pnl", 0))])

        lines += [
            "",
            "\u2501\u2501\u2501 <b>\U0001f9e0 НС2 (ВЫХОД)</b> \u2501\u2501\u2501",
            f"\U0001f4e1 От НС1: {active_sigs} актив / {closed_sigs} закрыто",
        ]
        if ex_cycles > 0:
            lines.append(f"\U0001f504 Модель #{ex_cycles} | Съедено: {ex_samples:,} из {ex_sigs} сиг")
            lines.append(f"\U0001f4c9 Loss: {ex_loss:.4f}")
        else:
            lines.append("\U0001f504 Модель #0 | \u23f3 Ждём первого обучения...")
        if sh_total > 0:
            sh_wr = sh_wins / sh_total * 100
            lines.append(f"\U0001f4b0 Paper ({period_name}): {sh_total} сдел | {sh_wr:.0f}% WR ({sh_wins}W/{sh_losses}L)")
            lines.append(f"\U0001f4b5 P&L: ${sh_pnl:+.2f} | НС2 лучше: {sh_ns2_better}/{sh_total}")
        else:
            lines.append(f"\U0001f4b0 Paper ({period_name}): нет сделок")

        lines += [
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

        return "\n".join([l for l in lines if l is not None and l != "None"])

    def _build_main_keyboard(self):
        errors = self._get_errors()
        err_count = len([e for e in errors if time.time() - e.get("ts", 0) < 86400])

        if self.session_active:
            row1 = [InlineKeyboardButton("\u23f9 Стоп сессии", callback_data="session_stop")]
        else:
            row1 = [InlineKeyboardButton("\u25b6\ufe0f Старт сессии", callback_data="session_start")]

        period_name = PERIODS[self.current_period][1]
        row2 = [
            InlineKeyboardButton("\U0001f4e6 НС1 пакеты", callback_data="batch_history"),
            InlineKeyboardButton("\U0001f4e6 НС2 пакеты", callback_data="exit_batch_history"),
        ]
        row3 = [
            InlineKeyboardButton("\U0001f4ca Монитор 1ч", callback_data="hourly_log"),
            InlineKeyboardButton("\U0001f4b0 НС2 торговля", callback_data="shadow_trades_0"),
        ]
        row3b = [
            InlineKeyboardButton("\U0001f4cb Токены", callback_data="tokens_0"),
            InlineKeyboardButton("\U0001f9e0 Модели", callback_data="model"),
        ]
        row4 = [
            InlineKeyboardButton(f"\u23f0 {period_name}", callback_data="period"),
        ]
        if err_count > 0:
            row4.append(InlineKeyboardButton(f"\u26a0\ufe0f ({err_count})", callback_data="errors"))
        row4.append(InlineKeyboardButton("\U0001f504 Обновить", callback_data="refresh"))

        rows = [row1, row2, row3, row3b, row4]
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
                    ns2 = s.get("ns2_score")
                    ns2_str = f" | NS2={ns2:.2f}" if ns2 is not None else ""
                    if pnl_val is not None:
                        usd = bet * pnl_val / 100
                        lines.append(f"{idx}. {sym} | \U0001f7e1 {pnl_val:+.1f}% (${usd:+.2f})")
                    else:
                        lines.append(f"{idx}. {sym} | \U0001f7e1 ожидание...")
                    lines.append(f"   conf={conf:.0f}% | открыт{ns2_str} | {time_str}")
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
        bs = self._get_batch_state()
        model_size = model.get("file_size_kb", 0)
        last_save = model.get("last_save_time", "---")
        cycles = model.get("cycles", 0)
        loss = model.get("loss", 0)
        initial_loss = model.get("initial_loss", 0)
        total_samples = model.get("total_samples", 0)
        total_wins = model.get("total_wins", 0)
        rockets_found = model.get("rockets_missed", 0)

        win_pct = (total_wins / total_samples * 100) if total_samples > 0 else 0
        loss_delta = ""
        if initial_loss > 0 and loss > 0:
            change = ((loss - initial_loss) / initial_loss) * 100
            loss_delta = f" ({change:+.1f}%)"

        history = bs.get("history", [])
        last_acc = history[-1].get("accuracy", 0) if history else 0
        avg_acc = 0.0
        if history:
            accs = [h.get("accuracy", 0) for h in history[-5:]]
            avg_acc = sum(accs) / len(accs) if accs else 0

        lines = [
            "\U0001f9e0 <b>НЕЙРОСЕТИ</b>",
            "",
            "\u2501\u2501\u2501 <b>НЕЙРОСЕТЬ 1 (ВХОД)</b> \u2501\u2501\u2501",
            f"\U0001f4c1 entry_model.pt ({model_size} KB)",
            f"\U0001f504 Циклов: {cycles}",
            f"\U0001f4c9 Loss: {loss:.4f}{loss_delta}" if loss > 0 else "\U0001f4c9 Loss: ---",
            f"\U0001f4ca Съедено: {total_samples:,} токенов",
            f"\U0001f3c6 Побед: {total_wins:,} ({win_pct:.0f}%)",
            f"\U0001f3af Accuracy: {last_acc:.1f}% (ср.5: {avg_acc:.1f}%)" if last_acc > 0 else "\U0001f3af Accuracy: ---",
            f"\U0001f680 Ракет: {rockets_found}",
        ]

        exit_info = self.state.get("exit_model_info", {})
        ex_cycles = exit_info.get("cycles", 0)
        ex_loss = exit_info.get("loss", 0)
        ex_initial = exit_info.get("initial_loss", 0)
        ex_samples = exit_info.get("total_samples", 0)
        ex_sigs = exit_info.get("total_signals_used", 0)

        ex_loss_delta = ""
        if ex_initial > 0 and ex_loss > 0:
            ch = ((ex_loss - ex_initial) / ex_initial) * 100
            ex_loss_delta = f" ({ch:+.1f}%)"

        lines.append("")
        lines.append("\u2501\u2501\u2501 <b>НЕЙРОСЕТЬ 2 (ВЫХОД)</b> \u2501\u2501\u2501")
        if ex_cycles > 0:
            lines.append(f"\U0001f504 Циклов: {ex_cycles}")
            lines.append(f"\U0001f4c9 Loss: {ex_loss:.4f}{ex_loss_delta}")
            lines.append(f"\U0001f4ca Съедено: {ex_samples:,} точек из {ex_sigs} сигналов")
            lines.append("\U0001f6d1 Статус: учится (выход по правилам)")
        else:
            lines.append("\u23f3 Ожидание закрытых сигналов...")
            lines.append("\U0001f6d1 Статус: не обучена")

        return "\n".join([l for l in lines if l is not None and l != ""])

    def _build_model_keyboard(self):
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("\U0001f3e0 Главная", callback_data="home")],
        ])

    def _build_batch_history_text(self):
        bs = self._get_batch_state()
        history = bs.get("history", [])
        batches_eaten = bs.get("total_batches", 0)
        tokens_fed = bs.get("total_tokens_fed", 0)

        lines = [
            "\U0001f4e6 <b>НС1 (ВХОД) — ИСТОРИЯ ПАКЕТОВ</b>",
            "",
            f"\U0001f37d Всего: {batches_eaten} пакетов, {tokens_fed:,} токенов",
            "",
        ]

        if not history:
            lines.append("Пакетов ещё нет. Первый будет через ~30 мин.")
        else:
            for h in reversed(history[-10:]):
                bid = h.get("id", 0)
                total = h.get("tokens_total", 0)
                fed = h.get("tokens_fed", 0)
                rockets = h.get("rockets", 0)
                acc = h.get("accuracy", 0)
                loss_val = h.get("loss", 0)
                wp = h.get("win_pct", 0)
                lines.append(
                    f"#{bid} | {fed}/{total} ток | "
                    f"{rockets} \U0001f680 | acc={acc:.0f}% | "
                    f"loss={loss_val:.4f} | win={wp:.0f}%"
                )

        return "\n".join(lines)

    def _build_batch_history_keyboard(self):
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("\U0001f3e0 Главная", callback_data="home")],
        ])

    def _build_exit_batch_history_text(self):
        bs = self._get_batch_state()
        history = bs.get("history", [])
        exit_info = self.state.get("exit_model_info", {})
        ex_cycles = exit_info.get("cycles", 0)
        ex_samples = exit_info.get("total_samples", 0)
        ex_sigs = exit_info.get("total_signals_used", 0)

        shadow = self.state.get("shadow_stats", {})
        all_trades = shadow.get("trades", [])
        sh_total_all = shadow.get("total", 0)
        sh_wins_all = shadow.get("wins", 0)
        sh_pnl_all = shadow.get("pnl_usd", 0)

        lines = [
            "\U0001f4e6 <b>НС2 (ВЫХОД) — ИСТОРИЯ МОДЕЛЕЙ</b>",
            "",
            f"\U0001f504 Текущая модель: #{ex_cycles}",
            f"\U0001f4ca Съедено: {ex_samples:,} точек из {ex_sigs} сигналов",
            f"\U0001f4b0 Paper всего: {sh_total_all} сдел | P&L: ${sh_pnl_all:+.2f}",
            "",
        ]

        has_exit = [h for h in history if h.get("exit_samples", 0) > 0]
        if not has_exit:
            lines.append("Моделей ещё нет. Ждём первое обучение...")
        else:
            model_num = 0
            for h in has_exit:
                model_num += 1
            shown = has_exit[-10:]
            start_num = max(1, model_num - len(shown) + 1)
            for i, h in enumerate(reversed(shown)):
                mn = model_num - i
                bid = h.get("id", 0)
                e_samp = h.get("exit_samples", 0)
                e_sigs = h.get("exit_signals", 0)
                e_loss = h.get("exit_loss", 0)
                lines.append(
                    f"Модель #{mn} (пакет #{bid}) | "
                    f"{e_samp} точек из {e_sigs} сиг | "
                    f"loss={e_loss:.4f}"
                )

        return "\n".join(lines)

    def _build_exit_batch_history_keyboard(self):
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("\U0001f3e0 Главная", callback_data="home")],
        ])

    def _build_hourly_text(self, page=0):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        stats_file = os.path.join(data_dir, "hourly_stats.json")

        lines = [
            "\U0001f4ca <b>ЧАСОВОЙ ЛОГ (НС1 + НС2)</b>",
            "",
        ]

        if not os.path.exists(stats_file):
            lines.append("Данных ещё нет. Первый снимок через ~1 час.")
            return "\n".join(lines), 0, 0

        try:
            with open(stats_file) as f:
                snapshots = json.load(f)
        except Exception:
            lines.append("Ошибка чтения файла.")
            return "\n".join(lines), 0, 0

        if not snapshots:
            lines.append("Снимков ещё нет.")
            return "\n".join(lines), 0, 0

        per_page = 12
        total_pages = max(1, (len(snapshots) + per_page - 1) // per_page)
        page = max(0, min(page, total_pages - 1))

        start_idx = len(snapshots) - (page + 1) * per_page
        end_idx = len(snapshots) - page * per_page
        start_idx = max(0, start_idx)
        page_snaps = snapshots[start_idx:end_idx]

        lines.append(f"Всего: {len(snapshots)} | Стр {page + 1}/{total_pages}")
        lines.append("")

        for snap in reversed(page_snaps):
            dt = snap.get("datetime", "")
            try:
                t = datetime.fromisoformat(dt).strftime("%d.%m %H:%M")
            except Exception:
                t = dt[:16]
            sigs = snap.get("hour_signals", 0)
            w = snap.get("hour_wins", 0)
            lo = snap.get("hour_losses", 0)
            pnl_val = snap.get("hour_pnl_usd", 0)
            loss_val = snap.get("model_loss", 0)
            acc = snap.get("model_accuracy", 0)
            ex_loss = snap.get("exit_loss", 0)
            ex_cyc = snap.get("exit_cycles", 0)
            ns2 = f" | НС2:{ex_loss:.4f}" if ex_cyc > 0 else ""
            lines.append(
                f"{t} | {sigs}sig {w}W/{lo}L "
                f"${pnl_val:+.2f} | НС1:{loss_val:.4f} acc{acc:.0f}%{ns2}"
            )

        return "\n".join(lines), page, total_pages

    def _build_hourly_keyboard(self, page=0, total_pages=1):
        nav = []
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton("\u25c0 Старше", callback_data=f"hourly_{page + 1}"))
        if page > 0:
            nav.append(InlineKeyboardButton("Новее \u25b6", callback_data=f"hourly_{page - 1}"))
        rows = []
        if nav:
            rows.append(nav)
        rows.append([InlineKeyboardButton("\U0001f3e0 Главная", callback_data="home")])
        return InlineKeyboardMarkup(rows)

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

    def _build_shadow_trades_text(self, page=0):
        shadow = self.state.get("shadow_stats", {})
        trades = shadow.get("trades", [])
        sh_total = shadow.get("total", 0)
        sh_wins = shadow.get("wins", 0)
        sh_losses = shadow.get("losses", 0)
        sh_pnl = shadow.get("pnl_usd", 0)
        sh_ns2 = shadow.get("ns2_better", 0)
        sh_rules = shadow.get("rules_better", 0)

        lines = [
            "\U0001f4b0 <b>НС2 PAPER TRADING</b>",
            "",
        ]
        if sh_total > 0:
            sh_wr = sh_wins / sh_total * 100
            lines.append(f"Сделок: {sh_total} | WR: {sh_wr:.0f}% ({sh_wins}W/{sh_losses}L)")
            lines.append(f"Paper P&L: ${sh_pnl:+.2f}")
            lines.append(f"НС2 лучше: {sh_ns2} | Правила лучше: {sh_rules}")
        else:
            lines.append("Сделок пока нет. Ждём закрытия сигналов...")
            return "\n".join(lines), 0, 0

        lines.append("")

        per_page = 8
        total_pages = max(1, (len(trades) + per_page - 1) // per_page)
        page = max(0, min(page, total_pages - 1))

        start_idx = len(trades) - (page + 1) * per_page
        end_idx = len(trades) - page * per_page
        start_idx = max(0, start_idx)
        page_trades = trades[start_idx:end_idx]

        lines.append(f"Стр {page + 1}/{total_pages}")
        lines.append("")

        for t in reversed(page_trades):
            sym = t.get("symbol", "?")
            ns2_pnl = t.get("ns2_pnl")
            rules_pnl = t.get("rules_pnl", 0)
            ns2_usd = t.get("ns2_usd")
            rules_usd = t.get("rules_usd", 0)
            rule = t.get("rule", "")
            dt = t.get("time", "")
            try:
                ts = datetime.fromisoformat(dt).strftime("%H:%M")
            except Exception:
                ts = dt[:5]
            if ns2_pnl is not None:
                better = "\u2705" if ns2_pnl > rules_pnl else "\u274c"
                lines.append(
                    f"{ts} {sym} | НС2:{ns2_pnl:+.1f}% вс Прав:{rules_pnl:+.1f}% {better}"
                )
            else:
                lines.append(
                    f"{ts} {sym} | НС2:н/д вс Прав:{rules_pnl:+.1f}%"
                )

        return "\n".join(lines), page, total_pages

    def _build_shadow_trades_keyboard(self, page=0, total_pages=1):
        nav = []
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton("\u25c0 Старше", callback_data=f"shadow_trades_{page + 1}"))
        if page > 0:
            nav.append(InlineKeyboardButton("Новее \u25b6", callback_data=f"shadow_trades_{page - 1}"))
        rows = []
        if nav:
            rows.append(nav)
        rows.append([InlineKeyboardButton("\U0001f3e0 Главная", callback_data="home")])
        return InlineKeyboardMarkup(rows)

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

        elif data == "batch_history":
            self._current_screen = "batch_history"
            text = self._build_batch_history_text()
            kb = self._build_batch_history_keyboard()
            await self._send_or_edit(chat_id, text, kb, msg_id)

        elif data == "exit_batch_history":
            self._current_screen = "exit_batch_history"
            text = self._build_exit_batch_history_text()
            kb = self._build_exit_batch_history_keyboard()
            await self._send_or_edit(chat_id, text, kb, msg_id)

        elif data.startswith("shadow_trades_"):
            self._current_screen = "shadow_trades"
            page = 0
            try:
                page = int(data.split("_")[2])
            except Exception:
                page = 0
            text, page, total_pages = self._build_shadow_trades_text(page)
            kb = self._build_shadow_trades_keyboard(page, total_pages)
            await self._send_or_edit(chat_id, text, kb, msg_id)

        elif data == "hourly_log" or data.startswith("hourly_"):
            self._current_screen = "hourly_log"
            page = 0
            if data.startswith("hourly_"):
                try:
                    page = int(data.split("_")[1])
                except Exception:
                    page = 0
            text, page, total_pages = self._build_hourly_text(page)
            kb = self._build_hourly_keyboard(page, total_pages)
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
