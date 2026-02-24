#!/usr/bin/env python3
"""
Maximum Sniper Bot v2 — realistic paper trading mode.

Watches pump.fun tokens at CREATION, buys ONLY after Phase 2 confirmation.
Includes real Solana transaction fees and extra slippage in simulation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import websockets
import websockets.exceptions

from config import (
    BET_SIZE_USD,
    BLACKLIST_PATH,
    BUY_SLIPPAGE_PCT,
    DATA_DIR,
    DEV_MAX_BUY_SOL,
    DEV_MIN_BUY_SOL,
    EMERGENCY_STOP_PCT,
    EXTRA_BUY_SLIPPAGE_PCT,
    EXTRA_SELL_SLIPPAGE_PCT,
    MAX_CONCURRENT,
    PHASE2_MIN_BUYERS,
    PHASE2_WAIT_SEC,
    PUMPFUN_FEE_PCT,
    PUMPPORTAL_WS_URL,
    SELL_SLIPPAGE_PCT,
    SOL_TX_FEE_PER_TRADE,
    STARTING_BALANCE,
    STATE_PATH,
    TIME_STOP_SEC,
    TRADES_LOG_PATH,
    TRAILING_STOP_PCT,
)

log = logging.getLogger("sniper")

SOL_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"


@dataclass
class WatchToken:
    mint: str
    symbol: str
    name: str
    dev_address: str
    create_time: float
    init_buy_sol: float
    unique_buyers: set[str] = field(default_factory=set)
    dev_sold: bool = False
    total_buys: int = 0
    total_sells: int = 0


@dataclass
class Position:
    mint: str
    symbol: str
    name: str
    dev_address: str
    entry_time: float
    sim_sol_spent: float
    sim_tokens_bought: float
    unique_buyers: set[str] = field(default_factory=set)
    dev_sold: bool = False
    peak_pnl_pct: float = 0.0
    current_pnl_pct: float = 0.0
    exit_reason: str = ""
    exit_time: float = 0.0
    pnl_usd: float = 0.0
    last_trade_time: float = 0.0
    total_buys: int = 0
    total_sells: int = 0
    phase: str = "confirmed"


class Sniper:
    def __init__(self) -> None:
        self.watching: dict[str, WatchToken] = {}
        self.positions: dict[str, Position] = {}
        self.token_curves: dict[str, dict[str, float]] = {}
        self.dev_blacklist: set[str] = set()
        self.sol_price_usd: float = 170.0
        self.paper_balance: float = STARTING_BALANCE
        self.reserved: float = 0.0
        self.total_trades: int = 0
        self.total_wins: int = 0
        self.total_pnl: float = 0.0
        self.total_fees: float = 0.0
        self.best_trade: float = 0.0
        self.worst_trade: float = 0.0
        self.trades_log: list[dict] = []
        self.tokens_seen: int = 0
        self.tokens_passed_p1: int = 0
        self.tokens_passed_p2: int = 0
        self.tokens_p2_failed: int = 0
        self.start_time: float = time.time()
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._running = True
        self._tasks: list[asyncio.Task[None]] = []
        self._load_blacklist()

    def _load_blacklist(self) -> None:
        p = Path(BLACKLIST_PATH)
        if p.exists():
            try:
                self.dev_blacklist = set(json.loads(p.read_text()))
                log.info("Loaded %d blacklisted devs", len(self.dev_blacklist))
            except Exception:
                pass

    def _save_blacklist(self) -> None:
        Path(BLACKLIST_PATH).write_text(json.dumps(list(self.dev_blacklist)))

    def _append_trade_log(self, rec: dict) -> None:
        with open(TRADES_LOG_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _save_state(self) -> None:
        state = {
            "paper_balance": self.paper_balance,
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "total_pnl": self.total_pnl,
            "total_fees": self.total_fees,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "tokens_seen": self.tokens_seen,
            "tokens_passed_p1": self.tokens_passed_p1,
            "tokens_passed_p2": self.tokens_passed_p2,
            "tokens_p2_failed": self.tokens_p2_failed,
            "start_time": self.start_time,
            "blacklist_size": len(self.dev_blacklist),
        }
        Path(STATE_PATH).write_text(json.dumps(state, indent=2))

    def _available_balance(self) -> float:
        return self.paper_balance - self.reserved

    def _open_count(self) -> int:
        return sum(1 for p in self.positions.values() if p.phase != "closed")

    def _watching_count(self) -> int:
        return len(self.watching)

    def _tx_fee_usd(self) -> float:
        return SOL_TX_FEE_PER_TRADE * self.sol_price_usd

    def _simulate_buy(
        self, v_sol: float, v_tokens: float, usd_amount: float
    ) -> tuple[float, float]:
        sol_amount = usd_amount / self.sol_price_usd
        sol_in = sol_amount * (1 - PUMPFUN_FEE_PCT) * (1 - BUY_SLIPPAGE_PCT) * (1 - EXTRA_BUY_SLIPPAGE_PCT)
        k = v_sol * v_tokens
        new_v_sol = v_sol + sol_in
        new_v_tokens = k / new_v_sol
        tokens_received = v_tokens - new_v_tokens
        return tokens_received, sol_amount

    def _simulate_sell(
        self, v_sol: float, v_tokens: float, tokens: float
    ) -> float:
        k = v_sol * v_tokens
        new_v_tokens = v_tokens + tokens
        new_v_sol = k / new_v_tokens
        gross_sol = v_sol - new_v_sol
        return gross_sol * (1 - PUMPFUN_FEE_PCT) * (1 - SELL_SLIPPAGE_PCT) * (1 - EXTRA_SELL_SLIPPAGE_PCT)

    def _calc_pnl_pct(self, pos: Position) -> float:
        if pos.sim_tokens_bought <= 0 or pos.sim_sol_spent <= 0:
            return 0.0
        curve = self.token_curves.get(pos.mint)
        if not curve:
            return 0.0
        sol_out = self._simulate_sell(
            curve["v_sol"], curve["v_tokens"], pos.sim_tokens_bought
        )
        return ((sol_out / pos.sim_sol_spent) - 1) * 100

    def _phase1_check(self, msg: dict) -> bool:
        dev = msg.get("traderPublicKey", "")
        if not dev or dev in self.dev_blacklist:
            return False
        init_buy = float(msg.get("initialBuy") or 0)
        init_buy_sol = init_buy / 1e9 if init_buy > 0 else 0.0
        if init_buy_sol < DEV_MIN_BUY_SOL or init_buy_sol > DEV_MAX_BUY_SOL:
            return False
        return True

    def _phase2_check(self, wt: WatchToken) -> bool:
        return len(wt.unique_buyers) >= PHASE2_MIN_BUYERS and not wt.dev_sold

    def _check_exit(self, pos: Position, pnl_pct: float) -> str | None:
        age = time.time() - pos.entry_time
        if pnl_pct <= EMERGENCY_STOP_PCT:
            return "EMERGENCY"
        if pos.peak_pnl_pct > 5.0:
            drop = pos.peak_pnl_pct - pnl_pct
            if drop >= TRAILING_STOP_PCT:
                return f"TRAILING(peak={pos.peak_pnl_pct:.0f}%)"
        if age >= TIME_STOP_SEC and pnl_pct < 5.0:
            return f"TIME({age:.0f}s)"
        return None

    def _close_position(self, pos: Position, reason: str, pnl_pct: float) -> None:
        pos.phase = "closed"
        pos.exit_reason = reason
        pos.exit_time = time.time()
        pos.current_pnl_pct = pnl_pct
        pnl_usd = BET_SIZE_USD * (pnl_pct / 100)

        fee_usd = self._tx_fee_usd()
        self.paper_balance -= fee_usd
        self.total_fees += fee_usd

        pos.pnl_usd = pnl_usd

        self.paper_balance += pnl_usd
        self.reserved -= BET_SIZE_USD
        self.total_trades += 1
        self.total_pnl += pnl_usd
        if pnl_usd > 0:
            self.total_wins += 1
        self.best_trade = max(self.best_trade, pnl_usd)
        self.worst_trade = min(self.worst_trade, pnl_usd)

        age = pos.exit_time - pos.entry_time
        if pnl_pct < -20 and age < 30:
            self.dev_blacklist.add(pos.dev_address)

        rec = {
            "mint": pos.mint,
            "symbol": pos.symbol,
            "dev": pos.dev_address,
            "entry_ts": pos.entry_time,
            "exit_ts": pos.exit_time,
            "hold_sec": round(age, 1),
            "pnl_pct": round(pnl_pct, 2),
            "pnl_usd": round(pnl_usd, 4),
            "fee_usd": round(fee_usd * 2, 4),
            "peak_pct": round(pos.peak_pnl_pct, 2),
            "reason": reason,
            "buyers": len(pos.unique_buyers),
            "buys": pos.total_buys,
            "sells": pos.total_sells,
        }
        self.trades_log.append(rec)
        self._append_trade_log(rec)

        tag = "WIN " if pnl_usd > 0 else "LOSS"
        log.info(
            "%s %s | %s | PnL: %+.2f%% ($%+.4f) | fee=$%.4f | peak=%+.0f%% | hold=%ds | buyers=%d | bal=$%.2f",
            tag,
            pos.symbol[:10],
            reason,
            pnl_pct,
            pnl_usd,
            fee_usd * 2,
            pos.peak_pnl_pct,
            int(age),
            len(pos.unique_buyers),
            self.paper_balance,
        )

    def _start_watching(self, msg: dict) -> str | None:
        mint = msg.get("mint", "")
        if not mint or mint in self.watching or mint in self.positions:
            return None

        dev = msg.get("traderPublicKey", "")
        init_buy = float(msg.get("initialBuy") or 0)
        init_buy_sol = init_buy / 1e9 if init_buy > 0 else 0.0

        v_sol = float(msg.get("vSolInBondingCurve") or 0)
        v_tokens = float(msg.get("vTokensInBondingCurve") or 0)
        if v_sol > 0 and v_tokens > 0:
            self.token_curves[mint] = {"v_sol": v_sol, "v_tokens": v_tokens}

        wt = WatchToken(
            mint=mint,
            symbol=msg.get("symbol", "???"),
            name=msg.get("name", ""),
            dev_address=dev,
            create_time=time.time(),
            init_buy_sol=init_buy_sol,
        )
        self.watching[mint] = wt
        self.tokens_passed_p1 += 1
        return mint

    def _open_position_from_watch(self, wt: WatchToken) -> None:
        mint = wt.mint
        if mint in self.positions:
            return
        if self._open_count() >= MAX_CONCURRENT:
            return

        fee_usd = self._tx_fee_usd()
        if self._available_balance() < BET_SIZE_USD + fee_usd:
            return

        curve = self.token_curves.get(mint)
        if not curve:
            return
        v_sol = curve["v_sol"]
        v_tokens = curve["v_tokens"]
        if v_sol <= 0 or v_tokens <= 0:
            return

        tokens_bought, sol_spent = self._simulate_buy(v_sol, v_tokens, BET_SIZE_USD)
        if tokens_bought <= 0:
            return

        self.paper_balance -= fee_usd
        self.total_fees += fee_usd

        pos = Position(
            mint=mint,
            symbol=wt.symbol,
            name=wt.name,
            dev_address=wt.dev_address,
            entry_time=time.time(),
            sim_sol_spent=sol_spent,
            sim_tokens_bought=tokens_bought,
            unique_buyers=wt.unique_buyers,
            dev_sold=wt.dev_sold,
            total_buys=wt.total_buys,
            total_sells=wt.total_sells,
            phase="confirmed",
        )
        self.positions[mint] = pos
        self.reserved += BET_SIZE_USD
        self.tokens_passed_p2 += 1
        log.info(
            "BUY  %s | buyers=%d | v_sol=%.1f | $%.2f bet | fee=$%.4f | open=%d",
            pos.symbol[:10],
            len(wt.unique_buyers),
            v_sol,
            BET_SIZE_USD,
            fee_usd,
            self._open_count(),
        )

    def _handle_trade(self, msg: dict) -> None:
        mint = msg.get("mint", "")
        if not mint:
            return

        v_sol = float(msg.get("vSolInBondingCurve") or 0)
        v_tokens = float(msg.get("vTokensInBondingCurve") or 0)
        if v_sol > 0 and v_tokens > 0:
            self.token_curves[mint] = {"v_sol": v_sol, "v_tokens": v_tokens}

        trader = msg.get("traderPublicKey", "")
        tx_type = msg.get("txType", "")

        wt = self.watching.get(mint)
        if wt:
            if tx_type == "buy":
                wt.total_buys += 1
                if trader != wt.dev_address:
                    wt.unique_buyers.add(trader)
            elif tx_type == "sell":
                wt.total_sells += 1
                if trader == wt.dev_address:
                    wt.dev_sold = True
            return

        pos = self.positions.get(mint)
        if not pos or pos.phase == "closed":
            return

        pos.last_trade_time = time.time()
        if tx_type == "buy":
            pos.total_buys += 1
            if trader != pos.dev_address:
                pos.unique_buyers.add(trader)
        elif tx_type == "sell":
            pos.total_sells += 1
            if trader == pos.dev_address:
                pos.dev_sold = True

        pnl = self._calc_pnl_pct(pos)
        pos.current_pnl_pct = pnl
        if pnl > pos.peak_pnl_pct:
            pos.peak_pnl_pct = pnl

        reason = self._check_exit(pos, pnl)
        if reason:
            self._close_position(pos, reason, pnl)

    async def _ws_listener(self) -> None:
        while self._running:
            try:
                async with websockets.connect(
                    PUMPPORTAL_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    log.info("Connected to PumpPortal WebSocket")
                    await ws.send(json.dumps({"method": "subscribeNewToken"}))
                    log.info("Subscribed to newToken events")

                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        tx_type = msg.get("txType", "")

                        if tx_type == "create":
                            self.tokens_seen += 1
                            if self._phase1_check(msg):
                                mint = self._start_watching(msg)
                                if mint:
                                    await ws.send(
                                        json.dumps(
                                            {
                                                "method": "subscribeTokenTrade",
                                                "keys": [mint],
                                            }
                                        )
                                    )
                        elif tx_type in ("buy", "sell"):
                            self._handle_trade(msg)

            except websockets.exceptions.ConnectionClosed:
                log.warning("WebSocket disconnected, reconnecting in 3s...")
            except Exception as exc:
                log.error("WebSocket error: %s, reconnecting in 5s...", exc)
                await asyncio.sleep(5)
                continue
            if self._running:
                await asyncio.sleep(3)

    async def _position_checker(self) -> None:
        while self._running:
            await asyncio.sleep(1)
            now = time.time()

            expired_watches: list[str] = []
            for mint, wt in list(self.watching.items()):
                age = now - wt.create_time
                if age >= PHASE2_WAIT_SEC:
                    if self._phase2_check(wt):
                        self._open_position_from_watch(wt)
                    else:
                        self.tokens_p2_failed += 1
                    expired_watches.append(mint)

            for mint in expired_watches:
                self.watching.pop(mint, None)

            to_close: list[tuple[Position, str, float]] = []
            for pos in list(self.positions.values()):
                if pos.phase == "closed":
                    continue

                pnl = self._calc_pnl_pct(pos)
                pos.current_pnl_pct = pnl
                if pnl > pos.peak_pnl_pct:
                    pos.peak_pnl_pct = pnl

                reason = self._check_exit(pos, pnl)
                if reason:
                    to_close.append((pos, reason, pnl))

            for pos, reason, pnl in to_close:
                self._close_position(pos, reason, pnl)

    async def _stats_printer(self) -> None:
        while self._running:
            await asyncio.sleep(60)
            uptime = time.time() - self.start_time
            hrs = uptime / 3600
            open_positions = self._open_count()
            watching = self._watching_count()
            wr = (self.total_wins / self.total_trades * 100) if self.total_trades > 0 else 0.0
            per_day = (self.total_pnl / hrs * 24) if hrs > 0 else 0.0

            log.info(
                "=== STATS %.1fh | bal=$%.2f | pnl=$%+.2f | fees=$%.2f | trades=%d | WR=%.1f%% | "
                "$/day=$%+.1f | open=%d | watch=%d | seen=%d | p1=%d | p2=%d | p2fail=%d | bl=%d ===",
                hrs,
                self.paper_balance,
                self.total_pnl,
                self.total_fees,
                self.total_trades,
                wr,
                per_day,
                open_positions,
                watching,
                self.tokens_seen,
                self.tokens_passed_p1,
                self.tokens_passed_p2,
                self.tokens_p2_failed,
                len(self.dev_blacklist),
            )
            self._save_state()
            self._save_blacklist()

    async def _sol_price_updater(self) -> None:
        while self._running:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(SOL_PRICE_URL)
                    if resp.status_code == 200:
                        data = resp.json()
                        price = data.get("solana", {}).get("usd", 0)
                        if price > 0:
                            self.sol_price_usd = price
                            log.info("SOL price: $%.2f", price)
            except Exception as exc:
                log.warning("SOL price fetch failed: %s", exc)
            await asyncio.sleep(300)

    async def _memory_cleaner(self) -> None:
        while self._running:
            await asyncio.sleep(120)
            now = time.time()
            stale_mints = [
                mint
                for mint, pos in self.positions.items()
                if pos.phase == "closed" and now - pos.exit_time > 600
            ]
            for mint in stale_mints:
                del self.positions[mint]
                self.token_curves.pop(mint, None)
            stale_watches = [
                mint
                for mint, wt in self.watching.items()
                if now - wt.create_time > 60
            ]
            for mint in stale_watches:
                del self.watching[mint]
                self.token_curves.pop(mint, None)
            stale_curves = [
                mint
                for mint in self.token_curves
                if mint not in self.positions and mint not in self.watching
            ]
            for mint in stale_curves:
                del self.token_curves[mint]
            cleaned = len(stale_mints) + len(stale_watches) + len(stale_curves)
            if cleaned:
                log.info(
                    "Cleaned %d positions, %d watches, %d curves",
                    len(stale_mints),
                    len(stale_watches),
                    len(stale_curves),
                )

    def stop(self) -> None:
        log.info("Shutting down...")
        self._running = False
        for t in self._tasks:
            t.cancel()

    async def run(self) -> None:
        log.info(
            "Maximum Sniper v2 starting | balance=$%.2f | bet=$%.2f | max_concurrent=%d",
            self.paper_balance,
            BET_SIZE_USD,
            MAX_CONCURRENT,
        )
        log.info(
            "Phase1: dev_buy=%.2f-%.1f SOL | Phase2: %ds, %d+ buyers (BUY ONLY AFTER P2)",
            DEV_MIN_BUY_SOL,
            DEV_MAX_BUY_SOL,
            PHASE2_WAIT_SEC,
            PHASE2_MIN_BUYERS,
        )
        log.info(
            "Exit: trailing=%g%%, time=%ds, emergency=%g%% | "
            "Extra slippage: buy=%g%%, sell=%g%% | TX fee=%.6f SOL",
            TRAILING_STOP_PCT,
            TIME_STOP_SEC,
            EMERGENCY_STOP_PCT,
            EXTRA_BUY_SLIPPAGE_PCT * 100,
            EXTRA_SELL_SLIPPAGE_PCT * 100,
            SOL_TX_FEE_PER_TRADE,
        )

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.stop)

        self._tasks = [
            asyncio.create_task(self._ws_listener()),
            asyncio.create_task(self._position_checker()),
            asyncio.create_task(self._stats_printer()),
            asyncio.create_task(self._sol_price_updater()),
            asyncio.create_task(self._memory_cleaner()),
        ]
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self._save_state()
            self._save_blacklist()
            log.info(
                "Final: bal=$%.2f | pnl=$%+.2f | fees=$%.2f | trades=%d | wins=%d",
                self.paper_balance,
                self.total_pnl,
                self.total_fees,
                self.total_trades,
                self.total_wins,
            )


def main() -> None:
    bot = Sniper()
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
