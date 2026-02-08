import asyncio
import json
import logging
import time

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CODEX_API_KEY, CODEX_GRAPHQL_URL, SOLANA_NETWORK_ID
from modules.trader import JupiterTrader

logger = logging.getLogger(__name__)

BANK_PERCENT_BY_SCORE = {
    10: 0.15,
    9: 0.13,
    8: 0.11,
    7: 0.09,
    6: 0.07,
}
MAX_POSITIONS = 5
TRAILING_STOP_PCT = 12.0
QUICK_EXIT_TIME_SEC = 180
QUICK_EXIT_DROP_PCT = 15.0
MAX_LOSS_PCT = 40.0
DEAD_TOKEN_TIME_SEC = 300
DEAD_TOKEN_MIN_CHANGE_PCT = 2.0
PROFIT_LOCK_THRESHOLD_PCT = 25.0
PROFIT_LOCK_STOP_PCT = 20.0
ROCKET_THRESHOLD_PCT = 100.0
ROCKET_SELL_FRACTION = 0.5
SOL_PRICE_CACHE_SEC = 60


class Position:
    def __init__(
        self,
        token_mint: str,
        symbol: str,
        entry_price: float,
        sol_spent: float,
        tokens_held: int,
        score: int,
        tx_signature: str,
    ):
        self.token_mint = token_mint
        self.symbol = symbol
        self.entry_price = entry_price
        self.current_price = entry_price
        self.peak_price = entry_price
        self.sol_spent = sol_spent
        self.tokens_held = tokens_held
        self.tokens_initial = tokens_held
        self.score = score
        self.tx_signature = tx_signature
        self.entry_time = time.time()
        self.last_update = time.time()
        self.status = "open"
        self.pnl_pct = 0.0
        self.peak_pnl_pct = 0.0
        self.trailing_stop_pct = TRAILING_STOP_PCT
        self.profit_locked = False
        self.partial_sold = False
        self.exit_reason = ""
        self.sol_received = 0.0

    def update_price(self, new_price: float):
        if new_price <= 0:
            return
        self.current_price = new_price
        self.last_update = time.time()

        if self.entry_price > 0:
            self.pnl_pct = ((new_price - self.entry_price) / self.entry_price) * 100

        if new_price > self.peak_price:
            self.peak_price = new_price
            self.peak_pnl_pct = self.pnl_pct

    def should_quick_exit(self) -> bool:
        age = time.time() - self.entry_time
        if age < QUICK_EXIT_TIME_SEC and self.pnl_pct <= -QUICK_EXIT_DROP_PCT:
            return True
        return False

    def should_max_loss_exit(self) -> bool:
        if self.pnl_pct <= -MAX_LOSS_PCT:
            return True
        return False

    def should_trailing_stop(self) -> bool:
        if self.peak_pnl_pct < 5:
            return False
        drop_from_peak = self.peak_pnl_pct - self.pnl_pct
        stop = self.trailing_stop_pct
        if self.profit_locked:
            stop = TRAILING_STOP_PCT * 0.5
        elif self.peak_pnl_pct >= 20:
            stop = TRAILING_STOP_PCT * 0.75
        if self.score >= 8:
            stop *= 1.15
        if drop_from_peak >= stop:
            return True
        return False

    def should_profit_lock(self) -> bool:
        if not self.profit_locked and self.pnl_pct >= PROFIT_LOCK_THRESHOLD_PCT:
            return True
        return False

    def should_partial_sell(self) -> bool:
        if not self.partial_sold and self.pnl_pct >= ROCKET_THRESHOLD_PCT:
            return True
        return False

    def should_dead_exit(self) -> bool:
        age = time.time() - self.entry_time
        if age > DEAD_TOKEN_TIME_SEC and abs(self.pnl_pct) < DEAD_TOKEN_MIN_CHANGE_PCT:
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "token_mint": self.token_mint,
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "peak_price": self.peak_price,
            "sol_spent": self.sol_spent,
            "tokens_held": self.tokens_held,
            "score": self.score,
            "pnl_pct": round(self.pnl_pct, 2),
            "peak_pnl_pct": round(self.peak_pnl_pct, 2),
            "status": self.status,
            "exit_reason": self.exit_reason,
            "sol_received": self.sol_received,
            "age_sec": int(time.time() - self.entry_time),
            "profit_locked": self.profit_locked,
            "partial_sold": self.partial_sold,
        }


class PositionManager:
    def __init__(self, trader: JupiterTrader, bank_sol: float = 0.0):
        self.trader = trader
        self.bank_sol = bank_sol
        self.positions: dict[str, Position] = {}
        self.closed_positions: list[dict] = []
        self.total_pnl_sol = 0.0

    async def get_token_price(self, address: str) -> float:
        import httpx

        query = """query($a: String!, $n: Int!) {
            getTokenPrices(inputs: [{address: $a, networkId: $n}]) { priceUsd }
        }"""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    CODEX_GRAPHQL_URL,
                    json={
                        "query": query,
                        "variables": {"a": address, "n": SOLANA_NETWORK_ID},
                    },
                    headers={
                        "Authorization": CODEX_API_KEY,
                        "Content-Type": "application/json",
                    },
                )
                data = resp.json()
                prices = data.get("data", {}).get("getTokenPrices", [])
                if prices and prices[0]:
                    p = prices[0].get("priceUsd")
                    if p and float(p) > 0:
                        return float(p)
        except Exception as e:
            logger.error("Price fetch error for %s: %s", address[:12], e)
        return 0.0

    def get_position_size_sol(self, score: int = 6) -> float:
        pct = BANK_PERCENT_BY_SCORE.get(score, 0.07)
        return self.bank_sol * pct

    async def open_position(self, signal: dict) -> Position | None:
        if len(self.positions) >= MAX_POSITIONS:
            logger.info("Max positions reached (%d), skipping", MAX_POSITIONS)
            return None

        token = signal["token"]
        address = token["address"]
        symbol = token["symbol"]

        if address in self.positions:
            logger.info("Already have position in %s", symbol)
            return None

        sol_amount = self.get_position_size_sol(signal["total_score"])
        if sol_amount < 0.001:
            logger.warning("Position size too small: %.4f SOL", sol_amount)
            return None

        logger.info(
            "Opening position: %s, %.4f SOL, score=%d",
            symbol,
            sol_amount,
            signal["total_score"],
        )

        sol_price = await self._get_sol_price()
        result = await self.trader.buy_token(
            address, sol_amount,
            token_price_usd=token["price_usd"],
            sol_price_usd=sol_price,
        )
        if not result:
            logger.warning("Failed to buy %s", symbol)
            return None

        self.bank_sol -= sol_amount

        pos = Position(
            token_mint=address,
            symbol=symbol,
            entry_price=token["price_usd"],
            sol_spent=sol_amount,
            tokens_held=result["tokens_received"],
            score=signal["total_score"],
            tx_signature=result["tx_signature"],
        )
        self.positions[address] = pos
        logger.info(
            "OPENED %s: %.4f SOL, tokens=%d, price=%.10f",
            symbol,
            sol_amount,
            pos.tokens_held,
            pos.entry_price,
        )
        return pos

    async def close_position(self, address: str, reason: str, fraction: float = 1.0) -> bool:
        pos = self.positions.get(address)
        if not pos:
            return False

        sell_amount = int(pos.tokens_held * fraction)
        if sell_amount <= 0:
            return False

        logger.info(
            "Closing %s (%.0f%%): reason=%s, pnl=%.1f%%",
            pos.symbol,
            fraction * 100,
            reason,
            pos.pnl_pct,
        )

        sol_price = await self._get_sol_price()
        result = await self.trader.sell_token(
            pos.token_mint, sell_amount,
            token_price_usd=pos.current_price,
            sol_price_usd=sol_price,
        )
        if not result:
            logger.warning("Failed to sell %s", pos.symbol)
            return False

        sol_back = result["sol_received"]
        pos.sol_received += sol_back
        pos.tokens_held -= sell_amount

        if fraction >= 1.0 or pos.tokens_held <= 0:
            pos.status = "closed"
            pos.exit_reason = reason
            net_pnl = pos.sol_received - pos.sol_spent
            self.total_pnl_sol += net_pnl
            self.bank_sol += pos.sol_received
            self.closed_positions.append(pos.to_dict())
            del self.positions[address]
            logger.info(
                "CLOSED %s: reason=%s, pnl=%.1f%%, net=%.4f SOL",
                pos.symbol,
                reason,
                pos.pnl_pct,
                net_pnl,
            )
        else:
            pos.partial_sold = True
            self.bank_sol += sol_back
            logger.info(
                "PARTIAL SELL %s: sold %.0f%%, remaining=%d tokens",
                pos.symbol,
                fraction * 100,
                pos.tokens_held,
            )

        return True

    async def check_positions(self):
        if not self.positions:
            return

        for address in list(self.positions.keys()):
            pos = self.positions.get(address)
            if not pos or pos.status != "open":
                continue

            price = await self.get_token_price(address)
            if price > 0:
                pos.update_price(price)

            if pos.should_quick_exit():
                await self.close_position(address, "quick_exit_dump")
                continue

            if pos.should_max_loss_exit():
                await self.close_position(address, "max_loss")
                continue

            if pos.should_profit_lock():
                pos.profit_locked = True
                logger.info(
                    "PROFIT LOCKED %s: pnl=%.1f%%, tighter stop",
                    pos.symbol,
                    pos.pnl_pct,
                )

            if pos.should_partial_sell():
                await self.close_position(
                    address, "rocket_partial_sell", fraction=ROCKET_SELL_FRACTION
                )
                continue

            if pos.should_trailing_stop():
                await self.close_position(address, "trailing_stop")
                continue

            if pos.should_dead_exit():
                await self.close_position(address, "dead_token")
                continue

            await asyncio.sleep(0.3)

    async def _get_sol_price(self) -> float:
        import httpx
        now = time.time()
        if hasattr(self, '_sol_price_cache') and now - self._sol_price_ts < SOL_PRICE_CACHE_SEC:
            return self._sol_price_cache
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://api.coingecko.com/api/v3/simple/price",
                    params={"ids": "solana", "vs_currencies": "usd"},
                )
                price = resp.json().get("solana", {}).get("usd", 200.0)
                self._sol_price_cache = price
                self._sol_price_ts = now
                return price
        except Exception:
            if hasattr(self, '_sol_price_cache'):
                return self._sol_price_cache
            return 200.0

    def get_stats(self) -> dict:
        open_positions = [p.to_dict() for p in self.positions.values()]
        total_open_value = sum(p.sol_spent for p in self.positions.values())
        wins = [c for c in self.closed_positions if c["pnl_pct"] > 0]
        losses = [c for c in self.closed_positions if c["pnl_pct"] <= 0]

        return {
            "bank_sol": round(self.bank_sol, 4),
            "total_pnl_sol": round(self.total_pnl_sol, 4),
            "open_positions": len(self.positions),
            "open_value_sol": round(total_open_value, 4),
            "closed_trades": len(self.closed_positions),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / max(len(self.closed_positions), 1) * 100, 1),
            "positions": open_positions,
            "closed": self.closed_positions[-10:],
        }
