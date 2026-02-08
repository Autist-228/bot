import asyncio
import logging
import time

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CODEX_API_KEY, CODEX_GRAPHQL_URL, SOLANA_NETWORK_ID

logger = logging.getLogger(__name__)

WATCH_CHECK_INTERVAL_SEC = 15
WATCH_MAX_TIME_SEC = 90
WATCH_MIN_OBSERVE_SEC = 15
WATCH_MIN_TIME_SEC = 45
WATCH_CONFIRM_GROWTH_PCT = 8.0
WATCH_FAST_CONFIRM_PCT = 20.0
WATCH_REJECT_DROP_PCT = -5.0
WATCH_MAX_SLOTS = 10
WATCH_ANTI_PEAK_PCT = 25.0
WATCH_DOUBLE_CONFIRM_SEC = 15
WATCH_DOUBLE_CONFIRM_MAX_DROP_PCT = 5.0


class WatchItem:
    def __init__(self, signal: dict, entry_price: float):
        self.signal = signal
        self.token_mint = signal["token"]["address"]
        self.symbol = signal["token"]["symbol"]
        self.entry_price = entry_price
        self.current_price = entry_price
        self.peak_price = entry_price
        self.added_at = time.time()
        self.checks = 0
        self.prices: list[float] = [entry_price]
        self.status = "watching"
        self.first_confirm_at: float = 0.0
        self.first_confirm_price: float = 0.0

    @property
    def age_sec(self) -> float:
        return time.time() - self.added_at

    @property
    def change_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100

    @property
    def peak_change_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return ((self.peak_price - self.entry_price) / self.entry_price) * 100

    def update_price(self, price: float):
        if price <= 0:
            return
        self.current_price = price
        self.prices.append(price)
        self.checks += 1
        if price > self.peak_price:
            self.peak_price = price

    def _meets_confirm_criteria(self) -> bool:
        if self.age_sec < WATCH_MIN_OBSERVE_SEC:
            return False
        if self.change_pct >= WATCH_ANTI_PEAK_PCT:
            return False
        if self.age_sec >= WATCH_MIN_TIME_SEC and self.change_pct >= WATCH_CONFIRM_GROWTH_PCT:
            if self._is_growth_stable():
                return True
        if self.age_sec >= WATCH_MIN_OBSERVE_SEC and self.change_pct >= WATCH_FAST_CONFIRM_PCT:
            if self._is_growth_stable():
                return True
        return False

    def is_confirmed(self) -> bool:
        if self.first_confirm_at == 0.0:
            if self._meets_confirm_criteria():
                self.first_confirm_at = time.time()
                self.first_confirm_price = self.current_price
            return False
        wait = time.time() - self.first_confirm_at
        if wait < WATCH_DOUBLE_CONFIRM_SEC:
            return False
        if self.first_confirm_price > 0:
            drop_since_confirm = ((self.current_price - self.first_confirm_price) / self.first_confirm_price) * 100
            if drop_since_confirm < -WATCH_DOUBLE_CONFIRM_MAX_DROP_PCT:
                self.first_confirm_at = 0.0
                self.first_confirm_price = 0.0
                return False
        return True

    def _is_growth_stable(self) -> bool:
        if len(self.prices) < 2:
            return True
        last_price = self.prices[-1]
        prev_price = self.prices[-2] if len(self.prices) >= 2 else self.entry_price
        if prev_price > 0:
            recent_drop = ((last_price - prev_price) / prev_price) * 100
            if recent_drop < -10:
                return False
        return True

    def is_rejected(self) -> bool:
        if self.age_sec >= WATCH_MIN_OBSERVE_SEC and self.change_pct <= WATCH_REJECT_DROP_PCT:
            return True
        if self.change_pct <= -20.0:
            return True
        if self.age_sec >= WATCH_MAX_TIME_SEC:
            if self.change_pct < WATCH_CONFIRM_GROWTH_PCT:
                return True
        if self.checks >= 2 and self.peak_change_pct > 10 and self.change_pct < 0:
            return True
        if self.change_pct >= WATCH_ANTI_PEAK_PCT:
            if self.age_sec >= WATCH_MIN_OBSERVE_SEC:
                return True
        return False


class Watchlist:
    def __init__(self):
        self.items: dict[str, WatchItem] = {}
        self.confirmed: list[WatchItem] = []
        self.rejected: list[str] = []

    async def add(self, signal: dict) -> bool:
        address = signal["token"]["address"]
        if address in self.items:
            return False
        if address in self.rejected:
            return False
        if len(self.items) >= WATCH_MAX_SLOTS:
            logger.info("Watchlist full (%d slots)", WATCH_MAX_SLOTS)
            return False

        price = signal["token"]["price_usd"]
        item = WatchItem(signal=signal, entry_price=price)
        self.items[address] = item
        logger.info(
            "WATCHLIST ADD: %s at $%.10f (score=%d)",
            item.symbol, price, signal["total_score"],
        )
        return True

    async def check(self) -> list[dict]:
        if not self.items:
            return []

        ready_signals = []

        for address in list(self.items.keys()):
            item = self.items.get(address)
            if not item:
                continue

            price = await self._get_price(address)
            if price > 0:
                item.update_price(price)

            if item.is_confirmed():
                item.signal["token"]["price_usd"] = item.current_price
                item.signal["watchlist_data"] = {
                    "watch_time_sec": int(item.age_sec),
                    "entry_price": item.entry_price,
                    "confirmed_price": item.current_price,
                    "growth_during_watch": round(item.change_pct, 1),
                    "peak_during_watch": round(item.peak_change_pct, 1),
                    "checks": item.checks,
                }
                ready_signals.append(item.signal)
                self.confirmed.append(item)
                del self.items[address]
                logger.info(
                    "WATCHLIST CONFIRMED: %s +%.1f%% in %ds (peak +%.1f%%)",
                    item.symbol, item.change_pct, int(item.age_sec), item.peak_change_pct,
                )

            elif item.is_rejected():
                self.rejected.append(address)
                del self.items[address]
                logger.info(
                    "WATCHLIST REJECTED: %s %.1f%% in %ds (peak +%.1f%%)",
                    item.symbol, item.change_pct, int(item.age_sec), item.peak_change_pct,
                )

        return ready_signals

    async def _get_price(self, address: str) -> float:
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
            logger.error("Watchlist price error for %s: %s", address[:12], e)
        return 0.0

    def get_stats(self) -> dict:
        return {
            "watching": len(self.items),
            "confirmed_total": len(self.confirmed),
            "rejected_total": len(self.rejected),
            "items": [
                {
                    "symbol": item.symbol,
                    "change_pct": round(item.change_pct, 1),
                    "peak_pct": round(item.peak_change_pct, 1),
                    "age_sec": int(item.age_sec),
                    "checks": item.checks,
                }
                for item in self.items.values()
            ],
        }
