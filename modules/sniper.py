import asyncio
import logging
import time

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CODEX_API_KEY, CODEX_GRAPHQL_URL, SOLANA_NETWORK_ID
from modules.codex_tracker import _query, _safe_float

logger = logging.getLogger(__name__)

SNIPER_MAX_AGE_SEC = 120
SNIPER_MIN_LIQUIDITY = 3000
SNIPER_MAX_HOLDERS = 200
SNIPER_MIN_BUYS_5M = 5
SNIPER_MAX_DEV_HELD = 5.0
SNIPER_MAX_BOT_HELD = 20.0

SNIPER_QUERY = """
query FilterTokens($filters: TokenFilters, $limit: Int, $rankings: [TokenRanking]) {
    filterTokens(filters: $filters, limit: $limit, rankings: $rankings) {
        results {
            token {
                address
                symbol
                name
                networkId
                isScam
            }
            priceUSD
            liquidity
            volume24
            buyCount5m
            sellCount5m
            change5m
            holders
            createdAt
            marketCap
            uniqueBuys5m
            sniperCount
            bundlerCount
            insiderCount
            devHeldPercentage
            sniperHeldPercentage
            bundlerHeldPercentage
            insiderHeldPercentage
        }
    }
}
"""


async def scan_new_tokens() -> list[dict]:
    now = int(time.time())
    created_after = now - SNIPER_MAX_AGE_SEC

    variables = {
        "filters": {
            "network": [SOLANA_NETWORK_ID],
            "liquidity": {"gte": SNIPER_MIN_LIQUIDITY},
            "createdAt": {"gte": created_after},
            "holders": {"lte": SNIPER_MAX_HOLDERS},
            "buyCount5m": {"gte": SNIPER_MIN_BUYS_5M},
        },
        "limit": 30,
        "rankings": [{"attribute": "createdAt", "direction": "DESC"}],
    }

    data = await _query(SNIPER_QUERY, variables)
    results = data.get("filterTokens", {}).get("results", [])

    candidates = []
    for r in results:
        token_info = r.get("token", {})
        symbol = token_info.get("symbol", "???")

        if token_info.get("isScam"):
            logger.info("SNIPER SKIP %s: scam", symbol)
            continue

        dev_held = _safe_float(r.get("devHeldPercentage"))
        if dev_held > SNIPER_MAX_DEV_HELD:
            logger.info("SNIPER SKIP %s: dev holds %.1f%%", symbol, dev_held)
            continue

        sniper_held = _safe_float(r.get("sniperHeldPercentage"))
        bundler_held = _safe_float(r.get("bundlerHeldPercentage"))
        insider_held = _safe_float(r.get("insiderHeldPercentage"))
        bot_total = sniper_held + bundler_held + insider_held
        if bot_total > SNIPER_MAX_BOT_HELD:
            logger.info("SNIPER SKIP %s: bots hold %.1f%%", symbol, bot_total)
            continue

        sell_5m = r.get("sellCount5m", 0) or 0
        buy_5m = r.get("buyCount5m", 0) or 0
        if sell_5m > 0 and buy_5m > 0 and sell_5m / buy_5m > 0.4:
            logger.info("SNIPER SKIP %s: high sell ratio %.0f%%", symbol, sell_5m / buy_5m * 100)
            continue

        change_5m = _safe_float(r.get("change5m"))
        if change_5m < -20:
            logger.info("SNIPER SKIP %s: already dumping %.1f%%", symbol, change_5m)
            continue

        age_sec = now - (r.get("createdAt") or now)
        liquidity = _safe_float(r.get("liquidity"))
        holders = r.get("holders", 0) or 0
        price_usd = _safe_float(r.get("priceUSD"))
        unique_buys = r.get("uniqueBuys5m", 0) or 0

        score = 7
        if unique_buys >= 10:
            score += 1
        if buy_5m >= 20:
            score += 1
        if liquidity >= 5000:
            score += 1

        candidates.append({
            "token": {
                "address": token_info.get("address", ""),
                "symbol": symbol,
                "name": token_info.get("name", ""),
                "network_id": token_info.get("networkId", SOLANA_NETWORK_ID),
                "price_usd": price_usd,
                "liquidity": liquidity,
                "volume_24h": _safe_float(r.get("volume24")),
                "holders": holders,
                "buy_count_5m": buy_5m,
                "unique_buys_5m": unique_buys,
                "change_5m": _safe_float(r.get("change5m")),
            },
            "total_score": score,
            "safety_score": 5,
            "details": {
                "safety_score": 5,
                "sniper_mode": True,
                "age_sec": age_sec,
                "dev_held_pct": dev_held,
                "bot_held_pct": bot_total,
                "liquidity": liquidity,
            },
            "social_data": {},
            "safety": {"is_safe": True, "warnings": [], "safety_score": 5, "top_holders_pct": None},
            "honeypot": {},
        })

        logger.info(
            "SNIPER CANDIDATE %s: age=%ds holders=%d liq=$%.0f buys=%d unique=%d dev=%.1f%% bots=%.1f%% score=%d",
            symbol, age_sec, holders, liquidity, buy_5m, unique_buys, dev_held, bot_total, score,
        )

    logger.info("SNIPER: %d raw, %d passed filters", len(results), len(candidates))
    return candidates
