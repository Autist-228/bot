import httpx
import logging
import time

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    CODEX_API_KEY, CODEX_GRAPHQL_URL, SOLANA_NETWORK_ID,
    MIN_LIQUIDITY_USD, MIN_VOLUME_24H_USD, MIN_BUY_COUNT_5M,
    MAX_TOKEN_AGE_HOURS, MAX_HOLDERS_EARLY, MIN_HOLDERS,
)

logger = logging.getLogger(__name__)

HEADERS = {
    "Authorization": CODEX_API_KEY,
    "Content-Type": "application/json",
}


async def _query(query: str, variables: dict | None = None) -> dict:
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(CODEX_GRAPHQL_URL, json=payload, headers=HEADERS)
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                logger.error("Codex GraphQL errors: %s", data["errors"])
            return data.get("data", {})
    except Exception as e:
        logger.error("Codex query error: %s", e)
        return {}


FILTER_QUERY = """
query FilterTokens($filters: TokenFilters, $limit: Int, $rankings: [TokenRanking]) {
    filterTokens(filters: $filters, limit: $limit, rankings: $rankings) {
        results {
            token {
                address
                symbol
                name
                networkId
                isScam
                info {
                    address
                    circulatingSupply
                    totalSupply
                }
            }
            priceUSD
            liquidity
            volume24
            buyCount5m
            sellCount5m
            buyCount1
            buyCount24
            change5m
            change1
            change24
            txnCount5m
            txnCount1
            txnCount24
            holders
            createdAt
            marketCap
            uniqueBuys5m
            uniqueSells5m
            uniqueBuys1
            uniqueBuys24
            sniperCount
            bundlerCount
            insiderCount
            devHeldPercentage
            sniperHeldPercentage
            bundlerHeldPercentage
            insiderHeldPercentage
            high5m
            low5m
            volumeChange5m
        }
    }
}
"""


def _parse_results(results: list) -> list[dict]:
    tokens = []
    for r in results:
        token_info = r.get("token", {})
        tokens.append({
            "address": token_info.get("address", ""),
            "symbol": token_info.get("symbol", ""),
            "name": token_info.get("name", ""),
            "network_id": token_info.get("networkId", SOLANA_NETWORK_ID),
            "is_scam": token_info.get("isScam", False),
            "price_usd": _safe_float(r.get("priceUSD")),
            "liquidity": _safe_float(r.get("liquidity")),
            "volume_24h": _safe_float(r.get("volume24")),
            "buy_count_5m": r.get("buyCount5m", 0),
            "sell_count_5m": r.get("sellCount5m", 0),
            "buy_count_1h": r.get("buyCount1", 0),
            "buy_count_24h": r.get("buyCount24", 0),
            "change_5m": _safe_float(r.get("change5m")),
            "change_1h": _safe_float(r.get("change1")),
            "change_24h": _safe_float(r.get("change24")),
            "txn_count_5m": r.get("txnCount5m", 0),
            "txn_count_1h": r.get("txnCount1", 0),
            "txn_count_24h": r.get("txnCount24", 0),
            "holders": r.get("holders", 0),
            "created_at": r.get("createdAt", 0),
            "market_cap": _safe_float(r.get("marketCap")),
            "unique_buys_5m": r.get("uniqueBuys5m", 0),
            "unique_buys_1h": r.get("uniqueBuys1", 0),
            "unique_buys_24h": r.get("uniqueBuys24", 0),
            "sniper_count": r.get("sniperCount", 0) or 0,
            "bundler_count": r.get("bundlerCount", 0) or 0,
            "insider_count": r.get("insiderCount", 0) or 0,
            "dev_held_pct": _safe_float(r.get("devHeldPercentage")),
            "sniper_held_pct": _safe_float(r.get("sniperHeldPercentage")),
            "bundler_held_pct": _safe_float(r.get("bundlerHeldPercentage")),
            "insider_held_pct": _safe_float(r.get("insiderHeldPercentage")),
            "high_5m": _safe_float(r.get("high5m")),
            "low_5m": _safe_float(r.get("low5m")),
            "volume_change_5m": _safe_float(r.get("volumeChange5m")),
        })
    return tokens


async def _fetch_ranked(ranking_attr: str, limit: int, extra_filters: dict | None = None) -> list[dict]:
    now = int(time.time())
    filters = {
        "network": [SOLANA_NETWORK_ID],
        "liquidity": {"gte": MIN_LIQUIDITY_USD},
        "volume24": {"gte": MIN_VOLUME_24H_USD},
        "buyCount5m": {"gte": MIN_BUY_COUNT_5M},
        "txnCount1": {"gte": 5},
        "createdAt": {"gte": now - MAX_TOKEN_AGE_HOURS * 3600},
        "holders": {"gte": MIN_HOLDERS, "lte": MAX_HOLDERS_EARLY},
    }
    if extra_filters:
        filters.update(extra_filters)
    variables = {
        "filters": filters,
        "limit": limit,
        "rankings": [{"attribute": ranking_attr, "direction": "DESC"}],
    }
    data = await _query(FILTER_QUERY, variables)
    results = data.get("filterTokens", {}).get("results", [])
    return _parse_results(results)


async def get_trending_tokens(limit: int = 50) -> list[dict]:
    import asyncio
    q1 = _fetch_ranked("change5m", limit)
    q2 = _fetch_ranked("buyCount5m", limit)
    q3 = _fetch_ranked("uniqueBuys5m", limit)

    results = await asyncio.gather(q1, q2, q3, return_exceptions=True)

    seen = set()
    merged = []
    for batch in results:
        if isinstance(batch, Exception):
            logger.error("Multi-query batch error: %s", batch)
            continue
        for token in batch:
            addr = token["address"]
            if addr in seen:
                continue
            seen.add(addr)
            merged.append(token)

    logger.info(
        "Multi-query: %d unique tokens from 3 rankings (change5m=%d, buyCount5m=%d, uniqueBuys5m=%d)",
        len(merged),
        len(results[0]) if not isinstance(results[0], Exception) else 0,
        len(results[1]) if not isinstance(results[1], Exception) else 0,
        len(results[2]) if not isinstance(results[2], Exception) else 0,
    )
    return merged


async def get_top_holders_percent(token_address: str) -> float | None:
    token_id = f"{token_address}:{SOLANA_NETWORK_ID}"
    query = """
    query Top10Holders($tokenId: String!) {
        top10HoldersPercent(tokenId: $tokenId)
    }
    """
    variables = {"tokenId": token_id}
    try:
        data = await _query(query, variables)
        val = data.get("top10HoldersPercent")
        if val is not None:
            return float(val)
    except Exception as e:
        logger.error("top10HoldersPercent error for %s: %s", token_address, e)
    return None


async def get_token_top_traders(token_address: str) -> list[dict]:
    query = """
    query TokenTopTraders($input: TokenTopTradersInput!) {
        tokenTopTraders(input: $input) {
            walletAddress
            tokenPnlUsd
            tokensBought
            tokensSold
            tokenBuyVolume
            tokenSellVolume
        }
    }
    """
    variables = {
        "input": {
            "tokenAddress": token_address,
            "networkId": SOLANA_NETWORK_ID,
        }
    }
    data = await _query(query, variables)
    return data.get("tokenTopTraders", [])


async def get_token_price(token_address: str) -> float | None:
    query = """
    query GetPrice($inputs: [GetPriceInput]) {
        getTokenPrices(inputs: $inputs) {
            priceUsd
        }
    }
    """
    variables = {
        "inputs": [{
            "address": token_address,
            "networkId": SOLANA_NETWORK_ID,
        }]
    }
    try:
        data = await _query(query, variables)
        prices = data.get("getTokenPrices", [])
        if prices and prices[0]:
            return prices[0].get("priceUsd")
    except Exception as e:
        logger.error("getTokenPrice error for %s: %s", token_address, e)
    return None


def _safe_float(val) -> float:
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0
