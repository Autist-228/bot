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


async def get_trending_tokens(limit: int = 25) -> list[dict]:
    query = """
    query FilterTokens($filters: TokenFilters, $limit: Int, $rankings: [TokenRanking]) {
        filterTokens(filters: $filters, limit: $limit, rankings: $rankings) {
            results {
                token {
                    address
                    symbol
                    name
                    networkId
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
            }
        }
    }
    """
    now = int(time.time())
    variables = {
        "filters": {
            "network": [SOLANA_NETWORK_ID],
            "liquidity": {"gte": MIN_LIQUIDITY_USD},
            "volume24": {"gte": MIN_VOLUME_24H_USD},
            "buyCount5m": {"gte": MIN_BUY_COUNT_5M},
            "txnCount1": {"gte": 5},
            "createdAt": {"gte": now - MAX_TOKEN_AGE_HOURS * 3600},
            "holders": {"gte": MIN_HOLDERS, "lte": MAX_HOLDERS_EARLY},
        },
        "limit": limit,
        "rankings": [
            {"attribute": "buyCount5m", "direction": "DESC"}
        ],
    }
    data = await _query(query, variables)
    results = data.get("filterTokens", {}).get("results", [])
    tokens = []
    for r in results:
        token_info = r.get("token", {})
        tokens.append({
            "address": token_info.get("address", ""),
            "symbol": token_info.get("symbol", ""),
            "name": token_info.get("name", ""),
            "network_id": token_info.get("networkId", SOLANA_NETWORK_ID),
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
        })
    return tokens


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
