import logging
import httpx

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import HELIUS_API_KEY, MAX_TOP_HOLDER_PERCENT
from modules.codex_tracker import get_top_holders_percent

logger = logging.getLogger(__name__)

HELIUS_API_URL = f"https://api-mainnet.helius-rpc.com/v0"


async def check_token_safety(token_address: str, token_data: dict) -> dict:
    results = {
        "is_safe": True,
        "warnings": [],
        "top_holders_pct": None,
        "holder_check_passed": True,
        "liquidity_check_passed": True,
        "volume_ratio_ok": True,
        "buy_sell_ratio_ok": True,
        "safety_score": 0,
    }

    top_pct = await get_top_holders_percent(token_address)
    results["top_holders_pct"] = top_pct
    if top_pct is not None and top_pct > MAX_TOP_HOLDER_PERCENT:
        results["holder_check_passed"] = False
        results["warnings"].append(
            f"Top 10 holders own {top_pct:.1f}% (>{MAX_TOP_HOLDER_PERCENT}%)"
        )
        results["is_safe"] = False

    liquidity = token_data.get("liquidity", 0)
    volume = token_data.get("volume_24h", 0)
    if liquidity > 0 and volume > 0:
        vol_liq_ratio = volume / liquidity
        if vol_liq_ratio > 50:
            results["volume_ratio_ok"] = False
            results["warnings"].append(
                f"Suspicious vol/liq ratio: {vol_liq_ratio:.1f}x"
            )

    buys_5m = token_data.get("buy_count_5m", 0)
    sells_5m = token_data.get("sell_count_5m", 0)
    if buys_5m > 0 and sells_5m > 0:
        ratio = buys_5m / sells_5m
        if ratio > 20:
            results["buy_sell_ratio_ok"] = False
            results["warnings"].append(
                f"Extreme buy/sell ratio: {ratio:.1f}x (possible wash trading)"
            )

    unique_buys = token_data.get("unique_buys_5m", 0)
    total_buys = token_data.get("buy_count_5m", 0)
    if total_buys > 10 and unique_buys > 0:
        if total_buys / unique_buys > 5:
            results["warnings"].append(
                f"Low unique buyers vs total buys ({unique_buys}/{total_buys})"
            )

    score = 0
    if results["holder_check_passed"]:
        score += 2
    if results["liquidity_check_passed"]:
        score += 1
    if results["volume_ratio_ok"]:
        score += 1
    if results["buy_sell_ratio_ok"]:
        score += 1
    if not results["warnings"]:
        score += 1

    results["safety_score"] = score
    return results


async def check_honeypot_helius(token_address: str) -> dict:
    url = f"{HELIUS_API_URL}/token-metadata"
    params = {"api-key": HELIUS_API_KEY}
    payload = {
        "mintAccounts": [token_address],
        "includeOffChain": True,
        "disableCache": False,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload, params=params)
            resp.raise_for_status()
            data = resp.json()
            if data and len(data) > 0:
                meta = data[0]
                on_chain = meta.get("onChainAccountInfo", {})
                account_info = on_chain.get("accountInfo", {})
                token_info = account_info.get("data", {}).get("parsed", {}).get("info", {})
                mint_authority = token_info.get("mintAuthority")
                freeze_authority = token_info.get("freezeAuthority")
                return {
                    "mint_authority": mint_authority,
                    "freeze_authority": freeze_authority,
                    "mint_disabled": mint_authority is None,
                    "freeze_disabled": freeze_authority is None,
                }
    except Exception as e:
        logger.error("Helius metadata check error: %s", e)
    return {
        "mint_authority": "unknown",
        "freeze_authority": "unknown",
        "mint_disabled": False,
        "freeze_disabled": False,
    }
