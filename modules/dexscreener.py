import httpx
import logging

logger = logging.getLogger(__name__)

DEXSCREENER_BASE = "https://api.dexscreener.com"


async def get_boosted_solana_tokens() -> list[dict]:
    url = f"{DEXSCREENER_BASE}/token-boosts/top/v1"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        tokens = []
        for item in data:
            if item.get("chainId") != "solana":
                continue
            addr = item.get("tokenAddress", "")
            if not addr:
                continue
            tokens.append({
                "address": addr,
                "boost_amount": item.get("totalAmount", 0),
                "description": (item.get("description") or "")[:100],
                "has_links": len(item.get("links", [])) > 0,
            })
        logger.info("DexScreener boosted: %d Solana tokens", len(tokens))
        return tokens
    except Exception as e:
        logger.error("DexScreener boosted error: %s", e)
        return []


async def get_latest_profiles_solana() -> list[dict]:
    url = f"{DEXSCREENER_BASE}/token-profiles/latest/v1"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        tokens = []
        for item in data:
            if item.get("chainId") != "solana":
                continue
            addr = item.get("tokenAddress", "")
            if not addr:
                continue
            tokens.append({
                "address": addr,
                "has_links": len(item.get("links", [])) > 0,
            })
        logger.info("DexScreener profiles: %d Solana tokens", len(tokens))
        return tokens
    except Exception as e:
        logger.error("DexScreener profiles error: %s", e)
        return []


async def get_dexscreener_signals() -> set[str]:
    import asyncio
    boosted, profiles = await asyncio.gather(
        get_boosted_solana_tokens(),
        get_latest_profiles_solana(),
        return_exceptions=True,
    )
    addresses = set()
    if not isinstance(boosted, Exception):
        for t in boosted:
            addresses.add(t["address"])
    if not isinstance(profiles, Exception):
        for t in profiles:
            addresses.add(t["address"])
    logger.info("DexScreener total unique Solana addresses: %d", len(addresses))
    return addresses
