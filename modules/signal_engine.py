import asyncio
import logging
import json

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SIGNAL_THRESHOLD
from modules.codex_tracker import get_trending_tokens
from modules.twitter_monitor import search_token_mentions, compute_social_score
from modules.safety_check import check_token_safety, check_honeypot_helius
from utils.database import already_signaled, save_signal

logger = logging.getLogger(__name__)


def compute_onchain_score(token: dict) -> int:
    score = 0

    vol = token.get("volume_24h", 0)
    if vol >= 20000:
        score += 2
    elif vol >= 5000:
        score += 1

    change_5m = token.get("change_5m", 0)
    if change_5m >= 15:
        score += 2
    elif change_5m >= 5:
        score += 1

    unique_buys = token.get("unique_buys_5m", 0)
    if unique_buys >= 8:
        score += 2
    elif unique_buys >= 4:
        score += 1

    holders = token.get("holders", 0)
    if 30 <= holders <= 300:
        score += 1

    buy_5m = token.get("buy_count_5m", 0)
    sell_5m = token.get("sell_count_5m", 0)
    if sell_5m > 0 and buy_5m / sell_5m >= 2:
        score += 1
    elif sell_5m == 0 and buy_5m >= 5:
        score += 1

    return min(score, 8)


async def scan_and_score() -> list[dict]:
    signals = []

    tokens = await get_trending_tokens(limit=25)
    logger.info("Codex returned %d trending tokens", len(tokens))

    for token in tokens:
        address = token["address"]
        symbol = token["symbol"]

        if await already_signaled(address):
            continue

        onchain_score = compute_onchain_score(token)

        await asyncio.sleep(1.5)
        social_data = await search_token_mentions(symbol)
        social_score = compute_social_score(social_data)

        safety = await check_token_safety(address, token)
        safety_score = safety["safety_score"]

        honeypot = await check_honeypot_helius(address)

        if onchain_score < 2:
            logger.debug("SKIP %s: weak on-chain score %d", symbol, onchain_score)
            continue

        top_pct = safety.get("top_holders_pct")
        if top_pct is not None and top_pct > 30:
            logger.debug("SKIP %s: top holders %.1f%%", symbol, top_pct)
            continue

        total_score = 0

        if onchain_score >= 5:
            total_score += 4
        elif onchain_score >= 3:
            total_score += 3
        elif onchain_score >= 2:
            total_score += 2

        tweet_count = social_data.get("tweet_count", 0)
        total_likes = social_data.get("total_likes", 0)
        influencers = social_data.get("influencer_mentions", 0)

        if tweet_count == 0:
            total_score -= 2
        elif tweet_count >= 15 and total_likes >= 5:
            total_score += 3
        elif social_score >= 3:
            total_score += 2
        elif social_score >= 2:
            total_score += 1

        if influencers >= 2:
            total_score += 1

        if safety_score >= 4:
            total_score += 2
        elif safety_score >= 3:
            total_score += 1

        if honeypot.get("mint_disabled") and honeypot.get("freeze_disabled"):
            total_score += 1

        if not safety["is_safe"]:
            total_score = max(0, total_score - 2)

        details = {
            "onchain_score": onchain_score,
            "social_score": social_score,
            "safety_score": safety_score,
            "total_score": total_score,
            "safety_warnings": safety["warnings"],
            "top_holders_pct": safety["top_holders_pct"],
            "mint_disabled": honeypot.get("mint_disabled"),
            "freeze_disabled": honeypot.get("freeze_disabled"),
            "social": {
                "tweet_count": social_data.get("tweet_count", 0),
                "influencer_mentions": social_data.get("influencer_mentions", 0),
                "total_likes": social_data.get("total_likes", 0),
                "total_views": social_data.get("total_views", 0),
                "top_tweet": social_data.get("top_tweet"),
            },
            "onchain": {
                "volume_24h": token.get("volume_24h"),
                "change_5m": token.get("change_5m"),
                "change_1h": token.get("change_1h"),
                "buy_count_5m": token.get("buy_count_5m"),
                "holders": token.get("holders"),
                "unique_buys_5m": token.get("unique_buys_5m"),
            },
        }

        if total_score >= SIGNAL_THRESHOLD:
            signal_id = await save_signal(
                token_address=address,
                token_symbol=symbol,
                token_name=token["name"],
                network_id=token["network_id"],
                price=token["price_usd"],
                liquidity=token["liquidity"],
                volume=token["volume_24h"],
                score=total_score,
                details=json.dumps(details, default=str),
            )
            signals.append({
                "signal_id": signal_id,
                "token": token,
                "details": details,
                "total_score": total_score,
                "social_data": social_data,
                "safety": safety,
                "honeypot": honeypot,
            })
            logger.info(
                "SIGNAL: %s (%s) score=%d [on-chain=%d social=%d safety=%d]",
                symbol, address[:12], total_score,
                onchain_score, social_score, safety_score,
            )

    return signals
