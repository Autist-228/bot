import asyncio
import logging
import json

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SIGNAL_THRESHOLD
from modules.codex_tracker import get_trending_tokens, get_token_price
from modules.twitter_monitor import search_token_mentions, compute_social_score
from modules.safety_check import check_token_safety, check_honeypot_helius
from modules.dexscreener import get_dexscreener_signals
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
    pending_signals = []

    codex_task = get_trending_tokens(limit=50)
    dex_task = get_dexscreener_signals()
    tokens, dex_addresses = await asyncio.gather(codex_task, dex_task, return_exceptions=True)
    if isinstance(tokens, Exception):
        logger.error("Codex fetch failed: %s", tokens)
        tokens = []
    if isinstance(dex_addresses, Exception):
        logger.error("DexScreener fetch failed: %s", dex_addresses)
        dex_addresses = set()
    logger.info("Codex returned %d tokens (multi-query), DexScreener %d addresses", len(tokens), len(dex_addresses))

    for token in tokens:
        address = token["address"]
        symbol = token["symbol"]

        if await already_signaled(address):
            continue

        if token.get("is_scam"):
            logger.info("SKIP %s: flagged as scam by Codex", symbol)
            continue

        sniper_count = token.get("sniper_count", 0)
        bundler_count = token.get("bundler_count", 0)
        insider_count = token.get("insider_count", 0)
        dev_held = token.get("dev_held_pct", 0)
        sniper_held = token.get("sniper_held_pct", 0)
        bundler_held = token.get("bundler_held_pct", 0)
        insider_held = token.get("insider_held_pct", 0)

        if dev_held > 20:
            logger.info("SKIP %s: dev holds %.1f%% (rug risk)", symbol, dev_held)
            continue

        if sniper_held + bundler_held + insider_held > 40:
            logger.info(
                "SKIP %s: bots hold %.1f%% (sniper=%.1f%% bundler=%.1f%% insider=%.1f%%)",
                symbol, sniper_held + bundler_held + insider_held,
                sniper_held, bundler_held, insider_held,
            )
            continue

        onchain_score = compute_onchain_score(token)

        await asyncio.sleep(0.8)
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

        sell_5m = token.get("sell_count_5m", 0)
        buy_5m = token.get("buy_count_5m", 0)
        if sell_5m > 0 and buy_5m > 0 and sell_5m / buy_5m > 0.8:
            logger.debug("SKIP %s: high sell pressure %.1f%%", symbol, sell_5m / buy_5m * 100)
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
        twitter_error = social_data.get("error")

        if twitter_error:
            pass
        elif tweet_count == 0:
            total_score -= 2
        elif tweet_count >= 15 and total_likes >= 5:
            total_score += 3
        elif social_score >= 3:
            total_score += 2
        elif social_score >= 2:
            total_score += 1

        if not twitter_error and influencers >= 2:
            total_score += 1

        if safety_score >= 4:
            total_score += 2
        elif safety_score >= 3:
            total_score += 1

        if honeypot.get("mint_disabled") and honeypot.get("freeze_disabled"):
            total_score += 1

        if address in dex_addresses:
            total_score += 1
            logger.info("BOOST %s: +1 from DexScreener (boosted/profiled)", symbol)

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
            "bot_data": {
                "sniper_count": sniper_count,
                "bundler_count": bundler_count,
                "insider_count": insider_count,
                "dev_held_pct": dev_held,
                "sniper_held_pct": sniper_held,
                "bundler_held_pct": bundler_held,
                "insider_held_pct": insider_held,
            },
        }

        if total_score >= SIGNAL_THRESHOLD:
            pending_signals.append({
                "token": token,
                "details": details,
                "total_score": total_score,
                "social_data": social_data,
                "safety": safety,
                "honeypot": honeypot,
                "onchain_score": onchain_score,
                "social_score": social_score,
                "safety_score": safety_score,
            })

    if pending_signals:
        await asyncio.sleep(5)
        momentum_tasks = [
            get_token_price(s["token"]["address"]) for s in pending_signals
        ]
        prices_after = await asyncio.gather(*momentum_tasks, return_exceptions=True)

        for sig_data, price_after in zip(pending_signals, prices_after):
            token = sig_data["token"]
            address = token["address"]
            symbol = token["symbol"]
            total_score = sig_data["total_score"]
            price_before = token["price_usd"]

            if isinstance(price_after, Exception) or not price_after or price_after <= 0:
                price_after = price_before

            if price_before > 0:
                momentum = ((price_after - price_before) / price_before) * 100
                if momentum < -10:
                    logger.info(
                        "SKIP %s: negative momentum %.1f%% (price dropped during confirmation)",
                        symbol, momentum,
                    )
                    continue
                if momentum > 0:
                    total_score += 1
                token["price_usd"] = price_after

            signal_id = await save_signal(
                token_address=address,
                token_symbol=symbol,
                token_name=token["name"],
                network_id=token["network_id"],
                price=token["price_usd"],
                liquidity=token["liquidity"],
                volume=token["volume_24h"],
                score=total_score,
                details=json.dumps(sig_data["details"], default=str),
            )
            signals.append({
                "signal_id": signal_id,
                "token": token,
                "details": sig_data["details"],
                "total_score": total_score,
                "social_data": sig_data["social_data"],
                "safety": sig_data["safety"],
                "honeypot": sig_data["honeypot"],
            })
            logger.info(
                "SIGNAL: %s (%s) score=%d [on-chain=%d social=%d safety=%d]",
                symbol, address[:12], total_score,
                sig_data["onchain_score"], sig_data["social_score"], sig_data["safety_score"],
            )

    return signals
