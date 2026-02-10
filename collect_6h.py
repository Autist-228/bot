"""
Autonomous 6-hour data collector for pump.fun tokens.
Tracks each token for 30 minutes with price checks at 1m, 5m, 10m, 20m, 30m.
Saves incrementally every 30 minutes. Runs fully unattended.
"""
import asyncio
import json
import logging
import time
import os
import signal
import sys
from datetime import datetime, timezone

import httpx
import websockets

from config import PUMPPORTAL_WS_URL, DEXSCREENER_API, DATA_DIR

DURATION = int(os.getenv("COLLECT_6H_DURATION", "25200"))
SAVE_INTERVAL = 1800
PRICE_CHECK_MINUTES = [1, 5, 10, 20, 30]
ROCKET_THRESHOLD = 500
WINNER_THRESHOLD = 100

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"collect6h_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file),
    ],
)
log = logging.getLogger("collect6h")

tokens: dict[str, dict] = {}
trade_counts: dict[str, dict] = {}
stats = {
    "total": 0, "enriched": 0, "migrated": 0, "trades": 0,
    "price_checks": 0, "saves": 0, "errors": 0, "start": 0,
}
SOL_PRICE_USD = 170.0
shutdown = False


def handle_signal(sig, frame):
    global shutdown
    log.info("Shutdown signal received, saving data...")
    shutdown = True


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


async def fetch_sol_price(client: httpx.AsyncClient):
    global SOL_PRICE_USD
    try:
        r = await client.get(
            f"{DEXSCREENER_API}/tokens/v1/solana/So11111111111111111111111111111111111111112",
            timeout=10,
        )
        if r.status_code == 200:
            pairs = r.json()
            if pairs and isinstance(pairs, list):
                SOL_PRICE_USD = float(pairs[0].get("priceUsd") or 170)
                log.info("SOL price: $%.2f", SOL_PRICE_USD)
    except Exception as e:
        log.warning("SOL price fetch failed: %s", e)


async def sol_price_loop(client: httpx.AsyncClient):
    while not shutdown:
        await asyncio.sleep(300)
        await fetch_sol_price(client)


async def listen_pumpportal():
    ws = None
    while not shutdown:
        try:
            ws = await websockets.connect(PUMPPORTAL_WS_URL, ping_interval=20, ping_timeout=10)
            log.info("Connected to PumpPortal WebSocket")
            await ws.send(json.dumps({"method": "subscribeNewToken"}))
            await ws.send(json.dumps({"method": "subscribeMigration"}))
            log.info("Subscribed: newToken + migration + trades")

            async for raw in ws:
                if shutdown:
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if msg.get("txType") == "migration":
                    mint = msg.get("mint", "")
                    if mint and mint in tokens:
                        tokens[mint]["migrated"] = True
                        tokens[mint]["migrated_at"] = time.time()
                        tokens[mint]["migration_age_sec"] = int(time.time() - tokens[mint]["created_ts"])
                        stats["migrated"] += 1
                        log.info("MIGRATED: %s after %ds", tokens[mint]["symbol"], tokens[mint]["migration_age_sec"])
                    continue

                if msg.get("txType") == "create":
                    mint = msg.get("mint", "")
                    if not mint or mint in tokens:
                        continue

                    now = time.time()
                    symbol = msg.get("symbol", "???")
                    name = msg.get("name", "")
                    v_sol = float(msg.get("vSolInBondingCurve") or 0)
                    v_tokens = float(msg.get("vTokensInBondingCurve") or 0)
                    mcap_sol = float(msg.get("marketCapSol") or 0)
                    init_buy = float(msg.get("initialBuy") or 0) / 1e9 if msg.get("initialBuy") else 0
                    price_sol = v_sol / v_tokens if v_tokens > 0 else 0
                    price_usd = price_sol * SOL_PRICE_USD
                    mcap_usd = mcap_sol * SOL_PRICE_USD

                    tokens[mint] = {
                        "mint": mint,
                        "symbol": symbol,
                        "name": name,
                        "created_ts": now,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "initial_buy_sol": init_buy,
                        "initial_price_sol": price_sol,
                        "initial_price_usd": price_usd,
                        "initial_mcap_usd": mcap_usd,
                        "initial_mcap_sol": mcap_sol,
                        "v_tokens_in_bonding": v_tokens,
                        "v_sol_in_bonding": v_sol,
                        "dev_address": msg.get("traderPublicKey", ""),
                        "migrated": False,
                        "migrated_at": None,
                        "migration_age_sec": None,
                        "peak_mcap_usd": mcap_usd,
                        "peak_mcap_sol": mcap_sol,
                        "peak_mcap_age_sec": 0,
                        "enriched": False,
                        "enrich_tried": False,
                    }
                    trade_counts[mint] = {
                        "buys": 0, "sells": 0,
                        "buy_sol": 0.0, "sell_sol": 0.0,
                        "unique_buyers": set(),
                        "unique_sellers": set(),
                        "first_trade_ts": None,
                        "last_trade_ts": None,
                    }
                    stats["total"] += 1

                    if stats["total"] % 50 == 0:
                        elapsed = int(now - stats["start"])
                        log.info("[%ds] #%d tokens | trades=%d | mcap=$%.0f | %s",
                                 elapsed, stats["total"], stats["trades"], mcap_usd, symbol)

                    await ws.send(json.dumps({
                        "method": "subscribeTokenTrade",
                        "keys": [mint],
                    }))

                elif msg.get("txType") in ("buy", "sell"):
                    mint = msg.get("mint", "")
                    if mint not in trade_counts:
                        continue
                    tx_type = msg["txType"]
                    sol_amount = float(msg.get("solAmount") or 0)
                    trader = msg.get("traderPublicKey", "")
                    tc = trade_counts[mint]
                    now = time.time()

                    if tx_type == "buy":
                        tc["buys"] += 1
                        tc["buy_sol"] += sol_amount
                        tc["unique_buyers"].add(trader)
                    else:
                        tc["sells"] += 1
                        tc["sell_sol"] += sol_amount
                        tc["unique_sellers"].add(trader)

                    if tc["first_trade_ts"] is None:
                        tc["first_trade_ts"] = now
                    tc["last_trade_ts"] = now

                    new_mcap_sol = float(msg.get("marketCapSol") or 0)
                    if new_mcap_sol > 0:
                        new_mcap_usd = new_mcap_sol * SOL_PRICE_USD
                        token = tokens[mint]
                        token["latest_mcap_sol"] = new_mcap_sol
                        token["latest_mcap_usd"] = new_mcap_usd
                        if new_mcap_usd > token.get("peak_mcap_usd", 0):
                            token["peak_mcap_usd"] = new_mcap_usd
                            token["peak_mcap_sol"] = new_mcap_sol
                            token["peak_mcap_age_sec"] = int(now - token["created_ts"])

                    stats["trades"] += 1

        except websockets.exceptions.ConnectionClosed:
            log.warning("PumpPortal disconnected, reconnecting in 3s...")
            await asyncio.sleep(3)
        except Exception as e:
            log.error("PumpPortal error: %s, reconnecting in 5s...", e)
            stats["errors"] += 1
            await asyncio.sleep(5)
        finally:
            if ws:
                try:
                    await ws.close()
                except Exception:
                    pass


async def price_checker_loop(client: httpx.AsyncClient):
    while not shutdown:
        await asyncio.sleep(10)
        now = time.time()
        batch: list[tuple[str, int]] = []

        for mint, token in list(tokens.items()):
            age_sec = now - token["created_ts"]
            if age_sec > 2100:
                continue
            for minutes in PRICE_CHECK_MINUTES:
                target_sec = minutes * 60
                key = f"price_{minutes}m"
                if age_sec >= target_sec and not token.get(key):
                    batch.append((mint, minutes))
                    break
            if len(batch) >= 5:
                break

        for mint, minutes in batch:
            try:
                r = await client.get(
                    f"{DEXSCREENER_API}/tokens/v1/solana/{mint}",
                    timeout=10,
                )
                if r.status_code != 200:
                    tokens[mint][f"price_{minutes}m"] = 0
                    tokens[mint][f"change_{minutes}m"] = -100.0
                    continue

                pairs = r.json()
                token = tokens[mint]

                if not pairs or not isinstance(pairs, list) or len(pairs) == 0:
                    token[f"price_{minutes}m"] = 0
                    token[f"change_{minutes}m"] = -100.0
                    continue

                p = pairs[0]
                price_usd = float(p.get("priceUsd") or 0)
                token[f"price_{minutes}m"] = price_usd

                initial = token["initial_price_usd"]
                if initial > 0 and price_usd > 0:
                    change = ((price_usd - initial) / initial) * 100
                    token[f"change_{minutes}m"] = round(change, 1)
                elif price_usd == 0:
                    token[f"change_{minutes}m"] = -100.0

                liq = float((p.get("liquidity") or {}).get("usd") or 0)
                mcap = float(p.get("marketCap") or 0)
                buys_5m = int((p.get("txns") or {}).get("m5", {}).get("buys") or 0)
                sells_5m = int((p.get("txns") or {}).get("m5", {}).get("sells") or 0)
                vol_5m = float((p.get("volume") or {}).get("m5") or 0)

                token[f"liq_{minutes}m"] = liq
                token[f"mcap_{minutes}m"] = mcap
                token[f"buys_{minutes}m"] = buys_5m
                token[f"sells_{minutes}m"] = sells_5m
                token[f"vol_{minutes}m"] = vol_5m

                if minutes == 1:
                    token["dex_liquidity_usd"] = liq
                    token["dex_market_cap"] = mcap
                    token["dex_buys_5m"] = buys_5m
                    token["dex_sells_5m"] = sells_5m
                    token["dex_volume_5m"] = vol_5m
                    token["dex_fdv"] = float(p.get("fdv") or 0)
                    token["dex_change_5m"] = float((p.get("priceChange") or {}).get("m5") or 0)
                    info = p.get("info", {}) or {}
                    token["has_website"] = len(info.get("websites", [])) > 0
                    token["has_socials"] = len(info.get("socials", [])) > 0

                stats["price_checks"] += 1
                await asyncio.sleep(1.2)
            except Exception as e:
                stats["errors"] += 1
                log.debug("Price check error %s %dm: %s", mint[:8], minutes, e)


async def enrich_batch(client: httpx.AsyncClient):
    while not shutdown:
        await asyncio.sleep(20)
        now = time.time()
        batch = []
        for mint, token in list(tokens.items()):
            age = now - token["created_ts"]
            if 90 <= age <= 300 and not token.get("enriched") and not token.get("enrich_tried"):
                batch.append(mint)
            if len(batch) >= 3:
                break

        for mint in batch:
            tokens[mint]["enrich_tried"] = True
            try:
                r = await client.get(
                    f"{DEXSCREENER_API}/tokens/v1/solana/{mint}",
                    timeout=10,
                )
                if r.status_code != 200:
                    continue
                pairs = r.json()
                if not pairs or not isinstance(pairs, list) or len(pairs) == 0:
                    continue
                p = pairs[0]
                token = tokens[mint]
                token["dex_price_usd"] = float(p.get("priceUsd") or 0)
                token["dex_liquidity_usd"] = float((p.get("liquidity") or {}).get("usd") or 0)
                token["dex_volume_5m"] = float((p.get("volume") or {}).get("m5") or 0)
                token["dex_volume_1h"] = float((p.get("volume") or {}).get("h1") or 0)
                token["dex_buys_5m"] = int((p.get("txns") or {}).get("m5", {}).get("buys") or 0)
                token["dex_sells_5m"] = int((p.get("txns") or {}).get("m5", {}).get("sells") or 0)
                token["dex_buys_1h"] = int((p.get("txns") or {}).get("h1", {}).get("buys") or 0)
                token["dex_sells_1h"] = int((p.get("txns") or {}).get("h1", {}).get("sells") or 0)
                token["dex_market_cap"] = float(p.get("marketCap") or 0)
                token["dex_fdv"] = float(p.get("fdv") or 0)
                token["dex_change_5m"] = float((p.get("priceChange") or {}).get("m5") or 0)
                token["dex_change_1h"] = float((p.get("priceChange") or {}).get("h1") or 0)
                info = p.get("info", {}) or {}
                token["has_website"] = len(info.get("websites", [])) > 0
                token["has_socials"] = len(info.get("socials", [])) > 0
                token["enriched"] = True
                stats["enriched"] += 1
                await asyncio.sleep(1.2)
            except Exception as e:
                stats["errors"] += 1


def finalize_tokens() -> list[dict]:
    result = []
    for mint, token in tokens.items():
        tc = trade_counts.get(mint, {})
        t = dict(token)
        t["total_buys"] = tc.get("buys", 0)
        t["total_sells"] = tc.get("sells", 0)
        t["total_buy_sol"] = round(tc.get("buy_sol", 0), 4)
        t["total_sell_sol"] = round(tc.get("sell_sol", 0), 4)
        t["buy_sell_ratio"] = round(tc.get("buys", 0) / max(1, tc.get("sells", 1)), 2)
        t["sell_pressure"] = round(tc.get("sells", 0) / max(1, tc.get("buys", 0) + tc.get("sells", 0)) * 100, 1)
        t["unique_buyers"] = len(tc.get("unique_buyers", set()))
        t["unique_sellers"] = len(tc.get("unique_sellers", set()))

        best_change = None
        for key in ["change_30m", "change_20m", "change_10m", "change_5m", "change_1m"]:
            if t.get(key) is not None:
                if best_change is None or t[key] > best_change:
                    best_change = t[key]

        peak_mcap = t.get("peak_mcap_usd", 0)
        initial_mcap = t.get("initial_mcap_usd", 0)
        if initial_mcap > 0 and peak_mcap > 0:
            peak_change = ((peak_mcap - initial_mcap) / initial_mcap) * 100
            t["peak_change_pct"] = round(peak_change, 1)
            if best_change is None or peak_change > best_change:
                best_change = peak_change

        t["best_change_pct"] = round(best_change, 1) if best_change is not None else None

        if best_change is None:
            t["outcome"] = "no_data"
        elif best_change >= ROCKET_THRESHOLD:
            t["outcome"] = "ROCKET"
        elif best_change >= WINNER_THRESHOLD:
            t["outcome"] = "winner"
        elif best_change >= 0:
            t["outcome"] = "flat"
        elif best_change >= -50:
            t["outcome"] = "loser"
        else:
            t["outcome"] = "dead"

        result.append(t)
    return result


def save_data(tag: str = ""):
    finalized = finalize_tokens()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"collect6h_{tag}_{ts}.json" if tag else f"collect6h_{ts}.json"
    filepath = os.path.join(DATA_DIR, filename)

    outcomes: dict[str, int] = {}
    for t in finalized:
        o = t["outcome"]
        outcomes[o] = outcomes.get(o, 0) + 1

    duration = time.time() - stats["start"]
    rockets = [t for t in finalized if t["outcome"] == "ROCKET"]
    winners = [t for t in finalized if t["outcome"] == "winner"]

    output = {
        "meta": {
            "start_time": datetime.fromtimestamp(stats["start"], tz=timezone.utc).isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "duration_sec": int(duration),
            "duration_hours": round(duration / 3600, 2),
            "sol_price_usd": SOL_PRICE_USD,
            "total_tokens": len(finalized),
            "enriched": stats["enriched"],
            "migrated": stats["migrated"],
            "total_trades_tracked": stats["trades"],
            "price_checks_done": stats["price_checks"],
            "errors": stats["errors"],
            "saves_done": stats["saves"],
            "tokens_per_min": round(len(finalized) / max(1, duration / 60), 1),
            "outcomes": outcomes,
            "rockets_count": len(rockets),
            "winners_count": len(winners),
            "rocket_threshold": f"+{ROCKET_THRESHOLD}%",
            "winner_threshold": f"+{WINNER_THRESHOLD}%",
            "tracking_duration": "30 minutes per token",
            "price_check_intervals": PRICE_CHECK_MINUTES,
        },
        "tokens": finalized,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, default=str)

    stats["saves"] += 1

    log.info("=" * 60)
    log.info("SAVED #%d: %d tokens to %s", stats["saves"], len(finalized), filename)
    log.info("Rate: %.1f tokens/min | Duration: %.1f hours", output["meta"]["tokens_per_min"], duration / 3600)
    log.info("Outcomes: %s", outcomes)
    if rockets:
        log.info("ROCKETS (+%d%%+):", ROCKET_THRESHOLD)
        for r in rockets[:10]:
            log.info("  %s: peak=+%.0f%% mcap=$%.0f→$%.0f buys=%d",
                     r["symbol"], r.get("peak_change_pct", 0), r["initial_mcap_usd"],
                     r.get("peak_mcap_usd", 0), r["total_buys"])
    log.info("=" * 60)
    return filepath


async def stats_printer():
    while not shutdown:
        await asyncio.sleep(120)
        elapsed = int(time.time() - stats["start"])
        hours = elapsed / 3600
        tpm = stats["total"] / max(1, elapsed / 60)

        outcomes: dict[str, int] = {}
        for token in tokens.values():
            peak = token.get("peak_mcap_usd", 0)
            initial = token.get("initial_mcap_usd", 0)
            if initial > 0 and peak > 0:
                change = ((peak - initial) / initial) * 100
                if change >= ROCKET_THRESHOLD:
                    outcomes["ROCKET"] = outcomes.get("ROCKET", 0) + 1
                elif change >= WINNER_THRESHOLD:
                    outcomes["winner"] = outcomes.get("winner", 0) + 1

        log.info(
            "=== %.1fh | tokens=%d (%.1f/min) | trades=%d | checks=%d | rockets=%d winners=%d ===",
            hours, stats["total"], tpm, stats["trades"], stats["price_checks"],
            outcomes.get("ROCKET", 0), outcomes.get("winner", 0),
        )


async def incremental_saver():
    save_num = 0
    while not shutdown:
        await asyncio.sleep(SAVE_INTERVAL)
        if shutdown:
            break
        save_num += 1
        save_data(tag=f"inc{save_num}")
        log.info("Incremental save #%d done", save_num)


async def cleanup_old_subscriptions():
    while not shutdown:
        await asyncio.sleep(300)
        now = time.time()
        old_mints = [m for m, t in tokens.items() if now - t["created_ts"] > 2400]
        if old_mints:
            for mint in old_mints:
                tc = trade_counts.get(mint)
                if tc and "unique_buyers" in tc:
                    tc["unique_buyers_count"] = len(tc["unique_buyers"])
                    tc["unique_sellers_count"] = len(tc["unique_sellers"])
                    tc["unique_buyers"] = set()
                    tc["unique_sellers"] = set()


async def main():
    global shutdown
    log.info("=" * 70)
    log.info("6-HOUR AUTONOMOUS COLLECTOR")
    log.info("Duration: %d seconds (%.1f hours)", DURATION, DURATION / 3600)
    log.info("Token tracking: 30 minutes per token")
    log.info("Price checks: %s minutes", PRICE_CHECK_MINUTES)
    log.info("ROCKET threshold: +%d%%", ROCKET_THRESHOLD)
    log.info("WINNER threshold: +%d%%", WINNER_THRESHOLD)
    log.info("Incremental saves: every %d minutes", SAVE_INTERVAL // 60)
    log.info("Log file: %s", log_file)
    log.info("Data dir: %s", DATA_DIR)
    log.info("=" * 70)

    stats["start"] = time.time()

    async with httpx.AsyncClient() as client:
        await fetch_sol_price(client)

        tasks = [
            asyncio.create_task(listen_pumpportal()),
            asyncio.create_task(price_checker_loop(client)),
            asyncio.create_task(enrich_batch(client)),
            asyncio.create_task(sol_price_loop(client)),
            asyncio.create_task(stats_printer()),
            asyncio.create_task(incremental_saver()),
            asyncio.create_task(cleanup_old_subscriptions()),
        ]

        try:
            await asyncio.sleep(DURATION)
        except asyncio.CancelledError:
            pass

        shutdown = True
        log.info("Collection period ended (%d hours). Final save...", DURATION // 3600)

        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

    filepath = save_data(tag="FINAL")
    elapsed = time.time() - stats["start"]
    log.info("=" * 70)
    log.info("COLLECTION COMPLETE!")
    log.info("Duration: %.1f hours", elapsed / 3600)
    log.info("Total tokens: %d", stats["total"])
    log.info("Total trades: %d", stats["trades"])
    log.info("Price checks: %d", stats["price_checks"])
    log.info("Final data: %s", filepath)
    log.info("=" * 70)
    return filepath


if __name__ == "__main__":
    asyncio.run(main())
