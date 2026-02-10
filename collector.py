import asyncio
import json
import logging
import time
import os
from datetime import datetime, timezone

import httpx
import websockets

from config import (
    PUMPPORTAL_WS_URL,
    DEXSCREENER_API,
    COLLECT_DURATION,
    DATA_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("collector")

tokens: dict[str, dict] = {}
trade_counts: dict[str, dict] = {}
migrations: set[str] = set()
stats = {"total": 0, "enriched": 0, "migrated": 0, "errors": 0, "trades": 0, "start": 0}

SOL_PRICE_USD = 200.0


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
                SOL_PRICE_USD = float(pairs[0].get("priceUsd") or 200)
                log.info("SOL price: $%.2f", SOL_PRICE_USD)
    except Exception as e:
        log.warning("Failed to fetch SOL price: %s, using $%.0f", e, SOL_PRICE_USD)


async def sol_price_updater(client: httpx.AsyncClient):
    while True:
        await asyncio.sleep(300)
        await fetch_sol_price(client)


async def enrich_batch(client: httpx.AsyncClient):
    while True:
        await asyncio.sleep(15)
        now = time.time()
        to_enrich = []
        for mint, token in list(tokens.items()):
            age = now - token["created_ts"]
            if age >= 120 and not token.get("enriched") and not token.get("enrich_tried"):
                to_enrich.append(mint)
            if len(to_enrich) >= 5:
                break

        for mint in to_enrich:
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
                price_usd = float(p.get("priceUsd") or 0)
                token["dex_price_usd"] = price_usd
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
                token["dex_dex_id"] = p.get("dexId", "")
                info = p.get("info", {}) or {}
                token["has_website"] = len(info.get("websites", [])) > 0
                token["has_socials"] = len(info.get("socials", [])) > 0

                if price_usd > 0 and token["initial_price_usd"] > 0:
                    change = ((price_usd - token["initial_price_usd"]) / token["initial_price_usd"]) * 100
                    token["change_at_enrich"] = round(change, 1)

                token["enriched"] = True
                stats["enriched"] += 1
                await asyncio.sleep(1.1)
            except Exception as e:
                stats["errors"] += 1
                log.debug("Enrich error %s: %s", mint[:8], e)


async def price_checker(client: httpx.AsyncClient):
    while True:
        await asyncio.sleep(20)
        now = time.time()
        batch = []
        for mint, token in list(tokens.items()):
            age = now - token["created_ts"]
            if age >= 300 and not token.get("price_5m"):
                batch.append((mint, 5))
            elif age >= 600 and not token.get("price_10m"):
                batch.append((mint, 10))
            if len(batch) >= 8:
                break

        for mint, minutes in batch:
            try:
                r = await client.get(
                    f"{DEXSCREENER_API}/tokens/v1/solana/{mint}",
                    timeout=10,
                )
                if r.status_code != 200:
                    continue
                pairs = r.json()
                if not pairs or not isinstance(pairs, list) or len(pairs) == 0:
                    token = tokens[mint]
                    token[f"price_{minutes}m"] = 0
                    token[f"change_{minutes}m"] = -100.0
                    continue
                price = float(pairs[0].get("priceUsd") or 0)
                token = tokens[mint]
                token[f"price_{minutes}m"] = price
                initial = token["initial_price_usd"]
                if initial > 0 and price > 0:
                    change = ((price - initial) / initial) * 100
                    token[f"change_{minutes}m"] = round(change, 1)
                elif price == 0:
                    token[f"change_{minutes}m"] = -100.0
                liq = float((pairs[0].get("liquidity") or {}).get("usd") or 0)
                buys = int((pairs[0].get("txns") or {}).get("m5", {}).get("buys") or 0)
                sells = int((pairs[0].get("txns") or {}).get("m5", {}).get("sells") or 0)
                token[f"liq_{minutes}m"] = liq
                token[f"buys_{minutes}m"] = buys
                token[f"sells_{minutes}m"] = sells
                await asyncio.sleep(1.1)
            except Exception:
                pass


async def listen_pumpportal():
    ws = None
    while True:
        try:
            ws = await websockets.connect(PUMPPORTAL_WS_URL)
            log.info("Connected to PumpPortal WebSocket")

            await ws.send(json.dumps({"method": "subscribeNewToken"}))
            await ws.send(json.dumps({"method": "subscribeMigration"}))
            log.info("Subscribed to: newToken + migration events")

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if "txType" in msg and msg.get("txType") == "migration":
                    mint = msg.get("mint", "")
                    if mint and mint in tokens:
                        migrations.add(mint)
                        tokens[mint]["migrated"] = True
                        tokens[mint]["migrated_at"] = time.time()
                        tokens[mint]["migration_age_sec"] = int(time.time() - tokens[mint]["created_ts"])
                        stats["migrated"] += 1
                        sym = tokens[mint]["symbol"]
                        log.info("MIGRATED: %s (%s) after %ds", sym, mint[:8], tokens[mint]["migration_age_sec"])
                    continue

                if msg.get("txType") == "create":
                    mint = msg.get("mint", "")
                    if not mint or mint in tokens:
                        continue

                    now = time.time()
                    elapsed = int(now - stats["start"])
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
                        "elapsed_sec": elapsed,
                        "initial_buy_sol": init_buy,
                        "initial_price_sol": price_sol,
                        "initial_price_usd": price_usd,
                        "initial_mcap_usd": mcap_usd,
                        "market_cap_sol": mcap_sol,
                        "v_tokens_in_bonding": v_tokens,
                        "v_sol_in_bonding": v_sol,
                        "dev_address": msg.get("traderPublicKey", ""),
                        "migrated": False,
                        "migrated_at": None,
                        "migration_age_sec": None,
                        "enriched": False,
                        "enrich_tried": False,
                        "dex_price_usd": 0,
                        "dex_liquidity_usd": 0,
                        "dex_volume_5m": 0,
                        "dex_volume_1h": 0,
                        "dex_buys_5m": 0,
                        "dex_sells_5m": 0,
                        "dex_buys_1h": 0,
                        "dex_sells_1h": 0,
                        "dex_market_cap": 0,
                        "dex_fdv": 0,
                        "dex_change_5m": 0,
                        "dex_change_1h": 0,
                        "dex_dex_id": "",
                        "has_website": False,
                        "has_socials": False,
                        "change_at_enrich": None,
                        "price_5m": None,
                        "price_10m": None,
                        "change_5m": None,
                        "change_10m": None,
                        "liq_5m": None,
                        "liq_10m": None,
                        "buys_5m": None,
                        "sells_5m": None,
                        "buys_10m": None,
                        "sells_10m": None,
                        "outcome": "unknown",
                    }

                    trade_counts[mint] = {"buys": 0, "sells": 0, "buy_sol": 0, "sell_sol": 0}

                    stats["total"] += 1
                    log.info(
                        "[%ds] #%d NEW: %s | mcap=$%.0f buy=%.3f SOL",
                        elapsed, stats["total"], symbol,
                        mcap_usd, init_buy,
                    )

                    await ws.send(json.dumps({
                        "method": "subscribeTokenTrade",
                        "keys": [mint],
                    }))

                elif msg.get("txType") in ("buy", "sell"):
                    mint = msg.get("mint", "")
                    if mint in trade_counts:
                        tx = msg["txType"]
                        sol_amount = float(msg.get("solAmount") or 0)
                        tc = trade_counts[mint]
                        if tx == "buy":
                            tc["buys"] += 1
                            tc["buy_sol"] += sol_amount
                        else:
                            tc["sells"] += 1
                            tc["sell_sol"] += sol_amount

                        new_mcap = float(msg.get("marketCapSol") or 0)
                        if new_mcap > 0:
                            tokens[mint]["latest_mcap_sol"] = new_mcap
                            tokens[mint]["latest_mcap_usd"] = new_mcap * SOL_PRICE_USD

                        stats["trades"] += 1

        except websockets.exceptions.ConnectionClosed:
            log.warning("PumpPortal disconnected, reconnecting in 3s...")
            await asyncio.sleep(3)
        except Exception as e:
            log.error("PumpPortal error: %s, reconnecting in 5s...", e)
            await asyncio.sleep(5)
        finally:
            if ws:
                try:
                    await ws.close()
                except Exception:
                    pass


def finalize_tokens():
    for mint, token in tokens.items():
        tc = trade_counts.get(mint, {})
        token["total_buys"] = tc.get("buys", 0)
        token["total_sells"] = tc.get("sells", 0)
        token["total_buy_sol"] = round(tc.get("buy_sol", 0), 4)
        token["total_sell_sol"] = round(tc.get("sell_sol", 0), 4)
        token["buy_sell_ratio"] = round(tc.get("buys", 0) / max(1, tc.get("sells", 1)), 2)
        token["sell_pressure"] = round(tc.get("sells", 0) / max(1, tc.get("buys", 1) + tc.get("sells", 0)) * 100, 1)

        best_change = None
        for key in ["change_10m", "change_5m", "change_at_enrich"]:
            if token.get(key) is not None:
                best_change = token[key]
                break

        latest_mcap = token.get("latest_mcap_usd", 0)
        initial_mcap = token.get("initial_mcap_usd", 0)
        if best_change is None and latest_mcap > 0 and initial_mcap > 0:
            best_change = ((latest_mcap - initial_mcap) / initial_mcap) * 100

        token["best_change_pct"] = round(best_change, 1) if best_change is not None else None

        if best_change is None:
            token["outcome"] = "no_data"
        elif best_change >= 100:
            token["outcome"] = "ROCKET"
        elif best_change >= 25:
            token["outcome"] = "winner"
        elif best_change >= -15:
            token["outcome"] = "flat"
        elif best_change >= -50:
            token["outcome"] = "loser"
        else:
            token["outcome"] = "dead"


def save_data():
    finalize_tokens()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(DATA_DIR, f"tokens_{ts}.json")

    outcomes: dict[str, int] = {}
    for t in tokens.values():
        o = t["outcome"]
        outcomes[o] = outcomes.get(o, 0) + 1

    duration = time.time() - stats["start"]
    output = {
        "meta": {
            "start_time": datetime.fromtimestamp(stats["start"], tz=timezone.utc).isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "duration_sec": int(duration),
            "sol_price_usd": SOL_PRICE_USD,
            "total_tokens": stats["total"],
            "enriched": stats["enriched"],
            "migrated": stats["migrated"],
            "total_trades_tracked": stats["trades"],
            "errors": stats["errors"],
            "tokens_per_min": round(stats["total"] / max(1, duration / 60), 1),
            "outcomes": outcomes,
        },
        "tokens": list(tokens.values()),
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log.info("=" * 60)
    log.info("SAVED %d tokens to %s", len(tokens), filepath)
    log.info("Rate: %.1f tokens/min", output["meta"]["tokens_per_min"])
    log.info("Enriched: %d | Migrated: %d | Trades tracked: %d",
             stats["enriched"], stats["migrated"], stats["trades"])
    log.info("Outcomes: %s", outcomes)
    log.info("=" * 60)

    return filepath


async def stats_printer():
    while True:
        await asyncio.sleep(60)
        elapsed = int(time.time() - stats["start"])
        tpm = stats["total"] / max(1, elapsed / 60)
        log.info(
            "=== [%ds] tokens=%d enriched=%d migrated=%d trades=%d rate=%.1f/min ===",
            elapsed, stats["total"], stats["enriched"], stats["migrated"],
            stats["trades"], tpm,
        )


async def final_price_check(client: httpx.AsyncClient):
    now = time.time()
    pending = [m for m, t in tokens.items()
               if not t.get("price_5m") and now - t["created_ts"] >= 300]
    log.info("Final price check for %d tokens...", len(pending))
    for i, mint in enumerate(pending):
        try:
            r = await client.get(
                f"{DEXSCREENER_API}/tokens/v1/solana/{mint}",
                timeout=10,
            )
            if r.status_code != 200:
                continue
            pairs = r.json()
            token = tokens[mint]
            age = now - token["created_ts"]
            minutes = 10 if age >= 600 else 5

            if not pairs or not isinstance(pairs, list) or len(pairs) == 0:
                token[f"price_{minutes}m"] = 0
                token[f"change_{minutes}m"] = -100.0
                continue

            price = float(pairs[0].get("priceUsd") or 0)
            token[f"price_{minutes}m"] = price
            initial = token["initial_price_usd"]
            if initial > 0 and price > 0:
                change = ((price - initial) / initial) * 100
                token[f"change_{minutes}m"] = round(change, 1)
            elif price == 0:
                token[f"change_{minutes}m"] = -100.0

            liq = float((pairs[0].get("liquidity") or {}).get("usd") or 0)
            token[f"liq_{minutes}m"] = liq

            if i % 5 == 4:
                await asyncio.sleep(1.1)
        except Exception:
            pass


async def main():
    duration = COLLECT_DURATION
    log.info("=" * 60)
    log.info("HIGH-THROUGHPUT COLLECTOR")
    log.info("Duration: %d seconds", duration)
    log.info("Sources: PumpPortal WS (new tokens + trades + migrations)")
    log.info("Enrichment: DexScreener API (delayed 2min)")
    log.info("=" * 60)
    stats["start"] = time.time()

    async with httpx.AsyncClient() as client:
        await fetch_sol_price(client)

        listener = asyncio.create_task(listen_pumpportal())
        enricher = asyncio.create_task(enrich_batch(client))
        checker = asyncio.create_task(price_checker(client))
        printer = asyncio.create_task(stats_printer())
        sol_updater = asyncio.create_task(sol_price_updater(client))

        await asyncio.sleep(duration)

        log.info("Collection period ended. Final checks...")
        listener.cancel()
        printer.cancel()

        await final_price_check(client)

        enricher.cancel()
        checker.cancel()
        sol_updater.cancel()

    filepath = save_data()
    log.info("DONE! Data: %s", filepath)
    return filepath


if __name__ == "__main__":
    asyncio.run(main())
