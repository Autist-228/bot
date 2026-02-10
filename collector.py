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
    COOLDOWN_DURATION,
    DATA_DIR,
    PRICE_SNAPSHOT_INTERVALS,
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
stats = {"total": 0, "enriched": 0, "migrated": 0, "trades": 0, "start": 0}

SOL_PRICE_USD = 200.0
collecting_active = True


def bonding_curve_price_sol(v_sol: float, v_tokens: float) -> float:
    if v_tokens <= 0:
        return 0.0
    return v_sol / v_tokens


def bonding_curve_price_usd(v_sol: float, v_tokens: float) -> float:
    return bonding_curve_price_sol(v_sol, v_tokens) * SOL_PRICE_USD


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


def take_snapshot(token: dict, label: str):
    v_sol = token.get("v_sol_current", token.get("v_sol_in_bonding", 0))
    v_tokens = token.get("v_tokens_current", token.get("v_tokens_in_bonding", 0))
    price_sol = bonding_curve_price_sol(v_sol, v_tokens)
    price_usd = price_sol * SOL_PRICE_USD
    mcap_usd = token.get("latest_mcap_sol", token.get("market_cap_sol", 0)) * SOL_PRICE_USD

    tc = trade_counts.get(token["mint"], {})

    snap = {
        "age_sec": int(time.time() - token["created_ts"]),
        "price_sol": price_sol,
        "price_usd": price_usd,
        "v_sol": v_sol,
        "v_tokens": v_tokens,
        "mcap_usd": mcap_usd,
        "buys": tc.get("buys", 0),
        "sells": tc.get("sells", 0),
        "buy_sol": round(tc.get("buy_sol", 0), 4),
        "sell_sol": round(tc.get("sell_sol", 0), 4),
        "unique_buyers": len(tc.get("buyers", set())),
        "unique_sellers": len(tc.get("sellers", set())),
    }

    if "snapshots" not in token:
        token["snapshots"] = {}
    token["snapshots"][label] = snap

    initial_price = token.get("initial_price_usd", 0)
    if initial_price > 0 and price_usd > 0:
        change_pct = ((price_usd - initial_price) / initial_price) * 100
        token[f"change_{label}"] = round(change_pct, 1)
    elif price_usd == 0:
        token[f"change_{label}"] = -100.0

    if price_usd > token.get("peak_price_usd", 0):
        token["peak_price_usd"] = price_usd
        token["peak_price_age_sec"] = snap["age_sec"]


async def snapshot_scheduler():
    while True:
        await asyncio.sleep(1)
        now = time.time()
        for mint, token in list(tokens.items()):
            age = now - token["created_ts"]
            for interval in PRICE_SNAPSHOT_INTERVALS:
                label = f"{interval}s"
                if age >= interval and label not in token.get("snapshots", {}):
                    take_snapshot(token, label)


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
                token["dex_liquidity_usd"] = float((p.get("liquidity") or {}).get("usd") or 0)
                token["dex_volume_5m"] = float((p.get("volume") or {}).get("m5") or 0)
                token["dex_volume_1h"] = float((p.get("volume") or {}).get("h1") or 0)
                token["dex_buys_5m"] = int((p.get("txns") or {}).get("m5", {}).get("buys") or 0)
                token["dex_sells_5m"] = int((p.get("txns") or {}).get("m5", {}).get("sells") or 0)
                token["dex_buys_1h"] = int((p.get("txns") or {}).get("h1", {}).get("buys") or 0)
                token["dex_sells_1h"] = int((p.get("txns") or {}).get("h1", {}).get("sells") or 0)
                token["dex_market_cap"] = float(p.get("marketCap") or 0)
                token["dex_fdv"] = float(p.get("fdv") or 0)
                info = p.get("info", {}) or {}
                token["has_website"] = len(info.get("websites", [])) > 0
                token["has_socials"] = len(info.get("socials", [])) > 0

                token["enriched"] = True
                stats["enriched"] += 1
                await asyncio.sleep(1.1)
            except Exception as e:
                log.debug("Enrich error %s: %s", mint[:8], e)


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
                        tokens[mint]["migration_age_sec"] = int(
                            time.time() - tokens[mint]["created_ts"]
                        )
                        stats["migrated"] += 1
                        sym = tokens[mint]["symbol"]
                        log.info(
                            "MIGRATED: %s (%s) after %ds",
                            sym, mint[:8], tokens[mint]["migration_age_sec"],
                        )
                    continue

                if msg.get("txType") == "create":
                    if not collecting_active:
                        continue

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
                    init_buy = (
                        float(msg.get("initialBuy") or 0) / 1e9
                        if msg.get("initialBuy")
                        else 0
                    )

                    price_sol = bonding_curve_price_sol(v_sol, v_tokens)
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
                        "v_sol_current": v_sol,
                        "v_tokens_current": v_tokens,
                        "latest_mcap_sol": mcap_sol,
                        "latest_mcap_usd": mcap_usd,
                        "peak_price_usd": price_usd,
                        "peak_price_age_sec": 0,
                        "dev_address": msg.get("traderPublicKey", ""),
                        "migrated": False,
                        "migrated_at": None,
                        "migration_age_sec": None,
                        "enriched": False,
                        "enrich_tried": False,
                        "dex_liquidity_usd": 0,
                        "dex_volume_5m": 0,
                        "dex_volume_1h": 0,
                        "dex_buys_5m": 0,
                        "dex_sells_5m": 0,
                        "dex_buys_1h": 0,
                        "dex_sells_1h": 0,
                        "dex_market_cap": 0,
                        "dex_fdv": 0,
                        "has_website": False,
                        "has_socials": False,
                        "snapshots": {},
                        "outcome": "unknown",
                    }

                    trade_counts[mint] = {
                        "buys": 0,
                        "sells": 0,
                        "buy_sol": 0,
                        "sell_sol": 0,
                        "buyers": set(),
                        "sellers": set(),
                    }

                    stats["total"] += 1
                    log.info(
                        "[%ds] #%d NEW: %s | mcap=$%.0f buy=%.3f SOL price=$%.10f",
                        elapsed, stats["total"], symbol, mcap_usd, init_buy, price_usd,
                    )

                    await ws.send(json.dumps({
                        "method": "subscribeTokenTrade",
                        "keys": [mint],
                    }))

                elif msg.get("txType") in ("buy", "sell"):
                    mint = msg.get("mint", "")
                    if mint not in trade_counts:
                        continue

                    tx = msg["txType"]
                    sol_amount = float(msg.get("solAmount") or 0)
                    trader_key = msg.get("traderPublicKey", "")
                    tc = trade_counts[mint]

                    if tx == "buy":
                        tc["buys"] += 1
                        tc["buy_sol"] += sol_amount
                        tc["buyers"].add(trader_key)
                    else:
                        tc["sells"] += 1
                        tc["sell_sol"] += sol_amount
                        tc["sellers"].add(trader_key)

                    new_v_sol = float(msg.get("vSolInBondingCurve") or 0)
                    new_v_tokens = float(msg.get("vTokensInBondingCurve") or 0)
                    new_mcap_sol = float(msg.get("marketCapSol") or 0)

                    if new_v_sol > 0 and new_v_tokens > 0:
                        tokens[mint]["v_sol_current"] = new_v_sol
                        tokens[mint]["v_tokens_current"] = new_v_tokens

                        price_usd = bonding_curve_price_usd(new_v_sol, new_v_tokens)
                        if price_usd > tokens[mint].get("peak_price_usd", 0):
                            tokens[mint]["peak_price_usd"] = price_usd
                            tokens[mint]["peak_price_age_sec"] = int(
                                time.time() - tokens[mint]["created_ts"]
                            )

                    if new_mcap_sol > 0:
                        tokens[mint]["latest_mcap_sol"] = new_mcap_sol
                        tokens[mint]["latest_mcap_usd"] = new_mcap_sol * SOL_PRICE_USD

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
        token["unique_buyers"] = len(tc.get("buyers", set()))
        token["unique_sellers"] = len(tc.get("sellers", set()))

        buys = tc.get("buys", 0)
        sells = tc.get("sells", 0)
        token["buy_sell_ratio"] = round(buys / max(1, sells), 2)
        token["sell_pressure"] = round(sells / max(1, buys + sells) * 100, 1)

        initial_price = token.get("initial_price_usd", 0)
        v_sol = token.get("v_sol_current", token.get("v_sol_in_bonding", 0))
        v_tokens = token.get("v_tokens_current", token.get("v_tokens_in_bonding", 0))
        final_price = bonding_curve_price_usd(v_sol, v_tokens)
        token["final_price_usd"] = final_price

        peak_price = token.get("peak_price_usd", initial_price)

        if initial_price > 0 and peak_price > 0:
            peak_change = ((peak_price - initial_price) / initial_price) * 100
        else:
            peak_change = 0.0
        token["peak_change_pct"] = round(peak_change, 1)

        if initial_price > 0 and final_price > 0:
            final_change = ((final_price - initial_price) / initial_price) * 100
        else:
            final_change = -100.0
        token["final_change_pct"] = round(final_change, 1)

        best_change = peak_change

        snapshots = token.get("snapshots", {})
        for label, snap in snapshots.items():
            snap_price = snap.get("price_usd", 0)
            if initial_price > 0 and snap_price > 0:
                change = ((snap_price - initial_price) / initial_price) * 100
                if change > best_change:
                    best_change = change

        token["best_change_pct"] = round(best_change, 1)

        if best_change >= 500:
            token["outcome"] = "ROCKET"
        elif best_change >= 100:
            token["outcome"] = "winner"
        elif best_change >= 50:
            token["outcome"] = "good"
        elif best_change >= -15:
            token["outcome"] = "flat"
        elif best_change >= -50:
            token["outcome"] = "loser"
        else:
            token["outcome"] = "dead"


def save_data(tag: str = "FINAL") -> str:
    finalize_tokens()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(DATA_DIR, f"collect_{tag}_{ts}.json")

    outcomes: dict[str, int] = {}
    for t in tokens.values():
        o = t["outcome"]
        outcomes[o] = outcomes.get(o, 0) + 1

    duration = time.time() - stats["start"]

    serializable_tokens = []
    for t in tokens.values():
        token_copy = dict(t)
        tc = trade_counts.get(t["mint"], {})
        if "buyers" in tc:
            token_copy["unique_buyer_addresses"] = list(tc["buyers"])[:50]
        if "sellers" in tc:
            token_copy["unique_seller_addresses"] = list(tc["sellers"])[:50]
        serializable_tokens.append(token_copy)

    output = {
        "meta": {
            "start_time": datetime.fromtimestamp(
                stats["start"], tz=timezone.utc
            ).isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "duration_sec": int(duration),
            "collect_duration": COLLECT_DURATION,
            "cooldown_duration": COOLDOWN_DURATION,
            "sol_price_usd": SOL_PRICE_USD,
            "total_tokens": stats["total"],
            "enriched": stats["enriched"],
            "migrated": stats["migrated"],
            "total_trades_tracked": stats["trades"],
            "tokens_per_min": round(stats["total"] / max(1, duration / 60), 1),
            "outcomes": outcomes,
            "price_source": "pumpfun_bonding_curve",
        },
        "tokens": serializable_tokens,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log.info("=" * 60)
    log.info("SAVED %d tokens to %s", len(tokens), filepath)
    log.info("Rate: %.1f tokens/min", output["meta"]["tokens_per_min"])
    log.info(
        "Enriched: %d | Migrated: %d | Trades tracked: %d",
        stats["enriched"], stats["migrated"], stats["trades"],
    )
    log.info("Outcomes: %s", outcomes)
    log.info("=" * 60)

    return filepath


async def stats_printer():
    while True:
        await asyncio.sleep(60)
        elapsed = int(time.time() - stats["start"])
        tpm = stats["total"] / max(1, elapsed / 60)
        active_str = "COLLECTING" if collecting_active else "COOLDOWN"
        log.info(
            "=== [%ds] [%s] tokens=%d enriched=%d migrated=%d trades=%d rate=%.1f/min ===",
            elapsed, active_str, stats["total"], stats["enriched"], stats["migrated"],
            stats["trades"], tpm,
        )


async def main():
    global collecting_active

    total_duration = COLLECT_DURATION + COOLDOWN_DURATION
    log.info("=" * 60)
    log.info("SOLANA SNIPER DATA COLLECTOR V2")
    log.info(
        "Collection: %d sec | Cooldown: %d sec | Total: %d sec",
        COLLECT_DURATION, COOLDOWN_DURATION, total_duration,
    )
    log.info("Price source: PumpFun bonding curve ONLY")
    log.info("Enrichment: DexScreener (liquidity/volume only)")
    log.info("Snapshot intervals: %s", PRICE_SNAPSHOT_INTERVALS)
    log.info("=" * 60)
    stats["start"] = time.time()

    async with httpx.AsyncClient() as client:
        await fetch_sol_price(client)

        listener = asyncio.create_task(listen_pumpportal())
        enricher = asyncio.create_task(enrich_batch(client))
        snapshotter = asyncio.create_task(snapshot_scheduler())
        printer = asyncio.create_task(stats_printer())
        sol_updater = asyncio.create_task(sol_price_updater(client))

        log.info("Phase 1: COLLECTING new tokens for %d seconds...", COLLECT_DURATION)
        await asyncio.sleep(COLLECT_DURATION)

        collecting_active = False
        log.info(
            "Phase 2: COOLDOWN - no new tokens, tracking existing for %d seconds...",
            COOLDOWN_DURATION,
        )
        log.info("Tokens collected: %d, now waiting for late data...", stats["total"])
        await asyncio.sleep(COOLDOWN_DURATION)

        log.info("Session complete. Finalizing...")
        listener.cancel()
        printer.cancel()
        snapshotter.cancel()
        enricher.cancel()
        sol_updater.cancel()

    filepath = save_data()
    log.info("DONE! Data saved to: %s", filepath)
    return filepath


if __name__ == "__main__":
    asyncio.run(main())
