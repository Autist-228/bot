#!/usr/bin/env python3
"""
historical_collector.py - Fetch historical PumpFun token data from Solana blockchain.

Uses Helius RPC + Enhanced Transactions API to:
1. Scan PumpFun program for recent token trades
2. Group by mint address, reconstruct price timeline
3. Calculate snapshots and outcomes in same format as live collector

Usage:
  python3 historical_collector.py --hours 12
  python3 historical_collector.py --hours 24 --max-tokens 10000
  python3 historical_collector.py --hours 6 --estimate
"""

import argparse
import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

import httpx

from config import DATA_DIR, PRICE_SNAPSHOT_INTERVALS, DEXSCREENER_API

HELIUS_API_KEY = os.getenv(
    "HELIUS_API_KEY", "7c8922d6-1031-42c1-b4ee-bf5daa29abd4"
)
HELIUS_RPC = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"
HELIUS_ENHANCED = (
    f"https://api-mainnet.helius-rpc.com/v0/transactions/?api-key={HELIUS_API_KEY}"
)
PUMPFUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

INITIAL_V_SOL = 30.0
INITIAL_V_TOKENS = 1_073_000_000.0

SOL_PRICE_USD = 82.0
MAX_RPS = 25
BATCH_PARSE_SIZE = 100
ENRICHMENT_BATCH = 30

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("historical")


class RateLimiter:
    def __init__(self, max_rps):
        self.min_interval = 1.0 / max_rps
        self.last_call = 0.0

    async def acquire(self):
        now = time.monotonic()
        wait = self.min_interval - (now - self.last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        self.last_call = time.monotonic()


async def rpc_call(client, method, params, limiter):
    await limiter.acquire()
    for attempt in range(3):
        try:
            resp = await client.post(
                HELIUS_RPC,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                timeout=30,
            )
            data = resp.json()
            if "error" in data:
                err = data["error"]
                if "429" in str(err) or "rate" in str(err).lower():
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise Exception(f"RPC error: {err}")
            return data.get("result")
        except httpx.TimeoutException:
            if attempt < 2:
                await asyncio.sleep(1)
                continue
            raise
    return None


async def fetch_sol_price(client):
    global SOL_PRICE_USD
    try:
        r = await client.get(
            f"{DEXSCREENER_API}/tokens/v1/solana/So11111111111111111111111111111111111111112",
            timeout=10,
        )
        pairs = r.json()
        if pairs and len(pairs) > 0:
            SOL_PRICE_USD = float(pairs[0].get("priceUsd", 82))
    except Exception:
        pass
    log.info("SOL price: $%.2f", SOL_PRICE_USD)


async def get_signatures(client, limiter, before=None, limit=1000):
    params = [PUMPFUN_PROGRAM, {"limit": limit, "commitment": "confirmed"}]
    if before:
        params[1]["before"] = before
    return await rpc_call(client, "getSignaturesForAddress", params, limiter)


async def parse_enhanced(client, limiter, signatures):
    await limiter.acquire()
    for attempt in range(3):
        try:
            resp = await client.post(
                HELIUS_ENHANCED,
                json={"transactions": signatures},
                timeout=30,
            )
            if resp.status_code == 429:
                await asyncio.sleep(2 ** attempt)
                continue
            if resp.status_code != 200:
                log.warning("Enhanced API %d: %s", resp.status_code, resp.text[:200])
                return []
            return resp.json()
        except httpx.TimeoutException:
            if attempt < 2:
                await asyncio.sleep(1)
                continue
            return []
    return []


def extract_events(parsed_txs, tokens_data):
    for tx in parsed_txs:
        source = tx.get("source", "")
        if source != "PUMP_FUN":
            continue

        timestamp = tx.get("timestamp", 0)
        if not timestamp:
            continue

        fee_payer = tx.get("feePayer", "")
        token_transfers = tx.get("tokenTransfers", [])
        native_transfers = tx.get("nativeTransfers", [])

        mint = None
        token_amount = 0.0
        token_from = ""
        token_to = ""

        for tt in token_transfers:
            m = tt.get("mint", "")
            if m and m != "So11111111111111111111111111111111111111112":
                mint = m
                token_amount = float(tt.get("tokenAmount", 0) or 0)
                token_from = tt.get("fromUserAccount", "")
                token_to = tt.get("toUserAccount", "")
                break

        if not mint:
            continue

        sol_in = 0.0
        sol_out = 0.0
        for nt in native_transfers:
            amount_sol = float(nt.get("amount", 0)) / 1e9
            if amount_sol < 0.0001:
                continue
            from_acc = nt.get("fromUserAccount", "")
            to_acc = nt.get("toUserAccount", "")
            if from_acc == fee_payer:
                sol_in += amount_sol
            elif to_acc == fee_payer:
                sol_out += amount_sol

        is_buy = token_to == fee_payer or (token_amount > 0 and sol_in > sol_out)
        sol_amount = sol_in if is_buy else sol_out

        if mint not in tokens_data:
            tokens_data[mint] = {
                "mint": mint,
                "first_ts": timestamp,
                "trades": [],
                "buy_count": 0,
                "sell_count": 0,
                "buy_sol": 0.0,
                "sell_sol": 0.0,
                "buyers": set(),
                "sellers": set(),
            }

        td = tokens_data[mint]
        if timestamp < td["first_ts"]:
            td["first_ts"] = timestamp

        price_sol = 0.0
        if token_amount > 0 and sol_amount > 0:
            price_sol = sol_amount / token_amount

        td["trades"].append({
            "ts": timestamp,
            "type": "buy" if is_buy else "sell",
            "sol": sol_amount,
            "tokens": token_amount,
            "price_sol": price_sol,
            "trader": fee_payer,
        })

        if is_buy:
            td["buy_count"] += 1
            td["buy_sol"] += sol_amount
            td["buyers"].add(fee_payer)
        else:
            td["sell_count"] += 1
            td["sell_sol"] += sol_amount
            td["sellers"].add(fee_payer)


def reconstruct_token(mint, td):
    trades = sorted(td["trades"], key=lambda t: t["ts"])
    if not trades:
        return None

    priced_trades = [t for t in trades if t["price_sol"] > 0]
    if not priced_trades:
        return None

    created_ts = td["first_ts"]
    first_buy = next((t for t in priced_trades if t["type"] == "buy"), priced_trades[0])

    initial_price_sol = first_buy["price_sol"]
    initial_price_usd = initial_price_sol * SOL_PRICE_USD
    initial_mcap_usd = initial_price_usd * 1_000_000_000

    initial_buy_sol = trades[0]["sol"] if trades[0]["type"] == "buy" else 0

    peak_price_sol = initial_price_sol
    peak_age = 0
    snapshots = {}

    for t in priced_trades:
        if t["price_sol"] > peak_price_sol:
            peak_price_sol = t["price_sol"]
            peak_age = t["ts"] - created_ts

    for interval in PRICE_SNAPSHOT_INTERVALS:
        target_ts = created_ts + interval
        nearest = None
        for t in priced_trades:
            if t["ts"] <= target_ts:
                nearest = t
            elif t["ts"] > target_ts:
                break
        snap_price = nearest["price_sol"] if nearest else initial_price_sol
        snapshots[f"{interval}s"] = {
            "price_sol": snap_price,
            "price_usd": snap_price * SOL_PRICE_USD,
            "age_sec": interval,
        }

    last_priced = priced_trades[-1]
    final_price_sol = last_priced["price_sol"]
    final_price_usd = final_price_sol * SOL_PRICE_USD
    peak_price_usd = peak_price_sol * SOL_PRICE_USD

    if initial_price_sol > 0:
        peak_change = ((peak_price_sol - initial_price_sol) / initial_price_sol) * 100
        final_change = ((final_price_sol - initial_price_sol) / initial_price_sol) * 100
    else:
        peak_change = 0.0
        final_change = -100.0

    best_change = peak_change
    for snap in snapshots.values():
        if initial_price_usd > 0 and snap["price_usd"] > 0:
            ch = ((snap["price_usd"] - initial_price_usd) / initial_price_usd) * 100
            if ch > best_change:
                best_change = ch

    if best_change >= 500:
        outcome = "ROCKET"
    elif best_change >= 100:
        outcome = "winner"
    elif best_change >= 50:
        outcome = "good"
    elif best_change >= -15:
        outcome = "flat"
    elif best_change >= -50:
        outcome = "loser"
    else:
        outcome = "dead"

    buys = td["buy_count"]
    sells = td["sell_count"]

    return {
        "mint": mint,
        "symbol": "",
        "name": "",
        "created_ts": created_ts,
        "created_at": datetime.fromtimestamp(created_ts, tz=timezone.utc).isoformat(),
        "initial_buy_sol": initial_buy_sol,
        "initial_price_sol": initial_price_sol,
        "initial_price_usd": initial_price_usd,
        "initial_mcap_usd": initial_mcap_usd,
        "v_tokens_in_bonding": INITIAL_V_TOKENS,
        "v_sol_in_bonding": INITIAL_V_SOL,
        "peak_price_usd": peak_price_usd,
        "peak_price_age_sec": int(peak_age),
        "final_price_usd": final_price_usd,
        "peak_change_pct": round(peak_change, 1),
        "final_change_pct": round(final_change, 1),
        "best_change_pct": round(best_change, 1),
        "dev_address": trades[0]["trader"] if trades else "",
        "migrated": False,
        "enriched": False,
        "total_buys": buys,
        "total_sells": sells,
        "total_buy_sol": round(td["buy_sol"], 4),
        "total_sell_sol": round(td["sell_sol"], 4),
        "unique_buyers": len(td["buyers"]),
        "unique_sellers": len(td["sellers"]),
        "buy_sell_ratio": round(buys / max(1, sells), 2),
        "sell_pressure": round(sells / max(1, buys + sells) * 100, 1),
        "snapshots": snapshots,
        "outcome": outcome,
        "trade_count": len(trades),
        "data_source": "historical_helius",
    }


async def enrich_tokens_dexscreener(client, final_tokens):
    log.info("Enriching %d tokens with DexScreener metadata...", len(final_tokens))
    enriched = 0
    for i in range(0, len(final_tokens), ENRICHMENT_BATCH):
        batch = final_tokens[i : i + ENRICHMENT_BATCH]
        for token in batch:
            try:
                r = await client.get(
                    f"{DEXSCREENER_API}/tokens/v1/solana/{token['mint']}",
                    timeout=8,
                )
                if r.status_code == 200:
                    pairs = r.json()
                    if pairs and len(pairs) > 0:
                        p = pairs[0]
                        info = p.get("info", {})
                        token["symbol"] = p.get("baseToken", {}).get("symbol", "")
                        token["name"] = p.get("baseToken", {}).get("name", "")
                        token["dex_liquidity_usd"] = float(
                            p.get("liquidity", {}).get("usd", 0) or 0
                        )
                        token["dex_volume_1h"] = float(
                            p.get("volume", {}).get("h1", 0) or 0
                        )
                        token["dex_market_cap"] = float(p.get("marketCap", 0) or 0)
                        token["has_website"] = bool(
                            info.get("websites") or info.get("website")
                        )
                        token["has_socials"] = bool(info.get("socials"))
                        token["enriched"] = True
                        enriched += 1
                await asyncio.sleep(0.3)
            except Exception:
                pass
        if (i // ENRICHMENT_BATCH) % 10 == 0 and i > 0:
            log.info("  Enriched %d/%d tokens", enriched, i + len(batch))
    log.info("Enrichment done: %d/%d tokens", enriched, len(final_tokens))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--enrich", action="store_true", help="Enrich with DexScreener")
    parser.add_argument("--save-interval", type=int, default=0, help="Save progress every N batches")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("HISTORICAL PUMPFUN DATA COLLECTOR")
    log.info("Looking back: %d hours", args.hours)
    log.info("Max tokens: %d", args.max_tokens)
    log.info("Enrich: %s", args.enrich)
    log.info("=" * 60)

    limiter = RateLimiter(MAX_RPS)
    tokens_data = {}
    cutoff_ts = time.time() - (args.hours * 3600)
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        await fetch_sol_price(client)

        log.info("Phase 1: Scanning PumpFun program signatures...")
        cursor = None
        total_sigs = 0
        batch_num = 0
        reached_cutoff = False
        phase1_start = time.time()

        while not reached_cutoff:
            batch_num += 1

            try:
                sigs = await get_signatures(client, limiter, before=cursor)
            except Exception as e:
                log.error("Sig fetch error: %s", e)
                await asyncio.sleep(2)
                continue

            if not sigs:
                break

            total_sigs += len(sigs)
            cursor = sigs[-1]["signature"]
            oldest_ts = sigs[-1].get("blockTime", 0)

            valid_sigs = [s["signature"] for s in sigs if not s.get("err")]

            for i in range(0, len(valid_sigs), BATCH_PARSE_SIZE):
                batch = valid_sigs[i : i + BATCH_PARSE_SIZE]
                try:
                    parsed = await parse_enhanced(client, limiter, batch)
                    if parsed:
                        extract_events(parsed, tokens_data)
                except Exception as e:
                    log.warning("Parse error: %s", e)

            if oldest_ts <= cutoff_ts:
                reached_cutoff = True

            if batch_num % 10 == 0:
                elapsed = time.time() - phase1_start
                sps = total_sigs / max(1, elapsed)
                now_ts = time.time()
                scanned_range = now_ts - oldest_ts if oldest_ts > 0 else 0
                total_range = now_ts - cutoff_ts
                pct = (scanned_range / total_range * 100) if total_range > 0 else 0
                eta = (elapsed / max(0.01, pct) * (100 - pct)) if pct > 0 else 0

                oldest_dt = datetime.fromtimestamp(
                    oldest_ts, tz=timezone.utc
                ).strftime("%H:%M")
                log.info(
                    "[%.0fs] batch=%d sigs=%d tokens=%d oldest=%s "
                    "progress=%.1f%% rate=%.0f/s ETA=%.0fmin",
                    elapsed, batch_num, total_sigs, len(tokens_data),
                    oldest_dt, pct, sps, eta / 60,
                )

            if args.save_interval and batch_num % args.save_interval == 0:
                _save_progress(tokens_data, args.hours, total_sigs, "progress")

            if len(tokens_data) >= args.max_tokens:
                log.info("Reached max tokens: %d", args.max_tokens)
                break

        phase1_time = time.time() - phase1_start
        log.info(
            "Phase 1 done: %d tokens from %d sigs in %.0fs (%.1f min)",
            len(tokens_data), total_sigs, phase1_time, phase1_time / 60,
        )

        if args.estimate:
            sps = total_sigs / max(1, phase1_time)
            log.info("Rate: %.0f sigs/sec", sps)
            log.info("Tokens found so far: %d", len(tokens_data))
            if not reached_cutoff and oldest_ts > cutoff_ts:
                remaining = oldest_ts - cutoff_ts
                total_est = (time.time() - oldest_ts) / max(1, phase1_time) * remaining
                log.info("Estimated total time for %dh: %.0f min", args.hours, (phase1_time + total_est) / 60)
            return

        log.info("Phase 2: Reconstructing %d tokens...", len(tokens_data))
        final_tokens = []
        outcomes = defaultdict(int)

        for mint, td in tokens_data.items():
            token = reconstruct_token(mint, td)
            if token:
                final_tokens.append(token)
                outcomes[token["outcome"]] += 1

        log.info("Reconstructed %d tokens", len(final_tokens))
        log.info("Outcomes: %s", dict(outcomes))

        if args.enrich and final_tokens:
            await enrich_tokens_dexscreener(client, final_tokens)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(DATA_DIR, f"historical_{args.hours}h_{ts}.json")

        total_time = time.time() - start_time
        output = {
            "meta": {
                "source": "historical_helius",
                "hours_back": args.hours,
                "start_time": datetime.fromtimestamp(
                    start_time, tz=timezone.utc
                ).isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "duration_sec": int(total_time),
                "sol_price_usd": SOL_PRICE_USD,
                "total_tokens": len(final_tokens),
                "total_signatures_scanned": total_sigs,
                "outcomes": dict(outcomes),
                "price_source": "pumpfun_trade_marginal_price",
                "snapshot_intervals": PRICE_SNAPSHOT_INTERVALS,
            },
            "tokens": final_tokens,
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)

        log.info("=" * 60)
        log.info("SAVED %d tokens to %s", len(final_tokens), filepath)
        log.info("Total time: %.0fs (%.1f min)", total_time, total_time / 60)
        log.info("Signatures scanned: %d", total_sigs)
        log.info("Outcomes: %s", dict(outcomes))
        log.info("=" * 60)

        return filepath


def _save_progress(tokens_data, hours, total_sigs, tag):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(DATA_DIR, f"historical_{tag}_{ts}.json")
    count = 0
    tokens_list = []
    for mint, td in tokens_data.items():
        token = reconstruct_token(mint, td)
        if token:
            tokens_list.append(token)
            count += 1
    output = {
        "meta": {"source": "historical_progress", "tokens": count, "sigs": total_sigs},
        "tokens": tokens_list,
    }
    with open(filepath, "w") as f:
        json.dump(output, f, default=str)
    log.info("Progress saved: %d tokens to %s", count, filepath)


if __name__ == "__main__":
    asyncio.run(main())
