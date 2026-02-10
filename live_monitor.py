import asyncio
import json
import logging
import sys
import time
import os
from datetime import datetime, timezone

import httpx
import websockets
import numpy as np
import joblib

from config import (
    PUMPPORTAL_WS_URL,
    DEXSCREENER_API,
    COLLECT_DURATION,
    DATA_DIR,
)

REAL_TRADING = "--real" in sys.argv
trader = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("live_monitor")

FEATURES = [
    "initial_buy_sol", "initial_price_usd", "initial_mcap_usd",
    "initial_v_sol", "initial_v_tokens",
    "v_sol_in_bonding", "total_buys", "total_sells", "total_buy_sol", "total_sell_sol",
    "buy_sell_ratio", "sell_pressure", "unique_buyers", "unique_sellers",
    "dex_liquidity_usd", "dex_volume_5m",
    "dex_volume_1h", "dex_buys_5m", "dex_sells_5m", "dex_buys_1h", "dex_sells_1h",
    "dex_market_cap", "dex_fdv", "dex_change_5m", "dex_change_1h",
    "has_website", "has_socials", "migrated",
]

MIN_BUYS_FOR_SIGNAL = 5
MAX_BUYS_FOR_SIGNAL = 999
MIN_RATIO_FOR_SIGNAL = 0.0
MIN_CONFIDENCE_PCT = 70.0
ROCKET_ONLY = True
STOP_LOSS_PCT = -10.0
TIME_STOP_SEC = 60
TIME_STOP_MIN_GAIN = 10.0
TRAILING_STOP_PCT = 15.0
PRICE_POLL_INTERVAL = 0.5
MAX_SELLS_PER_TOKEN = 1
MIN_LIQUIDITY_USD = 2000.0
MAX_SLOTS = int(os.getenv("MAX_SLOTS", "10"))
BET_SIZE_USD = float(os.getenv("BET_SIZE_USD", "5.0"))
BUY_SLIPPAGE_SIM = 0.02
SELL_SLIPPAGE_SIM = 0.03
PUMPFUN_FEE_PCT = 0.01

tokens: dict[str, dict] = {}
trade_counts: dict[str, dict] = {}
signals: list[dict] = []
stats = {"total": 0, "enriched": 0, "trades": 0, "signals": 0, "start": 0}

SOL_PRICE_USD = 200.0
model = None
scaler = None
seen_mints: set[str] = set()


def init_trader():
    global trader
    if not REAL_TRADING:
        return
    from real_trader import RealTrader
    trader = RealTrader()
    bal = trader.refresh_balance()
    trader.initial_balance = bal
    log.info("REAL TRADING MODE | Wallet: %s | Balance: %.4f SOL ($%.2f)", trader.pubkey, bal, bal * SOL_PRICE_USD)


def load_model():
    global model, scaler
    model_path = os.path.join(DATA_DIR, "model.pkl")
    scaler_path = os.path.join(DATA_DIR, "scaler.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        log.info("ML model loaded from %s", model_path)
        return True
    log.error("No model found at %s", model_path)
    return False


def predict_token(token_data: dict) -> tuple[str, float]:
    if model is None or scaler is None:
        return "no_model", 0.0
    values = []
    for f in FEATURES:
        v = token_data.get(f, 0)
        if f in ("has_website", "has_socials", "migrated"):
            v = int(v) if v else 0
        values.append(float(v) if v else 0.0)
    X = np.array([values])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    label = {0: "trash", 1: "winner", 2: "ROCKET"}.get(pred, "unknown")
    confidence = max(proba) * 100
    return label, confidence


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
        log.warning("Failed to fetch SOL price: %s", e)


async def ml_scanner():
    while True:
        await asyncio.sleep(5)
        now = time.time()
        for mint, token in list(tokens.items()):
            age = now - token["created_ts"]
            if age < 30 or age > 120:
                continue
            if token.get("ml_checked"):
                continue

            tc = trade_counts.get(mint, {})
            buys = tc.get("buys", 0)
            sells = tc.get("sells", 0)
            if buys < MIN_BUYS_FOR_SIGNAL:
                continue
            if buys > MAX_BUYS_FOR_SIGNAL:
                token["ml_checked"] = True
                continue

            token["total_buys"] = buys
            token["total_sells"] = sells
            token["total_buy_sol"] = tc.get("buy_sol", 0)
            token["total_sell_sol"] = tc.get("sell_sol", 0)
            token["buy_sell_ratio"] = round(buys / max(1, sells), 2)
            token["sell_pressure"] = round(sells / max(1, buys + sells) * 100, 1)
            token["unique_buyers"] = len(tc.get("buyers", set()))
            token["unique_sellers"] = len(tc.get("sellers", set()))

            label, confidence = predict_token(token)
            token["ml_checked"] = True
            token["ml_label"] = label
            token["ml_confidence"] = confidence

            if label in ("ROCKET", "winner"):
                if ROCKET_ONLY and label != "ROCKET":
                    continue
                if confidence < MIN_CONFIDENCE_PCT:
                    log.info("SKIP %s %s: conf=%.0f%% < %.0f%%", label, token["symbol"], confidence, MIN_CONFIDENCE_PCT)
                    continue
                if mint in seen_mints:
                    continue
                seen_mints.add(mint)

                active_count = sum(1 for s in signals if s["status"] == "ACTIVE")
                if active_count >= MAX_SLOTS:
                    log.info("SKIP %s %s: %d/%d active slots full", label, token["symbol"], active_count, MAX_SLOTS)
                    continue

                if buys < MIN_BUYS_FOR_SIGNAL or buys > MAX_BUYS_FOR_SIGNAL:
                    log.info("SKIP %s %s: buys=%d (range %d-%d)", label, token["symbol"], buys, MIN_BUYS_FOR_SIGNAL, MAX_BUYS_FOR_SIGNAL)
                    continue
                ratio = token["buy_sell_ratio"]
                if ratio < MIN_RATIO_FOR_SIGNAL:
                    log.info("SKIP %s %s: ratio=%.1f < %.1f", label, token["symbol"], ratio, MIN_RATIO_FOR_SIGNAL)
                    continue

                liq = token.get("dex_liquidity_usd", 0)
                if liq < MIN_LIQUIDITY_USD and token.get("enriched"):
                    log.info("SKIP %s %s: liquidity=$%.0f < $%.0f", label, token["symbol"], liq, MIN_LIQUIDITY_USD)
                    continue

                signal_time = time.time()
                initial_mcap = token.get("initial_mcap_usd", 0)

                sol_amount = round(BET_SIZE_USD / SOL_PRICE_USD, 4)
                v_sol = token.get("v_sol_in_bonding", 0)
                v_tokens = token.get("v_tokens_in_bonding", 0)
                sim_tokens = 0.0
                sim_sol_spent = 0.0
                if v_sol > 0 and v_tokens > 0:
                    sol_after_fee = sol_amount * (1 - PUMPFUN_FEE_PCT)
                    sol_effective = sol_after_fee * (1 - BUY_SLIPPAGE_SIM)
                    k = v_sol * v_tokens
                    new_v_sol = v_sol + sol_effective
                    new_v_tokens = k / new_v_sol
                    sim_tokens = v_tokens - new_v_tokens
                    sim_sol_spent = sol_amount

                entry_cost_pct = 0.0
                if sim_tokens > 0 and sim_sol_spent > 0 and v_sol > 0 and v_tokens > 0:
                    k_entry = v_sol * v_tokens
                    vt_after = v_tokens + sim_tokens
                    vs_after = k_entry / vt_after
                    gross_out = v_sol - vs_after
                    net_out = gross_out * (1 - PUMPFUN_FEE_PCT) * (1 - SELL_SLIPPAGE_SIM)
                    entry_cost_pct = ((net_out / sim_sol_spent) - 1) * 100

                signal = {
                    "mint": mint,
                    "symbol": token["symbol"],
                    "name": token["name"],
                    "signal_time": signal_time,
                    "signal_at": datetime.now(timezone.utc).isoformat(),
                    "signal_age_sec": int(age),
                    "ml_label": label,
                    "ml_confidence": round(confidence, 1),
                    "initial_mcap_usd": initial_mcap,
                    "buys_at_signal": buys,
                    "sells_at_signal": sells,
                    "buy_ratio_at_signal": token["buy_sell_ratio"],
                    "sell_pressure_at_signal": token["sell_pressure"],
                    "entry_price_usd": token.get("initial_price_usd", 0),
                    "current_price_usd": None,
                    "pnl_pct": None,
                    "pnl_usd_per_dollar": None,
                    "peak_pnl_pct": 0.0,
                    "gain_from_entry": 0.0,
                    "peak_gain": 0.0,
                    "entry_cost_pct": round(entry_cost_pct, 1),
                    "position_remaining_pct": 100.0,
                    "tp_hits": [],
                    "realized_pnl": 0.0,
                    "status": "ACTIVE",
                    "close_reason": None,
                    "close_pnl_pct": None,
                    "close_time": None,
                    "checked_at": None,
                    "sells_done": 0,
                    "sim_tokens_bought": sim_tokens,
                    "sim_sol_spent": sim_sol_spent,
                    "sim_v_sol_at_buy": v_sol,
                    "sim_v_tokens_at_buy": v_tokens,
                }
                signals.append(signal)
                stats["signals"] += 1
                log.info(
                    "*** SIGNAL #%d: %s %s (%s) conf=%.0f%% buys=%d ratio=%.1f | sim: %.4f SOL -> %.0f tokens | entry_cost=%.1f%% ***",
                    stats["signals"], label, token["symbol"], mint[:8],
                    confidence, buys, token["buy_sell_ratio"], sim_sol_spent, sim_tokens, entry_cost_pct,
                )

                if REAL_TRADING and trader:
                    sol_amount = round(BET_SIZE_USD / SOL_PRICE_USD, 4)
                    asyncio.create_task(execute_real_buy(signal, sol_amount))


def check_tp_ladder(sig: dict, current_pnl: float):
    sells_done = sig.get("sells_done", 0)
    for tp in TP_LADDER:
        level = tp["level"]
        sell_pct = tp["sell_pct"]
        if current_pnl >= level and level not in sig["tp_hits"]:
            if sells_done >= MAX_SELLS_PER_TOKEN:
                log.info("TP %s +%.0f%%: SKIP (max %d sells reached)", sig["symbol"], level, MAX_SELLS_PER_TOKEN)
                continue
            remaining = sig["position_remaining_pct"]
            if sell_pct == 100:
                sold_portion = remaining
            else:
                sold_portion = remaining * (sell_pct / 100.0)
            realized = sold_portion * (current_pnl / 100.0) / 100.0
            sig["realized_pnl"] += realized
            sig["position_remaining_pct"] = remaining - sold_portion
            sig["tp_hits"].append(level)
            sig["sells_done"] = sells_done + 1
            sells_done += 1
            log.info(
                "TP HIT %s +%.0f%%: sold %.0f%% (%.1f%% remaining) | realized +$%.4f per $1",
                sig["symbol"], level, sold_portion, sig["position_remaining_pct"], realized,
            )

            if REAL_TRADING and trader:
                real_sell_pct = 100 if sell_pct == 100 else sell_pct
                asyncio.create_task(execute_real_sell(sig, real_sell_pct, f"TP+{level:.0f}%"))

            if sig["position_remaining_pct"] <= 0:
                sig["status"] = "CLOSED"
                sig["close_reason"] = f"TP+{level:.0f}% FULL EXIT"
                sig["close_pnl_pct"] = round(sig["realized_pnl"] * 100, 1)
                sig["close_time"] = datetime.now(timezone.utc).isoformat()
                sig["pnl_usd_per_dollar"] = round(1 + sig["realized_pnl"], 4)
                log.info("CLOSED %s %s: FULL EXIT at TP+%.0f%% | P&L: +%.1f%%", sig["ml_label"], sig["symbol"], level, sig["realized_pnl"] * 100)
                return


def check_exit_rules(sig: dict, current_pnl: float) -> str | None:
    if sig["status"] != "ACTIVE":
        return None

    entry_cost = sig.get("entry_cost_pct", 0)
    gain = current_pnl - entry_cost
    sig["gain_from_entry"] = round(gain, 1)
    if gain > sig.get("peak_gain", 0):
        sig["peak_gain"] = round(gain, 1)

    if gain <= STOP_LOSS_PCT:
        return f"STOP_LOSS (gain={gain:+.1f}% <= {STOP_LOSS_PCT}%, real_pnl={current_pnl:+.1f}%)"

    age_sec = time.time() - sig["signal_time"]
    if age_sec >= TIME_STOP_SEC and gain < TIME_STOP_MIN_GAIN:
        return f"TIME_STOP ({age_sec:.0f}s, gain={gain:+.1f}% < +{TIME_STOP_MIN_GAIN}%, real_pnl={current_pnl:+.1f}%)"

    peak_gain = sig.get("peak_gain", 0)
    if peak_gain >= 30.0:
        drop = peak_gain - gain
        if drop >= TRAILING_STOP_PCT:
            return f"TRAILING_STOP (peak_gain={peak_gain:+.1f}%, gain={gain:+.1f}%, drop={drop:.1f}%, real_pnl={current_pnl:+.1f}%)"

    if peak_gain >= 15.0 and gain < 0:
        return f"PROFIT_GONE (peak_gain={peak_gain:+.1f}%, gain={gain:+.1f}%, real_pnl={current_pnl:+.1f}%)"

    return None


def close_signal(sig: dict, reason: str, pnl: float):
    remaining = sig["position_remaining_pct"] / 100.0
    unrealized = remaining * (pnl / 100.0)
    total_pnl_per_dollar = sig["realized_pnl"] + unrealized
    sig["status"] = "CLOSED"
    sig["close_reason"] = reason
    sig["close_pnl_pct"] = round(total_pnl_per_dollar * 100, 1)
    sig["close_time"] = datetime.now(timezone.utc).isoformat()
    sig["pnl_usd_per_dollar"] = round(1 + total_pnl_per_dollar, 4)
    log.info(
        "CLOSED %s %s: %s | Total P&L: %+.1f%% (realized: +%.1f%%, remaining %.0f%% closed at %+.1f%%)",
        sig["ml_label"], sig["symbol"], reason,
        total_pnl_per_dollar * 100, sig["realized_pnl"] * 100, sig["position_remaining_pct"], pnl,
    )

    sells_done = sig.get("sells_done", 0)
    if REAL_TRADING and trader and sig.get("real_buy") and sig["position_remaining_pct"] > 0:
        if sells_done < MAX_SELLS_PER_TOKEN:
            sig["sells_done"] = sells_done + 1
            asyncio.create_task(execute_real_sell(sig, 100, reason))


async def execute_real_buy(sig: dict, sol_amount: float):
    try:
        result = await trader.buy_token(sig["mint"], sol_amount, sig["symbol"])
        sig["real_buy"] = result
        if result["success"]:
            sig["real_buy_tx"] = result["tx_hash"]
            sig["real_sol_spent"] = sol_amount
            sig["buy_confirmed"] = True
            log.info("REAL BUY OK %s: %.4f SOL | tx=%s", sig["symbol"], sol_amount, result["tx_hash"][:20])
        else:
            log.error("REAL BUY FAILED %s: %s", sig["symbol"], result["error"])
    except Exception as e:
        log.error("REAL BUY ERROR %s: %s", sig["symbol"], e)


async def execute_real_sell(sig: dict, sell_pct: int, reason: str):
    if REAL_TRADING and not sig.get("buy_confirmed"):
        log.warning("SELL SKIP %s: buy not confirmed yet, waiting...", sig["symbol"])
        for _ in range(60):
            await asyncio.sleep(0.5)
            if sig.get("buy_confirmed"):
                break
        if not sig.get("buy_confirmed"):
            log.error("SELL ABORT %s: buy never confirmed after 30s", sig["symbol"])
            return {"success": False, "error": "buy not confirmed"}
    try:
        result = await trader.sell_token(sig["mint"], sell_pct, sig["symbol"], reason)
        if result["success"]:
            log.info("REAL SELL OK %s %d%% (%s): tx=%s | balance=%.4f SOL", sig["symbol"], sell_pct, reason, result["tx_hash"][:20], trader.sol_balance)
        else:
            log.error("REAL SELL FAILED %s (%s): %s", sig["symbol"], reason, result.get("error", "not confirmed"))
        return result
    except Exception as e:
        log.error("REAL SELL ERROR %s: %s", sig["symbol"], e)
        return {"success": False, "error": str(e)}


def calc_bonding_curve_pnl(sig: dict, token_data: dict) -> float | None:
    sim_tokens = sig.get("sim_tokens_bought", 0)
    sim_sol_spent = sig.get("sim_sol_spent", 0)
    if sim_tokens <= 0 or sim_sol_spent <= 0:
        return None
    cur_v_sol = token_data.get("v_sol_in_bonding", 0)
    cur_v_tokens = token_data.get("v_tokens_in_bonding", 0)
    if cur_v_sol <= 0 or cur_v_tokens <= 0:
        return None
    k = cur_v_sol * cur_v_tokens
    new_v_tokens = cur_v_tokens + sim_tokens
    new_v_sol = k / new_v_tokens
    gross_sol_out = cur_v_sol - new_v_sol
    sol_after_fee = gross_sol_out * (1 - PUMPFUN_FEE_PCT)
    sol_after_slippage = sol_after_fee * (1 - SELL_SLIPPAGE_SIM)
    pnl_pct = ((sol_after_slippage / sim_sol_spent) - 1) * 100
    return pnl_pct


async def signal_price_updater(client: httpx.AsyncClient):
    while True:
        await asyncio.sleep(PRICE_POLL_INTERVAL)
        active = [s for s in signals if s["status"] == "ACTIVE"]
        if not active:
            continue

        for sig in active:
            try:
                mint = sig["mint"]
                token_data = tokens.get(mint, {})
                current_pnl = calc_bonding_curve_pnl(sig, token_data)

                if current_pnl is not None:
                    sig["pnl_pct"] = round(current_pnl, 1)
                    if current_pnl > sig.get("peak_pnl_pct", 0):
                        sig["peak_pnl_pct"] = round(current_pnl, 1)
                    sig["checked_at"] = datetime.now(timezone.utc).isoformat()

                    reason = check_exit_rules(sig, current_pnl)
                    if reason:
                        close_signal(sig, reason, current_pnl)
                    else:
                        remaining = sig["position_remaining_pct"] / 100.0
                        unrealized = remaining * (current_pnl / 100.0)
                        total = sig["realized_pnl"] + unrealized
                        sig["pnl_usd_per_dollar"] = round(1 + total, 4)

            except Exception:
                pass


async def listen_pumpportal():
    ws = None
    while True:
        try:
            ws = await websockets.connect(PUMPPORTAL_WS_URL)
            log.info("Connected to PumpPortal WebSocket")
            await ws.send(json.dumps({"method": "subscribeNewToken"}))
            log.info("Subscribed to newToken events")

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
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
                    init_buy = float(msg.get("initialBuy") or 0)

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
                        "initial_v_sol": v_sol,
                        "initial_v_tokens": v_tokens,
                        "market_cap_sol": mcap_sol,
                        "v_tokens_in_bonding": v_tokens,
                        "v_sol_in_bonding": v_sol,
                        "dev_address": msg.get("traderPublicKey", ""),
                        "migrated": False,
                        "enriched": False,
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
                        "ml_checked": False,
                        "ml_label": None,
                        "ml_confidence": None,
                    }

                    trade_counts[mint] = {"buys": 0, "sells": 0, "buy_sol": 0, "sell_sol": 0, "buyers": set(), "sellers": set()}
                    stats["total"] += 1

                    if stats["total"] % 20 == 0:
                        log.info("[%ds] tokens=%d signals=%d trades=%d", elapsed, stats["total"], stats["signals"], stats["trades"])

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
                        trader_key = msg.get("traderPublicKey", "")
                        if tx == "buy":
                            tc["buys"] += 1
                            tc["buy_sol"] += sol_amount
                            if trader_key:
                                tc["buyers"].add(trader_key)
                        else:
                            tc["sells"] += 1
                            tc["sell_sol"] += sol_amount
                            if trader_key:
                                tc["sellers"].add(trader_key)

                        new_mcap = float(msg.get("marketCapSol") or 0)
                        new_v_sol = float(msg.get("vSolInBondingCurve") or 0)
                        new_v_tokens = float(msg.get("vTokensInBondingCurve") or 0)
                        if mint in tokens:
                            if new_mcap > 0:
                                tokens[mint]["latest_mcap_sol"] = new_mcap
                                tokens[mint]["latest_mcap_usd"] = new_mcap * SOL_PRICE_USD
                            if new_v_sol > 0:
                                tokens[mint]["v_sol_in_bonding"] = new_v_sol
                            if new_v_tokens > 0:
                                tokens[mint]["v_tokens_in_bonding"] = new_v_tokens

                        for sig in signals:
                            if sig["mint"] == mint and sig["status"] == "ACTIVE":
                                real_pnl = calc_bonding_curve_pnl(sig, tokens.get(mint, {}))
                                if real_pnl is not None:
                                    sig["pnl_pct"] = round(real_pnl, 1)
                                    if real_pnl > sig.get("peak_pnl_pct", 0):
                                        sig["peak_pnl_pct"] = round(real_pnl, 1)
                                    sig["checked_at"] = datetime.now(timezone.utc).isoformat()
                                    reason = check_exit_rules(sig, real_pnl)
                                    if reason:
                                        close_signal(sig, reason, real_pnl)
                                    else:
                                        remaining = sig["position_remaining_pct"] / 100.0
                                        unrealized = remaining * (real_pnl / 100.0)
                                        total = sig["realized_pnl"] + unrealized
                                        sig["pnl_usd_per_dollar"] = round(1 + total, 4)

                        stats["trades"] += 1

        except websockets.exceptions.ConnectionClosed:
            log.warning("Disconnected, reconnecting in 3s...")
            await asyncio.sleep(3)
        except Exception as e:
            log.error("Error: %s, reconnecting in 5s...", e)
            await asyncio.sleep(5)
        finally:
            if ws:
                try:
                    await ws.close()
                except Exception:
                    pass


def print_signals_report():
    active = [s for s in signals if s["status"] == "ACTIVE"]
    closed = [s for s in signals if s["status"] == "CLOSED"]

    lines = []
    lines.append(f"=== SIGNAL REPORT | Active: {len(active)} | Closed: {len(closed)} | Total: {len(signals)} ===")

    balance = 0.0
    invested = 0

    if active:
        lines.append("--- ACTIVE POSITIONS ---")
        for sig in active:
            age = int(time.time() - sig["signal_time"])
            pnl = sig.get("pnl_pct")
            peak = sig.get("peak_pnl_pct", 0)
            remaining = sig.get("position_remaining_pct", 100)
            tp_str = f" TP:{','.join(f'+{int(t)}%' for t in sig.get('tp_hits', []))}" if sig.get("tp_hits") else ""

            gain = sig.get("gain_from_entry", 0)
            peak_gain = sig.get("peak_gain", 0)
            if pnl is not None:
                pnl_str = f"real={pnl:+.1f}% gain={gain:+.1f}%"
                val = sig.get("pnl_usd_per_dollar", 1.0)
            else:
                pnl_str = "waiting..."
                val = 1.0

            balance += val
            invested += 1

            lines.append(
                f"  ACTIVE {sig['ml_label']} {sig['symbol']} | "
                f"{pnl_str} | peak_gain: {peak_gain:+.1f}% | pos: {remaining:.0f}%{tp_str} | "
                f"{age}s ago | buys={sig['buys_at_signal']} ratio={sig['buy_ratio_at_signal']}"
            )

    if closed:
        lines.append("--- CLOSED POSITIONS ---")
        wins = 0
        losses = 0
        for sig in closed:
            cpnl = sig.get("close_pnl_pct", 0)
            val = sig.get("pnl_usd_per_dollar", 1 + cpnl / 100)
            balance += val
            invested += 1
            if cpnl > 0:
                wins += 1
            else:
                losses += 1
            tp_str = f" TP:{','.join(f'+{int(t)}%' for t in sig.get('tp_hits', []))}" if sig.get("tp_hits") else ""
            lines.append(
                f"  CLOSED {sig['symbol']} | real={cpnl:+.1f}% ${val:.2f}{tp_str} | {sig['close_reason']}"
            )
        if wins + losses > 0:
            lines.append(f"  Closed stats: {wins}W / {losses}L ({wins/(wins+losses)*100:.0f}% win rate)")

    if invested > 0:
        total_pnl = ((balance / invested) - 1) * 100
        lines.append(f"  BALANCE: ${invested} invested -> ${balance:.2f} ({total_pnl:+.1f}%)")

    if not signals:
        lines.append("  No signals yet...")

    return "\n".join(lines)


def save_session():
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(DATA_DIR, f"live_session_{ts}.json")

    for mint, token in tokens.items():
        tc = trade_counts.get(mint, {})
        token["total_buys"] = tc.get("buys", 0)
        token["total_sells"] = tc.get("sells", 0)
        token["total_buy_sol"] = round(tc.get("buy_sol", 0), 4)
        token["total_sell_sol"] = round(tc.get("sell_sol", 0), 4)
        token["buy_sell_ratio"] = round(tc.get("buys", 0) / max(1, tc.get("sells", 1)), 2)
        token["sell_pressure"] = round(tc.get("sells", 0) / max(1, tc.get("buys", 1) + tc.get("sells", 0)) * 100, 1)

    active_count = sum(1 for s in signals if s["status"] == "ACTIVE")
    closed_count = sum(1 for s in signals if s["status"] == "CLOSED")
    closed_wins = sum(1 for s in signals if s["status"] == "CLOSED" and (s.get("close_pnl_pct", 0) or 0) > 0)

    output = {
        "meta": {
            "start_time": datetime.fromtimestamp(stats["start"], tz=timezone.utc).isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "duration_sec": int(time.time() - stats["start"]),
            "sol_price_usd": SOL_PRICE_USD,
            "total_tokens": stats["total"],
            "total_signals": stats["signals"],
            "total_trades": stats["trades"],
            "active_signals": active_count,
            "closed_signals": closed_count,
            "closed_wins": closed_wins,
        },
        "config": {
            "min_buys": MIN_BUYS_FOR_SIGNAL,
            "min_ratio": MIN_RATIO_FOR_SIGNAL,
            "stop_loss_pct": STOP_LOSS_PCT,
            "time_stop_sec": TIME_STOP_SEC,
            "time_stop_min_gain": TIME_STOP_MIN_GAIN,
            "trailing_stop_pct": TRAILING_STOP_PCT,
            "tp_ladder": "DISABLED - trailing stop only",
        },
        "signals": signals,
        "tokens": list(tokens.values()),
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log.info("Session saved: %s", filepath)
    return filepath


async def report_printer():
    while True:
        await asyncio.sleep(120)
        report = print_signals_report()
        log.info("\n%s", report)


async def main():
    duration = COLLECT_DURATION
    log.info("=" * 60)
    mode_str = "REAL TRADING" if REAL_TRADING else "SIMULATION"
    log.info("LIVE ML MONITOR v6 [%s] — BONDING CURVE P&L", mode_str)
    log.info("Duration: %d seconds", duration)
    log.info("Bet size: $%.0f | Max slots: %d | 1 buy + 1 sell per token", BET_SIZE_USD, MAX_SLOTS)
    rocket_str = "ROCKET only" if ROCKET_ONLY else "ROCKET + winner"
    log.info("Filters: %s, conf >= %.0f%%, buys >= %d, liquidity >= $%.0f", rocket_str, MIN_CONFIDENCE_PCT, MIN_BUYS_FOR_SIGNAL, MIN_LIQUIDITY_USD)
    log.info("Stop-loss: %.0f%% | Time-stop: %ds (min +%.0f%%)", STOP_LOSS_PCT, TIME_STOP_SEC, TIME_STOP_MIN_GAIN)
    log.info("Trailing stop: -%.0f%% from peak (activates at +30%%) | Price poll: %.1fs", TRAILING_STOP_PCT, PRICE_POLL_INTERVAL)
    log.info("P&L: bonding curve simulation (buy slip %.0f%%, sell slip %.0f%%, fee %.0f%%)", BUY_SLIPPAGE_SIM*100, SELL_SLIPPAGE_SIM*100, PUMPFUN_FEE_PCT*100)
    log.info("Strategy: NO TP ladder, trailing stop only (1 sell per token)")
    log.info("=" * 60)

    if not load_model():
        log.error("Cannot start without ML model. Run train_model.py first.")
        return

    if REAL_TRADING:
        init_trader()
        if not trader:
            log.error("Failed to init trader. Check SOLANA_PRIVATE_KEY in .env")
            return

    stats["start"] = time.time()

    async with httpx.AsyncClient() as client:
        await fetch_sol_price(client)

        listener = asyncio.create_task(listen_pumpportal())
        scanner = asyncio.create_task(ml_scanner())
        price_updater = asyncio.create_task(signal_price_updater(client))
        reporter = asyncio.create_task(report_printer())

        await asyncio.sleep(duration)

        log.info("Signal collection ended. Tracking active positions for 2 more minutes...")
        listener.cancel()
        scanner.cancel()
        reporter.cancel()

        track_extra = 120
        await asyncio.sleep(track_extra)

        log.info("Tracking period ended. Closing all active positions...")
        for sig in signals:
            if sig["status"] == "ACTIVE":
                pnl = sig.get("pnl_pct", 0) or 0
                close_signal(sig, "SESSION_END", pnl)

        log.info("Final price update...")
        await asyncio.sleep(5)
        price_updater.cancel()

    filepath = save_session()
    report = print_signals_report()
    log.info("\n%s", report)

    if REAL_TRADING and trader:
        await asyncio.sleep(3)
        final_bal = trader.get_sol_balance()
        summary = trader.get_trade_summary()
        log.info("WALLET: %.4f SOL ($%.2f) | initial: %.4f SOL", final_bal, final_bal * SOL_PRICE_USD, trader.initial_balance)
        log.info("TRADES: %s", summary)

    log.info("DONE! Session: %s", filepath)


if __name__ == "__main__":
    asyncio.run(main())
