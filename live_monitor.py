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
    "initial_buy_sol", "initial_price_usd", "initial_mcap_usd", "market_cap_sol",
    "v_sol_in_bonding", "total_buys", "total_sells", "total_buy_sol", "total_sell_sol",
    "buy_sell_ratio", "sell_pressure", "dex_liquidity_usd", "dex_volume_5m",
    "dex_volume_1h", "dex_buys_5m", "dex_sells_5m", "dex_buys_1h", "dex_sells_1h",
    "dex_market_cap", "dex_fdv", "dex_change_5m", "dex_change_1h",
    "has_website", "has_socials", "migrated",
]

MIN_BUYS_FOR_SIGNAL = 50
MIN_RATIO_FOR_SIGNAL = 1.5
STOP_LOSS_PCT = -10.0
TIME_STOP_SEC = 120
TIME_STOP_MIN_GAIN = 10.0
TRAILING_STOP_PCT = 10.0
TP_LADDER = [
    {"level": 200.0, "sell_pct": 50},
    {"level": 400.0, "sell_pct": 100},
]
MAX_SELLS_PER_TOKEN = 3
MIN_LIQUIDITY_USD = 1500.0
MAX_SLOTS = int(os.getenv("MAX_SLOTS", "10"))
BET_SIZE_USD = float(os.getenv("BET_SIZE_USD", "5.0"))

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
        await asyncio.sleep(10)
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
            if buys < 5:
                continue

            token["total_buys"] = buys
            token["total_sells"] = sells
            token["total_buy_sol"] = tc.get("buy_sol", 0)
            token["total_sell_sol"] = tc.get("sell_sol", 0)
            token["buy_sell_ratio"] = round(buys / max(1, sells), 2)
            token["sell_pressure"] = round(sells / max(1, buys + sells) * 100, 1)

            label, confidence = predict_token(token)
            token["ml_checked"] = True
            token["ml_label"] = label
            token["ml_confidence"] = confidence

            if label in ("ROCKET", "winner"):
                if mint in seen_mints:
                    continue
                seen_mints.add(mint)

                if stats["signals"] >= MAX_SLOTS:
                    log.info("SKIP %s %s: max slots %d reached", label, token["symbol"], MAX_SLOTS)
                    continue

                if buys < MIN_BUYS_FOR_SIGNAL:
                    log.info("SKIP %s %s: buys=%d < %d", label, token["symbol"], buys, MIN_BUYS_FOR_SIGNAL)
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
                    "position_remaining_pct": 100.0,
                    "tp_hits": [],
                    "realized_pnl": 0.0,
                    "status": "ACTIVE",
                    "close_reason": None,
                    "close_pnl_pct": None,
                    "close_time": None,
                    "checked_at": None,
                    "sells_done": 0,
                }
                signals.append(signal)
                stats["signals"] += 1
                log.info(
                    "*** SIGNAL #%d: %s %s (%s) conf=%.0f%% buys=%d ratio=%.1f ***",
                    stats["signals"], label, token["symbol"], mint[:8],
                    confidence, buys, token["buy_sell_ratio"],
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

    if current_pnl <= STOP_LOSS_PCT:
        return f"STOP_LOSS ({current_pnl:+.1f}% <= {STOP_LOSS_PCT}%)"

    age_sec = time.time() - sig["signal_time"]
    if age_sec >= TIME_STOP_SEC and current_pnl < TIME_STOP_MIN_GAIN:
        return f"TIME_STOP ({age_sec:.0f}s, pnl={current_pnl:+.1f}% < +{TIME_STOP_MIN_GAIN}%)"

    peak = sig.get("peak_pnl_pct", 0)
    if peak >= 30.0:
        drop_from_peak = peak - current_pnl
        if drop_from_peak >= TRAILING_STOP_PCT:
            return f"TRAILING_STOP (peak={peak:+.1f}%, now={current_pnl:+.1f}%, drop={drop_from_peak:.1f}%)"

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
            log.info("REAL BUY OK %s: %.4f SOL | tx=%s", sig["symbol"], sol_amount, result["tx_hash"][:20])
        else:
            log.error("REAL BUY FAILED %s: %s", sig["symbol"], result["error"])
    except Exception as e:
        log.error("REAL BUY ERROR %s: %s", sig["symbol"], e)


async def execute_real_sell(sig: dict, sell_pct: int, reason: str):
    try:
        result = await trader.sell_token(sig["mint"], sell_pct, sig["symbol"], reason)
        if result["success"]:
            sol_got = result.get("sol_received", 0)
            log.info("REAL SELL OK %s %d%% (%s): tx=%s | +%.6f SOL", sig["symbol"], sell_pct, reason, result["tx_hash"][:20], sol_got)
        else:
            log.error("REAL SELL FAILED %s (%s): %s", sig["symbol"], reason, result.get("error", "not confirmed"))
        return result
    except Exception as e:
        log.error("REAL SELL ERROR %s: %s", sig["symbol"], e)
        return {"success": False, "error": str(e)}


async def signal_price_updater(client: httpx.AsyncClient):
    while True:
        await asyncio.sleep(15)
        active = [s for s in signals if s["status"] == "ACTIVE"]
        if not active:
            continue

        for sig in active:
            current_pnl = None
            try:
                mint = sig["mint"]
                latest_mcap = tokens.get(mint, {}).get("latest_mcap_usd", 0)
                initial_mcap = sig["initial_mcap_usd"]

                if latest_mcap > 0 and initial_mcap > 0:
                    current_pnl = ((latest_mcap - initial_mcap) / initial_mcap) * 100

                r = await client.get(
                    f"{DEXSCREENER_API}/tokens/v1/solana/{mint}",
                    timeout=10,
                )
                if r.status_code == 200:
                    pairs = r.json()
                    if pairs and isinstance(pairs, list) and len(pairs) > 0:
                        price = float(pairs[0].get("priceUsd") or 0)
                        sig["current_price_usd"] = price
                        entry = sig["entry_price_usd"]
                        if entry > 0 and price > 0:
                            current_pnl = ((price - entry) / entry) * 100
                        elif price == 0:
                            current_pnl = -100.0

                if current_pnl is not None:
                    sig["pnl_pct"] = round(current_pnl, 1)
                    if current_pnl > sig.get("peak_pnl_pct", 0):
                        sig["peak_pnl_pct"] = round(current_pnl, 1)
                    sig["checked_at"] = datetime.now(timezone.utc).isoformat()

                    check_tp_ladder(sig, current_pnl)

                    reason = check_exit_rules(sig, current_pnl)
                    if reason:
                        close_signal(sig, reason, current_pnl)
                    else:
                        remaining = sig["position_remaining_pct"] / 100.0
                        unrealized = remaining * (current_pnl / 100.0)
                        total = sig["realized_pnl"] + unrealized
                        sig["pnl_usd_per_dollar"] = round(1 + total, 4)

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

                    trade_counts[mint] = {"buys": 0, "sells": 0, "buy_sol": 0, "sell_sol": 0}
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
                        if tx == "buy":
                            tc["buys"] += 1
                            tc["buy_sol"] += sol_amount
                        else:
                            tc["sells"] += 1
                            tc["sell_sol"] += sol_amount

                        new_mcap = float(msg.get("marketCapSol") or 0)
                        if new_mcap > 0 and mint in tokens:
                            tokens[mint]["latest_mcap_sol"] = new_mcap
                            tokens[mint]["latest_mcap_usd"] = new_mcap * SOL_PRICE_USD

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

            if pnl is not None:
                pnl_str = f"{pnl:+.1f}%"
                val = sig.get("pnl_usd_per_dollar", 1.0)
            else:
                latest_mcap = tokens.get(sig["mint"], {}).get("latest_mcap_usd", 0)
                initial_mcap = sig["initial_mcap_usd"]
                if latest_mcap > 0 and initial_mcap > 0:
                    pnl_calc = ((latest_mcap - initial_mcap) / initial_mcap) * 100
                    pnl_str = f"{pnl_calc:+.1f}% (mcap)"
                    val = 1 + pnl_calc / 100
                else:
                    pnl_str = "checking..."
                    val = 1.0

            balance += val
            invested += 1

            lines.append(
                f"  ACTIVE {sig['ml_label']} {sig['symbol']} | "
                f"P&L: {pnl_str} | peak: {peak:+.1f}% | pos: {remaining:.0f}%{tp_str} | "
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
                f"  CLOSED {sig['symbol']} | P&L: {cpnl:+.1f}% | ${val:.2f}{tp_str} | {sig['close_reason']}"
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
            "tp_ladder": TP_LADDER,
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
    log.info("LIVE ML MONITOR v4 [%s]", mode_str)
    log.info("Duration: %d seconds", duration)
    log.info("Bet size: $%.0f | Max slots: %d | Max sells/token: %d", BET_SIZE_USD, MAX_SLOTS, MAX_SELLS_PER_TOKEN)
    log.info("Filters: buys >= %d, ratio >= %.1f, liquidity >= $%.0f", MIN_BUYS_FOR_SIGNAL, MIN_RATIO_FOR_SIGNAL, MIN_LIQUIDITY_USD)
    log.info("Stop-loss: %.0f%% | Time-stop: %ds (min +%.0f%%)", STOP_LOSS_PCT, TIME_STOP_SEC, TIME_STOP_MIN_GAIN)
    log.info("Trailing stop: -%.0f%% from peak (activates at +30%%)", TRAILING_STOP_PCT)
    log.info("TP Ladder: %s", ", ".join(f"+{tp['level']:.0f}%->sell {tp['sell_pct']}%%" for tp in TP_LADDER))
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
